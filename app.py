"""
app.py — Hybrid Inventory Forecast Backend (Memory-Optimized for Render Free Tier)

Memory strategy (targets ~512 MB Render free tier):
  1. Lazy loading — stores are loaded on first request; only ONE store lives in RAM at a time.
  2. Immediate checkpoint eviction — `del ckpt; gc.collect()` after extracting all values.
  3. float32 DataFrames — halves RAM vs pandas float64 default with no accuracy loss.
  4. last_window only — the full scaled matrix is discarded after slicing the tail window.
  5. On-demand metrics — never stored in cache; computed per /metrics request and discarded.
  6. psutil RSS logging — every critical point emits MB so Render logs show live pressure.
"""

import gc
import os
import pickle
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import mean_squared_error

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("LightGBM not found. Run: pip install lightgbm")

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

app = Flask(__name__)

SERVICE_LEVEL_Z = 1.65
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Memory helpers ────────────────────────────────────────────────────────────

def rss_mb() -> float:
    """Return current process RSS in MB (requires psutil)."""
    if not _PSUTIL:
        return -1.0
    return psutil.Process(os.getpid()).memory_info().rss / 1_048_576


def log_mem(tag: str):
    mb = rss_mb()
    if mb >= 0:
        print(f"[mem] {tag}: {mb:.1f} MB RSS")
    else:
        print(f"[mem] {tag}: (install psutil for live tracking)")


# ── LSTM model (must match train.py exactly) ──────────────────────────────────

class MultiCategoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=2,
            batch_first=True, dropout=0.2,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── Inference helpers ─────────────────────────────────────────────────────────

def predict_lstm(cache: dict) -> dict:
    last_window    = cache["last_window"]   # float32, shape (window_size, num_features)
    scaler         = cache["lstm_scaler"]
    all_cols       = cache["all_cols"]
    dense_cats     = cache["dense_cats"]
    target_indices = cache["lstm_target_indices"]
    num_features   = len(all_cols)
    df             = cache["df"]

    inp = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(device)

    cache["lstm_model"].eval()
    with torch.no_grad():
        pred_scaled = cache["lstm_model"](inp).cpu().numpy()

    dummy = np.zeros((1, num_features), dtype=np.float32)
    dummy[0, target_indices] = pred_scaled[0]
    pred_real = scaler.inverse_transform(dummy)[0]

    result = {}
    for i, cat in enumerate(dense_cats):
        lstm_val = float(max(0, pred_real[target_indices[i]]))
        ma_val   = float(df[cat].values[-7:].mean())
        result[cat] = round(0.7 * lstm_val + 0.3 * ma_val, 2)
    return result


def predict_lgbm(cache: dict) -> dict:
    df                    = cache["df"]
    lgbm_models           = cache["lgbm_models"]
    feature_cols          = cache["lgbm_feature_cols"]
    cross_store_feat_cols = cache.get("cross_store_feat_cols", {})
    cross_store_dfs       = cache.get("cross_store_dfs", {})
    result = {}
    for cat, booster in lgbm_models.items():
        if cat in cross_store_dfs:
            cols     = cross_store_feat_cols[cat]
            last_row = cross_store_dfs[cat][cols].values[-1].reshape(1, -1)
        else:
            last_row = df[feature_cols].values[-1].reshape(1, -1)
        result[cat] = round(float(max(0, booster.predict(last_row)[0])), 2)
    return result


def predict_ma(cache: dict) -> dict:
    return dict(cache["ma_values"])


def run_all_predictions(cache: dict) -> dict:
    forecasts = {}
    if cache.get("lstm_model") is not None and cache["dense_cats"]:
        forecasts.update(predict_lstm(cache))
    if cache.get("lgbm_models"):
        forecasts.update(predict_lgbm(cache))
    forecasts.update(predict_ma(cache))
    return forecasts


# ── Metrics (computed on-demand, NOT cached) ──────────────────────────────────

def compute_mape(actual: np.ndarray, predicted: np.ndarray):
    mask = actual != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def compute_metrics_on_demand(cache: dict) -> dict:
    """
    Compute metrics on-demand. Re-scales df.values for LSTM evaluation,
    then immediately discards the full scaled array and test tensors.
    Nothing is stored back into the cache — RAM is fully reclaimed after the request.
    """
    df            = cache["df"]
    all_cols      = cache["all_cols"]
    dense_cats    = cache["dense_cats"]
    sparse_cats   = cache["sparse_cats"]
    nearzero_cats = cache["nearzero_cats"]
    window_size   = cache["window_size"]
    scaler        = cache["lstm_scaler"]
    feature_cols  = cache["lgbm_feature_cols"]
    lgbm_models   = cache["lgbm_models"]
    ma_values     = cache["ma_values"]
    num_features  = len(all_cols)
    metrics       = {}

    # ── LSTM metrics ──────────────────────────────────────────────────────────
    if cache.get("lstm_model") is not None and dense_cats:
        target_indices = cache["lstm_target_indices"]

        # Re-scale the full df on-demand (float32 to minimise RAM)
        scaled = scaler.transform(df.values).astype(np.float32)
        n      = len(scaled)

        X, y = [], []
        for i in range(n - window_size):
            X.append(scaled[i: i + window_size])
            y.append(scaled[i + window_size, target_indices])

        X        = torch.tensor(np.array(X, dtype=np.float32), dtype=torch.float32)
        y_scaled = np.array(y, dtype=np.float32)
        del scaled  # free immediately
        gc.collect()

        split  = int(len(X) * 0.8)
        X_test = X[split:].to(device)
        y_test = y_scaled[split:]
        del X, y_scaled
        gc.collect()

        cache["lstm_model"].eval()
        with torch.no_grad():
            y_pred_scaled = cache["lstm_model"](X_test).cpu().numpy()
        del X_test
        gc.collect()

        def inv(arr):
            d = np.zeros((len(arr), num_features), dtype=np.float32)
            d[:, target_indices] = arr
            return scaler.inverse_transform(d)[:, target_indices]

        y_true_r = inv(y_test)
        y_pred_r = np.clip(inv(y_pred_scaled), 0, None)

        for i, cat in enumerate(dense_cats):
            actual = y_true_r[:, i]
            pred   = y_pred_r[:, i]
            mape   = compute_mape(actual, pred)
            metrics[cat] = {
                "model": "LSTM",
                "rmse":  round(float(np.sqrt(mean_squared_error(actual, pred))), 2),
                "mape":  round(mape, 2) if mape is not None else None,
            }

    # ── LightGBM metrics ──────────────────────────────────────────────────────
    if lgbm_models:
        cross_store_feat_cols = cache.get("cross_store_feat_cols", {})
        cross_store_dfs       = cache.get("cross_store_dfs", {})
        X_feat = df[feature_cols].values
        split  = int(len(X_feat) * 0.8)

        for cat, booster in lgbm_models.items():
            if cat in cross_store_dfs:
                cdf      = cross_store_dfs[cat]
                cols     = cross_store_feat_cols[cat]
                X_cat    = cdf[cols].values
                y_all    = cdf["sales"].values
                cs       = int(len(X_cat) * 0.8)
                X_test_l = X_cat[cs:-1]
                y_true   = y_all[cs + 1:]
            else:
                X_test_l = X_feat[split:-1]
                y_true   = df[cat].values[split + 1:]

            y_pred  = np.clip(booster.predict(X_test_l), 0, None)
            min_len = min(len(y_true), len(y_pred))
            y_true, y_pred = y_true[:min_len], y_pred[:min_len]
            mape = compute_mape(y_true, y_pred)
            metrics[cat] = {
                "model": "LightGBM",
                "rmse":  round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
                "mape":  round(mape, 2) if mape is not None else None,
            }

    # ── MA metrics ────────────────────────────────────────────────────────────
    for cat in nearzero_cats:
        series = df[cat].values
        split  = int(len(series) * 0.8)
        y_true = series[split:]
        y_pred = np.full(len(y_true), fill_value=ma_values.get(cat, 0.0), dtype=np.float32)
        mape   = compute_mape(y_true, y_pred)
        has_signal = y_true.sum() > 0
        rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2) if has_signal else None
        metrics[cat] = {
            "model": "MA-30",
            "rmse":  rmse,
            "mape":  round(mape, 2) if mape is not None else None,
        }

    all_mapes = [v["mape"] for v in metrics.values() if v["mape"] is not None]
    all_rmses = [v["rmse"] for v in metrics.values() if v["rmse"] is not None]

    return {
        "summary": {
            "mean_mape": round(float(np.mean(all_mapes)), 2) if all_mapes else None,
            "mean_rmse": round(float(np.mean(all_rmses)), 2) if all_rmses else None,
            "model_split": {
                "lstm":     len(dense_cats),
                "lightgbm": len(sparse_cats),
                "ma":       len(nearzero_cats),
            },
        },
        "by_category": metrics,
    }


# ── Replenishment ─────────────────────────────────────────────────────────────

def compute_replenishment(
    forecasts: dict,
    history_df: pd.DataFrame,
    family_cols: list,
    lead_time_days: int,
) -> dict:
    recs = {}
    for cat in family_cols:
        if cat not in history_df.columns:
            continue
        history  = history_df[cat].values
        std_d    = float(np.std(history[-90:]))
        mean_d   = float(np.mean(history[-90:]))
        forecast = forecasts.get(cat, 0.0)

        safety_stock  = round(SERVICE_LEVEL_Z * std_d * np.sqrt(lead_time_days), 1)
        reorder_point = round(mean_d * lead_time_days + safety_stock, 1)
        approx_stock  = round(float(np.mean(history[-7:])), 1)
        recommended   = round(max(0.0, reorder_point - approx_stock), 1)

        recs[cat] = {
            "forecast_tomorrow": forecast,
            "safety_stock":      safety_stock,
            "reorder_point":     reorder_point,
            "approx_stock":      approx_stock,
            "approx_stock_note": "7-day sales proxy — replace with WMS data",
            "recommended_order": recommended,
            "action":            "ORDER" if recommended > 0 else "OK",
        }
    return recs


# ── Store cache & lazy loader ─────────────────────────────────────────────────

print(f"[startup] Device: {device}")

STORE_CACHE: dict      = {}
AVAILABLE_STORES: list = []

ckpt_files = sorted(
    f for f in os.listdir(".")
    if f.startswith("store_") and f.endswith("_forecast.pth")
)
for f in ckpt_files:
    try:
        AVAILABLE_STORES.append(int(f.split("_")[1]))
    except (IndexError, ValueError):
        continue

if not AVAILABLE_STORES:
    print("[startup] WARNING — No checkpoint files found.")


def load_store(store_nbr: int) -> bool:
    """
    Lazy-load a store on first request.

    Rules (Render free tier = ~512 MB):
    - Evict ALL cached stores before loading a new one.
    - Extract every value needed from ckpt THEN del ckpt — never reference
      it afterwards.
    - Downcast df to float32 (halves DataFrame RAM vs pandas float64 default).
    - Keep only last_window (window_size rows) of the scaled array; discard
      the full scaled matrix immediately after slicing.
    - Never pre-compute or cache metrics — compute them on-demand per request.
    """
    if store_nbr in STORE_CACHE:
        return True

    if STORE_CACHE:
        print("[memory] Evicting cached stores to free RAM …")
        STORE_CACHE.clear()
        gc.collect()
        log_mem("after eviction")

    ckpt_file = f"store_{store_nbr}_forecast.pth"
    if not os.path.exists(ckpt_file):
        return False

    try:
        print(f"[loading] Reading {ckpt_file} …")
        log_mem("before torch.load")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        log_mem("after torch.load")

        # ── Extract everything needed BEFORE del ckpt ─────────────────────
        all_cols              = ckpt["all_cols"]
        family_cols           = ckpt["family_cols"]
        lgbm_feature_cols     = ckpt["lgbm_feature_cols"]
        dense_cats            = ckpt["dense_cats"]
        sparse_cats           = ckpt["sparse_cats"]
        nearzero_cats         = ckpt["nearzero_cats"]
        window_size           = ckpt["window_size"]
        ma_values             = ckpt["ma_values"]
        lstm_scaler           = ckpt["lstm_scaler"]
        lstm_target_indices   = ckpt.get("lstm_target_indices", [])
        cross_store_feat_cols = ckpt.get("cross_store_feat_cols", {})
        cross_store_dfs       = ckpt.get("cross_store_dfs", {})

        if "df_feat" not in ckpt:
            raise RuntimeError("Checkpoint is missing df_feat.")

        # float32 cast — halves DataFrame RAM vs pandas float64 default
        df_feat: pd.DataFrame = ckpt["df_feat"]
        for col in all_cols:
            if col not in df_feat.columns:
                df_feat[col] = np.float32(0.0)
        df_feat = df_feat[all_cols].astype(np.float32)

        # Scale full array, keep only the tail window, discard the rest
        scaled_full = lstm_scaler.transform(df_feat.values).astype(np.float32)
        last_window = scaled_full[-window_size:].copy()
        del scaled_full
        gc.collect()
        log_mem("after scale+slice")

        # ── LSTM ──────────────────────────────────────────────────────────
        lstm_model = None
        if dense_cats and ckpt.get("lstm_state_dict") is not None:
            lstm_model = MultiCategoryLSTM(
                ckpt["lstm_num_features"],
                ckpt["lstm_hidden_size"],
                ckpt["lstm_num_targets"],
            ).to(device)
            lstm_model.load_state_dict(ckpt["lstm_state_dict"])
            lstm_model.eval()

        # ── LightGBM ──────────────────────────────────────────────────────
        lgbm_models = {
            cat: pickle.loads(raw)
            for cat, raw in ckpt.get("lgbm_models_bytes", {}).items()
        }

        # ── Free the heavy checkpoint dict now that we have everything ────
        del ckpt
        gc.collect()
        log_mem("after del ckpt")

        STORE_CACHE[store_nbr] = {
            "all_cols":              all_cols,
            "family_cols":           family_cols,
            "lgbm_feature_cols":     lgbm_feature_cols,
            "dense_cats":            dense_cats,
            "sparse_cats":           sparse_cats,
            "nearzero_cats":         nearzero_cats,
            "window_size":           window_size,
            "lstm_model":            lstm_model,
            "lstm_scaler":           lstm_scaler,
            "lstm_target_indices":   lstm_target_indices,
            "lgbm_models":           lgbm_models,
            "cross_store_feat_cols": cross_store_feat_cols,
            "cross_store_dfs":       cross_store_dfs,
            "ma_values":             ma_values,
            "df":                    df_feat,
            "last_window":           last_window,
        }

        print(f"[loading] Store {store_nbr} ready.")
        log_mem("store loaded")
        return True

    except Exception:
        print(f"[loading] Store {store_nbr} FAILED:")
        traceback.print_exc()
        STORE_CACHE.pop(store_nbr, None)
        return False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    """Lightweight liveness probe — also exposes RSS for Render logs."""
    return jsonify({
        "status":           "ok",
        "rss_mb":           round(rss_mb(), 1),
        "available_stores": AVAILABLE_STORES,
        "loaded_stores":    [int(s) for s in sorted(STORE_CACHE.keys())],
    })


@app.route("/status")
def status():
    info = {}
    for s, c in STORE_CACHE.items():
        info[int(s)] = {
            "categories":  len(c["family_cols"]),
            "lstm":        len(c["dense_cats"]),
            "lightgbm":    len(c["sparse_cats"]),
            "ma_fallback": len(c["nearzero_cats"]),
        }
    return jsonify({
        "device":            str(device),
        "rss_mb":            round(rss_mb(), 1),
        "available_stores":  AVAILABLE_STORES,
        "currently_loaded":  [int(s) for s in sorted(STORE_CACHE.keys())],
        "loaded_store_info": info,
    })


@app.route("/predict/all")
def predict_all():
    try:
        if not AVAILABLE_STORES:
            return jsonify({"error": "No store checkpoints available."}), 404

        store = int(request.args.get("store", AVAILABLE_STORES[0]))
        if store not in AVAILABLE_STORES:
            return jsonify({"error": f"Store {store} not found. Available: {AVAILABLE_STORES}"}), 404
        if not load_store(store):
            return jsonify({"error": f"Failed to load store {store}."}), 500

        forecasts = run_all_predictions(STORE_CACHE[store])
        gc.collect()
        log_mem("/predict/all done")
        return jsonify({"status": "success", "store": store, "predictions": forecasts})

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/metrics")
def metrics():
    """
    Metrics are computed on-demand and NOT cached — this keeps the store
    cache lean between prediction requests.
    """
    try:
        if not AVAILABLE_STORES:
            return jsonify({"error": "No store checkpoints available."}), 404

        store = int(request.args.get("store", AVAILABLE_STORES[0]))
        if store not in AVAILABLE_STORES:
            return jsonify({"error": f"Store {store} not found. Available: {AVAILABLE_STORES}"}), 404
        if not load_store(store):
            return jsonify({"error": f"Failed to load store {store}."}), 500

        result = compute_metrics_on_demand(STORE_CACHE[store])
        gc.collect()
        log_mem("/metrics done")
        return jsonify({"store": store, **result})

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/replenishment")
def replenishment():
    try:
        if not AVAILABLE_STORES:
            return jsonify({"error": "No store checkpoints available."}), 404

        store     = int(request.args.get("store", AVAILABLE_STORES[0]))
        lead_time = int(request.args.get("lead_time", 3))

        if store not in AVAILABLE_STORES:
            return jsonify({"error": f"Store {store} not found."}), 404
        if not (1 <= lead_time <= 30):
            return jsonify({"error": "lead_time must be 1–30."}), 400
        if not load_store(store):
            return jsonify({"error": f"Failed to load store {store}."}), 500

        cache     = STORE_CACHE[store]
        forecasts = run_all_predictions(cache)
        recs      = compute_replenishment(
            forecasts, cache["df"], cache["family_cols"], lead_time
        )
        gc.collect()
        log_mem("/replenishment done")
        return jsonify({
            "store":          store,
            "lead_time_days": lead_time,
            "service_level":  "95%",
            "orders_needed":  sum(1 for v in recs.values() if v["action"] == "ORDER"),
            "replenishment":  recs,
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[server] Listening on port {port}  |  RSS: {rss_mb():.1f} MB")
    app.run(host="0.0.0.0", port=port, debug=False)