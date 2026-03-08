"""
app.py — Hybrid Inventory Forecast Backend
==========================================
Matches train.py hybrid architecture:
  • Dense categories   → LSTM
  • Sparse categories  → LightGBM + Tweedie
  • Near-zero cats     → 30-day MA fallback

Each store loads its own checkpoint (store_N_forecast.pth).
Metrics are pre-computed once at startup per store.
NO CSV files needed — all data is stored inside the .pth checkpoint.

Run (dev):   python app.py
Run (prod):  gunicorn -w 2 -b 0.0.0.0:$PORT app:app
"""

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

app = Flask(__name__)

SERVICE_LEVEL_Z = 1.65
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── LSTM model (must match train.py exactly) ──────────────────────────────────
class MultiCategoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── Inference helpers ─────────────────────────────────────────────────────────
def predict_lstm(cache):
    window_size    = cache["window_size"]
    scaled         = cache["scaled"]
    scaler         = cache["lstm_scaler"]
    all_cols       = cache["all_cols"]
    dense_cats     = cache["dense_cats"]
    target_indices = cache["lstm_target_indices"]
    num_features   = len(all_cols)
    df             = cache["df"]

    last_window = scaled[-window_size:]
    inp         = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(device)

    cache["lstm_model"].eval()
    with torch.no_grad():
        pred_scaled = cache["lstm_model"](inp).cpu().numpy()

    dummy = np.zeros((1, num_features))
    dummy[0, target_indices] = pred_scaled[0]
    pred_real = scaler.inverse_transform(dummy)[0]

    result = {}
    for i, cat in enumerate(dense_cats):
        lstm_val = float(max(0, pred_real[target_indices[i]]))
        ma_val   = float(df[cat].values[-7:].mean())
        blended  = round(0.7 * lstm_val + 0.3 * ma_val, 2)
        result[cat] = blended
    return result


def predict_lgbm(cache):
    df                    = cache["df"]
    lgbm_models           = cache["lgbm_models"]
    feature_cols          = cache["lgbm_feature_cols"]
    cross_store_feat_cols = cache.get("cross_store_feat_cols", {})
    cross_store_dfs       = cache.get("cross_store_dfs", {})
    result = {}
    for cat, booster in lgbm_models.items():
        if cat in cross_store_dfs:
            cols     = cross_store_feat_cols[cat]
            cdf      = cross_store_dfs[cat]
            last_row = cdf[cols].values[-1].reshape(1, -1)
        else:
            last_row = df[feature_cols].values[-1].reshape(1, -1)
        result[cat] = round(float(max(0, booster.predict(last_row)[0])), 2)
    return result


def predict_ma(cache):
    return dict(cache["ma_values"])


def run_all_predictions(cache):
    forecasts = {}
    if cache.get("lstm_model") is not None and cache["dense_cats"]:
        forecasts.update(predict_lstm(cache))
    if cache.get("lgbm_models"):
        forecasts.update(predict_lgbm(cache))
    forecasts.update(predict_ma(cache))
    return forecasts


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_mape(actual, predicted):
    mask = actual != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def precompute_metrics(cache):
    df            = cache["df"]
    all_cols      = cache["all_cols"]
    dense_cats    = cache["dense_cats"]
    sparse_cats   = cache["sparse_cats"]
    nearzero_cats = cache["nearzero_cats"]
    window_size   = cache["window_size"]
    scaled        = cache["scaled"]
    scaler        = cache["lstm_scaler"]
    feature_cols  = cache["lgbm_feature_cols"]
    lgbm_models   = cache["lgbm_models"]
    ma_values     = cache["ma_values"]
    num_features  = len(all_cols)
    n             = len(scaled)

    metrics = {}

    # ── LSTM ──
    if cache.get("lstm_model") is not None and dense_cats:
        target_indices = cache["lstm_target_indices"]
        X, y = [], []
        for i in range(n - window_size):
            X.append(scaled[i: i + window_size])
            y.append(scaled[i + window_size, target_indices])

        X        = torch.tensor(np.array(X), dtype=torch.float32)
        y_scaled = np.array(y)
        split    = int(len(X) * 0.8)
        X_test   = X[split:].to(device)
        y_test   = y_scaled[split:]

        cache["lstm_model"].eval()
        with torch.no_grad():
            y_pred_scaled = cache["lstm_model"](X_test).cpu().numpy()

        def inv(arr):
            d = np.zeros((len(arr), num_features))
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

    # ── LightGBM ──
    if lgbm_models:
        cross_store_feat_cols = cache.get("cross_store_feat_cols", {})
        cross_store_dfs       = cache.get("cross_store_dfs", {})
        X_feat   = df[feature_cols].values
        split    = int(len(X_feat) * 0.8)

        for cat, booster in lgbm_models.items():
            if cat in cross_store_dfs:
                # use the saved cross_df — correct feature set
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

    # ── MA fallback ──
    for cat in nearzero_cats:
        series = df[cat].values
        split  = int(len(series) * 0.8)
        y_true = series[split:]
        y_pred = np.full(len(y_true), fill_value=ma_values.get(cat, 0.0), dtype=float)
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
            "mean_mape":   round(float(np.mean(all_mapes)), 2) if all_mapes else None,
            "mean_rmse":   round(float(np.mean(all_rmses)), 2) if all_rmses else None,
            "model_split": {
                "lstm":     len(dense_cats),
                "lightgbm": len(sparse_cats),
                "ma":       len(nearzero_cats),
            },
        },
        "by_category": metrics,
    }


# ── Replenishment ─────────────────────────────────────────────────────────────
def compute_replenishment(forecasts, history_df, family_cols, lead_time_days):
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


# ── Startup — load from .pth only, no CSV needed ──────────────────────────────
print(f"[startup] Device: {device}")

STORE_CACHE = {}

# Find all checkpoint files in current directory
ckpt_files = sorted([f for f in os.listdir(".") if f.startswith("store_") and f.endswith("_forecast.pth")])
print(f"[startup] Found checkpoints: {ckpt_files}")

if not ckpt_files:
    raise SystemExit("[startup] FATAL — No checkpoint files found. Add store_N_forecast.pth to your repo.")

for ckpt_file in ckpt_files:
    try:
        store_nbr = int(ckpt_file.split("_")[1])
    except (IndexError, ValueError):
        continue

    try:
        print(f"[startup] Loading store {store_nbr} from {ckpt_file}...")
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)

        all_cols          = ckpt["all_cols"]
        family_cols       = ckpt["family_cols"]
        lgbm_feature_cols = ckpt["lgbm_feature_cols"]
        dense_cats        = ckpt["dense_cats"]
        sparse_cats       = ckpt["sparse_cats"]
        nearzero_cats     = ckpt["nearzero_cats"]
        window_size       = ckpt["window_size"]
        ma_values         = ckpt["ma_values"]
        lstm_scaler       = ckpt["lstm_scaler"]

        # ── Load df_feat from checkpoint (no CSV needed) ──
        if "df_feat" in ckpt:
            df_feat = ckpt["df_feat"]
            print(f"[startup] Store {store_nbr}: loaded df_feat from checkpoint ({len(df_feat)} rows)")
        else:
            raise SystemExit(
                f"[startup] FATAL — checkpoint {ckpt_file} does not contain df_feat.\n"
                "Please retrain with the updated train.py that saves df_feat inside the checkpoint."
            )

        # Ensure all expected columns exist
        for col in all_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0.0
        df_feat = df_feat[all_cols]
        scaled  = lstm_scaler.transform(df_feat.values)

        # ── Load LSTM ──
        lstm_model = None
        if dense_cats and ckpt.get("lstm_state_dict") is not None:
            lstm_model = MultiCategoryLSTM(
                ckpt["lstm_num_features"],
                ckpt["lstm_hidden_size"],
                ckpt["lstm_num_targets"],
            ).to(device)
            lstm_model.load_state_dict(ckpt["lstm_state_dict"])
            lstm_model.eval()

        # ── Load LightGBM ──
        lgbm_models           = {
            cat: pickle.loads(raw)
            for cat, raw in ckpt.get("lgbm_models_bytes", {}).items()
        }
        cross_store_feat_cols = ckpt.get("cross_store_feat_cols", {})
        cross_store_dfs       = ckpt.get("cross_store_dfs", {})

        cache = {
            "all_cols":            all_cols,
            "family_cols":         family_cols,
            "lgbm_feature_cols":   lgbm_feature_cols,
            "dense_cats":          dense_cats,
            "sparse_cats":         sparse_cats,
            "nearzero_cats":       nearzero_cats,
            "window_size":         window_size,
            "lstm_model":          lstm_model,
            "lstm_scaler":         lstm_scaler,
            "lstm_target_indices": ckpt.get("lstm_target_indices", []),
            "lgbm_models":            lgbm_models,
            "cross_store_feat_cols":  cross_store_feat_cols,
            "cross_store_dfs":        cross_store_dfs,
            "ma_values":           ma_values,
            "df":                  df_feat,
            "scaled":              scaled,
        }

        print(f"[startup] Store {store_nbr}: computing metrics "
              f"(LSTM={len(dense_cats)} LGB={len(sparse_cats)} MA={len(nearzero_cats)})...")
        cache["metrics"] = precompute_metrics(cache)

        STORE_CACHE[store_nbr] = cache
        mape = cache["metrics"]["summary"].get("mean_mape", "n/a")
        print(f"[startup] Store {store_nbr} ready — mean_mape={mape}%")

    except SystemExit:
        raise
    except Exception:
        print(f"[startup] Store {store_nbr}: FAILED to load")
        traceback.print_exc()

if not STORE_CACHE:
    raise SystemExit("[startup] FATAL — No stores loaded successfully.")

print(f"[startup] Ready. Stores loaded: {sorted(STORE_CACHE.keys())}")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/status")
def status():
    info = {}
    for s, c in STORE_CACHE.items():
        summary = c["metrics"]["summary"]
        info[int(s)] = {
            "categories":  len(c["family_cols"]),
            "lstm":        len(c["dense_cats"]),
            "lightgbm":    len(c["sparse_cats"]),
            "ma_fallback": len(c["nearzero_cats"]),
            "mean_mape":   summary.get("mean_mape"),
        }
    return jsonify({
        "device":        str(device),
        "stores_cached": [int(s) for s in sorted(STORE_CACHE.keys())],
        "store_info":    info,
    })


@app.route("/predict/all")
def predict_all():
    try:
        default = sorted(STORE_CACHE.keys())[0]
        store   = int(request.args.get("store", default))
        if store not in STORE_CACHE:
            return jsonify({"error": f"Store {store} not loaded. Available: {sorted(STORE_CACHE.keys())}"}), 404
        forecasts = run_all_predictions(STORE_CACHE[store])
        return jsonify({"status": "success", "store": store, "predictions": forecasts})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/metrics")
def metrics():
    try:
        default = sorted(STORE_CACHE.keys())[0]
        store   = int(request.args.get("store", default))
        if store not in STORE_CACHE:
            return jsonify({"error": f"Store {store} not loaded."}), 404
        return jsonify({"store": store, **STORE_CACHE[store]["metrics"]})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/replenishment")
def replenishment():
    try:
        default   = sorted(STORE_CACHE.keys())[0]
        store     = int(request.args.get("store", default))
        lead_time = int(request.args.get("lead_time", 3))

        if store not in STORE_CACHE:
            return jsonify({"error": f"Store {store} not loaded."}), 404
        if not (1 <= lead_time <= 30):
            return jsonify({"error": "lead_time must be 1–30"}), 400

        cache     = STORE_CACHE[store]
        forecasts = run_all_predictions(cache)
        recs      = compute_replenishment(
            forecasts, cache["df"], cache["family_cols"], lead_time
        )
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
    print(f"[server] Running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
