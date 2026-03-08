import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("LightGBM not found. Run: pip install lightgbm")

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--store",            type=int,   default=1)
parser.add_argument("--epochs",           type=int,   default=150)
parser.add_argument("--batch",            type=int,   default=64)
parser.add_argument("--window",           type=int,   default=60)
parser.add_argument("--hidden",           type=int,   default=256)
parser.add_argument("--lr",               type=float, default=1e-3)
parser.add_argument("--patience",         type=int,   default=15)
parser.add_argument("--sparse_threshold", type=float, default=0.20)
parser.add_argument("--nearzero_threshold", type=float, default=0.05)
parser.add_argument("--data_dir", type=str,
                    default="store-sales-time-series-forecasting")
args = parser.parse_args()

STORE_NBR          = args.store
WINDOW_SIZE        = args.window
HIDDEN_SIZE        = args.hidden
BATCH_SIZE         = args.batch
MAX_EPOCHS         = args.epochs
PATIENCE           = args.patience
LR                 = args.lr
SPARSE_THRESHOLD   = args.sparse_threshold
NEARZERO_THRESHOLD = args.nearzero_threshold
DATA_DIR           = args.data_dir
MODEL_PATH         = f"store_{STORE_NBR}_forecast.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train] Device: {device}  |  Store: {STORE_NBR}")
print(f"[train] Sparse threshold: non-zero rate ≤ {SPARSE_THRESHOLD:.0%} → LightGBM")
print(f"[train] Near-zero threshold: non-zero rate ≤ {NEARZERO_THRESHOLD:.0%} → MA fallback")


# ── LSTM model ────────────────────────────────────────────────────────────────
class MultiCategoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── Data loading ──────────────────────────────────────────────────────────────
print("[train] Loading CSVs...")
try:
    df_train    = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["date"])
    df_oil      = pd.read_csv(os.path.join(DATA_DIR, "oil.csv"),   parse_dates=["date"])
    hfp         = os.path.join(DATA_DIR, "holidays_events.csv")
    df_holidays = pd.read_csv(hfp, parse_dates=["date"]) if os.path.exists(hfp) else None
except FileNotFoundError as e:
    raise SystemExit(f"[train] ERROR: {e}")

print(f"[train] Holidays: {'yes' if df_holidays is not None else 'no'}")


# ── Shared feature engineering ────────────────────────────────────────────────
def build_features(df_train, df_oil, df_holidays, store_nbr):
    df_store = df_train[df_train["store_nbr"] == store_nbr].copy()

    df_sales = df_store.pivot_table(
        index="date", columns="family", values="sales", aggfunc="sum"
    ).fillna(0).sort_index()

    df_promo = df_store.pivot_table(
        index="date", columns="family", values="onpromotion", aggfunc="mean"
    ).fillna(0).sort_index()
    df_promo.columns = [f"{c}_promo" for c in df_promo.columns]

    df = df_sales.join(df_promo, how="left").fillna(0)

    idx = df.index
    df["day_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["day_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    df["week_sin"]  = np.sin(2 * np.pi * idx.isocalendar().week.values / 52)
    df["week_cos"]  = np.cos(2 * np.pi * idx.isocalendar().week.values / 52)

    if df_holidays is not None:
        holiday_dates = set(df_holidays["date"])
        df["is_holiday"] = idx.isin(holiday_dates).astype(float)
    else:
        df["is_holiday"] = 0.0

    family_cols = df_sales.columns.tolist()
    extra = []
    for lag in [7, 14, 28]:
        tmp = df[family_cols].shift(lag).fillna(0)
        tmp.columns = [f"{c}_lag{lag}" for c in family_cols]
        extra.append(tmp)
    roll7  = df[family_cols].shift(1).rolling(7,  min_periods=1).mean().fillna(0)
    roll28 = df[family_cols].shift(1).rolling(28, min_periods=1).mean().fillna(0)
    roll7.columns  = [f"{c}_roll7"  for c in family_cols]
    roll28.columns = [f"{c}_roll28" for c in family_cols]
    df = pd.concat([df] + extra + [roll7, roll28], axis=1)

    df = pd.merge(df, df_oil.set_index("date"),
                  left_index=True, right_index=True, how="left").ffill().fillna(0)

    df = df.iloc[28:].copy()
    return df, family_cols


# ── Category classification ───────────────────────────────────────────────────
def classify_categories(df_feat, family_cols, train_end,
                         sparse_threshold, nearzero_threshold):
    train_sales   = df_feat[family_cols].values[:train_end]
    nonzero_rates = (train_sales > 0).mean(axis=0)
    means = train_sales.mean(axis=0)
    stds  = train_sales.std(axis=0)
    cvs   = np.where(means > 0, stds / means, 0)

    dense_cats, sparse_cats, nearzero_cats = [], [], []
    for cat, rate, cv in zip(family_cols, nonzero_rates, cvs):
        if rate <= nearzero_threshold:
            nearzero_cats.append(cat)
        elif rate <= sparse_threshold or cv > 3.0:
            sparse_cats.append(cat)
        else:
            dense_cats.append(cat)

    return dense_cats, sparse_cats, nearzero_cats


# ── Build tabular features for LightGBM ──────────────────────────────────────
def build_lgbm_features(df_feat, cat, feature_cols):
    target = df_feat[cat].shift(-1).dropna()
    idx    = target.index
    X      = df_feat.loc[idx, feature_cols].values
    y      = target.values
    return X, y


# ── MA fallback ───────────────────────────────────────────────────────────────
def ma_forecast(series, window=30):
    return float(np.mean(series[-window:]))


# ── Main ──────────────────────────────────────────────────────────────────────
print("[train] Engineering features...")
df_feat, family_cols = build_features(df_train, df_oil, df_holidays, STORE_NBR)

n_total   = len(df_feat)
train_end = int(n_total * 0.8)
all_cols  = df_feat.columns.tolist()

sales_set         = set(family_cols)
lgbm_feature_cols = [c for c in all_cols if c not in sales_set]

dense_cats, sparse_cats, nearzero_cats = classify_categories(
    df_feat, family_cols, train_end, SPARSE_THRESHOLD, NEARZERO_THRESHOLD
)

print(f"[train] Total categories : {len(family_cols)}")
print(f"[train] Dense  → LSTM    : {len(dense_cats)}  {dense_cats}")
print(f"[train] Sparse → LightGBM: {len(sparse_cats)}  {sparse_cats}")
print(f"[train] Near-zero → MA   : {len(nearzero_cats)}  {nearzero_cats}")


# ══════════════════════════════════════════════════════════════════════════════
# PART A — LSTM
# ══════════════════════════════════════════════════════════════════════════════
lstm_model     = None
lstm_scaler    = None
target_indices = []

if dense_cats:
    print(f"\n[train] ── LSTM training on {len(dense_cats)} dense categories ──")

    target_indices = [all_cols.index(c) for c in dense_cats]
    num_features   = len(all_cols)
    num_targets    = len(dense_cats)

    lstm_scaler = MinMaxScaler()
    lstm_scaler.fit(df_feat.values[:train_end])
    scaled_data = lstm_scaler.transform(df_feat.values)

    X_all, y_all = [], []
    for i in range(len(scaled_data) - WINDOW_SIZE):
        X_all.append(scaled_data[i: i + WINDOW_SIZE])
        y_all.append(scaled_data[i + WINDOW_SIZE, target_indices])

    X_all = torch.tensor(np.array(X_all), dtype=torch.float32)
    y_all = torch.tensor(np.array(y_all), dtype=torch.float32)

    split   = int(len(X_all) * 0.8)
    X_train_t, X_val_t = X_all[:split], X_all[split:]
    y_train_t, y_val_t = y_all[:split], y_all[split:]

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),
                              batch_size=BATCH_SIZE, shuffle=False)

    print(f"[train] LSTM sequences — train: {len(X_train_t)}  val: {len(X_val_t)}")

    raw_sales   = df_feat[dense_cats].values[:train_end]
    mean_sales  = np.mean(raw_sales, axis=0)
    cap         = np.percentile(mean_sales[mean_sales > 0], 10)
    mean_sales  = mean_sales.clip(min=max(cap, 1.0))
    weights     = 1.0 / mean_sales
    weights     = weights / weights.sum() * num_targets
    loss_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    def weighted_mse(pred, target, w):
        return ((pred - target) ** 2 * w).mean()

    lstm_model = MultiCategoryLSTM(num_features, HIDDEN_SIZE, num_targets).to(device)
    optimizer  = optim.Adam(lstm_model.parameters(), lr=LR)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val, best_state, no_improve = float("inf"), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        lstm_model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = weighted_mse(lstm_model(xb), yb, loss_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(X_train_t)

        lstm_model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl_loss += weighted_mse(lstm_model(xb), yb, loss_weights).item() * len(xb)
        vl_loss /= len(X_val_t)
        scheduler.step(vl_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS}  train={tr_loss:.6f}  val={vl_loss:.6f}")

        if vl_loss < best_val:
            best_val   = vl_loss
            best_state = {k: v.clone() for k, v in lstm_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[train] LSTM early stop at epoch {epoch}")
                break

    lstm_model.load_state_dict(best_state)
    print(f"[train] LSTM best val loss: {best_val:.6f}")

    lstm_model.eval()
    with torch.no_grad():
        lw = torch.tensor(scaled_data[-WINDOW_SIZE:], dtype=torch.float32).unsqueeze(0).to(device)
        ps = lstm_model(lw).cpu().numpy()
    dummy = np.zeros((1, len(all_cols)))
    dummy[0, target_indices] = ps[0]
    pr = lstm_scaler.inverse_transform(dummy)[0]
    print("\n[train] LSTM sanity (dense categories, sample):")
    for i, cat in enumerate(dense_cats[:5]):
        print(f"  {cat:25s}: {max(0, pr[target_indices[i]]):8.1f} units")
else:
    print("[train] No dense categories found — skipping LSTM")
    lstm_scaler = MinMaxScaler()
    lstm_scaler.fit(df_feat.values[:train_end])


# ══════════════════════════════════════════════════════════════════════════════
# PART B — LightGBM
# ══════════════════════════════════════════════════════════════════════════════
lgbm_models = {}
cross_store_feat_cols = {}  # cat -> feature cols for cross-store LightGBM models
cross_store_dfs       = {}  # cat -> cross_df for cross-store LightGBM models

if sparse_cats:
    print(f"\n[train] ── LightGBM training on {len(sparse_cats)} sparse categories ──")

    for cat in sparse_cats:
        X, y = build_lgbm_features(df_feat, cat, lgbm_feature_cols)
        split_idx = int(len(X) * 0.8)
        X_tr, X_vl = X[:split_idx], X[split_idx:]
        y_tr, y_vl = y[:split_idx], y[split_idx:]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)

        params = {
            "objective": "tweedie", "tweedie_variance_power": 1.5,
            "metric": "rmse", "learning_rate": 0.05, "num_leaves": 31,
            "min_data_in_leaf": 5, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1,
        }
        booster = lgb.train(params, dtrain, num_boost_round=500,
                            valid_sets=[dval],
                            callbacks=[lgb.early_stopping(20, verbose=False),
                                       lgb.log_evaluation(-1)])
        lgbm_models[cat] = booster
        print(f"  {cat:25s}: best_iter={booster.best_iteration}")

    print(f"[train] LightGBM done — {len(lgbm_models)} models trained")
else:
    print("[train] No sparse categories — skipping LightGBM")


# ══════════════════════════════════════════════════════════════════════════════
# PART C — Near-zero categories
# ══════════════════════════════════════════════════════════════════════════════
ma_values = {}

if nearzero_cats:
    print(f"\n[train] ── Cross-store LightGBM for {len(nearzero_cats)} near-zero categories ──")

    all_store_sales = df_train[df_train["family"].isin(nearzero_cats)].copy()

    for cat in nearzero_cats:
        cat_data  = all_store_sales[all_store_sales["family"] == cat].copy()
        cat_pivot = cat_data.pivot_table(
            index="date", columns="store_nbr", values="sales", aggfunc="sum"
        ).fillna(0).sort_index()
        cross_store_mean = cat_pivot.mean(axis=1)
        cross_df = pd.DataFrame({"sales": cross_store_mean})
        cross_df = pd.merge(cross_df, df_oil.set_index("date"),
                            left_index=True, right_index=True, how="left").ffill().fillna(0)

        if cross_df["sales"].sum() == 0:
            print(f"  {cat:25s}: skipped (all-zero series)")
            ma_values[cat] = 0.0
            continue

        for lag in [7, 14, 28]:
            cross_df[f"lag{lag}"] = cross_df["sales"].shift(lag)
        cross_df["roll7"]  = cross_df["sales"].shift(1).rolling(7,  min_periods=1).mean()
        cross_df["roll28"] = cross_df["sales"].shift(1).rolling(28, min_periods=1).mean()
        cross_df = cross_df.fillna(0).iloc[28:].copy()

        feat_cols = [c for c in cross_df.columns if c != "sales"]
        target    = cross_df["sales"].shift(-1).dropna()

        if target.sum() == 0:
            print(f"  {cat:25s}: skipped (target all zero)")
            ma_values[cat] = 0.0
            continue

        X = cross_df.loc[target.index, feat_cols].values
        y = target.values
        split_idx = int(len(X) * 0.8)
        train_X, val_X = X[:split_idx], X[split_idx:]
        train_y, val_y = y[:split_idx], y[split_idx:]

        if train_y.sum() == 0:
            print(f"  {cat:25s}: skipped (train labels all zero)")
            ma_values[cat] = 0.0
            continue

        dtrain  = lgb.Dataset(train_X, label=train_y)
        dval    = lgb.Dataset(val_X, label=val_y, reference=dtrain)
        params  = {"objective": "tweedie", "tweedie_variance_power": 1.5,
                   "metric": "rmse", "learning_rate": 0.05, "num_leaves": 15,
                   "feature_fraction": 0.9, "bagging_fraction": 0.8,
                   "bagging_freq": 1, "verbose": -1}
        booster = lgb.train(params, dtrain, num_boost_round=300,
                            valid_sets=[dval],
                            callbacks=[lgb.early_stopping(20, verbose=False),
                                       lgb.log_evaluation(-1)])
        lgbm_models[cat] = booster
        cross_store_feat_cols[cat] = feat_cols
        cross_store_dfs[cat]       = cross_df   # save full cross_df for metrics
        sparse_cats.append(cat)
        print(f"  {cat:25s}: cross-store LGB, best_iter={booster.best_iteration}")

    nearzero_cats = []
else:
    print("[train] No near-zero categories — skipping")

# Compute MA values for any remaining nearzero cats
for cat in nearzero_cats:
    ma_values[cat] = ma_forecast(df_feat[cat].values)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE CHECKPOINT — includes df_feat so app.py needs NO CSV files
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[train] Saving checkpoint to {MODEL_PATH} ...")

torch.save({
    # ── feature matrix (eliminates CSV dependency in app.py) ──
    "df_feat":             df_feat,

    # ── column metadata ──
    "all_cols":            all_cols,
    "family_cols":         family_cols,
    "lgbm_feature_cols":   lgbm_feature_cols,

    # ── category splits ──
    "dense_cats":          dense_cats,
    "sparse_cats":         sparse_cats,
    "nearzero_cats":       nearzero_cats,

    # ── hyperparams ──
    "window_size":         WINDOW_SIZE,

    # ── MA values ──
    "ma_values":           ma_values,

    # ── LSTM ──
    "lstm_scaler":         lstm_scaler,
    "lstm_state_dict":     lstm_model.state_dict() if lstm_model else None,
    "lstm_num_features":   len(all_cols),
    "lstm_hidden_size":    HIDDEN_SIZE,
    "lstm_num_targets":    len(dense_cats),
    "lstm_target_indices": target_indices,

    # ── LightGBM (serialised) ──
    "lgbm_models_bytes":        {cat: pickle.dumps(b) for cat, b in lgbm_models.items()},
    "cross_store_feat_cols":    cross_store_feat_cols,
    "cross_store_dfs":          cross_store_dfs,

}, MODEL_PATH)

print(f"[train] ✅ Checkpoint saved — {MODEL_PATH}")
print(f"[train] df_feat shape: {df_feat.shape}  (embedded in checkpoint)")
print("[train] Done! Push store_1_forecast.pth to GitHub — no CSV needed on Render.")