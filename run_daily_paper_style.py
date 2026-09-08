from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


MARKET_FEATURES = [
    "open", "high", "low", "close", "volume", "quote_volume", "num_trades",
    "log_return_1d", "momentum_3d", "momentum_7d",
    "close_vs_sma_7", "close_vs_sma_14",
    "volatility_7d", "volatility_14d", "volume_zscore_7d",
    "range_pct", "body_pct",
]

SENTIMENT_FEATURES = [
    "article_count", "sentiment_score", "positive_share", "negative_share",
    "neutral_share", "mean_confidence", "has_news",
    "sentiment_lag_1d", "sentiment_lag_3d", "sentiment_lag_7d",
    "sentiment_mean_3d", "sentiment_mean_7d",
    "sentiment_change_3d", "sentiment_change_7d",
    "article_count_3d", "article_count_7d",
    "positive_share_7d", "negative_share_7d",
    "sentiment_volatility_7d",
]


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _ensure_utc_day(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.floor("D")


def build_daily_dataset(cfg: dict) -> pd.DataFrame:
    market = pd.read_parquet(cfg["paths"]["market_hourly"]).copy()
    sent = pd.read_parquet(cfg["paths"]["sentiment_daily"]).copy()

    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    sent["timestamp"] = _ensure_utc_day(sent["timestamp"])
    market["date"] = market["timestamp"].dt.floor("D")

    # Build daily OHLCV strictly from data observed on day t.
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    for c in ["quote_volume", "num_trades"]:
        if c in market.columns:
            agg[c] = "sum"
    daily = market.groupby("date", as_index=False).agg(agg).rename(columns={"date": "timestamp"})
    for c in ["quote_volume", "num_trades"]:
        if c not in daily.columns:
            daily[c] = 0.0

    start = pd.Timestamp(cfg["experiment"]["start"], tz="UTC")
    end = pd.Timestamp(cfg["experiment"]["end"], tz="UTC")
    days = pd.DataFrame({"timestamp": pd.date_range(start, end, freq="D", tz="UTC")})
    daily = days.merge(daily, on="timestamp", how="left")

    # Market daily features.
    daily["log_return_1d"] = np.log(daily["close"] / daily["close"].shift(1))
    daily["momentum_3d"] = daily["close"].pct_change(3, fill_method=None)
    daily["momentum_7d"] = daily["close"].pct_change(7, fill_method=None)
    sma7 = daily["close"].rolling(7, min_periods=7).mean()
    sma14 = daily["close"].rolling(14, min_periods=14).mean()
    daily["close_vs_sma_7"] = daily["close"] / sma7 - 1.0
    daily["close_vs_sma_14"] = daily["close"] / sma14 - 1.0
    daily["volatility_7d"] = daily["log_return_1d"].rolling(7, min_periods=7).std(ddof=0)
    daily["volatility_14d"] = daily["log_return_1d"].rolling(14, min_periods=14).std(ddof=0)
    vmean = daily["volume"].rolling(7, min_periods=7).mean()
    vstd = daily["volume"].rolling(7, min_periods=7).std(ddof=0).replace(0, np.nan)
    daily["volume_zscore_7d"] = (daily["volume"] - vmean) / vstd
    daily["range_pct"] = (daily["high"] - daily["low"]) / daily["close"].replace(0, np.nan)
    daily["body_pct"] = (daily["close"] - daily["open"]) / daily["open"].replace(0, np.nan)

    # Sentiment file contains only days with accepted news. Inside verified
    # coverage, missing date = no accepted news, not neutral sentiment.
    sent = sent.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    sent["has_news"] = 1
    sent_cols = [
        "timestamp", "article_count", "mean_confidence", "positive_count",
        "negative_count", "neutral_count", "sentiment_score",
        "positive_share", "negative_share", "neutral_share", "has_news",
    ]
    sent = sent[[c for c in sent_cols if c in sent.columns]]
    daily = daily.merge(sent, on="timestamp", how="left")

    for c in [
        "article_count", "positive_count", "negative_count", "neutral_count",
        "sentiment_score", "positive_share", "negative_share", "neutral_share",
        "mean_confidence",
    ]:
        if c not in daily.columns:
            daily[c] = 0.0
        daily[c] = daily[c].fillna(0.0)
    daily["has_news"] = daily["has_news"].fillna(0).astype(int)

    for lag in [1, 3, 7]:
        daily[f"sentiment_lag_{lag}d"] = daily["sentiment_score"].shift(lag)
    for w in [3, 7]:
        daily[f"sentiment_mean_{w}d"] = daily["sentiment_score"].rolling(w, min_periods=w).mean()
        daily[f"article_count_{w}d"] = daily["article_count"].rolling(w, min_periods=w).sum()
    daily["sentiment_change_3d"] = daily["sentiment_score"] - daily["sentiment_score"].shift(3)
    daily["sentiment_change_7d"] = daily["sentiment_score"] - daily["sentiment_score"].shift(7)

    pos7 = daily["positive_count"].rolling(7, min_periods=7).sum()
    neg7 = daily["negative_count"].rolling(7, min_periods=7).sum()
    art7 = daily["article_count_7d"]
    daily["positive_share_7d"] = np.where(art7 > 0, pos7 / art7, 0.0)
    daily["negative_share_7d"] = np.where(art7 > 0, neg7 / art7, 0.0)
    daily["sentiment_volatility_7d"] = daily["sentiment_score"].rolling(7, min_periods=7).std(ddof=0)

    # Targets are strictly after day t.
    daily["target_next_close"] = daily["close"].shift(-1)
    daily["target_return_1d"] = np.log(daily["close"].shift(-1) / daily["close"])
    daily["target_direction_1d"] = (daily["target_return_1d"] > 0).astype("Int64")
    daily.loc[daily["target_return_1d"].isna(), "target_direction_1d"] = pd.NA

    needed = MARKET_FEATURES + SENTIMENT_FEATURES + [
        "target_next_close", "target_return_1d", "target_direction_1d"
    ]
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).reset_index(drop=True)
    return daily


def feature_sets():
    return {
        "market_only": MARKET_FEATURES,
        "sentiment_only": SENTIMENT_FEATURES,
        "market_plus_sentiment": MARKET_FEATURES + SENTIMENT_FEATURES,
    }


def split_final(df: pd.DataFrame, fraction: float, gap: int):
    test_start = int(len(df) * (1 - fraction))
    dev_end = test_start - gap
    if dev_end <= 0:
        raise ValueError("Final split leaves no development rows")
    return df.iloc[:dev_end].copy(), df.iloc[dev_end:test_start].copy(), df.iloc[test_start:].copy()


def folds_for(n: int, n_splits: int, gap: int):
    return list(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(np.arange(n)))


def cls_metrics(y, prob):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, prob),
        "mcc": matthews_corrcoef(y, pred),
    }
    try:
        out["roc_auc"] = roc_auc_score(y, prob)
    except Exception:
        out["roc_auc"] = float("nan")
    return {k: float(v) for k, v in out.items()}


def reg_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mae = mean_absolute_error(y, pred)
    rmse = math.sqrt(mean_squared_error(y, pred))
    nonzero = np.abs(y) > 1e-12
    mape = float(np.mean(np.abs((y[nonzero] - pred[nonzero]) / y[nonzero]))) if nonzero.any() else float("nan")
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2_score(y, pred)),
    }


def cls_model(name, seed):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
    raise KeyError(name)


def reg_model(name, seed):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
    raise KeyError(name)


if TORCH_AVAILABLE:
    class GRUHead(nn.Module):
        def __init__(self, n_features, hidden=48, dropout=0.2):
            super().__init__()
            self.gru = nn.GRU(n_features, hidden, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1))
        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1]).squeeze(-1)


def _seq(df, cols, target, seq_len):
    a = df[cols].to_numpy(np.float32)
    y = df[target].to_numpy(np.float32)
    X, Y = [], []
    for i in range(seq_len - 1, len(df)):
        X.append(a[i-seq_len+1:i+1])
        Y.append(y[i])
    return np.asarray(X), np.asarray(Y)


def _eval_seq(context, ev, cols, target, seq_len):
    combo = pd.concat([context, ev], ignore_index=True)
    X, y = _seq(combo, cols, target, seq_len)
    return X, y[-len(ev):]


def fit_gru(train, context, val, cols, target, cfg, task="classification"):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch unavailable")
    seed = int(cfg["training"]["seed"])
    set_seed(seed)
    seq_len = int(cfg["experiment"]["sequence_length_days"])
    scaler = StandardScaler().fit(train[cols])
    tr, ctx, va = train.copy(), context.copy(), val.copy()
    tr[cols] = tr[cols].astype(float)
    ctx[cols] = ctx[cols].astype(float)
    va[cols] = va[cols].astype(float)
    tr.loc[:, cols] = scaler.transform(train[cols])
    ctx.loc[:, cols] = scaler.transform(context[cols])
    va.loc[:, cols] = scaler.transform(val[cols])

    target_scaler = None
    if task == "regression":
        target_scaler = StandardScaler().fit(train[[target]])
        tr.loc[:, target] = target_scaler.transform(train[[target]]).ravel()
        ctx.loc[:, target] = target_scaler.transform(context[[target]]).ravel()
        va.loc[:, target] = target_scaler.transform(val[[target]]).ravel()

    Xtr, ytr = _seq(tr, cols, target, seq_len)
    Xva, yva = _eval_seq(ctx, va, cols, target, seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUHead(len(cols), int(cfg["training"]["gru_hidden"]), float(cfg["training"]["dropout"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    best_state, best_loss, best_epoch, bad = None, float("inf"), 0, 0
    max_epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"]["patience"])
    min_delta = float(cfg["training"]["min_delta"])
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(torch.tensor(Xva).to(device))
            lv = float(loss_fn(pv, torch.tensor(yva).to(device)).cpu())
        if lv < best_loss - min_delta:
            best_loss, best_epoch, bad = lv, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(Xva).to(device)).cpu().numpy()
    if task == "classification":
        pred = 1 / (1 + np.exp(-raw))
        y_true = yva
    else:
        pred = target_scaler.inverse_transform(raw.reshape(-1, 1)).ravel()
        y_true = target_scaler.inverse_transform(yva.reshape(-1, 1)).ravel()
    return model, scaler, target_scaler, y_true, pred, {"best_epoch": best_epoch, "epochs_ran": epoch, "best_validation_loss": best_loss}



def fit_gru_fixed(train, context, ev, cols, target, cfg, task, epochs):
    """Train on all development rows for a CV-selected fixed epoch count.

    No early stopping or test-based model selection is performed here. The
    evaluation block is used only after the epoch count has already been fixed.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch unavailable")
    seed = int(cfg["training"]["seed"])
    set_seed(seed)
    seq_len = int(cfg["experiment"]["sequence_length_days"])
    scaler = StandardScaler().fit(train[cols])
    tr, ctx, vv = train.copy(), context.copy(), ev.copy()
    tr[cols] = tr[cols].astype(float)
    ctx[cols] = ctx[cols].astype(float)
    vv[cols] = vv[cols].astype(float)
    tr.loc[:, cols] = scaler.transform(train[cols])
    ctx.loc[:, cols] = scaler.transform(context[cols])
    vv.loc[:, cols] = scaler.transform(ev[cols])

    target_scaler = None
    if task == "regression":
        target_scaler = StandardScaler().fit(train[[target]])
        tr.loc[:, target] = target_scaler.transform(train[[target]]).ravel()
        ctx.loc[:, target] = target_scaler.transform(context[[target]]).ravel()
        vv.loc[:, target] = target_scaler.transform(ev[[target]]).ravel()

    Xtr, ytr = _seq(tr, cols, target, seq_len)
    Xev, yev = _eval_seq(ctx, vv, cols, target, seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUHead(len(cols), int(cfg["training"]["gru_hidden"]), float(cfg["training"]["dropout"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=int(cfg["training"]["batch_size"]), shuffle=False)
    for _ in range(int(epochs)):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(Xev).to(device)).cpu().numpy()
    if task == "classification":
        pred = 1 / (1 + np.exp(-raw))
        y_true = yev
    else:
        pred = target_scaler.inverse_transform(raw.reshape(-1, 1)).ravel()
        y_true = target_scaler.inverse_transform(yev.reshape(-1, 1)).ravel()
    return model, scaler, target_scaler, y_true, pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/btc_daily_paper_style.yaml")
    ap.add_argument("--stage", choices=["validate", "classify", "regress", "all"], default="all")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    seed = int(cfg["training"]["seed"])
    set_seed(seed)

    out = Path(cfg["paths"]["output_root"]) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    df = build_daily_dataset(cfg)
    df.to_parquet(out / "daily_aligned_dataset.parquet", index=False)

    dev, purge, test = split_final(df, float(cfg["experiment"]["final_test_fraction"]), int(cfg["experiment"]["final_test_gap_days"]))
    folds = folds_for(len(dev), int(cfg["cv"]["n_splits"]), int(cfg["cv"]["gap_days"]))
    with open(out / "daily_manifest.json", "w") as f:
        json.dump({
            "rows": len(df), "start": str(df.timestamp.min()), "end": str(df.timestamp.max()),
            "dev_rows": len(dev), "purge_rows": len(purge), "test_rows": len(test),
            "market_features": MARKET_FEATURES, "sentiment_features": SENTIMENT_FEATURES,
            "feature_sets": {k: len(v) for k, v in feature_sets().items()},
            "targets": ["target_next_close", "target_return_1d", "target_direction_1d"],
            "max_epochs": cfg["training"]["epochs"], "patience": cfg["training"]["patience"],
        }, f, indent=2)

    print("Daily dataset ready")
    print(" rows:", len(df), "|", df.timestamp.min(), "->", df.timestamp.max())
    print(" feature sets:", {k: len(v) for k, v in feature_sets().items()})
    print(" dev:", len(dev), "| purge:", len(purge), "| final test:", len(test))
    if args.stage == "validate":
        return

    all_cv, final_rows = [], []

    if args.stage in {"classify", "all"}:
        target = "target_direction_1d"
        for fs_name, cols in feature_sets().items():
            for model_name in cfg["models"]["classification"]:
                fold_rows = []
                best_epochs = []
                for fold_id, (tr_idx, va_idx) in enumerate(folds, 1):
                    tr, va = dev.iloc[tr_idx].copy(), dev.iloc[va_idx].copy()
                    if model_name == "gru":
                        seq_len = int(cfg["experiment"]["sequence_length_days"])
                        start_pos = int(va_idx[0]) - (seq_len - 1)
                        if start_pos < 0:
                            continue
                        ctx = dev.iloc[start_pos:int(va_idx[0])].copy()
                        _, _, _, yv, prob, info = fit_gru(tr, ctx, va, cols, target, cfg, "classification")
                        m = cls_metrics(yv, prob)
                        m.update(info)
                        best_epochs.append(int(info["best_epoch"]))
                    else:
                        sc = StandardScaler().fit(tr[cols])
                        Xtr, Xva = sc.transform(tr[cols]), sc.transform(va[cols])
                        model = cls_model(model_name, seed)
                        model.fit(Xtr, tr[target].astype(int))
                        prob = model.predict_proba(Xva)[:, 1]
                        m = cls_metrics(va[target], prob)
                    m.update({"task": "direction_classification", "model": model_name, "feature_set": fs_name, "fold": fold_id})
                    fold_rows.append(m)
                    all_cv.append(m)

                if model_name == "gru":
                    seq_len = int(cfg["experiment"]["sequence_length_days"])
                    test_start = int(test.index[0])
                    ctx = df.iloc[test_start-(seq_len-1):test_start].copy()
                    fixed_epochs = max(1, int(round(np.median(best_epochs))))
                    model, sc, ys, y_true, prob = fit_gru_fixed(
                        dev, ctx, test, cols, target, cfg, "classification", fixed_epochs
                    )
                    m = cls_metrics(y_true, prob)
                    m.update({
                        "task": "direction_classification", "model": model_name,
                        "feature_set": fs_name, "split": "final_holdout",
                        "fixed_epochs_from_cv_median": fixed_epochs,
                    })
                    final_rows.append(m)
                else:
                    sc = StandardScaler().fit(dev[cols])
                    model = cls_model(model_name, seed)
                    model.fit(sc.transform(dev[cols]), dev[target].astype(int))
                    prob = model.predict_proba(sc.transform(test[cols]))[:, 1]
                    m = cls_metrics(test[target], prob)
                    m.update({"task": "direction_classification", "model": model_name, "feature_set": fs_name, "split": "final_holdout"})
                    final_rows.append(m)

    if args.stage in {"regress", "all"}:
        for target in ["target_return_1d", "target_next_close"]:
            for fs_name, cols in feature_sets().items():
                for model_name in cfg["models"]["regression"]:
                    best_epochs = []
                    for fold_id, (tr_idx, va_idx) in enumerate(folds, 1):
                        tr, va = dev.iloc[tr_idx].copy(), dev.iloc[va_idx].copy()
                        if model_name == "gru":
                            seq_len = int(cfg["experiment"]["sequence_length_days"])
                            start_pos = int(va_idx[0]) - (seq_len - 1)
                            if start_pos < 0:
                                continue
                            ctx = dev.iloc[start_pos:int(va_idx[0])].copy()
                            _, _, _, yv, pred, info = fit_gru(tr, ctx, va, cols, target, cfg, "regression")
                            m = reg_metrics(yv, pred)
                            m.update(info)
                            best_epochs.append(int(info["best_epoch"]))
                        else:
                            sc = StandardScaler().fit(tr[cols])
                            model = reg_model(model_name, seed)
                            model.fit(sc.transform(tr[cols]), tr[target])
                            pred = model.predict(sc.transform(va[cols]))
                            m = reg_metrics(va[target], pred)
                        # Convert regression predictions into directional accuracy too.
                        if target == "target_return_1d":
                            ydir = (np.asarray(yv if model_name == "gru" else va[target]) > 0).astype(int)
                            pdir = (np.asarray(pred) > 0).astype(int)
                        else:
                            current_close = va["close"].to_numpy()[-len(pred):]
                            ydir = (np.asarray(yv if model_name == "gru" else va[target]) > current_close).astype(int)
                            pdir = (np.asarray(pred) > current_close).astype(int)
                        m["directional_accuracy"] = float(accuracy_score(ydir, pdir))
                        if target == "target_next_close":
                            m["mape_accuracy_pct"] = float(100 * (1 - m["mape"]))
                        m.update({"task": target, "model": model_name, "feature_set": fs_name, "fold": fold_id})
                        all_cv.append(m)

                    if model_name == "gru":
                        seq_len = int(cfg["experiment"]["sequence_length_days"])
                        test_start = int(test.index[0])
                        ctx = df.iloc[test_start-(seq_len-1):test_start].copy()
                        fixed_epochs = max(1, int(round(np.median(best_epochs))))
                        _, _, _, y_true, pred = fit_gru_fixed(
                            dev, ctx, test, cols, target, cfg, "regression", fixed_epochs
                        )
                        m = reg_metrics(y_true, pred)
                        if target == "target_return_1d":
                            m["directional_accuracy"] = float(accuracy_score((y_true > 0).astype(int), (pred > 0).astype(int)))
                        else:
                            current_close = test["close"].to_numpy()[-len(pred):]
                            m["directional_accuracy"] = float(accuracy_score((y_true > current_close).astype(int), (pred > current_close).astype(int)))
                            m["mape_accuracy_pct"] = float(100 * (1 - m["mape"]))
                        m.update({
                            "task": target, "model": model_name, "feature_set": fs_name,
                            "split": "final_holdout", "fixed_epochs_from_cv_median": fixed_epochs,
                        })
                        final_rows.append(m)
                        continue
                    sc = StandardScaler().fit(dev[cols])
                    model = reg_model(model_name, seed)
                    model.fit(sc.transform(dev[cols]), dev[target])
                    pred = model.predict(sc.transform(test[cols]))
                    m = reg_metrics(test[target], pred)
                    if target == "target_return_1d":
                        m["directional_accuracy"] = float(accuracy_score((test[target] > 0).astype(int), (pred > 0).astype(int)))
                    else:
                        m["directional_accuracy"] = float(accuracy_score((test[target] > test["close"]).astype(int), (pred > test["close"].to_numpy()).astype(int)))
                        m["mape_accuracy_pct"] = float(100 * (1 - m["mape"]))
                    m.update({"task": target, "model": model_name, "feature_set": fs_name, "split": "final_holdout"})
                    final_rows.append(m)

        # Naive price persistence baseline: tomorrow's close = today's close.
        pred = test["close"].to_numpy()
        m = reg_metrics(test["target_next_close"], pred)
        m["directional_accuracy"] = float(accuracy_score(test["target_direction_1d"].astype(int), np.zeros(len(test), dtype=int)))
        m["mape_accuracy_pct"] = float(100 * (1 - m["mape"]))
        m.update({"task": "target_next_close", "model": "naive_persistence", "feature_set": "current_close_only", "split": "final_holdout"})
        final_rows.append(m)

        # Zero-return baseline.
        pred_r = np.zeros(len(test))
        m = reg_metrics(test["target_return_1d"], pred_r)
        m["directional_accuracy"] = float(accuracy_score(test["target_direction_1d"].astype(int), np.zeros(len(test), dtype=int)))
        m.update({"task": "target_return_1d", "model": "zero_return", "feature_set": "none", "split": "final_holdout"})
        final_rows.append(m)

    cv_df = pd.DataFrame(all_cv)
    final_df = pd.DataFrame(final_rows)
    cv_df.to_csv(out / "daily_cv_results.csv", index=False)
    final_df.to_csv(out / "daily_final_holdout_results.csv", index=False)

    if not cv_df.empty:
        summary = cv_df.groupby(["task", "model", "feature_set"]).mean(numeric_only=True).reset_index()
        summary.to_csv(out / "daily_cv_summary.csv", index=False)
        print("\nDaily CV summary:")
        cols = [c for c in ["task", "model", "feature_set", "accuracy", "f1", "roc_auc", "mae", "rmse", "r2", "directional_accuracy", "mape_accuracy_pct"] if c in summary.columns]
        print(summary[cols].to_string(index=False))
    if not final_df.empty:
        print("\nFinal holdout summary (including GRU retrained for CV-selected fixed epochs):")
        print(final_df.to_string(index=False))
    print("\nOutputs saved to:", out)


if __name__ == "__main__":
    main()
