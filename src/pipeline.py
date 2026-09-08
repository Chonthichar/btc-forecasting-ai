
from __future__ import annotations
import os, json, math, random, platform, sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
)
from sklearn.model_selection import TimeSeriesSplit

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from src.features import FEATURE_SETS


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def ensure_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.floor("h")


def load_and_merge(cfg: dict) -> pd.DataFrame:
    market = pd.read_parquet(cfg["paths"]["market"]).copy()
    sent = pd.read_parquet(cfg["paths"]["sentiment"]).copy()

    market["timestamp"] = ensure_utc(market["timestamp"])
    sent["timestamp"] = ensure_utc(sent["timestamp"])

    market = market.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    sent = sent.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    # Sentiment pipeline stores only hours containing accepted news.
    # Determine the period for which the sentiment collection itself has coverage
    # from the configured experiment dates, not from first/last news article.
    start = pd.Timestamp(cfg["experiment"]["start"], tz="UTC")
    end = pd.Timestamp(cfg["experiment"]["end"], tz="UTC") + pd.Timedelta(hours=23)

    sent["has_news"] = 1
    merged = market.merge(sent, on="timestamp", how="left", suffixes=("", "_sent"))

    # Restrict to the configured period BEFORE constructing sentiment lags.
    # This prevents hours before verified sentiment coverage from being encoded
    # as artificial zero-sentiment history at the beginning of the experiment.
    merged = merged[
        (merged["timestamp"] >= start) &
        (merged["timestamp"] <= end)
    ].copy().sort_values("timestamp").reset_index(drop=True)

    merged["coverage_available"] = 1

    # Inside the verified coverage window, a missing sentiment row means
    # no accepted news for that hour, so zero is a meaningful encoding.
    zero_fill = [
        "article_count","weight_sum","positive_count","negative_count","neutral_count",
        "unique_sources","sentiment_score","positive_share","negative_share","neutral_share"
    ]
    for c in zero_fill:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0.0)

    merged["has_news"] = merged["has_news"].fillna(0).astype(int)

    # Confidence/quality do not semantically exist on no-news hours.
    # Encode zero while has_news preserves the distinction.
    for c in ["mean_confidence", "mean_quality"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0.0)

    # Recompute causal sentiment lags only within verified coverage.
    # The first 1/6/24 hours remain NaN and are removed in prepare_dataset().
    for h in [1, 6, 24]:
        merged[f"sentiment_score_lag_{h}h"] = merged["sentiment_score"].shift(h)

    # Causal rolling sentiment features for delayed news effects.
    # All rolling windows use only the current and previous hours; no centered
    # windows and no future values are used. Full windows are required so the
    # first 24 hours become a transparent warm-up period rather than imputed data.
    for w in [3, 6, 12, 24]:
        merged[f"sentiment_mean_{w}h"] = (
            merged["sentiment_score"].rolling(window=w, min_periods=w).mean()
        )

    merged["sentiment_change_3h"] = (
        merged["sentiment_score"] - merged["sentiment_score"].shift(3)
    )
    merged["sentiment_change_6h"] = (
        merged["sentiment_score"] - merged["sentiment_score"].shift(6)
    )

    merged["article_count_6h"] = (
        merged["article_count"].rolling(window=6, min_periods=6).sum()
    )
    merged["article_count_24h"] = (
        merged["article_count"].rolling(window=24, min_periods=24).sum()
    )

    # Rolling polarity shares are computed from article counts rather than by
    # averaging hourly shares, so no-news hours do not receive artificial weight.
    pos6 = merged["positive_count"].rolling(window=6, min_periods=6).sum()
    neg6 = merged["negative_count"].rolling(window=6, min_periods=6).sum()
    articles6 = merged["article_count_6h"]
    merged["positive_share_6h"] = np.where(articles6 > 0, pos6 / articles6, 0.0)
    merged["negative_share_6h"] = np.where(articles6 > 0, neg6 / articles6, 0.0)

    merged["sentiment_volatility_24h"] = (
        merged["sentiment_score"].rolling(window=24, min_periods=24).std(ddof=0)
    )

    return merged.reset_index(drop=True)


def validate_dataset(df: pd.DataFrame, cfg: dict) -> dict:
    market_features = FEATURE_SETS[cfg["features"]["market_set"]]
    sentiment_features = FEATURE_SETS[cfg["features"]["sentiment_set"]]
    required = ["timestamp", cfg["experiment"]["target_column"]] + market_features + sentiment_features

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps remain after merge.")

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not sorted.")

    numeric = df[market_features + sentiment_features].select_dtypes(include="number")
    inf_count = int(np.isinf(numeric.to_numpy(dtype=float)).sum())

    missing = df[required].isna().sum()
    report = {
        "rows_before_dropna": int(len(df)),
        "first_timestamp": str(df["timestamp"].min()),
        "last_timestamp": str(df["timestamp"].max()),
        "duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        "infinite_values": inf_count,
        "missing_by_column": {k: int(v) for k, v in missing.items() if v > 0},
        "target_distribution": {
            str(k): int(v)
            for k, v in df[cfg["experiment"]["target_column"]].value_counts(dropna=False).items()
        },
    }
    return report


def prepare_dataset(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, List[str], List[str]]:
    market_features = FEATURE_SETS[cfg["features"]["market_set"]]
    sentiment_features = FEATURE_SETS[cfg["features"]["sentiment_set"]]
    target = cfg["experiment"]["target_column"]

    needed = market_features + sentiment_features + [target]
    clean = df.dropna(subset=needed).copy().reset_index(drop=True)

    # Do not allow future target columns to leak into feature lists.
    assert not any("target_" in c for c in market_features + sentiment_features)

    return clean, market_features, sentiment_features


def reserve_final_test(
    df: pd.DataFrame, fraction: float, gap: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve the newest fraction as an untouched final test set.

    `gap` rows immediately before the test set are purged from development.
    For an hourly target with horizon h, use gap=h so the final development
    labels cannot depend on prices inside the final test period.
    """
    n = len(df)
    test_start = int(n * (1 - fraction))
    dev_end_exclusive = test_start - int(gap)
    if dev_end_exclusive <= 0 or test_start >= n:
        raise ValueError("final_test_fraction/gap creates an empty dev or test set")
    dev = df.iloc[:dev_end_exclusive].copy()
    purged = df.iloc[dev_end_exclusive:test_start].copy()
    test = df.iloc[test_start:].copy()
    return dev, purged, test


def get_walk_forward_folds(n_rows: int, n_splits: int, gap: int):
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    dummy = np.arange(n_rows)
    return list(tscv.split(dummy))


def binary_metrics(y_true, prob) -> dict:
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)
    y_true = np.asarray(y_true, dtype=int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, prob),
        "mcc": matthews_corrcoef(y_true, pred),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, prob)
    except Exception:
        out["roc_auc"] = float("nan")
    return {k: float(v) for k, v in out.items()}


def baseline_factory(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=350, min_samples_leaf=3, n_jobs=-1,
            class_weight="balanced_subsample", random_state=seed
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, random_state=seed
        )
    raise KeyError(name)


def run_baseline_cv(df: pd.DataFrame, feature_cols: List[str], cfg: dict, model_name: str, seed: int):
    target = cfg["experiment"]["target_column"]
    final_gap = cfg["experiment"].get(
        "final_test_gap_hours", cfg["experiment"].get("target_horizon_hours", 0)
    )
    dev, _, final_test = reserve_final_test(
        df, cfg["experiment"]["final_test_fraction"], final_gap
    )
    folds = get_walk_forward_folds(
        len(dev), cfg["cv"]["n_splits"], cfg["cv"]["gap_hours"]
    )

    fold_metrics, fold_rows = [], []

    for fold_id, (tr_idx, va_idx) in enumerate(folds, 1):
        tr, va = dev.iloc[tr_idx], dev.iloc[va_idx]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(tr[feature_cols])
        Xva = scaler.transform(va[feature_cols])

        model = baseline_factory(model_name, seed)
        model.fit(Xtr, tr[target].astype(int))
        prob = model.predict_proba(Xva)[:, 1] if hasattr(model, "predict_proba") else model.predict(Xva)

        m = binary_metrics(va[target], prob)
        m.update({
            "fold": fold_id,
            "train_start": str(tr["timestamp"].min()),
            "train_end": str(tr["timestamp"].max()),
            "val_start": str(va["timestamp"].min()),
            "val_end": str(va["timestamp"].max()),
        })
        fold_metrics.append(m)

        tmp = pd.DataFrame({
            "timestamp": va["timestamp"].values,
            "y_true": va[target].astype(int).values,
            "prob_up": prob,
            "fold": fold_id,
        })
        fold_rows.append(tmp)

    return pd.DataFrame(fold_metrics), pd.concat(fold_rows, ignore_index=True), dev, final_test


def fit_final_baseline(dev: pd.DataFrame, test: pd.DataFrame, feature_cols: List[str], cfg: dict, model_name: str, seed: int, outdir: Path):
    target = cfg["experiment"]["target_column"]
    scaler = StandardScaler()
    Xdev = scaler.fit_transform(dev[feature_cols])
    Xtest = scaler.transform(test[feature_cols])

    model = baseline_factory(model_name, seed)
    model.fit(Xdev, dev[target].astype(int))
    prob = model.predict_proba(Xtest)[:, 1] if hasattr(model, "predict_proba") else model.predict(Xtest)
    metrics = binary_metrics(test[target], prob)

    joblib.dump(scaler, outdir / "scaler.joblib")
    joblib.dump(model, outdir / "model.joblib")

    preds = pd.DataFrame({
        "timestamp": test["timestamp"].values,
        "y_true": test[target].astype(int).values,
        "prob_up": prob,
        "prediction": (prob >= 0.5).astype(int),
    })
    preds.to_parquet(outdir / "final_test_predictions.parquet", index=False)
    return metrics


def make_sequences(df: pd.DataFrame, feature_cols: List[str], target: str, seq_len: int):
    X, y, ts = [], [], []
    arr = df[feature_cols].to_numpy(np.float32)
    labels = df[target].to_numpy(np.float32)
    times = df["timestamp"].to_numpy()
    for i in range(seq_len - 1, len(df)):
        X.append(arr[i-seq_len+1:i+1])
        y.append(labels[i])
        ts.append(times[i])
    return np.asarray(X), np.asarray(y), np.asarray(ts)


def make_eval_sequences(
    context_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    seq_len: int,
):
    """Build one contiguous sequence per evaluation row.

    `context_df` must be the immediately preceding seq_len-1 hourly rows.
    It may include purged gap rows as features, but those rows never become
    training targets. This preserves real hourly continuity without leakage.
    """
    if len(context_df) != seq_len - 1:
        raise ValueError(
            f"Expected {seq_len-1} context rows, got {len(context_df)}."
        )
    combo = pd.concat([context_df, eval_df], ignore_index=True)
    deltas = combo["timestamp"].diff().dropna()
    if not (deltas == pd.Timedelta(hours=1)).all():
        bad = combo.loc[deltas[deltas != pd.Timedelta(hours=1)].index, "timestamp"]
        raise ValueError(
            "Sequence context is not hourly-contiguous around: "
            + ", ".join(map(str, bad.head(5).tolist()))
        )
    X, y, ts = make_sequences(combo, feature_cols, target, seq_len)
    if len(X) != len(eval_df):
        raise RuntimeError("Evaluation sequence count does not match evaluation rows.")
    return X, y, ts


def build_eval_context(full_df: pd.DataFrame, eval_start_pos: int, seq_len: int) -> pd.DataFrame:
    """Return the immediate seq_len-1 rows before an evaluation block."""
    start = eval_start_pos - (seq_len - 1)
    if start < 0:
        raise ValueError("Not enough historical rows for requested sequence length.")
    return full_df.iloc[start:eval_start_pos].copy()


if TORCH_AVAILABLE:
    class SequenceClassifier(nn.Module):
        def __init__(self, n_features: int, hidden: int = 64, kind: str = "lstm", dropout: float = 0.2):
            super().__init__()
            RNN = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = RNN(n_features, hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1]).squeeze(-1)


    class SinusoidalPositionalEncoding(nn.Module):
        """Deterministic positional encoding so attention knows time order."""
        def __init__(self, d_model: int, max_len: int = 4096):
            super().__init__()
            position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(max_len, d_model, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]


    class VAETransformer(nn.Module):
        """VAE-augmented Transformer with an optional temporal sentiment branch.

        Market features are encoded per time step into a variational latent z.
        A decoder reconstructs the standardized market inputs, giving the VAE a
        genuine reconstruction objective. Positional encoding is added before
        Transformer attention. Sentiment, when enabled, is encoded through a GRU
        over the same contiguous sequence and fused with the market representation.
        """
        def __init__(
            self,
            n_market: int,
            n_sent: int,
            latent: int = 24,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.15,
        ):
            super().__init__()
            if d_model % nhead != 0:
                raise ValueError("transformer d_model must be divisible by nhead")

            enc_hidden = max(64, d_model)
            self.market_enc = nn.Sequential(
                nn.Linear(n_market, enc_hidden),
                nn.GELU(),
                nn.LayerNorm(enc_hidden),
            )
            self.mu = nn.Linear(enc_hidden, latent)
            self.logvar = nn.Linear(enc_hidden, latent)

            self.decoder = nn.Sequential(
                nn.Linear(latent, enc_hidden),
                nn.GELU(),
                nn.Linear(enc_hidden, n_market),
            )

            self.market_proj = nn.Linear(latent, d_model)
            self.positional = SinusoidalPositionalEncoding(d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.market_norm = nn.LayerNorm(d_model)

            self.n_sent = n_sent
            if n_sent > 0:
                sent_hidden = max(16, d_model // 2)
                self.sent_gru = nn.GRU(n_sent, sent_hidden, batch_first=True)
                self.sent_norm = nn.LayerNorm(sent_hidden)
                fusion_in = d_model + sent_hidden
            else:
                self.sent_gru = None
                self.sent_norm = None
                fusion_in = d_model

            self.head = nn.Sequential(
                nn.LayerNorm(fusion_in),
                nn.Dropout(dropout),
                nn.Linear(fusion_in, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def reparam(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, market_seq, sent_seq=None):
            h = self.market_enc(market_seq)
            mu = self.mu(h)
            logvar = torch.clamp(self.logvar(h), min=-10.0, max=10.0)
            z = self.reparam(mu, logvar) if self.training else mu

            reconstruction = self.decoder(z)

            tokens = self.market_proj(z)
            tokens = self.positional(tokens)
            tokens = self.transformer(tokens)
            market_repr = self.market_norm(tokens[:, -1])

            if self.n_sent > 0:
                if sent_seq is None:
                    raise ValueError("sent_seq is required when n_sent > 0")
                _, h_sent = self.sent_gru(sent_seq)
                sent_repr = self.sent_norm(h_sent[-1])
                fused = torch.cat([market_repr, sent_repr], dim=-1)
            else:
                fused = market_repr

            logits = self.head(fused).squeeze(-1)
            return logits, reconstruction, mu, logvar


def _deep_params(cfg: dict) -> dict:
    p = cfg.get("deep_model", {})
    return {
        "rnn_hidden": int(p.get("rnn_hidden", 64)),
        "dropout": float(p.get("dropout", 0.2)),
        "vae_latent_dim": int(p.get("vae_latent_dim", 24)),
        "transformer_d_model": int(p.get("transformer_d_model", 64)),
        "transformer_heads": int(p.get("transformer_heads", 4)),
        "transformer_layers": int(p.get("transformer_layers", 2)),
        "transformer_dropout": float(p.get("transformer_dropout", 0.15)),
        "vae_kl_beta": float(p.get("vae_kl_beta", 1e-3)),
        "vae_reconstruction_weight": float(p.get("vae_reconstruction_weight", 0.10)),
    }


def _scale_train_context_eval(train_df, context_df, eval_df, cols):
    scaler = StandardScaler().fit(train_df[cols])
    tr = train_df.copy()
    ctx = context_df.copy()
    ev = eval_df.copy()
    # Cast first so integer indicators such as has_news can safely receive
    # standardized floating-point values without pandas dtype warnings.
    tr[cols] = tr[cols].astype(float)
    ctx[cols] = ctx[cols].astype(float)
    ev[cols] = ev[cols].astype(float)
    tr.loc[:, cols] = scaler.transform(train_df[cols])
    ctx.loc[:, cols] = scaler.transform(context_df[cols])
    ev.loc[:, cols] = scaler.transform(eval_df[cols])
    return scaler, tr, ctx, ev


def train_torch_sequence_model(
    train_df,
    val_context_df,
    val_df,
    feature_cols,
    cfg,
    kind,
    seed,
    *,
    epochs_override=None,
    early_stopping=True,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")
    set_seed(seed)
    target = cfg["experiment"]["target_column"]
    seq_len = cfg["experiment"]["sequence_length"]
    p = _deep_params(cfg)

    scaler, tr, ctx, va = _scale_train_context_eval(
        train_df, val_context_df, val_df, feature_cols
    )

    Xtr, ytr, _ = make_sequences(tr, feature_cols, target, seq_len)
    Xva, yva, tsva = make_eval_sequences(ctx, va, feature_cols, target, seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceClassifier(
        len(feature_cols), hidden=p["rnn_hidden"], kind=kind, dropout=p["dropout"]
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    epochs = int(epochs_override or cfg["training"]["epochs"])
    min_delta = float(cfg["training"].get("min_delta", 1e-4))
    best_state, best_val_loss, best_epoch, bad = None, float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        if early_stopping:
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.tensor(Xva).to(device))
                val_y = torch.tensor(yva, dtype=torch.float32).to(device)
                val_loss = float(loss_fn(val_logits, val_y).cpu())
            if val_loss < (best_val_loss - min_delta):
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= cfg["training"]["patience"]:
                    break

    if early_stopping:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(Xva).to(device))).cpu().numpy()

    info = {
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epoch),
        "early_stopping_monitor": "validation_bce_loss",
        "best_validation_loss": (float(best_val_loss) if early_stopping else None),
    }
    return model, scaler, binary_metrics(yva, prob), pd.DataFrame({
        "timestamp": tsva, "y_true": yva.astype(int), "prob_up": prob
    }), info


def train_vae_transformer_fold(
    train_df,
    val_context_df,
    val_df,
    market_cols,
    sent_cols,
    cfg,
    seed,
    *,
    epochs_override=None,
    early_stopping=True,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")
    set_seed(seed)
    target = cfg["experiment"]["target_column"]
    seq_len = cfg["experiment"]["sequence_length"]
    p = _deep_params(cfg)

    market_scaler, tr, ctx, va = _scale_train_context_eval(
        train_df, val_context_df, val_df, market_cols
    )

    if sent_cols:
        sent_scaler = StandardScaler().fit(train_df[sent_cols])
        tr[sent_cols] = tr[sent_cols].astype(float)
        ctx[sent_cols] = ctx[sent_cols].astype(float)
        va[sent_cols] = va[sent_cols].astype(float)
        tr.loc[:, sent_cols] = sent_scaler.transform(train_df[sent_cols])
        ctx.loc[:, sent_cols] = sent_scaler.transform(val_context_df[sent_cols])
        va.loc[:, sent_cols] = sent_scaler.transform(val_df[sent_cols])
    else:
        sent_scaler = None

    Xm_tr, ytr, _ = make_sequences(tr, market_cols, target, seq_len)
    Xm_va, yva, tsva = make_eval_sequences(ctx, va, market_cols, target, seq_len)

    if sent_cols:
        Xs_tr, _, _ = make_sequences(tr, sent_cols, target, seq_len)
        Xs_va, _, _ = make_eval_sequences(ctx, va, sent_cols, target, seq_len)
    else:
        Xs_tr = Xs_va = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAETransformer(
        len(market_cols),
        len(sent_cols),
        latent=p["vae_latent_dim"],
        d_model=p["transformer_d_model"],
        nhead=p["transformer_heads"],
        num_layers=p["transformer_layers"],
        dropout=p["transformer_dropout"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    beta = p["vae_kl_beta"]
    recon_w = p["vae_reconstruction_weight"]

    tensors = [torch.tensor(Xm_tr)]
    if sent_cols:
        tensors.append(torch.tensor(Xs_tr))
    tensors.append(torch.tensor(ytr))
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    epochs = int(epochs_override or cfg["training"]["epochs"])
    min_delta = float(cfg["training"].get("min_delta", 1e-4))
    best_state, best_val_loss, best_epoch, bad = None, float("inf"), 0, 0
    last_components = {}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_cls = epoch_kl = epoch_recon = 0.0
        batches = 0
        for batch in loader:
            if sent_cols:
                xm, xs, yb = batch
                xs = xs.to(device)
            else:
                xm, yb = batch
                xs = None
            xm, yb = xm.to(device), yb.to(device)

            opt.zero_grad()
            logits, reconstruction, mu, logvar = model(xm, xs)
            cls_loss = bce(logits, yb)
            recon_loss = mse(reconstruction, xm)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = cls_loss + recon_w * recon_loss + beta * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            epoch_cls += float(cls_loss.detach().cpu())
            epoch_recon += float(recon_loss.detach().cpu())
            epoch_kl += float(kl.detach().cpu())
            batches += 1

        last_components = {
            "classification_loss": epoch_cls / max(batches, 1),
            "reconstruction_loss": epoch_recon / max(batches, 1),
            "kl_loss": epoch_kl / max(batches, 1),
        }

        if early_stopping:
            model.eval()
            with torch.no_grad():
                xm = torch.tensor(Xm_va).to(device)
                xs = torch.tensor(Xs_va).to(device) if sent_cols else None
                yv = torch.tensor(yva, dtype=torch.float32).to(device)
                logits, reconstruction, mu_v, logvar_v = model(xm, xs)
                val_cls = bce(logits, yv)
                val_recon = mse(reconstruction, xm)
                val_kl = -0.5 * torch.mean(1 + logvar_v - mu_v.pow(2) - logvar_v.exp())
                val_loss = float((val_cls + recon_w * val_recon + beta * val_kl).cpu())
            if val_loss < (best_val_loss - min_delta):
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= cfg["training"]["patience"]:
                    break

    if early_stopping:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs

    model.eval()
    with torch.no_grad():
        xm = torch.tensor(Xm_va).to(device)
        xs = torch.tensor(Xs_va).to(device) if sent_cols else None
        logits, reconstruction, mu, logvar = model(xm, xs)
        prob = torch.sigmoid(logits).cpu().numpy()
        eval_recon = float(mse(reconstruction, xm).cpu())
        eval_kl = float((-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())).cpu())

    info = {
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epoch),
        "train_loss_components_last_epoch": last_components,
        "validation_reconstruction_loss": eval_recon,
        "validation_kl_loss": eval_kl,
        "vae_kl_beta": beta,
        "vae_reconstruction_weight": recon_w,
        "early_stopping_monitor": "validation_total_vae_loss",
        "best_validation_loss": (float(best_val_loss) if early_stopping else None),
    }
    return model, market_scaler, sent_scaler, binary_metrics(yva, prob), pd.DataFrame({
        "timestamp": tsva, "y_true": yva.astype(int), "prob_up": prob
    }), info


def save_torch_state(model, path: Path):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")
    torch.save(model.state_dict(), path)


def environment_info() -> dict:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch_available": TORCH_AVAILABLE,
    }
    if TORCH_AVAILABLE:
        info.update({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    return info
