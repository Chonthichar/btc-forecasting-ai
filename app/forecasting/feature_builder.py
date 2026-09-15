"""Feature preparation for the frozen deployment models.

The nine derived features below are reproduced verbatim from the training code
(integration brief section 11). They must not be approximated or renamed: a
changed denominator still yields a confident number, just a wrong one, and
nothing downstream would raise.

Feature ORDER always comes from the shipped CSVs, never from DataFrame column
order, so a reordering upstream cannot silently permute the model inputs.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


class FeatureError(RuntimeError):
    """Raised when live data cannot support a valid prediction."""


def add_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """The exact Version-B derived features used to train the direction models.

    Requires the base indicators (sma_24, sma_168, macd) to exist already.
    """
    missing = [c for c in ("open", "high", "low", "close", "volume", "quote_volume",
                           "num_trades", "sma_24", "sma_168", "macd") if c not in df.columns]
    if missing:
        raise FeatureError(f"base indicators missing before derived features: {missing}")

    out = df.copy()
    out["open_close_log"] = np.log(out["open"].clip(lower=EPS) / out["close"].clip(lower=EPS))
    out["high_close_log"] = np.log(out["high"].clip(lower=EPS) / out["close"].clip(lower=EPS))
    out["low_close_log"] = np.log(out["low"].clip(lower=EPS) / out["close"].clip(lower=EPS))
    out["volume_logchg_1h"] = np.log1p(out["volume"].clip(lower=0)).diff()
    out["quote_volume_logchg_1h"] = np.log1p(out["quote_volume"].clip(lower=0)).diff()
    out["num_trades_logchg_1h"] = np.log1p(out["num_trades"].clip(lower=0)).diff()
    out["close_sma24_log"] = np.log(out["close"].clip(lower=EPS) / out["sma_24"].clip(lower=EPS))
    out["close_sma168_log"] = np.log(out["close"].clip(lower=EPS) / out["sma_168"].clip(lower=EPS))
    out["macd_pct"] = out["macd"] / out["close"].clip(lower=EPS)
    return out


def read_feature_order(path: Path) -> list[str]:
    """Feature names in their stored index order."""
    with open(path, encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: int(r["feature_index"]))
    names = [r["feature"] for r in rows]
    if not names:
        raise FeatureError(f"empty feature list: {path}")
    return names


def validate_timeline(df: pd.DataFrame) -> None:
    """Brief section 12: refuse to predict rather than predict on bad data."""
    if "timestamp" not in df.columns:
        raise FeatureError("market frame has no timestamp column")
    stamps = df["timestamp"]
    if stamps.duplicated().any():
        raise FeatureError("duplicate candle timestamps in the market frame")
    if not stamps.is_monotonic_increasing:
        raise FeatureError("candles are not in ascending time order")


def latest_row(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """The most recent closed candle as one finite feature vector."""
    absent = [f for f in features if f not in df.columns]
    if absent:
        raise FeatureError(f"missing required features: {absent}")
    values = df.iloc[[-1]][features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [f for f, ok in zip(features, np.isfinite(values[0])) if not ok]
        raise FeatureError(f"latest candle has non-finite features: {bad}")
    return values


def latest_sequence(df: pd.DataFrame, features: list[str], length: int) -> np.ndarray:
    """The last `length` closed candles, proven hourly-contiguous and finite."""
    absent = [f for f in features if f not in df.columns]
    if absent:
        raise FeatureError(f"missing required features: {absent}")
    if len(df) < length:
        raise FeatureError(f"need {length} candles for the sequence, have {len(df)}")

    window = df.iloc[-length:]
    steps = window["timestamp"].diff().dropna()
    if not (steps == pd.Timedelta(hours=1)).all():
        raise FeatureError("the model input window is not hourly-contiguous")

    values = window[features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        rows, cols = np.where(~np.isfinite(values))
        bad = sorted({features[c] for c in cols})
        raise FeatureError(f"sequence has non-finite features: {bad}")
    return values
