"""Produces all six deployment predictions from one live market frame.

Reuses the existing MarketDataAgent, so closed-candle handling, the Binance
mirror fallback and the indicator set are shared with the rest of the app
rather than duplicated here.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ..agents.market_agent import MarketDataAgent
from .feature_builder import FeatureError, add_stationary_features, validate_timeline
from .model_loader import HORIZONS, DeploymentModels
from .predictors import predict_direction, predict_risk

# Walk-forward research diagnostics from the integration brief (section 15).
# They qualify confidence in the UI and are never used to alter a model score.
DIRECTION_WALK_FORWARD_ROC_AUC = {1: 0.5483, 6: 0.5243, 24: 0.5188}


class DeploymentForecaster:
    def __init__(self, project_root: Path, models_dir: Path | None = None, symbol: str = "BTCUSDT"):
        self.project_root = Path(project_root)
        self.models = DeploymentModels(
            models_dir or self.project_root / os.getenv("BTC_MODELS_DIR", "btc_deployment_models"))
        self.market_agent = MarketDataAgent(symbol=symbol, interval="1h")
        self.symbol = symbol

    def build_frame(self, candles: pd.DataFrame | None = None, limit: int = 500) -> pd.DataFrame:
        """Closed candles plus every feature the deployment models require.

        `limit` must cover the 168h rolling warm-up and the 48h model window.
        """
        raw = candles if candles is not None else self.market_agent.fetch_closed_candles(limit=limit)
        frame = add_stationary_features(self.market_agent.engineer_features(raw))
        validate_timeline(frame)
        return frame

    def predict(self, frame: pd.DataFrame | None = None, limit: int = 500) -> dict:
        frame = self.build_frame(limit=limit) if frame is None else frame
        latest = frame.iloc[-1]

        risk, direction = {}, {}
        for horizon in HORIZONS:
            risk[str(horizon)] = predict_risk(self.models, frame, horizon)
            entry = predict_direction(self.models, frame, horizon)
            entry["walk_forward_roc_auc"] = DIRECTION_WALK_FORWARD_ROC_AUC.get(horizon)
            direction[str(horizon)] = entry

        return {
            "symbol": self.symbol,
            "timeframe": "1h",
            "forecast_candle_utc": pd.Timestamp(latest["timestamp"]).isoformat(),
            "btc_close": float(latest["close"]),
            "movement_risk": risk,
            "direction": direction,
        }


__all__ = ["DeploymentForecaster", "FeatureError", "DIRECTION_WALK_FORWARD_ROC_AUC"]
