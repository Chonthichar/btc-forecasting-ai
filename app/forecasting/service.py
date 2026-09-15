"""Serves the six deployment predictions and logs each one prospectively.

Inference is cached per forecast candle: the models are deterministic given a
closed candle, so recomputing within the same hour would burn CPU to produce
an identical answer.

Every new candle is written to the forward-test log BEFORE its outcome exists,
which is the property that makes the log usable as evidence later.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from .feature_builder import FeatureError
from .realtime_forecast import DeploymentForecaster

logger = logging.getLogger(__name__)


class DeploymentPredictionService:
    def __init__(self, project_root: Path, log_path: Path | None = None):
        self.project_root = Path(project_root)
        self.forecaster = DeploymentForecaster(self.project_root)
        self.log_path = log_path or self.project_root / "runtime" / "deployment_forecasts.jsonl"
        self._lock = threading.Lock()
        self._cache: dict | None = None
        self._logged: set[str] = set()

    def current(self, refresh: bool = False) -> dict:
        """The six predictions for the most recent closed candle."""
        with self._lock:
            frame = self.forecaster.build_frame()
            candle = str(frame.iloc[-1]["timestamp"])
            if refresh or self._cache is None or self._cache.get("_candle") != candle:
                payload = self.forecaster.predict(frame=frame)
                payload["_candle"] = candle
                self._cache = payload
                self._record(payload)
            return {k: v for k, v in self._cache.items() if not k.startswith("_")}

    def _record(self, payload: dict) -> None:
        """Append one flat row per forecast candle; never duplicate a candle."""
        candle = payload["forecast_candle_utc"]
        if candle in self._logged:
            return
        row = {
            "generated_at_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
            "forecast_candle_utc": candle,
            "symbol": payload["symbol"],
            "btc_close": payload["btc_close"],
        }
        for horizon, entry in payload["movement_risk"].items():
            row[f"risk_{horizon}h_score"] = entry["model_score"]
            row[f"risk_{horizon}h_band"] = entry["risk_level"]
        for horizon, entry in payload["direction"].items():
            row[f"direction_{horizon}h"] = entry["label"]
            row[f"direction_{horizon}h_up_score"] = entry["up_score"]

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if candle in self._existing_candles():
                self._logged.add(candle)
                return
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            self._logged.add(candle)
        except OSError as exc:
            # A forecast is still valid if the log cannot be written; say so
            # rather than failing the request.
            logger.warning("Could not append to the forward-test log: %s", exc)

    def _existing_candles(self) -> set[str]:
        """Candles already on disk, so a restart does not re-log them."""
        if self._logged:
            return self._logged
        if not self.log_path.exists():
            return set()
        found = set()
        try:
            with open(self.log_path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        found.add(json.loads(line)["forecast_candle_utc"])
                    except (ValueError, KeyError):
                        continue
        except OSError:
            return set()
        self._logged = found
        return found

    def log_size(self) -> int:
        return len(self._existing_candles())


__all__ = ["DeploymentPredictionService", "FeatureError"]
