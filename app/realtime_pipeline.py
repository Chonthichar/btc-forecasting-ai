from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import yaml

if __package__:
    from .agents.market_agent import MarketDataAgent
    from .agents.sentiment_agent import SentimentAgent
    from .agents.quality_agent import DataQualityAgent
    from .agents.forecast_agent import ForecastAgent
else:
    # Support the direct-script UI and notebooks that import from the app directory.
    from agents.market_agent import MarketDataAgent
    from agents.sentiment_agent import SentimentAgent
    from agents.quality_agent import DataQualityAgent
    from agents.forecast_agent import ForecastAgent

class RealtimeCryptoPipeline:
    def __init__(self, project_root, sentiment_script, live_work_dir=None, alpha_vantage_key=None):
        self.project_root = Path(project_root)
        self.work_dir = Path(live_work_dir or (self.project_root / "runtime"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.market_path = self.work_dir / "live_market.parquet"
        self.sent_dir = self.work_dir / "sentiment"

        import sys
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        from src.pipeline import VAETransformer, SequenceClassifier, load_and_merge
        from src.features import MARKET_CORE_V1, SENTIMENT_EXTENDED_V2

        self.load_and_merge = load_and_merge
        self.market_features = list(MARKET_CORE_V1)
        self.sentiment_features = list(SENTIMENT_EXTENDED_V2)
        self.sequence_length = 48

        registry_path = Path(__file__).resolve().parent / "deployment_registry.yaml"
        raw = yaml.safe_load(open(registry_path, "r", encoding="utf-8"))
        deployed = {}
        for k, spec in raw.items():
            h = int(k)
            deployed[h] = {
                "kind": spec["kind"],
                "feature_set": spec["feature_set"],
                "dir": self.project_root / spec["output_subdir"],
                "config": self.project_root / spec["config"],
            }

        self.market_agent = MarketDataAgent()
        self.sentiment_agent = SentimentAgent(
            sentiment_script,
            output_dir=self.sent_dir,
            alpha_vantage_key=alpha_vantage_key,
        )
        self.quality_agent = DataQualityAgent(
            self.market_features, self.sentiment_features, self.sequence_length
        )
        self.forecast_agent = ForecastAgent(
            deployed,
            self.market_features,
            self.sentiment_features,
            {"VAETransformer": VAETransformer, "SequenceClassifier": SequenceClassifier},
        )
        self.cache = {}

    def _build_multimodal_frame(self):
        now = pd.Timestamp.now(tz="UTC")
        start = now - pd.Timedelta(hours=96)
        cfg_path = self.project_root / "config/btc_24h.yaml"
        cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
        cfg = json.loads(json.dumps(cfg))
        cfg["paths"]["market"] = str(self.market_path)
        cfg["paths"]["sentiment"] = str(self.sent_dir / "04_btc_sentiment_hourly.parquet")
        cfg["experiment"]["start"] = start.strftime("%Y-%m-%d")
        cfg["experiment"]["end"] = now.strftime("%Y-%m-%d")

        merged = self.load_and_merge(cfg)
        needed = self.market_features + self.sentiment_features
        live = merged.dropna(subset=needed).copy().reset_index(drop=True)
        if len(live) < self.sequence_length:
            raise RuntimeError(f"Only {len(live)} prepared rows; need {self.sequence_length}.")
        return live, live.tail(self.sequence_length).copy()

    def refresh(self, use_gdelt=True, market_candles=None):
        market_df, market_snapshot = self.market_agent.run(limit=500, candles=market_candles)
        market_df.to_parquet(self.market_path, index=False)

        _, _, sentiment_snapshot = self.sentiment_agent.run(hours_back=96, use_gdelt=use_gdelt)
        live_frame, sequence = self._build_multimodal_frame()

        quality = self.quality_agent.run(sequence)
        if not quality["ok"]:
            raise RuntimeError("; ".join(quality["issues"]))

        forecasts = {str(h): self.forecast_agent.predict(h, sequence) for h in [1, 6, 24]}
        self.cache = {
            "refreshed_at_utc": str(pd.Timestamp.now(tz="UTC")),
            "market": market_snapshot,
            "sentiment": sentiment_snapshot,
            "quality": quality,
            "forecasts": forecasts,
        }
        self.live_frame = live_frame
        self.sequence = sequence
        return self.cache

    def get_latest_market(self):
        return self.cache.get("market", {})

    def get_latest_sentiment(self):
        q = self.cache.get("quality", {})
        s = self.cache.get("sentiment", {})
        return {
            **s,
            "latest_model_sentiment_score": q.get("latest_sentiment_score"),
            "hours_with_news_in_latest_sequence": q.get("hours_with_news_in_latest_sequence"),
        }

    def get_forecast(self, horizon_hours):
        return self.cache["forecasts"][str(int(horizon_hours))]

    def get_system_health(self):
        return self.cache.get("quality", {})

    def tool_map(self):
        return {
            "get_latest_market": lambda **kwargs: self.get_latest_market(),
            "get_latest_sentiment": lambda **kwargs: self.get_latest_sentiment(),
            "get_forecast": lambda **kwargs: self.get_forecast(**kwargs),
            "get_system_health": lambda **kwargs: self.get_system_health(),
            "refresh_live_data": lambda **kwargs: self.refresh(),
        }
