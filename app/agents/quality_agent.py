from __future__ import annotations
import pandas as pd

class DataQualityAgent:
    def __init__(self, market_features, sentiment_features, sequence_length=48):
        self.market_features = list(market_features)
        self.sentiment_features = list(sentiment_features)
        self.sequence_length = int(sequence_length)

    def run(self, sequence_df):
        issues = []
        if len(sequence_df) != self.sequence_length:
            issues.append(f"Expected {self.sequence_length} rows, got {len(sequence_df)}.")

        deltas = sequence_df["timestamp"].diff().dropna()
        if not (deltas == pd.Timedelta(hours=1)).all():
            issues.append("Inference sequence is not hourly-contiguous.")

        m_nan = int(sequence_df[self.market_features].isna().sum().sum())
        s_nan = int(sequence_df[self.sentiment_features].isna().sum().sum())
        if m_nan:
            issues.append(f"{m_nan} market-feature NaNs.")
        if s_nan:
            issues.append(f"{s_nan} sentiment-feature NaNs.")

        latest = sequence_df.iloc[-1]
        latest_ts = pd.Timestamp(latest["timestamp"])
        age_hours = (pd.Timestamp.now(tz="UTC") - latest_ts).total_seconds() / 3600.0
        if age_hours > 2.1:
            issues.append(f"Latest closed candle is stale ({age_hours:.1f}h old).")

        return {
            "ok": not issues,
            "issues": issues,
            "latest_timestamp": str(latest_ts),
            "sequence_rows": int(len(sequence_df)),
            "market_feature_count": len(self.market_features),
            "sentiment_feature_count": len(self.sentiment_features),
            "hours_with_news_in_latest_sequence": int(sequence_df["has_news"].sum()),
            "latest_sentiment_score": float(latest["sentiment_score"]),
        }
