from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

class SentimentAgent:
    """Run the existing thesis 3-agent BTC sentiment pipeline for a recent live window."""

    def __init__(self, script_path, output_dir="/content/realtime_btc_sentiment", alpha_vantage_key=None):
        self.script_path = Path(script_path)
        self.output_dir = Path(output_dir)
        self.alpha_vantage_key = alpha_vantage_key

    def run(self, hours_back=96, use_gdelt=True):
        if not self.script_path.exists():
            raise FileNotFoundError(f"Sentiment pipeline not found: {self.script_path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        end = pd.Timestamp.now(tz="UTC")
        start = end - pd.Timedelta(hours=hours_back)
        cmd = [
            sys.executable, str(self.script_path),
            "--start-date", start.strftime("%Y-%m-%d"),
            "--end-date", end.strftime("%Y-%m-%d"),
            "--output-dir", str(self.output_dir),
        ]
        if not use_gdelt:
            cmd.append("--skip-gdelt")

        env = os.environ.copy()
        for key in ("MONITOR_BACKUP_TOKEN", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
            env.pop(key, None)
        if self.alpha_vantage_key:
            env["ALPHA_VANTAGE_API_KEY"] = self.alpha_vantage_key

        # Bound a cycle so a stalled provider cannot hold the hourly lock.
        log_path = self.output_dir / "latest_collection.log"
        with log_path.open("w", encoding="utf-8") as log:
            try:
                subprocess.run(cmd, check=True, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError("News collection failed. See runtime/sentiment/latest_collection.log.") from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("News collection exceeded 15 minutes; the next cycle will retry.") from exc

        hourly = pd.read_parquet(self.output_dir / "04_btc_sentiment_hourly.parquet")
        articles = pd.read_parquet(self.output_dir / "03_final_btc_sentiment_articles.parquet")
        latest_score = float(hourly.sort_values("timestamp").iloc[-1]["sentiment_score"]) if len(hourly) else 0.0

        return hourly, articles, {
            "accepted_articles": int(len(articles)),
            "hours_with_news": int(len(hourly)),
            "latest_sentiment_score": latest_score,
            "window_start": str(start),
            "window_end": str(end),
        }
