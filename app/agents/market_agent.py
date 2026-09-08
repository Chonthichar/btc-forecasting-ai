from __future__ import annotations
import numpy as np
import pandas as pd
import requests

BINANCE_KLINES = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
]

RAW_COLS = [
    "open_time_ms","open","high","low","close","volume",
    "close_time_ms","quote_volume","num_trades",
    "taker_buy_base_volume","taker_buy_quote_volume","ignore",
]

class MarketDataAgent:
    """Fetch live closed BTCUSDT 1h candles and reproduce thesis market features."""

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1h"):
        self.symbol = symbol
        self.interval = interval
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "btc-thesis-realtime-v2/1.0"})

    def fetch_closed_candles(self, limit: int = 500, start_time=None) -> pd.DataFrame:
        last_error = None
        params = {"symbol": self.symbol, "interval": self.interval, "limit": limit}
        if start_time is not None:
            params["startTime"] = int(pd.Timestamp(start_time).timestamp() * 1000)
        for endpoint in BINANCE_KLINES:
            try:
                r = self.session.get(
                    endpoint,
                    params=params,
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict):
                    raise RuntimeError(str(data))
                df = pd.DataFrame(data, columns=RAW_COLS)
                df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
                df["close_timestamp"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
                for c in [
                    "open","high","low","close","volume","quote_volume","num_trades",
                    "taker_buy_base_volume","taker_buy_quote_volume",
                ]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

                now = pd.Timestamp.now(tz="UTC")
                df = df[df["close_timestamp"] < now].copy()
                keep = [
                    "timestamp","close_timestamp","open","high","low","close",
                    "volume","quote_volume","num_trades",
                    "taker_buy_base_volume","taker_buy_quote_volume",
                ]
                return (
                    df[keep]
                    .sort_values("timestamp")
                    .drop_duplicates("timestamp", keep="last")
                    .reset_index(drop=True)
                )
            except Exception as e:
                last_error = e
        raise RuntimeError(f"Binance live fetch failed: {last_error}")

    @staticmethod
    def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out = 100 - 100 / (1 + rs)
        return out.where(avg_loss != 0, 100.0).where(avg_gain != 0, 0.0)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.sort_values("timestamp").copy().reset_index(drop=True)
        prev_close = x["close"].shift(1)

        x["log_return"] = np.log(x["close"] / prev_close)
        x["return_1h"] = x["close"].pct_change()
        x["sma_24"] = x["close"].rolling(24, min_periods=24).mean()
        x["sma_168"] = x["close"].rolling(168, min_periods=168).mean()
        x["ema_12"] = x["close"].ewm(span=12, adjust=False).mean()
        x["ema_26"] = x["close"].ewm(span=26, adjust=False).mean()
        x["macd"] = x["ema_12"] - x["ema_26"]
        x["macd_signal_9"] = x["macd"].ewm(span=9, adjust=False).mean()
        x["macd_hist"] = x["macd"] - x["macd_signal_9"]
        x["volatility_24h"] = x["log_return"].rolling(24, min_periods=24).std()
        x["volatility_168h"] = x["log_return"].rolling(168, min_periods=168).std()
        x["rsi_14"] = self._rsi(x["close"])

        tr = pd.concat(
            [
                x["high"] - x["low"],
                (x["high"] - prev_close).abs(),
                (x["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        x["atr_14"] = tr.rolling(14, min_periods=14).mean()
        x["atr_pct_14"] = x["atr_14"] / x["close"]

        mid = x["close"].rolling(20, min_periods=20).mean()
        sd = x["close"].rolling(20, min_periods=20).std()
        x["bb_mid_20"] = mid
        x["bb_upper_20"] = mid + 2 * sd
        x["bb_lower_20"] = mid - 2 * sd
        x["bb_width_20"] = (x["bb_upper_20"] - x["bb_lower_20"]) / mid
        x["bb_position_20"] = (
            (x["close"] - x["bb_lower_20"])
            / (x["bb_upper_20"] - x["bb_lower_20"])
        )

        for h in [3, 6, 12, 24, 72, 168]:
            x[f"momentum_{h}h"] = x["close"].pct_change(h)
        for h in [1, 2, 3, 6, 12, 24]:
            x[f"log_return_lag_{h}h"] = x["log_return"].shift(h)

        x["volume_sma_24"] = x["volume"].rolling(24, min_periods=24).mean()
        x["volume_zscore_24"] = (
            (x["volume"] - x["volume_sma_24"])
            / x["volume"].rolling(24, min_periods=24).std().replace(0, np.nan)
        )
        x["quote_volume_sma_24"] = x["quote_volume"].rolling(24, min_periods=24).mean()
        x["quote_volume_zscore_24"] = (
            (x["quote_volume"] - x["quote_volume_sma_24"])
            / x["quote_volume"].rolling(24, min_periods=24).std().replace(0, np.nan)
        )
        x["trades_sma_24"] = x["num_trades"].rolling(24, min_periods=24).mean()
        x["taker_buy_ratio"] = x["taker_buy_base_volume"] / x["volume"].replace(0, np.nan)
        x["taker_buy_quote_ratio"] = x["taker_buy_quote_volume"] / x["quote_volume"].replace(0, np.nan)
        x["buy_sell_imbalance"] = 2 * x["taker_buy_ratio"] - 1
        x["range_pct"] = (x["high"] - x["low"]) / x["open"]
        x["body_pct"] = (x["close"] - x["open"]) / x["open"]
        x["close_location"] = (
            (x["close"] - x["low"])
            / (x["high"] - x["low"]).replace(0, np.nan)
        )
        return x

    def run(self, limit: int = 500, candles=None):
        raw = candles if candles is not None else self.fetch_closed_candles(limit=limit)
        features = self.engineer_features(raw)
        latest = features.iloc[-1]
        snapshot = {
            "timestamp": str(latest["timestamp"]),
            "price": float(latest["close"]),
            "rsi_14": float(latest["rsi_14"]),
            "momentum_24h": float(latest["momentum_24h"]),
            "volatility_24h": float(latest["volatility_24h"]),
            "taker_buy_ratio": float(latest["taker_buy_ratio"]),
        }
        return features, snapshot
