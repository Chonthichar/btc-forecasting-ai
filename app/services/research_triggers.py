"""Deterministic monitoring features and event rules; never model inputs."""
from __future__ import annotations
import hashlib
import json
import math
import pandas as pd


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def market_metrics(candles, snapshot, volatility_multiplier=1.5):
    # Match existing market features: sample std of hourly log returns and
    # fractional close-to-close momentum. Use completed, contiguous candles.
    frame = candles.sort_values("timestamp").drop_duplicates("timestamp").copy()
    times = pd.to_datetime(frame["timestamp"], utc=True)
    close = pd.Series(pd.to_numeric(frame["close"], errors="coerce").values, index=times)
    close = close.asfreq("h")
    returns = close.map(lambda x: math.log(x) if pd.notna(x) and x > 0 else float("nan")).diff()
    def change(hours):
        if len(close) <= hours or pd.isna(close.iloc[-hours-1]) or close.iloc[-hours-1] <= 0:
            return None
        return finite((close.iloc[-1] / close.iloc[-hours-1] - 1) * 100)
    volatility = finite(returns.rolling(24, min_periods=24).std().iloc[-1])
    # Prior 168 hourly returns: baseline excludes the current 24-hour window.
    baseline = finite(returns.shift(24).rolling(168, min_periods=168).std().iloc[-1])
    move24 = change(24)
    market = snapshot.get("market", {})
    regime = market.get("regime")
    basis = "upstream" if regime else "monitor-only: 24h momentum versus 24h realized volatility"
    if not regime and volatility is not None and baseline is not None and baseline > 0 and move24 is not None:
        # Fixed descriptive regime boundaries, independent of research thresholds.
        # These labels never alter model features, probabilities or reliability.
        if volatility >= baseline * volatility_multiplier:
            regime = "HIGH_VOL"
        elif move24 / 100 > volatility * math.sqrt(24):
            regime = "BULL"
        elif move24 / 100 < -volatility * math.sqrt(24):
            regime = "BEAR"
        else:
            regime = "RANGE"
    return {"market_timestamp": times.iloc[-1].isoformat(), "price": finite(close.iloc[-1]),
            "price_change_1h": change(1), "price_change_24h": move24,
            "volatility": volatility, "volatility_baseline": baseline,
            "momentum": None if move24 is None else move24 / 100,
            "regime": regime, "regime_basis": basis}


class ResearchTriggerEngine:
    def __init__(self, settings):
        self.settings = settings

    def evaluate(self, current, previous=None):
        previous = previous or {}
        reasons, details = [], {}
        # Repeated scans of the same closed candle do not constitute new events.
        fresh = current.get("data_fresh") is True
        changed_candle = current.get("market_timestamp") != previous.get("market_timestamp")
        if not fresh or not changed_candle:
            return {"should_research": False, "triggered_by": [], "details": {}, "fingerprint": None}
        move = finite(current.get("price_change_1h"))
        if move is not None and abs(move) >= self.settings.price_move_trigger_pct:
            reasons.append("large_price_move")
            details["price_change_1h"] = move
        old_regime, regime = previous.get("regime"), current.get("regime")
        if old_regime and regime and old_regime != regime:
            reasons.append("regime_change")
            details.update(previous_regime=old_regime, current_regime=regime)
        vol, baseline = finite(current.get("volatility")), finite(current.get("volatility_baseline"))
        if vol is not None and baseline is not None and baseline > 0 and vol >= baseline * self.settings.volatility_trigger_multiplier:
            reasons.append("unusual_volatility")
            details.update(volatility=vol, volatility_baseline=baseline)
        sentiment, old_sentiment = finite(current.get("sentiment")), finite(previous.get("sentiment"))
        if sentiment is not None and old_sentiment is not None:
            delta = sentiment - old_sentiment
            # Meaningful reversal requires both sides outside a small neutral band.
            reversal = sentiment * old_sentiment < 0 and min(abs(sentiment), abs(old_sentiment)) >= self.settings.sentiment_shift_threshold / 2
            if abs(delta) >= self.settings.sentiment_shift_threshold or reversal:
                reasons.append("sentiment_shift")
                details.update(sentiment_change=delta, sentiment=sentiment)
        transitions = {}
        for horizon, forecast in current.get("forecasts", {}).items():
            old = previous.get("forecasts", {}).get(horizon, {})
            before, after = old.get("signal_state"), forecast.get("signal_state")
            if before and after != before and after in {"UP", "DOWN", "NO_SIGNAL"} and (before in {"UP", "DOWN"} or after in {"UP", "DOWN"}):
                transitions[horizon] = {"before": before, "after": after}
        if transitions:
            reasons.append("forecast_state_change")
            details["signal_transitions"] = transitions
        signature = {"reasons": sorted(reasons), "regime": regime,
                     "move_bucket": None if move is None else math.trunc(move / self.settings.price_move_trigger_pct),
                     "sentiment_direction": None if sentiment is None else (0 if abs(sentiment) < self.settings.sentiment_shift_threshold / 2 else 1 if sentiment > 0 else -1),
                     "signals": {h: f.get("signal_state") for h, f in current.get("forecasts", {}).items()}}
        return {"should_research": bool(reasons), "triggered_by": reasons, "details": details,
                "fingerprint": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest() if reasons else None}
