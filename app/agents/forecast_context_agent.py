"""Read-only extraction of internal forecast facts and conservative signal states."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from numbers import Real

from ..schemas.agent_models import ForecastContext, ForecastView, MarketContext, SentimentContext

UTC = timezone.utc


def aware_time(value):
    """Parse only explicitly timezone-aware timestamps; never assume a timezone."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime) or value.utcoffset() is None:
        return None
    return value.astimezone(UTC)


def current_time(now=None):
    if now is None:
        return datetime.now(UTC)
    parsed = aware_time(now)
    if parsed is None:
        raise ValueError("now must be a timezone-aware datetime")
    return parsed


def number(value, *, probability=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    if not math.isfinite(value) or (probability and not 0 <= value <= 1) or (positive and value <= 0):
        return None
    return value


def text(value):
    return value if isinstance(value, str) and value.strip() else None


def time_text(value):
    if aware_time(value) is None:
        return None
    return value if isinstance(value, str) else value.isoformat()


def mapping(value):
    return value if isinstance(value, dict) else {}


def first_present(values, keys):
    for key in keys:
        if values.get(key) is not None:
            return values[key]
    return None


def count(value):
    parsed = number(value)
    return int(parsed) if parsed is not None and parsed >= 0 and parsed.is_integer() else None


class ForecastContextAgent:
    """No network, model calls, synthetic reliability, or state changes."""

    def __init__(self, settings):
        self.settings = settings

    def _fresh(self, opened, issued, now):
        opened, issued = aware_time(opened), aware_time(issued)
        if opened is None or issued is None:
            return None
        closed = opened + timedelta(hours=1)
        max_age = timedelta(minutes=self.settings.context_max_age_minutes)
        return (not (opened.minute or opened.second or opened.microsecond)
                and closed <= issued <= now and now - closed <= max_age
                and now - issued <= max_age)

    def _forecast(self, horizon, raw, snapshot, now):
        p = number(raw.get("probability_up"), probability=True)
        direction = "UNKNOWN" if p is None else "UP" if p > .5 else "DOWN" if p < .5 else "NEUTRAL"
        validated = number(raw.get("validated_reliability"), probability=True)
        meta = number(raw.get("meta_reliability"), probability=True)
        raw_confidence = number(first_present(raw, ("raw_confidence", "confidence")), probability=True)
        reliability, source = None, None
        if validated is not None:
            reliability, source = validated, "validated_reliability"
        elif meta is not None:
            reliability, source = meta, "meta_reliability"
        elif raw.get("reliability_validated") is True:
            reliability = number(raw.get("reliability"), probability=True)
            source = "reliability (reliability_validated=true)" if reliability is not None else None
        if reliability is None and raw.get("confidence_validated") is True:
            reliability = raw_confidence
            source = "confidence (confidence_validated=true)" if reliability is not None else None
        gates = [raw.get(key) for key in ("selective_gate", "selective_gate_passed", "gate_passed")
                 if type(raw.get(key)) is bool]
        gate = all(gates) if gates else None
        issued = time_text(first_present(raw, ("issued_at", "model_timestamp")))
        if issued is None and not any(raw.get(key) is not None for key in ("issued_at", "model_timestamp")):
            issued = time_text(snapshot.get("refreshed_at_utc"))
        opened = time_text(raw.get("latest_candle"))
        fresh = self._fresh(opened, issued, now)
        reference = number(raw.get("current_price"), positive=True)
        upstream_no_signal = any(raw.get(key) == "NO_SIGNAL" for key in ("signal_state", "signal_status", "signal", "direction"))
        if gate is False or upstream_no_signal:
            state, reason = "NO_SIGNAL", "The explicit selective gate or upstream signal vetoes this forecast."
        elif reliability is not None and reliability < self.settings.reliability_threshold:
            state, reason = "NO_SIGNAL", "Validated reliability is below the configured threshold."
        elif p is None or reference is None:
            state, reason = "UNKNOWN", "Forecast probability or reference close is unavailable or invalid."
        elif fresh is not True:
            state, reason = "UNKNOWN", "The source candle or forecast issue timestamp is missing, future, or stale."
        elif reliability is None:
            state, reason = "UNKNOWN", "Validated reliability is unavailable; directional probability is not reliability."
        elif direction == "NEUTRAL":
            state, reason = "NO_SIGNAL", "An equal UP/DOWN probability provides no directional signal."
        else:
            state, reason = direction, "Fresh internal forecast meets the validated reliability threshold."
        return ForecastView(
            horizon_hours=horizon, raw_direction=direction, probability_up=p,
            probability_down=number(raw.get("probability_down"), probability=True),
            raw_confidence=raw_confidence, meta_reliability=meta, validated_reliability=validated,
            reliability=reliability, reliability_source=source, selective_gate=gate,
            signal_state=state, reliability_reason=reason, model=text(raw.get("model")),
            feature_set=text(raw.get("feature_set")), model_version=text(raw.get("model_version")),
            model_timestamp=issued, latest_candle=opened, reference_price=reference,
            regime=text(raw.get("regime")),
        ), fresh

    def run(self, snapshot: dict, now=None) -> ForecastContext:
        now = current_time(now)
        snapshot = mapping(snapshot)
        market = mapping(snapshot.get("market"))
        sentiment = mapping(snapshot.get("sentiment"))
        quality = mapping(snapshot.get("quality"))
        forecasts, freshness, warnings = {}, [], []
        for key, value in mapping(snapshot.get("forecasts")).items():
            if str(key) not in ("1", "6", "24") or not isinstance(value, dict):
                warnings.append("An unsupported or malformed forecast entry was omitted.")
                continue
            forecast, fresh = self._forecast(int(key), value, snapshot, now)
            forecasts[str(key)] = forecast
            freshness.append(fresh)
        market_timestamp = time_text(market.get("timestamp"))
        source_timestamp = market_timestamp or time_text(quality.get("latest_timestamp"))
        refresh = time_text(snapshot.get("refreshed_at_utc"))
        if source_timestamp is not None:
            freshness.append(self._fresh(source_timestamp, refresh, now))
        data_fresh = None if not freshness or any(item is None for item in freshness) else all(freshness)
        if any(item is False for item in freshness):
            data_fresh = False
        if data_fresh is not True:
            for forecast in forecasts.values():
                if forecast.signal_state in {"UP", "DOWN"}:
                    forecast.signal_state = "UNKNOWN"
                    forecast.reliability_reason = "Freshness of the internal market and forecast context could not be confirmed."
        if not forecasts:
            warnings.append("No saved internal forecasts are available.")
        if any(f.reliability is None for f in forecasts.values()):
            warnings.append("One or more forecasts have no validated reliability; confidence and holdout accuracy do not substitute for it.")
        if data_fresh is not True:
            warnings.append("Context freshness could not be confirmed for every available forecast and market timestamp.")
        return ForecastContext(
            timestamp=source_timestamp, captured_at=now.isoformat(), last_refresh=refresh,
            data_fresh=data_fresh,
            market=MarketContext(price=number(market.get("price"), positive=True), timestamp=market_timestamp,
                                 regime=text(market.get("regime")), volatility_24h=number(market.get("volatility_24h")),
                                 momentum_24h=number(market.get("momentum_24h")), rsi_14=number(market.get("rsi_14"))),
            sentiment=SentimentContext(
                score=number(first_present(sentiment, ("latest_sentiment_score", "sentiment_score", "score"))),
                label=text(sentiment.get("label")),
                article_count=count(first_present(sentiment, ("accepted_articles", "article_count"))),
                hours_with_news=count(quality.get("hours_with_news_in_latest_sequence")),
                timestamp=time_text(sentiment.get("timestamp")),
            ), forecasts=forecasts, warnings=warnings,
        )
