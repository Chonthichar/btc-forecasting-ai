"""Authoritative post-validation for LLM decision proposals."""
from datetime import datetime, timezone
from app.schemas.agent_models import MarketDecision


def enforce_decision(context, research, validation, requested, proposal=None, explanation=""):
    # The user-selected horizon scope is authoritative; the model cannot hide an
    # unqualified requested horizon by selecting a different one.
    scope = list(dict.fromkeys(requested))
    horizon = scope[0] if len(scope) == 1 else "mixed"
    views = context.forecasts

    def qualified(h):
        view = views.get(h[:-1])
        return bool(view and context.data_fresh and validation.context_valid
                    and view.reliability is not None and view.reliability >= validation.reliability_threshold
                    and view.signal_state in ("UP", "DOWN")
                    and validation.signal_states.get(h[:-1]) == view.signal_state)

    if proposal and proposal.time_horizon == "6-24h" and set(scope) == {"1h", "6h", "24h"}:
        if all(qualified(h) for h in ("6h", "24h")) and views["6"].signal_state == views["24"].signal_state:
            scope, horizon = ["6h", "24h"], "6-24h"
    states = {h: views[h[:-1]].signal_state if h[:-1] in views else "UNKNOWN" for h in requested}
    directions = {views[h[:-1]].signal_state for h in scope if qualified(h)}
    complete = bool(scope) and all(qualified(h) for h in scope)
    ml = ("NO_RELIABLE_SIGNAL" if not complete else "MIXED" if len(directions) > 1
          else "BULLISH" if directions == {"UP"} else "BEARISH")
    # Only useful, reviewed evidence can contribute a directional research view.
    reviewed_ids = set(research.review.accepted_evidence) if research.review and research.review.status == "ok" else set()
    trusted = [e for e in research.evidence if e.id in reviewed_ids and e.recency_verified and e.source_quality != "unverified"]
    evidence_directions = {e.direction for e in trusted}
    external = ("NO_EVIDENCE" if not trusted else "MIXED" if {"bullish", "bearish"} <= evidence_directions
                else "BULLISH" if "bullish" in evidence_directions else "BEARISH" if "bearish" in evidence_directions else "NEUTRAL")
    stance = "NO_RELIABLE_VIEW" if ml == "NO_RELIABLE_SIGNAL" else ml
    # Disagreement adds uncertainty; news never reverses a qualified ML view.
    if stance in ("BULLISH", "BEARISH") and external in ("BULLISH", "BEARISH", "MIXED") and external != stance:
        stance = "MIXED"
    agreement = ml in ("BULLISH", "BEARISH") and ml == external
    strength = "MODERATE" if agreement and stance in ("BULLISH", "BEARISH") else "LOW"
    risk = "Forecasts remain uncertain; external context does not establish causation."
    if horizon == "6-24h" and qualified("1h") and views["1"].signal_state != views["6"].signal_state:
        risk = "The qualified 1h model points " + views["1"].signal_state + ", opposing the 6h/24h view."
    elif horizon == "6-24h" and not qualified("1h"):
        risk = "The 1h forecast does not have a qualified directional signal."
    canonical = ("No reliable model view is available for the selected horizons. External evidence cannot supply missing reliability."
                 if stance == "NO_RELIABLE_VIEW" else "The selected qualified forecasts and reviewed evidence yield a " + stance.lower() + " interpretation.")
    return MarketDecision(market_stance=stance, decision_strength=strength, time_horizon=horizon,
        ml_view=ml, research_view=external, agreement=agreement, short_term_risk=risk,
        explanation=canonical + ("\n\n" + explanation if explanation else ""), evidence_ids=[e.id for e in trusted],
        generated_at=datetime.now(timezone.utc).isoformat(), context_timestamp=context.timestamp,
        signal_states=states)
