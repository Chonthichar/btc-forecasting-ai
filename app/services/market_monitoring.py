"""Persist quantitative scans, then optionally research a deterministic event."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
from app.agents.forecast_context_agent import ForecastContextAgent
from app.agents.market_research_agent import MarketResearchAgent
from app.agents.validation_agent import ValidationAgent
from app.agents.decision_agent import DecisionAgent
from app.schemas.agent_models import ResearchResult
from .research_triggers import ResearchTriggerEngine, market_metrics
from .research_workflow import ResearchWorkflow, persist_research
from .market_decision import enforce_decision


class MarketMonitoringAgent:
    def __init__(self, monitoring, research_agent=None, llm=None):
        self.monitoring = monitoring
        self.settings = monitoring.settings
        self.store = monitoring.event_store
        self.forecast = ForecastContextAgent(self.settings)
        self.engine = ResearchTriggerEngine(self.settings)
        self.research = research_agent or ResearchWorkflow(monitoring, self.settings)
        self.validation = ValidationAgent(self.settings)
        self.decision = DecisionAgent(self.settings, self.validation, llm=llm)

    def prepare(self, snapshot, candles, now=None):
        metrics = market_metrics(candles, snapshot, self.settings.volatility_trigger_multiplier)
        context = self.forecast.run(snapshot, now=now)
        sentiment = context.sentiment.score
        if context.sentiment.article_count == 0:
            sentiment = None
        return {**metrics, "data_fresh": context.data_fresh, "sentiment": sentiment,
                "sentiment_scale": "weighted signed score [-1, 1]",
                "forecasts": {h: f.model_dump() for h, f in context.forecasts.items()}}

    def state(self, **values):
        current = self.monitoring.store.get_state("research_operation", {})
        self.monitoring.store.set_state("research_operation", {**current, **values})

    def process(self, run_id, snapshot, candles, now=None):
        now = now or datetime.now(timezone.utc)
        current = self.prepare(snapshot, candles, now)
        previous = self.store.scans(1)
        trigger = self.engine.evaluate(current, previous[0]["payload"] if previous else None)
        scan_id = self.store.scan(run_id, current, trigger, now)
        self.monitoring.store.set_state("last_market_scan", now.isoformat())
        # No paid provider or Decision Agent call on the ordinary path.
        if not trigger["should_research"]:
            self.state(status="IDLE", monitor="done", forecast="done", trigger="done", validation="idle", decision="idle")
            return trigger
        self.monitoring.store.set_state("last_research_trigger", {"timestamp": now.isoformat(), **trigger})
        event, leader = self.store.claim(trigger["fingerprint"], current["market_timestamp"], trigger["triggered_by"], self.settings.research_cooldown_minutes, now)
        self.store.link(scan_id, event["id"])
        if not leader:
            # Validate reused evidence again. Cooldown never extends source freshness.
            cached = ResearchResult.model_validate(event["result"]) if event.get("result") else ResearchResult()
            context = self.forecast.run(snapshot, now=now)
            _, cached, _ = self.validation.validate(context, snapshot, cached, now=now)
            self.state(status="CACHED" if cached.evidence else "IDLE", monitor="done", forecast="done", trigger="done", validation="done", decision="idle",
                       event_id=event["id"], cache_reason="Repeated event suppressed; no provider calls.")
            return {**trigger, "cached": bool(cached.evidence), "suppressed": True}
        self.state(status="SEARCHING", monitor="done", forecast="done", trigger="done", validation="idle", decision="idle", event_id=event["id"])
        try:
            question = "Explain the latest Bitcoin market context. Research topics: " + ", ".join(trigger["triggered_by"])
            result = self.research.run(question, trigger_types=trigger["triggered_by"], user_requested=False, event_id=event["id"],
                on_state=lambda stage, status: self.state(**{stage: status}))
            self.state(status="CACHED" if result.cached else "SEARCHING", validation="running", decision="idle", event_id=event["id"])
            context = self.forecast.run(snapshot, now=now)
            context, result, validation = self.validation.validate(context, snapshot, result)
            useful = any(item.recency_verified and item.source_quality in {"primary", "established"} for item in result.evidence)
            summary, llm_status = None, None
            decision = enforce_decision(context, result, validation, ["1h", "6h", "24h"])
            if useful:
                self.state(status="CACHED" if result.cached else "SEARCHING", validation="done", decision="running", event_id=event["id"])
                answer, llm_status, _, decision = self.decision.run(question, [], context, result, validation, structured=True)
                if llm_status == "ok":
                    summary = answer
            failed = result.status == "unavailable" or llm_status in {"unavailable", "invalid_output"}
            result.market_decision = decision
            if result.workflow and llm_status in {"unavailable", "invalid_output"}:
                result.workflow.errors.append("Decision Agent " + llm_status)
            persist_research(self.monitoring, result, decision,
                useful and bool(self.settings.openai_key and self.settings.openai_model))
            self.store.finish(event["id"], "error" if failed else "done" if useful else "empty", result.model_dump(), summary, llm_status)
            self.state(status="ERROR" if failed else "CACHED" if result.cached else "IDLE", monitor="done", forecast="done", trigger="done", validation="done",
                       decision="error" if failed and useful else "done" if useful else "idle", event_id=event["id"])
        except Exception:
            # Forecasts and the scan already committed; research must never undo them.
            self.store.finish(event["id"], "error")
            self.state(status="ERROR", monitor="done", forecast="done", trigger="done", validation="error", decision="idle", event_id=event["id"], error="Context research failed; market forecasts remain saved.")
        return trigger
