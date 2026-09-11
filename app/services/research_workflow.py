"""Authorized Research -> deterministic validation -> Review -> validation.

Uses the existing Tavily boundary, cache, SQLite events and reliability gate.
No scheduler and no provider work is created by a read endpoint.
"""
import copy
import re
import threading
import uuid
from datetime import datetime, timezone
from app.agents.market_research_agent import MarketResearchAgent
from app.agents.forecast_context_agent import ForecastContextAgent, aware_time
from app.agents.validation_agent import ValidationAgent
from app.schemas.agent_models import ResearchResult, ResearchAudit, ReviewReport, ReviewPlan, ResearchPlan
from .agent_config import redact_values
from .reasoning_agents import ReasoningAgents


def persist_research(monitoring, research, decision=None, decision_executed=False):
    audit = research.workflow
    if audit:
        audit.decision_agent_executed = decision_executed
        if decision:
            research.market_decision = decision
            audit.final_market_stance = decision.market_stance
            audit.decision_strength = decision.decision_strength
            audit.final_signal_states = decision.signal_states
        events = getattr(monitoring, "event_store", None)
        if events and audit.event_id:
            events.finish(audit.event_id, "error" if audit.errors else "done" if research.evidence else "empty",
                          result=redact_values(research.model_dump()),
                          summary=decision.explanation if decision else None)


class ResearchWorkflow:
    def __init__(self, monitoring, settings, collector=None, reasoning=None):
        self.monitoring, self.settings = monitoring, settings
        self.collector = collector or MarketResearchAgent(settings, store=getattr(monitoring, "event_store", None))
        self.reasoning = reasoning or ReasoningAgents(settings)
        self.forecast = ForecastContextAgent(settings)
        self.validator = ValidationAgent(settings)

    def _cached(self, mode, fresh, now):
        if fresh or mode != "live" or not getattr(self.monitoring, "store", None):
            return ResearchResult()
        candidates = [self.monitoring.store.get_state("analyst_evidence")]
        event = self.monitoring.event_store.latest_event(with_evidence=True)
        candidates.append(event["result"] if event else None)
        for candidate in sorted(filter(None, candidates), key=lambda x: x.get("retrieved_at") or "", reverse=True):
            stamp = aware_time(candidate.get("retrieved_at"))
            if stamp and 0 <= (now - stamp).total_seconds() < self.settings.news_cache_minutes * 60:
                result = ResearchResult.model_validate(candidate)
                return result.model_copy(update={"cached": True, "provider_call_count": 0,
                    "returned_source_count": len(result.evidence),
                    "review": None, "workflow": None, "market_decision": None}, deep=True)
        return ResearchResult()

    def run(self, question, mode="live", prediction_timestamp=None, fresh=False,
            trigger_types=None, user_requested=False, event_id=None, on_state=None):
        now = datetime.now(timezone.utc)
        audit = ResearchAudit(timestamp=now.isoformat(), trigger_reason=trigger_types or ["user_request"],
                              user_requested=user_requested, mode=mode, event_id=event_id)
        events = getattr(self.monitoring, "event_store", None)
        if events and event_id is None:
            event, _ = events.claim("request:" + uuid.uuid4().hex, None, audit.trigger_reason, 0, now)
            audit.event_id = event["id"]
        snapshot = copy.deepcopy(self.monitoring.snapshot())
        context = self.forecast.run(snapshot)
        audit.final_signal_states = {key: view.signal_state for key, view in context.forecasts.items()}
        options = dict(mode=mode, prediction_timestamp=prediction_timestamp)
        result = ResearchResult(status="empty")
        def notify(stage, status):
            if on_state:
                on_state(stage, status)
        try:
            result = self._cached(mode, fresh, now)
            context, result, _ = self.validator.validate(context, snapshot, result, **options)
            used, guard = False, threading.Lock()

            def search(queries):
                nonlocal used, result
                with guard:
                    if used:
                        return {"error": "The one-batch research budget is exhausted."}
                    used = True
                if not isinstance(queries, list) or not 1 <= len(queries) <= 3 or any(
                    not isinstance(q, str) or not 1 <= len(q.strip()) <= 180 or
                    not re.search(r"\b(?:bitcoin|btc)\b", q, re.I) or
                    re.search(r"\b(?:bullish|bearish|buy|sell)\b|why.*(?:will|should).*(?:rise|fall)", q, re.I)
                    for q in queries):
                    return {"error": "Use at most three short, neutral BTC queries."}
                cleaned = redact_values(list(dict.fromkeys(queries)))
                result = self.collector.run(question, queries=cleaned, fresh=fresh, **options)
                audit.queries_used = result.queries
                audit.tavily_called = result.provider_call_count > 0
                audit.cache_used = result.cached or result.cached_query_count > 0
                audit.returned_sources = result.returned_source_count or len(result.evidence)
                return result.model_dump(exclude={"review", "workflow", "market_decision"})

            payload = dict(question=question, trigger_reason=audit.trigger_reason, user_requested=user_requested,
                context=context.model_dump(), timestamp=now.isoformat(), mode=mode,
                prediction_timestamp=prediction_timestamp.isoformat() if hasattr(prediction_timestamp, "isoformat") else prediction_timestamp,
                fresh_research=fresh, cached_evidence=result.model_dump(), max_queries=3,
                # Once the deterministic router authorizes research, an empty
                # cache must produce one bounded search rather than relying on
                # the model to decide whether to call its only tool.
                search_required=not bool(result.evidence))
            notify("research", "running")
            plan, status, audit.research_agent_executed = self.reasoning.run("Research", payload, search)
            notify("research", "done" if status == "ok" else "unavailable")
            if status != "ok" or plan is None:
                raise ValueError("Research Agent unavailable; no unreviewed evidence is passed onward.")
            plan = ResearchPlan.model_validate(plan)
            if fresh and not used:
                raise ValueError("Fresh research was requested but the agent did not search.")
            if any(key not in {item.id for item in result.evidence} for key in plan.evidence_ids):
                raise ValueError("Research Agent referenced evidence that was not retrieved.")
            audit.queries_used, audit.cache_used = result.queries, result.cached or result.cached_query_count > 0
            audit.returned_sources = result.returned_source_count or len(result.evidence)
            # Give the independent reviewer all collected evidence, including
            # opposing items the Research Agent may not have selected.
            context, result, validated = self.validator.validate(context, snapshot, result, **options)
            audit.final_signal_states = validated.signal_states
            if result.evidence:
                notify("review", "running")
                review, status, audit.review_agent_executed = self.reasoning.run("Review", dict(
                    question=question, mode=mode, trigger_reason=audit.trigger_reason,
                    research_note=plan.research_summary, evidence=[e.model_dump() for e in result.evidence]))
                if status != "ok" or review is None:
                    raise ValueError("Review Agent unavailable; unreviewed evidence withheld.")
                review = ReviewPlan.model_validate(review)
                ids = [item.evidence_id for item in review.assessments]
                allowed = {item.id for item in result.evidence}
                if len(ids) != len(set(ids)) or set(ids) != allowed:
                    raise ValueError("Review Agent must assess each supplied ID exactly once.")
                assessments = {item.evidence_id: item for item in review.assessments}
                accepted = [item.model_copy(update={"direction": assessments[item.id].stance.lower(),
                    "direction_basis": "Semantic review of supplied source text; not verified price impact."}, deep=True)
                    for item in result.evidence if assessments[item.id].accept]
                result.evidence = accepted
                if not accepted:
                    result.status = "empty"
                # The reviewer cannot edit dates/URLs or revive rejected sources.
                context, result, validated = self.validator.validate(context, snapshot, result, **options)
                accepted_ids = {item.id for item in result.evidence}
                result.review = ReviewReport(status="ok", accepted_evidence=sorted(accepted_ids),
                    rejected_evidence=sorted(allowed - accepted_ids),
                    bullish_count=validated.bullish_evidence_count, bearish_count=validated.bearish_evidence_count,
                    neutral_count=validated.neutral_evidence_count, mixed_evidence=validated.mixed_evidence,
                    evidence_quality={"strong": "HIGH", "moderate": "MEDIUM", "limited": "LOW", "none": "NONE"}[validated.evidence_quality],
                    contradictions=redact_values(review.contradictions), unsupported_claims=redact_values(review.unsupported_claims),
                    review_summary=redact_values(review.review_summary), additional_research_needed=review.additional_research_needed)
                if review.additional_research_needed:
                    result.warnings.append("Review requested more evidence; automatic follow-up is disabled by the per-workflow budget.")
                notify("review", "done")
            elif result.status == "unavailable":
                audit.errors.append("Tavily research unavailable.")
        except Exception as exc:
            # Only our fixed diagnostics cross this boundary, never provider errors.
            message = str(exc) if type(exc) is ValueError and str(exc).startswith(("Research Agent", "Review Agent", "Fresh research")) else "Research/review workflow unavailable; evidence withheld."
            audit.errors.append(message)
            notify("review" if audit.review_agent_executed else "research", "unavailable")
            result = ResearchResult(status="unavailable", queries=audit.queries_used, cached=audit.cache_used,
                reason=message, review=ReviewReport(status="unavailable"))
        audit.accepted_sources = len(result.evidence)
        if result.status == "not_requested":
            result.status = "empty"
        audit.rejected_sources = max(0, audit.returned_sources - audit.accepted_sources)
        result.workflow = audit
        persist_research(self.monitoring, result)
        return result
