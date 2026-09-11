"""Deterministic routing around Research, Review and Decision SDK agents."""
from __future__ import annotations
import copy
import re
import threading
import time
import uuid
from collections import OrderedDict
from app.schemas.agent_models import AgentActivity, ChatResponse, ResearchResult, SafeChatMessage
from app.services.agent_config import AgentSettings, redact
from .forecast_context_agent import ForecastContextAgent
from .market_research_agent import MarketResearchAgent
from .validation_agent import ValidationAgent
from .decision_agent import DecisionAgent, horizons_for
from app.services.research_workflow import ResearchWorkflow, persist_research
from app.services.market_decision import enforce_decision

def needs_research(question):
    text = question.lower()
    if re.search(r"(?:no|without|do not|don't)\s+(?:web|search|research|brows)|internal (?:context|data|only)", text):
        return False
    if (re.search(r"(?:what|which).*(?:model|feature|input)|(?:does|is|are).*(?:model|feature|input).*(?:use|include|support)", text)
            and not re.search(r"(?:latest|current).*(?:news|headline)|search|research", text)):
        return False
    if re.search(r"why.*(?:model|predict|forecast).*(?:up|down)|what happened.*market|explain.*(?:latest|current).*mov|current events|external.*(?:events?|context|evidence)|fresh.*research", text):
        return True
    if re.search(r"news|headlines|moving|market.moving|etf|fed\b|macro|inflation|cpi\b|employment|jobs report|regulat|liquidat|funding|open interest|whale|institution|bullish.*bearish|bearish.*bullish|factors|geopolit|hack", text):
        return True
    if (re.search(r"\b(?:btc|bitcoin)\b", text) and re.search(r"\b(?:why|today|now|current)\b", text)
            and re.search(r"\b(?:up|down|surged?|rall\w*|fell|fall\w*|ris\w*|drop\w*|crash\w*|mov\w*)\b", text)):
        return True
    return bool(re.search(r"summari[sz]e.*(?:now|today)|market.*(?:today|now)", text))

class AgentOrchestrator:
    def __init__(self, monitoring, settings=None, research_agent=None, llm=None):
        self.monitoring = monitoring
        self.settings = settings or AgentSettings.from_env()
        self.forecast_agent = ForecastContextAgent(self.settings)
        self.research_agent = research_agent or ResearchWorkflow(monitoring, self.settings)
        self.validation_agent = ValidationAgent(self.settings)
        self.decision_agent = DecisionAgent(self.settings, self.validation_agent, llm=llm)
        self._lock = threading.RLock()
        self._activity = OrderedDict()
        self._evidence = ResearchResult()
        self._slots = threading.BoundedSemaphore(3)

    def _set(self, request_id, **fields):
        with self._lock:
            current = self._activity.get(request_id, (0, AgentActivity()))[1]
            current = current.model_copy(update=fields)
            self._activity[request_id] = (time.monotonic(), current)
            self._activity.move_to_end(request_id)
            while len(self._activity) > 128:
                self._activity.popitem(last=False)

    def activity(self, request_id=None):
        with self._lock:
            item = self._activity.get(request_id)
            state = item[1] if item else AgentActivity()
            return {"agent_status": state.model_dump(), "configuration": self.settings.public_status()}

    def _failed(self, request_id):
        states = self.activity(request_id)["agent_status"]
        self._set(request_id, **{key: "error" for key, value in states.items() if value == "running"})

    def context(self):
        snapshot = copy.deepcopy(self.monitoring.snapshot())
        context = self.forecast_agent.run(snapshot)
        context, _, validation = self.validation_agent.validate(context, snapshot, ResearchResult())
        research = self.evidence()
        decision = enforce_decision(context, research, validation, ["1h", "6h", "24h"])
        return {"context": context.model_dump(), "validation": validation.model_dump(), "configuration": self.settings.public_status(),
                "market_decision": decision.model_dump()}

    def evidence(self):
        with self._lock:
            latest = self._evidence.model_copy(deep=True)
        store = getattr(self.monitoring, "event_store", None)
        if store:
            saved = store.latest_event(with_evidence=True)
            candidates = [saved["result"] if saved else None, self.monitoring.store.get_state("analyst_evidence")]
            for candidate in candidates:
                if candidate and (candidate.get("retrieved_at") or "") > (latest.retrieved_at or ""):
                    latest = ResearchResult.model_validate(candidate)
        # Revalidate cached evidence at read time; a source can age out while the
        # dashboard stays open. No new provider calls are made by GET requests.
        snapshot = copy.deepcopy(self.monitoring.snapshot())
        context = self.forecast_agent.run(snapshot)
        context, latest, validation = self.validation_agent.validate(context, snapshot, latest)
        if latest.market_decision:
            latest.market_decision = enforce_decision(context, latest, validation, ["1h", "6h", "24h"], proposal=latest.market_decision)
        return latest

    def research(self, request):
        if not self._slots.acquire(blocking=False):
            raise RuntimeError("The analyst is busy. Please retry shortly.")
        request_id = request.request_id or uuid.uuid4().hex
        try:
            return self._research(request, request_id)
        except Exception:
            self._failed(request_id)
            raise
        finally:
            self._slots.release()

    def _research(self, request, request_id):
        self._set(request_id, forecast="running")
        snapshot = copy.deepcopy(self.monitoring.snapshot())
        context = self.forecast_agent.run(snapshot)
        self._set(request_id, forecast="done", research="running")
        options = {"fresh": True} if request.fresh_research else {}
        research = self.research_agent.run(redact(request.question), mode=request.mode, prediction_timestamp=request.prediction_timestamp,
            user_requested=True, on_state=lambda stage, status: self._set(request_id, **{stage: status}), **options)
        self._set(request_id, research="unavailable" if research.status == "unavailable" else "done", validation="running")
        context, research, validation = self.validation_agent.validate(context, snapshot, research, mode=request.mode, prediction_timestamp=request.prediction_timestamp)
        decision = enforce_decision(context, research, validation, horizons_for(request.question))
        executed = False
        if any(e.recency_verified and e.source_quality != "unverified" for e in research.evidence):
            self._set(request_id, decision="running")
            _, llm_status, _, decision = self.decision_agent.run(request.question, [], context, research, validation, structured=True)
            executed = bool(self.settings.openai_key and self.settings.openai_model)
            self._set(request_id, decision="done" if llm_status == "ok" else "unavailable")
            if research.workflow and llm_status != "ok":
                research.workflow.errors.append("Decision Agent " + llm_status)
        research.market_decision = decision
        persist_research(self.monitoring, research, decision, executed)
        if request.mode == "live":
            with self._lock:
                self._evidence = research.model_copy(deep=True)
            if getattr(self.monitoring, "event_store", None):
                self.monitoring.store.set_state("analyst_evidence", research.model_dump())
        self._set(request_id, validation="done", review="done" if research.review and research.review.status == "ok" else "unavailable" if research.review else "idle")
        return {"research": research.model_dump(), "validation": validation.model_dump(), "request_id": request_id,
                "market_decision": decision.model_dump(),
                "agent_status": self.activity(request_id)["agent_status"]}

    def chat(self, request):
        request_id = request.request_id or uuid.uuid4().hex
        if not self._slots.acquire(blocking=False):
            raise RuntimeError("The analyst is busy. Please retry shortly.")
        try:
            question = redact(request.message)
            history = [SafeChatMessage(role=m.role, content=redact(m.content)) for m in request.history[-12:]]
            self._set(request_id, forecast="running")
            snapshot = copy.deepcopy(self.monitoring.snapshot())
            context = self.forecast_agent.run(snapshot)
            self._set(request_id, forecast="done")
            if needs_research(question) or request.fresh_research:
                self._set(request_id, research="running")
                fresh = request.fresh_research or bool(re.search(r"\b(?:fresh|new)\s+(?:web\s+)?(?:search|research)|search again|ignore.*cache", question, re.I))
                research = self.research_agent.run(question, user_requested=True,
                    on_state=lambda stage, status: self._set(request_id, **{stage: status}), **({"fresh": True} if fresh else {}))
                self._set(request_id, research="unavailable" if research.status == "unavailable" else "done")
            else:
                research = ResearchResult()
                self._set(request_id, research="skipped")
            self._set(request_id, validation="running")
            context, research, validation = self.validation_agent.validate(context, snapshot, research)
            if research.status != "not_requested":
                with self._lock:
                    self._evidence = research.model_copy(deep=True)
                if getattr(self.monitoring, "event_store", None):
                    self.monitoring.store.set_state("analyst_evidence", research.model_dump())
            self._set(request_id, validation="done", decision="running")
            answer, llm_status, sources, decision = self.decision_agent.run(question, history, context, research, validation, structured=True)
            research.market_decision = decision
            if research.workflow and llm_status != "ok":
                research.workflow.errors.append("Decision Agent " + llm_status)
            persist_research(self.monitoring, research, decision, bool(self.settings.openai_key and self.settings.openai_model))
            if research.status != "not_requested":
                with self._lock:
                    self._evidence = research.model_copy(deep=True)
                if getattr(self.monitoring, "event_store", None):
                    self.monitoring.store.set_state("analyst_evidence", research.model_dump())
            self._set(request_id, decision="done" if llm_status == "ok" else "unavailable",
                review="done" if research.review and research.review.status == "ok" else "unavailable" if research.review else "idle")
            states = [context.forecasts[h[:-1]].signal_state for h in horizons_for(question) if h[:-1] in context.forecasts]
            unique = set(states)
            state = states[0] if len(unique) == 1 else "NO_SIGNAL" if "NO_SIGNAL" in unique else "UNKNOWN"
            return ChatResponse(answer=answer, signal_state=state, context_timestamp=context.timestamp,
                research_used=research.status != "not_requested", research_status=research.status, sources=sources,
                agent_status=AgentActivity(**self.activity(request_id)["agent_status"]), validation=validation,
                llm_status=llm_status, request_id=request_id, market_decision=decision)
        except Exception:
            self._failed(request_id)
            raise
        finally:
            self._slots.release()
