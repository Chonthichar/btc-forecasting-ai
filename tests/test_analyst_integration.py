"""Offline integration checks for the analyst API and grounded decision flow."""
from __future__ import annotations

import copy
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import app.agents
import app.services

# Allow the parent to test the bounded component artifacts before copying them.
_here = Path(__file__).resolve().parent
for path in (_here, _here.parent / "research"):
    app.agents.__path__.insert(0, str(path))
app.services.__path__.insert(0, str(_here.parent / "research"))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.analyst_api import create_analyst_router, safe_response
from app.agents.orchestrator import AgentOrchestrator
from app.schemas.agent_models import DecisionPlan, EvidenceItem, ResearchResult
from app.services.agent_config import AgentSettings
from app.services.llm_service import LLMService

UTC = timezone.utc


def settings(**updates):
    values = dict(openai_key="", openai_model="", tavily_key="", reliability_threshold=.5,
                  news_cache_minutes=10, news_max_age_hours=72, context_max_age_minutes=90,
                  tavily_timeout_seconds=12, openai_timeout_seconds=45)
    values.update(updates)
    return AgentSettings(**values)


def saved_snapshot(reliability=None, p=.1):
    now = datetime.now(UTC)
    closed = now.replace(minute=0, second=0, microsecond=0)
    opened = closed - timedelta(hours=1)
    forecast = {"horizon_hours": 1, "probability_up": p, "probability_down": 1 - p,
                "current_price": 65000.25, "latest_candle": opened.isoformat(),
                "issued_at": closed.isoformat(), "model": "vae_transformer_v2",
                "feature_set": "market_only", "model_version": "offline-fixture"}
    if reliability is not None:
        forecast["validated_reliability"] = reliability
    return {"refreshed_at_utc": closed.isoformat(),
            "market": {"price": 65000.25, "timestamp": opened.isoformat()},
            "sentiment": {"latest_sentiment_score": -.1, "accepted_articles": 4},
            "quality": {"latest_timestamp": opened.isoformat(), "hours_with_news_in_latest_sequence": 3},
            "forecasts": {"1": forecast}}


def article(id="bull", **updates):
    now = datetime.now(UTC)
    values = dict(id=id, headline="Bitcoin ETF inflows increase", source="Reuters", domain="reuters.com",
                  url="https://www.reuters.com/bitcoin-" + id,
                  published_at=(now - timedelta(minutes=20)).isoformat(), retrieved_at=now.isoformat(),
                  summary="Bitcoin ETF flows were reported.", direction="bullish", category="ETF", source_quality="established")
    values.update(updates)
    return EvidenceItem(**values)


class MonitoringStub:
    def __init__(self, snapshot):
        self.saved = copy.deepcopy(snapshot)

    def snapshot(self):
        return copy.deepcopy(self.saved)


class ResearchStub:
    def __init__(self, result=None):
        self.result = result or ResearchResult(status="empty")
        self.calls = []

    def run(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return self.result.model_copy(deep=True)


class PlanStub:
    def __init__(self, plan=None):
        self.value = plan
        self.calls = []

    def plan(self, *args):
        self.calls.append(args)
        if self.value is None:
            return None, "unavailable", "OpenAI is unavailable. Showing verified context."
        return self.value, "ok", None


class AnalystIntegrationTests(unittest.TestCase):
    def build(self, snapshot=None, research=None, llm=None, config=None):
        config = config or settings()
        monitor = MonitoringStub(saved_snapshot() if snapshot is None else snapshot)
        research = research if research is not None else ResearchStub()
        analyst = AgentOrchestrator(monitor, settings=config, research_agent=research, llm=llm)
        app = FastAPI()
        app.include_router(create_analyst_router(lambda: analyst))
        return TestClient(app), analyst, monitor, research

    def test_providers_absent_and_model_question_never_requests_research(self):
        client, _, _, research = self.build()
        response = client.post("/chat", json={"message": "Which model is deployed for 1h?"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["llm_status"], "unavailable")
        self.assertEqual(body["research_status"], "not_requested")
        self.assertEqual(research.calls, [])
        self.assertIn("vae_transformer_v2", body["answer"])
        self.assertIn("No reliable forecast", body["answer"])

    def test_numeric_snapshot_keys_produce_down_and_mandatory_reliability(self):
        client, _, _, _ = self.build(snapshot=saved_snapshot(reliability=.8, p=.1))
        response = client.post("/chat", json={"message": "Explain the 1h forecast"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["signal_state"], "DOWN")
        self.assertIn("80.0%", body["answer"])
        self.assertIn("1h DOWN", body["answer"])
        self.assertIn("P(UP) = 10.0%", body["answer"])

    def test_valid_plan_cannot_omit_mandatory_reliability_notice(self):
        selected = DecisionPlan(focus_horizons=["24h"], fact_ids=["market.price"], evidence_ids=[],
                                interpretation="model_only", answer_style="concise")
        client, _, _, _ = self.build(snapshot=saved_snapshot(reliability=.1), llm=PlanStub(selected))
        response = client.post("/chat", json={"message": "Explain 1h forecast"})
        body = response.json()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(body["signal_state"], "NO_SIGNAL")
        self.assertIn("1h: No reliable forecast", body["answer"])
        self.assertIn("below the configured threshold", body["answer"])

    def test_forged_plan_ids_fall_back_without_invented_price_or_cause(self):
        forged = DecisionPlan(focus_horizons=["1h"], fact_ids=["invented_price_999999"], evidence_ids=[],
                              interpretation="model_only", answer_style="analytical")
        client, _, _, _ = self.build(llm=PlanStub(forged))
        response = client.post("/chat", json={"message": "Explain 1h forecast because of secret signals"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["llm_status"], "invalid_output")
        self.assertNotIn("999999", body["answer"])
        self.assertIn("No reliable forecast", body["answer"])

    def test_invalid_provider_output_and_exception_have_safe_fallbacks(self):
        config = settings(openai_key="fixture-key", openai_model="fixture-model")
        for effect in ({"answer": "BTC is 999999 because of the Fed"},
                       RuntimeError("Authorization: fixture-key secret provider payload")):
            service = LLMService(config)
            service._run_agent = AsyncMock()
            if isinstance(effect, Exception):
                service._run_agent.side_effect = effect
            else:
                service._run_agent.return_value = effect
            client, _, _, _ = self.build(config=config, llm=service)
            response = client.post("/chat", json={"message": "Explain 1h forecast"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertNotIn("999999", response.text)
            self.assertNotIn("fixture-key", response.text)
            self.assertNotIn("secret provider payload", response.text)
            self.assertIn(response.json()["llm_status"], ("invalid_output", "unavailable"))
            service._run_agent.assert_awaited_once()

    def test_news_calls_research_but_never_promotes_unknown_model(self):
        bearish = article("bear", headline="Bitcoin ETF outflows increase", direction="bearish")
        research = ResearchStub(ResearchResult(status="ok", evidence=[article(), bearish]))
        client, _, _, research = self.build(research=research)
        response = client.post("/chat", json={"message": "What bullish and bearish news affects the 1h forecast?"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(len(research.calls), 1)
        self.assertEqual(body["signal_state"], "UNKNOWN")
        self.assertTrue(body["validation"]["mixed_evidence"])
        self.assertIn("BULLISH", body["answer"])
        self.assertIn("BEARISH", body["answer"])
        self.assertIn("No reliable forecast", body["answer"])
        self.assertFalse(body["validation"]["causal_claim_allowed"])

    def test_context_route_uses_internal_prices_and_does_not_mutate_snapshot(self):
        client, _, monitor, research = self.build()
        before = copy.deepcopy(monitor.saved)
        response = client.get("/context")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["context"]["market"]["price"], 65000.25)
        self.assertEqual(body["context"]["forecasts"]["1"]["signal_state"], "UNKNOWN")
        self.assertIsNone(body["context"]["forecasts"]["1"]["reliability"])
        self.assertTrue(body["validation"]["context_valid"])
        self.assertEqual(monitor.saved, before)
        self.assertEqual(research.calls, [])

    def test_research_route_and_cached_read_revalidate_sources(self):
        now = datetime.now(UTC)
        old = article("old", published_at=(now - timedelta(days=8)).isoformat())
        unsafe = article("unsafe", url="javascript:alert(1)")
        result = ResearchResult(status="ok", evidence=[article(), old, unsafe])
        client, analyst, _, research = self.build(research=ResearchStub(result))
        response = client.post("/research", json={"question": "Bitcoin latest news"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([item["id"] for item in response.json()["research"]["evidence"]], ["bull"])
        self.assertEqual(response.json()["validation"]["rejected_evidence_count"], 2)
        analyst._evidence = ResearchResult(status="ok", evidence=[old])
        cached = client.get("/evidence")
        self.assertEqual(cached.status_code, 200, cached.text)
        self.assertEqual(cached.json()["evidence"], [])
        self.assertEqual(len(research.calls), 1)

    def test_invalid_historical_cutoffs_are_422_without_provider_calls(self):
        client, _, _, research = self.build()
        future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
        for cutoff in (None, "2026-01-01T12:00:00", future):
            payload = {"mode": "historical", "question": "Bitcoin ETF news"}
            if cutoff is not None:
                payload["prediction_timestamp"] = cutoff
            response = client.post("/research", json=payload)
            self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(research.calls, [])

    def test_historical_research_rejects_later_and_undated_evidence(self):
        cutoff = datetime.now(UTC) - timedelta(hours=1)
        items = [article("before", published_at=(cutoff - timedelta(minutes=5)).isoformat()),
                 article("after", headline="Bitcoin exchange activity rises", published_at=(cutoff + timedelta(minutes=5)).isoformat()),
                 article("undated", headline="Bitcoin macro context", published_at=None)]
        client, _, _, _ = self.build(research=ResearchStub(ResearchResult(status="ok", evidence=items)))
        response = client.post("/research", json={"mode": "historical", "prediction_timestamp": cutoff.isoformat(),
                                                   "question": "Bitcoin news before this forecast"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([item["id"] for item in response.json()["research"]["evidence"]], ["before"])

    def test_all_response_boundaries_redact_secrets_without_corrupting_json_types(self):
        secrets = {"OPENAI_API_KEY": "fixture-openai-secret-value", "TAVILY_API_KEY": "fixture-tavily-secret-value",
                   "ALPHA_VANTAGE_API_KEY": "fixture-alpha-secret-value"}
        with patch.dict(os.environ, secrets):
            raw = saved_snapshot()
            raw["market"]["regime"] = "unvalidated " + secrets["OPENAI_API_KEY"]
            config = settings(openai_key=secrets["OPENAI_API_KEY"], openai_model="fixture-model", tavily_key=secrets["TAVILY_API_KEY"])
            item = article(headline="Bitcoin source " + secrets["TAVILY_API_KEY"])
            result = ResearchResult(status="ok", evidence=[item], reason="api_key=" + secrets["ALPHA_VANTAGE_API_KEY"])
            client, _, _, _ = self.build(snapshot=raw, config=config, research=ResearchStub(result), llm=PlanStub())
            responses = [client.get("/context"), client.get("/agents/status"),
                         client.post("/research", json={"question": "Bitcoin latest news"}),
                         client.get("/evidence"), client.post("/chat", json={"message": "Explain 1h forecast"})]
            for response in responses:
                self.assertEqual(response.status_code, 200, response.text)
                for secret in secrets.values():
                    self.assertNotIn(secret, response.text)
            cleaned = safe_response({"enabled": True, "amount": 1.25, "items": ["api_key=untrusted-token", None]})
            self.assertIs(cleaned["enabled"], True)
            self.assertEqual(cleaned["amount"], 1.25)
            self.assertIsNone(cleaned["items"][1])
            self.assertNotIn("untrusted-token", cleaned["items"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
