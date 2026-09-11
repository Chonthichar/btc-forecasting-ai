"""Offline SDK, routing, concurrency, and request-boundary hardening checks."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import unittest
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Reuse only fixtures; these imports also expose the unintegrated artifact paths.
from test_analyst_integration import MonitoringStub, ResearchStub, PlanStub, settings, saved_snapshot
from app.analyst_api import create_analyst_router
from app.agents.decision_agent import build_facts
from app.agents.forecast_context_agent import ForecastContextAgent
from app.agents.orchestrator import AgentOrchestrator, needs_research
from app.agents.validation_agent import ValidationAgent
from app.schemas.agent_models import ChatRequest, DecisionPlan, ResearchRequest, ResearchResult, SafeChatMessage
from app.services.llm_service import LLMService


def client_for(analyst):
    app = FastAPI()
    app.include_router(create_analyst_router(lambda: analyst))
    return TestClient(app)


def response_payload(plan):
    return {
        "id": "resp_offline_fixture", "object": "response", "created_at": 1,
        "status": "completed", "model": "fixture-model", "error": None,
        "incomplete_details": None, "instructions": None, "parallel_tool_calls": False,
        "tool_choice": "auto", "tools": [], "temperature": 1, "top_p": 1,
        "output": [{"id": "msg_offline_fixture", "type": "message", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text",
                    "text": json.dumps(plan), "annotations": []}]}],
    }


class SDKContractTests(unittest.TestCase):
    def test_agents_provider_failure_does_not_retry_or_expose_private_error(self):
        config = settings(openai_key="fixture-key", openai_model="fixture-model")
        raw = saved_snapshot()
        context = ForecastContextAgent(config).run(raw)
        context, research, validation = ValidationAgent(config).validate(context, raw, ResearchResult())
        requests = []
        def fail(request):
            requests.append(request)
            return httpx.Response(503, json={"error": {"message": "private-provider-body", "type": "server_error"}})
        sdk = AsyncOpenAI(api_key="fixture-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(fail)), max_retries=0)
        result, status, message = LLMService(config, client=sdk).plan(
            "hello", [], context, research, validation, build_facts(context))
        self.assertIsNone(result)
        self.assertEqual(status, "unavailable")
        self.assertEqual(len(requests), 1)
        self.assertNotIn("private-provider-body", message)

    def test_real_agents_runner_uses_strict_schema_one_request_and_redacted_json(self):
        secret = "fixture-openai-private-value"
        other_secret = "fixture-tavily-private-value"
        config = settings(openai_key=secret, openai_model="fixture-model", tavily_key=other_secret)
        raw = saved_snapshot()
        raw["market"]["regime"] = "provider-note token=" + other_secret
        context = ForecastContextAgent(config).run(raw)
        context, research, validation = ValidationAgent(config).validate(context, raw, ResearchResult())
        facts = build_facts(context)
        selected = dict(focus_horizons=["1h"], fact_ids=["market.price"], evidence_ids=[],
                        interpretation="model_only", answer_style="concise")
        captured = []

        def respond(request):
            self.assertEqual(str(request.url), "https://api.openai.com/v1/responses")
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.headers["Authorization"], "Bearer " + secret)
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=response_payload(selected))

        with patch.dict(os.environ, {"OPENAI_API_KEY": secret, "TAVILY_API_KEY": other_secret}):
            with patch("agents.Runner.run", wraps=__import__("agents").Runner.run) as runner:
                sdk = AsyncOpenAI(api_key=secret, base_url="https://api.openai.com/v1",
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)), max_retries=0)
                service = LLMService(config, client=sdk)
                output, status, message = service.plan(
                    "Explain 1h; api_key=" + secret,
                    [SafeChatMessage(role="user", content="token=" + other_secret)],
                    context, research, validation, facts,
                )
                self.assertEqual(runner.call_count, 1)
                self.assertEqual(runner.call_args.kwargs["max_turns"], 1)
                self.assertTrue(runner.call_args.kwargs["run_config"].tracing_disabled)
                self.assertFalse(runner.call_args.kwargs["run_config"].trace_include_sensitive_data)
        self.assertEqual(status, "ok", message)
        self.assertIsInstance(output, DecisionPlan)
        self.assertEqual(output.model_dump(), {**selected, "answer": None, "market_decision": None})
        self.assertEqual(len(captured), 1)
        request = captured[0]
        self.assertFalse(request["store"])
        self.assertEqual(request["max_output_tokens"], 2200)
        self.assertEqual(request["model"], "fixture-model")
        serialized = json.dumps(request)
        self.assertNotIn(secret, serialized)
        self.assertNotIn(other_secret, serialized)
        # Valid JSON survives key=value redaction inside individual string fields.
        model_input = json.loads(request["input"][0]["content"])
        self.assertIn("[redacted]", model_input["question"])
        self.assertIn("[redacted]", model_input["history"][0]["content"])
        self.assertEqual(model_input["context"]["market"]["price"], 65000.25)
        form = request["text"]["format"]
        self.assertEqual(form["type"], "json_schema")
        self.assertTrue(form["strict"])
        self.assertFalse(form["schema"]["additionalProperties"])
        self.assertEqual(set(form["schema"]["properties"]),
                         {"focus_horizons", "fact_ids", "evidence_ids", "interpretation", "answer_style", "answer", "market_decision"})

    def test_real_sdk_rejects_causal_prose_extra_without_exposing_response_body(self):
        config = settings(openai_key="fixture-key", openai_model="fixture-model")
        raw = saved_snapshot()
        context = ForecastContextAgent(config).run(raw)
        context, research, validation = ValidationAgent(config).validate(context, raw, ResearchResult())
        invalid = dict(focus_horizons=["1h"], fact_ids=["market.price"], evidence_ids=[],
                       interpretation="model_only", answer_style="concise",
                       because="Unsupported model causation: private-fixture-response")
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=response_payload(invalid)))
        sdk = AsyncOpenAI(api_key="fixture-key", http_client=httpx.AsyncClient(transport=transport), max_retries=0)
        output, status, message = LLMService(config, client=sdk).plan(
            "Explain 1h forecast", [], context, research, validation, build_facts(context))
        self.assertIsNone(output)
        self.assertIn(status, ("unavailable", "invalid_output"))
        self.assertNotIn("private-fixture-response", message)
        self.assertNotIn("Unsupported model causation", message)


class RoutingTests(unittest.TestCase):
    def test_external_events_route_to_research_and_model_or_no_web_questions_do_not(self):
        cases = {
            "Why is BTC down today?": True,
            "Bitcoin surged today; why?": True,
            "How could the latest CPI report affect Bitcoin?": True,
            "Explain US employment news and BTC": True,
            "Does this model use ETF flows as inputs?": False,
            "Which model features include ETF data?": False,
            "Without web research, why is BTC down today?": False,
            "No web search: explain current CPI implications": False,
            "Use internal context only to explain the 1h forecast": False,
            "Which model is deployed for 6h?": False,
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(needs_research(question), expected)
                research = ResearchStub()
                analyst = AgentOrchestrator(MonitoringStub(saved_snapshot()), settings=settings(),
                                            research_agent=research, llm=PlanStub())
                result = analyst.chat(ChatRequest(message=question))
                self.assertEqual(len(research.calls), int(expected))
                self.assertEqual(result.research_used, expected)


class BlockingResearch:
    def __init__(self):
        self.condition = threading.Condition()
        self.started = 0
        self.release = threading.Event()

    def run(self, question, **kwargs):
        with self.condition:
            self.started += 1
            self.condition.notify_all()
        if not self.release.wait(timeout=8):
            raise RuntimeError("Offline blocking fixture timed out")
        return ResearchResult(status="empty")


class OperationalTests(unittest.TestCase):
    def test_research_and_chat_share_exactly_three_concurrent_slots(self):
        research = BlockingResearch()
        analyst = AgentOrchestrator(MonitoringStub(saved_snapshot()), settings=settings(),
                                    research_agent=research, llm=PlanStub())
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(analyst.research, ResearchRequest(question="Bitcoin ETF news", request_id="r1")),
                       executor.submit(analyst.research, ResearchRequest(question="Bitcoin CPI news", request_id="r2")),
                       executor.submit(analyst.chat, ChatRequest(message="Bitcoin headlines", request_id="c1"))]
            try:
                with research.condition:
                    self.assertTrue(research.condition.wait_for(lambda: research.started == 3, timeout=4))
                with self.assertRaises(RuntimeError):
                    analyst.research(ResearchRequest(question="Bitcoin news", request_id="r4"))
                with self.assertRaises(RuntimeError):
                    analyst.chat(ChatRequest(message="Which model is deployed?", request_id="c4"))
                self.assertEqual(research.started, 3)
            finally:
                research.release.set()
                for future in futures:
                    future.result(timeout=4)
        # Slots are released after both kinds of request complete.
        result = analyst.chat(ChatRequest(message="Which model is deployed for 1h?"))
        self.assertEqual(result.research_status, "not_requested")

    def test_operational_failures_update_activity_and_release_slots(self):
        for operation in ("research", "forecast", "validation", "decision"):
            with self.subTest(operation=operation):
                analyst = AgentOrchestrator(MonitoringStub(saved_snapshot()), settings=settings(),
                                            research_agent=ResearchStub(), llm=PlanStub())
                client = client_for(analyst)
                target, method = {"research": (analyst.research_agent, "run"),
                                  "forecast": (analyst.forecast_agent, "run"),
                                  "validation": (analyst.validation_agent, "validate"),
                                  "decision": (analyst.decision_agent, "run")}[operation]
                with patch.object(target, method, side_effect=RuntimeError("sensitive-fixture-payload")):
                    route = "/research" if operation == "research" else "/chat"
                    field = "question" if route == "/research" else "message"
                    response = client.post(route, json={field: "Explain 1h forecast", "request_id": operation})
                self.assertEqual(response.status_code, 503, response.text)
                self.assertNotIn("sensitive-fixture-payload", response.text)
                states = client.get("/agents/status", params={"request_id": operation}).json()["agent_status"]
                self.assertEqual(states[operation], "error")
                self.assertNotIn("running", states.values())
                self.assertEqual(client.post("/chat", json={"message": "Which model is deployed for 1h?"}).status_code, 200)


class RequestBoundaryTests(unittest.TestCase):
    def test_malformed_secret_inputs_are_never_echoed_by_standalone_router(self):
        secret = "fixture-secret-submitted-invalid-input"
        analyst = AgentOrchestrator(MonitoringStub(saved_snapshot()), settings=settings(),
                                    research_agent=ResearchStub(), llm=PlanStub())
        client = client_for(analyst)
        requests = [
            ("/chat", {"message": {"OPENAI_API_KEY": secret}}),
            ("/chat", {"message": "hello", "history": [{"role": secret, "content": "hello"}]}),
            ("/chat", {"message": "hello", "unrecognized": secret}),
            ("/research", {"question": "Bitcoin news", "mode": secret}),
            ("/research", {"mode": "historical", "prediction_timestamp": secret}),
        ]
        for route, payload in requests:
            with self.subTest(route=route, fields=list(payload)):
                response = client.post(route, json=payload)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertNotIn(secret, response.text)
                self.assertNotIn("input", response.json().get("detail", ""))
        response = client.post("/chat", content='{"message":"' + secret + '"',
                               headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertNotIn(secret, response.text)
        response = client.get("/agents/status", params={"request_id": "!" + secret})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertNotIn(secret, response.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
