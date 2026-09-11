"""Scheduled scans and research gating with real SQLite and offline providers."""
import copy
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from fastapi.testclient import TestClient
from app import api
from app.monitoring_service import MonitoringService
from app.monitor_worker import MonitoringWorker
from app.services.market_monitoring import MarketMonitoringAgent
from app.services.monitoring_events_store import EventStore
from app.services.research_triggers import ResearchTriggerEngine, market_metrics
from app.services.monitor_schedule import next_cycle
from app.schemas.agent_models import ResearchResult, ChatRequest, DecisionPlan
from app.agents.market_research_agent import MarketResearchAgent
from app.agents.orchestrator import AgentOrchestrator, needs_research
from test_analyst_integration import settings, saved_snapshot, article, ResearchStub, PlanStub

ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc


class EventMonitoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = MonitoringService(ROOT, Path(self.temp.name) / "events.sqlite3")
        self.service.settings = settings()
        self.raw = saved_snapshot(p=.1)
        one = self.raw["forecasts"]["1"]
        one["current_price"] = 100.
        self.raw["forecasts"] = {str(h): {**copy.deepcopy(one), "horizon_hours": h} for h in (1, 6, 24)}
        self.raw["market"].update(price=100., regime="RANGE")
        self.origin = datetime.fromisoformat(one["latest_candle"])
        self.market = pd.DataFrame({"timestamp": pd.date_range(end=self.origin, periods=500, freq="h"), "close": [100.] * 500})
        self.research = ResearchStub(ResearchResult(status="ok", evidence=[article()]))
        selected = DecisionPlan(focus_horizons=["1h"], fact_ids=[], evidence_ids=["bull"], interpretation="bullish_context", answer_style="concise",
                                answer="The report describes ETF inflows. {{source:bull}}")
        self.llm = PlanStub(selected)
        self.monitor = MarketMonitoringAgent(self.service, self.research, self.llm)
        self.service._market_monitor = self.monitor
        self.pipeline = Mock()
        self.pipeline.market_agent.fetch_closed_candles.side_effect = lambda **_: self.market.copy()
        self.pipeline.refresh.side_effect = lambda **_: copy.deepcopy(self.raw)
        self.service._pipeline = self.pipeline
        self.old_service = api._service
        api._service = self.service
        self.client = TestClient(api.app)
        self.seed_previous()

    def tearDown(self):
        api._service = self.old_service
        self.temp.cleanup()

    def seed_previous(self, transform=None):
        previous = self.monitor.prepare(self.raw, self.market)
        previous["market_timestamp"] = (self.origin - timedelta(hours=1)).isoformat()
        if transform:
            transform(previous)
        # Replace only this fixture's baseline scan; never production history.
        with self.service.store._connection() as db:
            db.execute("DELETE FROM monitoring_scans")
        self.service.event_store.scan(-1, previous, {"should_research": False, "triggered_by": []})

    def price_move(self):
        self.market.loc[self.market.index[-1], "close"] = 102.
        self.raw["market"]["price"] = 102.
        for forecast in self.raw["forecasts"].values():
            forecast["current_price"] = 102.

    def test_ordinary_cycle_records_scan_without_paid_calls(self):
        result = self.service.run()
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["research_trigger"]["should_research"])
        self.assertEqual(self.research.calls, [])
        self.assertEqual(self.llm.calls, [])
        self.assertEqual(len(self.service.event_store.scans()), 2)
        self.assertTrue(self.service.monitoring_activity()["last_cycle_completed"])

    def test_large_price_move_calls_research_and_saves_context_reference(self):
        self.price_move()
        result = self.service.run()
        self.assertIn("large_price_move", result["research_trigger"]["triggered_by"])
        self.assertEqual(len(self.research.calls), 1)
        self.assertEqual(len(self.llm.calls), 1)
        event = self.service.event_store.latest_event()
        self.assertEqual(event["status"], "done")
        self.assertIn("ETF inflows", event["summary"])
        scan = self.service.event_store.scans(1)[0]
        self.assertEqual(scan["research_event_id"], event["id"])
        self.assertNotIn("evidence", scan["payload"])

    def test_regime_transition_triggers_research(self):
        self.raw["market"]["regime"] = "HIGH_VOL"
        self.assertIn("regime_change", self.service.run()["research_trigger"]["triggered_by"])
        self.assertEqual(len(self.research.calls), 1)

    def test_signed_internal_sentiment_shift_triggers_research(self):
        self.raw["sentiment"]["latest_sentiment_score"] = .4
        self.seed_previous()
        self.raw["sentiment"]["latest_sentiment_score"] = -.4
        self.assertIn("sentiment_shift", self.service.run()["research_trigger"]["triggered_by"])
        self.assertEqual(len(self.research.calls), 1)

    def test_small_probability_change_does_not_trigger(self):
        for forecast in self.raw["forecasts"].values():
            forecast["probability_up"] = .51
        self.seed_previous()
        for forecast in self.raw["forecasts"].values():
            forecast["probability_up"] = .52
        self.assertFalse(self.service.run()["research_trigger"]["should_research"])
        self.assertEqual(self.research.calls, [])

    def test_reliability_crossing_no_signal_to_down_triggers(self):
        for forecast in self.raw["forecasts"].values():
            forecast["validated_reliability"] = .4
        self.seed_previous()
        for forecast in self.raw["forecasts"].values():
            forecast["validated_reliability"] = .8
        self.assertIn("forecast_state_change", self.service.run()["research_trigger"]["triggered_by"])
        self.assertEqual(len(self.research.calls), 1)

    def test_same_candle_never_researches_again_even_after_restart(self):
        self.price_move()
        self.service.run()
        self.service._market_monitor = MarketMonitoringAgent(self.service, self.research, self.llm)
        result = self.service.run()
        self.assertEqual(result["status"], "up_to_date")
        self.assertEqual(len(self.research.calls), 1)
        self.assertEqual(len(self.llm.calls), 1)

    def test_persistent_cooldown_reuses_event_without_research_or_decision(self):
        self.price_move()
        self.service.run()
        current = self.monitor.prepare(self.service.snapshot(), self.market)
        current["market_timestamp"] = (self.origin + timedelta(hours=1)).isoformat()
        reopened = MarketMonitoringAgent(self.service, self.research, self.llm)
        with patch.object(reopened, "prepare", return_value=current):
            result = reopened.process(999, self.service.snapshot(), self.market)
        self.assertTrue(result["suppressed"])
        self.assertEqual(len(self.research.calls), 1)
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual(self.service.monitoring_activity()["research"]["status"], "CACHED")

    def test_tavily_failure_does_not_fail_market_cycle(self):
        self.price_move()
        self.research.result = ResearchResult(status="unavailable")
        result = self.service.run()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["recorded"]["inserted"], 3)
        self.assertEqual(self.llm.calls, [])
        self.assertIsNone(self.service.store.get_state("last_error"))
        self.assertEqual(self.service.monitoring_activity()["research"]["status"], "ERROR")

    def test_openai_failure_preserves_forecasts_and_scan(self):
        self.price_move()
        self.llm.value = None
        result = self.service.run()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["recorded"]["inserted"], 3)
        self.assertEqual(self.service.event_store.latest_event()["status"], "error")
        self.assertIsNone(self.service.event_store.latest_event()["summary"])

    def test_bearish_news_cannot_promote_unreliable_model(self):
        self.price_move()
        for forecast in self.raw["forecasts"].values():
            forecast["validated_reliability"] = .42
        self.research.result = ResearchResult(status="ok", evidence=[article("bear", headline="Bitcoin ETF outflows rise", direction="bearish")])
        self.service.run()
        for view in self.service.event_store.scans(1)[0]["payload"]["forecasts"].values():
            self.assertEqual(view["signal_state"], "NO_SIGNAL")
            self.assertEqual(view["reliability"], .42)
        self.assertEqual(self.service.summary()["qualification"]["no_signal_frequency"], 1.)

    def test_no_useful_validated_evidence_means_no_decision_call(self):
        self.price_move()
        self.research.result = ResearchResult(status="ok", evidence=[article(published_at=None)])
        self.service.run()
        self.assertEqual(self.llm.calls, [])

    def test_forward_outcome_resolution_keeps_original_prediction_and_gate(self):
        self.raw["forecasts"]["1"]["validated_reliability"] = .8
        self.service.run()
        original = self.service.store.history(horizon=1)["items"][0]
        now = self.origin + timedelta(hours=2, minutes=1)
        self.service.store.settle([{"timestamp": self.origin + timedelta(hours=1), "close": 90}], now=now)
        row = self.service.store.history(horizon=1, now=now)["items"][0]
        self.assertEqual(row["probability_up"], original["probability_up"])
        self.assertEqual(row["issued_at"], original["issued_at"])
        self.assertEqual(row["correct"], 1)
        metric = self.service.store.summary(horizon=1, now=now)
        self.assertEqual(metric["qualification"]["qualified_performance"]["accuracy"], 1.)
        self.assertAlmostEqual(metric["brier_score"], .01)

    def test_two_workers_and_restart_execute_without_browser_or_duplicates(self):
        client = self.client
        class Transport:
            def post(self, url, json, timeout):
                return client.post(url.replace("http://testserver", ""), json=json)
        first = MonitoringWorker("http://testserver", session=Transport())
        second = MonitoringWorker("http://testserver", session=Transport())
        self.assertEqual(first.step(), "success")
        self.assertEqual(second.step(), "standby")
        self.assertEqual(first.step(), "waiting")
        self.service.event_store.release("scheduler", first.worker_id)
        self.assertEqual(second.step(), "waiting")
        self.assertEqual(first.step(), "standby")
        self.assertEqual(self.pipeline.refresh.call_count, 1)
        self.assertEqual(self.research.calls, [])

    def test_explicit_fresh_why_research_bypasses_monitoring_cooldown(self):
        self.price_move()
        self.service.run()
        analyst = AgentOrchestrator(self.service, self.service.settings, self.research, self.llm)
        analyst.chat(ChatRequest(message="Why is the model predicting DOWN?", fresh_research=True))
        self.assertEqual(len(self.research.calls), 2)
        self.assertTrue(self.research.calls[-1][1]["fresh"])


class TriggerUnitTests(unittest.TestCase):
    def test_volatility_trigger_and_missing_baseline(self):
        engine = ResearchTriggerEngine(settings())
        point = {"data_fresh": True, "market_timestamp": "new", "volatility": .03, "volatility_baseline": .01}
        self.assertIn("unusual_volatility", engine.evaluate(point)["triggered_by"])
        point["volatility_baseline"] = None
        self.assertFalse(engine.evaluate(point)["should_research"])

    def test_stale_context_never_triggers_paid_research(self):
        point = {"data_fresh": False, "market_timestamp": "new", "price_change_1h": 20}
        self.assertFalse(ResearchTriggerEngine(settings()).evaluate(point)["should_research"])

    def test_price_threshold_is_configurable(self):
        point = {"data_fresh": True, "market_timestamp": "new", "price_change_1h": 1.5}
        self.assertFalse(ResearchTriggerEngine(settings(price_move_trigger_pct=2)).evaluate(point)["should_research"])

    def test_query_plans_are_bounded_relevant_and_direction_neutral(self):
        for trigger, keyword in (("large_price_move", "liquidations"), ("unusual_volatility", "liquidations"), ("sentiment_shift", "regulation")):
            queries = MarketResearchAgent.query_plan("model DOWN", trigger_types=[trigger])
            self.assertEqual(len(queries), 3)
            self.assertIn(keyword, " ".join(queries))
            self.assertNotIn("will fall", " ".join(queries))

    def test_requested_market_explanations_route_to_research(self):
        for question in ("Why is BTC moving?", "Why is the model predicting DOWN?", "What happened in the market?", "What news is affecting BTC?", "Explain the latest move."):
            self.assertTrue(needs_research(question), question)

    def test_configurable_cadence_and_midnight_rollover(self):
        now = datetime(2026, 9, 8, 23, 59, tzinfo=UTC)
        self.assertEqual(next_cycle(now, 60), datetime(2026, 9, 9, 0, 2, tzinfo=UTC))
        self.assertEqual(next_cycle(now.replace(hour=9, minute=5), 15), now.replace(hour=9, minute=17))

    def test_metrics_do_not_invent_returns_across_missing_candles(self):
        times = pd.date_range("2026-01-01", periods=26, freq="h", tz="UTC")
        frame = pd.DataFrame({"timestamp": times, "close": [100.] * 26}).drop(index=24)
        point = market_metrics(frame, {})
        self.assertIsNone(point["price_change_1h"])
        self.assertIsNone(point["volatility"])


class DurableQueryCacheTests(unittest.TestCase):
    def test_restart_reuses_query_cache_but_explicit_fresh_bypasses_it(self):
        with tempfile.TemporaryDirectory() as directory:
            service = MonitoringService(ROOT, Path(directory) / "cache.sqlite3")
            tavily = Mock()
            tavily.search.return_value = {"status": "empty", "results": []}
            config = settings()
            first = MarketResearchAgent(config, tavily, service.event_store)
            first.run("Bitcoin latest market news")
            self.assertEqual(tavily.search.call_count, 3)
            second = MarketResearchAgent(config, tavily, EventStore(service.store))
            self.assertTrue(second.run("Bitcoin latest market news").cached)
            self.assertEqual(tavily.search.call_count, 3)
            second.run("Bitcoin latest market news", fresh=True)
            self.assertEqual(tavily.search.call_count, 6)


if __name__ == "__main__":
    unittest.main()
