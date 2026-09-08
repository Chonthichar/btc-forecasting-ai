import copy
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from fastapi.testclient import TestClient

from app import api
from app.monitoring_service import MonitoringService
from app.monitor_worker import next_hourly_run

ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc


class MonitoringIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = MonitoringService(ROOT, Path(self.temp.name) / "monitor.sqlite3")
        self.origin = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        self.market = pd.DataFrame([{"timestamp": self.origin, "close": 100.0}])
        self.pipeline = Mock()
        self.pipeline.market_agent.fetch_closed_candles.return_value = self.market
        self.pipeline.refresh.side_effect = lambda **_: {
            "refreshed_at_utc": datetime.now(UTC).isoformat(), "market": {"price": 100.0},
            "sentiment": {"accepted_articles": 3},
            "quality": {"ok": True, "latest_timestamp": self.origin.isoformat(), "hours_with_news_in_latest_sequence": 3},
            "forecasts": {str(h): {"latest_candle": self.origin.isoformat(), "current_price": 100.0,
                "probability_up": .7, "direction": "UP", "model": "test-model", "feature_set": "market_only"} for h in (1, 6, 24)},
        }
        self.service._pipeline = self.pipeline
        self.old_service = api._service
        api._service = self.service
        self.client = TestClient(api.app)

    def tearDown(self):
        api._service = self.old_service
        self.temp.cleanup()

    def test_query_strings_and_empty_state_for_every_horizon_and_window(self):
        for h in (1, 6, 24):
            for days in (7, 30):
                response = self.client.get(f"/monitoring/summary?horizon={h}&days={days}")
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIsNone(response.json()["accuracy"])
                self.assertEqual(response.json()["total"], 0)
                self.assertEqual(self.client.get(f"/monitoring/history?horizon={h}&days={days}").status_code, 200)
        self.assertEqual(self.client.get("/monitoring/summary?horizon=5").status_code, 422)
        self.assertEqual(self.client.get("/monitoring/history?days=1000").status_code, 422)
        self.assertEqual(self.client.get("/monitoring/history?offset=-1").status_code, 422)
        self.assertEqual(self.client.get("/forecasts").status_code, 409)

    def test_run_records_once_and_api_reads_persisted_snapshot_after_restart(self):
        self.assertEqual(self.service.run()["recorded"]["inserted"], 3)
        original = self.client.get("/forecasts").json()
        self.assertEqual(self.service.run()["status"], "up_to_date")
        self.assertEqual(self.pipeline.refresh.call_count, 1)
        api._service = MonitoringService(ROOT, self.service.store.path)
        self.assertEqual(self.client.get("/forecasts").json(), original)
        summary = self.client.get("/monitoring/summary?horizon=1&days=7").json()
        self.assertEqual(summary["pending"], 1)
        self.assertIsNone(summary["accuracy"])
        self.assertEqual(self.client.get("/health").json()["status"], "ok")

    def test_due_outcome_scored_even_if_new_collection_fails(self):
        old_origin = self.origin - timedelta(hours=1)
        self.service.store.record_forecasts({"1": {"latest_candle": old_origin, "current_price": 90,
            "probability_up": .8, "model_version": self.service.models[1]["model_version"]}},
            issued_at=old_origin + timedelta(hours=1, minutes=2))
        self.pipeline.refresh.side_effect = RuntimeError("provider unavailable")
        self.assertEqual(self.service.run()["status"], "error")
        summary = self.service.summary()
        self.assertEqual(summary["scored"], 1)
        self.assertEqual(summary["accuracy"], 1)
        self.assertEqual(summary["system"]["last_error"], "provider unavailable")
        self.assertFalse(summary["system"]["active_run"])
        self.assertTrue(self.service.store.try_acquire_lock("test-after-error"))

    def test_concurrent_request_returns_busy_without_running_pipeline(self):
        self.service.store.try_acquire_lock("other-process")
        response = self.client.post("/refresh", json={})
        self.assertEqual(response.status_code, 409)
        self.pipeline.refresh.assert_not_called()

    def test_process_guard_prevents_overlap_even_if_database_lease_is_lost(self):
        entered, release = threading.Event(), threading.Event()
        original_refresh = self.pipeline.refresh.side_effect
        def delayed_refresh(**kwargs):
            entered.set()
            release.wait(5)
            return original_refresh(**kwargs)
        self.pipeline.refresh.side_effect = delayed_refresh
        results = []
        worker = threading.Thread(target=lambda: results.append(self.service.run()))
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            owner = self.service.store.get_state("active_run")["owner"]
            self.service.store.release_lock(owner)
            self.assertEqual(self.service.run()["status"], "busy")
        finally:
            release.set()
            worker.join(5)
        self.assertEqual(results[0]["status"], "success")
        self.assertEqual(self.pipeline.refresh.call_count, 1)

    def test_manual_background_refresh_and_static_assets(self):
        result = self.client.post("/monitoring/run?background=true", json={})
        self.assertEqual(result.status_code, 202)
        self.assertEqual(self.service.store.history(days=7)["total"], 3)
        self.assertEqual(self.client.get("/monitor").status_code, 200)
        self.assertEqual(self.client.get("/monitor/assets/dashboard.js").status_code, 200)
        self.assertEqual(self.client.get("/monitor/assets/styles.css").status_code, 200)
        self.assertEqual(self.client.get("/monitor/assets/../../.env").status_code, 404)

    def test_crash_recovery_preserves_original_reference_price(self):
        forecasts = copy.deepcopy(self.pipeline.refresh())
        for h, item in forecasts["forecasts"].items():
            item["model_version"] = self.service.models[int(h)]["model_version"]
            item["current_price"] = 99.0
            item["probability_up"] = .6
        self.service.store.record_forecasts(forecasts["forecasts"])
        self.assertEqual(self.service.run()["status"], "success")
        data = self.service.snapshot()["forecasts"]["1"]
        self.assertEqual(data["current_price"], 99.0)
        self.assertEqual(data["probability_up"], .6)
        self.assertEqual(self.service.store.history()["total"], 3)

    def test_scheduler_runs_two_minutes_after_hour_and_rolls_midnight(self):
        before = datetime(2026, 9, 7, 23, 1, tzinfo=UTC)
        self.assertEqual(next_hourly_run(before), before.replace(minute=2))
        at_slot = before.replace(minute=2)
        self.assertEqual(next_hourly_run(at_slot), datetime(2026, 9, 8, 0, 2, tzinfo=UTC))


if __name__ == "__main__":
    unittest.main()
