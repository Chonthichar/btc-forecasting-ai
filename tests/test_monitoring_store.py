import math
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.monitoring_store import MonitoringStore


UTC = timezone.utc
ORIGIN = datetime(2026, 9, 1, 0, tzinfo=UTC)


def forecast(origin=ORIGIN, probability=.8, version="v1", price=100, **extra):
    return {"latest_candle": origin.isoformat(), "current_price": price,
            "probability_up": probability, "model_version": version,
            "model": "test-model", "feature_set": ["close", "volume"], **extra}


def candle(hours, price, **extra):
    return {"timestamp": ORIGIN + timedelta(hours=hours), "close": price, **extra}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "monitoring.sqlite3"
        self.store = MonitoringStore(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def issue(self, horizon=1, data=None, hours=1):
        return self.store.record_forecasts({str(horizon): data or forecast()},
                                           issued_at=ORIGIN + timedelta(hours=hours))

    def test_closed_origin_and_exact_target_close_are_required(self):
        with self.assertRaises(ValueError):
            self.issue(hours=.999)
        self.issue()
        before = ORIGIN + timedelta(hours=2) - timedelta(microseconds=1)
        self.assertEqual(self.store.settle([candle(1, 120)], before)["scored"], 0)
        self.assertEqual(self.store.summary(now=before)["pending"], 1)
        at_close = ORIGIN + timedelta(hours=2)
        self.assertEqual(self.store.settle([candle(1, 120)], at_close)["scored"], 1)
        row = self.store.history(now=at_close)["items"][0]
        self.assertEqual(row["status"], "scored")
        self.assertEqual(row["actual_up"], 1)
        self.assertEqual(row["correct"], 1)
        self.assertEqual(row["target_open"], (ORIGIN + timedelta(hours=1)).isoformat(timespec="microseconds"))
        self.assertEqual(row["target_close"], at_close.isoformat(timespec="microseconds"))

    def test_duplicate_refresh_does_not_change_original_or_rescore(self):
        self.issue()
        duplicate = self.issue(data=forecast(probability=.01, price=500), hours=1.5)
        self.assertEqual(duplicate["duplicates"], 1)
        now = ORIGIN + timedelta(hours=2)
        self.store.settle([candle(1, 120)], now)
        self.assertEqual(self.store.settle([candle(1, 50)], now)["scored"], 0)
        row = self.store.history(now=now)["items"][0]
        self.assertEqual(row["current_price"], 100)
        self.assertEqual(row["probability_up"], .8)
        self.assertEqual(row["target_price"], 120)
        self.assertEqual(row["issued_at"], (ORIGIN + timedelta(hours=1)).isoformat(timespec="microseconds"))

    def test_late_prediction_audited_and_excluded_from_accuracy(self):
        result = self.issue(hours=2)
        self.assertEqual(result["excluded_late"], 1)
        now = ORIGIN + timedelta(hours=3)
        self.assertEqual(self.store.settle([candle(1, 120)], now)["scored"], 0)
        metrics = self.store.summary(now=now)
        self.assertEqual(metrics["excluded"], 1)
        self.assertEqual(metrics["scored"], 0)
        self.assertIsNone(metrics["accuracy"])
        self.assertIsNone(metrics["brier_score"])
        self.assertEqual(self.store.history(status="excluded_late", now=now)["total"], 1)

    def test_missing_target_is_awaiting_and_wrong_candle_cannot_settle(self):
        self.issue(horizon=6)
        now = ORIGIN + timedelta(hours=7)
        self.assertEqual(self.store.settle([candle(5, 120), candle(7, 120)], now),
                         {"scored": 0, "awaiting_market_data": 1})
        metrics = self.store.summary(horizon=6, now=now)
        self.assertEqual(metrics["awaiting_market_data"], 1)
        self.assertEqual(metrics["pending"], 0)
        self.assertEqual(self.store.history(status="awaiting_market_data", now=now)["total"], 1)
        self.assertEqual(self.store.history(status="pending", now=now)["total"], 0)
        self.assertEqual(self.store.settle([candle(6, 90)], now)["scored"], 1)

    def test_ties_are_down_and_probability_metrics_match_outcomes(self):
        self.issue(data=forecast(probability=.25))
        now = ORIGIN + timedelta(hours=2)
        self.store.settle([candle(1, 100)], now)
        metrics = self.store.summary(now=now)
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["baseline_accuracy"], 0)
        self.assertEqual(metrics["brier_score"], .0625)

    def test_versions_and_windows_do_not_mix_metrics(self):
        self.issue(data=forecast(version="v1", probability=.8))
        self.issue(data=forecast(version="v2", probability=.2), hours=1.1)
        now = ORIGIN + timedelta(hours=2)
        self.store.settle([candle(1, 120)], now)
        latest = self.store.summary(now=now)
        self.assertEqual(latest["model_version"], "v2")
        self.assertEqual(latest["accuracy"], 0)
        self.assertEqual(latest["total"], 1)
        self.assertAlmostEqual(latest["brier_score"], .64)
        old = self.store.summary(model_version="v1", now=now)
        self.assertEqual(old["accuracy"], 1)
        self.assertEqual(len(self.store.versions(1)), 2)
        self.assertEqual(self.store.history(now=now)["total"], 2)
        self.assertEqual(self.store.history(model_version="v1", now=now)["total"], 1)
        self.assertEqual(self.store.summary(days=1, now=now + timedelta(days=2))["total"], 0)

    def test_persistence_snapshot_state_runs_and_latest_forecasts(self):
        self.issue()
        snapshot = {"forecasts": {"1": forecast()}, "updated_at": ORIGIN}
        self.store.save_snapshot(snapshot)
        self.store.set_state("worker_heartbeat", {"at": ORIGIN})
        run_id = self.store.start_run("scheduled", now=ORIGIN)
        self.store.finish_run(run_id, "success", {"inserted": 1}, now=ORIGIN + timedelta(seconds=2))
        reopened = MonitoringStore(self.path)
        self.assertEqual(reopened.latest_forecasts()["1"]["model_version"], "v1")
        self.assertEqual(reopened.get_state("worker_heartbeat")["at"], ORIGIN.isoformat(timespec="microseconds"))
        self.assertEqual(reopened.get_state("missing", "default"), "default")
        run = reopened.recent_runs()[0]
        self.assertEqual(run["detail"], {"inserted": 1})
        self.assertEqual(run["duration_ms"], 2000)
        self.assertFalse(reopened.finish_run(run_id, "error", now=ORIGIN + timedelta(seconds=3)))
        self.assertEqual(reopened.history(now=ORIGIN + timedelta(hours=2))["total"], 1)

    def test_latest_forecasts_falls_back_to_stored_records(self):
        self.issue()
        self.assertEqual(self.store.latest_forecasts()["1"]["probability_up"], .8)

    def test_lock_exclusion_renewal_expiry_and_owner_release(self):
        self.assertTrue(self.store.try_acquire_lock("a", ttl_seconds=60, now=ORIGIN))
        self.assertFalse(self.store.try_acquire_lock("b", now=ORIGIN))
        self.assertFalse(self.store.release_lock("b"))
        self.assertTrue(self.store.try_acquire_lock("a", ttl_seconds=120, now=ORIGIN))
        self.assertFalse(self.store.try_acquire_lock("b", now=ORIGIN + timedelta(seconds=60)))
        self.assertTrue(self.store.try_acquire_lock("b", now=ORIGIN + timedelta(seconds=120)))
        self.assertFalse(self.store.release_lock("a"))
        self.assertTrue(self.store.release_lock("b"))

    def test_concurrent_workers_cannot_both_acquire_lock(self):
        barrier = threading.Barrier(8)

        def acquire(index):
            other = MonitoringStore(self.path)
            barrier.wait()
            return other.try_acquire_lock(f"worker-{index}", now=ORIGIN)

        with ThreadPoolExecutor(max_workers=8) as pool:
            result = list(pool.map(acquire, range(8)))
        self.assertEqual(sum(result), 1)

    def test_concurrent_forecasts_are_inserted_once(self):
        barrier = threading.Barrier(6)

        def insert(index):
            other = MonitoringStore(self.path)
            barrier.wait()
            return other.record_forecasts({"1": forecast()}, issued_at=ORIGIN + timedelta(hours=1))

        with ThreadPoolExecutor(max_workers=6) as pool:
            result = list(pool.map(insert, range(6)))
        self.assertEqual(sum(item["inserted"] for item in result), 1)
        self.assertEqual(sum(item["duplicates"] for item in result), 5)

    def test_bad_inputs_fail_before_inserting_any_forecasts(self):
        for probability in (math.nan, math.inf, -1, 2):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                self.issue(data=forecast(probability=probability))
        for price in (0, -1, math.nan, math.inf):
            with self.subTest(price=price), self.assertRaises(ValueError):
                self.issue(data=forecast(price=price))
        with self.assertRaises(ValueError):
            self.issue(data=forecast(origin=ORIGIN.replace(tzinfo=None)))
        with self.assertRaises(ValueError):
            self.issue(data=forecast(origin=ORIGIN + timedelta(minutes=1)))
        with self.assertRaises(ValueError):
            self.store.record_forecasts({"1": forecast(), "6": forecast(probability=10)},
                                        issued_at=ORIGIN + timedelta(hours=1))
        self.assertEqual(self.store.history(now=ORIGIN + timedelta(hours=2))["total"], 0)

    def test_wrong_close_timestamp_and_asset_cannot_supply_ground_truth(self):
        self.issue()
        now = ORIGIN + timedelta(hours=2)
        with self.assertRaises(ValueError):
            self.store.settle([candle(1, 120, close_timestamp=now + timedelta(hours=1))], now)
        self.assertEqual(self.store.settle([candle(1, 120, asset="ETH")], now)["scored"], 0)
        self.assertEqual(self.store.summary(now=now)["scored"], 0)

    def test_outstanding_targets_are_matured_unique_sorted_and_removed_after_settling(self):
        self.issue(horizon=1)
        self.issue(horizon=1, data=forecast(version="v2"))
        self.issue(horizon=6)
        self.assertEqual(self.store.outstanding_targets(now=ORIGIN + timedelta(hours=1)), [])
        now = ORIGIN + timedelta(hours=7)
        expected = [(ORIGIN + timedelta(hours=hour)).isoformat(timespec="microseconds") for hour in (1, 6)]
        self.assertEqual(self.store.outstanding_targets(now=now), expected)
        self.assertEqual(self.store.outstanding_targets(limit=1, now=now), expected[:1])
        self.store.settle([candle(1, 110)], now)
        self.assertEqual(self.store.outstanding_targets(now=now), expected[1:])

    def test_history_pagination_and_daily_series_are_consistent(self):
        for index in range(3):
            origin = ORIGIN + timedelta(hours=index)
            self.store.record_forecasts({"1": forecast(origin=origin)}, issued_at=origin + timedelta(hours=1))
        now = ORIGIN + timedelta(hours=4)
        self.store.settle([candle(1, 120), candle(2, 90), candle(3, 120)], now)
        first = self.store.history(limit=2, now=now)
        second = self.store.history(limit=2, offset=2, now=now)
        self.assertEqual(first["total"], 3)
        self.assertEqual(len(first["items"]), 2)
        self.assertEqual(len(second["items"]), 1)
        self.assertFalse({row["id"] for row in first["items"]} & {row["id"] for row in second["items"]})
        metrics = self.store.summary(now=now)
        self.assertEqual(len(metrics["series"]), 7)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["series"][-1]["scored"], metrics["scored"])
        self.assertEqual(metrics["series"][-1]["accuracy"], metrics["accuracy"])


if __name__ == "__main__":
    unittest.main()
