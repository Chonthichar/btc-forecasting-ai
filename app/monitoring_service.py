"""One owner for live inference, durable predictions, and delayed evaluation."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .monitoring_store import MonitoringStore
from .services.agent_config import AgentSettings
from .services.monitoring_events_store import EventStore
from .services.monitor_schedule import next_cycle
from .services.monitoring_backup import MonitoringBackup

UTC = timezone.utc


def utcnow():
    return datetime.now(UTC)


def safe_error(exc):
    from .services.agent_config import redact
    return redact(str(exc))[-600:]


class MonitoringService:
    def __init__(self, project_root, db_path=None):
        self.root = Path(project_root)
        path = db_path or os.getenv("BTC_MONITOR_DB", str(self.root / "runtime/monitoring.sqlite3"))
        self.backup = MonitoringBackup()
        restored = self.backup.restore(path)
        self.store = MonitoringStore(path)
        if restored:
            self.store.set_state("backup", {"status": "RESTORED", "restored_at": utcnow().isoformat()})
        self.work_dir = Path(self.store.path).parent
        self.settings = AgentSettings.from_env()
        self.event_store = EventStore(self.store)
        self._market_monitor = None
        self._pipeline = None
        self._pipeline_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self.models = self._model_specs()

    def _model_specs(self):
        registry = yaml.safe_load((self.root / "app/deployment_registry.yaml").read_text(encoding="utf-8"))
        specs = {}
        for h, spec in registry.items():
            directory = self.root / spec["output_subdir"]
            fingerprint = hashlib.sha256()
            files = sorted([directory / "model_state.pt", *directory.glob("*.joblib")])
            files += [self.root / spec["config"], self.root / "src/features.py", self.root / "src/pipeline.py", self.root / "app/agents/forecast_agent.py", self.root / "app/agents/market_agent.py"]
            for file in files:
                fingerprint.update(file.name.encode())
                fingerprint.update(file.read_bytes())
            specs[int(h)] = {"horizon_hours": int(h), "model": spec["kind"], "feature_set": spec["feature_set"], "model_version": fingerprint.hexdigest()[:16]}
        return specs

    def get_pipeline(self):
        if self._pipeline is None:
            with self._pipeline_lock:
                if self._pipeline is None:
                    from .realtime_pipeline import RealtimeCryptoPipeline
                    self._pipeline = RealtimeCryptoPipeline(
                        self.root, os.getenv("BTC_SENTIMENT_SCRIPT", str(self.root / "sentiment_pipeline/btc_sentiment_agents.py")),
                        live_work_dir=self.work_dir, alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
                    )
        return self._pipeline

    def snapshot(self):
        return self.store.latest_snapshot() or {}

    def heartbeat(self, next_run_at=None, worker_id=None):
        if worker_id and not self.event_store.acquire("scheduler", worker_id, ttl=90):
            return {"accepted": False, "next_cycle_due": self.store.get_state("next_cycle_due")}
        due = self.store.get_state("next_cycle_due")
        started = self.store.get_state("last_cycle_started")
        completed = self.store.get_state("last_cycle_finished") or self.store.get_state("last_cycle_completed")
        active = self.store.get_state("active_run") or {}
        if started and (not completed or started > completed):
            last_beat = active.get("heartbeat_at", active.get("started_at", started))
            if (utcnow() - datetime.fromisoformat(last_beat)).total_seconds() > 120:
                due = utcnow().isoformat()
                self.store.set_state("next_cycle_due", due)
        if due is None:
            last = self.store.get_state("last_cycle_completed") or self.store.get_state("last_success")
            due = next_cycle(datetime.fromisoformat(last), self.settings.monitor_interval_minutes).isoformat() if last else utcnow().isoformat()
            self.store.set_state("next_cycle_due", due)
        if worker_id:
            next_run_at = due
        state = {"at": utcnow().isoformat(), "next_run_at": next_run_at}
        if worker_id:
            state["worker_id"] = worker_id
        self.store.set_state("worker", state)
        return {**state, "accepted": True, "next_cycle_due": due, "monitor_interval_minutes": self.settings.monitor_interval_minutes,
                "last_cycle_started": self.store.get_state("last_cycle_started"),
                "last_cycle_completed": self.store.get_state("last_cycle_completed"),
                "last_research_trigger": self.store.get_state("last_research_trigger"),
                "last_error": self.store.get_state("last_error")}

    def market_monitor(self):
        if self._market_monitor is None:
            from .services.market_monitoring import MarketMonitoringAgent
            self._market_monitor = MarketMonitoringAgent(self)
        return self._market_monitor

    def monitoring_activity(self):
        system = self.system_status()
        event = self.event_store.latest_event()
        return {"worker_alive": system["worker_alive"],
                "last_cycle_started": self.store.get_state("last_cycle_started"),
                "last_cycle_completed": self.store.get_state("last_cycle_completed"),
                "last_market_scan": self.store.get_state("last_market_scan"),
                "last_successful_prediction_cycle": self.store.get_state("last_success"),
                "next_cycle_due": self.store.get_state("next_cycle_due"),
                "last_research_trigger": self.store.get_state("last_research_trigger"),
                "research": self.store.get_state("research_operation", {"status": "IDLE"}),
                "last_research": event, "last_error": system["last_error"],
                "backup": self.store.get_state("backup", {"status": "CONFIGURED" if self.backup.repo else "LOCAL_ONLY"}),
                "monitor_interval_minutes": self.settings.monitor_interval_minutes}

    def system_status(self):
        worker = self.store.get_state("worker", {}) or {}
        worker_at = worker.get("at")
        worker_alive = bool(worker_at and (utcnow() - datetime.fromisoformat(worker_at)).total_seconds() < 100)
        active = self.store.get_state("active_run", {}) or {}
        if active and (utcnow() - datetime.fromisoformat(active.get("heartbeat_at", active["started_at"]))).total_seconds() > 120:
            active = {}
        snapshot = self.snapshot()
        quality = snapshot.get("quality", {})
        latest_open = quality.get("latest_timestamp")
        latest_close = (pd.Timestamp(latest_open) + pd.Timedelta(hours=1)).isoformat() if latest_open else None
        stale = bool(latest_close and (pd.Timestamp.now(tz="UTC") - pd.Timestamp(latest_close)).total_seconds() > 4200)
        return {"worker_alive": worker_alive, "worker": worker, "active_run": active,
                "last_success": self.store.get_state("last_success"), "last_error": self.store.get_state("last_error"),
                "last_evaluation_error": self.store.get_state("last_evaluation_error"),
                "latest_candle_close": latest_close, "data_stale": stale,
                "quality": quality, "sentiment": snapshot.get("sentiment", {}), "runs": self.store.recent_runs(5)}

    def summary(self, horizon=1, days=7, model_version=None):
        model = self.models[horizon]
        version = model_version or model["model_version"]
        result = self.store.summary(horizon=horizon, days=days, model_version=version)
        history = self.store.history(horizon=horizon, days=365, model_version=version, limit=1)
        versions = self.store.versions(horizon)
        if not any(v["model_version"] == model["model_version"] for v in versions):
            versions.insert(0, {"model_version": model["model_version"], "total": 0, "latest_issued_at": None})
        return {**result, "model": model, "models": list(self.models.values()), "versions": versions,
                "latest": next(iter(history["items"]), None), "system": self.system_status(), "generated_at": utcnow().isoformat(),
                "label_rule": "UP when target close > reference close; unchanged counts as not UP.",
                "window_basis": "Forecast issue time; scored outcomes only; selected model version."}

    def _settle(self, pipeline, candles):
        # Binance close_timestamp is inclusive (xx:59:59.999). The store uses
        # the canonical boundary at the next full hour, derived from open time.
        result = self.store.settle(candles[["timestamp", "close"]].to_dict("records"))
        targets = self.store.outstanding_targets(limit=1000)
        for _ in range(3):
            if not targets:
                break
            page_start = pd.Timestamp(targets[0])
            backfill = pipeline.market_agent.fetch_closed_candles(limit=1000, start_time=page_start)
            scored = self.store.settle(backfill[["timestamp", "close"]].to_dict("records"))
            result["scored"] += scored["scored"]
            result["awaiting_market_data"] = scored["awaiting_market_data"]
            # Advance even if this page contains a permanently missing candle.
            boundary = page_start + pd.Timedelta(hours=1000)
            targets = [t for t in targets if pd.Timestamp(t) >= boundary]
        return result

    def run(self, trigger="manual", use_gdelt=True, worker_id=None):
        owner = uuid.uuid4().hex
        if not self._run_lock.acquire(blocking=False):
            return {"status": "busy", "detail": "A refresh is already running."}
        try:
            acquired = self.store.try_acquire_lock(owner, ttl_seconds=120)
        except Exception:
            self._run_lock.release()
            raise
        if not acquired:
            self._run_lock.release()
            return {"status": "busy", "detail": "A refresh is already running."}
        if trigger == "hourly":
            due = self.store.get_state("next_cycle_due")
            if (worker_id and not self.event_store.owns("scheduler", worker_id)) or (due and utcnow() < datetime.fromisoformat(due)):
                self.store.release_lock(owner)
                self._run_lock.release()
                return {"status": "not_due", "next_cycle_due": due}
        started = utcnow()
        clock_start = time.monotonic()
        renew_stop, lease_lost = threading.Event(), threading.Event()
        run_id = None

        def renew():
            while not renew_stop.wait(20):
                try:
                    if not self.store.try_acquire_lock(owner, ttl_seconds=120):
                        lease_lost.set()
                        return
                    active = self.store.get_state("active_run")
                    if active and active.get("owner") == owner:
                        active["heartbeat_at"] = utcnow().isoformat()
                        self.store.set_state("active_run", active)
                except Exception:
                    lease_lost.set()
                    return

        renewal = threading.Thread(target=renew, daemon=True)
        try:
            run_id = self.store.start_run(trigger)
            self.store.set_state("last_cycle_started", started.isoformat())
            self.store.set_state("next_cycle_due", next_cycle(started, self.settings.monitor_interval_minutes).isoformat())
            self.store.set_state("research_operation", {"status": "IDLE", "monitor": "running", "forecast": "running", "trigger": "idle", "validation": "idle", "decision": "idle"})
            old_active = self.store.get_state("active_run")
            if old_active and old_active.get("id"):
                self.store.finish_run(old_active["id"], "failed", detail="Previous process stopped; retrying with saved history.")
            self.store.set_state("active_run", {"id": run_id, "owner": owner, "started_at": started.isoformat(), "trigger": trigger})
            renewal.start()
            pipeline = self.get_pipeline()
            market = pipeline.market_agent.fetch_closed_candles(limit=500)
            if market.empty:
                raise RuntimeError("Market provider returned no closed candles.")
            try:
                settled = self._settle(pipeline, market)
                self.store.set_state("last_evaluation_error", None)
            except Exception as exc:
                self.store.set_state("last_evaluation_error", safe_error(exc))
                settled = {"scored": 0, "awaiting_market_data": None}
            origin = pd.Timestamp(market.iloc[-1]["timestamp"])
            previous_forecasts = self.snapshot().get("forecasts", {})
            unchanged = all(str(h) in previous_forecasts
                and previous_forecasts[str(h)].get("model_version") == model["model_version"]
                and pd.Timestamp(previous_forecasts[str(h)]["latest_candle"]) == origin
                for h, model in self.models.items())
            if unchanged:
                status, recorded = "up_to_date", {"inserted": 0}
            else:
                data = pipeline.refresh(use_gdelt=use_gdelt, market_candles=market)
                if lease_lost.is_set() or not self.store.try_acquire_lock(owner, ttl_seconds=120):
                    raise RuntimeError("Refresh ownership expired; no forecasts were recorded. The worker will retry.")
                issued_at = utcnow()
                for h, forecast in data["forecasts"].items():
                    forecast["model_version"] = self.models[int(h)]["model_version"]
                    forecast["issued_at"] = issued_at.isoformat()
                quantitative = self.market_monitor().prepare(data, market, now=issued_at)
                data["market"]["regime"] = quantitative["regime"]
                recorded = self.store.record_forecasts(data["forecasts"], issued_at=issued_at,
                    context_by_horizon=quantitative["forecasts"], regime=quantitative["regime"])
                # Recover the first-issued values after a crash between record
                # insertion and snapshot saving, without rewriting predictions.
                for h, forecast in data["forecasts"].items():
                    rows = self.store.history(horizon=int(h), days=2, model_version=forecast["model_version"], limit=5)["items"]
                    original = next((r for r in rows if pd.Timestamp(r["origin_open"]) == pd.Timestamp(forecast["latest_candle"])), None)
                    if original:
                        forecast["probability_up"] = original["probability_up"]
                        forecast["probability_down"] = 1 - original["probability_up"]
                        forecast["direction"] = "UP" if original["probability_up"] >= .5 else "DOWN"
                        forecast["issued_at"] = original["issued_at"]
                        forecast["current_price"] = original["current_price"]
                        forecast["model"] = original["model"]
                        forecast["feature_set"] = original["feature_set"]
                self.store.save_snapshot(json.loads(json.dumps(data, default=str)))
                self.store.set_state("last_success", issued_at.isoformat())
                status = "success"
            try:
                research_trigger = self.market_monitor().process(run_id, self.snapshot(), market)
            except Exception:
                research_trigger = {"should_research": False, "triggered_by": [], "error": "Monitoring context processing failed; forecasts remain saved."}
                self.store.set_state("research_operation", {"status": "ERROR", "error": research_trigger["error"]})
            self.store.set_state("last_error", None)
            self.store.set_state("last_cycle_completed", utcnow().isoformat())
            duration = round((time.monotonic() - clock_start) * 1000)
            self.store.finish_run(run_id, "skipped" if status == "up_to_date" else status, duration_ms=duration)
            return {"status": status, "recorded": recorded, "settled": settled, "duration_ms": duration, "research_trigger": research_trigger}
        except Exception as exc:
            detail = safe_error(exc)
            self.store.set_state("last_error", detail)
            from datetime import timedelta
            self.store.set_state("next_cycle_due", (utcnow() + timedelta(minutes=min(5, self.settings.monitor_interval_minutes))).isoformat())
            self.store.set_state("research_operation", {"status": "IDLE", "monitor": "error", "forecast": "error", "trigger": "idle", "validation": "idle", "decision": "idle"})
            if run_id is not None:
                self.store.finish_run(run_id, "error", detail=detail, duration_ms=round((time.monotonic() - clock_start) * 1000))
            return {"status": "error", "detail": detail}
        finally:
            try:
                self.store.set_state("last_cycle_finished", utcnow().isoformat())
                renew_stop.set()
                if renewal.is_alive():
                    renewal.join(timeout=2)
                active = self.store.get_state("active_run")
                if active and active.get("owner") == owner:
                    self.store.set_state("active_run", None)
                self.store.release_lock(owner)
            finally:
                self._run_lock.release()
            self.backup.save(self.store)
