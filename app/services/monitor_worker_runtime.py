"""Browser-independent scheduler; API owns its durable lease and due time."""
from __future__ import annotations
import os
import signal
import threading
import uuid
from datetime import datetime, timezone
import requests
from .monitor_schedule import next_cycle

UTC = timezone.utc


def next_hourly_run(now):
    return next_cycle(now, 60)


class MonitoringWorker:
    def __init__(self, api, session=None, now=None):
        self.api = api.rstrip("/")
        # Per-call sessions keep heartbeat and the long scan request independent.
        self.session = session or requests
        self.now = now or (lambda: datetime.now(UTC))
        self.worker_id = uuid.uuid4().hex
        self.stop = threading.Event()

    def heartbeat(self):
        response = self.session.post(self.api + "/monitoring/heartbeat", json={"worker_id": self.worker_id}, timeout=10)
        response.raise_for_status()
        return response.json()

    def step(self):
        status = self.heartbeat()
        if not status.get("accepted"):
            return "standby"
        due = status.get("next_cycle_due")
        if due and self.now() < datetime.fromisoformat(due):
            return "waiting"
        response = self.session.post(self.api + "/monitoring/run", json={
            "trigger": "hourly", "worker_id": self.worker_id,
            "use_gdelt": os.getenv("MONITOR_USE_GDELT", "true").lower() == "true"}, timeout=1500)
        response.raise_for_status()
        return response.json()["status"]

    def run(self):
        def renew():
            while not self.stop.wait(25):
                try:
                    self.heartbeat()
                except (requests.RequestException, ValueError):
                    pass
        thread = threading.Thread(target=renew, daemon=True)
        thread.start()
        try:
            while not self.stop.is_set():
                try:
                    result = self.step()
                    if result not in {"waiting", "standby"}:
                        print(f"Monitoring cycle: {result}", flush=True)
                except (requests.RequestException, ValueError, KeyError) as error:
                    print(f"Monitoring API unavailable ({type(error).__name__}); retry shortly.", flush=True)
                self.stop.wait(10)
        finally:
            self.stop.set()
            thread.join(timeout=12)
            try:
                self.session.post(self.api + "/monitoring/worker/release", json={"worker_id": self.worker_id}, timeout=5)
            except requests.RequestException:
                pass


def main():
    worker = MonitoringWorker(os.getenv("BTC_API_URL", "http://btc-api:8000"))
    signal.signal(signal.SIGTERM, lambda *_: worker.stop.set())
    signal.signal(signal.SIGINT, lambda *_: worker.stop.set())
    worker.run()
