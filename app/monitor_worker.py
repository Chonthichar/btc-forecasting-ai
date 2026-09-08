"""Hourly scheduler; FastAPI is the single inference and database owner."""
from __future__ import annotations
import os
import signal
import threading
from datetime import datetime, timedelta, timezone
import requests

UTC = timezone.utc

def next_hourly_run(now):
    slot = now.replace(minute=2, second=0, microsecond=0)
    return slot if now < slot else slot + timedelta(hours=1)

def main():
    api = os.getenv("BTC_API_URL", "http://btc-api:8000").rstrip("/")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    next_run = [datetime.now(UTC)]

    def heartbeat():
        while not stop.is_set():
            try:
                requests.post(f"{api}/monitoring/heartbeat", json={"next_run_at": next_run[0].isoformat()}, timeout=10).raise_for_status()
            except requests.RequestException:
                pass
            stop.wait(25)

    threading.Thread(target=heartbeat, daemon=True).start()
    while not stop.is_set():
        now = datetime.now(UTC)
        if now < next_run[0]:
            stop.wait(min(30, (next_run[0] - now).total_seconds()))
            continue
        try:
            response = requests.post(f"{api}/monitoring/run", json={"use_gdelt": os.getenv("MONITOR_USE_GDELT", "true").lower() == "true", "trigger": "hourly"}, timeout=1500)
            response.raise_for_status()
            status = response.json()["status"]
            print(f"Monitoring cycle: {status}", flush=True)
            if status in {"error", "busy"}:
                next_run[0] = datetime.now(UTC) + timedelta(minutes=5)
            else:
                next_run[0] = next_hourly_run(datetime.now(UTC))
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(f"Monitoring API unavailable ({type(exc).__name__}); retry in 60s.", flush=True)
            next_run[0] = datetime.now(UTC) + timedelta(seconds=60)
    print("Monitoring scheduler stopped.", flush=True)

if __name__ == "__main__":
    main()
