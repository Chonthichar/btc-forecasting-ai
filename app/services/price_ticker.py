"""Spot quotes for the dashboard's asset switcher.

These are exchange quotes, nothing more. A price here says nothing about whether
an asset has a trained model: Ethereum and Cardano quote normally while having no
forecast at all. Model inputs come from MarketDataAgent's closed candles, never
from this cache.
"""
from __future__ import annotations
import json
import threading
import time

import requests

ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/ticker/24hr",
    "https://api.binance.com/api/v3/ticker/24hr",
]


class PriceTicker:
    def __init__(self, ttl_seconds: int = 60):
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._fetched_at = 0.0
        self._quotes: dict[str, dict] = {}
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "btc-thesis-dashboard/1.0"})

    def quotes(self, symbols) -> dict[str, dict]:
        symbols = [str(symbol) for symbol in symbols if symbol]
        if not symbols:
            return {}
        with self._lock:
            fresh = self._quotes and time.monotonic() - self._fetched_at < self.ttl
            if fresh:
                return dict(self._quotes)
        fetched = self._fetch(symbols)
        with self._lock:
            if fetched:
                self._quotes = fetched
                self._fetched_at = time.monotonic()
            # On failure the previous quotes are served rather than blanking the
            # switcher; they carry their own timestamp for the caller to judge.
            return dict(self._quotes)

    def _fetch(self, symbols) -> dict[str, dict]:
        params = {"symbols": json.dumps(symbols, separators=(",", ":"))}
        for endpoint in ENDPOINTS:
            try:
                response = self._session.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                rows = response.json()
                if isinstance(rows, dict):
                    rows = [rows]
                return {
                    row["symbol"]: {
                        "price": float(row["lastPrice"]),
                        "change_24h": float(row["priceChangePercent"]) / 100,
                    }
                    for row in rows
                    if row.get("symbol") and row.get("lastPrice") is not None
                }
            except Exception:
                continue
        return {}
