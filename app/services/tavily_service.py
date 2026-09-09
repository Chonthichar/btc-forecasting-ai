"""Small server-side Tavily boundary with a fixed provider and safe failures."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import requests


def prediction_cutoff(value):
    """Return an aware UTC cutoff; never assume a timezone for historical work."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime) or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def unavailable(reason):
    return {"status": "unavailable", "reason": reason, "results": []}


class TavilyService:
    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, settings, session=None):
        self.settings = settings
        # The requests module creates a separate session per call. Do not share
        # mutable Session state between concurrent callers by default.
        self.session = session if session is not None else requests

    def search(self, query, mode="live", prediction_timestamp=None):
        if mode not in ("live", "historical"):
            return unavailable("Unsupported research mode.")
        cutoff = prediction_cutoff(prediction_timestamp) if mode == "historical" else None
        if mode == "historical" and cutoff is None:
            return unavailable("Historical research requires a timezone-aware prediction timestamp.")
        if not self.settings.tavily_key:
            return unavailable("Tavily is not configured.")
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 512:
            return unavailable("The research query is invalid.")
        payload = {"query": query.strip(), "topic": "news", "search_depth": "basic",
                   "max_results": 5, "include_answer": False, "include_raw_content": False}
        if mode == "historical":
            payload.update(start_date=(cutoff - timedelta(days=7)).date().isoformat(),
                           end_date=cutoff.date().isoformat())
        else:
            payload["time_range"] = "day" if self.settings.news_max_age_hours <= 24 else "week"
        try:
            timeout = float(self.settings.tavily_timeout_seconds)
            timeout = min(60., max(1., timeout)) if math.isfinite(timeout) else 12.
            response = self.session.post(self.ENDPOINT, json=payload,
                                         headers={"Authorization": "Bearer " + self.settings.tavily_key,
                                                  "Content-Type": "application/json"},
                                         timeout=timeout, allow_redirects=False)
            status_code = response.status_code
            if status_code == 429:
                return unavailable("Tavily is rate limited; try again later.")
            if status_code in (401, 403):
                return unavailable("Tavily authentication is unavailable.")
            if status_code in (432, 433):
                return unavailable("Tavily usage limits were reached.")
            if status_code != 200:
                return unavailable("Tavily could not complete the research request.")
            body = response.json()
        except requests.Timeout:
            return unavailable("Tavily timed out; try again later.")
        except Exception:
            # Exception messages and response bodies can contain credentials.
            return unavailable("Tavily returned an unavailable or invalid response.")
        if not isinstance(body, dict) or not isinstance(body.get("results"), list):
            return unavailable("Tavily returned an invalid result format.")
        if not body["results"]:
            return {"status": "empty", "reason": "No news results were returned.", "results": []}
        results = []
        for item in body["results"][:5]:
            if not isinstance(item, dict):
                continue
            if not all(isinstance(item.get(field), str) and item[field].strip() for field in ("title", "url")):
                continue
            # Only known evidence fields cross the provider boundary.
            result = {key: item.get(key) for key in
                      ("title", "url", "content", "score", "published_date", "published_at")}
            if not isinstance(result["content"], str):
                result["content"] = ""
            results.append(result)
        if not results:
            return unavailable("Tavily returned no usable result records.")
        return {"status": "ok", "reason": None, "results": results}
