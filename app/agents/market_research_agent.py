"""Explicit, model-independent BTC news research and bounded evidence records."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, TimeoutError as FutureTimeout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import ipaddress
import math
import re
import threading
import time
import uuid
import warnings
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from dateutil import parser as date_parser

from app.schemas.agent_models import EvidenceItem, ResearchResult
from app.services.tavily_service import TavilyService, prediction_cutoff, unavailable
from app.services.agent_config import redact_values

PRIMARY = {
    "federalreserve.gov": "Federal Reserve", "sec.gov": "SEC", "treasury.gov": "US Treasury",
    "bls.gov": "BLS", "bea.gov": "BEA", "cftc.gov": "CFTC", "ecb.europa.eu": "ECB",
    "imf.org": "IMF", "blackrock.com": "BlackRock", "ishares.com": "iShares",
    "fidelity.com": "Fidelity", "coinbase.com": "Coinbase", "kraken.com": "Kraken", "binance.com": "Binance",
    "cmegroup.com": "CME Group",
}
ESTABLISHED = {
    "reuters.com": "Reuters", "bloomberg.com": "Bloomberg", "ft.com": "Financial Times",
    "wsj.com": "Wall Street Journal", "cnbc.com": "CNBC", "apnews.com": "Associated Press",
    "coindesk.com": "CoinDesk", "theblock.co": "The Block", "cointelegraph.com": "Cointelegraph", "decrypt.co": "Decrypt",
}
TOPICS = (
    (r"\betfs?\b|\bflows?\b|\binstitution", "Bitcoin spot ETF net inflows outflows institutional demand"),
    (r"\bmacro\b|\bfed\b|\bfomc\b|\brates?\b|\binflation\b|\bdollar\b|\bcpi\b|\bemployment\b|\bpayrolls?\b",
     "Bitcoin Federal Reserve interest rates inflation CPI employment dollar macroeconomic news"),
    (r"\bfutures?\b|\bfunding\b|\bliquidat|\bderivatives?\b|\bleverage\b|\bopen interest\b",
     "Bitcoin futures funding open interest long short liquidations"),
    (r"\bregulat|\bsec\b|\bpolicy\b|\blegal\b|\bapproval\b|\benforcement\b",
     "Bitcoin regulation SEC approval enforcement exchange policy"),
    (r"\bhack|\bsecurity\b|\bexchange\b|\boutage\b|\bwithdrawal",
     "Bitcoin exchange security incidents withdrawals trading operations"),
    (r"\bwhales?\b|\bon[ -]?chain\b|\bwallet\b|\blarge transfers?\b",
     "Bitcoin whale wallet activity exchange inflows outflows on-chain transfers"),
    (r"\bgeopolit|\brisk[ -]?off\b|\brisk[ -]?on\b|\bwar\b|\bconflict\b|\btariffs?\b|\bsanctions?\b",
     "Bitcoin geopolitical developments global risk appetite risk-off risk-on market news"),
)
SENSITIVE_QUERY = re.compile(r"(?i)^(?:api_?key|key|token|access_?token|refresh_?token|auth|authorization|"
                             r"password|passwd|secret|signature|credential|session|sessionid|access_?key|auth_?code|"
                             r"x_amz_(?:credential|signature|security_token)|awsaccesskeyid)$")
TOKEN_PATTERN = re.compile(r"(?i)\b(?:tvly-[a-z0-9_-]{8,}|sk-[a-z0-9_-]{10,})")


def _now():
    return datetime.now(timezone.utc).isoformat()


def safe_source_url(value):
    """Validate a public source link without fetching it or resolving DNS."""
    if not isinstance(value, str) or not 1 <= len(value) <= 2048:
        return None
    decoded = unquote(unquote(value))
    if re.search(r"[\x00-\x20\x7f\\]", value) or re.search(r"[\x00-\x1f\x7f\\]", decoded):
        return None
    if TOKEN_PATTERN.search(decoded):
        return None
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() not in ("http", "https") or parts.username is not None or parts.password is not None:
            return None
        host = (parts.hostname or "").rstrip(".").lower().encode("idna").decode("ascii")
        if not host or parts.port not in (None, 80, 443):
            return None
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".test")):
            return None
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if "." not in host or not re.fullmatch(r"[a-z0-9.-]+", host) or host.rsplit(".", 1)[-1].isdigit():
                return None
            if len(host) > 253 or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                                      for label in host.split(".")):
                return None
        else:
            if not address.is_global:
                return None
        query = parse_qsl(parts.query, keep_blank_values=True, max_num_fields=100)
        for key, _ in query:
            if SENSITIVE_QUERY.fullmatch(unquote(key).replace("-", "_")):
                return None
        query = [(key, val) for key, val in query if not key.lower().startswith("utm_") and
                 key.lower() not in {"gclid", "fbclid", "msclkid", "mc_cid", "mc_eid"}]
        netloc = "[" + host + "]" if ":" in host else host
        if parts.port is not None and not (parts.scheme == "https" and parts.port == 443 or parts.scheme == "http" and parts.port == 80):
            netloc += ":" + str(parts.port)
        return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", urlencode(query), ""))
    except (ValueError, UnicodeError):
        return None


def source_quality(domain):
    for quality, mapping in (("primary", PRIMARY), ("established", ESTABLISHED)):
        for known, label in mapping.items():
            if domain == known or domain.endswith("." + known):
                return quality, label
    return "unverified", domain


def publication_timestamp(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 100:
        return None
    value = value.strip()
    if not (re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[Tt\s])", value) or
            re.search(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b|\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\b", value)):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = date_parser.parse(value, fuzzy=False)
    except (ValueError, OverflowError, TypeError):
        return None
    # Preserve provider date-only/naive precision for the validator; never fill
    # a missing publication timezone or timestamp from retrieval time.
    if parsed.utcoffset() is None:
        return value
    return parsed.astimezone(timezone.utc).isoformat()


def headline_direction(headline):
    text = headline.lower()
    if not re.search(r"\b(?:bitcoin|btc)\b", text):
        return "neutral"
    if re.search(r"\b(?:no|not|may|might|could|would|prediction|predictions|forecast|forecasts|expected)\b", text):
        return "neutral"
    if re.search(r"\b(?:inflows?|outflows?)\b.{0,35}\b(?:slow\w*|ease\w*|declin\w*|fall\w*|drop\w*)\b", text):
        return "neutral"
    bullish = bool(re.search(r"\binflows?\b|\b(?:bitcoin|btc)(?:\s+price)?\s+(?:rises?|rose|rallies|gains?|surges?|climbs?|rebounds?)\b", text))
    bullish = bullish or bool(re.search(r"\betfs?\b.{0,32}\b(?:pulls? in|attracts?|draws?)\s+\$?\d|"
        r"\binstitutional demand (?:recovers|rises|rebounds|strengthens)\b", text))
    bearish = bool(re.search(r"\boutflows?\b|\b(?:bitcoin|btc)(?:\s+price)?\s+(?:falls?|fell|drops?|slumps?|declines?|plunges?|dips?|crashes?|sinks?)\b", text))
    return "neutral" if bullish == bearish else ("bullish" if bullish else "bearish")


def evidence_category(headline, snippet):
    text = (headline + " " + snippet[:200]).lower()
    for pattern, category in (
        (r"\betfs?\b", "ETF"), (r"\bliquidat", "LIQUIDATIONS"),
        (r"\bhack|\bexploit|\bsecurity breach", "SECURITY"),
        (r"\bfed\b|\bfederal reserve\b|\bfomc\b", "FED"),
        (r"\bregulat|\bsec\b|\bcftc\b|\benforcement\b", "REGULATION"),
        (r"\binflation\b|\bcpi\b|\binterest rates?\b|\bdollar\b|\bmacro|\bemployment\b|\bpayrolls?\b|\bgeopolit|\brisk[ -]?(?:off|on)\b", "MACRO"),
        (r"\bfutures?\b|\bfunding\b|\bopen interest\b|\bderivative", "DERIVATIVES"),
        (r"\bwhales?\b", "WHALE"), (r"\bexchange\b|\bcoinbase\b|\bkraken\b", "EXCHANGE"),
        (r"\binstitution|\btreasury\b", "INSTITUTIONAL"),
        (r"\bliquidity\b|\border book\b|\bmarket structure\b", "MARKET_STRUCTURE"),
    ):
        if re.search(pattern, text):
            return category
    return "OTHER"


class MarketResearchAgent:
    def __init__(self, settings, tavily=None, store=None):
        self.settings = settings
        self.tavily = tavily if tavily is not None else TavilyService(settings)
        self._cache = OrderedDict()
        self._inflight = {}
        self._guard = threading.Lock()
        self.store = store

    @staticmethod
    def query_plan(question, mode="live", trigger_types=None):
        if trigger_types:
            topics = []
            rules = {
                "large_price_move": [TOPICS[2][1], TOPICS[0][1]],
                "unusual_volatility": [TOPICS[2][1], TOPICS[1][1]],
                "regime_change": [TOPICS[1][1], TOPICS[2][1]],
                "sentiment_shift": [TOPICS[0][1], TOPICS[3][1]],
                "forecast_state_change": [TOPICS[0][1], TOPICS[1][1]],
            }
            for reason in trigger_types:
                for query in rules.get(reason, []):
                    if query not in topics:
                        topics.append(query)
            return ["Bitcoin BTC latest market news price drivers"] + topics[:2]
        text = str(question).lower()[:2000]
        chosen = [query for pattern, query in TOPICS if re.search(pattern, text)][:2]
        for _, query in TOPICS[:2]:
            if len(chosen) == 2:
                break
            if query not in chosen:
                chosen.append(query)
        prefix = "Bitcoin BTC latest market news price drivers" if mode == "live" else "Bitcoin BTC market news price drivers"
        return [prefix] + chosen

    def _search(self, query, mode, cutoff, fresh=False):
        key = (query, mode, cutoff.isoformat() if cutoff else None)
        disk_key = hashlib.sha256(repr((key, self.settings.news_max_age_hours)).encode()).hexdigest()
        if self.store and not fresh:
            try:
                disk = self.store.cache_get(disk_key)
            except Exception:
                return {**unavailable("The shared research cache is unavailable."), "retrieved_at": _now()}, False
            if disk is not None:
                return disk, True
        with self._guard:
            entry = self._cache.get(key)
            if not fresh and entry and entry[0] > time.monotonic():
                self._cache.move_to_end(key)
                return deepcopy(entry[1]), True
            future = self._inflight.get(key)
            leader = future is None
            if leader:
                future = Future()
                self._inflight[key] = future
        if not leader:
            try:
                return deepcopy(future.result(timeout=self.settings.tavily_timeout_seconds * 2 + 5)), True
            except FutureTimeout:
                return {**unavailable("The shared research request is still in progress."), "retrieved_at": _now()}, False
        lease_owner = uuid.uuid4().hex
        if self.store:
            acquired, disk = False, None
            try:
                acquired = self.store.acquire("query:" + disk_key, lease_owner, ttl=self.settings.tavily_timeout_seconds + 30)
                # Another process may have filled the cache after our first read.
                if not fresh or not acquired:
                    disk = self.store.cache_get(disk_key)
            except Exception:
                acquired = False
            if disk is not None or not acquired:
                payload = disk or {**unavailable("The shared research request is in progress or unavailable."), "retrieved_at": _now()}
                if acquired:
                    try:
                        self.store.release("query:" + disk_key, lease_owner)
                    except Exception:
                        pass
                with self._guard:
                    self._inflight.pop(key, None)
                    future.set_result(deepcopy(payload))
                return payload, True
        try:
            payload = self.tavily.search(query, mode=mode, prediction_timestamp=cutoff)
            if not isinstance(payload, dict) or payload.get("status") not in ("ok", "empty", "unavailable"):
                payload = unavailable("The news provider returned an invalid response.")
            payload = {**payload, "retrieved_at": _now(), "provider_called": bool(self.settings.tavily_key)}
        except Exception:
            payload = {**unavailable("The news provider is unavailable."), "retrieved_at": _now(), "provider_called": bool(self.settings.tavily_key)}
        ttl = max(0., self.settings.news_cache_minutes * 60)
        if payload["status"] == "unavailable":
            ttl = min(ttl, 30.)
        if self.store:
            try:
                # Retain bounded snippets only; do not persist whole article bodies.
                disk_payload = deepcopy(payload)
                disk_payload["results"] = [{k: (v[:1000] if isinstance(v, str) else v)
                    for k, v in item.items() if k in {"title", "url", "content", "score", "published_at", "published_date"}}
                    for item in payload.get("results", [])[:5] if isinstance(item, dict)]
                self.store.cache_put(disk_key, redact_values(disk_payload), ttl)
            except Exception:
                pass  # A cache write failure must not strand in-flight waiters.
            finally:
                try:
                    self.store.release("query:" + disk_key, lease_owner)
                except Exception:
                    pass  # The lease expires; waiting callers still receive a result.
        with self._guard:
            self._cache[key] = (time.monotonic() + ttl, deepcopy(payload))
            self._cache.move_to_end(key)
            while len(self._cache) > 256:
                self._cache.popitem(last=False)
            self._inflight.pop(key, None)
            future.set_result(deepcopy(payload))
        return payload, False

    def _normalize(self, raw, retrieved_at):
        if not isinstance(raw, dict):
            return None
        title, content = raw.get("title"), raw.get("content", "")
        if not isinstance(title, str) or not title.strip() or len(title) > 1000:
            return None
        title = title.strip()
        content = content if isinstance(content, str) else ""
        url = safe_source_url(raw.get("url"))
        if not url:
            return None
        text = title + " " + content + " " + url
        if TOKEN_PATTERN.search(text) or any(key and key in text for key in
                                             (self.settings.tavily_key, self.settings.openai_key)):
            return None
        domain = (urlsplit(url).hostname or "").removeprefix("www.")
        quality, source = source_quality(domain)
        if quality == "unverified" and re.search(
                r"(?i)\b(?:price predictions?|price targets?|presale|guaranteed|buy now|\d+x|moon)\b|"
                r"\b(?:could|will|set to)\s+(?:hit|reach|explode)\b", title):
            return None
        score = raw.get("score")
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            score = None
        publication = publication_timestamp(raw.get("published_at")) or publication_timestamp(raw.get("published_date"))
        return EvidenceItem(id="news_" + hashlib.sha256(url.encode()).hexdigest()[:16],
                            headline=title, source=source, domain=domain, url=url,
                            published_at=publication, retrieved_at=retrieved_at,
                            summary=content.strip()[:500], relevance_score=score,
                            direction=headline_direction(title), category=evidence_category(title, content),
                            source_quality=quality, supports_model_direction=None, recency_verified=False)

    def run(self, question, mode="live", prediction_timestamp=None, fresh=False, trigger_types=None, queries=None):
        if mode not in ("live", "historical"):
            return ResearchResult(status="unavailable", reason="Unsupported research mode.")
        cutoff = prediction_cutoff(prediction_timestamp) if mode == "historical" else None
        if mode == "historical" and cutoff is None:
            return ResearchResult(status="unavailable", reason="Historical research requires a timezone-aware prediction timestamp.")
        queries = self.query_plan(question, mode, trigger_types) if queries is None else queries
        if not isinstance(queries, list) or not 1 <= len(queries) <= 3 or any(
                not isinstance(q, str) or not 1 <= len(q.strip()) <= 180 for q in queries):
            return ResearchResult(status="unavailable", reason="Research query budget or format is invalid.")
        payloads, cached_flags, evidence = [], [], []
        rejected = 0
        for query in queries:
            payload, cached = self._search(query, mode, cutoff, fresh=fresh)
            payloads.append(payload)
            cached_flags.append(cached)
            results = payload.get("results", [])
            if not isinstance(results, list):
                results = []
            for raw in results[:5]:
                item = self._normalize(raw, payload["retrieved_at"])
                if item is not None:
                    evidence.append(item)
                else:
                    rejected += 1
        evidence.sort(key=lambda item: ({"primary": 0, "established": 1, "unverified": 2}[item.source_quality],
                                       -(item.relevance_score or 0)))
        unique, seen_urls, seen_titles = [], set(), set()
        for item in evidence:
            parts = urlsplit(item.url)
            canonical = (parts.hostname.removeprefix("www."), parts.path.rstrip("/"),
                         tuple(sorted(parse_qsl(parts.query, keep_blank_values=True))))
            headline = re.sub(r"\W+", " ", item.headline.casefold()).strip()
            if canonical not in seen_urls and headline not in seen_titles:
                unique.append(item)
                seen_urls.add(canonical)
                seen_titles.add(headline)
        failures = sum(payload["status"] == "unavailable" for payload in payloads)
        warnings = []
        if failures:
            warnings.append(f"{failures} of {len(queries)} focused news searches were unavailable.")
        if rejected:
            warnings.append(f"{rejected} unsafe or unsuitable source records were omitted.")
        if mode == "historical":
            warnings.append("Historical search dates can reflect publication or update dates; exact publication cutoff still requires validation.")
        status = ("partial" if failures else "ok") if unique else ("unavailable" if failures else "empty")
        reason = None if unique else ("Focused news research is unavailable." if failures else "No usable news evidence was found.")
        return ResearchResult(status=status, evidence=unique, queries=queries,
                              retrieved_at=max(payload["retrieved_at"] for payload in payloads),
                              cached=all(cached_flags), reason=reason, warnings=warnings,
                              cached_query_count=sum(cached_flags),
                              returned_source_count=sum(len(p.get("results", [])[:5]) for p in payloads if isinstance(p.get("results"), list)),
                              provider_call_count=sum(bool(p.get("provider_called")) and not cached
                                  for p, cached in zip(payloads, cached_flags)))
