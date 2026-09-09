"""Deterministic guards for internal facts, dated research, and ID-only plans."""
from __future__ import annotations

import ipaddress
import re
from datetime import timedelta
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..schemas.agent_models import DecisionPlan, ForecastContext, ResearchResult, ValidationResult
from .forecast_context_agent import ForecastContextAgent, aware_time, current_time

_BTC = re.compile(r"\b(?:bitcoin|btc)\b", re.I)
_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")
_PRIMARY = {"sec.gov", "cftc.gov", "federalreserve.gov", "treasury.gov", "cmegroup.com",
            "blackrock.com", "ishares.com", "fidelity.com", "coinbase.com", "binance.com",
            "bls.gov", "bea.gov", "ecb.europa.eu", "imf.org", "kraken.com"}
_ESTABLISHED = {"reuters.com", "apnews.com", "ft.com", "wsj.com", "bloomberg.com",
                "coindesk.com", "cointelegraph.com", "theblock.co", "decrypt.co", "cnbc.com"}
_QUALITY = {"unverified": 0, "established": 1, "primary": 2}
_TRACKING = {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid"}


def _in_domains(domain, domains):
    return any(domain == allowed or domain.endswith("." + allowed) for allowed in domains)


def canonical_url(value):
    """Accept public HTTP(S) links, remove tracking, and return actual host identity."""
    if not isinstance(value, str) or not value or len(value) > 4000:
        raise ValueError("Missing or oversized source URL")
    if re.search(r"[\x00-\x20\x7f\\]", value):
        raise ValueError("Unsafe source URL characters")
    parts = urlsplit(value)
    if parts.scheme.lower() not in {"https", "http"} or parts.username or parts.password:
        raise ValueError("Source URL must be a public HTTP(S) URL without credentials")
    domain = (parts.hostname or "").lower().rstrip(".")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("Invalid URL port") from exc
    if not domain or port not in (None, 80, 443):
        raise ValueError("Source URL has an invalid host or nonstandard port")
    try:
        address = ipaddress.ip_address(domain)
    except ValueError:
        if ("." not in domain or not re.fullmatch(r"[a-z0-9.-]+", domain)
                or any(domain == suffix or domain.endswith("." + suffix)
                       for suffix in ("localhost", "local", "internal", "invalid", "test", "onion"))
                or any(not label or label.startswith("-") or label.endswith("-") for label in domain.split("."))):
            raise ValueError("Source URL does not name a public domain")
    else:
        if not address.is_global:
            raise ValueError("Source URL refers to a nonpublic IP address")
    host = "[" + domain + "]" if ":" in domain else domain
    if port is not None and not (parts.scheme == "https" and port == 443 or parts.scheme == "http" and port == 80):
        host += ":" + str(port)
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                             if not k.lower().startswith("utm_") and k.lower() not in _TRACKING))
    path = parts.path.rstrip("/") or "/"
    canonical = urlunsplit((parts.scheme.lower(), host, path, query, ""))
    identity = (domain.removeprefix("www."), path, query)
    return canonical, identity, domain


class ValidationAgent:
    def __init__(self, settings):
        self.settings = settings
        self.context_agent = ForecastContextAgent(settings)

    def _research(self, research, now, cutoff, historical):
        # Revalidation also catches model_construct() bypasses and mutation after parsing.
        research = ResearchResult.model_validate(research.model_dump())
        accepted, warnings, rejected = [], list(research.warnings), 0
        for original in research.evidence:
            item = original.model_copy(deep=True)
            try:
                if not _ID.fullmatch(item.id) or not item.headline.strip():
                    raise ValueError("missing headline or invalid evidence ID")
                url, identity, domain = canonical_url(item.url)
                if not _BTC.search(item.headline + " " + item.summary):
                    raise ValueError("BTC relevance is not explicit in the article text")
                retrieved = aware_time(item.retrieved_at)
                if retrieved is None or retrieved > now:
                    raise ValueError("retrieval timestamp is missing, invalid, or in the future")
                published = aware_time(item.published_at)
                if published is None and historical:
                    raise ValueError("historical evidence requires an explicit publication timestamp")
                if published is not None:
                    if published > cutoff:
                        raise ValueError("publication is after the allowed cutoff")
                    if cutoff - published > timedelta(hours=self.settings.news_max_age_hours):
                        raise ValueError("publication is older than the configured news window")
                    if published > retrieved:
                        raise ValueError("publication is later than the recorded retrieval")
                declared_domain = item.domain.lower().removeprefix("www.").rstrip(".")
                matching_domain = declared_domain == domain.removeprefix("www.")
                allowed_quality = ("primary" if _in_domains(domain, _PRIMARY) else
                                   "established" if _in_domains(domain, _ESTABLISHED) else "unverified")
                quality = min((item.source_quality, allowed_quality), key=_QUALITY.get)
                if not matching_domain or published is None:
                    quality = "unverified"
                item.url, item.domain = url, domain
                if not matching_domain:
                    # Never retain an impersonated publisher label from mismatched metadata.
                    item.source = domain
                    warnings.append(f"{item.id}: publisher metadata did not match its URL; source is unverified.")
                item.source_quality = quality
                item.recency_verified = published is not None
                if published is None:
                    item.published_at = None
                    warnings.append(f"{item.id}: publication date is unverified; retained as lower-priority live context.")
                item.supports_model_direction = None
                item.direction_basis = "Provided article direction metadata; contextual evidence, not model causation."
                headline_key = " ".join(re.findall(r"\w+", item.headline.casefold()))
                accepted.append((item, identity, headline_key, published))
            except ValueError as exc:
                rejected += 1
                warnings.append(f"{original.id}: rejected ({exc}).")
        accepted.sort(key=lambda row: (-_QUALITY[row[0].source_quality], not row[0].recency_verified,
                                      -(row[3].timestamp() if row[3] else 0), row[0].id))
        evidence, urls, headlines, ids = [], set(), set(), set()
        for item, identity, headline, _ in accepted:
            if identity in urls or headline in headlines or item.id in ids:
                rejected += 1
                warnings.append(f"{item.id}: duplicate URL, headline, or evidence ID was removed.")
                continue
            evidence.append(item)
            urls.add(identity)
            headlines.add(headline)
            ids.add(item.id)
        status = research.status
        if evidence:
            status = "partial" if (research.status == "partial" or rejected or any(
                not e.recency_verified or e.source_quality == "unverified" for e in evidence)) else "ok"
        elif research.evidence:
            status = "empty"
        cleaned = research.model_copy(update={"evidence": evidence, "status": status,
                                               "warnings": list(dict.fromkeys(warnings))}, deep=True)
        return cleaned, rejected

    def validate(self, context: ForecastContext, snapshot: dict, research: ResearchResult,
                 now=None, mode="live", prediction_timestamp=None):
        now = current_time(now)
        if mode not in {"live", "historical"}:
            raise ValueError("mode must be live or historical")
        cutoff = now
        if mode == "historical":
            cutoff = aware_time(prediction_timestamp)
            if cutoff is None or cutoff > now:
                raise ValueError("Historical mode requires a timezone-aware prediction timestamp no later than now")
        context_valid = False
        try:
            context = ForecastContext.model_validate(context.model_dump())
            captured = aware_time(context.captured_at)
            if captured is not None and captured <= now:
                expected = self.context_agent.run(snapshot, now=captured)
                context_valid = context.model_dump() == expected.model_dump()
        except (ValueError, TypeError, AttributeError):
            pass
        # Source facts are always regenerated; stale context cannot retain an earlier UP/DOWN state.
        checked = self.context_agent.run(snapshot, now=cutoff)
        warnings = list(checked.warnings)
        if not context_valid:
            warning = "Caller context differs from the internal snapshot and was rejected."
            warnings.append(warning)
            for forecast in checked.forecasts.values():
                forecast.signal_state = "UNKNOWN"
                forecast.reliability_reason = warning
        cleaned, rejected = self._research(research, now, cutoff, mode == "historical")
        counts = {direction: sum(item.direction == direction for item in cleaned.evidence)
                  for direction in ("bullish", "bearish", "neutral")}
        trusted = [item for item in cleaned.evidence if item.recency_verified and item.source_quality != "unverified"]
        independent_domains = {item.domain.removeprefix("www.") for item in trusted}
        quality = ("none" if not cleaned.evidence else "limited" if not trusted else
                   "strong" if len(independent_domains) >= 3 else "moderate")
        states = {key: value.signal_state for key, value in checked.forecasts.items()}
        usable = context_valid and any(state in {"UP", "DOWN"} for state in states.values())
        reason = ("At least one fresh internal forecast meets the validated reliability threshold."
                  if usable else " ".join(dict.fromkeys(f.reliability_reason for f in checked.forecasts.values()))
                  or "No saved internal forecasts are available.")
        checked.warnings = list(dict.fromkeys(warnings))
        result = ValidationResult(
            context_valid=context_valid, forecast_usable=usable, forecast_reason=reason,
            signal_states=states, reliability_threshold=self.settings.reliability_threshold,
            evidence_quality=quality, bullish_evidence_count=counts["bullish"],
            bearish_evidence_count=counts["bearish"], neutral_evidence_count=counts["neutral"],
            mixed_evidence=bool(counts["bullish"] and counts["bearish"]),
            causal_claim_allowed=False, rejected_evidence_count=rejected,
            warnings=list(dict.fromkeys(warnings + cleaned.warnings)),
        )
        return checked, cleaned, result

    def validate_plan(self, plan: DecisionPlan, facts: dict[str, str], research: ResearchResult,
                      validation: ValidationResult) -> DecisionPlan:
        plan = DecisionPlan.model_validate(plan.model_dump())
        if validation.causal_claim_allowed:
            raise ValueError("Model causal attribution is never permitted")
        evidence = {item.id: item for item in research.evidence}
        if len(evidence) != len(research.evidence):
            raise ValueError("Evidence IDs must be unique")
        for selected, allowed in ((plan.fact_ids, facts), (plan.evidence_ids, evidence)):
            if len(selected) != len(set(selected)) or any(not _ID.fullmatch(key) or key not in allowed for key in selected):
                raise ValueError("The plan contains duplicate or unsupported fact/evidence IDs")
        counts = {direction: sum(item.direction == direction for item in research.evidence)
                  for direction in ("bullish", "bearish", "neutral")}
        if (counts["bullish"] != validation.bullish_evidence_count
                or counts["bearish"] != validation.bearish_evidence_count
                or counts["neutral"] != validation.neutral_evidence_count
                or bool(counts["bullish"] and counts["bearish"]) != validation.mixed_evidence):
            raise ValueError("Evidence counts do not match validation")
        selected_directions = {evidence[key].direction for key in plan.evidence_ids}
        interpretation = plan.interpretation
        if validation.mixed_evidence:
            if interpretation != "mixed_context" or not {"bullish", "bearish"} <= selected_directions:
                raise ValueError("Mixed research must retain both bullish and bearish evidence")
        elif interpretation == "mixed_context":
            raise ValueError("Mixed interpretation requires bullish and bearish evidence")
        elif interpretation in {"bullish_context", "bearish_context"}:
            direction = interpretation.removesuffix("_context")
            opposite = "bearish" if direction == "bullish" else "bullish"
            if not counts[direction] or counts[opposite] or direction not in selected_directions:
                raise ValueError("Directional interpretation is unsupported by the evidence")
        elif interpretation == "neutral_context":
            if counts["bullish"] or counts["bearish"] or "neutral" not in selected_directions:
                raise ValueError("Neutral interpretation is incompatible with evidence counts")
        elif interpretation == "model_only":
            if plan.evidence_ids or (not plan.fact_ids and not plan.answer):
                raise ValueError("Model-only plans must select internal facts without research evidence")
        elif interpretation == "insufficient_evidence":
            if plan.evidence_ids and selected_directions - {"neutral"}:
                raise ValueError("Insufficient-evidence interpretation cannot present directional evidence")
        return plan
