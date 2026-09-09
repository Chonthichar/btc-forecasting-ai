"""Network-free tests; fixture predictions are isolated from production history."""
from __future__ import annotations

import copy
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Also run directly from the artifact directory before the parent integrates files.
import app.agents
app.agents.__path__.insert(0, str(Path(__file__).resolve().parent))

from app.agents.forecast_context_agent import ForecastContextAgent
from app.agents.validation_agent import ValidationAgent
from app.schemas.agent_models import DecisionPlan, EvidenceItem, ResearchResult
from app.services.agent_config import AgentSettings

UTC = timezone.utc
NOW = datetime(2026, 9, 8, 12, 10, tzinfo=UTC)
SETTINGS = AgentSettings("", "", "", .5, 10, 72, 90, 12, 45)


def snapshot(p=.7, reliability=.8, **updates):
    forecast = {"horizon_hours": 1, "probability_up": p, "probability_down": 1 - p,
                "current_price": 65000.25, "latest_candle": "2026-09-08T11:00:00+00:00",
                "issued_at": "2026-09-08T12:02:00+00:00", "model": "vae_transformer_v2",
                "feature_set": "market_only", "model_version": "fixture-v1"}
    if reliability is not None:
        forecast["validated_reliability"] = reliability
    forecast.update(updates)
    return {"refreshed_at_utc": "2026-09-08T12:02:00+00:00",
            "market": {"price": 65000.25, "timestamp": "2026-09-08T11:00:00+00:00",
                       "momentum_24h": -.013, "volatility_24h": .002, "rsi_14": 41.2},
            "sentiment": {"latest_sentiment_score": -.12, "accepted_articles": 19,
                          "window_start": "2026-09-04T12:02:00+00:00"},
            "quality": {"latest_timestamp": "2026-09-08T11:00:00+00:00",
                        "hours_with_news_in_latest_sequence": 11}, "forecasts": {"1": forecast}}


def evidence(id="n1", **updates):
    value = {"id": id, "headline": "Bitcoin ETF reports new inflows", "source": "Reuters",
             "domain": "reuters.com", "url": "https://www.reuters.com/bitcoin-etf",
             "published_at": "2026-09-08T10:00:00+00:00", "retrieved_at": NOW.isoformat(),
             "summary": "Bitcoin ETF fund flows were reported.", "direction": "bullish",
             "category": "ETF", "source_quality": "established"}
    value.update(updates)
    return EvidenceItem(**value)


def plan(**updates):
    value = {"focus_horizons": ["1h"], "fact_ids": ["forecast.1h"], "evidence_ids": [],
             "interpretation": "model_only", "answer_style": "concise"}
    value.update(updates)
    return DecisionPlan(**value)


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.agent = ForecastContextAgent(SETTINGS)

    def view(self, **values):
        return self.agent.run(snapshot(**values), NOW).forecasts["1"]

    def test_reliable_low_up_probability_is_down(self):
        result = self.view(p=.14, reliability=.8)
        self.assertEqual((result.raw_direction, result.signal_state), ("DOWN", "DOWN"))
        self.assertEqual(result.reliability, .8)

    def test_low_reliability_vetoes_both_directions_and_zero(self):
        for p in (.1, .9, 0.):
            with self.subTest(p=p):
                self.assertEqual(self.view(p=p, reliability=.49).signal_state, "NO_SIGNAL")

    def test_missing_reliability_does_not_use_probability_confidence_or_holdout(self):
        result = self.view(p=.01, reliability=None, confidence=.99,
                           historical_holdout_metrics={"accuracy": .999})
        self.assertEqual(result.signal_state, "UNKNOWN")
        self.assertIsNone(result.reliability)
        self.assertIsNone(result.selective_gate)
        self.assertEqual(result.raw_confidence, .99)

    def test_explicit_reliability_sources_and_validation_flags(self):
        cases = [({"meta_reliability": .8}, "DOWN", "meta_reliability"),
                 ({"reliability": .8}, "UNKNOWN", None),
                 ({"reliability": .8, "reliability_validated": True}, "DOWN", "reliability (reliability_validated=true)"),
                 ({"confidence": .8}, "UNKNOWN", None),
                 ({"confidence": .8, "confidence_validated": True}, "DOWN", "confidence (confidence_validated=true)")]
        for updates, state, source in cases:
            raw = snapshot(p=.1, reliability=None)
            raw["forecasts"]["1"].update(updates)
            result = self.agent.run(raw, NOW).forecasts["1"]
            self.assertEqual((result.signal_state, result.reliability_source), (state, source))

    def test_false_gate_and_upstream_veto_cannot_be_promoted(self):
        for updates in ({"selective_gate": False}, {"gate_passed": False},
                        {"selective_gate_passed": False}, {"signal_state": "NO_SIGNAL"},
                        {"selective_gate": True, "gate_passed": False}):
            self.assertEqual(self.view(**updates).signal_state, "NO_SIGNAL")

    def test_probability_zero_and_one_are_values_and_tie_is_neutral(self):
        self.assertEqual(self.view(p=0).signal_state, "DOWN")
        self.assertEqual(self.view(p=1).signal_state, "UP")
        self.assertEqual(self.view(p=.5).raw_direction, "NEUTRAL")
        self.assertEqual(self.view(p=.5).signal_state, "NO_SIGNAL")
        self.assertEqual(self.view(p=.6, reliability=.5).signal_state, "UP")

    def test_stale_candle_stale_issue_future_issue_and_preclose_issue(self):
        for updates in ({"latest_candle": "2026-09-08T09:00:00+00:00"},
                        {"issued_at": "2026-09-08T10:00:00+00:00"},
                        {"issued_at": "2026-09-08T13:00:00+00:00"},
                        {"issued_at": "2026-09-08T11:59:59+00:00"},
                        {"issued_at": "not a date"}):
            self.assertEqual(self.view(**updates).signal_state, "UNKNOWN")

    def test_stale_market_context_also_blocks_fresh_forecast(self):
        raw = snapshot()
        raw["market"]["timestamp"] = "2026-09-08T08:00:00+00:00"
        context = self.agent.run(raw, NOW)
        self.assertFalse(context.data_fresh)
        self.assertEqual(context.forecasts["1"].signal_state, "UNKNOWN")

    def test_missing_unknowns_are_not_synthesized_and_input_is_unchanged(self):
        raw = snapshot(reliability=None)
        original = copy.deepcopy(raw)
        result = self.agent.run(raw, NOW)
        self.assertEqual(raw, original)
        self.assertEqual(result.market.price, raw["market"]["price"])
        self.assertEqual(result.sentiment.score, -.12)
        self.assertEqual(result.sentiment.hours_with_news, 11)
        self.assertIsNone(result.market.regime)
        self.assertIsNone(result.sentiment.label)
        self.assertIsNone(result.sentiment.timestamp)
        self.assertIsNone(result.forecasts["1"].raw_confidence)
        self.assertEqual(self.agent.run({}, NOW).forecasts, {})

    def test_explicit_refresh_time_fallback_and_invalid_values(self):
        raw = snapshot()
        del raw["forecasts"]["1"]["issued_at"]
        self.assertEqual(self.agent.run(raw, NOW).forecasts["1"].model_timestamp, raw["refreshed_at_utc"])
        raw["forecasts"]["1"]["probability_up"] = float("nan")
        self.assertEqual(self.agent.run(raw, NOW).forecasts["1"].signal_state, "UNKNOWN")
        with self.assertRaises(ValueError):
            self.agent.run(raw, NOW.replace(tzinfo=None))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.agent = ValidationAgent(SETTINGS)
        self.raw = snapshot(p=.1)
        self.context = ForecastContextAgent(SETTINGS).run(self.raw, NOW)

    def validate(self, items=None, **kwargs):
        return self.agent.validate(self.context, self.raw,
                                   ResearchResult(status="ok", evidence=items or []), now=NOW, **kwargs)

    def test_modified_price_or_signal_is_rejected_and_recomputed(self):
        for mutation in (lambda c: setattr(c.market, "price", 1.),
                         lambda c: setattr(c.forecasts["1"], "signal_state", "UP")):
            self.context = ForecastContextAgent(SETTINGS).run(self.raw, NOW)
            mutation(self.context)
            checked, _, result = self.validate()
            self.assertFalse(result.context_valid)
            self.assertFalse(result.forecast_usable)
            self.assertEqual(checked.market.price, 65000.25)
            self.assertEqual(checked.forecasts["1"].signal_state, "UNKNOWN")

    def test_validation_time_rechecks_freshness_without_false_tamper(self):
        checked, _, result = self.agent.validate(self.context, self.raw, ResearchResult(), now=NOW + timedelta(hours=2))
        self.assertTrue(result.context_valid)
        self.assertFalse(result.forecast_usable)
        self.assertEqual(checked.forecasts["1"].signal_state, "UNKNOWN")

    def test_news_never_promotes_a_gated_or_unknown_forecast(self):
        for reliability in (None, .1):
            self.raw = snapshot(p=.9, reliability=reliability)
            self.context = ForecastContextAgent(SETTINGS).run(self.raw, NOW)
            checked, cleaned, result = self.validate([evidence()])
            self.assertEqual(len(cleaned.evidence), 1)
            self.assertFalse(result.forecast_usable)
            self.assertIn(checked.forecasts["1"].signal_state, ("UNKNOWN", "NO_SIGNAL"))
            self.assertFalse(result.causal_claim_allowed)

    def test_old_future_irrelevant_macro_and_unsafe_sources_are_rejected(self):
        items = [evidence("old", published_at=(NOW - timedelta(hours=73)).isoformat()),
                 evidence("future", published_at=(NOW + timedelta(seconds=1)).isoformat()),
                 evidence("macro", headline="Fed keeps interest rates unchanged", summary="Broad economic outlook.", category="FED"),
                 evidence("unsafe", url="javascript:alert(1)"),
                 evidence("private", url="http://127.0.0.1/admin"),
                 evidence("credentials", url="https://user:password@reuters.com/story"),
                 evidence("slash", url="https://reuters.com\\@evil.com/news")]
        _, cleaned, result = self.validate(items)
        self.assertEqual(cleaned.evidence, [])
        self.assertEqual(result.rejected_evidence_count, len(items))

    def test_canonical_url_and_headline_deduplication(self):
        items = [evidence("a", url="https://www.reuters.com/story/?utm_source=mail#fragment"),
                 evidence("b", url="http://reuters.com/story", headline="Another Bitcoin ETF headline"),
                 evidence("c", url="https://coindesk.com/different", domain="coindesk.com")]
        _, cleaned, result = self.validate(items)
        self.assertEqual(len(cleaned.evidence), 1)
        self.assertEqual(result.rejected_evidence_count, 2)
        self.assertNotIn("utm_", cleaned.evidence[0].url)
        self.assertNotIn("#", cleaned.evidence[0].url)

    def test_missing_live_date_retained_unverified_and_downranked(self):
        dated = evidence("dated")
        undated = evidence("undated", headline="Bitcoin derivatives activity increases", published_at=None,
                           url="https://www.reuters.com/derivatives")
        _, cleaned, result = self.validate([undated, dated])
        self.assertEqual([e.id for e in cleaned.evidence], ["dated", "undated"])
        self.assertIsNone(cleaned.evidence[1].published_at)
        self.assertFalse(cleaned.evidence[1].recency_verified)
        self.assertEqual(cleaned.evidence[1].source_quality, "unverified")
        self.assertEqual(result.rejected_evidence_count, 0)

    def test_unknown_and_impersonated_domains_cannot_claim_primary_quality(self):
        spam = evidence("spam", headline="Bitcoin promotional claims", domain="reuters.com", source_quality="primary",
                        url="https://bitcoin-spam.example.org/promotion")
        _, cleaned, result = self.validate([spam])
        self.assertEqual(cleaned.evidence[0].source_quality, "unverified")
        self.assertEqual(cleaned.evidence[0].source, "bitcoin-spam.example.org")
        self.assertEqual(result.evidence_quality, "limited")

    def test_upstream_partial_coverage_is_preserved_after_validation(self):
        research = ResearchResult(status="partial", evidence=[evidence()], reason="A provider query timed out.")
        _, cleaned, result = self.agent.validate(self.context, self.raw, research, now=NOW)
        self.assertEqual(cleaned.status, "partial")
        self.assertEqual(cleaned.reason, research.reason)
        self.assertEqual(result.rejected_evidence_count, 0)
        self.assertTrue(cleaned.evidence[0].recency_verified)

    def test_publisher_quality_matches_research_union(self):
        primary = ("federalreserve.gov", "sec.gov", "treasury.gov", "bls.gov", "bea.gov", "cftc.gov",
                   "ecb.europa.eu", "imf.org", "blackrock.com", "ishares.com", "fidelity.com", "coinbase.com",
                   "kraken.com", "cmegroup.com", "binance.com")
        established = ("reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com", "apnews.com",
                       "coindesk.com", "theblock.co", "cointelegraph.com", "decrypt.co")
        for quality, domains in (("primary", primary), ("established", established)):
            for domain in domains:
                with self.subTest(domain=domain):
                    item = evidence(domain=domain, url="https://" + domain + "/bitcoin-report", source_quality=quality)
                    _, cleaned, _ = self.validate([item])
                    self.assertEqual(cleaned.evidence[0].source_quality, quality)

    def test_historical_cutoff_excludes_future_and_missing_publication(self):
        cutoff = NOW - timedelta(hours=1)
        items = [evidence("before", published_at=(cutoff - timedelta(minutes=1)).isoformat()),
                 evidence("after", headline="Bitcoin futures news after forecast", url="https://reuters.com/after",
                          published_at=(cutoff + timedelta(seconds=1)).isoformat()),
                 evidence("missing", headline="Undated Bitcoin story", url="https://reuters.com/missing", published_at=None)]
        _, cleaned, result = self.validate(items, mode="historical", prediction_timestamp=cutoff)
        self.assertEqual([item.id for item in cleaned.evidence], ["before"])
        self.assertEqual(result.rejected_evidence_count, 2)
        with self.assertRaises(ValueError):
            self.validate(items, mode="historical")
        with self.assertRaises(ValueError):
            self.validate(items, mode="historical", prediction_timestamp=cutoff.replace(tzinfo=None))

    def test_mixed_counts_and_plan_preserve_both_sides(self):
        bearish = evidence("bear", headline="Bitcoin ETF reports outflows", url="https://reuters.com/outflows", direction="bearish")
        _, research, validation = self.validate([evidence("bull"), bearish])
        self.assertTrue(validation.mixed_evidence)
        self.assertEqual((validation.bullish_evidence_count, validation.bearish_evidence_count), (1, 1))
        facts = {"forecast.1h": "The validated internal model signal is DOWN."}
        valid = plan(evidence_ids=["bull", "bear"], interpretation="mixed_context")
        self.assertEqual(self.agent.validate_plan(valid, facts, research, validation), valid)
        for invalid in (plan(evidence_ids=["bull"], interpretation="bullish_context"),
                        plan(evidence_ids=["bull"], interpretation="mixed_context")):
            with self.assertRaises(ValueError):
                self.agent.validate_plan(invalid, facts, research, validation)

    def test_plan_unknown_ids_wrong_interpretation_and_causal_fields_rejected(self):
        _, research, validation = self.validate([evidence()])
        facts = {"forecast.1h": "The validated internal model signal is DOWN."}
        for invalid in (plan(fact_ids=["invented_price"]),
                        plan(evidence_ids=["made_up"], interpretation="bullish_context"),
                        plan(evidence_ids=["n1"], interpretation="bearish_context"),
                        plan(fact_ids=["because Bitcoin will rise"])):
            with self.assertRaises(ValueError):
                self.agent.validate_plan(invalid, facts, research, validation)
        with self.assertRaises(ValueError):
            DecisionPlan(**plan().model_dump(), because="The model rose because of the Fed")
        bypass = DecisionPlan.model_construct(**plan().model_dump(), causal_explanation="Invented cause")
        # model_construct can ignore extras; it never creates a prose channel in the contract.
        self.assertNotIn("causal_explanation", bypass.model_dump())


if __name__ == "__main__":
    unittest.main(verbosity=2)
