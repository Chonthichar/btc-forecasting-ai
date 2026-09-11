from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

SignalState = Literal["UP", "DOWN", "NO_SIGNAL", "UNKNOWN"]
Direction = Literal["bullish", "bearish", "neutral"]

class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

class MarketContext(Contract):
    price: float | None = None
    timestamp: str | None = None
    regime: str | None = None
    volatility_24h: float | None = None
    momentum_24h: float | None = None
    rsi_14: float | None = None

class SentimentContext(Contract):
    score: float | None = None
    label: str | None = None
    article_count: int | None = None
    hours_with_news: int | None = None
    timestamp: str | None = None

class ForecastView(Contract):
    horizon_hours: int
    raw_direction: Literal["UP", "DOWN", "NEUTRAL", "UNKNOWN"] = "UNKNOWN"
    probability_up: float | None = Field(None, ge=0, le=1)
    probability_down: float | None = Field(None, ge=0, le=1)
    raw_confidence: float | None = Field(None, ge=0, le=1)
    meta_reliability: float | None = Field(None, ge=0, le=1)
    validated_reliability: float | None = Field(None, ge=0, le=1)
    reliability: float | None = Field(None, ge=0, le=1)
    reliability_source: str | None = None
    selective_gate: bool | None = None
    signal_state: SignalState = "UNKNOWN"
    reliability_reason: str = "Validated reliability is unavailable."
    model: str | None = None
    feature_set: str | None = None
    model_version: str | None = None
    model_timestamp: str | None = None
    latest_candle: str | None = None
    reference_price: float | None = None
    regime: str | None = None

class ForecastContext(Contract):
    timestamp: str | None = None
    captured_at: str
    last_refresh: str | None = None
    data_fresh: bool | None = None
    market: MarketContext = Field(default_factory=MarketContext)
    sentiment: SentimentContext = Field(default_factory=SentimentContext)
    forecasts: dict[str, ForecastView] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

class EvidenceItem(Contract):
    id: str
    headline: str
    source: str
    domain: str
    url: str
    published_at: str | None = None
    retrieved_at: str
    summary: str
    relevance_score: float | None = None
    direction: Direction = "neutral"
    direction_basis: str = "Conservative headline classification; not a verified market impact."
    category: Literal["ETF", "MACRO", "FED", "REGULATION", "SECURITY", "EXCHANGE", "INSTITUTIONAL", "LIQUIDATIONS", "DERIVATIVES", "WHALE", "MARKET_STRUCTURE", "OTHER"] = "OTHER"
    supports_model_direction: bool | None = None
    source_quality: Literal["primary", "established", "unverified"] = "unverified"
    recency_verified: bool = False

class ResearchResult(Contract):
    status: Literal["not_requested", "ok", "partial", "unavailable", "empty"] = "not_requested"
    evidence: list[EvidenceItem] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    retrieved_at: str | None = None
    cached: bool = False
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    review: ReviewReport | None = None
    workflow: ResearchAudit | None = None
    market_decision: MarketDecision | None = None
    provider_call_count: int = 0
    cached_query_count: int = 0
    returned_source_count: int = 0

class ValidationResult(Contract):
    context_valid: bool = True
    forecast_usable: bool = False
    forecast_reason: str
    signal_states: dict[str, SignalState] = Field(default_factory=dict)
    reliability_threshold: float
    evidence_quality: Literal["none", "limited", "moderate", "strong"] = "none"
    bullish_evidence_count: int = 0
    bearish_evidence_count: int = 0
    neutral_evidence_count: int = 0
    mixed_evidence: bool = False
    causal_claim_allowed: bool = False
    rejected_evidence_count: int = 0
    warnings: list[str] = Field(default_factory=list)

class SafeChatMessage(Contract):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=6000)

class ChatRequest(Contract):
    fresh_research: bool = False
    message: str = Field(min_length=1, max_length=2000)
    history: list[SafeChatMessage] = Field(default_factory=list, max_length=12)
    request_id: str | None = Field(None, pattern=r"^[a-zA-Z0-9_-]{1,64}$")

class ResearchRequest(Contract):
    fresh_research: bool = False
    question: str = Field(default="Bitcoin latest market news today", min_length=1, max_length=2000)
    mode: Literal["live", "historical"] = "live"
    prediction_timestamp: datetime | None = None
    request_id: str | None = Field(None, pattern=r"^[a-zA-Z0-9_-]{1,64}$")

class AgentActivity(Contract):
    forecast: Literal["idle", "running", "done", "error"] = "idle"
    research: Literal["idle", "running", "done", "skipped", "unavailable", "error"] = "idle"
    validation: Literal["idle", "running", "done", "error"] = "idle"
    review: Literal["idle", "running", "done", "unavailable", "error"] = "idle"
    decision: Literal["idle", "running", "done", "unavailable", "error"] = "idle"

class ChatResponse(Contract):
    answer: str
    signal_state: SignalState
    context_timestamp: str | None = None
    research_used: bool = False
    research_status: str
    sources: list[EvidenceItem] = Field(default_factory=list)
    agent_status: AgentActivity
    validation: ValidationResult
    llm_status: Literal["ok", "unavailable", "invalid_output"]
    request_id: str
    market_decision: MarketDecision | None = None

class DecisionPlan(Contract):
    """A conversational answer with verifiable references to supplied facts."""
    focus_horizons: list[Literal["1h", "6h", "24h"]]
    fact_ids: list[str]
    evidence_ids: list[str]
    interpretation: Literal["mixed_context", "bullish_context", "bearish_context", "neutral_context", "model_only", "insufficient_evidence"]
    answer_style: Literal["concise", "analytical"]
    answer: str | None = Field(None, max_length=7000)
    market_decision: DecisionProposal | None = None


class ResearchPlan(Contract):
    evidence_ids: list[str] = Field(max_length=15)
    research_summary: str = Field(max_length=1200)


class EvidenceReview(Contract):
    evidence_id: str = Field(max_length=120)
    accept: bool
    stance: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    reason: str = Field(max_length=400)


class ReviewPlan(Contract):
    assessments: list[EvidenceReview] = Field(max_length=15)
    contradictions: list[str] = Field(max_length=10)
    unsupported_claims: list[str] = Field(max_length=10)
    review_summary: str = Field(max_length=1200)
    additional_research_needed: bool


class ReviewReport(Contract):
    status: Literal["ok", "unavailable", "invalid_output"]
    accepted_evidence: list[str] = Field(default_factory=list)
    rejected_evidence: list[str] = Field(default_factory=list)
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    mixed_evidence: bool = False
    evidence_quality: Literal["HIGH", "MEDIUM", "LOW", "NONE"] = "NONE"
    contradictions: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    review_summary: str = ""
    additional_research_needed: bool = False


class DecisionProposal(Contract):
    market_stance: Literal["BULLISH", "BEARISH", "MIXED", "NO_RELIABLE_VIEW"]
    decision_strength: Literal["LOW", "MODERATE", "HIGH"]
    time_horizon: Literal["1h", "6h", "24h", "6-24h", "mixed"]
    ml_view: Literal["BULLISH", "BEARISH", "MIXED", "NO_RELIABLE_SIGNAL"]
    research_view: Literal["BULLISH", "BEARISH", "NEUTRAL", "MIXED", "NO_EVIDENCE"]
    agreement: bool
    short_term_risk: str = Field(max_length=600)
    explanation: str = Field(max_length=7000)
    evidence_ids: list[str] = Field(max_length=15)


class MarketDecision(DecisionProposal):
    generated_at: str
    context_timestamp: str | None = None
    signal_states: dict[str, SignalState] = Field(default_factory=dict)
    reliability_enforced: bool = True


class ResearchAudit(Contract):
    event_id: str | None = None
    timestamp: str
    trigger_reason: list[str] = Field(default_factory=list)
    user_requested: bool = False
    mode: Literal["live", "historical"] = "live"
    tavily_called: bool = False
    cache_used: bool = False
    queries_used: list[str] = Field(default_factory=list)
    returned_sources: int = 0
    accepted_sources: int = 0
    rejected_sources: int = 0
    research_agent_executed: bool = False
    review_agent_executed: bool = False
    decision_agent_executed: bool = False
    final_market_stance: str | None = None
    decision_strength: str | None = None
    final_signal_states: dict[str, SignalState] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


ResearchResult.model_rebuild()
ChatResponse.model_rebuild()
DecisionPlan.model_rebuild()
