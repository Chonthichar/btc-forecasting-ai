"""Question-specific conversational answers grounded in application facts."""
from __future__ import annotations
import re
from app.schemas.agent_models import DecisionPlan
from app.services.agent_config import redact
from app.services.llm_service import LLMService
from .answer_guard import render_grounded_answer

def horizons_for(question):
    matches = re.findall(r"\b(1|6|24)\s*(?:h(?:ou?r?s?)?|hour[s]?)\b", question.lower())
    return list(dict.fromkeys(h+"h" for h in matches)) or ["1h", "6h", "24h"]

def reliability_text(view):
    if view.signal_state in {"NO_SIGNAL", "UNKNOWN"}:
        return f"{view.horizon_hours}h: No reliable forecast right now. {view.reliability_reason}"
    score = f"Validated reliability: {view.reliability:.1%}. " if view.reliability is not None else ""
    return f"{view.horizon_hours}h: {score}{view.reliability_reason} This is a qualified signal, not a guarantee."

def build_facts(context):
    facts = {}
    market = context.market
    if market.price is not None:
        facts["market.price"] = f"The latest recorded BTC closing price is ${market.price:,.2f} (market timestamp: {market.timestamp or 'unavailable'})."
    facts["market.freshness"] = "The internal market snapshot is fresh." if context.data_fresh else "The internal snapshot is missing or stale; current conditions are not confirmed."
    facts["market.regime"] = f"Market regime: {market.regime}." if market.regime else "The current pipeline does not provide a market-regime label."
    for field, label in (("volatility_24h", "24h volatility"), ("momentum_24h", "24h momentum"), ("rsi_14", "RSI (14)")):
        value = getattr(market, field)
        if value is not None:
            facts["market."+field] = f"{label}: {value:.6g}."
    sentiment = context.sentiment
    facts["sentiment.score"] = f"Internal news sentiment score: {sentiment.score:.6g}." if sentiment.score is not None else "The internal sentiment score is unavailable."
    facts["sentiment.label"] = f"Internal sentiment label: {sentiment.label}." if sentiment.label else "The current snapshot does not provide a sentiment label."
    facts["concept.probability_up"] = "P(UP) estimates the chance that the target closing price will exceed the saved reference close. It describes predicted direction, not validated reliability."
    facts["concept.reliability"] = "Validated reliability requires a separately evaluated reliability or calibration measure. A model's directional probability alone does not supply that measure."
    if context.forecasts and all(view.reliability is None for view in context.forecasts.values()):
        facts["model.reliability_status"] = "The current model snapshots do not include a validated reliability score. The app therefore cannot qualify their raw directional predictions as reliable signals."
    if sentiment.article_count is not None:
        facts["sentiment.count"] = f"The internal collection contains {sentiment.article_count} accepted articles."
    for key, view in context.forecasts.items():
        key = str(view.horizon_hours) + "h"
        facts[f"{key}.model"] = f"The {key} forecast uses {view.model or 'an unavailable model'} with feature set {view.feature_set or 'unavailable'}."
        if view.probability_up is not None:
            lean = "has no directional preference" if view.raw_direction == "NEUTRAL" else f"leans {view.raw_direction}"
            facts[f"{key}.raw"] = f"The raw {key} model {lean}; P(UP) = {view.probability_up:.1%}."
        else:
            facts[f"{key}.raw"] = f"The raw {key} forecast is unavailable."
        facts[f"{key}.reliability"] = reliability_text(view)
        if view.model_timestamp:
            facts[f"{key}.timestamp"] = f"{key} forecast recorded at {view.model_timestamp}."
        if view.raw_confidence is not None:
            facts[f"{key}.confidence"] = f"{key} reported raw confidence: {view.raw_confidence:.1%}; this is distinct from validated reliability."
    return facts

class DecisionAgent:
    def __init__(self, settings, validator, llm=None):
        self.settings, self.validator = settings, validator
        self.llm = llm or LLMService(settings)

    def run(self, question, history, context, research, validation, structured=False):
        from app.services.market_decision import enforce_decision
        plan = None
        def finish(answer, status, sources):
            if not structured:
                return answer, status, sources
            decision = enforce_decision(context, research, validation, horizons_for(question),
                proposal=plan.market_decision if plan else None, explanation=answer)
            return answer, status, sources, decision
        facts = build_facts(context)
        facts["app.llm"] = (f"The Decision Agent is configured to use OpenAI {self.settings.openai_model}."
                            if self.settings.openai_model else "No OpenAI model is configured.")
        facts["app.horizons"] = "The dashboard covers the 1h, 6h, and 24h forecast horizons."
        requested = horizons_for(question)
        fallback_ids = []
        if "sentiment" in question.lower():
            fallback_ids += [k for k in facts if k.startswith("sentiment.")]
        elif re.search(r"\bmodel\b", question.lower()) and not re.search(r"why|explain|predict", question.lower()):
            fallback_ids += [f"{h}.model" for h in requested]
        else:
            fallback_ids += [k for k in ("market.price", "market.freshness") if k in facts]
            fallback_ids += [f"{h}.raw" for h in requested]
        interpretation = ("mixed_context" if validation.mixed_evidence else
            "bullish_context" if validation.bullish_evidence_count else
            "bearish_context" if validation.bearish_evidence_count else
            "neutral_context" if research.evidence else "model_only")
        fallback = DecisionPlan(focus_horizons=requested, fact_ids=[k for k in fallback_ids if k in facts],
            evidence_ids=[e.id for e in research.evidence[:6]], interpretation=interpretation, answer_style="analytical")
        plan, llm_status, message = self.llm.plan(question, history, context, research, validation, facts)
        if plan is not None:
            try:
                plan = self.validator.validate_plan(plan, facts, research, validation)
                if plan.answer is not None:
                    answer, used_facts, sources = render_grounded_answer(plan, facts, research, validation)
                    focused = [context.forecasts[h[:-1]] for h in requested if h[:-1] in context.forecasts]
                    discusses_forecast = (any(re.match(r"(?:1|6|24)h\.|model\.reliability_status", key) for key in used_facts)
                        or bool(re.search(r"forecast|predict|signal|compare|why.*(?:unknown|unavailable)", question, re.I)))
                    if discusses_forecast:
                        remaining = focused
                        if remaining:
                            reasons = {view.reliability_reason for view in remaining}
                            if len(remaining) > 1 and len(reasons) == 1 and all(view.signal_state in {"UNKNOWN", "NO_SIGNAL"} for view in remaining):
                                answer += "\n\nNo reliable forecast right now for " + ", ".join(str(view.horizon_hours)+"h" for view in remaining) + ". " + remaining[0].reliability_reason
                            else:
                                answer += "\n\n" + "\n".join(reliability_text(view) for view in remaining)
                        elif not focused:
                            answer += "\n\nNo reliable forecast right now. Saved forecasts are unavailable."
                    if research.status == "unavailable":
                        answer += "\n\nLive web research is currently unavailable."
                    if sources:
                        if research.status == "partial":
                            answer += "\n\nResearch coverage or source verification is incomplete."
                        if any(not item.recency_verified for item in sources):
                            answer += " Some source publication times are unverified."
                        answer += "\n\nExternal news is context, not a verified cause of the model output or an increase in its reliability."
                    return finish(redact(answer), llm_status, sources)
            except (ValueError, TypeError):
                plan, llm_status, message = None, "invalid_output", None
        if re.fullmatch(r"\s*(?:hi|hello|hey|thanks|thank you)[!.\s]*", question, re.I):
            return finish(redact((message + "\n\n" if message else "") + "Hi! I can help explain your BTC forecasts, model reliability, and market news. What would you like to know?"), llm_status, [])
        if re.search(r"what (?:does|is).*(?:probability|reliability)|explain that|more simply|that difference", question, re.I):
            explanation = facts["concept.probability_up"] + "\n\n" + facts["concept.reliability"]
            return finish(redact((message + "\n\n" if message else "") + explanation), llm_status, [])
        if "model.reliability_status" in facts and re.search(r"why.*reliab|reliab.*(?:unknown|missing|unavailable)", question, re.I):
            explanation = facts["model.reliability_status"] + "\n\n" + facts["concept.reliability"]
            return finish(redact((message + "\n\n" if message else "") + explanation + "\n\nNo reliable forecast right now."), llm_status, [])
        if research.status != "not_requested" and not re.search(r"model|forecast|predict|signal|compare", question, re.I):
            selected = list(research.evidence[:6])
            if validation.mixed_evidence:
                for direction in ("bullish", "bearish"):
                    if not any(item.direction == direction for item in selected):
                        item = next((item for item in research.evidence if item.direction == direction), None)
                        if item:
                            selected.append(item)
            news = "\n".join(f"- [{item.direction.upper()}] {item.source}: {item.headline}" for item in selected)
            if not news:
                news = "Live web research is currently unavailable. No useful current sources passed validation."
            else:
                news += "\n\nThese are external news reports, not a verified cause of the model's prediction."
            return finish(redact((message + "\n\n" if message else "") + news), llm_status, selected)
        plan = plan or fallback
        # User-requested horizons and mandatory reliability notices cannot be
        # omitted or changed by the LLM's selection.
        focused = [context.forecasts[h[:-1]] for h in requested if h[:-1] in context.forecasts]
        model_facts = [facts[k] for k in plan.fact_ids if k in facts and not k.endswith(".reliability")]
        if not model_facts:
            model_facts = [facts[k] for k in fallback.fact_ids]
        sections = []
        if message:
            sections.append(message)
        sections.append("\n".join(dict.fromkeys(model_facts)))
        sections.append(("\n".join(reliability_text(v) for v in focused)
            or "No reliable forecast right now. Saved forecasts for the requested horizons are unavailable."))
        selected = [e for e in research.evidence if e.id in plan.evidence_ids]
        if not selected and research.evidence:
            selected = research.evidence[:6]
        if validation.mixed_evidence:
            for direction in ("bullish", "bearish"):
                if not any(e.direction == direction for e in selected):
                    match = next((e for e in research.evidence if e.direction == direction), None)
                    if match:
                        selected.append(match)
        if research.status == "unavailable":
            market_text = "Live web research is currently unavailable."
        elif research.status == "not_requested":
            market_text = "Web research was not needed for this question; this answer uses internal application context."
        elif not selected:
            market_text = "Live web research is currently unavailable. No sufficiently relevant current sources passed validation."
        else:
            market_text = "\n".join(f"- [{e.direction.upper()}] {e.source}: {e.headline}" for e in selected)
            if any(not e.recency_verified for e in selected):
                market_text += "\nSome sources have no verified publication time; their freshness is uncertain."
            if research.status == "partial":
                market_text += "\nResearch coverage or source verification is incomplete."
        if research.status != "not_requested":
            sections.append(market_text)
        # These conclusions are deterministic functions of validated evidence,
        # not arbitrary generated claims or explanations of model causality.
        if validation.mixed_evidence:
            interpretation_text = "External evidence is mixed: both bullish and bearish developments are represented."
        elif validation.bullish_evidence_count:
            interpretation_text = "The retrieved evidence includes bullish developments; this is limited to the sources found."
        elif validation.bearish_evidence_count:
            interpretation_text = "The retrieved evidence includes bearish developments; this is limited to the sources found."
        else:
            interpretation_text = "There is insufficient validated external evidence for a directional market interpretation."
        interpretation_text += " External news provides context; it does not establish the cause of a model prediction or increase its validated reliability."
        if research.evidence:
            sections.append(interpretation_text)
        usable = [v for v in focused if v.signal_state in {"UP", "DOWN"}]
        if not usable:
            conclusion = "No reliable forecast right now. The model does not currently have a reliable directional view."
        else:
            conclusion = "Qualified model signals: " + "; ".join(f"{v.horizon_hours}h {v.signal_state}" for v in usable) + ". These remain uncertain."
            missing = [str(v.horizon_hours)+"h" for v in focused if v.signal_state not in {"UP", "DOWN"}]
            if missing:
                conclusion += " No reliable forecast for " + ", ".join(missing) + "."
        if usable:
            sections.append(conclusion)
        if selected:
            sections.append("Sources:\n" + "\n".join(f"- {e.source} — {e.headline}\n  {e.url}" for e in selected))
        return finish(redact("\n\n".join(sections)), llm_status, selected)
