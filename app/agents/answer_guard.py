"""Validate references and selected high-risk claims before rendering prose.

These deterministic checks supplement grounding instructions; they are not a
general proof that every possible natural-language statement is true.
"""
from __future__ import annotations
import re
from decimal import Decimal, InvalidOperation

MARKER = re.compile(r"\{\{(fact|source):([A-Za-z0-9_.:-]{1,120})\}\}")
NUMBER = re.compile(r"[+-]?\d+(?:[.,:/-]\d+)*(?:%|)")

def numeric_values(text):
    values = set()
    for value in NUMBER.findall(text):
        unit = "%" if value.endswith("%") else ""
        raw = value.rstrip("%").replace(",", "")
        try:
            values.add((Decimal(raw), unit))
        except InvalidOperation:
            values.add((value, unit))  # Dates/time sequences must match exactly.
    return values

def render_grounded_answer(plan, facts, research, validation):
    answer = plan.answer
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Missing conversational answer")
    source_map = {item.id: item for item in research.evidence}
    # Some provider replies shorten {{fact:known.id}} to {{known.id}}.
    # Normalize only IDs already selected and present in the verified catalogs.
    def canonical_reference(match):
        key = match.group(1)
        if key in plan.fact_ids and key in facts:
            return "{{fact:" + key + "}}"
        if key in plan.evidence_ids and key in source_map:
            return "{{source:" + key + "}}"
        return match.group(0)
    answer = re.sub(r"\{\{([A-Za-z0-9_.:-]{1,120})\}\}", canonical_reference, answer)
    used_facts, used_sources = set(), set()

    def expand(match):
        kind, key = match.groups()
        if kind == "fact":
            if key not in plan.fact_ids or key not in facts:
                raise ValueError("Unsupported fact reference")
            used_facts.add(key)
            return ""
        if key not in plan.evidence_ids or key not in source_map:
            raise ValueError("Unsupported source reference")
        used_sources.add(key)
        source = source_map[key]
        return f"[{source.direction.upper()}] {source.source}: {source.headline}"

    rendered = MARKER.sub(expand, answer).strip()
    rendered = re.sub(r"[ \t]+([.,;!?])", r"\1", rendered)
    rendered = re.sub(r"[ \t]{2,}", " ", rendered)
    rendered = re.sub(r"\n[ \t]+", "\n", rendered)
    if not rendered:
        raise ValueError("The answer contained only citations")
    # Repeated numeric values must already occur in the cited original records.
    # Horizon labels are fixed application constants; dates remain whole tokens.
    grounding = " ".join(facts[key] for key in used_facts) + " " + " ".join(
        source_map[key].headline + " " + source_map[key].summary for key in used_sources)
    allowed = numeric_values(grounding + " 1 6 24")
    prose = MARKER.sub("", answer)
    if re.search(r"https?://|www\.|\{\{|\}\}", prose) or not numeric_values(prose) <= allowed:
        raise ValueError("An ungrounded value, URL, or reference was generated")
    if re.search(r"\b(?:guaranteed (?:profit|return|gain)|will (?:definitely|certainly)|"
                 r"(?:bitcoin|btc|the price) (?:will|is going to) (?:rise|fall|rally|crash|increase|decrease))\b", prose, re.I):
        raise ValueError("Unsupported directional certainty")
    for sentence in re.split(r"[.!?\n]+", prose.lower()):
        negated = bool(re.search(r"\b(?:not|never|cannot|can't|doesn't|isn't|no|without|would|could|if|means|refers|measures|requires|needs|until|whether)\b", sentence))
        if not negated:
            model = re.search(r"\b(?:model|forecast|prediction|signal)\b", sentence)
            external = re.search(r"\b(?:etf|news|outflows?|inflows?|fed|macro|liquidations?)\b", sentence)
            cause = re.search(r"\b(?:because|caus\w*|driv\w*|due to|explains?|reason for)\b", sentence)
            if model and external and cause:
                raise ValueError("Unsupported model causal attribution")
            if any(state in {"UNKNOWN", "NO_SIGNAL"} for state in validation.signal_states.values()):
                if re.search(r"\b(?:reliable|qualified|confident|high.confidence|strong) (?:bullish|bearish|directional|signal|forecast|prediction)\b|"
                             r"\b(?:signal|forecast|prediction) (?:is|looks|remains) (?:reliable|qualified|strong)\b", sentence):
                    raise ValueError("Unqualified signal was promoted")
    if plan.evidence_ids and not used_sources:
        raise ValueError("Selected sources were not grounded in the answer")
    if validation.mixed_evidence:
        directions = {source_map[key].direction for key in used_sources}
        if not {"bullish", "bearish"} <= directions:
            raise ValueError("The answer omitted opposing evidence")
    return rendered, used_facts, [item for item in research.evidence if item.id in used_sources]
