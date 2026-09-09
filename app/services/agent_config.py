"""One environment-only configuration boundary; secrets are never serialized."""
from __future__ import annotations
import os
import re
from dataclasses import dataclass, field

def bounded_env(name, default, low, high):
    try:
        result = float(os.getenv(name, str(default)))
        if not low <= result <= high:
            raise ValueError()
        return result
    except ValueError:
        raise ValueError(f"Invalid {name}; expected a number between {low} and {high}.") from None

@dataclass(frozen=True)
class AgentSettings:
    openai_key: str = field(repr=False)
    openai_model: str
    tavily_key: str = field(repr=False)
    reliability_threshold: float
    news_cache_minutes: float
    news_max_age_hours: float
    context_max_age_minutes: float
    tavily_timeout_seconds: float
    openai_timeout_seconds: float

    @classmethod
    def from_env(cls):
        return cls(os.getenv("OPENAI_API_KEY", "").strip(), os.getenv("OPENAI_MODEL", "").strip(),
                   os.getenv("TAVILY_API_KEY", "").strip(), bounded_env("RELIABILITY_THRESHOLD", .50, 0, 1),
                   bounded_env("NEWS_CACHE_MINUTES", 10, 0, 120), bounded_env("NEWS_MAX_AGE_HOURS", 72, 1, 720),
                   bounded_env("CONTEXT_MAX_AGE_MINUTES", 90, 1, 1440),
                   bounded_env("TAVILY_TIMEOUT_SECONDS", 12, 1, 60), bounded_env("OPENAI_TIMEOUT_SECONDS", 45, 1, 120))

    def public_status(self):
        return {"openai_configured": bool(self.openai_key and self.openai_model),
                "tavily_configured": bool(self.tavily_key), "openai_model": self.openai_model or None,
                "agent_framework": "OpenAI Agents SDK",
                "reliability_threshold": self.reliability_threshold, "news_cache_minutes": self.news_cache_minutes}

def redact(value):
    text = str(value)
    for name in ("OPENAI_API_KEY", "TAVILY_API_KEY", "ALPHA_VANTAGE_API_KEY"):
        key = os.getenv(name)
        if key:
            text = text.replace(key, "[redacted]")
    text = re.sub(r"(?i)\b(?:sk-[A-Za-z0-9_-]{10,}|tvly-[A-Za-z0-9_-]{8,})", "[redacted]", text)
    return re.sub(r"(?i)(apikey|api_key|token|authorization)\s*[=:]\s*[^\s&,;]+", r"\1=[redacted]", text)

def redact_values(value):
    if isinstance(value, dict):
        return {key: redact_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_values(item) for item in value]
    return redact(value) if isinstance(value, str) else value
