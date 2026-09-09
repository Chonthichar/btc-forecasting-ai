"""Additive endpoints for context, grounded answers, and Tavily evidence."""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from app.schemas.agent_models import ChatRequest, ResearchRequest
from app.services.agent_config import redact

logger = logging.getLogger(__name__)

class SafeAnalystRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()
        async def validated(request):
            try:
                return await handler(request)
            except RequestValidationError:
                # FastAPI's default validation detail echoes rejected inputs.
                # A pasted credential must not be reflected back to the UI.
                raise HTTPException(422, "Invalid analyst request. Check the message length, history roles, request ID and timestamp format.") from None
        return validated

def safe_response(value):
    # Redact individual values without rewriting JSON delimiters or field names.
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: safe_response(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_response(item) for item in value]
    return redact(value) if isinstance(value, str) else value

def create_analyst_router(get_analyst):
    router = APIRouter(tags=["AI market analyst"], route_class=SafeAnalystRoute)

    def run_safely(fn):
        try:
            return safe_response(fn())
        except HTTPException:
            raise
        except RuntimeError:
            raise HTTPException(503, "The analyst is busy or temporarily unavailable. Please retry shortly.") from None
        except Exception as exc:
            logger.warning("Analyst operation unavailable (%s)", type(exc).__name__)
            raise HTTPException(503, "The analyst is temporarily unavailable. Forecasting and monitoring remain available.") from None

    @router.get("/context")
    def context():
        return run_safely(lambda: get_analyst().context())

    @router.get("/evidence")
    def evidence():
        return run_safely(lambda: get_analyst().evidence())

    @router.get("/agents/status")
    def status(request_id: str | None = Query(None, pattern=r"^[a-zA-Z0-9_-]{1,64}$")):
        return run_safely(lambda: get_analyst().activity(request_id))

    @router.post("/chat")
    def chat(request: ChatRequest):
        return run_safely(lambda: get_analyst().chat(request))

    @router.post("/research")
    def research(request: ResearchRequest):
        if request.mode == "historical":
            stamp = request.prediction_timestamp
            if stamp is None or stamp.utcoffset() is None or stamp > datetime.now(timezone.utc):
                raise HTTPException(422, "Historical research requires a timezone-aware prediction_timestamp no later than now.")
            request.prediction_timestamp = stamp.astimezone(timezone.utc)
        return run_safely(lambda: get_analyst().research(request))

    return router
