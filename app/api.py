from __future__ import annotations
import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Literal
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import yaml
from pydantic import BaseModel, Field
from .monitoring_service import MonitoringService

PROJECT = Path(os.getenv("BTC_PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))
app = FastAPI(title="BTC Forecasting API", version="1.2.0", description="Live forecasting, durable performance monitoring, and validated AI market research.")
_assets = Path(__file__).resolve().parent / "static/monitor"
app.mount("/monitor/assets", StaticFiles(directory=_assets), name="monitor-assets")
_service = None
_lock = threading.Lock()
_analyst = None
_analyst_lock = threading.Lock()
_catalog = None
_assets_version = None
_ticker = None
_predictions = None
_predictions_lock = threading.Lock()

def get_service():
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                _service = MonitoringService(PROJECT)
    return _service

def get_pipeline():
    return get_service().get_pipeline()

def get_analyst():
    global _analyst
    service = get_service()
    if _analyst is None or _analyst.monitoring is not service:
        with _analyst_lock:
            if _analyst is None or _analyst.monitoring is not service:
                from .agents.orchestrator import AgentOrchestrator
                _analyst = AgentOrchestrator(service)
    return _analyst

from .analyst_api import create_analyst_router
app.include_router(create_analyst_router(get_analyst))

class RefreshRequest(BaseModel):
    use_gdelt: bool = True

class MonitoringRunRequest(RefreshRequest):
    trigger: Literal["manual", "hourly"] = "manual"
    worker_id: str | None = Field(None, pattern=r"^[a-f0-9]{32}$")

class HeartbeatRequest(BaseModel):
    next_run_at: str | None = None
    worker_id: str | None = Field(None, pattern=r"^[a-f0-9]{32}$")

def _asset_version():
    """Fingerprint of the dashboard's own JS and CSS. Stamped into the asset
    URLs so a browser cannot keep serving a previous build from cache: the URL
    itself changes whenever the content does. Hand-edited version strings were
    forgotten between deploys, which left visitors on stale assets."""
    global _assets_version
    if _assets_version is None:
        digest = hashlib.sha256()
        for path in sorted(_assets.glob("*")):
            if path.suffix in {".js", ".css"}:
                digest.update(path.name.encode())
                digest.update(path.read_bytes())
        _assets_version = digest.hexdigest()[:12]
    return _assets_version

@app.get("/monitor", include_in_schema=False)
@app.get("/monitor/", include_in_schema=False)
def dashboard():
    page = (_assets / "index.html").read_text(encoding="utf-8")
    # The trailing guard keeps manifest.json from matching as manifest.js.
    page = re.sub(r"(?<=/monitor/assets/)([\w.-]+\.(?:js|css))(?!\w)(\?v=[^\"']*)?",
                  lambda m: f"{m.group(1)}?v={_asset_version()}", page)
    return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

@app.get("/")
def root():
    return {"service": "BTC Forecasting API", "status": "running", "horizons": [1, 6, 24], "docs": "/docs", "dashboard": "/monitor", "mode": "production-first"}

@app.get("/health")
def health():
    service = get_service()
    loaded = bool(service.snapshot())
    return {"status": "ok" if loaded else "ready_no_live_snapshot", "live_snapshot_loaded": loaded,
            "last_error": service.store.get_state("last_error"), "deployed_horizons": [1, 6, 24],
            "worker_alive": service.system_status()["worker_alive"]}

@app.post("/refresh")
def refresh(req: RefreshRequest):
    service = get_service()
    result = service.run(use_gdelt=req.use_gdelt)
    if result["status"] == "error":
        raise HTTPException(503, result["detail"])
    if result["status"] == "busy":
        raise HTTPException(409, result["detail"])
    return service.snapshot()

def snapshot():
    data = get_service().snapshot()
    if not data:
        raise HTTPException(409, "Awaiting the first successful refresh.")
    return data

@app.get("/forecasts")
def forecasts():
    return snapshot()["forecasts"]

@app.get("/forecast/{horizon_hours}")
def forecast(horizon_hours: int):
    if horizon_hours not in (1, 6, 24):
        raise HTTPException(400, "horizon must be 1, 6, or 24")
    return snapshot()["forecasts"][str(horizon_hours)]

@app.get("/market")
def market():
    return snapshot()["market"]

@app.get("/sentiment")
def sentiment():
    data = snapshot()
    quality = data.get("quality", {})
    return {**data["sentiment"], "latest_model_sentiment_score": quality.get("latest_sentiment_score"),
            "hours_with_news_in_latest_sequence": quality.get("hours_with_news_in_latest_sequence")}

@app.get("/system")
def system():
    return {"deployed_models": get_service().models, "monitoring": get_service().system_status()}

def _load_catalog():
    global _catalog
    if _catalog is None:
        _catalog = yaml.safe_load((PROJECT / "app/asset_catalog.yaml").read_text(encoding="utf-8"))
    return _catalog

@app.get("/prices")
def prices():
    """Spot quotes for the asset switcher. Quotes only: a price here does not
    imply that the asset has a deployed model."""
    global _ticker
    if _ticker is None:
        from .services.price_ticker import PriceTicker
        _ticker = PriceTicker()
    catalog = _load_catalog()
    entries = (catalog.get("assets") or {})
    quotes = _ticker.quotes([entry.get("symbol") for entry in entries.values()])
    return {"prices": {str(code).strip().upper(): {
                "price": (quotes.get(entry.get("symbol")) or {}).get("price"),
                "change_24h": (quotes.get(entry.get("symbol")) or {}).get("change_24h")}
            for code, entry in entries.items()}}

@app.get("/assets")
def assets():
    """Thesis asset scope. An asset is only reported as deployed when trained
    artifacts for it were actually loaded, so a planned coin cannot advertise
    itself as a live model."""
    _catalog = _load_catalog()
    horizons = sorted(get_service().models)
    deployed_code = str(_catalog.get("deployed_asset", "")).strip().upper()
    items = []
    for code, entry in (_catalog.get("assets") or {}).items():
        code = str(code).strip().upper()
        live = bool(horizons) and code == deployed_code
        items.append({"code": code, "name": entry.get("name"), "pair": entry.get("pair"),
                      "symbol": entry.get("symbol"), "deployed": live,
                      "horizons": horizons if live else [], "requires": entry.get("requires")})
    return {"assets": items}

def get_predictions_service():
    """Lazy: importing torch and scikit-learn is expensive, and the dashboard
    must still serve monitoring pages if the deployment models are absent."""
    global _predictions
    if _predictions is None:
        with _predictions_lock:
            if _predictions is None:
                from .forecasting.service import DeploymentPredictionService
                _predictions = DeploymentPredictionService(PROJECT)
    return _predictions

@app.get("/predictions")
def predictions(refresh: bool = False):
    """All six frozen-model outputs for the latest closed candle.

    Risk is a two-sided +/-2.5% barrier-touch score, not a crash probability;
    direction scores are model scores, not calibrated probabilities."""
    from .forecasting.feature_builder import FeatureError
    try:
        return get_predictions_service().current(refresh=refresh)
    except FeatureError as exc:
        # Brief section 12: refuse rather than predict on unusable data.
        raise HTTPException(503, f"Live data cannot support a prediction: {exc}") from None
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Deployment model artifact missing: {exc}") from None

@app.get("/monitoring/summary")
def monitoring_summary(horizon: int = 1, days: int = 7, model_version: str | None = Query(None, max_length=100)):
    if horizon not in (1, 6, 24) or days not in (7, 30):
        raise HTTPException(422, "Choose horizon 1, 6, or 24 and a 7- or 30-day window.")
    return get_service().summary(horizon, days, model_version)

@app.get("/monitoring/history")
def monitoring_history(horizon: int = 1, days: int = 7,
                       model_version: str | None = Query(None, max_length=100),
                       status: Literal["scored", "pending", "awaiting_market_data", "excluded_late"] | None = None,
                       limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)):
    if horizon not in (1, 6, 24) or days not in (7, 30):
        raise HTTPException(422, "Choose horizon 1, 6, or 24 and a 7- or 30-day window.")
    service = get_service()
    version = model_version or service.models[horizon]["model_version"]
    return service.store.history(horizon=horizon, days=days, model_version=version, status=status, limit=limit, offset=offset)

@app.post("/monitoring/run")
def monitoring_run(req: MonitoringRunRequest, tasks: BackgroundTasks, background: bool = False):
    service = get_service()
    if background:
        if service.system_status()["active_run"]:
            return JSONResponse({"status": "busy", "detail": "A refresh is already running."}, status_code=202)
        tasks.add_task(service.run, req.trigger, req.use_gdelt, req.worker_id)
        return JSONResponse({"status": "queued"}, status_code=202)
    return service.run(req.trigger, req.use_gdelt, req.worker_id)

@app.post("/monitoring/heartbeat", include_in_schema=False)
def monitoring_heartbeat(req: HeartbeatRequest):
    return get_service().heartbeat(req.next_run_at, req.worker_id)

@app.post("/monitoring/worker/release", include_in_schema=False)
def release_worker(req: HeartbeatRequest):
    if req.worker_id:
        get_service().event_store.release("scheduler", req.worker_id)
    return {"released": True}

@app.get("/monitoring/activity")
def monitoring_activity():
    return get_service().monitoring_activity()

@app.get("/monitoring/scans")
def monitoring_scans(limit: int = Query(50, ge=1, le=200)):
    return {"items": get_service().event_store.scans(limit)}
