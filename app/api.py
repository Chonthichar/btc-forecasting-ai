from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Literal
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .monitoring_service import MonitoringService

PROJECT = Path(os.getenv("BTC_PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))
app = FastAPI(title="BTC Forecasting API", version="1.1.0", description="Live forecasting and durable forward performance monitoring.")
_assets = Path(__file__).resolve().parent / "static/monitor"
app.mount("/monitor/assets", StaticFiles(directory=_assets), name="monitor-assets")
_service = None
_lock = threading.Lock()

def get_service():
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                _service = MonitoringService(PROJECT)
    return _service

def get_pipeline():
    return get_service().get_pipeline()

class RefreshRequest(BaseModel):
    use_gdelt: bool = True

class MonitoringRunRequest(RefreshRequest):
    trigger: Literal["manual", "hourly"] = "manual"

class HeartbeatRequest(BaseModel):
    next_run_at: str | None = None

@app.get("/monitor", include_in_schema=False)
@app.get("/monitor/", include_in_schema=False)
def dashboard():
    return FileResponse(_assets / "index.html", headers={"Cache-Control": "no-cache"})

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
        tasks.add_task(service.run, req.trigger, req.use_gdelt)
        return JSONResponse({"status": "queued"}, status_code=202)
    return service.run(req.trigger, req.use_gdelt)

@app.post("/monitoring/heartbeat", include_in_schema=False)
def monitoring_heartbeat(req: HeartbeatRequest):
    return get_service().heartbeat(req.next_run_at)
