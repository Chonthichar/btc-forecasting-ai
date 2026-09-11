"""Serve the dashboard, API and optional Gradio view on one public Space port."""
from __future__ import annotations
from contextlib import asynccontextmanager
import os
import subprocess
import sys
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

def create_app():
    port = int(os.getenv("PORT", "7860"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be a valid TCP port")
    local_api = f"http://127.0.0.1:{port}"
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr
    from . import api, app_gradio
    app_gradio.API = local_api

    @asynccontextmanager
    async def lifespan(_):
        worker = None
        if os.getenv("SPACE_START_MONITOR", "true").lower() == "true":
            env = dict(os.environ, BTC_API_URL=local_api)
            for key in ("OPENAI_API_KEY", "TAVILY_API_KEY", "ALPHA_VANTAGE_API_KEY", "MONITOR_BACKUP_TOKEN", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
                env.pop(key, None)
            worker = subprocess.Popen([sys.executable, "-m", "app.monitor_worker"], env=env)
        try:
            yield
        finally:
            if worker is not None and worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)

    public = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @public.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/monitor", status_code=307)

    public = gr.mount_gradio_app(public, app_gradio.demo, path="/gradio")
    public.mount("/", api.app)
    return public

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
