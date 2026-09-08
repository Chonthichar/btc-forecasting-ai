"""Optional compact UI. The API owns all inference and saved forecast state."""
from __future__ import annotations
import os
import gradio as gr
import requests

API = os.getenv("BTC_API_URL", "http://127.0.0.1:8000").rstrip("/")
def dashboard():
    try:
        response = requests.get(f"{API}/forecasts", timeout=15)
        if response.status_code == 409:
            return "Awaiting the first successful hourly refresh. Open the monitoring dashboard for progress."
        response.raise_for_status()
        forecasts = response.json()
        return "\n".join(f"{h}h | {forecasts[str(h)]['direction']} | P(up)={forecasts[str(h)]['probability_up']:.3f} | {forecasts[str(h)]['model']}" for h in [1, 6, 24])
    except (requests.RequestException, ValueError, KeyError):
        return "The forecasting API is unavailable. Please check the monitoring dashboard or container status."

def refresh():
    try:
        response = requests.post(f"{API}/monitoring/run?background=true", json={"trigger": "manual"}, timeout=15)
        response.raise_for_status()
        return "Refresh requested. The monitoring dashboard shows its progress.\n\n" + dashboard()
    except requests.RequestException:
        return "Unable to request a refresh from the forecasting API."

with gr.Blocks(title="BTC Forecasting") as demo:
    gr.Markdown(
    """
    # ₿ BTC Forecasting AI

    **Multi-horizon forecasting • Live market intelligence • AI research**

    Experimental research system — not financial advice.
    """
)
    gr.Markdown(f"[Open the full dashboard and AI Market Analyst]({os.getenv('BTC_PUBLIC_API_URL', '')}/monitor)")
    status = gr.Textbox(value="Connecting to forecasting API…", label="Current forecasts", lines=6)
    gr.Button("Refresh live data").click(refresh, outputs=status)
    gr.Button("Reload forecasts").click(dashboard, outputs=status)
    demo.load(dashboard, outputs=status)
    gr.Timer(30).tick(dashboard, outputs=status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
