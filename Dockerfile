FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    API_PORT=8000 \
    UI_PORT=7860

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=1000:1000 . .
RUN chown 1000:1000 /app

ENV BTC_PROJECT_ROOT=/app \
    BTC_SENTIMENT_SCRIPT=/app/sentiment_pipeline/btc_sentiment_agents.py \
    BTC_API_URL=http://127.0.0.1:7860 \
    PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HOME=/tmp/huggingface \
    GRADIO_TEMP_DIR=/tmp/gradio

# Hugging Face exposes 7860 publicly.
# Dashboard, FastAPI and Gradio share the same public port.
EXPOSE 7860

CMD ["python", "-m", "app.space_server"]
