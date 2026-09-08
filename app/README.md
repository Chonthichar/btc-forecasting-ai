# Realtime Agentic BTC Forecasting App

Application layer for the reproducible BTC thesis pipeline.

## Flow
User/UI → OpenAI orchestrator → deterministic tools → live market + live sentiment + validation → trained thesis model → probability → LLM explanation.

The LLM does not generate the numerical BTC forecast.

## Agents
- `MarketDataAgent`: Binance closed 1h BTCUSDT candles + exact thesis market features.
- `SentimentAgent`: existing collector → validator/FinBERT → manager pipeline.
- `DataQualityAgent`: sequence length, hourly continuity, freshness, NaNs.
- `ForecastAgent`: selected saved 1h/6h/24h PyTorch models and scalers.
- `CryptoLLMOrchestrator`: OpenAI Responses API function tools.

## Secrets
Use Colab Secrets / environment variables:
- `OPENAI_API_KEY`
- `ALPHA_VANTAGE_API_KEY` (optional)
- `OPENAI_MODEL` (optional; default `gpt-4.1`)

## Notebook
Open `notebooks/Realtime_Prediction_V2_Agentic.ipynb`.

## Deployment registry
`deployment_registry.yaml` points to the currently selected v3 200-epoch models. Update it after V4 only if model selection changes.

## Next deployment step
Expose `RealtimeCryptoPipeline` through FastAPI and deploy on Cloud Run/Render. Log every forecast for prospective forward testing.
