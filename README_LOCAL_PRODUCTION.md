# BTC Forecasting — Local + Production Bundle

This folder is a self-contained snapshot of the BTC thesis forecasting project for local development and container deployment.

## Included

- core reproducible pipeline (`src/`, `config/`, runner scripts)
- latest selective-forecasting notebook
- corrected-sentiment collection notebook and canonical sentiment collector
- historical BTC market parquet used by the experiments
- historical hourly/article sentiment parquet used for the current development bundle
- realtime agentic Gradio application
- currently deployed 1h / 6h / 24h model artifacts referenced by `app/deployment_registry.yaml`
- Docker and local-run scaffolding

## Important model note

The realtime application currently loads the **saved v3 deployed models** in `app/deployment_registry.yaml`:

- 1h: VAE-Transformer, market-only
- 6h: GRU, market + sentiment
- 24h: VAE-Transformer, market + sentiment

The newest `BTC_1Y_Regime_MetaLabel_Selective_Forecasting.ipynb` is an **experiment notebook**. It does not yet export a production model registry compatible with the realtime app. Keep it for research/training; promote a new model only after validation and after saving model/scaler artifacts.

## Local setup

### macOS / Linux

```bash
./scripts/setup_local.sh
source .venv/bin/activate
# edit .env and add keys
./scripts/run_local.sh
```

Open: `http://localhost:7860`

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env
$env:BTC_PROJECT_ROOT = (Get-Location).Path
$env:BTC_SENTIMENT_SCRIPT = "$((Get-Location).Path)\sentiment_pipeline\btc_sentiment_agents.py"
python app\app_gradio.py
```

## Docker

```bash
cp .env.example .env
# edit .env
docker build -t btc-forecast-app .
docker run --env-file .env -p 7860:7860 btc-forecast-app
```

Or:

```bash
docker compose up --build
```

## Production hosting

The Docker image can be deployed to services that accept containers, for example Cloud Run, Render, Azure Container Apps, ECS, or a VM.

Production configuration should inject secrets as environment variables rather than committing `.env`.

Required/optional secrets:

- `OPENAI_API_KEY` — needed for the LLM chat/explanation layer
- `ALPHA_VANTAGE_API_KEY` — optional/recommended for the live sentiment collector
- `OPENAI_MODEL` — optional
- `PORT` — defaults to `7860`

## Data layout

```text
data/
  market/
    04_btc_training_ready.parquet
  sentiment/
    04_btc_sentiment_hourly.parquet
    03_final_btc_sentiment_articles.parquet
```

The historical files are useful for reproducibility and local experimentation. A production container generally does not need the full historical dataset once final model artifacts are frozen, so you can later make a smaller deployment-only image.

## Research notebooks

- `notebooks/BTC_reproducible_pipeline.ipynb`
- `notebooks/BTC_1Y_Regime_MetaLabel_Selective_Forecasting.ipynb`
- `notebooks/run_btc_sentiment_1y_2y_corrected_colab.ipynb`

## Recommended production promotion flow

1. finish the corrected sentiment collection;
2. retrain/evaluate the selected final model;
3. freeze the untouched test result;
4. export model state + scaler(s) + training metadata;
5. update `app/deployment_registry.yaml`;
6. smoke-test locally;
7. build the Docker image;
8. deploy;
9. log every live forecast for forward testing.
