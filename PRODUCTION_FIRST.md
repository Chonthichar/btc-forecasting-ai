# Production-first deployment

The plan is:

1. Deploy the full system now with the current model artifacts.
2. Stabilize API, live data, containers, credentials, logs, and uptime.
3. Improve model accuracy separately.
4. Replace model artifacts later without changing the API contract.

## Live model routes

Live monitoring dashboard: **http://localhost:8000/monitor**.
The hourly monitoring worker now starts with the API. See [MONITORING.md](MONITORING.md)
for accuracy definitions, durable history, and worker controls.

- `GET /forecast/1`
- `GET /forecast/6`
- `GET /forecast/24`
- `GET /forecasts`
- `GET /health`
- `POST /refresh`

The newest regime/meta-label experiment is included in `notebooks/` but is not required for the service to boot.

## Start locally with Docker

```bash
cp .env.example .env
docker compose up --build -d
```

API:
- http://localhost:8000
- http://localhost:8000/docs
- http://localhost:8000/health

Optional UI:

```bash
docker compose --profile ui up --build -d
```

UI:
- http://localhost:7860

## First refresh

```bash
curl -X POST http://localhost:8000/refresh \
  -H "Content-Type: application/json" \
  -d '{"use_gdelt": true}'
```

Then:

```bash
curl http://localhost:8000/forecasts
```

## Later accuracy upgrades

Export the improved model/scaler artifacts, update `app/deployment_registry.yaml`,
rebuild/restart the container, and keep the same external API.
