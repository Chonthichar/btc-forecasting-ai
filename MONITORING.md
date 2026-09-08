# BTC Observatory

The monitoring dashboard is served by the existing API at **http://localhost:8000/monitor**.
The compact UI remains at http://localhost:7860 and now reads the same API records.

## Start and stop

From this Production First bundle:

```powershell
docker compose --profile ui up --build -d
docker compose ps
```

`btc-api` owns inference and SQLite, `btc-monitor` schedules hourly cycles, and
`btc-ui` is the optional compact UI. Docker Desktop must stay running. The worker
runs once after startup and then at two minutes past each UTC hour. Failed runs
retry after five minutes; API connection failures retry after one minute.
The dashboard polls every 30 seconds, or every five seconds during a refresh.

```powershell
docker compose stop btc-monitor
docker compose start btc-monitor
docker compose logs --tail=50 btc-api btc-monitor
```

Stopping the worker pauses automatic collection. Run refresh remains available
on the dashboard. Multiple refresh requests cannot run inference concurrently.
The first news collection may take several minutes; it is limited to 15 minutes.
Its details are saved locally in `runtime/sentiment/latest_collection.log`.

## What is measured

- Every original prediction is saved with a fingerprint of the deployed model,
  scalers, configuration, and feature/inference code. One record per model version,
  horizon, and reference candle is allowed. Repeated refreshes do not add votes.
- `timestamp` in the market data is the candle **open**. For a 09:00–10:00 UTC
  reference candle, the 1h forecast compares its close with the 10:00–11:00 close,
  and becomes eligible for evaluation at 11:00. The 6h and 24h horizons follow
  the same convention. A forecast issued after its outcome was available is
  retained as `excluded_late`, and contributes to no accuracy metric.
- Direction is UP when the future close is strictly greater than the reference
  close. Equal prices count as not UP. This matches every observable non-tied
  label in the bundled data; the historical dataset contains no exact ties and
  the original tie-rule source is not included.
- Accuracy = correct / scored; Brier score = mean squared difference between
  P(UP) and the eventual UP label. Always-UP accuracy uses the same scored rows.
  These are direction/probability metrics, not trading returns.
- Seven- and thirty-day windows use forecast issue time and one selected model
  version. The curve shows the trailing window as it was known each day;
  pending, missing-price, and excluded forecasts are not scored. Counts remain
  visible, and no historical holdout predictions are imported as live results.
- Longer-horizon forecasts issued hourly overlap; their outcomes are correlated.
  Small samples and overlapping outcomes should not be treated as independent
  evidence of improvement. No independence-based confidence intervals are shown.

## Persistence and failure recovery

The database is `runtime/monitoring.sqlite3`, mounted from the host into the API.
Forecasts, outcomes, snapshots, run logs, and worker status survive container
recreation. Do not delete `runtime` to restart the app. For a filesystem backup,
stop the API and worker first and copy the database (plus any WAL/SHM files still
present), or use SQLite's backup API while running.

After downtime, old predictions are evaluated against their exact finalized
target candles. Missing prices remain `awaiting_market_data` and are retried;
new forecasts are never invented for hours when the service was offline.
The last successful snapshot stays visible when a provider fails, with its
original timestamp and a stale-data indicator.

The API and worker continue the existing local deployment's access model. This
dashboard adds no login and is intended for the local Docker deployment. Remote
hosting would need a separately configured deployment and access controls.

## Endpoints

- `GET /monitoring/summary?horizon=1&days=7`
- `GET /monitoring/history?horizon=1&days=7&limit=20&offset=0`
- Both accept `model_version`; history also accepts `status`.
- `POST /monitoring/run?background=true` with `{"trigger":"manual"}`
- Existing `/refresh`, `/forecast/{horizon}`, `/forecasts`, `/market`, `/sentiment`,
  `/health`, and `/system` routes remain available.

Run the monitoring regression suite in the app image (tests are omitted from
production image build context, so mount the source tree read-only):

```powershell
docker run --rm --network none --mount "type=bind,source=$((Get-Location).Path),target=/app,readonly" btc_prod-btc-api python -m unittest discover -s tests -v
```
