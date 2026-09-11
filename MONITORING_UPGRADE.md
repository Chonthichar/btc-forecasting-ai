# Continuous monitoring upgrade

Implemented September 9, 2026 in the Production First bundle. All 134 offline tests pass against the built Docker image. The existing trained models, inference inputs, model fingerprints, reliability gate, heartbeat and forward settlement logic are preserved. No model was retrained.

## Deployment status

- Tested source installed in `C:\Users\Windows PC\Downloads\BTC_Forecasting_Production_First_Bundle\btc_prod`.
- Reviewed release also available in the Local Production bundle's `monitor-upgrade` directory; Docker image `btc-monitor-upgrade` builds successfully.
- All 57 existing Hugging Face forecast records were exported and restored without changing their IDs, probabilities, reference prices, issue timestamps or scored outcomes. All three model fingerprints match the running release.
- The verified database was uploaded to the approved **private** dataset repository, `chonthichar/crypto-forecasting-ai-monitoring`, commit `067dbbb6fddf0131b1eb99760a859881002cb982`.
- Five recent run logs were retained. The legacy API exposes only those five logs; older run logs and unpersisted chat evidence could not be exported. Every forecast listed by the legacy version inventory was retained.
- The upgrade is deployed and running at https://chonthichar-crypto-forecasting-ai.hf.space/monitor, commit `6f8f8778e293e200367752a3ba255fa0e41585dc`. The user supplied a restricted `MONITOR_BACKUP_TOKEN` through Space Secrets. Startup restored all 57 forecasts, and a controlled real scan successfully uploaded a fresh backup at 2026-09-09 13:23:24 UTC. The scan generated no research trigger and left Tavily and automatic Decision idle. Next scheduled cycle: 14:02 UTC. The local deployment token subsequently became invalid, so the final hosted restart test is pending a restart from Space Settings; running app and backup access remain operational.
- No provider secret, `.env`, local runtime directory or database is included in the public source release. Existing OpenAI/Tavily secrets and model settings remain unchanged.

## 1. Files changed

`app/api.py`, `app/monitoring_service.py`, `app/monitoring_store.py`, `app/monitor_worker.py`, `app/space_server.py`; `app/agents/market_research_agent.py`, `app/agents/orchestrator.py`, `app/agents/sentiment_agent.py`; `app/schemas/agent_models.py`, `app/services/agent_config.py`; `app/static/monitor/index.html`, `dashboard.js`, `analyst.js`, `analyst.css`; `scripts/check_space_bundle.py`, `requirements-agents.txt`, `.env.example`.

The sentiment-agent change only removes backup credentials from its child-process environment; its collection and inference behavior is unchanged. Replaced local source files are copied to `monitor-upgrade-source-backup` in the Local Production bundle.

## 2. Files created

- `app/services/research_triggers.py`: deterministic metrics and event rules.
- `app/services/market_monitoring.py`: persisted scan → trigger → research → validation → decision flow.
- `app/services/monitoring_events_store.py`: scans, events, query cache and ownership leases.
- `app/services/monitor_schedule.py`: configurable UTC schedule.
- `app/services/monitor_worker_runtime.py`: browser-independent worker and heartbeat.
- `app/services/monitoring_backup.py`: consistent private Hub snapshots and startup restore.
- `app/static/monitor/monitoring-events.js`: live operational dashboard updates.
- `tests/test_event_monitoring.py`, `tests/test_monitoring_persistence.py`.
- This report. A one-time `migrate_monitoring_history.py` helper and private local export remain outside the public release directory.

## 3. Scheduler behavior

`MONITOR_INTERVAL_MINUTES=60` preserves the existing hourly schedule at `HH:02 UTC`, allowing the preceding candle to close. Other intervals align to a UTC epoch with a two-minute offset. It does not depend on a browser, UI refresh or chat request.

The API persists the next due time. One worker holds a renewable 90-second scheduler lease; competing workers stay on standby. The existing inference lease and process lock remain in place, renewed during long cycles. Heartbeats continue during inference and research. A restart resumes the persisted schedule; an interrupted cycle becomes eligible after its stale ownership expires. Failed market cycles retry after at most five minutes. Heartbeats never count as successful scans.

An already-processed candle does not generate duplicate forecasts or repeat its research. It can still settle due outcomes and record a quantitative scan. A 30-second dashboard refresh only reads API state.

## 4. Trigger rules

Defaults, configurable through the existing environment settings:

| Setting | Default | Rule |
| --- | --- | --- |
| `PRICE_MOVE_TRIGGER_PCT` | `1.0` | Absolute one-hour close change ≥ 1% |
| `VOLATILITY_TRIGGER_MULTIPLIER` | `1.5` | Current 24-hour realized volatility ≥ prior baseline × multiplier |
| `SENTIMENT_SHIFT_THRESHOLD` | `0.25` | Signed sentiment delta ≥ threshold, or meaningful sign reversal |
| `RELIABILITY_THRESHOLD` | `0.50` | Existing gate; qualified signal transitions can trigger research |
| `RESEARCH_COOLDOWN_MINUTES` | `30` | Suppress recently reserved matching event fingerprints |
| `NEWS_CACHE_MINUTES` | `10` | Reuse matching query responses |

Genuine known regime changes and transitions involving a qualified UP/DOWN signal also trigger research. Small probability changes alone do not. Missing sentiment, unavailable volatility baselines and stale market data do not produce fabricated changes.

The monitoring regime is descriptive: HIGH_VOL compares realized volatility to a prior 168-return baseline excluding the current 24-hour window; BULL/BEAR compare 24-hour momentum with realized volatility; otherwise RANGE. These additional labels do not enter trained model features or alter probabilities. Sentiment uses the project's actual signed `[-1, 1]` scale.

## 5. Tavily and OpenAI cost control

An ordinary scan with no trigger performs **zero Tavily calls and zero automatic Decision calls**. A triggered event chooses at most three neutral, relevant queries. Source evidence is classified afterwards; search terms do not assume the forecast direction.

Event fingerprints and query caches persist in SQLite. Query leases and in-process request coalescing reduce concurrent duplicate searches. Reused evidence is revalidated for freshness; cooldown does not make an old source fresh. Repeated conditions for the same candle remain suppressed even if research failed. A genuinely new candle with the same event may be researched after cooldown.

Explicit news/context questions permit research immediately; explicit fresh research bypasses cache and monitoring cooldown. The dashboard's research button requests fresh research. Ordinary chat retains its existing Decision Agent behavior.

Automatic Decision runs only after triggered research produces useful, timestamp-verified evidence from primary or established sources. Tavily or OpenAI failure leaves committed forecasts intact. External news never raises validated reliability or overrides NO_SIGNAL/UNKNOWN. The existing OpenAI Agents SDK Decision Agent remains; deterministic monitoring and validation stages are ordinary Python services, not additional LLM agents.

## 6. Storage and persistence

SQLite remains the local working database. Additive tables store quantitative scans, normalized research events, cached queries, operation leases and immutable issue-time forecast annotations. Scans reference a shared research event rather than duplicating article bodies. Forecast annotations capture reliability, signal state and regime only at first issuance; legacy rows remain explicitly unannotated.

Forward metrics retain directional accuracy, Brier score and baseline comparison, and add qualified signal coverage, NO_SIGNAL/UNKNOWN frequencies, reliability buckets and regime performance. Outcomes are resolved only when the appropriate future candle exists. Historical predictions are never rewritten by research or settlement.

For ephemeral Spaces, set `MONITOR_BACKUP_REPO` to the private repository and supply `MONITOR_BACKUP_TOKEN` (or `HF_TOKEN`) as a **Space secret**. Startup restores the database before creating the scheduler. A missing/corrupt/inaccessible configured backup fails closed instead of starting empty and overwriting history. Existing nonempty local databases remain authoritative.

After each cycle, SQLite's online backup API creates a standalone file, removes stale runtime ownership, and uploads it using a parent-commit check. Backup conflicts or provider failures are surfaced separately in the dashboard; local forecasts remain saved. Only one deployment should write to a given backup repository. Local Docker keeps its existing bind-mounted runtime; leave backup settings empty locally when the repository belongs to the Space.

## 7. UI and API changes

The existing dark monitoring dashboard now displays worker status, last/next scan, last successful prediction cycle, research state, last trigger reason, stored contextual summary, operational agent states and backup health. It explains that research is event-triggered to reduce API calls. Forward performance adds coverage and grouped qualification metrics. No chain-of-thought is exposed.

New reads: `GET /monitoring/activity`, `GET /monitoring/scans`. Scheduler requests carry a worker ID; `POST /monitoring/worker/release` releases its own lease. Existing routes remain available.

## 8. Validation

- **134 unit/integration tests passed**, including all 14 requested scenarios.
- Offline tests exercise no-trigger cost control; price/regime/sentiment/qualified-state triggers; small probability noise; durable cooldown/cache expiry; explicit fresh research; provider failures; reliability preservation; browser-free scheduling; competing workers; restart recovery; settlement; private backup restoration and corruption handling.
- Eight concurrent lease contenders produce one scheduler owner.
- Backup restoration verifies a self-contained database and removes stale worker ownership.
- The Docker image builds, including required dashboard/module/model bundle checks. All three changed JavaScript files pass `node --check`.
- The legacy export verifies every forecast field and unchanged model fingerprints before upload to the private repository.
- Hosted checks passed: private restore and backup upload; exact preservation of all 57 forecast records and scored outcomes; unchanged model versions; active worker; dashboard HTML and static assets; and configured OpenAI Agents SDK / gpt-5.6. A second worker was rejected (`accepted=false`), and a duplicate scheduled run returned `not_due`. The no-trigger live scan recorded no new forecasts and performed no automatic research or Decision call. Final hosted restart verification is pending because the local deployment token became invalid. Browser rendering was not visually inspected. Test output contains pre-existing asynchronous event-loop ResourceWarnings; tests still pass.

## 9. Known limitations

- A worker is independent of browser sessions **while its host is running**. The current CPU Basic Space can sleep or be paused; continuous 24/7 uptime needs an always-running host. Hardware and billing have not been changed.
- Remote backups are after-cycle snapshots. An abrupt host loss before upload can lose work since the last successful backup; external API requests cannot be guaranteed exactly once across that crash window. Failed backups remain visible and require attention.
- Private repository snapshots retain database history in Hub revisions. Scans and event history currently grow over time; no automatic historical deletion was introduced.
- The deployed models do not currently provide validated production reliability. UNKNOWN and unavailable qualified metrics remain honest outcomes; directional probability and historical holdout accuracy are not substituted for reliability.
- Strict evidence validation can leave no useful sources, in which case automatic Decision is skipped. News is context, not proof of causation.
- Accuracy is delayed until the 1h/6h/24h outcomes become observable. A page refresh cannot produce immediate true accuracy.

## 10. Restart and verify

The hosted upgrade is deployed. For future CLI maintenance, first renew the invalid local deployment credential (it needs access to manage the Space); keep the restricted backup token in Space Secrets unchanged. Alternatively, use Space Settings to restart. The commands below use the existing safe wrapper, which reads the local deployment credential without printing it:

```powershell
Set-Location 'C:\Users\Windows PC\Downloads\BTC_Forecasting_Local_Production_Bundle\btc_prod'
.\hf-auth-command.ps1 spaces variables add chonthichar/crypto-forecasting-ai --env MONITOR_BACKUP_REPO=chonthichar/crypto-forecasting-ai-monitoring --env MONITOR_INTERVAL_MINUTES=60
.\hf-auth-command.ps1 upload chonthichar/crypto-forecasting-ai monitor-upgrade . --type space --commit-message 'Add event-triggered monitoring and durable private history'
.\hf-auth-command.ps1 spaces restart chonthichar/crypto-forecasting-ai
.\hf-auth-command.ps1 spaces info chonthichar/crypto-forecasting-ai --format json
Invoke-RestMethod 'https://chonthichar-crypto-forecasting-ai.hf.space/monitoring/activity'
Invoke-RestMethod 'https://chonthichar-crypto-forecasting-ai.hf.space/monitoring/scans?limit=2'
Invoke-RestMethod 'https://chonthichar-crypto-forecasting-ai.hf.space/monitoring/summary?horizon=1&days=30'
```

`MONITOR_BACKUP_REPO` and its private access secret are configured. The Space reported `RESTORED` then `OK`, with an active worker and all 57 preserved forecasts. A restart should retain forecast IDs, scan history and the next due time. The restricted backup credential should not be used to manage the Space itself.

Local Docker, from the installed Production First source:

```powershell
Set-Location 'C:\Users\Windows PC\Downloads\BTC_Forecasting_Production_First_Bundle\btc_prod'
docker compose up -d --build btc-api btc-monitor
docker compose ps
Invoke-RestMethod 'http://localhost:8000/monitoring/activity'
Invoke-RestMethod 'http://localhost:8000/monitoring/scans?limit=2'
docker compose logs --tail 60 btc-monitor
Start-Process 'http://localhost:8000/monitor'
```

To run the complete offline suite using the tested image:

```powershell
docker run --rm --network none --read-only --tmpfs /tmp --mount "type=bind,source=C:\Users\Windows PC\Downloads\BTC_Forecasting_Production_First_Bundle\btc_prod,target=/app,readonly" btc-monitor-upgrade python -m unittest discover -s tests -q
```

Store OpenAI and Tavily keys only in local `.env` for Docker or Hugging Face Settings → Secrets for the Space. For backup access, prefer a token restricted to read/write on the private monitoring repository. Never place credentials in README files, public source, dashboard code or chat messages.
