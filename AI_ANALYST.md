# BTC Forecasting AI

The existing monitoring dashboard now includes saved model signals, an AI Market Analyst, and Tavily evidence. Open **http://localhost:8000/monitor**. The original Gradio interface remains at **http://localhost:7860**.

## Keys and local startup

The running project is **Production_First_Bundle**, not the similarly named Local_Production_Bundle used for staging:

```powershell
cd 'C:\Users\Windows PC\Downloads\BTC_Forecasting_Production_First_Bundle\btc_prod'
notepad .env
```

Set these entries locally, without posting their values in chat or adding them to source files:

```dotenv
OPENAI_API_KEY=your-local-key
OPENAI_MODEL=gpt-4.1
TAVILY_API_KEY=your-local-key
RELIABILITY_THRESHOLD=0.50
NEWS_CACHE_MINUTES=10
```

Keep your existing `OPENAI_MODEL` selection if already configured. The deployed environment was checked only for presence of keys. The API reads keys from its environment; there is no browser key-entry form. `.env` and other environment files are excluded from Git and Docker build context; `.env.example` contains placeholders. Only the API container receives these keys. The Gradio UI and hourly worker do not receive them. Docker owners can inspect container environment values, so local machine access remains relevant.

Build and start:

```powershell
docker compose --profile ui up --build -d
docker compose --profile ui ps
Start-Process 'http://localhost:8000/monitor'
```

After changing keys, recreate the API container to load the new environment:

```powershell
docker compose up -d --force-recreate btc-api
```

Restarting alone does not reread `.env`. Ports bind to `127.0.0.1` for this local deployment. Do not use `docker compose config` or `docker inspect` output in public messages; they can reveal resolved environment values.

## What happens every 30 seconds

The browser reads saved forecasts, scored outcomes, system health, normalized context, and cached evidence from your local API. It does **not** trigger OpenAI, Tavily, model inference, or retraining. The hourly worker continues collecting and evaluating while the page is closed. The existing **Run refresh** button explicitly requests inference and evaluation; chat does not.

Accuracy changes only when a saved forecast reaches its target time and a finalized outcome candle is available. A 24-hour forecast cannot be scored after 30 seconds. The price strip displays the latest saved **hourly closing price**, not a tick-by-tick exchange feed. LIVE means the stored context passes freshness checks; OFFLINE also covers stale or unavailable market context.

## Architecture

1. `ForecastContextAgent` reads a copy of the saved application snapshot. It extracts actual values and preserves nulls for absent fields. No provider call or model inference happens here.
2. `MarketResearchAgent` performs conditional Tavily searches. It normalizes bounded source records, dates, conservative direction labels, categories, quality, and links.
3. `ValidationAgent` recomputes internal context to detect modified facts, enforces gating/freshness, and checks publication times, BTC relevance, duplicates, source identity, and conflicting evidence.
4. `DecisionAgent` uses the official **OpenAI Agents SDK 0.22.1**, with `Agent`, `Runner`, `OpenAIResponsesModel` and a typed `DecisionPlan` output. The run is limited to one turn and one OpenAI Responses call. OpenAI writes a question-specific conversational answer with fact and evidence citations. The server validates and hides fact citations, checks numeric values against cited records, and inserts attributed source headlines. Fixed horizon labels are allowed; links come from validated sources. Fact sentences are not repeated after the explanation. Reliability notices are attached when discussing forecasts; greetings and general concepts do not trigger the old fixed market report. Deterministic checks reject selected forms of unsupported certainty and model causation; these checks supplement grounding instructions rather than prove every possible natural-language statement true.

OpenAI receives normalized context, validated research, validation decisions, the question, a bounded history, and a catalog derived from those facts. It has no independent search tools. `store=False` is requested and SDK tracing is disabled. The other three workflow stages retain deterministic Python and Tavily logic; they do not make additional LLM calls. Provider errors produce safe fallback answers without exposing raw exceptions or request bodies. Missing OpenAI configuration still allows verified internal summaries; missing Tavily still allows model analysis.

The browser keeps at most 12 history messages in memory and sends them only with chat requests. Clear removes that history; it does not remove forecasts or evidence. Clearing while a request runs discards its eventual answer in the UI but does not cancel work already accepted by the API. No chat is stored in SQLite or browser local storage. Four agent indicators show operational states only.

## Reliability rules

`RELIABILITY_THRESHOLD` is centralized in `AgentSettings` and defaults to 0.50.

| Input condition | Final signal |
| --- | --- |
| Fresh valid forecast, independently validated reliability at/above threshold, no veto | UP or DOWN according to raw probability |
| Reliability below threshold or an explicit selective-gate/upstream veto | NO_SIGNAL |
| Missing reliability, invalid required values, or unconfirmed freshness | UNKNOWN |
| Equal UP/DOWN probabilities with otherwise usable information | NO_SIGNAL |

P(UP) is direction, not reliability. For example, P(UP)=0.14 and validated reliability=0.80 can qualify as DOWN. P(UP)=0.90 with reliability=0.40 remains NO_SIGNAL. These examples are test fixtures, never inserted into live monitoring history.

The context reader accepts explicit `validated_reliability`, `meta_reliability`, or a reliability/confidence value carrying its corresponding explicit validation flag. Raw confidence and holdout accuracy are not substitutes. Explicit false gates always veto qualification. Source candles must be closed, forecast issuance must be after the source close and not in the future, and the context must pass the age limit.

**The current deployed model snapshots do not provide validated reliability, selective gates, or regime labels.** Therefore the new cards honestly show UNKNOWN / “No reliable forecast right now,” with the raw direction and probability underneath. Missing regime/labels remain unavailable. This integration does not calibrate models, train a reliability model, or manufacture a confidence score. News never upgrades a signal, and no causal attribution to model inputs is asserted.

## Tavily behavior

- Direct HTTPS calls to Tavily Search with Bearer authentication, news topic, basic depth, up to five results per query, no generated Tavily answer, and no raw page content.
- Two or three balanced focused queries selected from the question, independent of forecast direction. Topics include general BTC news, ETF flows, institutions, macro/Fed/CPI/employment, regulation, exchanges/security, derivatives/liquidations, whale activity, and geopolitics.
- Model/feature and internal sentiment questions do not automatically search. Current movement/news questions do. Explicit no-web instructions skip research.
- Identical queries share a thread-safe bounded cache and concurrent work. Default TTL is 10 minutes; failures are cached for no more than 30 seconds. Historical cache keys include the exact cutoff. Chat and explicit research share a limit of three concurrent operations.
- Quality favors named financial publishers and official sources using actual hostnames. Anonymous prediction/spam material is rejected; retained unverified sources are flagged. URL and normalized-headline deduplication do not guarantee that every syndicated paraphrase is found.
- Bullish/bearish labels are conservative headline classifications, not proven market impact. Neutral and mixed evidence are preserved; both sides appear when validation finds conflicting directions.
- Live articles older than `NEWS_MAX_AGE_HOURS` (default 72) are rejected when dates are verifiable. Missing or uncertain publication times may remain in live research with freshness explicitly unverified. Retrieval time is never substituted for publication time.
- Historical research requires an aware, nonfuture `prediction_timestamp`. Articles after that timestamp or without a verified publication time are rejected. Provider date filters alone are insufficient. This protects the evidence cutoff; it does not claim an archival replay of historic web search rankings or original page versions.
- Timeouts, rate limits, absent keys, bad responses, empty results and partial coverage remain structured states. No alternate search provider is silently substituted. The existing model sentiment pipeline remains unchanged.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /context` | Validated saved context, per-horizon signal states, provider configuration booleans |
| `GET /evidence` | Last live research, revalidated at read time; no provider calls |
| `GET /agents/status?request_id=...` | Operational status and safe configuration |
| `POST /chat` | Conditional research and a grounded answer |
| `POST /research` | Explicit Tavily research, optionally historical |

All existing forecasting, monitoring, health and refresh routes are preserved. Interactive contracts are at **http://localhost:8000/docs**.

```powershell
Invoke-RestMethod 'http://localhost:8000/context'
Invoke-RestMethod 'http://localhost:8000/chat' -Method Post -ContentType 'application/json' -Body '{"message":"What model is used for 6h?","history":[]}'
Invoke-RestMethod 'http://localhost:8000/research' -Method Post -ContentType 'application/json' -Body '{"question":"Bitcoin latest market news today"}'
```

The last two commands can use provider credits. GET routes do not.

## Files

Created:

- `app/agents/forecast_context_agent.py`, `market_research_agent.py`, `validation_agent.py`, `decision_agent.py`, `orchestrator.py`
- `app/agents/answer_guard.py`, `app/space_server.py`
- `app/schemas/__init__.py`, `app/schemas/agent_models.py`
- `app/services/__init__.py`, `agent_config.py`, `llm_service.py`, `tavily_service.py`
- `app/analyst_api.py`
- `app/static/monitor/analyst.css`, `analyst.js`
- `tests/test_reliability.py`, `test_market_research.py`, `test_analyst_integration.py`, `test_analyst_hardening.py`
- `tests/test_conversational_analyst.py`, `tests/test_space_server.py`
- `AI_ANALYST.md`

Changed: `app/api.py`, `app/static/monitor/index.html`, `docker-compose.yml`, `Dockerfile`, `app/app_gradio.py`, `.env.example`, `.gitignore`. Only absent analyst configuration entries were appended to the local `.env`; existing key values were preserved. The Gradio heading and README Space metadata added by the user are retained.

Model artifacts, deployment registry, inference implementation, Gradio behavior, hourly scheduler, SQLite history and original forecast outputs are preserved. The existing monitoring CSS/JavaScript remain; the additive stylesheet makes the latest raw prediction neutral to distinguish it from qualified signals.

## Validation and maintenance

The conversational correction and Agents SDK integration pass **100 automated tests**. The real SDK Runner is exercised with mocked HTTP for strict typed output, one-call limits, disabled tracing, secret redaction, and provider failures. Live checks cover greetings, horizon-specific model questions, missing-reliability explanations, follow-ups, probability definitions and a Tavily-backed news summary using gpt-4.1. One generated news answer was rejected by validation; a subsequent live answer passed. Rejected answers remain visibly labeled as fallbacks. Replies now identify successful OpenAI responses separately from fallback responses in the UI. The single-port Space layout was tested locally as UID 1000: dashboard, context, health, chat and Gradio routes are reachable through port 7860. Remote Hugging Face deployment is not part of this verification.

Deployment verified on 2026-09-08:

- **80 automated tests passed**, including 23 existing monitoring tests and 57 analyst/research/reliability tests. The actual installed OpenAI SDK parser was exercised with an offline HTTP mock.
- Docker built successfully; API, Gradio UI and hourly monitor are all healthy.
- Live OpenAI `gpt-4.1` returned valid structured selections for a 6-hour model question and a current-news summary.
- Live Tavily ran three focused searches and retained nine dated sources. Coverage is honestly marked partial; evidence quality is moderate. The subsequent news chat reused cached research.
- The model-only chat did not call Tavily. The news chat preserved UNKNOWN signals and causal attribution remained disabled.
- Original and new endpoints, the dashboard and all four CSS/JavaScript assets returned successfully. The Gradio homepage returned HTTP 200.
- All 12 preexisting forecast records retained their immutable values after restart. No model artifacts were included in the 24-file installation.
- Provider keys are absent from the UI/worker environment, `.env` is absent from the Docker image filesystem, and checked responses did not expose configured secrets.

The production context was fresh at verification, with all three signals UNKNOWN because validated reliability is not supplied by these models.

The tests use isolated fixtures and mock providers; they do not insert synthetic forecasts into production. Coverage includes the requested 14 reliability, provider-failure and historical-integrity cases, plus API integration, provider payloads, cache concurrency, secret handling, and all original persistence/scheduler/monitoring checks.

Run the complete suite against the built image without network access:

```powershell
docker run --rm --network none --read-only --tmpfs /tmp --mount "type=bind,source=$($PWD.Path),target=/app,readonly" btc_prod-btc-api python -m unittest discover -s tests -v
```

New JavaScript also passes `node --check`. Served page/assets and API responses are checked during deployment. The environment currently has no connected browser or Computer Use helper, so visual and interactive browser verification could not be performed. The tested container has OpenAI SDK 3.8.0 and Pydantic 2.13.5; the original model dependency file is retained, and `requirements-agents.txt` adds the pinned framework. For a non-Docker installation run `pip install -r requirements.txt -r requirements-agents.txt`.

Review fixes included inconsistent horizon identifiers between snapshot and chat, partial search results being mislabeled as complete, missed market-movement research questions, inconsistent publisher quality lists, future historical cutoffs reaching the provider, and unsafe redaction of serialized JSON. Analyst request-validation errors do not echo rejected user inputs. Invalid structured selections fall back to deterministic verified facts.

Optional limits: `NEWS_MAX_AGE_HOURS=72`, `CONTEXT_MAX_AGE_MINUTES=90`, `TAVILY_TIMEOUT_SECONDS=12`, `OPENAI_TIMEOUT_SECONDS=45`. All configuration is read server-side. The research cache and activity log are process-local and clear on API recreation; durable monitoring history remains in `runtime/monitoring.sqlite3`.

Provider references: [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search).
