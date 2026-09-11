# Three-agent reasoning upgrade

Implemented September 9, 2026. This is an incremental upgrade of the existing Production First application. No retraining, model replacement, scheduler replacement, endpoint rename, database replacement, or dashboard redesign was performed.

## 1. Architecture

```text
Existing deterministic layer
Market Monitor -> trained Forecast Engine -> Reliability Gate -> Trigger Engine
                                                                  |
                                               authorized research only
                                                                  v
BTC Research Agent [OpenAI SDK, one bounded Tavily tool batch]
    -> existing deterministic source/context validation
    -> BTC Review Agent [separate OpenAI SDK run, no tools]
    -> existing deterministic validation again
    -> BTC Decision Agent [existing OpenAI SDK agent, no tools]
    -> deterministic market-decision post-validation
    -> existing SQLite events / API responses
```

Internal-only chat calls the Decision Agent with stored context. An ordinary scheduled scan with no trigger calls none of the three reasoning agents. Browser polling calls neither Tavily nor OpenAI.

## 2. Changed and created files

Changed:

- `app/agents/orchestrator.py`: routes permitted research through the new workflow; adds structured decisions and review status to existing responses.
- `app/agents/decision_agent.py`: preserves the existing three-value return contract by default; optional `structured=True` adds the post-validated decision.
- `app/agents/market_research_agent.py`: retains the existing Tavily adapter/cache; supports bounded agent-selected queries and provider/cache/source counters.
- `app/schemas/agent_models.py`: additive structured research/review/audit/decision schemas and optional response fields.
- `app/services/llm_service.py`: existing Decision Agent instructions now describe independently reviewed evidence and structured decisions.
- `app/services/market_monitoring.py`: retains triggering/cooldown logic and uses the shared research workflow after an event is claimed.
- `app/services/monitoring_events_store.py`: historical research records no longer appear as the latest live evidence; existing tables remain unchanged.
- `app/services/agent_config.py`: exposes the three reasoning roles and query cap through existing public configuration.
- `app/static/monitor/analyst.js`: only the POST timeout changes from 120 to 240 seconds to accommodate the bounded multi-agent chain. GET timeout and polling remain unchanged.
- `scripts/check_space_bundle.py`: requires the three new service modules at build time.
- `tests/test_analyst_hardening.py`: updates the strict schema expectation for the additive decision field.

Created:

- `app/services/reasoning_agents.py`: actual SDK Research and Review agent definitions.
- `app/services/research_workflow.py`: permitted research execution, semantic review, deterministic validation and existing-store audit integration.
- `app/services/market_decision.py`: authoritative decision/reliability post-validation.
- `tests/test_three_reasoning_agents.py`: focused workflow, SDK, API, PWA, cost and reliability tests.

The deployed patch consists of these 15 source/test files. This report and verification helpers remain local. Replaced production source files are preserved in the Local Production bundle's `multiagent-source-backup-20260909` directory.

## 3. Components that remain deterministic

Market fetching and scheduled updates; trained forecast inference and probabilities; forecast reliability/gating; trigger detection, cooldown and fingerprints; Tavily request limits/cache; URL/date/age/duplicate checks; delayed forecast settlement; final structured decision enforcement; and persistence.

The existing internal sentiment feature pipeline is unchanged. It continues updating according to its own schedule. Its news inputs are separate from explanatory Tavily research.

## 4. Actual OpenAI agents

There are exactly three reasoning roles, created with the installed OpenAI Agents SDK:

| SDK name | Role | Tools | Maximum SDK turns per invocation |
| --- | --- | --- | --- |
| `BTC Research Agent` | Chooses focused searches or recent cached evidence | `search_tavily` | 2 |
| `BTC Review Agent` | Independently assesses every supplied source ID | None | 1 |
| `BTC Decision Agent` | Writes the grounded explanation and proposes structured interpretation | None | 1 |

All use the existing `OPENAI_MODEL` configuration. The Space is configured for `gpt-5.6`; local settings are preserved and may differ. SDK dependency versions were not changed. There is no OpenAI Forecast Agent and no LLM-generated price probability.

The Research Agent's tool permits one batch with at most three short neutral BTC queries. It cannot start another batch even if the model asks repeatedly. The reviewer receives all retrieved sources that passed Python checks, including valid opposing evidence the researcher did not select. It can reject weak evidence, revise stance classifications, flag contradictions and request more research. Such a request is recorded; automatic follow-up searches are disabled to keep the budget bounded.

## 5. Exact Tavily call path

`ResearchWorkflow.run` defines the bounded `search(queries)` callback. `ReasoningAgents._run` exposes it as the Research Agent's `search_tavily` SDK function tool.

The callback uses the existing `MarketResearchAgent.run(..., queries=...)` -> `MarketResearchAgent._search(...)` -> `TavilyService.search(...)`. Only `TavilyService.search` sends the HTTP POST to `https://api.tavily.com/search`.

`MarketResearchAgent` retains its legacy class name and standalone Python adapter interface for compatibility; it is not a fourth OpenAI reasoning agent. Production callers use `ResearchWorkflow` around that adapter. No second Tavily HTTP client implementation was added.

## 6. When research is allowed

- The existing deterministic monitoring Trigger Engine fires and its event claim passes cooldown/deduplication.
- The user presses Research (`POST /research`).
- A chat question genuinely asks for external/current events, such as â€œWhy is BTC dropping right now?â€ or â€œWhat external events might explain today's move?â€

Explaining stored forecasts, identifying their models, explaining probability/reliability, and comparing stored horizons do not call Tavily. Explicit fresh research bypasses the cache as before; it cannot silently return old evidence instead of searching.

Existing defaults remain `RESEARCH_COOLDOWN_MINUTES=30` and `NEWS_CACHE_MINUTES=10`. Repeated automatic events within cooldown do not repeat any reasoning agent. Per-query caching and leases remain in the existing SQLite store. Agent-selected queries remain subject to a hard three-query/one-batch cap and neutral-query validation.

## 7. Exact OpenAI call paths

- `ReasoningAgents.run("Research", ...)` -> `ReasoningAgents._run(...)` -> SDK `Runner.run` with `BTC Research Agent`.
- `ReasoningAgents.run("Review", ...)` -> `ReasoningAgents._run(...)` -> SDK `Runner.run` with `BTC Review Agent`.
- `DecisionAgent.run(...)` -> `LLMService.plan(...)` -> `LLMService._run_agent(...)` -> SDK `Runner.run` with `BTC Decision Agent`.

These use `AsyncOpenAI`, `OpenAIResponsesModel`, explicit provider endpoint, structured output schemas, disabled tracing/storage and zero provider retries. A complete external workflow has at most four model turns: up to two for Research, one Review, one Decision. Review is skipped when no source survives deterministic validation. Automatic Decision is skipped without useful validated evidence. No live OpenAI/Tavily requests were made during tests.

## 8. Decision and reliability safeguards

`enforce_decision` recomputes the allowed result from the checked forecasts and reviewed evidence. It ignores invented reliability, source IDs, agreement, strength, risk text and stance in an LLM proposal.

- Unknown/unqualified requested horizons yield `NO_RELIABLE_VIEW` and `NO_RELIABLE_SIGNAL`. Bullish or bearish research cannot repair them.
- Conflicting qualified horizons can yield `MIXED`.
- For a broad three-horizon question, a proposed `6-24h` scope is allowed only when both longer horizons have qualified agreeing signals. An opposing or unqualified 1h view is explicitly noted. A specific 1h request cannot be silently changed to a longer horizon.
- Opposing news can add uncertainty; it cannot reverse an ML signal into a different reliable forecast.
- Only semantically reviewed, timestamp-verified primary/established evidence contributes to the structured directional research view.
- Strength is conservatively capped at `MODERATE`; `HIGH` is accepted in an LLM proposal schema but not promoted by the current post-validator. Strength is not a reliability probability.
- Existing prose/fact/source guards still validate the generated answer. LLM-proposed explanation/risk text is not copied directly into trusted fields.
- Historical source cutoffs, age limits, required fields and URL deduplication remain authoritative before and after semantic review.

## 9. Compatibility, persistence and observability

All existing API routes remain. Additions include `market_decision` on `/context`, `/evidence`, `/research` and `/chat`, plus optional `review` and `workflow` metadata. Existing research/evidence/forecast fields remain available. Old stored research JSON loads with defaults for the new fields. GET decisions are derived from current checked state without running an LLM.

Each external workflow records timestamp, trigger reason, user-request flag, mode, actual provider-call/cache counts, queries, returned/accepted/rejected source counts, Research/Review/Decision execution flags, final stance/strength/signal states and sanitized errors. The existing `research_events.result` JSON stores these records; scans retain their existing event references. No new database or scheduler was created.

Dashboard layout, manifest, icons, mobile project, FastAPI routes, monitoring worker/service/store, model registry, trained weights and inference files are preserved. Thirty-two selected model/inference/scheduler/API/PWA files were verified byte-for-byte unchanged. The single browser timeout adjustment accommodates the extra reasoning turns.

## 10. Validation

**157 tests passed**, with OpenAI and Tavily mocked and Docker networking disabled. This includes the original suite and 23 focused new tests covering all 14 requested scenarios, plus tool budgets, biased queries, historical cutoffs, malformed reviewer IDs, failure handling, fresh/cache behavior, persisted audits and horizon scope enforcement.

Tests exercise the actual OpenAI Agents SDK with mocked HTTP responses, including a real Research function-tool turn followed by structured output and an independent tool-free Review run. Separate checks verify all three actual SDK agent names and tool access. Docker build/preflight and JavaScript syntax checks pass. Pre-existing event-loop ResourceWarnings remain in test output.

Hosted verification results are appended after deployment below. No paid live research is needed for deployment checks; those use read-only endpoints.

## 11. Configuration and maintenance

No new secrets, model configuration or database migration are required. Continue using the existing OpenAI/Tavily secrets and private monitoring backup settings. The local deployment token is separate from the Space's restricted backup token.

Source is installed in:
`C:\Users\Windows PC\Downloads\BTC_Forecasting_Production_First_Bundle\btc_prod`

For local Docker:

```powershell
Set-Location 'C:\Users\Windows PC\Downloads\BTC_Forecasting_Production_First_Bundle\btc_prod'
docker compose up -d --build btc-api btc-monitor
Invoke-RestMethod 'http://localhost:8000/agents/status'
Invoke-RestMethod 'http://localhost:8000/context'
```

Read-only hosted checks:

```powershell
Invoke-RestMethod 'https://chonthichar-crypto-forecasting-ai.hf.space/agents/status'
Invoke-RestMethod 'https://chonthichar-crypto-forecasting-ai.hf.space/context'
Invoke-RestMethod 'https://chonthichar-crypto-forecasting-ai.hf.space/monitoring/activity'
```

The first endpoint should list Research, Review and Decision under `configuration.reasoning_agents`. Opening or polling these endpoints does not invoke them. Use the dashboard's explicit Research action when you actually want current external research; that action consumes provider resources.

## 12. Existing limitations intentionally retained

- **Models:** 1h is VAE-Transformer (`market_only`); **6h remains GRU** (`market_plus_sentiment`); 24h is VAE-Transformer (`market_plus_sentiment`). Only those three trained artifacts are present. The existing registry/inference interface supports VAE-Transformer per horizon, but switching 6h requires a compatible validated artifact, scalers and configuration. No substitute was manufactured.
- Current snapshots lack validated per-prediction reliability. UNKNOWN remains valid behavior despite available raw probabilities or historical holdout results.
- Review is semantic assessment of supplied Tavily excerpts, not independent full-article factual verification. Multiple agents can share mistakes; agreement is not proof. Python guards constrain dates, IDs and reliability but cannot prove every news claim.
- A missing/failing Research or Review agent withholds unreviewed evidence. More research can be requested by the reviewer, but another pass requires a later permitted workflow.
- Backup cadence, possible crash loss since the last successful backup, free Space sleep behavior and historical scan retention remain as documented in `MONITORING_UPGRADE.md`.
- Provider latency may still cause a timeout, especially with non-default larger timeout settings. The UI reports failure rather than presenting unchecked output.
- No new browser visual review, live paid-agent evaluation, retraining, PWA redesign or mobile build was performed.

SDK references used: [agent definitions](https://developers.openai.com/api/docs/guides/agents/define-agents), [running agents](https://developers.openai.com/api/docs/guides/agents/running-agents).

## Deployment verification — 2026-09-09

Uploaded the minimal 15-file patch to Space commit `811c110f2305ac54c4784787608149ca7f724f0c`. The Docker build and bundle preflight passed. Hugging Face reports RUNNING, but application readiness is blocked: logs report `Monitoring history restore failed` and analyst endpoints return HTTP 503. Hosted functional verification is therefore incomplete.

Using the refreshed deployment credential, downloaded the private monitoring snapshot and verified SQLite integrity (`ok`) and all 72 pre-deployment forecast IDs. Forecast values match the pre-deployment API snapshot; only the serialized `feature_set` representation differs. No empty history was created or uploaded. The existing fail-closed restore protection remains enabled.

The Space's separate MONITOR_BACKUP_TOKEN value cannot be retrieved through the Hub API, and no separate backup token is present in the local production .env. Its access needs to be checked/replaced in Space Settings with a dedicated token that can read and write the private monitoring dataset. Do not publish tokens in repository files. After that, verify hosted readiness and history restoration before considering deployment complete.

### Follow-up verification — 2026-09-10

After `MONITOR_BACKUP_TOKEN` was replaced, the Space restored the private snapshot successfully. `/health` reports a live snapshot and worker, `/monitoring/activity` reports backup status `RESTORED`, and the forecast history is available. `/agents/status` lists the three OpenAI Agents SDK roles: Research, Review and Decision, with both OpenAI and Tavily configured.

Two minimal live Decision-Agent requests returned `llm_status: ok`, including a grounded model-specific answer. The earlier `invalid_output` event was a generated response rejected by deterministic validation, rather than an OpenAI connection failure. The dashboard now labels these cases separately.

A research edge case was also corrected: when the deterministic router authorizes research and no acceptable cached evidence exists, the Research Agent's first Tavily tool call is required. The existing one-batch and three-query limits still apply. Cached evidence continues to avoid Tavily. The complete network-disabled suite remains at 157 passing tests.

The follow-up patch is deployed at Space commit `1d764e7b2a164dd8f689dd1cb11cca3a8135a5ec`. Final hosted checks report `RUNNING`, health `ok`, worker alive, backup `RESTORED`, 46 history rows for each horizon, and the updated fallback labels. A post-deployment Decision-Agent request returned `llm_status: ok`. No live Tavily research workflow was invoked for verification.

### Market overview UI — 2026-09-10

Space commit `0a03c3d97c927dcfa810a8f73ee35d57ae3f00a8` adds a white responsive market overview, three asset selectors, a BTC saved-price/directional-probability chart with 1h, 6h and 24h controls, and a desktop right-side AI analyst. Bitcoin uses the existing production context and durable history APIs. Ethereum and Cardano are intentionally marked `Not configured` because no trained artifacts or production data pipelines exist for them. The chart explicitly avoids presenting a synthetic price target from direction-classification models. The visible activity strip now includes the Review Agent.

All 157 offline tests and JavaScript syntax checks pass. Hosted structural verification confirms the white PWA theme, asset and horizon controls, right-column analyst, healthy worker, restored backup and idle GET behavior. Browser-based pixel inspection was unavailable in the development session, so final visual review should be performed directly in the deployed Space.

### Simplified overview UI — 2026-09-10

Space commit `9892c7ac3a885d6cd8773932b2d143b7c5daf52d` changes only presentation. The visible BTC-specific sidebar and redundant Forecast Signals cards are removed, the page heading is now `Market overview`, the desktop analyst column is narrower, and `Live Market Evidence` is presented as a compact `News analysis` list. The evidence remains real validated workflow output; no API, model, agent, persistence or reliability behavior changed. Hosted checks report health `ok`, worker alive, and all agents idle during read-only verification.

### Chatbot verified-fallback UX — 2026-09-10

Space commit `2aa366a176733e88cc82e6b2f15c9572db87595d` removes the internal `The AI answer could not be verified` diagnostic from user-visible chat replies. If deterministic validation rejects a generated Decision Agent plan, the API keeps `llm_status: invalid_output` for monitoring and returns the existing grounded deterministic answer. The analyst labels that state `VERIFIED FALLBACK`; it continues to distinguish provider failures as `OpenAI unavailable · fallback shown`.

The validation policy remains strict: unsupported values, sources, confidence, reliability and causal claims are still rejected. The complete network-disabled suite passes with 157 tests. A live summary request returned `llm_status: ok`, and final hosted checks confirm the exact commit is `RUNNING`, health is `ok`, the forecast worker is alive, OpenAI and Tavily are configured, all three reasoning agents are idle, and the deployed JavaScript contains the new label without the old rejected-response label.

### Chat panel layout fix — 2026-09-10

The desktop analyst is now a bounded flex column: its header, agent strip, suggested questions and composer remain visible while only the conversation log shrinks and scrolls. Explicit dark-panel colors restore contrast for the analyst and welcome headings, and the Clear and Send buttons use the dark chat palette. Below 1180px the panel returns to normal document flow with automatic height. This is a stylesheet-only change; agent, model and API behavior are unchanged.

The follow-up layout at Space commit `1e7903cd5635640ac3a2be80b5829db90739b0fd` makes both the agent states and suggested prompts single-row horizontal strips, reduces the conversation minimum height, and compacts the header and composer. A headless live render at 1536 × 768 confirms the textarea and Send button remain visible inside the panel. The page uses the `chat-layout-v2` stylesheet URL, and hosted health remains `ok` with the forecast worker alive.

### Research sidebar restoration — 2026-09-10

Space commit `4a2b3992ffcbc27f933b4c896e86a4c5b81b520e` restores a compact light sidebar with Overview, News analysis, AI assistant, About the framework and Thesis prototype navigation. It also includes the research-purpose card, workspace identity, live API indicator, active-section tracking and a compact horizontal navigation mode below 900px. A live 1536 × 768 render confirmed the sidebar and complete chat composer fit together. Hosted health is `ok` and the forecast worker remains alive.
