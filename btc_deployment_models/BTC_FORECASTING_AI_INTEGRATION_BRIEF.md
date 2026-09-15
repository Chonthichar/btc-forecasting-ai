# BTC Forecasting App Integration Brief

## Purpose

This document is the source-of-truth brief for integrating the final BTC forecasting models into the local `btc-forecasting-ai` application.

The application must expose **six model outputs**:

- **Movement risk:** 1h, 6h, 24h
- **Direction:** 1h, 6h, 24h

The application should reuse the existing live Binance 1-hour market-data pipeline, load the frozen trained artifacts, compute features exactly as they were computed during training, and return deterministic model outputs to the frontend and LLM.

The LLM is only allowed to **explain** model outputs. It must not invent, alter, smooth, or replace numeric model predictions.

---

# 1. High-Level System

```text
Binance closed 1h candles
        │
        ▼
existing MarketDataAgent
        │
        ├── base market features
        │
        └── add exact Version-B stationary direction features
        │
        ▼
feature validation + exact feature ordering
        │
        ├───────────────┐
        ▼               ▼
risk models         direction models
1h / 6h / 24h      1h / 6h / 24h
LogisticRegression Transformer
        │               │
        └──────┬────────┘
               ▼
         unified forecast object
               │
        ┌──────┴──────┐
        ▼             ▼
      API/UI          LLM
                    explanation only
```

---

# 2. Existing Live Market Pipeline

The repository already contains:

```text
app/agents/market_agent.py
```

Use the existing `MarketDataAgent`.

Expected interface:

```python
MarketDataAgent(symbol="BTCUSDT", interval="1h")
market_agent.run(limit=500)
```

The current live agent returns closed Binance 1h candles and approximately 53 market columns, including all 27 risk features.

Important requirements:

- Use **closed candles only**
- Use UTC timestamps
- Never use the still-forming current hour
- Require an hourly-contiguous timeline
- Fetch enough history for:
  - 168h rolling features
  - 48h Transformer input sequence

A limit around `500` closed hourly candles is sufficient.

---

# 3. Final Local Model Folder Structure

Use this layout:

```text
btc-forecasting-ai/
│
├── app/
│   └── ...
│
├── models/
│   │
│   ├── risk/
│   │   ├── risk_1h_logistic.joblib
│   │   ├── risk_1h_scaler.joblib
│   │   ├── risk_6h_logistic.joblib
│   │   ├── risk_6h_scaler.joblib
│   │   ├── risk_24h_logistic.joblib
│   │   ├── risk_24h_scaler.joblib
│   │   ├── risk_feature_list.csv
│   │   ├── provisional_risk_band_thresholds.json
│   │   └── risk_band_empirical_results.csv
│   │
│   └── direction/
│       ├── direction_1h_transformer.pt
│       ├── direction_1h_scaler.joblib
│       ├── direction_6h_transformer.pt
│       ├── direction_6h_scaler.joblib
│       ├── direction_24h_transformer.pt
│       ├── direction_24h_scaler.joblib
│       └── direction_stationary_feature_list.csv
│
└── ...
```

Do not retrain any model.

---

# 4. Movement-Risk Task Definition

## What the risk models predict

The risk models do **not** predict direction.

They predict whether BTC will make a sufficiently large move in **either direction**.

For horizon `H`:

> Will BTC touch either **+2.5%** or **-2.5%** from the current candle close at any point during the next `H` hours?

The event definition is:

```python
upper_boundary = close[t] * 1.025
lower_boundary = close[t] * 0.975

risk = 1 if (
    max(high[t+1 : t+H]) >= upper_boundary
    or
    min(low[t+1 : t+H]) <= lower_boundary
)

risk = 0 otherwise
```

Horizons:

```text
1h  -> inspect candle t+1
6h  -> inspect t+1 ... t+6
24h -> inspect t+1 ... t+24
```

Important:

```text
+2.5% touch -> risk = 1
-2.5% touch -> risk = 1
```

Direction is ignored.

This is a **barrier-touch / large-movement event**, not:

- crash probability
- downside probability
- expected return
- future close ±2.5%
- volatility forecast

The future **high** and **low** are what define the event.

---

# 5. Risk Model Specification

Each risk horizon uses a separate model.

```text
Model: Logistic Regression
Penalty: L2
C: 0.001
Scaling: StandardScaler fit on training only
Input features: 27 BTC market features
Horizons: 1h, 6h, 24h
```

Load the exact scaler matching each horizon before inference.

The raw model output should be called:

```text
model_score
```

Do not claim that it is a perfectly calibrated probability.

---

# 6. Exact 27 Risk Features

The order must come from:

```text
models/risk/risk_feature_list.csv
```

The expected feature names are:

```python
RISK_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "log_return",
    "log_return_lag_1h",
    "log_return_lag_6h",
    "log_return_lag_24h",
    "sma_24",
    "sma_168",
    "macd",
    "rsi_14",
    "momentum_6h",
    "momentum_24h",
    "volatility_24h",
    "volatility_168h",
    "atr_pct_14",
    "bb_width_20",
    "volume_zscore_24",
    "taker_buy_ratio",
    "buy_sell_imbalance",
    "range_pct",
    "body_pct",
    "close_location",
]
```

Never hard-code a different order if the CSV exists. Load the CSV and use its exact stored order.

---

# 7. Risk-Band Interpretation

Use:

```text
models/risk/provisional_risk_band_thresholds.json
models/risk/risk_band_empirical_results.csv
```

For each risk horizon return:

- model score
- risk band
- historical empirical event rate for that band

Risk levels:

```text
LOW
TYPICAL
ELEVATED
HIGH
```

The app should describe the historical event rate as:

> Historical observed event frequency for similar out-of-fold model-score bands.

Do not describe the band event rate as the model's exact probability.

---

# 8. Direction Task Definition

The direction models predict whether BTC closes higher than the current close after the requested horizon.

Targets:

```python
direction_1h  = 1 if close[t+1]  > close[t] else 0
direction_6h  = 1 if close[t+6]  > close[t] else 0
direction_24h = 1 if close[t+24] > close[t] else 0
```

All three direction horizons must remain available in the app:

```text
1h
6h
24h
```

Do not remove 6h or 24h.

The 1h model validated better than the longer horizons, but all three should still be displayed.

---

# 9. Direction Model Specification

Each direction horizon uses a separate saved Transformer and scaler.

Shared setup:

```text
Input sequence: 48 consecutive closed 1h candles
Features per timestep: 26
d_model: 64
attention heads: 4
Transformer layers: 2
feed-forward dimension: 128
dropout: 0.15
activation: GELU
```

Conceptual architecture:

```text
26 features
   │
Linear 26 -> 64
   │
GELU
   │
LayerNorm
   │
learned positional embedding
   │
TransformerEncoder
   │
last sequence token
   │
LayerNorm
   │
Dropout
   │
Linear -> binary logit
```

Use the architecture and metadata in the saved checkpoints as the source of truth when loading models.

Output:

```python
up_score = sigmoid(logit)
down_score = 1 - up_score

label = "UP" if up_score >= 0.50 else "DOWN"
```

Call these **model scores**, not guaranteed calibrated probabilities.

---

# 10. Exact 26 Direction Features

The exact order must come from:

```text
models/direction/direction_stationary_feature_list.csv
```

Expected features:

```python
DIRECTION_FEATURES = [
    "open_close_log",
    "high_close_log",
    "low_close_log",

    "volume_logchg_1h",
    "quote_volume_logchg_1h",
    "num_trades_logchg_1h",

    "log_return",
    "log_return_lag_1h",
    "log_return_lag_6h",
    "log_return_lag_24h",

    "close_sma24_log",
    "close_sma168_log",
    "macd_pct",

    "rsi_14",
    "momentum_6h",
    "momentum_24h",
    "volatility_24h",
    "volatility_168h",
    "atr_pct_14",
    "bb_width_20",
    "volume_zscore_24",
    "taker_buy_ratio",
    "buy_sell_imbalance",
    "range_pct",
    "body_pct",
    "close_location",
]
```

---

# 11. Exact 9 Derived Stationary Direction Features

These 9 features are not currently computed by the base repo market pipeline.

They must be added exactly as they were defined during training.

Use:

```python
EPS = 1e-12
```

Exact formulas:

```python
df["open_close_log"] = np.log(
    df["open"].clip(lower=EPS)
    /
    df["close"].clip(lower=EPS)
)

df["high_close_log"] = np.log(
    df["high"].clip(lower=EPS)
    /
    df["close"].clip(lower=EPS)
)

df["low_close_log"] = np.log(
    df["low"].clip(lower=EPS)
    /
    df["close"].clip(lower=EPS)
)

df["volume_logchg_1h"] = (
    np.log1p(
        df["volume"].clip(lower=0)
    )
    .diff()
)

df["quote_volume_logchg_1h"] = (
    np.log1p(
        df["quote_volume"].clip(lower=0)
    )
    .diff()
)

df["num_trades_logchg_1h"] = (
    np.log1p(
        df["num_trades"].clip(lower=0)
    )
    .diff()
)

df["close_sma24_log"] = np.log(
    df["close"].clip(lower=EPS)
    /
    df["sma_24"].clip(lower=EPS)
)

df["close_sma168_log"] = np.log(
    df["close"].clip(lower=EPS)
    /
    df["sma_168"].clip(lower=EPS)
)

df["macd_pct"] = (
    df["macd"]
    /
    df["close"].clip(lower=EPS)
)
```

These formulas must not be approximated or renamed.

Feature-engineering order:

```text
1. Fetch/build base market indicators.
2. Ensure sma_24, sma_168, macd and raw activity columns exist.
3. Compute the 9 exact Version-B derived features.
4. Load exact 26-feature order from CSV.
5. Validate all features are finite.
6. Take the latest 48 consecutive valid hourly rows.
7. Scale using the horizon-specific saved scaler.
8. Run the horizon-specific Transformer.
```

---

# 12. Live Data Validation

Before inference, enforce:

```text
- no duplicate timestamps
- hourly spacing exactly 1 hour
- latest candle is closed
- all required risk features exist
- all required direction features exist
- latest risk feature vector contains only finite values
- latest direction sequence contains exactly 48 rows
- 48h direction sequence is hourly-contiguous
- all 26 direction values are finite
```

If any requirement fails, return a data-quality error instead of silently predicting.

---

# 13. Unified Inference Output

The backend should expose one clean object similar to:

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "forecast_candle_utc": "2026-09-11T17:00:00+00:00",
  "btc_close": 77534.0,

  "movement_risk": {
    "1h": {
      "model_score": 0.0065,
      "risk_level": "TYPICAL",
      "historical_event_rate": 0.0030,
      "event_definition": "BTC touches +2.5% or -2.5% from current close within the next 1 hour"
    },
    "6h": {
      "model_score": 0.0952,
      "risk_level": "TYPICAL",
      "historical_event_rate": 0.0936,
      "event_definition": "BTC touches +2.5% or -2.5% from current close at any point within the next 6 hours"
    },
    "24h": {
      "model_score": 0.4876,
      "risk_level": "TYPICAL",
      "historical_event_rate": 0.4582,
      "event_definition": "BTC touches +2.5% or -2.5% from current close at any point within the next 24 hours"
    }
  },

  "direction": {
    "1h": {
      "label": "UP",
      "up_score": 0.5631,
      "down_score": 0.4369
    },
    "6h": {
      "label": "UP_OR_DOWN",
      "up_score": 0.0,
      "down_score": 0.0
    },
    "24h": {
      "label": "UP_OR_DOWN",
      "up_score": 0.0,
      "down_score": 0.0
    }
  }
}
```

The 6h and 24h values above are placeholders only; use actual model inference.

---

# 14. UI Requirements

The app should clearly separate:

## Direction

```text
1h  UP / DOWN
6h  UP / DOWN
24h UP / DOWN
```

For each horizon show:

- direction label
- UP model score
- DOWN model score
- optional confidence wording

Do not represent these scores as guaranteed probabilities.

The UI may indicate that:

```text
1h direction had the strongest historical validation.
6h and 24h direction models were weaker and should be interpreted cautiously.
```

Do not hide the longer-horizon models.

## Large-Movement Risk

For each horizon show:

- 1h / 6h / 24h
- model score
- LOW / TYPICAL / ELEVATED / HIGH
- historical event rate
- short explanation of the ±2.5% barrier event

Recommended label:

```text
Large Movement Risk
```

Avoid vague labels such as only `Risk`.

---

# 15. Historical Direction Validation Context

These are research diagnostics and should not be presented as user-facing forecast probabilities.

Approximate mean walk-forward ROC-AUC:

```text
1h  : 0.5483
6h  : 0.5243
24h : 0.5188
```

Interpretation:

- 1h was strongest
- 6h was weaker
- 24h was weaker
- all three remain in the application by product decision

The app may use this information to qualify confidence, but must not modify the actual scores based on these validation metrics.

---

# 16. LLM Rules

The LLM must consume deterministic backend results.

Allowed:

```text
"The 1h direction model currently leans UP."
"The 24h large-movement risk is within its typical historical range."
"The short-term direction signal is stronger than the longer-horizon research signals."
```

Not allowed:

```text
inventing a new probability
changing model scores
averaging scores without an explicit algorithm
claiming certainty
turning large-movement risk into crash probability
using news or intuition to overwrite model outputs
```

The LLM can add context and explanation, but the numeric model outputs remain authoritative.

---

# 17. Prospective Forward Testing

Every live forecast should be logged before its future outcome is known.

Recommended file/database fields:

```text
generated_at_utc
forecast_candle_utc
btc_close

risk_1h_score
risk_1h_band

risk_6h_score
risk_6h_band

risk_24h_score
risk_24h_band

direction_1h
direction_1h_up_score

direction_6h
direction_6h_up_score

direction_24h
direction_24h_up_score
```

Do not duplicate the same forecast candle if the pipeline is rerun.

Future evaluation can later attach realized outcomes.

This forward-test log is important for the thesis because the prediction is stored before the outcome occurs.

---

# 18. Current Working Realtime Test

A successful prospective inference was already produced from the closed BTC candle:

```text
Forecast candle:
2026-09-11 17:00 UTC

BTC close:
$77,534.00
```

Observed model outputs in that test:

```text
Movement risk

1h
model score: 0.65%
risk level: TYPICAL
historical event rate: 0.30%

6h
model score: 9.52%
risk level: TYPICAL
historical event rate: 9.36%

24h
model score: 48.76%
risk level: TYPICAL
historical event rate: 45.82%
```

Direction test at that time:

```text
1h prediction: UP
UP score: 56.31%
DOWN score: 43.69%
```

That test proved that:

```text
- Binance closed-candle data works
- all 27 risk features are present
- all 26 direction features can be built
- the latest risk row is finite
- a valid 48h direction sequence can be constructed
- the frozen artifacts can run prospective inference
```

The local application now needs to reproduce the same inference logic.

---

# 19. Implementation Tasks for the Coding AI

Implement the following without redesigning the whole project:

1. Inspect the current repository before changing anything.
2. Reuse `app/agents/market_agent.py`.
3. Add the exact 9 Version-B stationary features to the market/inference feature pipeline.
4. Add a clean local model-loading layer for `models/risk/` and `models/direction/`.
5. Load all three risk models and matching scalers.
6. Load all three direction models and matching scalers.
7. Always use feature-order CSVs rather than relying on DataFrame column order.
8. Validate finite values and sequence continuity before inference.
9. Run:
   - risk 1h
   - risk 6h
   - risk 24h
   - direction 1h
   - direction 6h
   - direction 24h
10. Apply risk-band interpretation from the saved JSON/CSV files.
11. Return a unified backend forecast object.
12. Update API endpoints so the frontend can access all six outputs.
13. Update the frontend to display all six predictions cleanly.
14. Update the LLM context so it can explain all six outputs.
15. Keep the LLM read-only with respect to numeric predictions.
16. Add prospective logging for all six outputs.
17. Remove or stop using outdated old-model paths where they conflict with the new deployment models.
18. Do not retrain models.
19. Do not change model architecture or feature formulas.
20. Keep changes minimal and compatible with the existing project structure.

---

# 20. Important Do-Not-Do List

Do **not**:

```text
- retrain the models
- substitute different indicators
- reorder model inputs
- use a single scaler across horizons
- use unfinished/current candles
- approximate the 9 stationary formulas
- call risk score "crash probability"
- call direction score a guaranteed probability
- remove 6h or 24h direction
- let the LLM generate numeric forecasts
- silently make predictions when required features are missing
```

---

# 21. Suggested Code Organization

A clean minimal structure could be:

```text
app/
├── agents/
│   └── market_agent.py
│
├── forecasting/
│   ├── feature_builder.py
│   ├── risk_predictor.py
│   ├── direction_predictor.py
│   ├── model_loader.py
│   └── realtime_forecast.py
│
├── api/
│   └── ...
│
└── ...
```

Responsibilities:

```text
feature_builder.py
    exact derived stationary features
    feature validation
    feature ordering

model_loader.py
    load models/scalers/configs once

risk_predictor.py
    1h/6h/24h risk inference
    risk-band interpretation

direction_predictor.py
    1h/6h/24h Transformer inference

realtime_forecast.py
    orchestrate market data + all six predictions
    build final forecast object
    log prospective prediction
```

Do not force this exact file structure if the repository already has a clean equivalent. Prefer the smallest maintainable changes.

---

# 22. Definition of Done

The integration is complete when a local run can:

```text
1. fetch the latest closed BTCUSDT 1h data
2. validate the live timeline
3. build all required features
4. produce risk 1h / 6h / 24h
5. produce direction 1h / 6h / 24h
6. expose all six results through the backend/API
7. display all six in the frontend
8. provide them to the LLM for explanation
9. log the forecast prospectively
10. run without Google Drive or Colab
```

Final product behavior:

```text
RISK
1h   model score + band + historical rate
6h   model score + band + historical rate
24h  model score + band + historical rate

DIRECTION
1h   UP/DOWN + scores
6h   UP/DOWN + scores
24h  UP/DOWN + scores
```

This document should be treated as the integration specification for the current frozen deployment models.
