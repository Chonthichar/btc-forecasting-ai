---
title: Crypto Forecasting AI
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
# BTC Reproducible Multimodal Forecasting Pipeline — Sentiment Ablation V4

This project tests whether BTC news sentiment has **standalone** and **incremental** predictive value at **1h, 6h, and 24h** horizons, and adds a separate daily paper-style experiment for comparison with published Bitcoin-sentiment studies.

## Hourly ablation matrix

Every horizon now compares three feature conditions:

- **Market only:** 27 `market_core_v1` features
- **Sentiment only:** 21 `sentiment_extended_v2` features
- **Market + sentiment:** 27 + 21 = 48 features

This distinction is important:

- `sentiment_only` asks whether sentiment contains predictive signal by itself.
- `market_plus_sentiment` asks whether sentiment adds value beyond historical market information.

The 21 sentiment features include article activity, polarity shares, confidence, no-news indicator, causal lags, rolling means, sentiment changes, rolling article counts, rolling polarity shares, and sentiment volatility. All are constructed only from information available at or before time `t`.

## Forecast horizons

| Config | Target | CV/final purge gap |
|---|---|---:|
| `config/btc_1h.yaml` | `target_direction_1h` | 1 hour |
| `config/btc_6h.yaml` | `target_direction_6h` | 6 hours |
| `config/btc_24h.yaml` | `target_direction_24h` | 24 hours |

## Deep training

LSTM, GRU, and VAE-Transformer use:

- maximum epochs: **200**
- early stopping patience: **20**
- min delta: `0.0001`
- LSTM/GRU monitor: validation BCE loss
- VAE-Transformer monitor: validation total VAE loss

The 200 epochs are a ceiling, not a requirement. Training stops earlier when validation loss does not improve for 20 epochs.

For the sentiment-only VAE-Transformer ablation, the variational encoder/decoder operates on the sentiment sequence itself. For market+sentiment, the VAE reconstructs market features and the sentiment sequence enters through the separate GRU branch.

## Leakage-controlled hourly evaluation

1. Restrict to verified sentiment coverage.
2. Encode no-news hours as `has_news=0`, zero counts/shares/score.
3. Build causal sentiment lags/rolling features after coverage restriction.
4. Drop only transparent feature/target warm-up rows.
5. Reserve the newest 15% as final chronological holdout.
6. Purge `h` hours at every train/validation and development/test boundary for an `h`-hour target.
7. Fit scalers on training rows only.
8. Purged rows may provide historical sequence context but never training targets.
9. Select model/configuration from walk-forward CV only.
10. Retrain the selected deep model for the median CV-selected epoch count and evaluate the holdout once.

## Hourly models

Traditional baselines:

- Logistic Regression
- Random Forest
- HistGradientBoosting

Deep models:

- LSTM
- GRU
- VAE-Transformer V2

## Daily paper-style experiment

`run_daily_paper_style.py` constructs daily BTC OHLCV from the hourly market dataset and merges it with the existing daily FinBERT news-sentiment file.

It evaluates the same three feature sets:

- market only
- sentiment only
- market + sentiment

and predicts three targets:

1. **Next-day direction** — binary UP/DOWN classification
2. **Next-day log return** — regression
3. **Next-day close price** — regression

Models include Logistic/RF/HistGB for classification, Ridge/RF/HistGB for regression, and a GRU for both classification and regression. The GRU also uses max 200 epochs with patience 20 in CV, then the final holdout model is retrained for the median CV-selected best epoch count.

The daily experiment also reports:

- MAE, RMSE, MAPE and R² for regression
- directional accuracy derived from return/price predictions
- a clearly labeled `mape_accuracy_pct = 100 × (1 − MAPE)` for comparison with papers that call regression error an “accuracy”
- a **naive persistence baseline**: tomorrow's close = today's close
- a zero-return baseline

This makes it possible to distinguish genuinely useful directional forecasting from an apparently high price-level score caused mainly by BTC price persistence.

## Data sources in Drive

Hourly market:
`/content/drive/MyDrive/crypto historical 2 year hourly data/data/04_btc_training_ready.parquet`

Hourly sentiment:
`/content/drive/MyDrive/btc sentiment year data/data/04_btc_sentiment_hourly.parquet`

Daily sentiment:
`/content/drive/MyDrive/btc sentiment year data/data/05_btc_sentiment_daily.parquet`

## Colab setup

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!pip install -q pandas numpy pyarrow scikit-learn joblib pyyaml torch
%cd "/content/drive/MyDrive/BTC reproducible multimodal pipeline"
```

## Hourly sentiment-ablation runs

Validate all three horizons:

```bash
!python run_horizon_suite.py --stage validate --horizons 1 6 24
```

Run baseline ablations:

```bash
!python run_horizon_suite.py --stage baselines --horizons 1 6 24
```

Run all deep ablations with max 200 epochs + early stopping:

```bash
!python run_horizon_suite.py --stage deep --horizons 1 6 24
```

Or run one horizon at a time:

```bash
!python run_pipeline.py --config config/btc_1h.yaml --stage deep
!python run_pipeline.py --config config/btc_6h.yaml --stage deep
!python run_pipeline.py --config config/btc_24h.yaml --stage deep
```

New V4 output folders are isolated from earlier experiments:

- `outputs/BTC_1h_target1h_sentiment_v4_ablation_200ep/`
- `outputs/BTC_1h_target6h_sentiment_v4_ablation_200ep/`
- `outputs/BTC_1h_target24h_sentiment_v4_ablation_200ep/`

## Daily paper-style runs

Validate daily alignment first:

```bash
!python run_daily_paper_style.py --config config/btc_daily_paper_style.yaml --stage validate
```

Run direction classification only:

```bash
!python run_daily_paper_style.py --config config/btc_daily_paper_style.yaml --stage classify
```

Run return/price regression only:

```bash
!python run_daily_paper_style.py --config config/btc_daily_paper_style.yaml --stage regress
```

Run everything:

```bash
!python run_daily_paper_style.py --config config/btc_daily_paper_style.yaml --stage all
```

Daily outputs are saved under:

`outputs/BTC_daily_paper_style_sentiment_v1/`

## Main research questions

1. Does BTC news sentiment alone predict future direction better than chance?
2. Does adding sentiment improve on market-only information?
3. Does sentiment value increase from 1h to 6h to 24h?
4. Does sentiment help next-day **return/direction**, or only make next-day **price-level** regression look accurate?
5. Does a sophisticated model outperform naive persistence on the daily price task?
