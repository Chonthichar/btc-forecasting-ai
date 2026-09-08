#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export BTC_PROJECT_ROOT="$ROOT"
export BTC_SENTIMENT_SCRIPT="$ROOT/sentiment_pipeline/btc_sentiment_agents.py"
python app/app_gradio.py
