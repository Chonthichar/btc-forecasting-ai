MARKET_CORE_V1 = [
    "open","high","low","close","volume","quote_volume","num_trades",
    "log_return","log_return_lag_1h","log_return_lag_6h","log_return_lag_24h",
    "sma_24","sma_168","macd",
    "rsi_14","momentum_6h","momentum_24h",
    "volatility_24h","volatility_168h","atr_pct_14","bb_width_20",
    "volume_zscore_24","taker_buy_ratio","buy_sell_imbalance",
    "range_pct","body_pct","close_location",
]

SENTIMENT_CORE_V1 = [
    "article_count","sentiment_score","positive_share","negative_share",
    "neutral_share","mean_confidence","has_news",
    "sentiment_score_lag_1h","sentiment_score_lag_6h",
    "sentiment_score_lag_24h",
]

# Extended causal sentiment representation for delayed-news-effect experiments.
# These features are created only from current/past hourly sentiment observations.
SENTIMENT_EXTENDED_V2 = SENTIMENT_CORE_V1 + [
    "sentiment_mean_3h",
    "sentiment_mean_6h",
    "sentiment_mean_12h",
    "sentiment_mean_24h",
    "sentiment_change_3h",
    "sentiment_change_6h",
    "article_count_6h",
    "article_count_24h",
    "positive_share_6h",
    "negative_share_6h",
    "sentiment_volatility_24h",
]

FEATURE_SETS = {
    "market_core_v1": MARKET_CORE_V1,
    "sentiment_core_v1": SENTIMENT_CORE_V1,
    "sentiment_extended_v2": SENTIMENT_EXTENDED_V2,
}
