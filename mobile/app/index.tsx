import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

import {
  ContextResponse,
  ForecastContext,
  getContext,
} from "../services/api";

export default function HomeScreen() {
  const [data, setData] = useState<ContextResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadData() {
    try {
      setError(null);

      const result = await getContext();

      setData(result);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load BTC market data."
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
        <Text style={styles.loading}>
          Loading BTC intelligence...
        </Text>
      </View>
    );
  }

  const context = data?.context;

  function renderForecast(
    horizon: "1" | "6" | "24"
  ) {
    const forecast: ForecastContext | undefined =
      context?.forecasts[horizon];

    if (!forecast) {
      return null;
    }

    const probabilityUp =
      (forecast.probability_up * 100).toFixed(1);

    const probabilityDown =
      (forecast.probability_down * 100).toFixed(1);

    const hasReliableSignal =
      forecast.signal_state === "UP" ||
      forecast.signal_state === "DOWN";

    return (
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Text style={styles.horizon}>
            {horizon}H FORECAST
          </Text>

          <Text
            style={
              hasReliableSignal
                ? styles.signal
                : styles.unknown
            }
          >
            {hasReliableSignal
              ? forecast.signal_state
              : "NO RELIABLE SIGNAL"}
          </Text>
        </View>

        <Text style={styles.rawDirection}>
          Raw model lean: {forecast.raw_direction}
        </Text>

        <View style={styles.row}>
          <Text style={styles.label}>Probability UP</Text>
          <Text style={styles.value}>{probabilityUp}%</Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Probability DOWN</Text>
          <Text style={styles.value}>{probabilityDown}%</Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Reliability</Text>
          <Text style={styles.value}>
            {forecast.validated_reliability !== null
              ? `${(
                  forecast.validated_reliability * 100
                ).toFixed(1)}%`
              : "Unavailable"}
          </Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Model</Text>
          <Text style={styles.value}>
            {forecast.model}
          </Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Features</Text>
          <Text style={styles.value}>
            {forecast.feature_set}
          </Text>
        </View>

        {!hasReliableSignal && (
          <Text style={styles.reason}>
            {forecast.reliability_reason}
          </Text>
        )}
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.container}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={() => {
            setRefreshing(true);
            loadData();
          }}
        />
      }
    >
      <Text style={styles.title}>
        ₿ BTC Forecasting AI
      </Text>

      <Text style={styles.subtitle}>
        Multi-horizon forecasting • AI market intelligence
      </Text>

      {error && (
        <View style={styles.errorBox}>
          <Text style={styles.errorText}>
            {error}
          </Text>
        </View>
      )}

      <View style={styles.marketCard}>
        <Text style={styles.marketLabel}>
          BTC PRICE
        </Text>

        <Text style={styles.price}>
          {context?.market.price
            ? `$${context.market.price.toLocaleString(
                undefined,
                {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                }
              )}`
            : "Unavailable"}
        </Text>

        <View style={styles.row}>
          <Text style={styles.label}>Market regime</Text>
          <Text style={styles.value}>
            {context?.market.regime ?? "Not provided"}
          </Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Sentiment score</Text>
          <Text style={styles.value}>
            {context?.sentiment.score !== null &&
            context?.sentiment.score !== undefined
              ? context.sentiment.score.toFixed(3)
              : "Unavailable"}
          </Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>News articles</Text>
          <Text style={styles.value}>
            {context?.sentiment.article_count ?? 0}
          </Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>24h volatility</Text>
          <Text style={styles.value}>
            {context?.market.volatility_24h !== null &&
            context?.market.volatility_24h !== undefined
              ? `${(
                  context.market.volatility_24h * 100
                ).toFixed(2)}%`
              : "Unavailable"}
          </Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>RSI 14</Text>
          <Text style={styles.value}>
            {context?.market.rsi_14?.toFixed(1) ??
              "Unavailable"}
          </Text>
        </View>
      </View>

      <Text style={styles.sectionTitle}>
        Forecast Signals
      </Text>

      {renderForecast("1")}
      {renderForecast("6")}
      {renderForecast("24")}

      <View style={styles.notice}>
        <Text style={styles.noticeTitle}>
          Reliability Notice
        </Text>

        <Text style={styles.noticeText}>
          {data?.validation.forecast_reason ??
            "No validated reliability information available."}
        </Text>
      </View>

      <Text style={styles.updated}>
        Last refresh:{" "}
        {context?.last_refresh ?? "Unavailable"}
      </Text>

      <Text style={styles.disclaimer}>
        Experimental research system. Not financial advice.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#0b1020",
  },

  container: {
    padding: 20,
    paddingTop: 50,
    paddingBottom: 60,
    width: "100%",
    maxWidth: 700,
    alignSelf: "center",
  },

  center: {
    flex: 1,
    backgroundColor: "#0b1020",
    alignItems: "center",
    justifyContent: "center",
  },

  loading: {
    color: "white",
    marginTop: 12,
  },

  title: {
    fontSize: 30,
    fontWeight: "800",
    color: "white",
  },

  subtitle: {
    color: "#94a3b8",
    marginTop: 6,
    marginBottom: 24,
  },

  marketCard: {
    backgroundColor: "#151c30",
    padding: 20,
    borderRadius: 20,
  },

  marketLabel: {
    fontSize: 12,
    fontWeight: "700",
    color: "#94a3b8",
  },

  price: {
    fontSize: 34,
    fontWeight: "800",
    color: "white",
    marginTop: 6,
    marginBottom: 18,
  },

  sectionTitle: {
    color: "white",
    fontSize: 21,
    fontWeight: "700",
    marginTop: 26,
    marginBottom: 12,
  },

  card: {
    backgroundColor: "#151c30",
    borderRadius: 18,
    padding: 18,
    marginBottom: 14,
  },

  cardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },

  horizon: {
    color: "#94a3b8",
    fontWeight: "700",
  },

  signal: {
    color: "#ffffff",
    fontWeight: "800",
  },

  unknown: {
    color: "#fbbf24",
    fontWeight: "800",
    fontSize: 12,
  },

  rawDirection: {
    color: "white",
    fontSize: 22,
    fontWeight: "800",
    marginTop: 16,
    marginBottom: 10,
  },

  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 9,
    gap: 15,
  },

  label: {
    color: "#94a3b8",
  },

  value: {
    color: "white",
    fontWeight: "600",
    textAlign: "right",
    flexShrink: 1,
  },

  reason: {
    color: "#fbbf24",
    fontSize: 12,
    lineHeight: 18,
    marginTop: 15,
  },

  notice: {
    borderWidth: 1,
    borderColor: "#374151",
    borderRadius: 16,
    padding: 16,
    marginTop: 12,
  },

  noticeTitle: {
    color: "white",
    fontWeight: "700",
    marginBottom: 7,
  },

  noticeText: {
    color: "#94a3b8",
    lineHeight: 19,
  },

  updated: {
    color: "#64748b",
    fontSize: 12,
    marginTop: 20,
    textAlign: "center",
  },

  disclaimer: {
    color: "#64748b",
    fontSize: 12,
    marginTop: 8,
    textAlign: "center",
  },

  errorBox: {
    backgroundColor: "#3a1720",
    padding: 14,
    borderRadius: 12,
    marginBottom: 15,
  },

  errorText: {
    color: "#ffb4bd",
  },
});