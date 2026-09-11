const API_URL = process.env.EXPO_PUBLIC_API_URL;

async function apiRequest<T>(path: string): Promise<T> {
  if (!API_URL) {
    throw new Error("EXPO_PUBLIC_API_URL is not configured.");
  }

  const response = await fetch(`${API_URL}${path}`);

  if (!response.ok) {
    throw new Error(`API error ${response.status}`);
  }

  return response.json();
}

export type ForecastContext = {
  horizon_hours: number;
  raw_direction: "UP" | "DOWN";
  probability_up: number;
  probability_down: number;

  validated_reliability: number | null;
  reliability: number | null;
  signal_state: "UP" | "DOWN" | "NO_SIGNAL" | "UNKNOWN";

  reliability_reason: string;

  model: string;
  feature_set: string;
  model_version: string;

  reference_price: number;
};

export type ContextResponse = {
  context: {
    timestamp: string;
    captured_at: string;
    last_refresh: string;
    data_fresh: boolean;

    market: {
      price: number;
      timestamp: string;
      regime: string | null;
      volatility_24h: number | null;
      momentum_24h: number | null;
      rsi_14: number | null;
    };

    sentiment: {
      score: number | null;
      label: string | null;
      article_count: number;
      hours_with_news: number;
      timestamp: string | null;
    };

    forecasts: {
      "1": ForecastContext;
      "6": ForecastContext;
      "24": ForecastContext;
    };

    warnings: string[];
  };

  validation: {
    context_valid: boolean;
    forecast_usable: boolean;
    forecast_reason: string;
    reliability_threshold: number;
  };

  configuration: {
    openai_configured: boolean;
    tavily_configured: boolean;
    openai_model: string;
    reliability_threshold: number;
  };
};

export function getContext() {
  return apiRequest<ContextResponse>("/context");
}