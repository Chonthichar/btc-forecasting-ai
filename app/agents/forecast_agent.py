from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import torch
import yaml

class ForecastAgent:
    def __init__(self, deployed_registry, market_features, sentiment_features, model_classes):
        self.deployed = deployed_registry
        self.market_features = list(market_features)
        self.sentiment_features = list(sentiment_features)
        self.SequenceClassifier = model_classes["SequenceClassifier"]
        self.VAETransformer = model_classes["VAETransformer"]
        self.cache = {}

    def _metrics(self, model_dir):
        p = Path(model_dir) / "metrics.json"
        return json.load(open(p)) if p.exists() else {}

    def _cfg(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _make_vae(self, cfg, n_sent):
        p = cfg.get("deep_model", {})
        return self.VAETransformer(
            n_market=len(self.market_features),
            n_sent=n_sent,
            latent=int(p.get("vae_latent_dim", 24)),
            d_model=int(p.get("transformer_d_model", 64)),
            nhead=int(p.get("transformer_heads", 4)),
            num_layers=int(p.get("transformer_layers", 2)),
            dropout=float(p.get("transformer_dropout", 0.15)),
        )

    def load(self, horizon):
        h = int(horizon)
        spec = self.deployed[h]
        model_dir = Path(spec["dir"])
        cfg = self._cfg(spec["config"])
        kind = spec["kind"]
        feature_set = spec["feature_set"]

        if kind in {"lstm", "gru"}:
            features = (
                self.market_features if feature_set == "market_only"
                else self.sentiment_features if feature_set == "sentiment_only"
                else self.market_features + self.sentiment_features
            )
            scaler = joblib.load(model_dir / "scaler.joblib")
            p = cfg.get("deep_model", {})
            model = self.SequenceClassifier(
                n_features=len(features),
                hidden=int(p.get("rnn_hidden", 64)),
                kind=kind,
                dropout=float(p.get("dropout", 0.2)),
            )
            model.load_state_dict(torch.load(model_dir / "model_state.pt", map_location="cpu"))
            model.eval()
            bundle = {
                "kind": kind, "feature_set": feature_set, "model": model,
                "scaler": scaler, "features": features, "metrics": self._metrics(model_dir),
            }
        elif kind == "vae_transformer_v2":
            use_sent = feature_set == "market_plus_sentiment"
            model = self._make_vae(cfg, len(self.sentiment_features) if use_sent else 0)
            model.load_state_dict(torch.load(model_dir / "model_state.pt", map_location="cpu"))
            model.eval()
            bundle = {
                "kind": kind, "feature_set": feature_set, "model": model,
                "market_scaler": joblib.load(model_dir / "market_scaler.joblib"),
                "sent_scaler": joblib.load(model_dir / "sentiment_scaler.joblib") if use_sent else None,
                "metrics": self._metrics(model_dir),
            }
        else:
            raise KeyError(kind)

        self.cache[h] = bundle
        return bundle

    def predict(self, horizon, sequence_df):
        h = int(horizon)
        bundle = self.cache.get(h) or self.load(h)

        if bundle["kind"] in {"lstm", "gru"}:
            arr = bundle["scaler"].transform(sequence_df[bundle["features"]]).astype(np.float32)
            x = torch.tensor(arr[None, :, :], dtype=torch.float32)
            with torch.no_grad():
                prob_up = float(torch.sigmoid(bundle["model"](x)).item())
        else:
            m = bundle["market_scaler"].transform(sequence_df[self.market_features]).astype(np.float32)
            xm = torch.tensor(m[None, :, :], dtype=torch.float32)
            xs = None
            if bundle["sent_scaler"] is not None:
                s = bundle["sent_scaler"].transform(sequence_df[self.sentiment_features]).astype(np.float32)
                xs = torch.tensor(s[None, :, :], dtype=torch.float32)
            with torch.no_grad():
                logits, _, _, _ = bundle["model"](xm, xs)
                prob_up = float(torch.sigmoid(logits).item())

        latest = sequence_df.iloc[-1]
        return {
            "asset": "BTC",
            "horizon_hours": h,
            "probability_up": prob_up,
            "probability_down": 1.0 - prob_up,
            "direction": "UP" if prob_up >= 0.5 else "DOWN",
            "model": bundle["kind"],
            "feature_set": bundle["feature_set"],
            "latest_candle": str(latest["timestamp"]),
            "current_price": float(latest["close"]),
            "historical_holdout_metrics": bundle["metrics"],
        }
