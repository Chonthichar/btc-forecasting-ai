"""Loads the frozen deployment artifacts once per process.

Nothing here trains, fits or modifies a model. Every artifact is read from
btc_deployment_models/ exactly as shipped, and each horizon keeps its own
scaler: sharing one across horizons would silently mis-scale the inputs.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import torch
import torch.nn as nn

from .feature_builder import read_feature_order

HORIZONS = (1, 6, 24)


class DirectionTransformer(nn.Module):
    """Rebuilt from the checkpoint's own metadata (integration brief section 9).

    The checkpoint is the source of truth for every dimension, so a retrained
    artifact with different sizes loads without code changes.
    """

    def __init__(self, n_features: int, seq_len: int, d_model: int, n_heads: int,
                 n_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.activation = nn.GELU()
        self.input_norm = nn.LayerNorm(d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(self.activation(self.input_projection(x)))
        h = h + self.positional_embedding[:, : h.size(1), :]
        h = self.encoder(h)
        return self.classifier(self.dropout(self.output_norm(h[:, -1, :]))).squeeze(-1)


class DeploymentModels:
    def __init__(self, models_dir: Path):
        self.dir = Path(models_dir)
        self.risk_dir = self.dir / "risk"
        self.direction_dir = self.dir / "direction"
        self._lock = threading.Lock()
        self._risk: dict[int, dict] = {}
        self._direction: dict[int, dict] = {}
        self._bands: dict | None = None

        self.risk_features = read_feature_order(self.risk_dir / "risk_feature_list.csv")
        self.direction_features = read_feature_order(
            self.direction_dir / "direction_stationary_feature_list.csv")

    # joblib is imported lazily so that importing this module never requires
    # scikit-learn to be present just to inspect feature lists.
    @staticmethod
    def _load_joblib(path: Path):
        import joblib
        return joblib.load(path)

    def risk(self, horizon: int) -> dict:
        if horizon not in self._risk:
            with self._lock:
                if horizon not in self._risk:
                    self._risk[horizon] = {
                        "scaler": self._load_joblib(self.risk_dir / f"risk_{horizon}h_scaler.joblib"),
                        "model": self._load_joblib(self.risk_dir / f"risk_{horizon}h_logistic.joblib"),
                    }
        return self._risk[horizon]

    def direction(self, horizon: int) -> dict:
        if horizon not in self._direction:
            with self._lock:
                if horizon not in self._direction:
                    checkpoint = torch.load(
                        self.direction_dir / f"direction_{horizon}h_transformer.pt",
                        map_location="cpu", weights_only=False)
                    model = DirectionTransformer(
                        n_features=int(checkpoint["input_dim"]),
                        seq_len=int(checkpoint["seq_len"]),
                        d_model=int(checkpoint["d_model"]),
                        n_heads=int(checkpoint["n_heads"]),
                        n_layers=int(checkpoint["n_layers"]),
                        dim_feedforward=int(checkpoint["dim_feedforward"]),
                        dropout=float(checkpoint.get("dropout", 0.15)))
                    model.load_state_dict(checkpoint["state_dict"], strict=True)
                    model.eval()
                    self._direction[horizon] = {
                        "model": model,
                        "scaler": self._load_joblib(
                            self.direction_dir / f"direction_{horizon}h_scaler.joblib"),
                        "seq_len": int(checkpoint["seq_len"]),
                        "threshold": float(checkpoint.get("decision_threshold", 0.5)),
                        "target_definition": checkpoint.get("target_definition"),
                        "holdout_evaluated": bool(checkpoint.get("final_holdout_evaluated", False)),
                    }
        return self._direction[horizon]

    def risk_bands(self) -> dict:
        """Band cut-points plus the out-of-fold event rate observed in each."""
        if self._bands is None:
            import csv as _csv
            thresholds = json.loads(
                (self.risk_dir / "provisional_risk_band_thresholds.json").read_text(encoding="utf-8"))
            empirical: dict[tuple[int, str], dict] = {}
            with open(self.risk_dir / "risk_band_empirical_results.csv", encoding="utf-8") as handle:
                for row in _csv.DictReader(handle):
                    empirical[(int(row["horizon_hours"]), row["risk_band"])] = {
                        "event_rate": float(row["empirical_event_rate"]),
                        "baseline_rate": float(row["baseline_event_rate"]),
                        "lift": float(row["event_rate_lift"]),
                        "n": int(row["n"]),
                    }
            self._bands = {"thresholds": thresholds, "empirical": empirical}
        return self._bands
