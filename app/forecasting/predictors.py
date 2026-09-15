"""Inference for the frozen risk and direction models.

Both return the model's own number untouched. Nothing here rounds toward a
nicer value, blends horizons, or substitutes a fallback when the model is
unsure: an unusable input raises instead (integration brief section 12).
"""
from __future__ import annotations

import numpy as np
import torch

from .feature_builder import latest_row, latest_sequence
from .model_loader import DeploymentModels

# The risk models answer one question, and the wording matters: it is a
# two-sided barrier touch, not a crash or downside probability.
RISK_EVENT = ("BTC touches +2.5% or -2.5% from the current close at any point "
              "within the next {hours} hour{plural}")


def risk_event_definition(horizon: int) -> str:
    return RISK_EVENT.format(hours=horizon, plural="" if horizon == 1 else "s")


def band_for(score: float, thresholds: dict) -> str:
    if score <= thresholds["low_upper"]:
        return "LOW"
    if score <= thresholds["typical_upper"]:
        return "TYPICAL"
    if score <= thresholds["elevated_upper"]:
        return "ELEVATED"
    return "HIGH"


def predict_risk(models: DeploymentModels, frame, horizon: int) -> dict:
    bundle = models.risk(horizon)
    values = latest_row(frame, models.risk_features)
    score = float(bundle["model"].predict_proba(bundle["scaler"].transform(values))[0, 1])

    bands = models.risk_bands()
    level = band_for(score, bands["thresholds"][f"{horizon}h"])
    observed = bands["empirical"].get((horizon, level), {})
    return {
        "horizon_hours": horizon,
        "model_score": score,
        "risk_level": level,
        "historical_event_rate": observed.get("event_rate"),
        "baseline_event_rate": observed.get("baseline_rate"),
        "event_rate_lift": observed.get("lift"),
        "band_sample_size": observed.get("n"),
        "event_definition": risk_event_definition(horizon),
        # The bands come from out-of-fold scores only; the JSON says so itself.
        "band_basis": "PROVISIONAL_OOF_ONLY",
    }


def predict_direction(models: DeploymentModels, frame, horizon: int) -> dict:
    bundle = models.direction(horizon)
    window = latest_sequence(frame, models.direction_features, bundle["seq_len"])
    scaled = bundle["scaler"].transform(window)
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        up = float(torch.sigmoid(bundle["model"](tensor)).item())

    return {
        "horizon_hours": horizon,
        "label": "UP" if up >= bundle["threshold"] else "DOWN",
        "up_score": up,
        "down_score": 1.0 - up,
        "decision_threshold": bundle["threshold"],
        "target_definition": bundle["target_definition"],
        # False on every shipped artifact; surfaced so the UI cannot imply
        # a validated accuracy that does not exist.
        "holdout_evaluated": bundle["holdout_evaluated"],
        "sequence_length": bundle["seq_len"],
    }
