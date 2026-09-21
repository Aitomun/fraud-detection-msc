"""Pure model-loading, input-validation, and prediction helpers for the demo."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load_model_artifacts(model_path: Path, metadata_path: Path):
    """Load the fitted pipeline and metadata or raise a clear file error."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_order") != FEATURE_COLS:
        raise ValueError("Saved metadata feature order does not match the application schema")
    if not 0 < float(metadata.get("threshold", 0)) < 1:
        raise ValueError("Saved classification threshold must be between zero and one")
    return model, metadata


def validate_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate one or more transactions and return a numeric ordered copy."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("At least one transaction is required")
    missing = [column for column in FEATURE_COLS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in FEATURE_COLS]
    if missing or extra:
        raise ValueError(f"Input schema mismatch; missing={missing}, extra={extra}")
    numeric = frame[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("All feature values must be numeric and non-missing")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Feature values must be finite")
    if (numeric["Amount"] < 0).any():
        raise ValueError("Amount cannot be negative")
    return numeric


def predict_transactions(model, metadata: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Return immutable-threshold predictions for validated transactions."""
    validated = validate_input_frame(frame)
    threshold = float(metadata["threshold"])
    probabilities = model.predict_proba(validated)[:, 1]
    predictions = (probabilities >= threshold).astype(np.int8)
    return pd.DataFrame({
        "fraud_probability": probabilities,
        "predicted_class": predictions,
        "threshold": threshold,
    }, index=validated.index)
