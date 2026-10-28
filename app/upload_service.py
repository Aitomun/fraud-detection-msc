"""Pure helpers for scoring an uploaded dataset with the frozen model.

The model only understands the published dataset schema (Time, V1-V28, Amount).
V1-V28 are PCA components of confidential bank features whose loadings were never
released, so files with everyday columns (merchant, category, ...) are detected and
rejected with an explanation instead of being scored on invented values.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.model_service import FEATURE_COLS, predict_transactions


MAX_ROWS = 200_000
LABEL_COL = "Class"
V_COLS = [f"V{i}" for i in range(1, 29)]
LABEL_ALIASES = {"class", "label", "isfraud", "fraud", "target"}


def read_upload(name: str, data: bytes) -> pd.DataFrame:
    """Read an uploaded CSV or Parquet file, handling CSVs without a header row."""
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if suffix not in {"csv", "txt", "parquet"}:
        raise ValueError("Unsupported file type; upload a .csv or .parquet file")
    try:
        if suffix == "parquet":
            frame = pd.read_parquet(io.BytesIO(data))
        else:
            frame = pd.read_csv(io.BytesIO(data))
            if _header_is_numeric(frame.columns):
                frame = pd.read_csv(io.BytesIO(data), header=None)
    except Exception as error:
        raise ValueError(f"Could not read the file: {error}") from error
    if frame.empty:
        raise ValueError("The uploaded file contains no rows")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"The file has {len(frame):,} rows; the limit is {MAX_ROWS:,}")
    return frame


def _header_is_numeric(columns) -> bool:
    try:
        [float(str(column)) for column in columns]
    except ValueError:
        return False
    return True


def _canonical_name(column) -> str | None:
    key = re.sub(r"[^a-z0-9]", "", str(column).lower())
    if key == "time":
        return "Time"
    if key == "amount":
        return "Amount"
    if key in LABEL_ALIASES:
        return LABEL_COL
    match = re.fullmatch(r"(?:v|pca|pc)(\d{1,2})", key)
    if match and 1 <= int(match.group(1)) <= 28:
        return f"V{int(match.group(1))}"
    return None


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename recognised column spellings to the model schema.

    A header-less file with 30 or 31 columns is taken to be in the published order
    (Time, V1-V28, Amount[, Class]). Unrecognised columns are kept unchanged.
    """
    if all(isinstance(column, (int, np.integer)) for column in frame.columns):
        if frame.shape[1] == len(FEATURE_COLS):
            return frame.set_axis(FEATURE_COLS, axis=1)
        if frame.shape[1] == len(FEATURE_COLS) + 1:
            return frame.set_axis(FEATURE_COLS + [LABEL_COL], axis=1)
        frame = frame.set_axis([f"column_{i + 1}" for i in range(frame.shape[1])], axis=1)

    renamed = {}
    for column in frame.columns:
        canonical = _canonical_name(column)
        renamed[column] = canonical if canonical else column
    targets = list(renamed.values())
    duplicates = sorted({name for name in targets if targets.count(name) > 1})
    if duplicates:
        raise ValueError(f"Several columns map to the same feature: {duplicates}")
    return frame.rename(columns=renamed)


def detect_schema(frame: pd.DataFrame) -> tuple[str, list[str]]:
    """Return ("pca" | "partial" | "everyday", missing model features)."""
    missing = [column for column in FEATURE_COLS if column not in frame.columns]
    if not missing:
        return "pca", []
    if not any(column in frame.columns for column in V_COLS):
        return "everyday", missing
    return "partial", missing


def _row_problems(features: pd.DataFrame) -> pd.Series:
    """Describe why each row is unusable; empty string means the row is valid."""
    numeric = features.apply(pd.to_numeric, errors="coerce")
    problems = pd.Series("", index=features.index, dtype=object)
    missing = numeric.isna()
    for index in missing.index[missing.any(axis=1)]:
        problems[index] = "missing or non-numeric: " + ", ".join(missing.columns[missing.loc[index]])
    values = numeric.to_numpy(dtype=float)
    infinite = np.isinf(values).any(axis=1)
    problems[infinite & (problems == "")] = "infinite value"
    negative = (numeric["Amount"] < 0).to_numpy()
    problems[negative & (problems == "")] = "negative Amount"
    return problems


def score_upload(model, metadata: dict, frame: pd.DataFrame):
    """Score valid rows of a normalised "pca" frame.

    Returns (results, invalid): results holds the original columns of valid rows plus
    fraud_probability and predicted_class; invalid holds rejected rows with a reason.
    """
    problems = _row_problems(frame[FEATURE_COLS])
    valid_mask = problems == ""
    invalid = frame.loc[~valid_mask].assign(reason=problems[~valid_mask])
    results = frame.loc[valid_mask].copy()
    if not results.empty:
        predictions = predict_transactions(model, metadata, results[FEATURE_COLS])
        results["fraud_probability"] = predictions["fraud_probability"]
        results["predicted_class"] = predictions["predicted_class"]
    return results, invalid


def evaluate_labelled(y_true, probabilities, predictions) -> dict:
    """Metrics for an uploaded file that includes a 0/1 fraud label."""
    labels = pd.to_numeric(pd.Series(y_true), errors="coerce")
    if labels.isna().any() or not set(labels.unique()) <= {0, 1}:
        raise ValueError("The label column must contain only 0 (legitimate) and 1 (fraud)")
    labels = labels.astype(int).to_numpy()
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "pr_auc": average_precision_score(labels, probabilities) if labels.any() else None,
        "confusion_matrix": pd.DataFrame(
            matrix,
            index=["Actual legitimate", "Actual fraud"],
            columns=["Predicted legitimate", "Predicted fraud"],
        ),
    }
