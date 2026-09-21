"""Execute Milestone 2 cleaning and the single locked train/test split.

This phase removes verified exact duplicates, creates one reproducible
stratified split, saves the raw-feature partitions, and records integrity
evidence. It does not fit preprocessing or train a model.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    DATA_PROCESSED,
    METRICS_DIR,
    RANDOM_STATE,
    RAW_CSV,
    TARGET_COL,
    TEST_SIZE,
)
from src.data_processing import remove_duplicates, save_splits, split_data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts().sort_index().reindex([0, 1], fill_value=0)
    return {"legitimate": int(counts.loc[0]), "fraudulent": int(counts.loc[1])}


def row_hashes(frame: pd.DataFrame) -> set[int]:
    return set(pd.util.hash_pandas_object(frame, index=False).astype("uint64"))


def run() -> dict:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_CSV)
    raw_rows = len(raw)
    raw_counts = class_counts(raw[TARGET_COL])
    duplicates_before = int(raw.duplicated(keep="first").sum())

    cleaned = remove_duplicates(raw, decision=True)
    removed_rows = raw_rows - len(cleaned)
    cleaned_counts = class_counts(cleaned[TARGET_COL])

    if removed_rows != duplicates_before:
        raise AssertionError("Removed-row count does not match the duplicate audit")
    if cleaned.duplicated().any():
        raise AssertionError("Exact duplicates remain after cleaning")
    if cleaned.isna().any().any():
        raise AssertionError("Unexpected missing values appeared during cleaning")
    if np.isinf(cleaned.select_dtypes(include=[np.number]).to_numpy()).any():
        raise AssertionError("Unexpected infinite values appeared during cleaning")

    X_train, X_test, y_train, y_test = split_data(
        cleaned,
        target=TARGET_COL,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_full = X_train.copy()
    train_full[TARGET_COL] = y_train
    test_full = X_test.copy()
    test_full[TARGET_COL] = y_test
    overlap_count = len(row_hashes(train_full).intersection(row_hashes(test_full)))
    if overlap_count != 0:
        raise AssertionError(f"Detected {overlap_count} identical rows across partitions")

    expected_columns = list(raw.columns)
    if list(train_full.columns) != expected_columns or list(test_full.columns) != expected_columns:
        raise AssertionError("Saved partition schema would not match the raw dataset")

    save_splits(X_train, X_test, y_train, y_test, out_dir=DATA_PROCESSED)

    train_path = DATA_PROCESSED / "train.parquet"
    test_path = DATA_PROCESSED / "test.parquet"
    train_reloaded = pd.read_parquet(train_path)
    test_reloaded = pd.read_parquet(test_path)
    if len(train_reloaded) != len(train_full) or len(test_reloaded) != len(test_full):
        raise AssertionError("Reloaded partition row counts do not match")
    if list(train_reloaded.columns) != expected_columns or list(test_reloaded.columns) != expected_columns:
        raise AssertionError("Reloaded partition schema does not match")

    train_counts = class_counts(y_train)
    test_counts = class_counts(y_test)
    metadata = {
        "source_file": str(RAW_CSV.relative_to(RAW_CSV.parents[2])),
        "source_sha256": sha256_file(RAW_CSV),
        "duplicate_policy": "remove exact duplicates and retain first occurrence before split",
        "raw_rows": raw_rows,
        "raw_class_counts": raw_counts,
        "exact_duplicate_rows_removed": removed_rows,
        "cleaned_rows": int(len(cleaned)),
        "cleaned_class_counts": cleaned_counts,
        "test_size_parameter": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratified_on": TARGET_COL,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "train_fraud_percentage": float(y_train.mean() * 100),
        "test_fraud_percentage": float(y_test.mean() * 100),
        "identical_row_hash_overlap": overlap_count,
        "train_file": str(train_path.relative_to(RAW_CSV.parents[2])),
        "test_file": str(test_path.relative_to(RAW_CSV.parents[2])),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "test_set_policy": "locked until the one final evaluation after model and threshold selection",
    }

    metadata_path = DATA_PROCESSED / "split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "partition": "raw",
                "rows": raw_rows,
                **raw_counts,
                "fraud_percentage": raw_counts["fraudulent"] / raw_rows * 100,
            },
            {
                "partition": "cleaned",
                "rows": len(cleaned),
                **cleaned_counts,
                "fraud_percentage": cleaned_counts["fraudulent"] / len(cleaned) * 100,
            },
            {
                "partition": "train",
                "rows": len(X_train),
                **train_counts,
                "fraud_percentage": y_train.mean() * 100,
            },
            {
                "partition": "test_locked",
                "rows": len(X_test),
                **test_counts,
                "fraud_percentage": y_test.mean() * 100,
            },
        ]
    )
    summary.to_csv(METRICS_DIR / "cleaning_split_summary.csv", index=False)

    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    run()
