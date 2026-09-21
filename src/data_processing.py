"""
src/data_processing.py
======================
Handles all data loading, inspection, cleaning, and train/test splitting.

Rules enforced here:
- Duplicates are investigated before any removal decision.
- Missing values are checked and reported.
- The train/test split is stratified.
- The test set is NOT touched after splitting.
- No scaling is applied here (that happens inside the pipeline in feature_engineering.py).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    RAW_CSV, TARGET_COL, RANDOM_STATE, TEST_SIZE, DATA_PROCESSED
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_data(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV from disk and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Inspection helpers (called during EDA)
# ---------------------------------------------------------------------------

def dataset_overview(df: pd.DataFrame) -> dict:
    """
    Return a summary dict with shape, dtypes, missing values, and duplicates.
    Prints a human-readable report as a side effect.
    """
    n_rows, n_cols = df.shape
    n_missing      = df.isnull().sum().sum()
    n_duplicates   = df.duplicated().sum()

    overview = {
        "n_rows":       n_rows,
        "n_cols":       n_cols,
        "n_missing":    n_missing,
        "n_duplicates": n_duplicates,
        "dtypes":       df.dtypes.to_dict(),
    }

    print("=" * 55)
    print("DATASET OVERVIEW")
    print("=" * 55)
    print(f"  Rows        : {n_rows:,}")
    print(f"  Columns     : {n_cols}")
    print(f"  Missing vals: {n_missing}")
    print(f"  Duplicates  : {n_duplicates:,}")
    print()
    print("Column dtypes:")
    print(df.dtypes)
    print()

    return overview


def investigate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a thorough duplicate investigation without automatically removing rows.

    Returns a DataFrame of duplicated rows for inspection.
    Prints counts broken down by Class label.
    """
    dup_mask  = df.duplicated(keep=False)          # mark ALL copies
    dup_df    = df[dup_mask].copy()
    first_dup = df.duplicated(keep="first")        # mark all except first copy

    print("=" * 55)
    print("DUPLICATE INVESTIGATION")
    print("=" * 55)
    print(f"  Total rows flagged as duplicates : {dup_mask.sum():,}")
    print(f"  Rows that are exact copies       : {first_dup.sum():,}")
    print()
    print("Duplicate counts by Class:")
    print(dup_df[TARGET_COL].value_counts())
    print()

    return dup_df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return per-column missing-value counts and print a report."""
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found. No imputation required.")
    else:
        print("Missing values detected:")
        print(missing[missing > 0])
    return missing


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return a two-row DataFrame showing count and % for each class."""
    counts = df[TARGET_COL].value_counts().sort_index()
    pct    = (counts / len(df) * 100).round(4)
    dist   = pd.DataFrame({"count": counts, "percent": pct})
    dist.index = dist.index.map({0: "Legitimate (0)", 1: "Fraudulent (1)"})
    print("CLASS DISTRIBUTION")
    print(dist)
    print()
    return dist


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame, decision: bool = True) -> pd.DataFrame:
    """
    Remove exact duplicate rows if `decision` is True.

    This function must be called BEFORE the train/test split.
    The decision to remove should be documented based on investigate_duplicates().
    """
    if not decision:
        print("Duplicates retained by project decision.")
        return df

    before = len(df)
    df_clean = df.drop_duplicates(keep="first").reset_index(drop=True)
    after    = len(df_clean)
    print(f"Removed {before - after:,} duplicate rows ({before:,} -> {after:,}).")
    return df_clean


# ---------------------------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    target: str     = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Train : {len(X_train):,} rows | fraud rate = {y_train.mean():.4%}")
    print(f"Test  : {len(X_test):,}  rows | fraud rate = {y_test.mean():.4%}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Persistence helpers (save processed splits)
# ---------------------------------------------------------------------------

def save_splits(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    out_dir: Path = DATA_PROCESSED,
) -> None:
    """Save processed train/test splits to parquet for fast reloading."""
    out_dir.mkdir(parents=True, exist_ok=True)

    train = X_train.copy()
    train[TARGET_COL] = y_train.values

    test = X_test.copy()
    test[TARGET_COL] = y_test.values

    train.to_parquet(out_dir / "train.parquet", index=False)
    test.to_parquet(out_dir  / "test.parquet",  index=False)
    print(f"Splits saved to {out_dir}")


def load_splits(
    out_dir: Path = DATA_PROCESSED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reload train/test splits saved by save_splits()."""
    train = pd.read_parquet(out_dir / "train.parquet")
    test  = pd.read_parquet(out_dir / "test.parquet")

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test  = test.drop(columns=[TARGET_COL])
    y_test  = test[TARGET_COL]

    return X_train, X_test, y_train, y_test


def load_training_split(
    out_dir: Path = DATA_PROCESSED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load only the training partition for development experiments.

    Use this function for cross-validation, imbalance experiments, tuning, and
    threshold selection so development code does not open the locked test set.
    """
    train = pd.read_parquet(out_dir / "train.parquet")
    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    return X_train, y_train


def load_test_split(
    out_dir: Path = DATA_PROCESSED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the locked test partition for the one final evaluation only."""
    test = pd.read_parquet(out_dir / "test.parquet")
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]
    return X_test, y_test
