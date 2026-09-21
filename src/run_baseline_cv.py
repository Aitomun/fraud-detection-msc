"""Execute Milestone 3 training-only baseline cross-validation.

The locked test partition is deliberately not imported or loaded here.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CV_FOLDS, DATA_PROCESSED, METRICS_DIR, RANDOM_STATE, TARGET_COL
from src.data_processing import load_training_split
from src.feature_engineering import (
    build_dt_pipeline,
    build_lr_pipeline,
    build_rf_pipeline,
)
from src.train import cross_validate_model


METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def classifier_parameters(pipeline) -> dict:
    classifier = pipeline.named_steps["classifier"]
    keys = [
        "C",
        "solver",
        "max_iter",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "n_estimators",
        "max_features",
        "class_weight",
        "random_state",
    ]
    params = classifier.get_params()
    return {key: params[key] for key in keys if key in params}


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # This loader reads train.parquet only. Do not replace it with load_splits().
    X_train, y_train = load_training_split(DATA_PROCESSED)
    if TARGET_COL in X_train.columns:
        raise AssertionError("Target leakage: Class is present in the predictors")
    if list(X_train.columns) != ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]:
        raise AssertionError("Unexpected training feature schema")
    if set(y_train.unique()) != {0, 1}:
        raise AssertionError("Unexpected training target labels")

    models = {
        "Logistic Regression": build_lr_pipeline(class_weight=None),
        "Decision Tree": build_dt_pipeline(class_weight=None),
        # CV parallelizes folds; keep each forest single-process to avoid nested parallelism.
        "Random Forest": build_rf_pipeline(
            class_weight=None, n_estimators=100, n_jobs=1
        ),
    }

    all_folds: list[pd.DataFrame] = []
    metadata_models: dict[str, dict] = {}
    for model_name, pipeline in models.items():
        print(f"\n{'=' * 70}\n{model_name} baseline\n{'=' * 70}")
        fold_results = cross_validate_model(
            pipeline,
            X_train,
            y_train,
            n_splits=CV_FOLDS,
        )
        fold_results.insert(0, "fold", np.arange(1, len(fold_results) + 1))
        fold_results.insert(0, "imbalance_strategy", "original_distribution")
        fold_results.insert(0, "model", model_name)
        all_folds.append(fold_results)
        metadata_models[model_name] = classifier_parameters(pipeline)

    folds = pd.concat(all_folds, ignore_index=True)
    summary_rows = []
    for model_name, group in folds.groupby("model", sort=False):
        row: dict[str, object] = {
            "model": model_name,
            "imbalance_strategy": "original_distribution",
            "cv_folds": CV_FOLDS,
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        row["fit_time_seconds_mean"] = float(group["fit_time_seconds"].mean())
        row["fit_time_seconds_total"] = float(group["fit_time_seconds"].sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    folds_path = METRICS_DIR / "baseline_cv_fold_results.csv"
    summary_path = METRICS_DIR / "baseline_cv_summary.csv"
    metadata_path = METRICS_DIR / "baseline_cv_metadata.json"
    folds.to_csv(folds_path, index=False)
    summary.to_csv(summary_path, index=False)

    metadata = {
        "phase": "Milestone 3 baseline cross-validation",
        "data_loaded": "data/processed/train.parquet only",
        "test_data_loaded": False,
        "training_rows": int(len(X_train)),
        "training_fraud_cases": int(y_train.sum()),
        "features": list(X_train.columns),
        "random_state": RANDOM_STATE,
        "cross_validation": {
            "type": "StratifiedKFold",
            "folds": CV_FOLDS,
            "shuffle": True,
        },
        "imbalance_strategy": "original training distribution",
        "models": metadata_models,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nBaseline cross-validation summary")
    print(summary.to_string(index=False))
    print(f"\nSaved {folds_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {metadata_path}")
    return folds, summary


if __name__ == "__main__":
    run()
