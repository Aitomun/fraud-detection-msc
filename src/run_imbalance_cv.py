"""Run the class-weighting stage of Milestone 4 on training data only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CV_FOLDS, DATA_PROCESSED, METRICS_DIR, RANDOM_STATE
from src.data_processing import load_training_split
from src.feature_engineering import (
    build_dt_pipeline,
    build_lr_pipeline,
    build_rf_pipeline,
)
from src.train import cross_validate_model


METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, strategy), group in folds.groupby(
        ["model", "imbalance_strategy"], sort=False
    ):
        row: dict[str, object] = {
            "model": model,
            "imbalance_strategy": strategy,
            "cv_folds": len(group),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        row["fit_time_seconds_mean"] = float(group["fit_time_seconds"].mean())
        row["fit_time_seconds_total"] = float(group["fit_time_seconds"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train, y_train = load_training_split(DATA_PROCESSED)
    models = {
        "Logistic Regression": build_lr_pipeline(class_weight="balanced"),
        "Decision Tree": build_dt_pipeline(class_weight="balanced"),
        "Random Forest": build_rf_pipeline(
            class_weight="balanced", n_estimators=100, n_jobs=1
        ),
    }

    fold_frames = []
    for model_name, pipeline in models.items():
        print(f"\n{'=' * 70}\n{model_name} class-weighted\n{'=' * 70}")
        frame = cross_validate_model(
            pipeline, X_train, y_train, n_splits=CV_FOLDS
        )
        frame.insert(0, "fold", np.arange(1, len(frame) + 1))
        frame.insert(0, "imbalance_strategy", "class_weight_balanced")
        frame.insert(0, "model", model_name)
        fold_frames.append(frame)

    weighted_folds = pd.concat(fold_frames, ignore_index=True)
    weighted_summary = summarize(weighted_folds)

    weighted_folds.to_csv(
        METRICS_DIR / "class_weight_cv_fold_results.csv", index=False
    )
    weighted_summary.to_csv(
        METRICS_DIR / "class_weight_cv_summary.csv", index=False
    )

    baseline_summary = pd.read_csv(METRICS_DIR / "baseline_cv_summary.csv")
    comparison = pd.concat(
        [baseline_summary, weighted_summary], ignore_index=True, sort=False
    )
    comparison.to_csv(
        METRICS_DIR / "imbalance_strategy_comparison.csv", index=False
    )

    metadata = {
        "phase": "Milestone 4A class weighting",
        "data_loaded": "data/processed/train.parquet only",
        "test_data_loaded": False,
        "training_rows": int(len(X_train)),
        "training_fraud_cases": int(y_train.sum()),
        "cross_validation": {
            "type": "StratifiedKFold",
            "folds": CV_FOLDS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "comparison": ["original_distribution", "class_weight_balanced"],
        "smote_status": "not run; decision follows class-weight evidence",
    }
    (METRICS_DIR / "class_weight_cv_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nClass-weighted cross-validation summary")
    print(weighted_summary.to_string(index=False))
    return weighted_folds, comparison


if __name__ == "__main__":
    run()
