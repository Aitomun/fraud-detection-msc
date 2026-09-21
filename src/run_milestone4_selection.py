"""Controlled Milestone 4 model search and training-only threshold selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CV_FOLDS, DATA_PROCESSED, FIGURES_DIR, METRICS_DIR, RANDOM_STATE
from src.data_processing import load_training_split
from src.feature_engineering import (
    build_lr_pipeline,
    build_lr_smote_pipeline,
    build_rf_pipeline,
    build_rf_smote_pipeline,
)
from src.train import cross_validate_model


METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


NEW_CANDIDATES = [
    # Fast tests of the pending Amount transformation and LR regularization.
    {
        "candidate_id": "lr_log1p_none_c01",
        "model": "Logistic Regression",
        "strategy": "original_distribution",
        "amount_transform": "log1p",
        "params": {"C": 0.1, "class_weight": None},
    },
    {
        "candidate_id": "lr_log1p_none_c1",
        "model": "Logistic Regression",
        "strategy": "original_distribution",
        "amount_transform": "log1p",
        "params": {"C": 1.0, "class_weight": None},
    },
    {
        "candidate_id": "lr_log1p_none_c10",
        "model": "Logistic Regression",
        "strategy": "original_distribution",
        "amount_transform": "log1p",
        "params": {"C": 10.0, "class_weight": None},
    },
    {
        "candidate_id": "lr_log1p_balanced_c1",
        "model": "Logistic Regression",
        "strategy": "class_weight_balanced",
        "amount_transform": "log1p",
        "params": {"C": 1.0, "class_weight": "balanced"},
    },
    {
        "candidate_id": "lr_smote_original_c1",
        "model": "Logistic Regression",
        "strategy": "smote",
        "amount_transform": "original",
        "params": {"C": 1.0, "class_weight": None},
    },
    {
        "candidate_id": "lr_smote_log1p_c1",
        "model": "Logistic Regression",
        "strategy": "smote",
        "amount_transform": "log1p",
        "params": {"C": 1.0, "class_weight": None},
    },
    # A deliberately bounded RF search; each change has a specific hypothesis.
    {
        "candidate_id": "rf_none_200",
        "model": "Random Forest",
        "strategy": "original_distribution",
        "amount_transform": "original",
        "params": {"n_estimators": 200, "class_weight": None, "n_jobs": 1},
    },
    {
        "candidate_id": "rf_balanced_200",
        "model": "Random Forest",
        "strategy": "class_weight_balanced",
        "amount_transform": "original",
        "params": {
            "n_estimators": 200,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        },
    },
    {
        "candidate_id": "rf_balanced_subsample_200",
        "model": "Random Forest",
        "strategy": "class_weight_balanced_subsample",
        "amount_transform": "original",
        "params": {
            "n_estimators": 200,
            "class_weight": "balanced_subsample",
            "n_jobs": 1,
        },
    },
    {
        "candidate_id": "rf_balanced_200_leaf2",
        "model": "Random Forest",
        "strategy": "class_weight_balanced",
        "amount_transform": "original",
        "params": {
            "n_estimators": 200,
            "class_weight": "balanced",
            "min_samples_leaf": 2,
            "n_jobs": 1,
        },
    },
    {
        "candidate_id": "rf_smote_100",
        "model": "Random Forest",
        "strategy": "smote",
        "amount_transform": "original",
        "params": {"n_estimators": 100, "class_weight": None, "n_jobs": 1},
    },
]


def build_candidate(spec: dict):
    kwargs = dict(spec["params"])
    if spec["model"] == "Logistic Regression":
        builder = build_lr_smote_pipeline if spec["strategy"] == "smote" else build_lr_pipeline
        return builder(amount_transform=spec["amount_transform"], **kwargs)
    builder = build_rf_smote_pipeline if spec["strategy"] == "smote" else build_rf_pipeline
    return builder(**kwargs)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_id, group in folds.groupby("candidate_id", sort=False):
        first = group.iloc[0]
        row = {
            "candidate_id": candidate_id,
            "model": first["model"],
            "imbalance_strategy": first["imbalance_strategy"],
            "amount_transform": first["amount_transform"],
            "cv_folds": len(group),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        row["fit_time_seconds_mean"] = float(group["fit_time_seconds"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def existing_candidates() -> tuple[pd.DataFrame, list[dict]]:
    baseline = pd.read_csv(METRICS_DIR / "baseline_cv_fold_results.csv")
    weighted = pd.read_csv(METRICS_DIR / "class_weight_cv_fold_results.csv")
    mapping = {
        ("Logistic Regression", "original_distribution"): "lr_original_none_c1",
        ("Decision Tree", "original_distribution"): "dt_original_default",
        ("Random Forest", "original_distribution"): "rf_none_100",
        ("Logistic Regression", "class_weight_balanced"): "lr_original_balanced_c1",
        ("Decision Tree", "class_weight_balanced"): "dt_balanced_default",
        ("Random Forest", "class_weight_balanced"): "rf_balanced_100",
    }
    combined = pd.concat([baseline, weighted], ignore_index=True)
    combined["candidate_id"] = [
        mapping[(row.model, row.imbalance_strategy)]
        for row in combined.itertuples()
    ]
    combined["amount_transform"] = "original"
    registry = [
        {
            "candidate_id": candidate_id,
            "model": model,
            "strategy": strategy,
            "amount_transform": "original",
            "params": {},
            "source": "Milestone 3 baseline" if strategy == "original_distribution" else "Milestone 4A",
        }
        for (model, strategy), candidate_id in mapping.items()
    ]
    return combined, registry


def threshold_table(y: pd.Series, scores: np.ndarray) -> tuple[pd.DataFrame, dict]:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best_index = int(np.argmax(f1))
    best_threshold = float(thresholds[best_index])
    predictions = (scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    selected = {
        "threshold": best_threshold,
        "selection_objective": "maximum out-of-fold F1",
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y, scores)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    table = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
        "f1": f1,
    })
    return table, selected


def plot_thresholds(table: pd.DataFrame, selected: dict) -> None:
    display = table.iloc[::max(1, len(table) // 1500)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(display["threshold"], display["precision"], label="Precision")
    ax.plot(display["threshold"], display["recall"], label="Recall")
    ax.plot(display["threshold"], display["f1"], label="F1")
    ax.axvline(selected["threshold"], color="black", linestyle="--", linewidth=1,
               label=f"Selected = {selected['threshold']:.4f}")
    ax.set(xlabel="Probability threshold", ylabel="Score",
           title="Training-only out-of-fold threshold selection")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "oof_threshold_selection.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    X_train, y_train = load_training_split(DATA_PROCESSED)
    existing_folds, registry = existing_candidates()
    new_fold_frames = []
    for spec in NEW_CANDIDATES:
        print(f"\n{'=' * 72}\n{spec['candidate_id']}\n{'=' * 72}")
        result = cross_validate_model(build_candidate(spec), X_train, y_train, n_splits=CV_FOLDS)
        result.insert(0, "fold", np.arange(1, len(result) + 1))
        result.insert(0, "amount_transform", spec["amount_transform"])
        result.insert(0, "imbalance_strategy", spec["strategy"])
        result.insert(0, "model", spec["model"])
        result.insert(0, "candidate_id", spec["candidate_id"])
        new_fold_frames.append(result)

    all_folds = pd.concat([existing_folds, *new_fold_frames], ignore_index=True, sort=False)
    summary = summarize(all_folds).sort_values(
        ["pr_auc_mean", "pr_auc_std", "f1_mean"], ascending=[False, True, False]
    ).reset_index(drop=True)
    winner_id = str(summary.iloc[0]["candidate_id"])
    all_specs = registry + [{**spec, "source": "Milestone 4B"} for spec in NEW_CANDIDATES]
    winner_spec = next(spec for spec in all_specs if spec["candidate_id"] == winner_id)
    if winner_id in {"rf_none_100", "rf_balanced_100"}:
        winner_spec["params"] = {
            "n_estimators": 100,
            "class_weight": None if winner_id == "rf_none_100" else "balanced",
            "n_jobs": 1,
        }
    elif winner_id.startswith("lr_") and not winner_spec["params"]:
        winner_spec["params"] = {
            "C": 1.0,
            "class_weight": "balanced" if "balanced" in winner_id else None,
        }

    print(f"\nSelected by mean PR-AUC: {winner_id}")
    selected_pipeline = build_candidate(winner_spec)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_scores = cross_val_predict(
        selected_pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    thresholds, selected_threshold = threshold_table(y_train, oof_scores)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all_folds.to_csv(METRICS_DIR / "model_selection_fold_results.csv", index=False)
    summary.to_csv(METRICS_DIR / "model_selection_summary.csv", index=False)
    thresholds.to_csv(METRICS_DIR / "oof_threshold_analysis.csv", index=False)
    pd.DataFrame({"row_index": X_train.index, "y_true": y_train, "oof_probability": oof_scores}).to_csv(
        METRICS_DIR / "selected_model_oof_predictions.csv", index=False
    )
    plot_thresholds(thresholds, selected_threshold)

    frozen = {
        "phase": "Milestone 4 model and threshold selection",
        "selection_metric": "mean five-fold cross-validated average precision (PR-AUC)",
        "tie_break_evidence": ["lower PR-AUC standard deviation", "higher mean F1"],
        "selected_candidate": winner_spec,
        "selected_cv_results": summary.iloc[0].to_dict(),
        "threshold_selection": selected_threshold,
        "feature_order": list(X_train.columns),
        "cross_validation": {
            "type": "StratifiedKFold", "folds": CV_FOLDS,
            "shuffle": True, "random_state": RANDOM_STATE,
        },
        "training_rows": int(len(X_train)),
        "training_fraud_cases": int(y_train.sum()),
        "test_data_loaded": False,
        "candidate_registry": all_specs,
    }
    (METRICS_DIR / "frozen_model_specification.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )
    print(summary[["candidate_id", "pr_auc_mean", "pr_auc_std", "f1_mean", "recall_mean"]].to_string(index=False))
    print(json.dumps(selected_threshold, indent=2))


if __name__ == "__main__":
    run()
