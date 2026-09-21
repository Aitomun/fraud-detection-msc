"""Run the single Milestone 5 evaluation using the frozen specification."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    DATA_PROCESSED,
    FIGURES_DIR,
    FINAL_MODEL_FILE,
    METRICS_DIR,
    MODEL_METADATA_FILE,
)
from src.data_processing import load_test_split, load_training_split
from src.feature_engineering import build_rf_pipeline


FINAL_METADATA = METRICS_DIR / "final_evaluation_metadata.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_confusion_figure(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    image = ax.imshow(cm, cmap="Blues")
    for (row, column), value in np.ndenumerate(cm):
        ax.text(column, row, f"{value:,}", ha="center", va="center",
                color="white" if value > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Final Test Confusion Matrix (threshold = 0.485)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_test_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_roc_figure(y_true: pd.Series, probabilities: np.ndarray, auc: float) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(fpr, tpr, color="#376A8A", linewidth=2, label=f"Random Forest (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Final Test ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_test_roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pr_figure(y_true: pd.Series, probabilities: np.ndarray, ap: float) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    prevalence = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(recall, precision, color="#B7791F", linewidth=2,
            label=f"Random Forest (AP = {ap:.4f})")
    ax.axhline(prevalence, color="black", linestyle="--", linewidth=1,
               label=f"Fraud prevalence = {prevalence:.4%}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Final Test Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_test_precision_recall_curve.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    if FINAL_METADATA.exists():
        raise RuntimeError(
            "Final evaluation metadata already exists. The locked test evaluation "
            "is single-use; refusing to run again."
        )

    frozen_path = METRICS_DIR / "frozen_model_specification.json"
    split_path = DATA_PROCESSED / "split_metadata.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    selected = frozen["selected_candidate"]
    threshold = float(frozen["threshold_selection"]["threshold"])

    if selected["candidate_id"] != "rf_balanced_200" or threshold != 0.485:
        raise ValueError("Frozen model or threshold does not match the approved Milestone 4 result")

    train_path = DATA_PROCESSED / "train.parquet"
    test_path = DATA_PROCESSED / "test.parquet"
    train_hash_before = sha256(train_path)
    test_hash_before = sha256(test_path)
    if train_hash_before != split["train_sha256"] or test_hash_before != split["test_sha256"]:
        raise ValueError("A processed partition checksum does not match split metadata")

    X_train, y_train = load_training_split(DATA_PROCESSED)
    X_test, y_test = load_test_split(DATA_PROCESSED)
    feature_order = frozen["feature_order"]
    if list(X_train.columns) != feature_order or list(X_test.columns) != feature_order:
        raise ValueError("Saved feature order differs from the frozen specification")
    if len(X_train) != frozen["training_rows"] or int(y_train.sum()) != frozen["training_fraud_cases"]:
        raise ValueError("Training partition differs from the frozen specification")

    model = build_rf_pipeline(**selected["params"])
    print("Fitting the frozen pipeline on the complete training partition...")
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(np.int8)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(y_test, probabilities)),
        "threshold": threshold,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "test_rows": int(len(y_test)),
        "test_fraud_cases": int(y_test.sum()),
        "test_fraud_prevalence": float(y_test.mean()),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(METRICS_DIR / "final_test_metrics.csv", index=False)
    (METRICS_DIR / "final_test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    report = pd.DataFrame(classification_report(
        y_test, predictions, labels=[0, 1], target_names=["legitimate", "fraud"],
        output_dict=True, zero_division=0
    )).transpose()
    report.to_csv(METRICS_DIR / "final_test_classification_report.csv")

    error_type = np.select(
        [(y_test.to_numpy() == 1) & (predictions == 1),
         (y_test.to_numpy() == 0) & (predictions == 1),
         (y_test.to_numpy() == 1) & (predictions == 0)],
        ["true_positive", "false_positive", "false_negative"],
        default="true_negative",
    )
    pd.DataFrame({
        "test_row": np.arange(len(y_test)),
        "actual_class": y_test.to_numpy(dtype=np.int8),
        "fraud_probability": probabilities,
        "predicted_class": predictions,
        "error_type": error_type,
    }).to_csv(METRICS_DIR / "final_test_predictions.csv", index=False)

    save_confusion_figure(cm)
    save_roc_figure(y_test, probabilities, metrics["roc_auc"])
    save_pr_figure(y_test, probabilities, metrics["pr_auc_average_precision"])

    joblib.dump(model, FINAL_MODEL_FILE)
    model_hash = sha256(FINAL_MODEL_FILE)
    model_metadata = {
        "model": "Random Forest",
        "candidate_id": selected["candidate_id"],
        "threshold": threshold,
        "n_estimators": selected["params"]["n_estimators"],
        "class_weight": selected["params"]["class_weight"],
        "random_state": selected["params"]["random_state"],
        "feature_order": feature_order,
        "research_use": "MSc prototype; not for production banking use",
    }
    MODEL_METADATA_FILE.write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")

    train_hash_after = sha256(train_path)
    test_hash_after = sha256(test_path)
    if train_hash_after != train_hash_before or test_hash_after != test_hash_before:
        raise RuntimeError("A processed partition changed during final evaluation")

    metadata = {
        "phase": "Milestone 5 single final test evaluation",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_specification": str(frozen_path.relative_to(Path.cwd())),
        "selected_candidate": selected,
        "threshold": threshold,
        "fit_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_evaluation_count": 1,
        "post_test_model_or_threshold_changes_permitted": False,
        "train_sha256_before_and_after": train_hash_after,
        "test_sha256_before_and_after": test_hash_after,
        "model_file": str(FINAL_MODEL_FILE.relative_to(Path.cwd())),
        "model_sha256": model_hash,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "metrics": metrics,
    }
    FINAL_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved: {FINAL_MODEL_FILE}")
    print(f"Model SHA-256: {model_hash}")


if __name__ == "__main__":
    run()
