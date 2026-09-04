"""
src/evaluate.py
===============
Evaluation helpers:
- compute_metrics()          : full classification report as a dict
- print_metrics_table()      : formatted comparison table
- plot_confusion_matrix()    : save confusion matrix PNG
- plot_roc_curves()          : overlay ROC curves for multiple models
- plot_pr_curves()           : overlay Precision-Recall curves
- plot_threshold_analysis()  : precision/recall/F1 vs threshold
- save_metrics_csv()         : persist metric tables to CSV

All plots are saved to artifacts/figures/ (or artifacts/shap/).
Nothing is printed to the test set during model selection.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for scripts & notebooks)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    FIGURES_DIR,
    METRICS_DIR,
    THRESHOLD_RANGE_START,
    THRESHOLD_RANGE_STOP,
    THRESHOLD_RANGE_STEP,
)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_proba: np.ndarray,
    label:   str = "",
) -> dict:
    """
    Compute all required classification metrics.

    Parameters
    ----------
    y_true  : ground-truth labels
    y_pred  : hard predictions (0/1)
    y_proba : probability of class 1

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, pr_auc
    """
    metrics = {
        "model":     label,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc":    round(average_precision_score(y_true, y_proba), 4),
    }

    print(f"\n{'='*50}")
    print(f"METRICS — {label}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:12s}: {v}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraudulent"]))

    return metrics


def print_metrics_table(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Print a formatted comparison table from a list of metric dicts.
    Returns a DataFrame.
    """
    df = pd.DataFrame(metrics_list).set_index("model")
    print("\nMODEL COMPARISON TABLE")
    print(df.to_string())
    return df


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    model_name: str,
    out_dir:    Path = FIGURES_DIR,
) -> Path:
    """Plot and save a confusion matrix heatmap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Fraudulent"],
        yticklabels=["Legitimate", "Fraudulent"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    results: list[dict],   # each dict: {"label": str, "y_true": ..., "y_proba": ...}
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Overlay ROC curves for multiple models on a single plot.

    Parameters
    ----------
    results : list of dicts with keys label, y_true, y_proba
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    for r in results:
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_proba"])
        auc          = roc_auc_score(r["y_true"], r["y_proba"])
        ax.plot(fpr, tpr, label=f"{r['label']} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    plt.tight_layout()

    out_path = out_dir / "roc_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Precision-Recall curves
# ---------------------------------------------------------------------------

def plot_pr_curves(
    results: list[dict],
    out_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Overlay Precision-Recall curves for multiple models.
    PR-AUC is more informative than ROC-AUC for highly imbalanced datasets.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    for r in results:
        prec, rec, _ = precision_recall_curve(r["y_true"], r["y_proba"])
        ap            = average_precision_score(r["y_true"], r["y_proba"])
        ax.plot(rec, prec, label=f"{r['label']} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    plt.tight_layout()

    out_path = out_dir / "precision_recall_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------

def threshold_analysis(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "",
    out_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """
    Compute precision, recall, and F1 across a range of probability thresholds.

    Returns a DataFrame with columns: threshold, precision, recall, f1.
    Saves a plot to figures/.
    """
    thresholds = np.arange(
        THRESHOLD_RANGE_START,
        THRESHOLD_RANGE_STOP,
        THRESHOLD_RANGE_STEP,
    )

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 3),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        })

    df = pd.DataFrame(rows)

    # Plot
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["precision"], label="Precision", marker="o", ms=3)
    ax.plot(df["threshold"], df["recall"],    label="Recall",    marker="s", ms=3)
    ax.plot(df["threshold"], df["f1"],        label="F1-score",  marker="^", ms=3)
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Analysis — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / "threshold_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    return df


# ---------------------------------------------------------------------------
# Metric persistence
# ---------------------------------------------------------------------------

def save_metrics_csv(
    df:       pd.DataFrame,
    filename: str,
    out_dir:  Path = METRICS_DIR,
) -> Path:
    """Save a metrics DataFrame to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=True)
    print(f"Metrics saved → {out_path}")
    return out_path
