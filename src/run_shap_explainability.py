"""Generate reproducible global and local SHAP evidence for Milestone 6."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_PROCESSED, FINAL_MODEL_FILE, METRICS_DIR, RANDOM_STATE, SHAP_DIR
from src.data_processing import load_test_split, load_training_split
from src.explain import (
    build_shap_explainer,
    get_feature_names,
    get_transformed_data,
    positive_class_expected_value,
    positive_class_shap_values,
)


GLOBAL_SAMPLE_SIZE = 2000
LOCAL_TOP_N = 10


def representative_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Choose the probability-nearest-to-median row in each outcome category."""
    rows = []
    for category in ["true_positive", "true_negative", "false_positive", "false_negative"]:
        group = predictions[predictions["error_type"] == category].copy()
        if group.empty:
            continue
        median_probability = float(group["fraud_probability"].median())
        chosen = group.loc[(group["fraud_probability"] - median_probability).abs().idxmin()].copy()
        chosen["selection_rule"] = "probability nearest category median"
        chosen["category_size"] = int(len(group))
        rows.append(chosen)
    return pd.DataFrame(rows).reset_index(drop=True)


def plot_global_summary(values: np.ndarray, data: np.ndarray, names: list[str]) -> None:
    shap.summary_plot(values, data, feature_names=names, max_display=20, show=False)
    plt.title("Global SHAP Summary: Training-Distribution Sample")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "global_shap_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_global_importance(importance: pd.DataFrame) -> None:
    top = importance.head(20).sort_values("mean_absolute_shap")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.barh(top["feature"], top["mean_absolute_shap"], color="#376A8A")
    ax.set_xlabel("Mean absolute SHAP value for fraud probability")
    ax.set_title("Global Random Forest Feature Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SHAP_DIR / "global_shap_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_importance(cohorts: pd.DataFrame) -> None:
    top_features = cohorts.sort_values(
        "fraud_mean_absolute_shap", ascending=False
    ).head(15)["feature"].tolist()
    view = cohorts.set_index("feature").loc[top_features].iloc[::-1]
    y = np.arange(len(view))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y - 0.18, view["fraud_mean_absolute_shap"], height=0.36,
            label="Fraud cohort (all 378)", color="#B44C43")
    ax.barh(y + 0.18, view["legitimate_mean_absolute_shap"], height=0.36,
            label="Legitimate sample (2,000)", color="#376A8A")
    ax.set_yticks(y, view.index)
    ax.set_xlabel("Mean absolute SHAP value for fraud probability")
    ax.set_title("SHAP Importance by Actual-Class Cohort")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SHAP_DIR / "shap_importance_by_class_cohort.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)


def plot_local_case(
    values: np.ndarray,
    transformed: np.ndarray,
    names: list[str],
    base_value: float,
    category: str,
    probability: float,
) -> None:
    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=transformed,
        feature_names=names,
    )
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f"{category.replace('_', ' ').title()} (fraud probability = {probability:.3f})")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / f"local_shap_{category}.png", dpi=180, bbox_inches="tight")
    plt.close()


def run() -> None:
    final_metadata = json.loads(
        (METRICS_DIR / "final_evaluation_metadata.json").read_text(encoding="utf-8")
    )
    threshold = float(final_metadata["threshold"])
    model = joblib.load(FINAL_MODEL_FILE)
    X_train, y_train = load_training_split(DATA_PROCESSED)
    X_test, y_test = load_test_split(DATA_PROCESSED)
    predictions = pd.read_csv(METRICS_DIR / "final_test_predictions.csv")

    if len(predictions) != len(X_test) or not np.array_equal(
        predictions["actual_class"].to_numpy(), y_test.to_numpy()
    ):
        raise ValueError("Saved final predictions do not align with the test partition")

    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    sample = X_train.sample(n=min(GLOBAL_SAMPLE_SIZE, len(X_train)), random_state=RANDOM_STATE)
    sample_targets = y_train.loc[sample.index]
    transformed = get_transformed_data(model, sample)
    feature_names = get_feature_names(model, sample)
    explainer = build_shap_explainer(model, transformed)
    global_values = positive_class_shap_values(explainer.shap_values(transformed))
    base_value = positive_class_expected_value(explainer)

    if global_values.shape != transformed.shape:
        raise ValueError("SHAP output does not match the transformed feature matrix")
    reconstructed = base_value + global_values.sum(axis=1)
    predicted = model.predict_proba(sample)[:, 1]
    max_additivity_error = float(np.max(np.abs(reconstructed - predicted)))
    if max_additivity_error > 1e-6:
        raise ValueError(f"SHAP additivity check failed: {max_additivity_error}")

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": np.abs(global_values).mean(axis=0),
        "mean_signed_shap": global_values.mean(axis=0),
    }).sort_values("mean_absolute_shap", ascending=False).reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    importance.to_csv(SHAP_DIR / "global_shap_importance.csv", index=False)
    pd.DataFrame({
        "training_row_index": sample.index,
        "actual_class": sample_targets.to_numpy(),
        "model_probability": predicted,
    }).to_csv(SHAP_DIR / "global_shap_sample.csv", index=False)
    np.savez_compressed(
        SHAP_DIR / "global_shap_values.npz",
        shap_values=global_values,
        transformed_features=transformed,
        feature_names=np.asarray(feature_names),
        training_row_indices=sample.index.to_numpy(),
    )
    plot_global_summary(global_values, transformed, feature_names)
    plot_global_importance(importance)

    fraud_rows = X_train[y_train == 1]
    legitimate_rows = X_train[y_train == 0].sample(
        n=min(GLOBAL_SAMPLE_SIZE, int((y_train == 0).sum())), random_state=RANDOM_STATE
    )
    fraud_transformed = get_transformed_data(model, fraud_rows)
    legitimate_transformed = get_transformed_data(model, legitimate_rows)
    fraud_values = positive_class_shap_values(explainer.shap_values(fraud_transformed))
    legitimate_values = positive_class_shap_values(
        explainer.shap_values(legitimate_transformed)
    )
    cohort_importance = pd.DataFrame({
        "feature": feature_names,
        "fraud_mean_absolute_shap": np.abs(fraud_values).mean(axis=0),
        "legitimate_mean_absolute_shap": np.abs(legitimate_values).mean(axis=0),
        "fraud_mean_signed_shap": fraud_values.mean(axis=0),
        "legitimate_mean_signed_shap": legitimate_values.mean(axis=0),
    })
    cohort_importance.to_csv(SHAP_DIR / "shap_importance_by_class_cohort.csv", index=False)
    plot_cohort_importance(cohort_importance)

    cases = representative_rows(predictions)
    case_summaries = []
    contribution_rows = []
    for case in cases.itertuples(index=False):
        test_row = int(case.test_row)
        X_case = X_test.iloc[[test_row]]
        transformed_case = get_transformed_data(model, X_case)
        values = positive_class_shap_values(explainer.shap_values(transformed_case))[0]
        reconstructed_probability = float(base_value + values.sum())
        stored_probability = float(case.fraud_probability)
        if not np.isclose(reconstructed_probability, stored_probability, atol=1e-6):
            raise ValueError(f"Local SHAP additivity failed for test row {test_row}")

        plot_local_case(
            values, transformed_case[0], feature_names, base_value,
            str(case.error_type), stored_probability,
        )
        order = np.argsort(np.abs(values))[::-1]
        for rank, feature_index in enumerate(order[:LOCAL_TOP_N], start=1):
            contribution_rows.append({
                "error_type": case.error_type,
                "test_row": test_row,
                "rank": rank,
                "feature": feature_names[feature_index],
                "feature_value": float(transformed_case[0, feature_index]),
                "shap_value_fraud_class": float(values[feature_index]),
                "direction": "toward fraud" if values[feature_index] > 0 else "toward legitimate",
            })
        case_summaries.append({
            "error_type": case.error_type,
            "test_row": test_row,
            "actual_class": int(case.actual_class),
            "predicted_class": int(case.predicted_class),
            "fraud_probability": stored_probability,
            "threshold": threshold,
            "selection_rule": case.selection_rule,
            "category_size": int(case.category_size),
            "shap_baseline_model_output": base_value,
            "reconstructed_probability": reconstructed_probability,
        })

    pd.DataFrame(case_summaries).to_csv(SHAP_DIR / "representative_cases.csv", index=False)
    pd.DataFrame(contribution_rows).to_csv(
        SHAP_DIR / "representative_case_contributions.csv", index=False
    )
    metadata = {
        "phase": "Milestone 6 SHAP explainability",
        "model_file": str(FINAL_MODEL_FILE),
        "model_sha256": final_metadata["model_sha256"],
        "shap_version": shap.__version__,
        "explainer": "TreeExplainer",
        "explained_output": "class 1 fraud probability",
        "shap_baseline_model_output": base_value,
        "baseline_interpretation": "TreeExplainer reference output under the fitted class-weighted forest; not observed fraud prevalence",
        "global_sample": {
            "source": "training partition",
            "selection": "simple random sample without replacement",
            "random_state": RANDOM_STATE,
            "rows": int(len(sample)),
            "fraud_rows": int(sample_targets.sum()),
            "legitimate_rows": int((sample_targets == 0).sum()),
        },
        "maximum_global_additivity_error": max_additivity_error,
        "cohort_comparison": {
            "fraud_source": "all fraud observations in the training partition",
            "fraud_rows": int(len(fraud_rows)),
            "legitimate_source": "simple random training sample without replacement",
            "legitimate_rows": int(len(legitimate_rows)),
            "legitimate_random_state": RANDOM_STATE,
            "purpose": "supplement the prevalence-respecting global sample; not a prevalence estimate",
        },
        "representative_case_selection": "probability nearest the median within each final-test outcome category",
        "representative_cases": case_summaries,
        "limitations": [
            "SHAP describes model behaviour and does not establish causality.",
            "V1 through V28 are anonymized and cannot receive unsupported business meanings.",
            "Global importance is estimated from a deterministic 2,000-row training sample.",
        ],
    }
    (SHAP_DIR / "shap_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(importance.head(10).to_string(index=False))
    print(pd.DataFrame(case_summaries).to_string(index=False))
    print(f"Maximum additivity error: {max_additivity_error:.3e}")


if __name__ == "__main__":
    run()
