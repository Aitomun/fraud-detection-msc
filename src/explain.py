"""
src/explain.py
==============
SHAP-based Explainable AI module.

Provides:
- build_shap_explainer()    : create the right SHAP explainer for the model type
- compute_shap_values()     : compute SHAP values for a set of instances
- plot_global_summary()     : beeswarm summary plot (global feature importance)
- plot_global_bar()         : bar chart of mean |SHAP| values
- plot_local_waterfall()    : waterfall plot for a single prediction
- explain_single()          : human-readable local explanation dict

All plots are saved to artifacts/shap/.
Language conventions:
- "contributed toward / against" — NOT "caused"
- "the feature influenced the model's prediction" — NOT "caused fraud"
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SHAP_DIR, FIGURES_DIR, RANDOM_STATE


# ---------------------------------------------------------------------------
# Explainer construction
# ---------------------------------------------------------------------------

def build_shap_explainer(fitted_pipeline, X_train_transformed: np.ndarray):
    """
    Choose the appropriate SHAP explainer for the classifier inside the pipeline.

    - TreeExplainer for Random Forest and Decision Tree (fast, exact)
    - LinearExplainer for Logistic Regression
    - KernelExplainer as fallback (slow but model-agnostic)

    Parameters
    ----------
    fitted_pipeline       : a fitted sklearn Pipeline
    X_train_transformed   : training data after preprocessing (numpy array)

    Returns
    -------
    explainer : shap.Explainer instance
    """
    classifier = fitted_pipeline.named_steps.get("classifier")

    if classifier is None:
        # Try imblearn pipeline structure
        classifier = fitted_pipeline[-1]

    clf_type = type(classifier).__name__

    if clf_type in ("RandomForestClassifier", "DecisionTreeClassifier"):
        explainer = shap.TreeExplainer(classifier)
        print(f"Using TreeExplainer for {clf_type}")
    elif clf_type == "LogisticRegression":
        explainer = shap.LinearExplainer(
            classifier,
            X_train_transformed,
            feature_perturbation="interventional",
        )
        print(f"Using LinearExplainer for {clf_type}")
    else:
        print(f"Falling back to KernelExplainer for {clf_type} (this may be slow)")
        background = shap.sample(X_train_transformed, 100, random_state=RANDOM_STATE)
        explainer  = shap.KernelExplainer(classifier.predict_proba, background)

    return explainer


def get_transformed_data(fitted_pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Pass X through the preprocessing steps of a Pipeline to get
    the array that the classifier actually sees.
    """
    # Walk through all steps except the final classifier
    steps = list(fitted_pipeline.named_steps.items())
    arr = X
    for name, step in steps[:-1]:
        if name == "smote":
            continue    # SMOTE is training-only; skip at inference
        arr = step.transform(arr) if hasattr(step, "transform") else arr
    return arr if isinstance(arr, np.ndarray) else np.array(arr)


def get_feature_names(fitted_pipeline, X: pd.DataFrame) -> list[str]:
    """
    Recover feature names after ColumnTransformer preprocessing.
    Falls back to V1–V28 + Time + Amount ordering if get_feature_names_out fails.
    """
    try:
        preprocessor = fitted_pipeline.named_steps.get(
            "preprocessor",
            list(fitted_pipeline.named_steps.values())[0],
        )
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return list(X.columns)


# ---------------------------------------------------------------------------
# Compute SHAP values
# ---------------------------------------------------------------------------

def compute_shap_values(
    explainer,
    X_transformed: np.ndarray,
    max_display_rows: int = 500,
) -> np.ndarray:
    """
    Compute SHAP values.

    For large datasets, subsample to max_display_rows to keep computation fast.
    Returns shap_values for class 1 (fraud probability) as a 2-D array.
    """
    if len(X_transformed) > max_display_rows:
        rng  = np.random.default_rng(RANDOM_STATE)
        idx  = rng.choice(len(X_transformed), size=max_display_rows, replace=False)
        X_use = X_transformed[idx]
    else:
        X_use = X_transformed

    shap_values = explainer.shap_values(X_use)

    # TreeExplainer returns list [class0, class1] for binary classifiers
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]   # class 1 = fraud

    print(f"SHAP values computed: shape {shap_values.shape}")
    return shap_values, X_use


# ---------------------------------------------------------------------------
# Global explanations
# ---------------------------------------------------------------------------

def plot_global_summary(
    shap_values:    np.ndarray,
    X_transformed:  np.ndarray,
    feature_names:  list[str],
    out_dir:        Path = SHAP_DIR,
    max_display:    int  = 20,
) -> Path:
    """
    Beeswarm summary plot showing feature impact on fraud probability.
    Each dot is one transaction; colour = feature value.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    out_path = out_dir / "shap_summary_beeswarm.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_global_bar(
    shap_values:   np.ndarray,
    feature_names: list[str],
    out_dir:       Path = SHAP_DIR,
    max_display:   int  = 20,
) -> Path:
    """
    Bar chart: mean absolute SHAP value per feature (global importance).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_imp  = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    feat_imp.head(max_display).sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global Feature Importance (top {max_display})")
    plt.tight_layout()

    out_path = out_dir / "shap_feature_importance_bar.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Local explanation
# ---------------------------------------------------------------------------

def plot_local_waterfall(
    explainer,
    X_single_transformed: np.ndarray,
    feature_names: list[str],
    index: int = 0,
    out_dir: Path = SHAP_DIR,
) -> Path:
    """
    Waterfall plot for a single transaction.
    Shows which features pushed the prediction toward or away from fraud.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    sv = explainer.shap_values(X_single_transformed)
    if isinstance(sv, list):
        sv = sv[1]

    shap_exp = shap.Explanation(
        values          = sv[0] if sv.ndim == 2 else sv,
        base_values     = explainer.expected_value[1]
                          if isinstance(explainer.expected_value, (list, np.ndarray))
                          else explainer.expected_value,
        data            = X_single_transformed[0],
        feature_names   = feature_names,
    )

    shap.waterfall_plot(shap_exp, show=False)
    plt.tight_layout()
    out_path = out_dir / f"shap_local_waterfall_idx{index}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def explain_single(
    fitted_pipeline,
    explainer,
    X_single: pd.DataFrame,
    feature_names: list[str],
    threshold: float = 0.5,
    top_n: int = 5,
) -> dict:
    """
    Return a human-readable explanation dict for one transaction.

    Returns
    -------
    dict with keys:
        predicted_class, fraud_probability, top_toward_fraud,
        top_toward_legitimate, threshold_used
    """
    X_t     = get_transformed_data(fitted_pipeline, X_single)
    proba   = fitted_pipeline.predict_proba(X_single)[0, 1]
    pred    = int(proba >= threshold)

    sv = explainer.shap_values(X_t)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv[0] if sv.ndim == 2 else sv

    feat_shap = pd.Series(sv, index=feature_names).sort_values()

    explanation = {
        "predicted_class":       "Fraudulent" if pred == 1 else "Legitimate",
        "fraud_probability":     round(float(proba), 4),
        "threshold_used":        threshold,
        "top_toward_fraud":      feat_shap.tail(top_n)[::-1].to_dict(),
        "top_toward_legitimate": feat_shap.head(top_n).to_dict(),
    }

    print("\n--- Local Explanation ---")
    print(f"Prediction        : {explanation['predicted_class']}")
    print(f"Fraud Probability : {explanation['fraud_probability']:.2%}")
    print(f"Threshold         : {threshold}")
    print(f"\nTop features contributing toward FRAUD:")
    for feat, val in explanation["top_toward_fraud"].items():
        print(f"  {feat:20s}  SHAP={val:+.4f}")
    print(f"\nTop features contributing toward LEGITIMATE:")
    for feat, val in explanation["top_toward_legitimate"].items():
        print(f"  {feat:20s}  SHAP={val:+.4f}")

    return explanation
