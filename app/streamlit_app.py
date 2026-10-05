"""
app/streamlit_app.py
====================
Research prototype — NOT a production banking system.

This Streamlit application:
1. Loads the saved preprocessing pipeline and final model.
2. Offers two interaction modes:
   Option A — manual numeric input for all features.
   Option B — sample transaction selector (from the test set).
3. Generates a prediction and fraud probability.
4. Displays a SHAP local explanation.

NOTE: V1–V28 are anonymized/transformed features. The UI presents them
as numerical inputs rather than attempting to assign business meanings
that are not present in the dataset.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FINAL_MODEL_FILE,
    MODEL_METADATA_FILE,
    DATA_PROCESSED,
    DATA_SAMPLE,
    TARGET_COL,
)
from src.explain import (
    get_transformed_data,
    get_feature_names,
    build_shap_explainer,
    positive_class_expected_value,
    positive_class_shap_values,
)
from app.model_service import (
    FEATURE_COLS,
    load_model_artifacts,
    predict_transactions,
    validate_input_frame,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection — MSc Prototype",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Model loading (cached so it only happens once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model …")
def load_artifacts():
    """Load and validate the final model and metadata."""
    return load_model_artifacts(FINAL_MODEL_FILE, MODEL_METADATA_FILE)


@st.cache_resource(show_spinner="Building SHAP explainer …")
def get_explainer(_model, X_train_sample):
    """Build a SHAP explainer from the training sample."""
    X_t = get_transformed_data(_model, X_train_sample)
    return build_shap_explainer(_model, X_t)


@st.cache_data(show_spinner="Loading test set …")
def load_test_set():
    """Load the full test set, or the committed 1,000-row sample if it is absent."""
    test_path = DATA_PROCESSED / "test.parquet"
    if not test_path.exists():
        test_path = DATA_SAMPLE / "test_sample.parquet"
    if not test_path.exists():
        return None, None
    test = pd.read_parquet(test_path)
    X    = test.drop(columns=[TARGET_COL])
    y    = test[TARGET_COL]
    return X, y


# ---------------------------------------------------------------------------
# SHAP waterfall helper (returns a matplotlib figure)
# ---------------------------------------------------------------------------

def shap_waterfall_fig(model, explainer, X_single: pd.DataFrame, feature_names: list) -> plt.Figure:
    X_t = get_transformed_data(model, X_single)
    sv = positive_class_shap_values(explainer.shap_values(X_t))[0]
    expected = positive_class_expected_value(explainer)

    exp = shap.Explanation(
        values        = sv,
        base_values   = expected,
        data          = X_t[0],
        feature_names = feature_names,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(exp, show=False)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    st.title("🔍 Credit Card Fraud Detection")
    st.caption("MSc Research Prototype ")

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    try:
        model, metadata = load_artifacts()
    except (FileNotFoundError, ValueError) as error:
        st.error(
            "Model artifacts are unavailable or invalid. "
            "Complete Milestone 5 before launching this application."
        )
        st.code(str(error))
        return

    # ------------------------------------------------------------------
    # Sidebar — model info
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Model Information")
        st.write(f"**Classifier:** {metadata['model']}")
        st.write(f"**Trees:** {metadata['n_estimators']}")
        st.write(f"**Class weighting:** {metadata['class_weight']}")
        st.write(f"**Features:** {len(metadata['feature_order'])}")

        st.divider()
        threshold = float(metadata["threshold"])
        st.metric("Frozen threshold", f"{threshold:.3f}")
        st.caption("Fixed during training-only model selection; not adjustable after test evaluation.")

    # ------------------------------------------------------------------
    # Mode selector
    # ------------------------------------------------------------------
    mode = st.radio(
        "Input mode",
        ["Option A — Manual feature input", "Option B — Sample transaction selector"],
        horizontal=True,
    )

    # ------------------------------------------------------------------
    # Option A: manual input
    # ------------------------------------------------------------------
    if mode.startswith("Option A"):
        st.subheader("Transaction Features")
        st.info(
            "V1–V28 are anonymized/transformed numerical features derived from "
            "the original transaction data. Enter numerical values for each feature."
        )

        col1, col2, col3 = st.columns(3)
        input_vals = {}

        with col1:
            input_vals["Time"]   = st.number_input("Time (seconds)", value=0.0, format="%.2f")
            input_vals["Amount"] = st.number_input(
                "Amount (dataset units)", value=1.0, min_value=0.0, format="%.2f"
            )

        for i, vname in enumerate([f"V{j}" for j in range(1, 29)]):
            target_col = col2 if i < 14 else col3
            with target_col:
                input_vals[vname] = st.number_input(vname, value=0.0, format="%.4f")

        X_input = pd.DataFrame([input_vals])[FEATURE_COLS]

    # ------------------------------------------------------------------
    # Option B: sample selector
    # ------------------------------------------------------------------
    else:
        st.subheader("Select a Sample Transaction")
        X_test, y_test = load_test_set()

        if X_test is None:
            st.warning("Sample transactions not found. Run src/run_preprocessing.py first.")
            return

        fraud_filter = st.selectbox(
            "Show transactions of type:",
            ["All", "Legitimate (Class=0)", "Fraudulent (Class=1)"],
        )

        if fraud_filter == "Legitimate (Class=0)":
            X_pool = X_test[y_test == 0]
        elif fraud_filter == "Fraudulent (Class=1)":
            X_pool = X_test[y_test == 1]
        else:
            X_pool = X_test

        max_idx  = min(len(X_pool) - 1, 999)
        sel_idx  = st.slider("Transaction index", 0, max_idx, 0)
        X_input  = X_pool.iloc[[sel_idx]][FEATURE_COLS]

        st.dataframe(X_input.T.rename(columns={X_input.index[0]: "Value"}))

        if y_test is not None:
            true_label = y_test.loc[X_pool.index[sel_idx]]
            st.write(f"**True label:** {'Fraudulent' if true_label == 1 else 'Legitimate'}")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    if st.button("🔎 Predict Transaction", type="primary"):
        try:
            X_input = validate_input_frame(X_input)
            result = predict_transactions(model, metadata, X_input).iloc[0]
        except ValueError as error:
            st.error(f"Invalid transaction: {error}")
            return
        proba = float(result["fraud_probability"])
        pred = int(result["predicted_class"])

        st.divider()
        col_r, col_p = st.columns(2)

        with col_r:
            if pred == 1:
                st.error(f"⚠️  Prediction: **FRAUDULENT**")
            else:
                st.success(f"✅  Prediction: **LEGITIMATE**")

        with col_p:
            st.metric("Fraud Probability", f"{proba:.2%}")
            st.metric("Threshold used", f"{threshold:.3f}")

        # ------------------------------------------------------------------
        # SHAP explanation
        # ------------------------------------------------------------------
        st.divider()
        st.subheader("Why? — SHAP Local Explanation")
        st.caption(
            "The chart below shows which features contributed toward or away from "
            "a fraud prediction for this specific transaction. "
            "SHAP values reflect model behaviour — they do **not** establish causality."
        )

        try:
            explainer     = get_explainer(model, X_input)
            feature_names = get_feature_names(model, X_input)

            fig = shap_waterfall_fig(model, explainer, X_input, feature_names)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            # Tabular breakdown
            X_t = get_transformed_data(model, X_input)
            sv = positive_class_shap_values(explainer.shap_values(X_t))[0]

            feat_shap = pd.Series(sv, index=feature_names).sort_values()

            st.subheader("Feature Contributions")
            col_f, col_l = st.columns(2)
            with col_f:
                st.write("**Toward Fraud** ↑")
                toward_fraud = feat_shap.tail(10)[::-1]
                for feat, val in toward_fraud.items():
                    st.write(f"`{feat}` → {val:+.4f}")
            with col_l:
                st.write("**Toward Legitimate** ↓")
                toward_legit = feat_shap.head(10)
                for feat, val in toward_legit.items():
                    st.write(f"`{feat}` → {val:+.4f}")

        except Exception as e:
            st.warning(f"SHAP explanation could not be generated: {e}")

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.divider()
    st.caption(
        "Research Prototype | MSc Dissertation | "
        
        "V1–V28 are anonymized features | "
        "SHAP explains model behaviour, not causality."
    )


if __name__ == "__main__":
    main()
