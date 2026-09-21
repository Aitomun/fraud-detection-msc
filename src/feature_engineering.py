"""
src/feature_engineering.py
==========================
Builds Scikit-learn pipelines with preprocessing for each model type.

Key rules enforced here:
- Scalers are fitted ONLY on training data.
- Decision Tree and Random Forest do NOT require scaling.
- SMOTE is applied ONLY to training data, never the test set.
- Pipelines make the process reproducible and leakage-free.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# imbalanced-learn pipeline (drop-in for sklearn Pipeline when using SMOTE)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RANDOM_STATE


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
# Amount and Time need scaling for Logistic Regression.
# V1–V28 are already PCA-transformed so are on a comparable scale,
# but StandardScaler is still applied for LR numerical stability.

SCALE_COLS  = ["Time", "Amount"]              # these differ in magnitude from V1–V28
V_COLS      = [f"V{i}" for i in range(1, 29)] # already PCA-transformed; scaled for LR
ALL_NUM     = SCALE_COLS + V_COLS             # all 30 predictor columns


# ---------------------------------------------------------------------------
# Preprocessors
# ---------------------------------------------------------------------------

def build_lr_preprocessor(amount_transform: str = "original") -> ColumnTransformer:
    """
    Standard scaler applied to all numerical features.
    Used for Logistic Regression which is sensitive to feature magnitudes.
    RobustScaler is used for Amount/Time to handle outliers; StandardScaler for V-cols.
    """
    if amount_transform == "original":
        amount_pipeline = RobustScaler()
    elif amount_transform == "log1p":
        amount_pipeline = Pipeline([
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("robust_scale", RobustScaler()),
        ])
    else:
        raise ValueError("amount_transform must be 'original' or 'log1p'")

    preprocessor = ColumnTransformer(
        transformers=[
            ("time_scale",   RobustScaler(), ["Time"]),
            ("amount",       amount_pipeline, ["Amount"]),
            ("std_scale",    StandardScaler(), V_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def build_tree_preprocessor() -> ColumnTransformer:
    """
    Pass-through transformer for tree-based models (DT, RF).
    No scaling is needed; included so Pipeline structure is consistent.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("passthrough", "passthrough", ALL_NUM),
        ],
        remainder="drop",
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Full model pipelines — standard (no SMOTE)
# ---------------------------------------------------------------------------

def build_lr_pipeline(amount_transform: str = "original", **lr_kwargs) -> Pipeline:
    """
    Logistic Regression pipeline.
    Default kwargs set baseline configuration; overridden during tuning.
    """
    defaults = dict(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    defaults.update(lr_kwargs)

    return Pipeline([
        ("preprocessor", build_lr_preprocessor(amount_transform)),
        ("classifier",   LogisticRegression(**defaults)),
    ])


def build_dt_pipeline(**dt_kwargs) -> Pipeline:
    """Decision Tree pipeline (no scaling required)."""
    defaults = dict(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    defaults.update(dt_kwargs)

    return Pipeline([
        ("preprocessor", build_tree_preprocessor()),
        ("classifier",   DecisionTreeClassifier(**defaults)),
    ])


def build_rf_pipeline(**rf_kwargs) -> Pipeline:
    """Random Forest pipeline (no scaling required)."""
    defaults = dict(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    defaults.update(rf_kwargs)

    return Pipeline([
        ("preprocessor", build_tree_preprocessor()),
        ("classifier",   RandomForestClassifier(**defaults)),
    ])


# ---------------------------------------------------------------------------
# SMOTE pipelines — training-only oversampling via imblearn Pipeline
# NOTE: SMOTE must be fitted inside cross-validation folds, never on the
#       whole training set before CV.  Use these pipelines with
#       cross_validate / GridSearchCV so SMOTE is re-fitted per fold.
# ---------------------------------------------------------------------------

def build_lr_smote_pipeline(
    amount_transform: str = "original", **lr_kwargs
) -> ImbPipeline:
    """Logistic Regression pipeline with SMOTE oversampling."""
    defaults = dict(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    defaults.update(lr_kwargs)

    return ImbPipeline([
        ("preprocessor", build_lr_preprocessor(amount_transform)),
        ("smote",        SMOTE(random_state=RANDOM_STATE)),
        ("classifier",   LogisticRegression(**defaults)),
    ])


def build_rf_smote_pipeline(**rf_kwargs) -> ImbPipeline:
    """Random Forest pipeline with SMOTE oversampling."""
    defaults = dict(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    defaults.update(rf_kwargs)

    return ImbPipeline([
        ("preprocessor", build_tree_preprocessor()),
        ("smote",        SMOTE(random_state=RANDOM_STATE)),
        ("classifier",   RandomForestClassifier(**defaults)),
    ])
