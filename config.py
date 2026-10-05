"""
config.py
=========
Central configuration for the fraud-detection MSc project.
All paths, hyperparameter grids, and fixed seeds live here so that
every notebook and script imports from a single source of truth.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_RAW        = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED  = PROJECT_ROOT / "data" / "processed"
DATA_SAMPLE     = PROJECT_ROOT / "data" / "sample"   # committed demo sample

MODELS_DIR      = PROJECT_ROOT / "models"
ARTIFACTS_DIR   = PROJECT_ROOT / "artifacts"
METRICS_DIR     = ARTIFACTS_DIR / "metrics"
FIGURES_DIR     = ARTIFACTS_DIR / "figures"
SHAP_DIR        = ARTIFACTS_DIR / "shap"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
RAW_CSV         = DATA_RAW / "credit_card.csv"
TARGET_COL      = "Class"
RANDOM_STATE    = 42

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE       = 0.20          # 80 / 20 split
STRATIFY        = True          # always stratify on Class

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_FOLDS        = 5             # StratifiedKFold n_splits

# ---------------------------------------------------------------------------
# Scoring metric used for tuning
# ---------------------------------------------------------------------------
TUNING_SCORER   = "average_precision"   # PR-AUC; justified by class imbalance

# ---------------------------------------------------------------------------
# Model save filenames
# ---------------------------------------------------------------------------
FINAL_MODEL_FILE        = MODELS_DIR / "final_model.joblib"
PREPROCESSING_FILE      = MODELS_DIR / "preprocessing_pipeline.joblib"
MODEL_METADATA_FILE     = MODELS_DIR / "model_metadata.json"

# ---------------------------------------------------------------------------
# Hyperparameter grids (kept deliberately small for an MSc prototype)
# ---------------------------------------------------------------------------
LR_PARAM_GRID = {
    "classifier__C":            [0.01, 0.1, 1.0, 10.0],
    "classifier__solver":       ["lbfgs", "liblinear"],
    "classifier__class_weight": [None, "balanced"],
}

DT_PARAM_GRID = {
    "classifier__max_depth":         [3, 5, 7, 10, None],
    "classifier__min_samples_split": [2, 10, 50],
    "classifier__min_samples_leaf":  [1, 5, 20],
    "classifier__class_weight":      [None, "balanced"],
}

RF_PARAM_GRID = {
    "classifier__n_estimators":      [100, 200],
    "classifier__max_depth":         [5, 10, None],
    "classifier__min_samples_split": [2, 10],
    "classifier__min_samples_leaf":  [1, 5],
    "classifier__max_features":      ["sqrt", "log2"],
    "classifier__class_weight":      [None, "balanced"],
}

# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------
THRESHOLD_RANGE_START = 0.05
THRESHOLD_RANGE_STOP  = 0.96
THRESHOLD_RANGE_STEP  = 0.05
