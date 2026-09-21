"""
src/train.py
============
Training helpers:
- run_baseline()          : fit a pipeline, return predictions + probabilities
- cross_validate_model()  : stratified k-fold CV on training data only
- tune_model()            : GridSearchCV / RandomizedSearchCV wrapper
- save_model()            : persist with joblib
- load_model()            : reload from joblib
"""

import json
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    RANDOM_STATE, CV_FOLDS, TUNING_SCORER,
    FINAL_MODEL_FILE, PREPROCESSING_FILE, MODEL_METADATA_FILE,
    MODELS_DIR,
)


# ---------------------------------------------------------------------------
# Baseline fit
# ---------------------------------------------------------------------------

def run_baseline(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit `pipeline` on training data.
    Returns (y_pred, y_proba) on the test set.

    Parameters
    ----------
    pipeline : sklearn or imblearn Pipeline
    X_train, y_train : training data
    X_test           : test features (NOT fitted here, only transformed)

    Returns
    -------
    y_pred  : hard predictions (0 / 1)
    y_proba : probability of class 1 (fraud)
    """
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


# ---------------------------------------------------------------------------
# Stratified cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring:  dict | None = None,
    n_splits: int = CV_FOLDS,
) -> pd.DataFrame:
    """
    Stratified k-fold CV on the training set only.

    Returns a DataFrame with one row per fold and columns for each metric.
    Never touches the test set.
    """
    if scoring is None:
        scoring = {
            "accuracy":          "accuracy",
            "precision":         "precision",
            "recall":            "recall",
            "f1":                "f1",
            "roc_auc":           "roc_auc",
            "pr_auc":            "average_precision",
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    results_df = pd.DataFrame(
        {
            "fit_time_seconds": cv_results["fit_time"],
            "score_time_seconds": cv_results["score_time"],
            **{
                k.replace("test_", ""): v
                for k, v in cv_results.items()
                if k.startswith("test_")
            },
        }
    )

    print(f"CV results (mean +/- std) over {n_splits} folds:")
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        print(f"  {col:22s}: {results_df[col].mean():.4f} +/- {results_df[col].std():.4f}")
    print(f"  {'fit_time_seconds':22s}: {results_df['fit_time_seconds'].mean():.2f} seconds/fold")

    return results_df


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_model(
    pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str  = TUNING_SCORER,
    n_splits: int = CV_FOLDS,
    n_iter: int | None = None,          # if set → RandomizedSearchCV
    verbose: int = 1,
):
    """
    GridSearchCV or RandomizedSearchCV wrapper.

    Parameters
    ----------
    n_iter : int or None
        If None, GridSearchCV is used.
        If int, RandomizedSearchCV is used with that many iterations.

    Returns
    -------
    best_estimator : fitted pipeline with the best hyperparameters
    best_params    : dict of best hyperparameters
    cv_results_df  : full CV results DataFrame
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    SearchClass = RandomizedSearchCV if n_iter is not None else GridSearchCV
    search_kwargs = dict(
        estimator=pipeline,
        param_distributions=param_grid if n_iter else None,
        param_grid=None if n_iter else param_grid,
        scoring=scoring,
        cv=cv,
        refit=True,
        n_jobs=-1,
        verbose=verbose,
        return_train_score=False,
    )
    # Remove the key that doesn't apply
    if n_iter is not None:
        del search_kwargs["param_grid"]
        search_kwargs["n_iter"] = n_iter
    else:
        del search_kwargs["param_distributions"]

    search = SearchClass(**search_kwargs)
    search.fit(X_train, y_train)

    print(f"\nBest {scoring}: {search.best_score_:.4f}")
    print(f"Best params  : {search.best_params_}")

    cv_results_df = pd.DataFrame(search.cv_results_)

    return search.best_estimator_, search.best_params_, cv_results_df


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    model,
    path: Path = FINAL_MODEL_FILE,
    metadata: dict | None = None,
) -> None:
    """Save a fitted pipeline with joblib. Optionally write a JSON metadata file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved -> {path}")

    if metadata is not None:
        meta_path = path.parent / "model_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved -> {meta_path}")


def save_preprocessing_pipeline(pipeline, path: Path = PREPROCESSING_FILE) -> None:
    """Save just the preprocessing pipeline (if needed separately)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Preprocessing pipeline saved -> {path}")


def load_model(path: Path = FINAL_MODEL_FILE):
    """Load a saved model from joblib."""
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model


def load_metadata(path: Path = MODEL_METADATA_FILE) -> dict:
    """Load model metadata JSON."""
    with open(path) as f:
        return json.load(f)
