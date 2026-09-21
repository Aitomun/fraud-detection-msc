"""Automated tests for the Streamlit application's model boundary."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.model_service import (
    FEATURE_COLS,
    load_model_artifacts,
    predict_transactions,
    validate_input_frame,
)
from config import DATA_PROCESSED, FINAL_MODEL_FILE, MODEL_METADATA_FILE
from src.explain import (
    get_transformed_data,
    positive_class_expected_value,
    positive_class_shap_values,
)


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.valid = pd.DataFrame([{column: 0.0 for column in FEATURE_COLS}])

    def test_valid_input_is_reordered(self):
        reversed_frame = self.valid[list(reversed(FEATURE_COLS))]
        result = validate_input_frame(reversed_frame)
        self.assertEqual(result.columns.tolist(), FEATURE_COLS)

    def test_missing_feature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_input_frame(self.valid.drop(columns="V28"))

    def test_extra_feature_is_rejected(self):
        frame = self.valid.assign(Class=0)
        with self.assertRaisesRegex(ValueError, "extra"):
            validate_input_frame(frame)

    def test_missing_and_infinite_values_are_rejected(self):
        for value in ["not-a-number", np.nan, np.inf, -np.inf]:
            frame = self.valid.copy()
            if isinstance(value, str):
                frame["V1"] = frame["V1"].astype(object)
            frame.loc[0, "V1"] = value
            with self.assertRaises(ValueError):
                validate_input_frame(frame)

    def test_negative_amount_is_rejected(self):
        frame = self.valid.copy()
        frame.loc[0, "Amount"] = -1
        with self.assertRaisesRegex(ValueError, "negative"):
            validate_input_frame(frame)


class ArtifactAndPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metadata = load_model_artifacts(
            FINAL_MODEL_FILE, MODEL_METADATA_FILE
        )
        cls.test = pd.read_parquet(DATA_PROCESSED / "test.parquet")

    def test_missing_model_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.joblib"
            with self.assertRaisesRegex(FileNotFoundError, "Model artifact"):
                load_model_artifacts(missing, MODEL_METADATA_FILE)

    def test_metadata_freezes_expected_schema_and_threshold(self):
        self.assertEqual(self.metadata["feature_order"], FEATURE_COLS)
        self.assertEqual(self.metadata["threshold"], 0.485)
        self.assertEqual(self.model.n_features_in_, 30)

    def test_prediction_is_deterministic(self):
        frame = self.test.drop(columns="Class").iloc[[0]]
        first = predict_transactions(self.model, self.metadata, frame)
        second = predict_transactions(self.model, self.metadata, frame)
        pd.testing.assert_frame_equal(first, second)

    def test_representative_outcomes_match_saved_final_predictions(self):
        cases = pd.read_csv(PROJECT_ROOT / "artifacts/shap/representative_cases.csv")
        features = self.test.drop(columns="Class")
        for case in cases.itertuples(index=False):
            result = predict_transactions(
                self.model, self.metadata, features.iloc[[int(case.test_row)]]
            ).iloc[0]
            self.assertEqual(int(result.predicted_class), int(case.predicted_class))
            self.assertAlmostEqual(float(result.fraud_probability), case.fraud_probability, places=12)

    def test_shap_output_is_fraud_class_and_additive(self):
        frame = self.test.drop(columns="Class").iloc[[8815]]
        transformed = get_transformed_data(self.model, frame)
        explainer = shap.TreeExplainer(self.model.named_steps["classifier"])
        values = positive_class_shap_values(explainer.shap_values(transformed))
        baseline = positive_class_expected_value(explainer)
        probability = self.model.predict_proba(frame)[0, 1]
        self.assertEqual(values.shape, (1, 30))
        self.assertAlmostEqual(baseline + values[0].sum(), probability, places=9)


class StreamlitSmokeTest(unittest.TestCase):
    def test_app_renders_without_exception(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py"))
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertIn("Credit Card Fraud Detection", app.title[0].value)
        app.button[0].click().run(timeout=60)
        self.assertEqual(len(app.exception), 0)
        metric_values = {metric.label: metric.value for metric in app.metric}
        self.assertIn("Fraud Probability", metric_values)
        self.assertEqual(metric_values["Threshold used"], "0.485")


if __name__ == "__main__":
    unittest.main()
