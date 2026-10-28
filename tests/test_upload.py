"""Tests for dataset upload: schema handling, row validation, scoring and the UI."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.model_service import FEATURE_COLS, load_model_artifacts
from app.upload_service import (
    LABEL_COL,
    detect_schema,
    evaluate_labelled,
    normalise_columns,
    read_upload,
    score_upload,
)
from config import DATA_SAMPLE, FINAL_MODEL_FILE, MODEL_METADATA_FILE

SAMPLE = pd.read_parquet(DATA_SAMPLE / "test_sample.parquet").reset_index(drop=True)


class ReadAndNormaliseTests(unittest.TestCase):
    def test_csv_with_header_round_trips(self):
        frame = normalise_columns(read_upload("x.csv", SAMPLE.to_csv(index=False).encode()))
        self.assertEqual(frame.columns.tolist(), FEATURE_COLS + [LABEL_COL])

    def test_headerless_csv_is_read_positionally(self):
        for columns in (FEATURE_COLS, FEATURE_COLS + [LABEL_COL]):
            data = SAMPLE[columns].to_csv(index=False, header=False).encode()
            frame = normalise_columns(read_upload("x.csv", data))
            self.assertEqual(frame.columns.tolist(), columns)
            self.assertEqual(len(frame), len(SAMPLE))

    def test_parquet_is_supported(self):
        frame = read_upload("x.parquet", SAMPLE.to_parquet())
        self.assertEqual(len(frame), len(SAMPLE))

    def test_unsupported_and_empty_files_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            read_upload("x.xlsx", b"")
        with self.assertRaisesRegex(ValueError, "no rows"):
            read_upload("x.csv", b"Time,Amount\n")

    def test_column_spelling_variants_are_mapped(self):
        frame = pd.DataFrame(columns=["time", "V_1", "pca2", " AMOUNT ", "is_fraud", "merchant"])
        self.assertEqual(
            normalise_columns(frame).columns.tolist(),
            ["Time", "V1", "V2", "Amount", LABEL_COL, "merchant"],
        )

    def test_conflicting_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same feature"):
            normalise_columns(pd.DataFrame(columns=["V1", "v_1"]))


class SchemaDetectionTests(unittest.TestCase):
    def test_full_schema_is_pca(self):
        self.assertEqual(detect_schema(SAMPLE), ("pca", []))

    def test_everyday_columns_are_detected(self):
        frame = normalise_columns(pd.DataFrame(columns=["timestamp", "amount", "merchant"]))
        kind, missing = detect_schema(frame)
        self.assertEqual(kind, "everyday")
        self.assertIn("V14", missing)

    def test_partial_schema_lists_missing_features(self):
        self.assertEqual(detect_schema(SAMPLE.drop(columns=["V3", "V28"])), ("partial", ["V3", "V28"]))


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metadata = load_model_artifacts(FINAL_MODEL_FILE, MODEL_METADATA_FILE)

    def test_valid_rows_are_scored_and_invalid_rows_reported(self):
        frame = SAMPLE.head(6).copy()
        frame["V1"] = frame["V1"].astype(object)
        frame.loc[1, "V1"] = "abc"
        frame.loc[2, "V2"] = np.nan
        frame.loc[3, "V5"] = np.inf
        frame.loc[4, "Amount"] = -5
        results, invalid = score_upload(self.model, self.metadata, frame)
        self.assertEqual(results.index.tolist(), [0, 5])
        self.assertEqual(invalid.index.tolist(), [1, 2, 3, 4])
        self.assertTrue(invalid.loc[1, "reason"].startswith("missing or non-numeric"))
        self.assertEqual(invalid.loc[3, "reason"], "infinite value")
        self.assertEqual(invalid.loc[4, "reason"], "negative Amount")

    def test_predictions_use_frozen_threshold(self):
        results, invalid = score_upload(self.model, self.metadata, SAMPLE)
        self.assertTrue(invalid.empty)
        expected = (results["fraud_probability"] >= self.metadata["threshold"]).astype(int)
        self.assertTrue((results["predicted_class"] == expected).all())
        self.assertIn(LABEL_COL, results.columns)

    def test_metrics_for_labelled_upload(self):
        results, _ = score_upload(self.model, self.metadata, SAMPLE)
        metrics = evaluate_labelled(
            results[LABEL_COL], results["fraud_probability"], results["predicted_class"]
        )
        self.assertEqual(int(metrics["confusion_matrix"].to_numpy().sum()), len(SAMPLE))
        self.assertGreater(metrics["pr_auc"], 0.5)

    def test_non_binary_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "only 0"):
            evaluate_labelled([0, 2], [0.1, 0.9], [0, 1])


class UploadModeAppTests(unittest.TestCase):
    def _open_upload_mode(self, source: str) -> AppTest:
        app = AppTest.from_file(str(PROJECT_ROOT / "app/streamlit_app.py"))
        app.run(timeout=30)
        app.radio[0].set_value("Option C — Upload dataset").run(timeout=30)
        app.radio[1].set_value(source).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        return app

    def test_published_format_sample_is_scored(self):
        app = self._open_upload_mode("Demo — 1,000-row test sample (published format)")
        app.button[0].click().run(timeout=120)
        self.assertEqual(len(app.exception), 0)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Rows scored"], "1,000")
        self.assertIn("PR-AUC", metrics)

    def test_everyday_example_shows_explanation_not_predictions(self):
        app = self._open_upload_mode("Demo — everyday-format example")
        self.assertEqual(len(app.button), 0)
        self.assertTrue(any("everyday transaction columns" in e.value for e in app.error))
