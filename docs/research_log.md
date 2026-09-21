# Research Log

## 2026-09-05 Project plan finalized

- Confirmed the three required classifiers: Logistic Regression, Decision Tree, and Random Forest.
- Defined average precision or PR-AUC as the primary cross-validation ranking metric.
- Corrected the experimental order so model selection and threshold selection occur before one final test-set evaluation.
- Established Markdown as the continuous dissertation source; Word generation was subsequently deferred until final dissertation preparation by student decision.
- Added a project-local Python 3.12 environment and installed the declared dependencies.
- Added `pyarrow` because the project persists processed partitions as Parquet files.
- Added a reproducible Milestone 1 runner at `src/run_eda.py`.

## 2026-09-05 Milestone 1 completed

- Source: `data/raw/credit_card.csv`.
- SHA-256: `9ffcbfe0553c4350b834b7eb504b71ab83f1d3004b54b15af65c1f222c5cffb2`.
- Verified 284,807 rows, 31 columns, 30 predictors, and binary target `Class`.
- Verified 284,315 legitimate and 492 fraudulent transactions.
- Fraud rate: 0.1727486%; legitimate-to-fraud ratio: approximately 577.88 to 1.
- Missing values: 0. Infinite values: 0.
- Exact duplicate rows after retaining the first occurrence: 1,081.
- Duplicate copies by class: 1,062 legitimate and 19 fraudulent.
- Feature-identical groups with conflicting class labels: 0.
- Generated six EDA figures and nine metric or audit files.
- Visually inspected all generated figures; no clipping, overlap, or illegible labels were found.
- No model was trained and the test set has not been created or evaluated.
- Milestone 1 status: complete. Milestone 2 is next.

## 2026-09-05 Milestone 2 completed

- Removed 1,081 verified exact duplicate rows before partitioning.
- Cleaned dataset: 283,726 rows, comprising 283,253 legitimate and 473 fraudulent transactions.
- Created one stratified 80/20 split with `random_state=42`.
- Training partition: 226,980 rows, 378 fraud cases, fraud rate 0.1665345%.
- Locked test partition: 56,746 rows, 95 fraud cases, fraud rate 0.1674127%.
- Verified zero identical-row hash overlap between partitions.
- Saved `data/processed/train.parquet`, `data/processed/test.parquet`, and `data/processed/split_metadata.json`.
- Saved `artifacts/metrics/cleaning_split_summary.csv`.
- Verified the saved Parquet files can be reloaded with their expected row counts and schema.
- No preprocessing transformer or classifier was fitted.
- The test set has not been evaluated.
- Word document updates were deferred until the end by student decision; Markdown remains the current dissertation source.
- Milestone 2 status: complete. Milestone 3 baseline cross-validation is next.

## 2026-09-05 Milestone 3 completed

- Loaded `data/processed/train.parquet` only: 226,980 observations and 378 fraud cases.
- Verified that `Class` was absent from the predictor matrix and that all 30 expected features were present.
- Used shuffled stratified five-fold cross-validation with `random_state=42`.
- Evaluated Logistic Regression, Decision Tree, and Random Forest on the original training distribution without class weighting or resampling.
- Logistic Regression mean results: precision 0.8647, recall 0.6136, F1 0.7162, ROC-AUC 0.9752, PR-AUC 0.7535.
- Decision Tree mean results: precision 0.7526, recall 0.7590, F1 0.7544, ROC-AUC 0.8793, PR-AUC 0.5711.
- Random Forest mean results: precision 0.9451, recall 0.7725, F1 0.8477, ROC-AUC 0.9401, PR-AUC 0.8334.
- Random Forest is the baseline leader on PR-AUC and F1, but no final model has been selected.
- Saved 15 fold records to `artifacts/metrics/baseline_cv_fold_results.csv`.
- Saved aggregate results to `artifacts/metrics/baseline_cv_summary.csv` and reproducibility details to `artifacts/metrics/baseline_cv_metadata.json`.
- Generated and visually checked `artifacts/figures/baseline_cv_comparison.png`.
- Recomputed the locked test-file checksum after the run; it still matches the split metadata.
- The baseline script records `test_data_loaded=false` and does not import the test-data loader.
- Milestone 3 status: complete. Milestone 4 is next.

## 2026-09-05 Milestone 4 completed

- Loaded `data/processed/train.parquet` only and retained the locked test policy.
- Compared original-distribution and balanced class-weight configurations for all three required classifiers on the same five stratified folds.
- Evaluated a controlled total of 17 candidates, including Logistic Regression amount-transform and SMOTE variants plus bounded Random Forest refinements and SMOTE.
- Saved 85 complete fold records with no missing metric values.
- Selected the balanced 200-tree Random Forest by mean PR-AUC: 0.8368 +/- 0.0321.
- Selected model mean precision: 0.9213; recall: 0.7936; F1: 0.8515; ROC-AUC: 0.9632.
- Generated one independent out-of-fold probability for each of the 226,980 training observations.
- Froze threshold 0.485 by maximum out-of-fold F1.
- Out-of-fold threshold results: precision 0.9219, recall 0.8122, F1 0.8636, TP 307, FP 26, FN 71, TN 226,576.
- Verified all stored metrics and confusion counts by recomputation.
- Visually inspected `artifacts/figures/oof_threshold_selection.png`; labels, legend, and selected-threshold marker are legible.
- Recomputed the locked test-file SHA-256 as `cc2ffc96c527946cfe04d60108dc2cc7e88b6c4e9cf4e5913880420a9a0c9ff6`, matching split metadata.
- The model-selection metadata records `test_data_loaded=false`; no development script imported the test loader.
- Milestone 4 status: complete. Milestone 5 one-time final evaluation is next.

## 2026-09-05 Milestone 5 completed

- Verified the train and test Parquet checksums against `data/processed/split_metadata.json` before evaluation.
- Loaded the locked test partition for the first and only final evaluation after the candidate and threshold had been frozen.
- Fitted the frozen balanced 200-tree Random Forest on all 226,980 training observations.
- Applied the frozen threshold of 0.485 to all 56,746 test observations, including 95 fraud cases.
- Final accuracy: 0.999489; precision: 0.958333; recall: 0.726316; F1: 0.826347.
- Final specificity: 0.999947; ROC-AUC: 0.942811; PR-AUC/average precision: 0.814841.
- Final confusion counts: TN 56,648; FP 3; FN 26; TP 69.
- Saved the exact final predictions, metrics, classification report, evaluation metadata, and three figures.
- Saved the complete fitted pipeline to `models/final_model.joblib`.
- Model SHA-256: `3b13807fca9bf45934500150647b5208fb94520274e5a72a141bd4ef0790eb91`.
- Recomputed both processed-partition checksums after evaluation; both matched their original values.
- Recomputed every stored metric and confusion count from the saved prediction file; all checks passed.
- Reloaded the persisted model and verified that it expects the frozen 30 input features.
- Visually inspected the final confusion-matrix, ROC, and precision-recall figures.
- No model, preprocessing, predictor, or threshold change was made after observing test performance.
- Milestone 5 status: complete. Milestone 6 SHAP explainability is next.

## 2026-09-05 Milestone 6 completed

- Loaded the immutable fitted pipeline whose SHA-256 matches the final-evaluation metadata.
- Confirmed SHAP 0.52.0 returns binary Random Forest values with shape `(rows, features, classes)` and updated the shared helper to select the fraud-class dimension safely across SHAP versions.
- Recovered clean transformed feature names in the model's actual order: `Time`, `Amount`, and V1 through V28.
- Used TreeExplainer to explain class 1 model output.
- Generated the primary global explanation from a deterministic 2,000-row training sample containing 1,995 legitimate and five fraud observations.
- Generated a supplementary cohort comparison using all 378 training fraud observations and 2,000 sampled legitimate training observations.
- Primary global top features by mean absolute SHAP: V12 0.07967, V14 0.06880, V4 0.05312, V3 0.04874, V11 0.04624, V10 0.04622, and V17 0.03357.
- Fraud-cohort top features: V14 0.11791, V12 0.09111, V10 0.08003, and V17 0.07664.
- Selected representative local cases by probability nearest the median within each final-test outcome category.
- Representative cases: TP row 8,815 at 0.965; TN row 0 at 0.000; FP row 49,255 at 0.625; FN row 47,606 at 0.015.
- Verified a maximum global SHAP additivity error of approximately 2.1 × 10^-11 and checked local additivity for all four cases.
- Visually inspected the two global plots, cohort plot, and four local waterfall plots; no clipping or illegible labels were found.
- Recorded that the approximately 0.5001 SHAP baseline is a class-weighted model reference, not observed fraud prevalence.
- No model, threshold, or final-test metric was changed.
- Milestone 6 status: complete. Milestone 7 Streamlit application and tests is next.

## 2026-09-05 Milestone 7 completed

- Added `app/model_service.py` to isolate model loading, schema validation, and frozen-threshold prediction from Streamlit rendering.
- Enforced the exact 30-feature schema and canonical feature order.
- Added clear rejection for missing features, extra fields, nonnumeric or missing values, infinite values, and negative Amount.
- Replaced the adjustable UI threshold with the frozen value 0.485.
- Changed the unsupported currency-specific Amount label to `Amount (dataset units)`.
- Updated the UI's SHAP processing for SHAP 0.52 `(rows, features, classes)` output.
- Removed unnecessary training-data loading from interactive local explanation generation.
- Added 11 `unittest` tests covering validation, artifact errors, metadata, deterministic prediction, all four representative outcomes, SHAP shape/additivity, Streamlit rendering, the Predict button, metrics, and the local explanation path.
- All 11 tests passed with zero failures.
- Launched the project-local Streamlit server in headless mode and received HTTP 200 with `ok` from `http://127.0.0.1:8501/_stcore/health`.
- Stopped both verified project-local server processes after the health check.
- Recomputed the model SHA-256 after testing; it remained `3b13807fca9bf45934500150647b5208fb94520274e5a72a141bd4ef0790eb91`.
- Saved machine-readable evidence to `artifacts/metrics/application_test_summary.json`.
- Milestone 7 status: complete. Milestone 8 dissertation completion is next.

## 2026-09-05 Milestone 8 completed

- Finalized the continuously maintained `docs/dissertation.md` evidence source at approximately 7,000 words.
- Added the abstract, expanded introduction, completed literature review, literature-linked discussion, complete conclusion, explicit answers to all five research questions, future-work recommendations, 13 academic references, and a reproducibility appendix.
- Removed all obsolete pending statements and corrected the final test-set history.
- Added manuscript references for 4 verified result tables and 12 saved experimental figures.
- Replaced the milestone-only DOCX builder with a reproducible final builder driven by the Markdown source.
- Generated `dissertation.docx`, refreshed its table of contents in Microsoft Word, and exported it for visual verification.
- Inspected all 34 rendered pages at full resolution and corrected list numbering, table row splitting, a blank page, cover-page headers, and heading spacing.
- Verified 9 Heading 1 paragraphs, 34 Heading 2 paragraphs, 12 inline figures, and no floating image anchors.
- The accessibility audit reported zero high-, medium-, or low-severity findings.
- Final dissertation SHA-256: `b578479bb407ecd0148b290c6e1f284a757a4712bea1e62cab7d671bce699957`.
- Milestone 8 status: complete. All planned project milestones are complete.
