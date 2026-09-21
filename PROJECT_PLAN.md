# Final MSc Project Execution Plan

## Project objective

Build and evaluate a reproducible, explainable machine-learning research prototype for credit-card fraud detection. The implementation will compare Logistic Regression, Decision Tree, and Random Forest, investigate class imbalance, explain the selected model with SHAP, and provide a simple Streamlit demonstration.

The project is an MSc research prototype. It is not a production banking platform.

## Current progress

- Milestone 1 Dataset audit and exploratory analysis: complete.
- Milestone 2 Cleaning and experimental design: complete.
- Milestone 3 Baseline cross-validation: complete.
- Milestone 4 Imbalance handling tuning and threshold selection: complete.
- Milestone 5 One-time final evaluation: complete.
- Milestone 6 Explainability: complete.
- Milestone 7 Streamlit application and tests: complete.
- Milestone 8 Dissertation completion: complete.
- The locked test set was evaluated once after the model and threshold were frozen.

## Governing rules

1. Numerical claims must come from executed code and saved artifacts.
2. The test set must remain untouched until one final evaluation.
3. Preprocessing, resampling, tuning, model selection, and threshold selection must use training data only.
4. SMOTE, if retained, must run inside cross-validation folds.
5. Accuracy is reported but does not drive model selection.
6. V1 through V28 remain anonymized features; no unsupported meanings will be assigned to them.
7. SHAP explains model behaviour and does not establish causality.
8. The complete fitted preprocessing and classifier pipeline will be saved for the application.

## Algorithms and experimental factors

The required classifiers are:

- Logistic Regression: interpretable linear baseline.
- Decision Tree: interpretable nonlinear baseline.
- Random Forest: ensemble tree model.

For each appropriate classifier, the research will compare the original class distribution and class weighting. SMOTE will be evaluated as an optional training-only strategy. It is not a fourth classifier.

The primary ranking metric is cross-validated average precision, also reported as PR-AUC. Precision, recall, F1, ROC-AUC, fold stability, confusion-matrix counts, explainability, and computational cost will also inform the final decision.

## Leakage-safe execution phases

### Milestone 1 Dataset audit and exploratory analysis

- Verify file integrity, shape, columns, data types, missing and infinite values.
- Verify class counts and percentages.
- Investigate exact duplicates and feature-identical rows with conflicting labels.
- Produce focused EDA tables and figures.
- Identify potential leakage and preprocessing issues.
- Save findings and update the dissertation Markdown evidence source.

No model training is permitted in this milestone.

### Milestone 2 Cleaning and experimental design

- Record and implement the duplicate decision.
- Make one stratified 80/20 train/test split with random state 42.
- Save split provenance and integrity checks.
- Lock the test set from all development decisions.
- Define candidate feature transformations using training data only.
- Update the methodology chapter in Markdown.

**Status:** Completed on 5 September 2026. Word snapshots are deferred until final dissertation preparation by user decision; Markdown remains current.

### Milestone 3 Baseline cross-validation

- Build leakage-safe pipelines for the three classifiers.
- Evaluate unweighted baselines with stratified five-fold cross-validation on training data.
- Record fold-level and aggregate metrics and training time.
- Do not evaluate the test set.

**Status:** Completed on 5 September 2026 using the original training distribution only. The locked test set was not loaded.

### Milestone 4 Imbalance handling tuning and threshold selection

- Compare class weighting and, if justified, SMOTE inside cross-validation folds.
- Conduct controlled hyperparameter searches on training data only.
- Compare configurations using predefined metrics.
- Select the final classifier and preprocessing configuration.
- Select the operating threshold using out-of-fold training predictions.
- Freeze the pipeline, parameters, feature order, and threshold.
- Update the dissertation Markdown evidence source.

**Status:** Completed on 5 September 2026. Seventeen controlled configurations were compared using the same five training-only folds. A 200-tree class-weighted Random Forest was frozen with an out-of-fold threshold of 0.485. The locked test set was not loaded. Word snapshots remain deferred by user decision.

### Milestone 5 Final evaluation

- Fit the frozen pipeline on the complete training set.
- Evaluate it once on the untouched test set.
- Save final metrics, confusion matrix, ROC curve, precision-recall curve, predictions, and metadata.
- Do not change the selected model or threshold after seeing test results.
- Update the results chapter in Markdown.

**Status:** Completed on 5 September 2026. The frozen 200-tree balanced Random Forest was fitted on all 226,980 training observations and evaluated once on 56,746 untouched test observations at threshold 0.485. No post-test model or threshold changes are permitted.

### Milestone 6 Explainability

- Generate global SHAP summary and importance outputs.
- Explain representative true-positive, true-negative, false-positive, and false-negative cases when available.
- Record the limitation created by anonymized features.
- Update the explainability results in Markdown.

**Status:** Completed on 5 September 2026 using SHAP 0.52.0 and TreeExplainer. Global, actual-class cohort, and representative TP/TN/FP/FN explanations were generated for the immutable final model. Additivity and visual-quality checks passed.

### Milestone 7 Streamlit application and tests

- Load the exact frozen pipeline and threshold.
- Support sample transactions and manual numerical input.
- Display prediction, model risk score, and local SHAP contributions.
- Test valid input, invalid input, missing artifacts, deterministic prediction, and representative classes.
- Capture evidence and update the dissertation Markdown.

**Status:** Completed on 5 September 2026. The application loads the frozen fitted pipeline and threshold, supports validated manual and sample inputs, and renders SHAP 0.52-compatible local explanations. Eleven automated tests passed, the full Predict-button path passed, and the live server health endpoint returned HTTP 200.

### Milestone 8 Dissertation completion

- Finalize Chapters 1 through 6 using verified artifacts.
- Add and verify literature citations separately from experimental evidence.
- Check every table, figure, caption, cross-reference, limitation, and claim.
- Generate and visually inspect the final Word document.

**Status:** Completed on 5 September 2026. The authoritative Markdown manuscript now contains the abstract, completed Chapters 1 through 6, 13 verified academic references, 4 results tables, 12 figures, explicit research-question answers, limitations, recommendations, and a reproducibility appendix. The final 34-page Word document was generated, its table of contents was refreshed in Microsoft Word, every rendered page was visually inspected, and heading, image, and accessibility audits passed with no reported accessibility findings.

## Dissertation evidence workflow

`docs/dissertation.md` is the continuously updated working source. The student may rewrite its prose in their own voice while preserving verified facts, tables, figures, and citations.

Supporting records are:

- `docs/research_log.md`: commands, dates, checks, outputs, and blockers.
- `docs/methodology_decisions.md`: decisions and their evidence-based rationale.
- `docs/results_register.md`: verified numerical findings and artifact locations.

`dissertation.docx` generation is deferred until Milestone 8. Until then, Markdown is the authoritative working source. Unverified values remain explicitly pending and are never invented.
