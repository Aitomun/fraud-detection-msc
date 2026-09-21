# Methodology Decisions

## Decision 001 Reserve the test set for one final evaluation

**Status:** Accepted

**Decision:** Model comparison, imbalance experiments, hyperparameter tuning, and threshold selection will use training data only. The test set will be evaluated once after the complete pipeline and threshold are frozen.

**Rationale:** Repeatedly consulting test performance would allow development choices to adapt to the test set and would produce an optimistically biased estimate of generalization.

## Decision 002 Required classifiers

**Status:** Accepted

**Decision:** Compare Logistic Regression, Decision Tree, and Random Forest.

**Rationale:** These algorithms match the approved project scope and provide linear, single-tree, and ensemble-tree comparisons without unnecessary model complexity.

## Decision 003 Primary ranking metric

**Status:** Accepted provisionally

**Decision:** Use cross-validated average precision or PR-AUC as the primary ranking metric, supported by precision, recall, F1, ROC-AUC, fold stability, and error counts.

**Rationale:** PR-AUC focuses on minority-class retrieval quality and is more informative than accuracy for severe class imbalance. The decision will be revisited only if the verified dataset or research requirements materially differ.

## Decision 004 Duplicate handling

**Status:** Accepted and implemented

**Evidence:** The audit found 1,081 exact duplicate rows after the first occurrence: 1,062 legitimate and 19 fraudulent duplicate copies. It found no feature-identical groups with conflicting labels.

**Decision:** Exact duplicates were removed before the stratified split while retaining the first occurrence. The cleaned dataset contains 283,726 observations, including 473 fraud cases. Hash-based validation confirmed zero identical-row overlap between training and test partitions.

## Decision 006 Reproducible partition

**Status:** Accepted and implemented

**Decision:** Create one 80/20 split using `stratify=Class` and `random_state=42`. Save the untouched feature values and target to Parquet together with checksums and provenance metadata.

**Evidence:** The training partition contains 226,980 observations and 378 fraud cases. The test partition contains 56,746 observations and 95 fraud cases. The partition files and metadata are stored in `data/processed`.

**Constraint:** Development code for cross-validation, imbalance experiments, tuning, and threshold selection must not load `test.parquet`.

## Decision 005 Candidate amount treatment

**Status:** Rejected for the final pipeline after training-only comparison

**Evidence:** Amount has a mean of 88.35, median of 22.00, and maximum of 25,691.16, indicating strong right skew.

**Decision:** Logistic Regression candidates compared robust scaling of the original amount with `log1p(Amount)` inside each cross-validation fold. The best log-transformed Logistic Regression PR-AUC was 0.7561, while no Logistic Regression candidate approached the Random Forest candidates. Because monotonic transformations do not change the ordering available to ordinary tree splits, the selected Random Forest retains the original amount values.

## Decision 007 Baseline configurations

**Status:** Accepted and executed

**Decision:** Establish unweighted, non-resampled reference results for Logistic Regression, Decision Tree, and Random Forest using identical shuffled stratified five-fold splits of the training data.

**Configuration:** Logistic Regression used robust scaling for `Time` and `Amount`, standard scaling for V1 through V28, C=1.0, and lbfgs. Decision Tree used its unconstrained defaults with `random_state=42`. Random Forest used 100 trees, square-root feature sampling, and `random_state=42`. All classifiers used `class_weight=None`.

**Rationale:** These configurations provide a controlled reference against which class weighting, resampling, feature treatment, and tuning can be compared. They are not assumed to be optimal.

## Decision 008 Baseline leader is not the final model

**Status:** Accepted

**Evidence:** Random Forest leads the original-distribution baselines with mean PR-AUC 0.8334 and F1 0.8477. Its recall standard deviation is 0.0920, and no imbalance or tuning experiment has yet been performed.

**Decision:** Retain all three algorithms for Milestone 4. Do not select or save a final model based solely on the baseline results.

## Decision 009 Controlled imbalance and tuning scope

**Status:** Accepted and executed

**Decision:** First compare `class_weight="balanced"` with the original-distribution baselines for all three algorithms. Then use a bounded search to test the pending Logistic Regression amount transformation, Logistic Regression SMOTE, additional trees and weighting variants for Random Forest, and one Random Forest SMOTE candidate. SMOTE is fitted inside each training fold only. The weighted Decision Tree is not expanded into a large grid because its PR-AUC fell from 0.5711 to 0.5417 and its variability increased.

**Rationale:** This staged design answers the imbalance question while avoiding a computationally wasteful Cartesian grid after training evidence has identified weak branches.

## Decision 010 Final model configuration

**Status:** Accepted and frozen before test evaluation

**Decision:** Select the 200-tree Random Forest with Gini splits, square-root feature sampling, unconstrained depth, `min_samples_split=2`, `min_samples_leaf=1`, `class_weight="balanced"`, and `random_state=42`. Retain all 30 predictors in their saved order and use their original numerical values.

**Evidence:** This candidate ranked first among 17 configurations with mean five-fold PR-AUC 0.8368 (standard deviation 0.0321), mean precision 0.9213, mean recall 0.7936, and mean F1 0.8515. Its PR-AUC advantage over the unweighted 200-tree forest was small (0.0004), but it also had lower PR-AUC variability and higher mean recall and F1.

## Decision 011 Operating threshold

**Status:** Accepted and frozen before test evaluation

**Decision:** Use probability threshold 0.485, selected by maximum F1 from one out-of-fold training probability per observation. At this threshold, the pooled training-only results were precision 0.9219, recall 0.8122, and F1 0.8636, with 307 true positives, 26 false positives, 71 false negatives, and 226,576 true negatives.

**Rationale:** No application-specific false-positive/false-negative cost ratio was supplied, so F1 provides a reproducible balance between precision and recall. The threshold must not be changed after the final test results are seen.

## Decision 012 Final evaluation is immutable

**Status:** Accepted and executed

**Decision:** Fit the frozen pipeline on all 226,980 training observations and evaluate it once on the 56,746-observation test partition using threshold 0.485. Persist the model, predictions, metrics, figures, partition checksums, and model checksum. Do not revise the model, preprocessing, predictors, or threshold in response to test performance.

**Evidence:** Pre- and post-evaluation SHA-256 checks confirmed that both processed partitions were unchanged. The evaluator records `test_evaluation_count=1` and refuses to run again while the final-evaluation metadata exists.

## Decision 013 SHAP explanation design

**Status:** Accepted and executed

**Decision:** Use SHAP TreeExplainer on the immutable fitted Random Forest and explain the class 1 model output. Estimate global importance on a deterministic simple random sample of 2,000 training observations. Supplement this prevalence-respecting sample with an explicitly labelled cohort comparison using all 378 training fraud observations and 2,000 randomly sampled legitimate training observations. Select one local example from each available final-test outcome category by choosing the probability nearest that category's median.

**Rationale:** The global sample reflects the strong class imbalance, while the supplementary cohorts reveal whether importance differs for known fraud cases. The median-probability rule avoids selecting visually dramatic local cases after inspecting their explanations. Representative local explanations cover successful and unsuccessful decisions.

**Constraints:** SHAP values describe contributions to this fitted model's output, not causal effects. The approximately 0.5001 TreeExplainer baseline is a reference output of the class-weighted forest and is not the observed fraud prevalence. V1 through V28 remain anonymized and must not receive invented meanings.

## Decision 014 Application inference boundary

**Status:** Accepted and implemented

**Decision:** Place artifact loading, exact-schema validation, and prediction in a Streamlit-independent service module. The UI must use the saved 30-feature order and frozen threshold of 0.485. It must reject missing, extra, nonnumeric, missing, infinite, and negative-Amount inputs. The threshold is displayed but cannot be adjusted after final evaluation.

**Rationale:** Separating the inference boundary makes validation and deterministic behaviour testable without browser interaction. Fixing the threshold ensures that the demonstration represents the evaluated pipeline rather than an unreported post-test configuration.

## Decision 015 Application test strategy

**Status:** Accepted and executed

**Decision:** Use the Python standard-library `unittest` framework and Streamlit's installed `AppTest` support. Test artifact failures, schema validation, deterministic inference, representative final outcomes, SHAP fraud-class extraction and additivity, initial rendering, the Predict-button path, displayed metrics, and a real local server health endpoint.

**Evidence:** All 11 automated tests passed. The live `/_stcore/health` endpoint returned HTTP 200 with response `ok`, after which the verified project-local server processes were stopped.
