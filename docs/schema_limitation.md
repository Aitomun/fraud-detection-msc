# Schema limitation: why everyday transaction data cannot be scored

This note justifies a design decision in the Streamlit prototype. Its wording can be
reused in the dissertation's limitations and future-work sections.

## Decision

The deployed model accepts only data in the published schema of the ULB credit-card
dataset (`Time`, `V1`–`V28`, `Amount`). Uploaded files with everyday transaction
columns (merchant, category, location, card type, …) are detected and explained, not
scored. The app never fills in, estimates or recomputes V1–V28.

## Justification

1. **The transformation is not available.** V1–V28 are principal components of the
   original transaction features. The publishers withheld those features, and the PCA
   loadings and scaling, for confidentiality. Each component is a weighted
   combination of unknown inputs with unknown weights, so there is no function from
   everyday columns to V1–V28. This is a missing transformation, not a labelling
   task.
2. **Recomputing PCA is not equivalent.** Principal components depend on the data
   they are fitted on. Fitting PCA to a different dataset gives axes with different
   meanings, order and signs, so the model's learned `V14` would receive an unrelated
   quantity.
3. **The model's decisions depend on the anonymised components.** In the global SHAP
   analysis (`artifacts/shap/global_shap_importance.csv`), the ten most influential
   features are all V components (V12, V14, V4, V3, V11, V10, V17, V16, V7, V1).
   `Amount` ranks 19th and `Time` 26th of 30. Their mean |SHAP| (0.0047 and 0.0040)
   is about 1/17 and 1/20 of V12's (0.0797). An everyday file supplies only the
   features the model relies on least.
4. **Imputation would fail silently.** Each V component has a mean of about zero.
   Rows with zero-filled components look like an "average" transaction, and the
   random forest would still return a confident-looking probability. This
   training/serving skew gives plausible but meaningless output, which is worse than
   refusing the input.

## Consequence for validity

The results support conclusions about this dataset's feature space only. External
validity to other institutions' raw transaction data is not established, which is a
limitation common to studies that use this benchmark.

## Future work

To score everyday transaction data, train and evaluate a separate model on a labelled
dataset with interpretable features. Examples are simulated card-transaction datasets
with merchant, category, amount, time and location fields. Use the same protocol:
stratified cross-validation, PR-AUC selection, a threshold chosen out-of-fold and
SHAP explanation. Report it as a separate model, not an extension of this one.
