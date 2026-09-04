# MSc Project Implementation Specification

## Project Title

**Design and Implementation of an Explainable Machine Learning Model for Financial Fraud Detection**

---

# 1. Project Objective

Build a complete, reproducible machine-learning prototype for detecting fraudulent credit-card transactions using the provided dataset.

The system must:

1. Load and inspect the provided dataset.
2. Perform exploratory data analysis (EDA).
3. Clean and preprocess the data.
4. Investigate and address severe class imbalance.
5. Train and compare:
   - Logistic Regression
   - Decision Tree
   - Random Forest
6. Evaluate the models using appropriate classification metrics.
7. Perform controlled model improvement/tuning.
8. Select a final model based on evidence from the experiments.
9. Apply SHAP-based Explainable AI to explain predictions.
10. Build a simple Streamlit interface for testing individual transactions.
11. Save all important models, metrics, figures, and preprocessing artifacts.
12. Produce results that can be directly used in an MSc dissertation.

Do **not** introduce unnecessary technologies or architectures.

This is an MSc research prototype, not a production banking system.

---

# 2. Dataset

The dataset is:

`credit card.csv`

The dataset contains approximately:

- 284,807 transactions
- 31 columns
- 30 predictor features
- 1 target variable: `Class`

Columns:

```text
Time
V1
V2
V3
V4
V5
V6
V7
V8
V9
V10
V11
V12
V13
V14
V15
V16
V17
V18
V19
V20
V21
V22
V23
V24
V25
V26
V27
V28
Amount
Class
```

Target:

```text
Class = 0 → Legitimate transaction
Class = 1 → Fraudulent transaction
```

The dataset is extremely imbalanced:

- Legitimate: approximately 284,315
- Fraudulent: approximately 492
- Fraud rate: approximately 0.17%

This class imbalance is an important part of the project and must be explicitly investigated.

---

# 3. Important Dataset Rules

## 3.1 Do not invent meanings for V1–V28

`V1` through `V28` are anonymized/transformed numerical features.

Do not rename them or attempt to assign business meanings that are not present in the dataset.

They should remain as numerical model features.

The dissertation can describe them as anonymized/transformed transaction features.

---

## 3.2 Investigate duplicates

The dataset contains approximately 1,081 duplicate rows.

Do not automatically delete them without first investigating them.

Perform:

- exact duplicate count
- duplicate class-label consistency
- duplicate distribution by `Class`
- effect of removing duplicates if appropriate

Document the final decision.

If duplicates are removed, ensure this is done before the train/test split.

---

## 3.3 Missing values

Check for missing values.

If none exist, report this rather than creating unnecessary imputation logic.

---

# 4. Recommended Project Structure

Create a clean Python project:

```text
fraud-detection-msc/
│
├── data/
│   ├── raw/
│   │   └── credit card.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_model_improvement.ipynb
│   └── 05_shap_explainability.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
│
├── models/
│
├── artifacts/
│   ├── metrics/
│   ├── figures/
│   └── shap/
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
├── README.md
└── config.py
```

Keep the architecture simple.

Do not introduce:

- Docker
- Kubernetes
- Kafka
- microservices
- cloud infrastructure
- REST APIs
- React frontend
- databases
- deep learning

unless there is a later explicit requirement.

---

# 5. Exploratory Data Analysis

Before training models, perform proper EDA.

Include:

### Dataset overview

- number of rows
- number of columns
- data types
- missing values
- duplicates
- descriptive statistics

### Target distribution

Show:

- legitimate transaction count
- fraudulent transaction count
- percentages
- bar chart

Explicitly demonstrate the class imbalance.

### Feature analysis

Investigate:

- distributions of numerical features
- `Amount`
- `Time`
- differences between fraud and legitimate transactions
- correlations where useful

Because V1–V28 are transformed/anonymized features, avoid pretending that their numerical values have direct business meanings.

Save important plots into:

```text
artifacts/figures/
```

---

# 6. Data Preprocessing

The preprocessing pipeline must avoid data leakage.

Use:

```text
Train/Test Split
        ↓
Fit preprocessing ONLY on training data
        ↓
Transform training/test data
```

Use stratified splitting because the target is highly imbalanced.

Recommended initial split:

```text
80% training
20% testing
random_state = 42
stratify = Class
```

The test set must remain untouched until final evaluation.

---

# 7. Feature Scaling

Investigate feature scaling appropriately.

Logistic Regression should use scaled numerical features.

Decision Tree and Random Forest do not require scaling.

Use appropriate Scikit-learn pipelines so preprocessing is reproducible and leakage is avoided.

For example:

```text
ColumnTransformer
+
Pipeline
```

Do not manually fit scalers on the entire dataset before splitting.

---

# 8. Class Imbalance

This is one of the most important parts of the project.

Because fraud represents approximately 0.17% of transactions, accuracy alone is not sufficient.

Investigate at least:

### Experiment A — Original class distribution

Train baseline models using the original training distribution.

### Experiment B — Class weighting

Investigate:

```python
class_weight="balanced"
```

for applicable models.

### Optional Experiment C — SMOTE

If implemented, use SMOTE only on the training data.

Never apply SMOTE to the test set.

Use an appropriate imbalanced-learn pipeline.

Example concept:

```text
Original Training Data
        ↓
SMOTE
        ↓
Model
```

The untouched test set must preserve the real-world class distribution.

Do not oversample the test set.

---

# 9. Models

The three primary algorithms required by the project are:

## 9.1 Logistic Regression

Use Logistic Regression as the baseline linear classification model.

Investigate appropriate regularization and class weighting.

Example starting configuration:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
```

Do not assume this exact configuration is optimal.

---

## 9.2 Decision Tree

Use a Decision Tree classifier.

Investigate parameters such as:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `class_weight`

Avoid unnecessarily large trees that overfit.

---

## 9.3 Random Forest

Use Random Forest as the ensemble model.

Investigate:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `class_weight`

Do not assume Random Forest is automatically the best model.

The experimental results must determine the final model.

---

# 10. Baseline Experiment

Train all three models before extensive tuning.

Create a comparison table:

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | |
| Decision Tree | | | | | |
| Random Forest | | | | | |

Also generate confusion matrices.

Save the results.

---

# 11. Evaluation Metrics

Accuracy must NOT be the primary metric because of the severe class imbalance.

Report:

### Required

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

### Strongly recommended

- Precision-Recall curve
- PR-AUC / Average Precision

Explain why precision, recall, F1 and PR-AUC are particularly relevant to fraud detection.

Important:

**Recall measures how many actual fraudulent transactions were detected.**

**Precision measures how many transactions predicted as fraud were actually fraudulent.**

The project should discuss the trade-off between these metrics.

---

# 12. Cross-Validation

Use stratified cross-validation on the training data.

Recommended:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Use cross-validation during model comparison/tuning.

Do not use the final test set for model selection.

The test set should remain untouched until the final evaluation.

---

# 13. Hyperparameter Tuning

After baseline experiments, perform controlled hyperparameter tuning.

Use:

- `GridSearchCV`
or
- `RandomizedSearchCV`

Do not perform enormous searches.

Keep the parameter ranges reasonable.

Tune the three required models where appropriate.

The tuning objective should not simply be accuracy.

Consider optimizing for:

```text
F1
```

or another justified fraud-relevant metric such as:

```text
Average Precision / PR-AUC
```

The chosen scoring metric must be documented and justified.

---

# 14. Threshold Analysis

Do not assume that a probability threshold of exactly 0.50 is always optimal for fraud detection.

For the strongest candidate model, investigate how changing the classification threshold affects:

- Precision
- Recall
- F1-score

Produce a threshold analysis where practical.

Discuss the trade-off.

For example:

```text
Lower threshold
→ potentially detects more fraud
→ recall increases
→ false positives may increase
```

The final threshold should be selected based on the experimental evidence and clearly documented.

---

# 15. Final Model Selection

Do not automatically select Random Forest.

Select the final model based on the results.

Consider:

- Recall
- Precision
- F1-score
- PR-AUC
- ROC-AUC
- confusion matrix
- model complexity
- explainability

Clearly document why the selected model is preferable.

---

# 16. Explainable AI — SHAP

Use SHAP for model explainability.

The purpose is to answer:

> Why did the model classify this transaction as fraudulent or legitimate?

Implement both:

## Global explanation

Show which features generally influence predictions most strongly.

Produce appropriate SHAP visualizations, such as:

- SHAP feature importance
- SHAP beeswarm summary plot
- bar summary plot where appropriate

Save figures to:

```text
artifacts/shap/
```

---

## Local explanation

For an individual transaction, show:

- predicted class
- fraud probability
- features pushing prediction toward fraud
- features pushing prediction toward legitimate

For example:

```text
Prediction: Fraudulent

Fraud Probability: 91.4%

Main contributing features:
V14 → strong contribution toward fraud
V10 → contribution toward fraud
V12 → contribution toward fraud
Amount → smaller contribution
```

Do not claim that SHAP proves causality.

Use language such as:

> "The feature contributed strongly to the model's prediction."

Not:

> "The feature caused the transaction to be fraudulent."

---

# 17. Streamlit Application

Build a simple Streamlit interface.

The application should:

1. Load the saved preprocessing pipeline.
2. Load the saved final model.
3. Accept transaction feature values.
4. Generate a prediction.
5. Display fraud probability.
6. Display predicted class.
7. Display an explanation using SHAP.

Example:

```text
Credit Card Fraud Detection

Transaction Details
-------------------

Time: ______
V1: ______
V2: ______
...
V28: ______
Amount: ______

[Predict Transaction]

Prediction:
Potentially Fraudulent

Fraud Probability:
91.4%

Why?
-------------------
V14     █████████
V10     ███████
V12     █████
Amount  ███
```

The UI should remain simple and suitable for an academic demonstration.

Do not build a complicated frontend.

---

# 18. Important Streamlit Consideration

Because V1–V28 are anonymized numerical features, the UI cannot realistically ask a normal user questions such as:

> "What is the transaction's V14?"

Therefore, make the application clearly a **research prototype/demo**.

Possible approach:

### Option A

Provide all model features as numerical inputs.

### Option B

Provide a sample transaction selector:

```text
Select sample transaction
[ Transaction #12345 ]

[Predict]
```

and allow the user to modify values.

Use the approach that best demonstrates the model while remaining faithful to the actual features.

Do not invent fake business features such as:

```text
Country
Merchant Category
Card Type
Customer Age
```

because these columns do not exist in the dataset.

---

# 19. Model Persistence

Save the final trained components using `joblib`.

For example:

```text
models/
├── final_model.joblib
├── preprocessing_pipeline.joblib
└── model_metadata.json
```

The Streamlit application must load these artifacts rather than retraining the model every time.

---

# 20. Reproducibility

Set random seeds where appropriate.

Use:

```python
random_state=42
```

where supported.

Record:

- preprocessing decisions
- train/test split
- model parameters
- hyperparameters
- evaluation metrics
- final threshold
- final model

Create:

```text
requirements.txt
```

containing the required packages.

Likely packages include:

```text
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
shap
streamlit
joblib
jupyter
```

Only include packages actually used.

---

# 21. Results to Save

Save important results in:

```text
artifacts/metrics/
```

Examples:

```text
baseline_results.csv
tuned_results.csv
cross_validation_results.csv
threshold_results.csv
```

Save figures in:

```text
artifacts/figures/
```

Examples:

```text
class_distribution.png
confusion_matrix_logistic_regression.png
confusion_matrix_decision_tree.png
confusion_matrix_random_forest.png
roc_curve.png
precision_recall_curve.png
threshold_analysis.png
```

SHAP figures:

```text
artifacts/shap/
```

---

# 22. Research Questions

Structure the experiments so that they can answer questions such as:

### RQ1

How effectively can machine-learning models detect fraudulent credit-card transactions?

### RQ2

How do Logistic Regression, Decision Tree, and Random Forest compare in fraud detection performance?

### RQ3

How does severe class imbalance affect fraud detection performance?

### RQ4

Which transaction features have the greatest influence on fraud predictions?

### RQ5

How can SHAP improve the interpretability of machine-learning fraud predictions?

Do not fabricate conclusions before the experiments are performed.

---

# 23. Dissertation-Oriented Outputs

The implementation must produce enough evidence for the following dissertation chapters.

## Chapter 1 — Introduction

Cover:

- fraud detection problem
- motivation
- objectives
- research questions
- significance

## Chapter 2 — Literature Review

Review:

- financial fraud detection
- machine learning for fraud detection
- class imbalance
- Logistic Regression
- Decision Trees
- Random Forest
- explainable AI
- SHAP

## Chapter 3 — Methodology

Document:

- dataset
- preprocessing
- EDA
- train/test split
- class imbalance strategy
- models
- cross-validation
- hyperparameter tuning
- evaluation metrics
- SHAP
- Streamlit

## Chapter 4 — Results

Present:

- EDA findings
- baseline model results
- imbalance experiments
- tuned results
- confusion matrices
- ROC curves
- Precision-Recall curves
- final model
- SHAP results

## Chapter 5 — Discussion

Interpret:

- why models performed differently
- effect of class imbalance
- precision/recall trade-offs
- important features
- explainability findings
- comparison with literature

## Chapter 6 — Conclusion

Cover:

- objectives achieved
- research questions answered
- limitations
- recommendations
- future work

---

# 24. Limitations to Acknowledge

The final project should acknowledge limitations such as:

1. The dataset is a public dataset and may not represent every real-world banking environment.
2. The V1–V28 features are anonymized/transformed.
3. Fraud patterns can change over time.
4. A model trained on this dataset may not generalize directly to another financial institution.
5. SHAP explains model behaviour but does not establish causal relationships.
6. The Streamlit application is a research prototype rather than a production banking system.
7. False positives and false negatives have different financial consequences.

Do not claim the prototype is production-ready.

---

# 25. Testing

Test the application with:

### Legitimate transaction

Verify that the system returns a prediction and probability.

### Fraudulent transaction

Verify that the system returns a prediction and probability.

### Invalid input

Test:

- missing values
- non-numeric values
- impossible values where appropriate

### Model loading

Verify that the Streamlit application handles missing model artifacts appropriately.

### Reproducibility

Running the same input through the saved model should produce consistent results.

---

# 26. Code Quality Requirements

The code should be:

- modular
- readable
- commented where necessary
- reproducible
- reasonably documented

Avoid excessive abstraction.

Prefer clear Python/Scikit-learn code over unnecessary design patterns.

The project should be understandable to an MSc supervisor examining the implementation.

---

# 27. Important Anti-Leakage Rules

These rules are mandatory.

### Never:

- scale the entire dataset before splitting
- perform SMOTE before splitting
- tune hyperparameters using the test set
- select the final model based on test-set performance
- repeatedly inspect the test set while developing the model

### Correct process:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Train/Test Split
     ↓
Training Data
     ↓
Preprocessing / Scaling
     ↓
Imbalance Handling
     ↓
Cross-Validation
     ↓
Hyperparameter Tuning
     ↓
Final Model
     ↓
ONE final evaluation on untouched Test Set
```

---

# 28. What NOT To Do

Do not:

- replace the required models with XGBoost without justification
- introduce neural networks
- use deep learning
- create microservices
- create a React frontend
- create a database
- deploy to Kubernetes
- add cloud infrastructure
- artificially manufacture additional features
- invent meanings for V1–V28
- report accuracy as the main success criterion
- oversample the test set
- claim SHAP proves causality
- fabricate results
- fabricate literature findings
- fabricate performance numbers

The project must be driven by actual experimental results.

---

# 29. Definition of Done

The implementation is complete when:

- [ ] Dataset loaded successfully
- [ ] Dataset characteristics documented
- [ ] Missing values checked
- [ ] Duplicates investigated
- [ ] Class imbalance demonstrated
- [ ] EDA completed
- [ ] Data preprocessing implemented
- [ ] Leakage prevention verified
- [ ] Stratified train/test split implemented
- [ ] Logistic Regression implemented
- [ ] Decision Tree implemented
- [ ] Random Forest implemented
- [ ] Baseline comparison completed
- [ ] Precision calculated
- [ ] Recall calculated
- [ ] F1-score calculated
- [ ] Accuracy calculated
- [ ] ROC-AUC calculated
- [ ] PR-AUC calculated
- [ ] Confusion matrices generated
- [ ] Cross-validation performed
- [ ] Class imbalance strategy investigated
- [ ] Hyperparameter tuning performed
- [ ] Threshold analysis performed where appropriate
- [ ] Final model selected based on evidence
- [ ] SHAP global explanation implemented
- [ ] SHAP local explanation implemented
- [ ] Final model saved
- [ ] Preprocessing pipeline saved
- [ ] Streamlit application implemented
- [ ] Streamlit prediction tested
- [ ] Streamlit SHAP explanation tested
- [ ] Metrics saved
- [ ] Figures saved
- [ ] requirements.txt created
- [ ] README created
- [ ] Results are reproducible
- [ ] No fabricated results
- [ ] No data leakage

---

# 30. Final Instruction to the Agent

Build this project incrementally.

**Do not blindly generate the entire system without validating the dataset and intermediate results.**

Start by:

1. Inspecting `credit card.csv`.
2. Confirming the actual schema and target distribution.
3. Performing EDA.
4. Reporting the findings.
5. Implementing preprocessing.
6. Training the three baseline models.
7. Reporting their actual results.
8. Proceeding to imbalance experiments.
9. Performing tuning.
10. Selecting the final model.
11. Implementing SHAP.
12. Building Streamlit.

At every major stage, preserve the experimental results and explain important decisions.

**Most importantly: do not fabricate any metric, result, feature importance, or conclusion. All reported results must come from actually running the experiments on the provided dataset.**

The goal is a technically sound, reproducible **MSc-level explainable fraud detection prototype**, not an unnecessarily complex production system.