# Explainable ML for Credit-Card Fraud Detection
### MSc Research Prototype

---

## Project Overview

**Live demo:** https://fraud-detection-msc.streamlit.app/ (free tier; the first load may take a moment while the app wakes up)

This project builds a reproducible, explainable machine-learning prototype for detecting
fraudulent credit-card transactions. It is an **academic research prototype**, not a
production banking system.

**Research Questions addressed:**
- RQ1: How effectively can ML models detect fraudulent transactions?
- RQ2: How do Logistic Regression, Decision Tree, and Random Forest compare?
- RQ3: How does severe class imbalance affect performance?
- RQ4: Which features most influence fraud predictions?
- RQ5: How does SHAP improve interpretability?

---

## Project Structure

```
fraud-detection-msc/
├── data/
│   ├── raw/                    ← Place credit_card.csv here (not tracked; see Setup)
│   ├── processed/              ← Generated train/test splits (not tracked)
│   └── sample/                 ← 1,000-row test sample used by the deployed demo
├── src/
│   ├── data_processing.py      ← Load, clean, split
│   ├── feature_engineering.py  ← Scaling, pipeline construction
│   ├── train.py                ← Training and cross-validation helpers
│   ├── evaluate.py             ← Metrics and plots
│   ├── explain.py              ← SHAP helpers
│   └── run_*.py                ← One script per milestone (see below)
├── app/                        ← Streamlit demo and model-loading service
├── tests/                      ← Automated tests
├── scripts/                    ← Dissertation .docx builder
├── docs/                       ← Dissertation draft and evidence records
├── models/                     ← Saved final model and metadata
├── artifacts/                  ← Metrics, figures and SHAP outputs
├── notebooks/archive/          ← Superseded template notebooks (do not run)
├── config.py                   ← All paths and settings
├── requirements.txt            ← Pinned runtime dependencies (demo)
└── requirements-dev.txt        ← Adds seaborn and python-docx (pipeline, dissertation build)
```

---

## Setup

### 1. Place the dataset

The raw dataset is **not stored in this repository** (99 MB). Download the public
"Credit Card Fraud Detection" dataset (ULB Machine Learning Group, Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and save it as:

```
data/raw/credit_card.csv
```

It is only needed to re-run the pipeline. The Streamlit demo runs from the committed
model (`models/`) and the committed sample (`data/sample/`).

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements-dev.txt   # full pipeline; use requirements.txt for the demo only
```

On Windows PowerShell, this repository currently uses:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Versions are pinned (Python 3.12) because the saved model is a pickled scikit-learn pipeline.

## Phased execution

The leakage-safe execution order and milestone status are maintained in
[`PROJECT_PLAN.md`](PROJECT_PLAN.md). Run only the current phase; do not use the
locked test partition during development.

Completed reproducible phase commands:

```powershell
# Milestone 1: dataset audit and EDA only
.\.venv\Scripts\python.exe -B src\run_eda.py

# Milestone 2: duplicate cleaning and the single stratified split
.\.venv\Scripts\python.exe -B src\run_preprocessing.py

# Milestone 3: training-only baseline cross-validation
.\.venv\Scripts\python.exe -B src\run_baseline_cv.py

# Milestone 4A and 4B: imbalance comparison, model selection, threshold freeze
.\.venv\Scripts\python.exe -B src\run_imbalance_cv.py
.\.venv\Scripts\python.exe -B src\run_milestone4_selection.py

# Milestone 5: single-use final evaluation (do not rerun after completion)
.\.venv\Scripts\python.exe -B src\run_final_evaluation.py

# Milestone 6: global, cohort, and representative local SHAP explanations
.\.venv\Scripts\python.exe -B src\run_shap_explainability.py

# Milestone 7: automated application and inference tests
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The continuous dissertation draft and evidence records are in `docs/`.

### 4. Launch Streamlit demo

After the milestone scripts have been run (or with the committed model artifacts):

```bash
streamlit run app/streamlit_app.py
```

---

## Important Notes

- **No data leakage** — preprocessing is fit only on training data.
- **No accuracy as primary metric** — F1, PR-AUC, and Recall are used.
- **SHAP explains model behaviour** — it does not establish causality.
- **V1–V28 are anonymized** — do not assign business meanings to them.
- This is a research prototype. Do not deploy it to production.

---

## Reproducibility

All experiments use `random_state=42`. Re-running the milestone scripts in order
produces identical results given the same dataset.

---

## Limitations

1. Dataset is public and may not represent every banking environment.
2. V1–V28 are anonymized/transformed; direct business interpretation is not possible.
3. Fraud patterns evolve over time; the model may drift.
4. SHAP explains predictions, not causality.
5. The Streamlit app is a demo, not a production system.
6. False positives and false negatives carry different financial consequences.
