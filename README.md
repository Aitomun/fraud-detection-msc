# Explainable ML for Credit-Card Fraud Detection
### MSc Research Prototype

---

## Project Overview

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
│
├── data/
│   ├── raw/                    ← Place credit_card.csv here
│   └── processed/              ← Auto-generated preprocessed splits
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_model_improvement.ipynb
│   └── 05_shap_explainability.ipynb
│
├── src/
│   ├── data_processing.py      ← Load, clean, split
│   ├── feature_engineering.py  ← Scaling, pipeline construction
│   ├── train.py                ← Training, CV, hyperparameter tuning
│   ├── evaluate.py             ← Metrics, plots, threshold analysis
│   └── explain.py              ← SHAP global and local explanations
│
├── models/                     ← Saved joblib artifacts
├── artifacts/
│   ├── metrics/                ← CSV metric tables
│   ├── figures/                ← PNG plots
│   └── shap/                   ← SHAP plots
│
├── app/
│   └── streamlit_app.py        ← Demo UI
│
├── config.py                   ← All paths and settings
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Place the dataset

Copy `credit_card.csv` into:

```
data/raw/credit_card.csv
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run notebooks in order

```bash
jupyter notebook
```

Open and run each notebook in sequence:

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | EDA — shapes, distributions, imbalance |
| `02_preprocessing.ipynb` | Clean → split → scale → save processed data |
| `03_baseline_models.ipynb` | Train LR, DT, RF; generate comparison table |
| `04_model_improvement.ipynb` | Imbalance strategies, tuning, threshold analysis |
| `05_shap_explainability.ipynb` | Global + local SHAP, save the final model |

### 5. Launch Streamlit demo

After the notebooks have been run and models saved:

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

All experiments use `random_state=42`. Re-running the notebooks in order
produces identical results given the same dataset.

---

## Limitations

1. Dataset is public and may not represent every banking environment.
2. V1–V28 are anonymized/transformed; direct business interpretation is not possible.
3. Fraud patterns evolve over time; the model may drift.
4. SHAP explains predictions, not causality.
5. The Streamlit app is a demo, not a production system.
6. False positives and false negatives carry different financial consequences.
