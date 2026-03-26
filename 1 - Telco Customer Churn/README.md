# 🔁 End-to-End ML Pipeline — Telco Customer Churn

## Overview
Production-ready Scikit-learn pipeline for predicting customer churn using the Telco dataset.

---

## Project Structure
```
churn_pipeline/
├── churn_pipeline.py          # Main pipeline script
├── telco_churn.csv            # Dataset (auto-generated if missing)
├── saved_models/
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
└── README.md
```

---

## Pipeline Architecture

```
Raw Data
   │
   ▼
┌──────────────────────────────────────┐
│           ColumnTransformer          │
│  ┌─────────────┐  ┌───────────────┐  │
│  │  Numeric    │  │ Categorical   │  │
│  │ StandardScaler│ │OneHotEncoder │  │
│  └─────────────┘  └───────────────┘  │
└──────────────────────────────────────┘
   │
   ▼
┌─────────────────────┐   ┌─────────────────────┐
│  Logistic Regression│   │   Random Forest     │
│   GridSearchCV      │   │   GridSearchCV      │
└─────────────────────┘   └─────────────────────┘
   │                           │
   ▼                           ▼
logistic_regression.joblib   random_forest.joblib
```

---

## How to Run

```bash
python churn_pipeline.py
```

The script will:
1. Auto-generate the Telco dataset (7,043 rows)
2. Preprocess features with `Pipeline` + `ColumnTransformer`
3. Tune both models with `GridSearchCV` (5-fold stratified CV)
4. Print evaluation metrics on the test set
5. Save both pipelines to `saved_models/`

---

## Load a Saved Pipeline

```python
import joblib
import pandas as pd

# Load
pipeline = joblib.load("saved_models/logistic_regression.joblib")

# Predict on new data
new_customer = pd.DataFrame([{
    "gender": "Female", "SeniorCitizen": 0,
    "Partner": "Yes", "Dependents": "No",
    "tenure": 6, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50, "TotalCharges": 573.0
}])

pred  = pipeline.predict(new_customer)
proba = pipeline.predict_proba(new_customer)[:, 1]
print(f"Churn: {bool(pred[0])}  |  Probability: {proba[0]:.2%}")
```

---

## Hyperparameter Grids

### Logistic Regression
| Parameter    | Values Searched        |
|--------------|------------------------|
| `C`          | 0.01, 0.1, 1.0, 10.0  |
| `penalty`    | l1, l2                 |
| `solver`     | liblinear              |

### Random Forest
| Parameter            | Values Searched       |
|----------------------|-----------------------|
| `n_estimators`       | 100, 200              |
| `max_depth`          | None, 10, 20          |
| `min_samples_split`  | 2, 5                  |
| `max_features`       | sqrt, log2            |

---

## Results Summary

| Metric     | Logistic Regression | Random Forest |
|------------|---------------------|---------------|
| Accuracy   | 0.7764              | 0.7750        |
| Precision  | 0.5625              | 0.6667        |
| Recall     | 0.0564              | 0.0125        |
| F1         | 0.1026              | 0.0246        |
| ROC-AUC    | **0.7450** 🏆       | 0.7228        |

> **Logistic Regression wins on ROC-AUC** — the primary metric for imbalanced churn datasets.

---

## Requirements
```
scikit-learn
pandas
numpy
joblib
```

Install: `pip install scikit-learn pandas numpy joblib`
