"""
======================================================
  End-to-End ML Pipeline — Telco Customer Churn
======================================================
  - Preprocessing  : Pipeline (scaling + encoding)
  - Models         : Logistic Regression, Random Forest
  - Tuning         : GridSearchCV
  - Export         : joblib
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "telco_churn.csv")
MODEL_DIR  = os.path.join(SCRIPT_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  1.  DATA GENERATION  (synthetic Telco data so the script is self-contained)
# ══════════════════════════════════════════════════════════════════════════
def generate_telco_dataset(n=7043, seed=42):
    """Create a realistic Telco-style churn dataset and save to CSV."""
    rng = np.random.default_rng(seed)

    gender           = rng.choice(["Male", "Female"], n)
    senior_citizen   = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner          = rng.choice(["Yes", "No"], n)
    dependents       = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure           = rng.integers(0, 72, n)
    phone_service    = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines   = rng.choice(["No", "Yes", "No phone service"], n)
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    online_security  = rng.choice(["No", "Yes", "No internet service"], n)
    online_backup    = rng.choice(["No", "Yes", "No internet service"], n)
    device_protection= rng.choice(["No", "Yes", "No internet service"], n)
    tech_support     = rng.choice(["No", "Yes", "No internet service"], n)
    streaming_tv     = rng.choice(["No", "Yes", "No internet service"], n)
    streaming_movies = rng.choice(["No", "Yes", "No internet service"], n)
    contract         = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21])
    paperless_billing= rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method   = rng.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"], n
    )
    monthly_charges  = rng.uniform(18, 120, n).round(2)
    total_charges    = (tenure * monthly_charges + rng.normal(0, 20, n)).clip(0).round(2)

    # Churn logic: short tenure + month-to-month + high charges → more churn
    churn_prob = (
        0.05
        + 0.25 * (contract == "Month-to-month")
        + 0.10 * (internet_service == "Fiber optic")
        + 0.10 * (tenure < 12)
        - 0.05 * (tenure > 48)
        + 0.05 * (senior_citizen == 1)
        - 0.05 * (dependents == "Yes")
    ).clip(0, 1)
    churn = rng.binomial(1, churn_prob).astype(bool)
    churn_label = np.where(churn, "Yes", "No")

    df = pd.DataFrame({
        "gender": gender, "SeniorCitizen": senior_citizen,
        "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        "Churn": churn_label,
    })
    df.to_csv(DATA_PATH, index=False)
    print(f"[DATA] Dataset saved → {DATA_PATH}  (shape: {df.shape})")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  2.  LOAD & PREPROCESS
# ══════════════════════════════════════════════════════════════════════════
def load_and_prepare(path: str):
    df = pd.read_csv(path)
    print(f"\n[LOAD] {df.shape[0]} rows × {df.shape[1]} cols")

    # TotalCharges may be read as string in real Telco CSV
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Target
    df["Churn"] = (df["Churn"].str.strip().str.lower() == "yes").astype(int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print(f"[DATA] Churn rate: {y.mean():.1%}  |  positives: {y.sum()}")
    return X, y


def get_feature_groups(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"[FEAT] Numeric ({len(num_cols)}): {num_cols}")
    print(f"[FEAT] Categorical ({len(cat_cols)}): {cat_cols}")
    return num_cols, cat_cols


# ══════════════════════════════════════════════════════════════════════════
#  3.  BUILD SKLEARN PIPELINES
# ══════════════════════════════════════════════════════════════════════════
def build_preprocessor(num_cols, cat_cols):
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])
    return preprocessor


def build_lr_pipeline(preprocessor):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def build_rf_pipeline(preprocessor):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])


# ══════════════════════════════════════════════════════════════════════════
#  4.  HYPERPARAMETER GRIDS
# ══════════════════════════════════════════════════════════════════════════
LR_PARAM_GRID = {
    "classifier__C":       [0.01, 0.1, 1.0, 10.0],
    "classifier__penalty": ["l1", "l2"],
    "classifier__solver":  ["liblinear"],
}

RF_PARAM_GRID = {
    "classifier__n_estimators":      [100, 200],
    "classifier__max_depth":         [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__max_features":      ["sqrt", "log2"],
}


# ══════════════════════════════════════════════════════════════════════════
#  5.  TRAIN + TUNE
# ══════════════════════════════════════════════════════════════════════════
def tune_pipeline(pipeline, param_grid, X_train, y_train, name="Model"):
    print(f"\n{'─'*55}")
    print(f"  GridSearchCV → {name}")
    print(f"{'─'*55}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        pipeline, param_grid,
        cv=cv, scoring="roc_auc",
        n_jobs=-1, verbose=1, refit=True
    )
    gs.fit(X_train, y_train)
    print(f"  Best CV ROC-AUC : {gs.best_score_:.4f}")
    print(f"  Best params     : {gs.best_params_}")
    return gs


# ══════════════════════════════════════════════════════════════════════════
#  6.  EVALUATE
# ══════════════════════════════════════════════════════════════════════════
def evaluate(gs, X_test, y_test, name="Model"):
    best = gs.best_estimator_
    y_pred  = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall"   : recall_score(y_test, y_pred, zero_division=0),
        "F1"       : f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC"  : roc_auc_score(y_test, y_proba),
    }

    print(f"\n{'═'*55}")
    print(f"  Test Results — {name}")
    print(f"{'═'*55}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    return metrics


# ══════════════════════════════════════════════════════════════════════════
#  7.  EXPORT PIPELINES
# ══════════════════════════════════════════════════════════════════════════
def export_pipeline(gs, name: str):
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(gs.best_estimator_, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"[SAVE] {name} → {path}  ({size_kb:.1f} KB)")
    return path


def load_pipeline(path: str):
    pipeline = joblib.load(path)
    print(f"[LOAD] Pipeline loaded ← {path}")
    return pipeline


# ══════════════════════════════════════════════════════════════════════════
#  8.  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  Telco Customer Churn — ML Pipeline")
    print("=" * 55)

    # ── Generate / load data ───────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        generate_telco_dataset()
    X, y = load_and_prepare(DATA_PATH)
    num_cols, cat_cols = get_feature_groups(X)

    # ── Train/test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"\n[SPLIT] Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Build preprocessor ─────────────────────────────────────────────
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # ── Logistic Regression ────────────────────────────────────────────
    lr_pipeline = build_lr_pipeline(build_preprocessor(num_cols, cat_cols))
    gs_lr       = tune_pipeline(lr_pipeline, LR_PARAM_GRID, X_train, y_train, "Logistic Regression")
    lr_metrics  = evaluate(gs_lr, X_test, y_test, "Logistic Regression")
    lr_path     = export_pipeline(gs_lr, "logistic_regression")

    # ── Random Forest ──────────────────────────────────────────────────
    rf_pipeline = build_rf_pipeline(build_preprocessor(num_cols, cat_cols))
    gs_rf       = tune_pipeline(rf_pipeline, RF_PARAM_GRID, X_train, y_train, "Random Forest")
    rf_metrics  = evaluate(gs_rf, X_test, y_test, "Random Forest")
    rf_path     = export_pipeline(gs_rf, "random_forest")

    # ── Model comparison ───────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  Model Comparison Summary")
    print("═" * 55)
    print(f"  {'Metric':<12}  {'Logistic Reg':>14}  {'Random Forest':>14}")
    print(f"  {'─'*12}  {'─'*14}  {'─'*14}")
    for metric in lr_metrics:
        print(f"  {metric:<12}  {lr_metrics[metric]:>14.4f}  {rf_metrics[metric]:>14.4f}")

    winner = "Random Forest" if rf_metrics["ROC-AUC"] > lr_metrics["ROC-AUC"] else "Logistic Regression"
    print(f"\n  🏆  Best model by ROC-AUC: {winner}")

    # ── Round-trip verification ────────────────────────────────────────
    print("\n[VERIFY] Loading saved pipelines and re-predicting …")
    for label, path in [("Logistic Regression", lr_path), ("Random Forest", rf_path)]:
        loaded = load_pipeline(path)
        preds  = loaded.predict(X_test[:5])
        print(f"  {label}: first 5 predictions → {preds.tolist()}")

    print("\n✅  Pipeline complete. Models saved to:", MODEL_DIR)


if __name__ == "__main__":
    main()
