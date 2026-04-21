"""
train.py
--------
End-to-end model training script.

Steps:
  1. Load (or generate) data
  2. Split into train / test
  3. Train Logistic Regression, Random Forest, then XGBoost
  4. Evaluate all three — print metrics
  5. Save the best model (XGBoost pipeline) as models/churn_model.pkl

Usage:
    python pipelines/train.py
"""

import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.preprocessor import (
    CATEGORICAL_FEATURES,
    DROP_COLS,
    NUMERICAL_FEATURES,
    TARGET,
    build_preprocessor,
)

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "telecom_churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: evaluate & print a single model's metrics
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Print classification metrics and return a summary dict."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    recall  = recall_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    log.info(f"\n{'='*50}")
    log.info(f"Model: {name}")
    log.info(f"  Recall  : {recall:.4f}")
    log.info(f"  F1 Score: {f1:.4f}")
    log.info(f"  ROC-AUC : {roc_auc:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Stay','Churn'])}")

    return {"name": name, "recall": recall, "f1": f1, "roc_auc": roc_auc}


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────────────
def train():
    # ── 1. Load Data ──────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        log.warning("Dataset not found — generating synthetic data …")
        from data.generate_data import generate_telecom_data
        df = generate_telecom_data()
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    log.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    log.info(f"Churn rate: {df[TARGET].mean():.1%}")

    # ── 2. Feature / Target Split ─────────────────────────────────────────────
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    log.info(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    preprocessor = build_preprocessor()
    results = []

    # ── 3a. Logistic Regression (baseline) ───────────────────────────────────
    log.info("Training Logistic Regression …")
    lr_pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf",  LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    lr_pipeline.fit(X_train, y_train)
    results.append(evaluate_model("Logistic Regression", lr_pipeline, X_test, y_test))

    # ── 3b. Random Forest ────────────────────────────────────────────────────
    log.info("Training Random Forest …")
    rf_pipeline = Pipeline([
        ("prep", build_preprocessor()),   # fresh preprocessor instance
        ("clf",  RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", rf_pipeline, X_test, y_test))

    # ── 3c. XGBoost (final model) ─────────────────────────────────────────────
    log.info("Training XGBoost …")

    # Compute scale_pos_weight to handle class imbalance
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    log.info(f"scale_pos_weight = {scale_pos_weight:.2f}")

    xgb_pipeline = Pipeline([
        ("prep", build_preprocessor()),
        ("clf",  XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    xgb_pipeline.fit(X_train, y_train)
    results.append(evaluate_model("XGBoost", xgb_pipeline, X_test, y_test))

    # ── 4. Summary Table ──────────────────────────────────────────────────────
    summary = pd.DataFrame(results).set_index("name")
    log.info(f"\n{'='*50}\nFINAL COMPARISON\n{summary.round(4)}\n{'='*50}")

    # ── 5. Save Best Model ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(xgb_pipeline, MODEL_PATH)
    log.info(f"✅ XGBoost pipeline saved to: {MODEL_PATH}")

    return xgb_pipeline


if __name__ == "__main__":
    train()