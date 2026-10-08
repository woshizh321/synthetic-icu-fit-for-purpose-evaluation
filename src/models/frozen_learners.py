"""Frozen learner definitions used for real- and synthetic-trained models."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def fit_logistic_regression(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(
        penalty="l2", C=1.0, max_iter=5000, solver="lbfgs",
        class_weight=None, random_state=42,
    )
    model.fit(scaler.transform(X), y)
    return "lr", model, scaler


def fit_xgboost(X: np.ndarray, y: np.ndarray):
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=4,
    )
    model.fit(X, y)
    return "xgb", model, None


def predict_probability(fitted, X: np.ndarray) -> np.ndarray:
    kind, model, scaler = fitted
    transformed = scaler.transform(X) if kind == "lr" else X
    return model.predict_proba(transformed)[:, 1]
