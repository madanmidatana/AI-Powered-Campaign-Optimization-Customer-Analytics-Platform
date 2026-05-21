"""
prediction.py — Logistic Regression model to predict campaign response probability.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


FEATURE_COLS = ["recency_score", "frequency_score", "monetary_score", "email_opens"]
TARGET_COL = "campaign_response"


def train_model(df: pd.DataFrame):
    """
    Train a Logistic Regression model and return predictions + metrics.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of input with 'response_probability' column appended.
    metrics : dict
        {'accuracy': float, 'auc': float}
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_s, y_train)

    # Metrics on held-out set
    y_pred = model.predict(X_test_s)
    y_proba_test = model.predict_proba(X_test_s)[:, 1]
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    auc = round(roc_auc_score(y_test, y_proba_test), 4)

    # Predict on full dataset
    X_full_s = scaler.transform(X)
    probabilities = model.predict_proba(X_full_s)[:, 1]

    df_out = df.copy()
    df_out["response_probability"] = np.round(probabilities, 4)

    metrics = {"accuracy": accuracy, "auc": auc}
    return df_out, metrics
