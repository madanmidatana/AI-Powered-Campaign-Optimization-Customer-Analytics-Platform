"""
data_loader.py — Load, clean, and feature-engineer the customer dataset.
"""

import pandas as pd
import streamlit as st
from pathlib import Path


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load customers.csv, clean it, and derive RFM scores."""
    csv_path = Path(__file__).resolve().parent.parent / "data" / "customers.csv"
    df = pd.read_csv(csv_path)

    # --- drop nulls ---
    df.dropna(inplace=True)

    # --- encode categoricals ---
    df["gender_encoded"] = df["gender"].map({"Male": 0, "Female": 1, "Other": 2})

    # --- parse date column ---
    if "last_purchase_date" in df.columns:
        df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])

    # --- derive RFM scores (min-max scaled to 0-100) ---
    # Recency: lower days_since_last_purchase → higher score
    max_days = df["days_since_last_purchase"].max()
    df["recency_score"] = round(
        (1 - df["days_since_last_purchase"] / max_days) * 100, 2
    )

    # Frequency: higher num_purchases → higher score
    max_purchases = df["num_purchases"].max()
    df["frequency_score"] = round(
        (df["num_purchases"] / max_purchases) * 100, 2
    )

    # Monetary: higher total_spend → higher score
    max_spend = df["total_spend"].max()
    df["monetary_score"] = round(
        (df["total_spend"] / max_spend) * 100, 2
    )

    return df
