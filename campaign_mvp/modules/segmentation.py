"""
segmentation.py — KMeans clustering on RFM features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Human-readable segment names ordered by overall value (descending)
SEGMENT_NAMES = ["High Value", "Growth", "At Risk", "Dormant"]


def segment_customers(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """
    Fit KMeans on scaled RFM features and assign a named segment to each customer.
    Returns a copy of the DataFrame with 'cluster' and 'segment' columns added.
    """
    rfm_cols = ["recency_score", "frequency_score", "monetary_score"]
    X = df[rfm_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    df = df.copy()
    df["cluster"] = labels

    # Rank clusters by the mean of all three RFM scores (descending)
    cluster_means = (
        df.groupby("cluster")[rfm_cols]
        .mean()
        .mean(axis=1)
        .sort_values(ascending=False)
    )
    rank_map = {cluster: rank for rank, cluster in enumerate(cluster_means.index)}
    df["segment"] = df["cluster"].map(rank_map).map(
        {i: name for i, name in enumerate(SEGMENT_NAMES[:k])}
    )

    return df


def get_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table with key metrics per segment."""
    summary = (
        df.groupby("segment")
        .agg(
            Customers=("customer_id", "count"),
            Avg_Recency=("recency_score", "mean"),
            Avg_Frequency=("frequency_score", "mean"),
            Avg_Monetary=("monetary_score", "mean"),
            Avg_Spend=("total_spend", "mean"),
            Conversion_Rate=("campaign_response", "mean"),
        )
        .round(2)
        .sort_values("Avg_Monetary", ascending=False)
        .reset_index()
    )
    summary.columns = [
        "Segment", "Customers", "Avg Recency", "Avg Frequency",
        "Avg Monetary", "Avg Spend ($)", "Conversion Rate",
    ]
    return summary
