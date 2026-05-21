"""
analytics.py — Compute dashboard KPIs.
"""

import pandas as pd


def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Return a dictionary of high-level KPIs for the Overview page.
    """
    total_customers = len(df)
    avg_clv = round(df["total_spend"].mean(), 2)

    # Top segment by customer count
    if "segment" in df.columns:
        top_segment = df["segment"].value_counts().idxmax()
        top_segment_size = int(df["segment"].value_counts().max())
    else:
        top_segment = "N/A"
        top_segment_size = 0

    conversion_rate = round(df["campaign_response"].mean() * 100, 2)

    return {
        "total_customers": total_customers,
        "avg_clv": avg_clv,
        "top_segment": top_segment,
        "top_segment_size": top_segment_size,
        "conversion_rate": conversion_rate,
    }
