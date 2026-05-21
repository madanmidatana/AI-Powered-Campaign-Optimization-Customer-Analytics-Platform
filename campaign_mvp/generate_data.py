"""
Generate synthetic customers.csv with 500 rows for the Campaign MVP.
Run once: python generate_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 500

# --- base demographics ---
customer_ids = [f"CUST-{i:04d}" for i in range(1, n + 1)]
ages = np.random.randint(18, 70, size=n)
genders = np.random.choice(["Male", "Female", "Other"], size=n, p=[0.48, 0.48, 0.04])

# --- transactional features ---
num_purchases = np.random.poisson(lam=12, size=n).clip(1)
avg_order_value = np.round(np.random.lognormal(mean=3.5, sigma=0.6, size=n), 2)
total_spend = np.round(num_purchases * avg_order_value * np.random.uniform(0.8, 1.2, size=n), 2)
days_since_last_purchase = np.random.exponential(scale=60, size=n).astype(int).clip(0, 365)

# --- engagement ---
email_opens = np.random.poisson(lam=5, size=n).clip(0, 30)

# --- campaign response (correlated with engagement + spend) ---
logit = (
    0.02 * email_opens
    + 0.001 * total_spend / 100
    - 0.005 * days_since_last_purchase
    + 0.01 * num_purchases
    + np.random.normal(0, 0.3, size=n)
)
prob = 1 / (1 + np.exp(-logit))
campaign_response = (prob > np.random.uniform(size=n)).astype(int)

# --- purchase dates (for time-series chart) ---
# Generate a random purchase date in the last 365 days for each customer
base_date = pd.Timestamp("2025-01-01")
last_purchase_date = [
    (base_date + pd.Timedelta(days=int(np.random.randint(0, 365)))).strftime("%Y-%m-%d")
    for _ in range(n)
]

df = pd.DataFrame({
    "customer_id": customer_ids,
    "age": ages,
    "gender": genders,
    "total_spend": total_spend,
    "num_purchases": num_purchases,
    "avg_order_value": avg_order_value,
    "days_since_last_purchase": days_since_last_purchase,
    "email_opens": email_opens,
    "campaign_response": campaign_response,
    "last_purchase_date": last_purchase_date,
})

out = Path(__file__).parent / "data" / "customers.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"[OK] Generated {len(df)} rows -> {out}")
