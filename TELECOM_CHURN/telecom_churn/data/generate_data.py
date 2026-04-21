"""
generate_data.py
----------------
Creates a realistic synthetic telecom churn dataset.
Run this once to produce: data/telecom_churn.csv
"""

import numpy as np
import pandas as pd
import random

# Reproducible results every time
np.random.seed(42)
random.seed(42)

NUM_CUSTOMERS = 5000


def generate_telecom_data(n: int = NUM_CUSTOMERS) -> pd.DataFrame:
    """Generate synthetic telecom customer data with realistic patterns."""

    # ── Demographics ─────────────────────────────────────────────────────────
    age = np.random.randint(18, 75, n)
    gender = np.random.choice(["Male", "Female"], n)
    location = np.random.choice(
        ["Urban", "Suburban", "Rural"], n, p=[0.50, 0.35, 0.15]
    )

    # ── Contract & Account Info ───────────────────────────────────────────────
    contract_type = np.random.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        n,
        p=[0.55, 0.25, 0.20],   # most customers are on monthly plans
    )

    tenure = np.where(
        contract_type == "Month-to-Month",
        np.random.randint(1, 36, n),     # shorter tenure for monthly
        np.random.randint(6, 72, n),     # longer for yearly
    )

    # ── Usage & Financials ────────────────────────────────────────────────────
    monthly_recharge = np.round(
        np.random.normal(65, 25, n).clip(10, 200), 2
    )
    data_usage_gb = np.round(np.random.exponential(4, n).clip(0, 30), 2)
    call_minutes = np.random.randint(0, 600, n)
    num_services = np.random.randint(1, 6, n)

    # ── Support & Complaints ──────────────────────────────────────────────────
    # Customers with more complaints are more likely to churn
    num_complaints = np.random.poisson(0.8, n)       # average ~0.8 complaints
    support_calls = np.random.poisson(1.5, n)

    # ── Payment ──────────────────────────────────────────────────────────────
    payment_method = np.random.choice(
        ["Credit Card", "Bank Transfer", "E-Wallet", "Cash"],
        n,
        p=[0.35, 0.30, 0.25, 0.10],
    )
    paperless_billing = np.random.choice([0, 1], n, p=[0.40, 0.60])

    # ── Simulated Sentiment Score ─────────────────────────────────────────────
    # Based on complaints + support interactions — ranges from -1 (angry) to +1 (happy)
    base_sentiment = np.random.normal(0.3, 0.4, n)
    sentiment_penalty = (num_complaints * 0.25) + (support_calls * 0.05)
    sentiment_score = np.clip(base_sentiment - sentiment_penalty, -1, 1).round(3)

    # ── Churn Label (target) ──────────────────────────────────────────────────
    # Build churn probability from business rules
    churn_prob = (
        0.05                                                      # base rate
        + (contract_type == "Month-to-Month") * 0.20             # monthly → higher churn
        + (tenure < 12) * 0.15                                   # new customers churn more
        + (num_complaints > 1) * 0.20                            # complaints → churn
        + (monthly_recharge > 100) * 0.10                        # high bill → churn
        + (sentiment_score < 0) * 0.15                           # negative sentiment
        + (support_calls > 3) * 0.10                             # many support calls
        - (tenure > 36) * 0.10                                   # loyal customers stay
        - (num_services > 3) * 0.05                              # more services → sticky
    ).clip(0, 1)

    churn = (np.random.rand(n) < churn_prob).astype(int)

    # ── Introduce ~3% missing values (realistic real-world noise) ─────────────
    df = pd.DataFrame({
        "customer_id":       [f"CUST{i:05d}" for i in range(n)],
        "age":               age,
        "gender":            gender,
        "location":          location,
        "tenure_months":     tenure,
        "contract_type":     contract_type,
        "monthly_recharge":  monthly_recharge,
        "data_usage_gb":     data_usage_gb,
        "call_minutes":      call_minutes,
        "num_services":      num_services,
        "num_complaints":    num_complaints,
        "support_calls":     support_calls,
        "payment_method":    payment_method,
        "paperless_billing": paperless_billing,
        "sentiment_score":   sentiment_score,
        "churn":             churn,
    })

    # Randomly set ~3% of some columns to NaN
    for col in ["monthly_recharge", "data_usage_gb", "sentiment_score", "call_minutes"]:
        mask = np.random.rand(n) < 0.03
        df.loc[mask, col] = np.nan

    print(f"✅ Dataset generated: {n} rows | Churn rate: {churn.mean():.1%}")
    return df


if __name__ == "__main__":
    df = generate_telecom_data()
    df.to_csv("telecom_churn.csv", index=False)
    print("💾 Saved to data/telecom_churn.csv")
    print(df.head())
    print(f"\nColumn types:\n{df.dtypes}")