"""Data loading, validation, and cleaning for customer-level segmentation data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "annual_income",
    "months_active",
    "avg_monthly_spend",
    "purchase_frequency",
    "avg_order_value",
    "discount_usage_rate",
    "return_rate",
    "browsing_time_minutes",
    "support_interactions",
    "payment_method",
    "region",
    "customer_segment",
]

NUMERIC_COLUMNS = [
    "age",
    "annual_income",
    "months_active",
    "avg_monthly_spend",
    "purchase_frequency",
    "avg_order_value",
    "discount_usage_rate",
    "return_rate",
    "browsing_time_minutes",
    "support_interactions",
]

RATE_COLUMNS = ["discount_usage_rate", "return_rate"]
NON_NEGATIVE_COLUMNS = [
    "annual_income",
    "months_active",
    "avg_monthly_spend",
    "purchase_frequency",
    "avg_order_value",
    "browsing_time_minutes",
    "support_interactions",
]


def load_customer_data(path: str | Path) -> pd.DataFrame:
    """Load raw customer-level data from CSV."""
    return pd.read_csv(path)


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact column-level quality summary."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing_pct": [float(df[col].isna().mean() * 100) for col in df.columns],
            "n_unique": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )


def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning and validation-friendly transformations."""
    out = df.copy()
    out.columns = out.columns.str.strip().str.lower()

    missing_required = [col for col in REQUIRED_COLUMNS if col not in out.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    out = out[REQUIRED_COLUMNS].copy()
    out = out.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Fill numeric/categorical missing values before range checks.
    for col in NUMERIC_COLUMNS:
        out[col] = out[col].fillna(out[col].median())
    for col in ["payment_method", "region", "customer_segment"]:
        mode_series = out[col].mode(dropna=True)
        fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
        out[col] = out[col].fillna(fill_value)

    # Constrain rates to [0, 1].
    for col in RATE_COLUMNS:
        out[col] = out[col].clip(lower=0.0, upper=1.0)

    # Ensure expected non-negative features.
    for col in NON_NEGATIVE_COLUMNS:
        out[col] = out[col].clip(lower=0.0)

    # Keep age within reasonable adult bounds for analysis consistency.
    out["age"] = out["age"].clip(lower=18, upper=100)

    out["customer_id"] = out["customer_id"].astype(str)
    out["payment_method"] = out["payment_method"].astype(str)
    out["region"] = out["region"].astype(str)
    out["customer_segment"] = out["customer_segment"].astype(str)

    return out
