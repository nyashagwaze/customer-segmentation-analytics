"""Feature engineering for customer-level behaviour and value analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def build_behaviour_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build portfolio-ready behavioural features from customer-level inputs."""
    out = df.copy()

    out["annualized_spend"] = out["avg_monthly_spend"] * 12.0
    out["spend_to_income_ratio"] = np.where(
        out["annual_income"] > 0,
        out["annualized_spend"] / out["annual_income"],
        np.nan,
    )
    out["support_intensity"] = out["support_interactions"] / out["months_active"].replace(0, np.nan)
    out["discounted_order_value"] = out["avg_order_value"] * (1.0 - out["discount_usage_rate"])

    # Composite engagement proxy that balances browsing/purchase against friction signals.
    out["engagement_score"] = (
        _zscore(out["purchase_frequency"])
        + _zscore(out["browsing_time_minutes"])
        - _zscore(out["return_rate"])
        - _zscore(out["support_intensity"].fillna(0.0))
    )

    out["value_tier"] = pd.qcut(
        out["annualized_spend"].rank(method="first"),
        q=4,
        labels=["Low", "Medium", "High", "Top"],
    ).astype(str)

    return out
