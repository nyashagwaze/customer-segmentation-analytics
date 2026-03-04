"""Segmentation and classification modelling helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score


def fit_kmeans(features: pd.DataFrame | np.ndarray, n_clusters: int = 4, random_state: int = 42) -> KMeans:
    """Fit a KMeans model for customer segmentation."""
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    model.fit(features)
    return model


def evaluate_kmeans_range(
    features: pd.DataFrame | np.ndarray,
    k_values: Iterable[int] = range(2, 9),
    random_state: int = 42,
    silhouette_sample_size: int = 10000,
) -> pd.DataFrame:
    """Compute silhouette score across candidate k values."""
    records: list[dict[str, float]] = []
    n_rows = len(features)
    sample_size = min(int(silhouette_sample_size), n_rows)
    for k in k_values:
        model = fit_kmeans(features, n_clusters=int(k), random_state=random_state)
        score = silhouette_score(
            features,
            model.labels_,
            sample_size=sample_size if sample_size < n_rows else None,
            random_state=random_state,
        )
        records.append({"k": float(k), "silhouette_score": float(score)})
    return pd.DataFrame(records).sort_values("silhouette_score", ascending=False)


def build_segment_logistic_model(random_state: int = 42) -> LogisticRegression:
    """Create multinomial logistic regression for segment classification."""
    return LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        random_state=random_state,
    )


def build_segment_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Create random forest classifier for segment classification."""
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
