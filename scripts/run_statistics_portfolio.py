"""Build statistics-first portfolio artifacts from customer-level segmentation data.

This script avoids notebook execution dependencies by producing core outputs directly:
- cleaned dataset and engineered features
- statistical test summaries with effect sizes
- clustering outputs and figures
- optional segment-classification ML benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure local imports work when script is run from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import clean_customer_data, load_customer_data, quality_report
from src.feature_engineering import build_behaviour_features
from src.modelling import (
    build_segment_logistic_model,
    build_segment_random_forest,
    evaluate_kmeans_range,
    fit_kmeans,
)

RANDOM_STATE = 42
BOOTSTRAP_ITERATIONS = 300
BOOTSTRAP_MAX_SAMPLE = 10000
SHAPIRO_MAX_SAMPLE = 5000

SEGMENTATION_COLUMNS = [
    "annual_income",
    "avg_monthly_spend",
    "purchase_frequency",
    "avg_order_value",
    "return_rate",
    "engagement_score",
]

MODEL_FEATURE_COLUMNS = [
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
    "annualized_spend",
    "spend_to_income_ratio",
    "support_intensity",
    "discounted_order_value",
    "engagement_score",
    "payment_method",
    "region",
    "value_tier",
]


def ensure_output_dirs(root: Path) -> tuple[Path, Path, Path]:
    processed_dir = root / "data" / "processed"
    figures_dir = root / "outputs" / "figures"
    tables_dir = root / "outputs" / "tables"
    reports_dir = root / "reports"

    for directory in [processed_dir, figures_dir, tables_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return processed_dir, figures_dir, tables_dir


def cohen_d(sample_a: pd.Series, sample_b: pd.Series) -> float:
    a = sample_a.dropna().to_numpy()
    b = sample_b.dropna().to_numpy()
    n_a = len(a)
    n_b = len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_var <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled_var))


def cramers_v(contingency: pd.DataFrame, chi2: float) -> float:
    n = contingency.to_numpy().sum()
    if n == 0:
        return float("nan")
    r, c = contingency.shape
    denom = min(r - 1, c - 1)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / (n * denom)))


def eta_squared_from_groups(groups: list[np.ndarray]) -> float:
    if not groups:
        return float("nan")
    values = np.concatenate(groups)
    if values.size == 0:
        return float("nan")
    overall_mean = float(np.mean(values))
    ss_between = float(sum(len(group) * (float(np.mean(group)) - overall_mean) ** 2 for group in groups))
    ss_total = float(np.sum((values - overall_mean) ** 2))
    if ss_total <= 0:
        return float("nan")
    return ss_between / ss_total


def bootstrap_ci(
    stat_fn,
    n_rows: int,
    rng: np.random.Generator,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    max_sample_size: int | None = BOOTSTRAP_MAX_SAMPLE,
) -> tuple[float, float]:
    estimates: list[float] = []
    sample_size = n_rows if max_sample_size is None else min(max_sample_size, n_rows)
    for _ in range(n_iterations):
        indices = rng.integers(0, n_rows, size=sample_size)
        estimate = float(stat_fn(indices))
        if np.isfinite(estimate):
            estimates.append(estimate)
    if len(estimates) < max(20, n_iterations // 4):
        return float("nan"), float("nan")
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_ci_two_sample(
    stat_fn,
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    rng: np.random.Generator,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[float, float]:
    clean_a = sample_a[np.isfinite(sample_a)]
    clean_b = sample_b[np.isfinite(sample_b)]
    if len(clean_a) < 2 or len(clean_b) < 2:
        return float("nan"), float("nan")

    size_a = min(BOOTSTRAP_MAX_SAMPLE, len(clean_a))
    size_b = min(BOOTSTRAP_MAX_SAMPLE, len(clean_b))
    estimates: list[float] = []
    for _ in range(n_iterations):
        idx_a = rng.integers(0, len(clean_a), size=size_a)
        idx_b = rng.integers(0, len(clean_b), size=size_b)
        estimate = float(stat_fn(clean_a[idx_a], clean_b[idx_b]))
        if np.isfinite(estimate):
            estimates.append(estimate)
    if len(estimates) < max(20, n_iterations // 4):
        return float("nan"), float("nan")
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    ordered = np.array(p_values, dtype=float)[order]
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum.accumulate((ordered * m / (np.arange(m) + 1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    return adjusted.tolist()


def run_statistics_workflow(root: Path, run_ml: bool) -> None:
    sns.set_theme(style="whitegrid")
    processed_dir, figures_dir, tables_dir = ensure_output_dirs(root)

    raw_path = root / "data" / "raw" / "retail_customer_segmentation.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    # 1) Clean + feature engineering
    raw_df = load_customer_data(raw_path)
    clean_df = clean_customer_data(raw_df)
    features_df = build_behaviour_features(clean_df)

    clean_df.to_parquet(processed_dir / "customers_clean.parquet", index=False)
    features_df.to_parquet(processed_dir / "customer_features.parquet", index=False)
    quality_report(raw_df).to_csv(tables_dir / "data_quality_report.csv", index=False)

    # 2) EDA summary tables
    segment_summary = (
        features_df.groupby("customer_segment", dropna=False)
        .agg(
            customers=("customer_id", "nunique"),
            avg_income=("annual_income", "mean"),
            avg_monthly_spend=("avg_monthly_spend", "mean"),
            avg_purchase_frequency=("purchase_frequency", "mean"),
            avg_return_rate=("return_rate", "mean"),
            avg_engagement=("engagement_score", "mean"),
        )
        .sort_values("customers", ascending=False)
        .round(3)
    )
    segment_summary.to_csv(tables_dir / "segment_summary.csv")

    region_spend = (
        features_df.groupby("region", dropna=False)["avg_monthly_spend"]
        .mean()
        .sort_values(ascending=False)
        .round(3)
    )
    region_spend.to_csv(tables_dir / "region_spend_summary.csv", header=["avg_monthly_spend"])

    # 3) Statistical tests + confidence intervals + validity checks
    rng = np.random.default_rng(RANDOM_STATE)
    n_rows = len(features_df)

    spend_values = features_df["avg_monthly_spend"].to_numpy(dtype=float)
    frequency_values = features_df["purchase_frequency"].to_numpy(dtype=float)
    segment_values = features_df["customer_segment"].to_numpy()
    payment_values = features_df["payment_method"].to_numpy()

    rho, p_corr = stats.spearmanr(frequency_values, spend_values)
    pearson_r, pearson_p = stats.pearsonr(frequency_values, spend_values)
    rho_ci_low, rho_ci_high = bootstrap_ci(
        lambda idx: stats.spearmanr(frequency_values[idx], spend_values[idx])[0],
        n_rows=n_rows,
        rng=rng,
    )

    anova_groups = [
        group["avg_monthly_spend"].to_numpy()
        for _, group in features_df.groupby("customer_segment")
        if len(group) > 1
    ]
    anova_stat, anova_p = stats.f_oneway(*anova_groups)
    eta_sq = eta_squared_from_groups(anova_groups)
    eta_ci_low, eta_ci_high = bootstrap_ci(
        lambda idx: eta_squared_from_groups(
            [
                spend_values[idx][segment_values[idx] == seg]
                for seg in np.unique(segment_values[idx])
                if np.sum(segment_values[idx] == seg) > 1
            ]
        ),
        n_rows=n_rows,
        rng=rng,
    )

    contingency = pd.crosstab(payment_values, segment_values)
    chi2, chi_p, chi_dof, expected = stats.chi2_contingency(contingency)
    c_v = cramers_v(contingency, chi2)
    cv_ci_low, cv_ci_high = bootstrap_ci(
        lambda idx: (
            lambda sample_cont: cramers_v(
                sample_cont,
                stats.chi2_contingency(sample_cont)[0],
            )
        )(pd.crosstab(payment_values[idx], segment_values[idx])),
        n_rows=n_rows,
        rng=rng,
        max_sample_size=None,
    )

    threshold = float(features_df["discount_usage_rate"].median())
    high_discount = features_df.loc[
        features_df["discount_usage_rate"] > threshold, "avg_order_value"
    ]
    low_discount = features_df.loc[
        features_df["discount_usage_rate"] <= threshold, "avg_order_value"
    ]
    t_stat, t_p = stats.ttest_ind(
        high_discount,
        low_discount,
        equal_var=False,
        nan_policy="omit",
    )
    d_val = cohen_d(high_discount, low_discount)
    mean_diff = float(high_discount.mean() - low_discount.mean())
    d_ci_low, d_ci_high = bootstrap_ci_two_sample(
        lambda a, b: cohen_d(pd.Series(a), pd.Series(b)),
        sample_a=high_discount.to_numpy(dtype=float),
        sample_b=low_discount.to_numpy(dtype=float),
        rng=rng,
    )
    mean_diff_ci_low, mean_diff_ci_high = bootstrap_ci_two_sample(
        lambda a, b: float(np.mean(a) - np.mean(b)),
        sample_a=high_discount.to_numpy(dtype=float),
        sample_b=low_discount.to_numpy(dtype=float),
        rng=rng,
    )

    # Assumption and robustness checks for transparency.
    anova_residuals = features_df["avg_monthly_spend"] - features_df.groupby("customer_segment")[
        "avg_monthly_spend"
    ].transform("mean")
    shapiro_sample = anova_residuals.sample(
        n=min(SHAPIRO_MAX_SAMPLE, len(anova_residuals)),
        random_state=RANDOM_STATE,
    )
    shapiro_stat, shapiro_p = stats.shapiro(shapiro_sample.to_numpy())
    levene_stat, levene_p = stats.levene(*anova_groups, center="median")
    mann_u_stat, mann_u_p = stats.mannwhitneyu(
        high_discount.to_numpy(),
        low_discount.to_numpy(),
        alternative="two-sided",
    )
    min_expected = float(np.min(expected))
    pct_expected_lt5 = float(np.mean(expected < 5))

    assumption_checks = pd.DataFrame(
        [
            {
                "check_name": "anova_residual_normality_shapiro",
                "statistic": float(shapiro_stat),
                "p_value": float(shapiro_p),
                "criterion": "p_value > 0.05",
                "passed": bool(shapiro_p > 0.05),
                "notes": f"Residual sample size={len(shapiro_sample)}",
            },
            {
                "check_name": "anova_variance_homogeneity_levene",
                "statistic": float(levene_stat),
                "p_value": float(levene_p),
                "criterion": "p_value > 0.05",
                "passed": bool(levene_p > 0.05),
                "notes": "Median-centred Levene test across customer segments.",
            },
            {
                "check_name": "chi_square_expected_counts",
                "statistic": float(min_expected),
                "p_value": float("nan"),
                "criterion": "min_expected >= 1 and share_expected_lt5 <= 0.20",
                "passed": bool((min_expected >= 1.0) and (pct_expected_lt5 <= 0.20)),
                "notes": f"share_expected_lt5={pct_expected_lt5:.3f}",
            },
            {
                "check_name": "welch_ttest_robustness_mann_whitney",
                "statistic": float(mann_u_stat),
                "p_value": float(mann_u_p),
                "criterion": "compare directionality against Welch t-test",
                "passed": bool((mann_u_p < 0.05) == (t_p < 0.05)),
                "notes": "Non-parametric sensitivity check for two-group difference.",
            },
            {
                "check_name": "spearman_pearson_direction_consistency",
                "statistic": float(pearson_r),
                "p_value": float(pearson_p),
                "criterion": "sign(spearman_rho) == sign(pearson_r)",
                "passed": bool(np.sign(rho) == np.sign(pearson_r)),
                "notes": f"Spearman rho={rho:.4f}, Pearson r={pearson_r:.4f}",
            },
        ]
    )
    assumption_checks.to_csv(tables_dir / "assumption_checks_summary.csv", index=False)

    stats_rows = [
        {
            "test_name": "spearman_purchase_frequency_vs_spend",
            "statistic": float(rho),
            "statistic_ci_low": float(rho_ci_low),
            "statistic_ci_high": float(rho_ci_high),
            "p_value": float(p_corr),
            "effect_size_name": "spearman_rho",
            "effect_size_value": float(rho),
            "effect_size_ci_low": float(rho_ci_low),
            "effect_size_ci_high": float(rho_ci_high),
            "notes": "Monotonic relationship between purchase frequency and average monthly spend.",
        },
        {
            "test_name": "anova_spend_across_customer_segment",
            "statistic": float(anova_stat),
            "statistic_ci_low": float("nan"),
            "statistic_ci_high": float("nan"),
            "p_value": float(anova_p),
            "effect_size_name": "eta_squared",
            "effect_size_value": float(eta_sq),
            "effect_size_ci_low": float(eta_ci_low),
            "effect_size_ci_high": float(eta_ci_high),
            "notes": "Mean spend difference across provided customer segments.",
        },
        {
            "test_name": "chi_square_payment_method_vs_customer_segment",
            "statistic": float(chi2),
            "statistic_ci_low": float("nan"),
            "statistic_ci_high": float("nan"),
            "p_value": float(chi_p),
            "effect_size_name": "cramers_v",
            "effect_size_value": float(c_v),
            "effect_size_ci_low": float(cv_ci_low),
            "effect_size_ci_high": float(cv_ci_high),
            "notes": f"Degrees of freedom: {int(chi_dof)}",
        },
        {
            "test_name": "welch_t_avg_order_value_high_vs_low_discount_usage",
            "statistic": float(t_stat),
            "statistic_ci_low": float("nan"),
            "statistic_ci_high": float("nan"),
            "p_value": float(t_p),
            "effect_size_name": "cohens_d",
            "effect_size_value": float(d_val),
            "effect_size_ci_low": float(d_ci_low),
            "effect_size_ci_high": float(d_ci_high),
            "notes": (
                f"Median discount threshold: {threshold:.4f}; "
                f"mean_diff={mean_diff:.4f}; "
                f"mean_diff_ci=[{mean_diff_ci_low:.4f}, {mean_diff_ci_high:.4f}]"
            ),
        },
    ]
    stats_summary = pd.DataFrame(stats_rows)
    p_values = stats_summary["p_value"].tolist()
    bonferroni = [min(p * len(p_values), 1.0) for p in p_values]
    fdr_bh = benjamini_hochberg(p_values)
    stats_summary["p_value_bonferroni"] = bonferroni
    stats_summary["p_value_fdr_bh"] = fdr_bh
    stats_summary["significant_0_05_bonferroni"] = stats_summary["p_value_bonferroni"] < 0.05
    stats_summary["significant_0_05_fdr_bh"] = stats_summary["p_value_fdr_bh"] < 0.05
    stats_summary.to_csv(tables_dir / "statistical_tests_summary.csv", index=False)

    # 4) Clustering
    X_seg = features_df[SEGMENTATION_COLUMNS].copy()
    X_scaled = StandardScaler().fit_transform(X_seg)
    k_scores = evaluate_kmeans_range(
        X_scaled,
        k_values=range(2, 9),
        random_state=RANDOM_STATE,
        silhouette_sample_size=10000,
    )
    k_scores.to_csv(tables_dir / "kmeans_silhouette_scores.csv", index=False)

    best_k = int(k_scores.iloc[0]["k"])
    kmeans_model = fit_kmeans(X_scaled, n_clusters=best_k, random_state=RANDOM_STATE)
    clustered_df = features_df.copy()
    clustered_df["cluster"] = kmeans_model.labels_.astype(int)
    clustered_df.to_csv(tables_dir / "customer_clusters.csv", index=False)

    cluster_profile = (
        clustered_df.groupby("cluster")[SEGMENTATION_COLUMNS]
        .mean()
        .sort_index()
        .round(3)
    )
    cluster_profile.to_csv(tables_dir / "cluster_profile_summary.csv")

    cluster_vs_segment = pd.crosstab(
        clustered_df["cluster"],
        clustered_df["customer_segment"],
        normalize="index",
    ).round(4)
    cluster_vs_segment.to_csv(tables_dir / "cluster_vs_segment_share.csv")

    # 5) Figures
    plt.figure(figsize=(10, 5))
    sns.histplot(features_df["avg_monthly_spend"], bins=40, kde=True)
    plt.title("Distribution of Average Monthly Spend")
    plt.xlabel("avg_monthly_spend")
    plt.tight_layout()
    plt.savefig(figures_dir / "spend_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    region_spend.plot(kind="bar")
    plt.title("Average Monthly Spend by Region")
    plt.ylabel("avg_monthly_spend")
    plt.tight_layout()
    plt.savefig(figures_dir / "avg_spend_by_region.png", dpi=160)
    plt.close()

    corr_cols = [
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
    plt.figure(figsize=(10, 8))
    sns.heatmap(features_df[corr_cols].corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.lineplot(data=k_scores.sort_values("k"), x="k", y="silhouette_score", marker="o")
    plt.title("KMeans Silhouette Score by K")
    plt.tight_layout()
    plt.savefig(figures_dir / "kmeans_silhouette_by_k.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.heatmap(cluster_profile, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Cluster Profile Heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "cluster_profile_heatmap.png", dpi=160)
    plt.close()

    # 6) Optional ML benchmark
    ml_status_path = tables_dir / "ml_status.txt"
    if run_ml:
        try:
            run_ml_benchmark(features_df, tables_dir)
            ml_status_path.write_text("ML benchmark completed successfully.\n", encoding="utf-8")
        except Exception as exc:  # pragma: no cover - fallback path for unstable envs
            ml_status_path.write_text(
                f"ML benchmark skipped due to runtime error:\n{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
    else:
        ml_status_path.write_text(
            "ML benchmark intentionally skipped. Run with --run-ml to enable.\n",
            encoding="utf-8",
        )

    write_findings_markdown(root, stats_summary, assumption_checks, k_scores, best_k, cluster_profile)


def run_ml_benchmark(features_df: pd.DataFrame, tables_dir: Path) -> None:
    X = features_df[MODEL_FEATURE_COLUMNS].copy()
    y = features_df["customer_segment"].copy()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logistic = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", build_segment_logistic_model(random_state=RANDOM_STATE)),
        ]
    )
    random_forest = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", build_segment_random_forest(random_state=RANDOM_STATE)),
        ]
    )

    logistic.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    log_pred = logistic.predict(X_test)
    rf_pred = random_forest.predict(X_test)

    metrics_df = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "accuracy": float(accuracy_score(y_test, log_pred)),
                "macro_f1": float(f1_score(y_test, log_pred, average="macro")),
            },
            {
                "model": "random_forest",
                "accuracy": float(accuracy_score(y_test, rf_pred)),
                "macro_f1": float(f1_score(y_test, rf_pred, average="macro")),
            },
        ]
    )
    metrics_df.to_csv(tables_dir / "ml_metrics.csv", index=False)

    report_text = classification_report(y_test, rf_pred)
    (tables_dir / "rf_classification_report.txt").write_text(report_text, encoding="utf-8")

    proba = random_forest.predict_proba(X)
    classes = random_forest.classes_
    score_df = pd.DataFrame(proba, columns=[f"prob_{c}" for c in classes])
    score_df.insert(0, "customer_id", features_df["customer_id"].values)
    score_df.insert(1, "predicted_segment", random_forest.predict(X))
    score_df.to_csv(tables_dir / "segment_prediction_scores.csv", index=False)


def write_findings_markdown(
    root: Path,
    stats_summary: pd.DataFrame,
    assumption_checks: pd.DataFrame,
    k_scores: pd.DataFrame,
    best_k: int,
    cluster_profile: pd.DataFrame,
) -> None:
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    test_rows = {row["test_name"]: row for _, row in stats_summary.iterrows()}
    best_sil = float(k_scores.iloc[0]["silhouette_score"])
    n_failed_checks = int((~assumption_checks["passed"]).sum())
    n_total_checks = int(len(assumption_checks))

    cluster_profile_text = cluster_profile.to_string()

    content = f"""# Statistical Findings Summary

## Dataset Snapshot
- Rows analysed: 50,000 customers
- Source: `data/raw/retail_customer_segmentation.csv`

## Core Statistical Tests
1. Spearman correlation (`purchase_frequency` vs `avg_monthly_spend`):
   - rho = {test_rows['spearman_purchase_frequency_vs_spend']['statistic']:.4f}
   - 95% CI for rho = [{test_rows['spearman_purchase_frequency_vs_spend']['effect_size_ci_low']:.4f}, {test_rows['spearman_purchase_frequency_vs_spend']['effect_size_ci_high']:.4f}]
   - p-value (raw) = {test_rows['spearman_purchase_frequency_vs_spend']['p_value']:.4g}
   - p-value (Bonferroni) = {test_rows['spearman_purchase_frequency_vs_spend']['p_value_bonferroni']:.4g}
   - p-value (FDR-BH) = {test_rows['spearman_purchase_frequency_vs_spend']['p_value_fdr_bh']:.4g}
2. ANOVA (`avg_monthly_spend` across `customer_segment`):
   - F-statistic = {test_rows['anova_spend_across_customer_segment']['statistic']:.2f}
   - p-value (raw) = {test_rows['anova_spend_across_customer_segment']['p_value']:.4g}
   - p-value (Bonferroni) = {test_rows['anova_spend_across_customer_segment']['p_value_bonferroni']:.4g}
   - p-value (FDR-BH) = {test_rows['anova_spend_across_customer_segment']['p_value_fdr_bh']:.4g}
   - eta-squared = {test_rows['anova_spend_across_customer_segment']['effect_size_value']:.4f}
   - 95% CI for eta-squared = [{test_rows['anova_spend_across_customer_segment']['effect_size_ci_low']:.4f}, {test_rows['anova_spend_across_customer_segment']['effect_size_ci_high']:.4f}]
3. Chi-square (`payment_method` vs `customer_segment`):
   - chi-square = {test_rows['chi_square_payment_method_vs_customer_segment']['statistic']:.2f}
   - p-value (raw) = {test_rows['chi_square_payment_method_vs_customer_segment']['p_value']:.4g}
   - p-value (Bonferroni) = {test_rows['chi_square_payment_method_vs_customer_segment']['p_value_bonferroni']:.4g}
   - p-value (FDR-BH) = {test_rows['chi_square_payment_method_vs_customer_segment']['p_value_fdr_bh']:.4g}
   - Cramer's V = {test_rows['chi_square_payment_method_vs_customer_segment']['effect_size_value']:.4f}
   - 95% CI for Cramer's V = [{test_rows['chi_square_payment_method_vs_customer_segment']['effect_size_ci_low']:.4f}, {test_rows['chi_square_payment_method_vs_customer_segment']['effect_size_ci_high']:.4f}]
4. Welch t-test (`avg_order_value` high vs low discount usage):
   - t-statistic = {test_rows['welch_t_avg_order_value_high_vs_low_discount_usage']['statistic']:.4f}
   - p-value (raw) = {test_rows['welch_t_avg_order_value_high_vs_low_discount_usage']['p_value']:.4g}
   - p-value (Bonferroni) = {test_rows['welch_t_avg_order_value_high_vs_low_discount_usage']['p_value_bonferroni']:.4g}
   - p-value (FDR-BH) = {test_rows['welch_t_avg_order_value_high_vs_low_discount_usage']['p_value_fdr_bh']:.4g}
   - Cohen's d = {test_rows['welch_t_avg_order_value_high_vs_low_discount_usage']['effect_size_value']:.4f}
   - 95% CI for Cohen's d = [{test_rows['welch_t_avg_order_value_high_vs_low_discount_usage']['effect_size_ci_low']:.4f}, {test_rows['welch_t_avg_order_value_high_vs_low_discount_usage']['effect_size_ci_high']:.4f}]

## Statistical Validity Checks
- Checks passed: {n_total_checks - n_failed_checks}/{n_total_checks}
- Checks requiring caution: {n_failed_checks}
- Full table: `outputs/tables/assumption_checks_summary.csv`

## Segmentation Findings
- Best KMeans k by silhouette: {best_k}
- Best silhouette score: {best_sil:.4f}

## Cluster Profile Means
```
{cluster_profile_text}
```

## Artifacts Generated
- `outputs/tables/statistical_tests_summary.csv`
- `outputs/tables/assumption_checks_summary.csv`
- `outputs/tables/kmeans_silhouette_scores.csv`
- `outputs/tables/cluster_profile_summary.csv`
- `outputs/tables/cluster_vs_segment_share.csv`
- `outputs/figures/spend_distribution.png`
- `outputs/figures/avg_spend_by_region.png`
- `outputs/figures/correlation_heatmap.png`
- `outputs/figures/kmeans_silhouette_by_k.png`
- `outputs/figures/cluster_profile_heatmap.png`
"""

    (reports_dir / "statistics_findings.md").write_text(content + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run statistics-first portfolio pipeline.")
    parser.add_argument(
        "--run-ml",
        action="store_true",
        help="Enable optional segment classification benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_statistics_workflow(PROJECT_ROOT, run_ml=args.run_ml)
    print("Portfolio statistics pipeline completed.")
    if args.run_ml:
        print("ML benchmark attempted; check outputs/tables/ml_status.txt.")
    else:
        print("ML benchmark skipped by default. Use --run-ml to enable.")


if __name__ == "__main__":
    main()
