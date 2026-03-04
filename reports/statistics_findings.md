# Statistical Findings Summary

## Dataset Snapshot
- Rows analysed: 50,000 customers
- Source: `data/raw/retail_customer_segmentation.csv`

## Core Statistical Tests
1. Spearman correlation (`purchase_frequency` vs `avg_monthly_spend`):
   - rho = 0.0042
   - 95% CI for rho = [-0.0146, 0.0249]
   - p-value (raw) = 0.3515
   - p-value (Bonferroni) = 1
   - p-value (FDR-BH) = 0.7031
2. ANOVA (`avg_monthly_spend` across `customer_segment`):
   - F-statistic = 2566.58
   - p-value (raw) = 0
   - p-value (Bonferroni) = 0
   - p-value (FDR-BH) = 0
   - eta-squared = 0.1335
   - 95% CI for eta-squared = [0.1224, 0.1453]
3. Chi-square (`payment_method` vs `customer_segment`):
   - chi-square = 3.82
   - p-value (raw) = 0.7005
   - p-value (Bonferroni) = 1
   - p-value (FDR-BH) = 0.7472
   - Cramer's V = 0.0062
   - 95% CI for Cramer's V = [0.0049, 0.0149]
4. Welch t-test (`avg_order_value` high vs low discount usage):
   - t-statistic = 0.3223
   - p-value (raw) = 0.7472
   - p-value (Bonferroni) = 1
   - p-value (FDR-BH) = 0.7472
   - Cohen's d = 0.0029
   - 95% CI for Cohen's d = [-0.0246, 0.0300]

## Statistical Validity Checks
- Checks passed: 3/5
- Checks requiring caution: 2
- Full table: `outputs/tables/assumption_checks_summary.csv`

## Segmentation Findings
- Best KMeans k by silhouette: 5
- Best silhouette score: 0.2283

## Cluster Profile Means
```
         annual_income  avg_monthly_spend  purchase_frequency  avg_order_value  return_rate  engagement_score
cluster                                                                                                      
0            36033.445            259.676               3.837           59.934        0.097             0.240
1            39660.343            333.963              10.143           32.312        0.120             2.218
2            40005.534            624.427               2.728          199.033        0.131            -0.483
3            39094.058            288.566               4.200           66.179        0.302            -2.195
4           114866.690            309.084               4.485           68.025        0.131             0.064
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

