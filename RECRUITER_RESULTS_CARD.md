# Recruiter Results Card: Statistical Analysis Portfolio

## Project in 30 Seconds
I analysed 50,000 customer-level retail records to test which factors genuinely distinguish customer value and segment behaviour. I built a reproducible, script-first workflow that produces cleaned data, statistical tests, validation checks, clustering outputs, and presentation-ready figures.

## Business Objective
Identify reliable signals for customer segmentation and targeting, while explicitly ruling out weak or misleading signals.

## Data and Scope
- Dataset: `data/raw/retail_customer_segmentation.csv`
- Unit of analysis: customer-level records
- Constraint: no native churn event label, so findings are associational (not causal)

## Methods Used
1. Data quality checks and cleaning
2. Feature engineering for value/engagement metrics
3. Statistical inference:
- Spearman correlation
- ANOVA
- Chi-square test
- Welch t-test
4. Effect size reporting + bootstrap confidence intervals
5. Multiple-testing correction:
- Bonferroni
- FDR-BH
6. Assumption and robustness checks
7. Unsupervised clustering (KMeans + silhouette selection)

## Key Statistical Findings
| Question | Result | Interpretation |
|---|---|---|
| Does purchase frequency relate to average monthly spend? | Spearman rho `0.0042`, 95% CI `[-0.0146, 0.0249]`, FDR-BH p `0.7031` | No meaningful monotonic relationship |
| Do provided customer segments differ in spend? | ANOVA F `2566.58`, eta-squared `0.1335`, 95% CI `[0.1224, 0.1453]`, FDR-BH p `<0.001` | Strong and practically meaningful segment-level spend differences |
| Is payment method associated with segment? | Chi-square p `0.7472` (FDR-BH), Cramer's V `0.0062`, 95% CI `[0.0049, 0.0149]` | Negligible association |
| Do high-discount users differ in order value? | Welch p `0.7472` (FDR-BH), Cohen's d `0.0029`, 95% CI `[-0.0246, 0.0300]` | No meaningful difference |

## Validity and Transparency
- Assumption checks passed: `3/5`
- Checks flagged for caution: `2/5` (large-sample ANOVA normality/variance assumptions)
- Robustness checks included and reported
- Explicitly avoids causal claims

## Segmentation Output
- Best KMeans solution: `k = 5`
- Best silhouette score: `0.2283` (moderate separation, directional profiling use)

## Optional ML Benchmark
- Logistic Regression: accuracy `0.6669`, macro-F1 `0.6372`
- Random Forest: accuracy `0.7543`, macro-F1 `0.7604`

## Evidence Artefacts
- Statistical tests: `outputs/tables/statistical_tests_summary.csv`
- Assumption checks: `outputs/tables/assumption_checks_summary.csv`
- Findings write-up: `reports/statistics_findings.md`
- Cluster outputs: `outputs/tables/customer_clusters.csv`
- Figures: `outputs/figures/`

## Reproducibility
```powershell
python -m pip install -r requirements.txt
python scripts/run_statistics_portfolio.py
```
