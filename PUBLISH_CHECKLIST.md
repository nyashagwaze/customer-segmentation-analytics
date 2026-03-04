# Publish Checklist (Git + Quality)

Use this checklist before publishing to GitHub.

## 1) Local Validation
Run from project root:

```powershell
python -m compileall src scripts
python scripts/run_statistics_portfolio.py
python scripts/run_statistics_portfolio.py --run-ml
```

Confirm these outputs exist and are current:
- `outputs/tables/statistical_tests_summary.csv`
- `outputs/tables/assumption_checks_summary.csv`
- `outputs/tables/kmeans_silhouette_scores.csv`
- `outputs/tables/ml_metrics.csv`
- `reports/statistics_findings.md`
- `assets/screenshots/`

## 2) Content Review
1. Check README links render correctly in markdown preview.
2. Check screenshots display in `README.md`.
3. Ensure UK spelling consistency in key docs.
4. Confirm limitations and non-causal caveats are explicit.
5. Confirm dataset licence/terms allow public redistribution (or document source and exclude raw data if required).
6. Verify `DATASET_SOURCE.md` contains the source URL and licence details.

## 3) Git Hygiene
If this project has not been initialised as its own repository:

```powershell
git init
git branch -M main
```

Review what will be committed:

```powershell
git status
git add -A
git status
```

Create a clear initial commit:

```powershell
git commit -m "Initial publish: statistics-first customer segmentation portfolio"
```

## 4) Remote Publish
Create an empty GitHub repository, then run:

```powershell
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## 5) Post-Publish Check
1. Open the GitHub repo and verify README rendering.
2. Open `RECRUITER_RESULTS_CARD.md` and check metrics are readable.
3. Confirm large files are uploaded successfully.
4. Add repository topics such as:
- `data-analysis`
- `statistics`
- `customer-segmentation`
- `python`

## 6) Optional Professional Extras
1. Add `LICENSE` (MIT is common for portfolio projects).
2. Add a short pinned LinkedIn post linking to the repository.
3. Add one short video walkthrough (2-4 minutes) demonstrating findings.
