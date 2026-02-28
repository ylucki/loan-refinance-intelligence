# Technical Analysis Summary

## Objective
Build an AI-driven refinancing intelligence pipeline for:
- Credit Card to Personal Loan conversion (`cc_to_pl`)
- Personal Loan refinancing (`pl_refinance`)

The system combines risk screening, savings estimation, and explainable recommendation logic.

## Data Used in This Run
- Primary: LendingClub accepted loans (`accepted_2007_to_2018Q4.csv.gz`)
- Optional reference loaded: Home Credit (`application_train.csv`) for future feature transfer checks

## EDA Workflow Completed
1. Data loading and schema checks
2. Leakage-aware feature selection for EDA baseline
3. Outcome wrangling with resolved/unresolved status flags
4. Resolved-only modeling frame construction
5. Missingness and data quality profiling
6. Pathway segmentation (`cc_to_pl`, `pl_refinance`, `other`)
7. Class balance analysis for risk target readiness
8. Feature distribution analysis (`int_rate`, `dti`, `annual_inc`, `revol_util`)
9. Temporal trend analysis by `issue_year`

## Key Outputs
- Tables: `reports/tables/data_profile.csv`, `reports/tables/data_quality_checks.csv`, `reports/tables/leakage_exclusions.csv`, `reports/tables/pathway_summary.csv`, `reports/tables/default_rate_by_issue_year.csv`, `reports/tables/eda_summary.json`
- Figures: `reports/figures/eda_class_and_pathway_distribution.png`, `reports/figures/eda_feature_distributions.png`, `reports/figures/eda_default_rate_by_issue_year.png`

## Results Snapshot
- Records processed (all): `2,260,701`
- Resolved records used for supervised modeling analysis: `1,348,132`
- Unresolved records excluded from supervised labels: `912,569` (`40.37%`)
- Default-label rate (all rows): `11.91%`
- Default-label rate (resolved-only): `19.98%`
- Resolved pathway volumes:
  - `pl_refinance`: `781,442` (~58.0%)
  - `cc_to_pl`: `295,625` (~21.9%)
  - `other`: `271,065` (~20.1%)

### Pathway Differences (core signals)
- Mean interest rate:
  - `cc_to_pl`: `11.79`
  - `pl_refinance`: `13.62`
- Mean default-label rate:
  - `cc_to_pl`: `16.9%`
  - `pl_refinance`: `21.2%`
- Mean `dti` remains close for core pathways (~18.5-18.9), while `revol_util` is higher in `cc_to_pl` (expected behaviorally).

### Data Quality
- No severe missingness in selected modeling columns.
- Highest missingness among loaded selected fields is low (`dti` / `revol_util` <= ~0.07%).
- Quality checks flagged only a small number of out-of-range utilization rows (`2` rows).
- Duplicate loan ids in loaded data: `0`.
- Leakage exclusions are now explicitly documented (`total_pymnt`, `last_pymnt_d`, `last_credit_pull_d`).

### Temporal Observations
- Resolved-only default rate in late vintages remains materially higher than naive all-row rates.
- This confirms that unresolved statuses can heavily censor labels.
- Temporal splitting and resolved-outcome filtering are mandatory to avoid optimistic bias.

## Modeling Implications
1. Risk model training must use resolved outcomes only.
2. Class imbalance is material (~20% default on resolved frame) and requires threshold tuning and calibration.
3. Separate pathway models remain justified by persistent rate/default separation between `cc_to_pl` and `pl_refinance`.
4. Leakage columns must stay excluded from feature engineering and model training.

## Baseline Modeling Results
Baseline end-to-end modeling was run from `notebooks/end_to_end_pipeline.ipynb` on a resolved-row sample (`MODEL_MAX_ROWS=150000`).

### Classification (risk suitability)
- Best validation model: `random_forest`
- Test metrics:
  - ROC-AUC: `0.6899`
  - PR-AUC: `0.3579`
  - Brier score: `0.2068`
  - Precision: `0.3393`
  - Recall: `0.5803`
  - F1: `0.4282`

### Regression (rate estimation)
- Best validation model: `random_forest_regressor`
- Test metrics:
  - RMSE: `1.7074`
  - MAE: `0.9776`
  - R2: `0.9052`

### Recommendation layer (test split)
- `hold`: `21,151`
- `consider`: `4,019`
- Summary and samples saved to:
  - `reports/tables/recommendation_summary.csv`
  - `reports/tables/recommendation_sample.csv`
  - `reports/tables/pipeline_summary.json`

## Next Steps
1. Define strict temporal train/validation/test windows with explicit outcome-maturity filters.
2. Build a modeling dataset artifact under `data/processed/` using the resolved-only frame.
3. Train baseline risk model (classification) and baseline rate/savings model (regression).
4. Add calibration, pathway-level evaluation, and threshold-policy tables.
