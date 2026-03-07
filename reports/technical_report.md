# Technical Analysis Summary

## Objective
Build an AI-driven refinancing intelligence pipeline for:
- Credit Card to Personal Loan conversion (`cc_to_pl`)
- Personal Loan refinancing (`pl_refinance`)

The system combines risk screening, savings estimation, and explainable recommendation logic.

## Data Used in This Run
- Primary: LendingClub accepted loans (`accepted_2007_to_2018Q4.csv.gz`)
- Optional reference loaded: Home Credit (`application_train.csv`) for future feature transfer checks
- Current modeling benchmark profile:
  - `FAST_MODE=0`
  - `MODEL_MAX_ROWS=20000`
  - `MODEL_RANDOM_SEARCH_ITERS=2`
  - `MODEL_GRID_TOP_K=0`
  - `MODEL_CV_SPLITS=3`
  - `MODEL_ENABLE_SVM=0`

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
Updated end-to-end modeling was run from `notebooks/end_to_end_pipeline.ipynb` on a resolved-row sample of `20,000` rows in non-fast mode.

### Feature selection outcome
- Numeric candidates before selection: `18`
- Selected numeric features: `15`
- Categorical features retained: `4`
- Key retained numeric features:
  - `rate_x_term`
  - `annual_inc_log1p`
  - `fico_range_low`
  - `annual_inc`
  - `revol_bal_log1p`
  - `revol_util_num`
  - `dti`
  - `term_months`
  - `dti_x_revol_util`
  - `installment_to_monthly_income`
- Redundant bureau features removed by correlation/filtering:
  - `fico_avg`
  - `fico_range_high`

### Classification (risk suitability)
Model families benchmarked in this run:
- `logistic_l1`
- `logistic_l2`
- `logistic_elasticnet`
- `random_forest`
- `linear_svm_calibrated`

Search ranking by validation PR-AUC:
- `logistic_l1`: `0.3552`
- `logistic_l2`: `0.3525`
- `logistic_elasticnet`: `0.3431`
- `linear_svm_calibrated`: `0.3424`
- `random_forest`: `0.3384`

Selected champion:
- `logistic_l1`
- Threshold selected from validation sweep: `0.45`
- Calibration status: `uncalibrated` (sigmoid calibration path still failing under current API usage)

Champion metrics:
- Validation:
  - ROC-AUC: `0.7024`
  - PR-AUC: `0.3853`
  - Log loss: `0.6186`
  - Brier: `0.2150`
  - Precision: `0.3270`
  - Recall: `0.7273`
  - F1: `0.4512`
- Test:
  - ROC-AUC: `0.6842`
  - PR-AUC: `0.3336`
  - Log loss: `0.6535`
  - Brier: `0.2249`
  - Precision: `0.3009`
  - Recall: `0.7374`
  - F1: `0.4274`

Interpretation:
- The logistic family remains strongest overall for practical risk screening.
- `logistic_l1`, `logistic_l2`, and `logistic_elasticnet` are tightly grouped, which suggests the signal is robust and not dependent on one model specification.
- `random_forest` is competitive but did not beat the logistic models in the current benchmark.
- `linear_svm_calibrated` has acceptable ranking metrics but poor operating behavior at the chosen classification threshold.

### Regression (rate estimation)
Model families benchmarked in this run:
- `random_forest_regressor`
- `ridge`
- `lasso`
- `elasticnet`
- `linear_svr`

Search ranking by validation RMSE:
- `random_forest_regressor`: `0.5621`
- `ridge`: `0.6493`
- `lasso`: `0.6497`
- `elasticnet`: `0.6598`
- `linear_svr`: `0.6951`

Selected champion:
- `random_forest_regressor`

Test metrics:
- `random_forest_regressor`: RMSE `1.2261`, MAE `0.4883`, MAPE `0.0239`, R2 `0.9493`
- `linear_svr`: RMSE `8.2819`, MAE `0.7294`, MAPE `0.0481`, R2 `-1.3150`
- `lasso`: RMSE `8.8606`, MAE `0.7489`, MAPE `0.0515`, R2 `-1.6498`
- `elasticnet`: RMSE `10.0882`, MAE `0.7722`, MAPE `0.0535`, R2 `-2.4349`
- `ridge`: RMSE `15.2735`, MAE `0.9038`, MAPE `0.0618`, R2 `-6.8733`

Interpretation:
- The regression results are now materially stronger than the earlier fast-mode artifact set.
- The tree-based regressor clearly outperforms the linear family, suggesting nonlinear structure in rate determination.
- This is a strong benchmark result, but it should still be confirmed on a larger non-fast run with additional search depth.

### Recommendation layer (test split)
- `hold`: `1,752`
- `consider`: `1,591`
- Average PD by recommendation:
  - `hold`: `0.6085`
  - `consider`: `0.3007`
- Average risk-adjusted savings:
  - `hold`: `37.99`
  - `consider`: `52.18`

Policy table insight:
- At hold threshold `0.45`, the recommendation rule is more permissive than the earlier fast-mode run and produces a balanced split between `hold` and `consider`.
- Tightening the minimum savings filter sharply reduces recommendation volume while increasing average savings among recommended borrowers.

Saved outputs:
- `reports/tables/recommendation_summary.csv`
- `reports/tables/recommendation_sample.csv`
- `reports/tables/pipeline_summary.json`

## Next Steps
1. Re-run the benchmark on a larger non-fast sample with grid refinement enabled.
2. Fix classifier calibration compatibility with the current scikit-learn API.
3. Add pathway-specific evaluation to test whether separate models should be used for `cc_to_pl` and `pl_refinance`.
4. Build a stable processed modeling dataset artifact under `data/processed/`.
5. Replace scenario assumptions with production borrower and lender inputs for deployment.
