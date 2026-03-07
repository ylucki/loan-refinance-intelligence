# Executive Summary

## Problem Statement

This project builds a refinancing intelligence system for two pathways:
- `cc_to_pl`: credit card balance consolidation into a personal loan
- `pl_refinance`: refinancing an existing personal loan into a better personal loan

The business question is not only whether a borrower is risky, but whether refinancing is both feasible and economically meaningful. The end-to-end objective is therefore to answer three questions for each borrower:
1. Is the borrower a suitable refinance candidate?
2. What refinance rate and monthly savings appear achievable?
3. What recommendation should a lending team take: `consider` or `hold`?

## Data Sources and Run Context

Primary data source:
- LendingClub accepted loans dataset (public structural analog)

Secondary reference dataset:
- Home Credit Default Risk dataset (loaded for future transferability checks, not yet used in the main model pipeline)

Current repository artifacts reflect two scales of execution:
- EDA review run: `EDA_MAX_ROWS=50000`, `EDA_PLOT_SAMPLE=30000`
- Updated modeling benchmark run: `FAST_MODE=0`, `MODEL_MAX_ROWS=20000`, `MODEL_RANDOM_SEARCH_ITERS=2`, `MODEL_GRID_TOP_K=0`, `MODEL_CV_SPLITS=3`, `MODEL_ENABLE_SVM=0`

Even with sampled review artifacts, the repository is built to wrangle the full LendingClub corpus:
- Raw rows available to the pipeline: `2,260,701`
- Unresolved outcomes excluded from supervised labeling in the current pipeline logic: `912,569`

This matters because unresolved loans materially censor default labels. The project therefore uses a resolved-only modeling frame for supervised training and evaluation.

## End-to-End Approach

The pipeline is implemented as a sequence of reusable stages:
- Data loading and schema validation
- Core wrangling: status normalization, refinance pathway mapping, date parsing, and resolved/unresolved outcome flags
- Data quality checks: duplicates, invalid ranges, missingness, and leakage exclusions
- EDA cleaning: duplicate removal, simple imputation, and IQR-based outlier diagnostics
- Feature engineering: log transforms, affordability ratios, FICO aggregates, utilization interactions, and rate/term interactions
- Feature selection: correlation filtering plus L1 logistic and lasso coefficient signals
- Modeling:
  - classification for refinance suitability / default risk
  - regression for expected refinance rate
- Recommendation policy: combine predicted probability of default, predicted rate, and estimated savings into `consider` / `hold`
- Policy simulation: show how thresholds change business volume and savings quality

The current notebook uses temporal splitting:
- Train: loans issued up to 2015
- Validation: 2016
- Test: 2017 onward

This is an important design choice because a random split would overstate performance by leaking time structure.

## What Has Been Implemented So Far

Completed repository components:
- Reproducible EDA notebook and end-to-end modeling notebook
- Modular wrangling, feature engineering, feature selection, search, evaluation, and recommendation code under `src/`
- Data quality and leakage control artifacts under `reports/tables/`
- Visual EDA outputs under `reports/figures/`
- Policy simulation outputs for recommendation strategy tuning
- Supporting project context documents under `project_docs/`

Completed data preparation work:
- Duplicate handling
- Missing-value imputation for key modeling columns
- Outlier identification for `annual_inc`, `installment`, `int_rate_num`, `dti`, and `revol_util_num`
- Explicit exclusion of post-outcome leakage columns such as `total_pymnt`, `last_pymnt_d`, and `last_credit_pull_d`

## Key Data and Business Observations

From the latest sampled EDA run:
- Raw rows analyzed: `50,000`
- Resolved rows used for supervised analysis: `44,006`
- Unresolved rows excluded: `5,994` (`11.99%`)
- Resolved default rate: `20.52%`

Resolved pathway mix:
- `pl_refinance`: `24,710`
- `cc_to_pl`: `10,988`
- `other`: `8,308`

Pathway-level business observations:
- `pl_refinance` is the largest segment and carries the highest observed default rate in the sampled EDA output (`22.3%`)
- `cc_to_pl` shows lower default (`17.0%`) but higher revolving utilization (`55.0%`), which is consistent with debt-consolidation behavior
- `pl_refinance` has higher average interest rate (`12.43%`) than `cc_to_pl` (`10.47%`), suggesting greater gross savings opportunity but also higher credit risk

Recommendation-layer observations from the current baseline test split:
- `hold`: `1,752` borrowers
- `consider`: `1,591` borrowers
- Average predicted default probability:
  - `hold`: `0.608`
  - `consider`: `0.301`
- Average risk-adjusted savings:
  - `hold`: `37.99`
  - `consider`: `52.18`

Business interpretation:
- The recommendation policy is directionally separating lower-risk, higher-value borrowers from higher-risk cases
- Thresholds create a clear tradeoff between volume and expected quality
- For example, a stricter hold threshold plus a minimum savings filter sharply reduces outreach volume but improves average economics among recommended borrowers

## Technical Insights

### Feature engineering and selection

The updated modeling benchmark selected `15` numeric and `4` categorical features.

Top retained numeric signals by blended regularization ranking:
- `rate_x_term`
- `fico_range_low`
- `annual_inc_log1p`
- `term_months`
- `revol_bal_log1p`
- `dti`
- `annual_inc`
- `revol_util_num`
- `installment_to_monthly_income`

Important selection behavior:
- `fico_avg` and `fico_range_high` were removed by correlation filtering as redundant with other bureau-related signals
- The selected set mixes credit quality, affordability, utilization, and pricing structure, which is appropriate for refinance decisioning

### Classification model behavior

The full pipeline supports these classification families:
- `logistic_l2`
- `logistic_l1`
- `logistic_elasticnet`
- `random_forest`
- `linear_svm_calibrated`
- `svm_rbf`

The updated non-fast benchmark run evaluated this practical subset:
- `logistic_l2`
- `logistic_l1`
- `logistic_elasticnet`
- `random_forest`
- `linear_svm_calibrated`

This run excludes only the most expensive RBF kernel branch for runtime control.

Validation / model-search ranking by PR-AUC:
- `logistic_l1`: `0.355`
- `logistic_l2`: `0.353`
- `logistic_elasticnet`: `0.343`
- `linear_svm_calibrated`: `0.342`
- `random_forest`: `0.338`

Test:
- `logistic_l1`: ROC-AUC `0.684`, PR-AUC `0.334`, precision `0.301`, recall `0.737`, F1 `0.427`
- `logistic_l2`: ROC-AUC `0.702`, PR-AUC `0.383`, precision `0.347`, recall `0.627`, F1 `0.447` on validation
- `logistic_elasticnet`: ROC-AUC `0.700`, PR-AUC `0.383`, precision `0.345`, recall `0.636`, F1 `0.447` on validation
- `random_forest`: ROC-AUC `0.687`, PR-AUC `0.381`, precision `0.383`, recall `0.417`, F1 `0.399` on validation
- `linear_svm_calibrated`: ROC-AUC `0.699`, PR-AUC `0.383`, but extremely low recall at the default operating point, making it weak as a practical screening model in the current configuration

Interpretation:
- The logistic family remains the strongest practical choice in the current benchmark
- `logistic_l1` was selected as champion because it led the search objective and supported a high-recall screening policy after threshold tuning
- `logistic_l2` and `logistic_elasticnet` are close alternatives and show that performance is not dependent on a single narrow specification
- The calibrated linear SVM has acceptable ranking metrics but is not producing a usable operating point under the current calibration path
- The random forest is competitive, but in this benchmark the logistic models remain easier to justify and operationalize

Threshold tuning insight:
- The chosen operating threshold is `0.45`
- At this threshold on validation, precision is about `0.327`, recall about `0.727`, and F1 about `0.451`
- Lower thresholds produce near-total recall but an unusably high positive rate; higher thresholds improve precision but reduce campaign reach

### Regression model behavior

The full pipeline supports these regression families:
- `ridge`
- `lasso`
- `elasticnet`
- `random_forest_regressor`
- `linear_svr`
- `svr_rbf`

The updated non-fast benchmark run evaluated:
- `random_forest_regressor`
- `ridge`
- `lasso`
- `elasticnet`
- `linear_svr`

Validation / model-search ranking by RMSE:
- `random_forest_regressor`: RMSE `0.562` (CV metric)
- `ridge`: RMSE `0.649`
- `lasso`: RMSE `0.650`
- `elasticnet`: RMSE `0.660`
- `linear_svr`: RMSE `0.695`

Test:
- `random_forest_regressor`: RMSE `1.226`, MAE `0.488`, MAPE `0.024`, `R2 = 0.949`
- `linear_svr`: RMSE `8.282`, MAE `0.729`, MAPE `0.048`, `R2 = -1.315`
- `lasso`: RMSE `8.861`, MAE `0.749`, MAPE `0.052`, `R2 = -1.650`
- `elasticnet`: RMSE `10.088`, MAE `0.772`, MAPE `0.054`, `R2 = -2.435`
- `ridge`: RMSE `15.273`, MAE `0.904`, MAPE `0.062`, `R2 = -6.873`

Interpretation:
- The refreshed benchmark materially changes the regression story
- A nonlinear tree model (`random_forest_regressor`) is clearly outperforming the linear family and holds up well on the test split
- The large performance gap between the forest model and the linear models suggests meaningful nonlinear structure in rate formation
- This does not make the regression layer fully production-ready yet, but it is now a credible component rather than an obviously unstable baseline

### Open technical gap

The calibration step still fails in the current benchmark because the `CalibratedClassifierCV` call path is using an outdated `prefit` pattern against the installed scikit-learn API. This does not block the baseline classification workflow, but it should be fixed before presenting the pipeline as fully calibrated.

## What This Means for the Business

The project already demonstrates a credible decision-support workflow:
- identify a refinance pathway
- screen for risk
- estimate savings
- apply a transparent recommendation rule

The strongest business value today is still in prioritization, not full automation:
- pre-screening inbound leads
- routing sales effort toward lower-risk, higher-savings borrowers
- quantifying how policy thresholds change outreach volume and expected borrower value

The most defensible business conclusion at this stage is:
- the classification and recommendation pipeline is useful as a prioritization tool
- the refreshed benchmark shows that the savings/rate layer can also be useful, but it still needs validation on larger non-fast runs and, ultimately, on production-grade lender data

## What Remains to Be Done

Highest-priority next steps:
1. Re-run the end-to-end modeling pipeline on a larger non-fast sample so the repository has a stronger benchmark artifact set
2. Fix classifier calibration compatibility with the current scikit-learn version
3. Confirm the regression findings on a larger benchmark run with grid refinement enabled
4. Add pathway-specific model evaluation to test whether `cc_to_pl` and `pl_refinance` should be modeled separately
5. Replace public-data scenario assumptions with production borrower and lender data when moving toward real deployment

## Overall Status

This repository is beyond a pure EDA exercise. It already contains:
- a coherent refinance decision problem
- leakage-aware data preparation
- feature engineering and selection
- baseline classification and regression modeling
- recommendation logic
- policy simulation outputs

The project is best described as an end-to-end baseline system with a credible risk-screening layer, a useful recommendation framework, and a refreshed regression benchmark that now shows promising nonlinear performance but still needs larger-scale confirmation.
