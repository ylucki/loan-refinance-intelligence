# Loan Refinance Intelligence

AI-driven refinancing intelligence system focused on two pathways:
- Credit Card to Personal Loan (CC-to-PL) conversion
- Personal Loan (PL) refinancing

The goal is to build a practical, explainable ML pipeline that answers three core questions for a borrower:
1. Is this borrower a good refinancing candidate?
2. What interest rate and monthly savings are realistically achievable?
3. What recommendation should be given, and why?

## Scope
- End-to-end implementation using public datasets and reproducible notebooks
- Calibrated credit risk scoring (classification)
- Rate/savings prediction (regression)
- Hybrid recommendation logic (ML + rules)
- Explainability with SHAP/LIME and fairness checks
- Streamlit demo for interactive inference

## Data
- LendingClub Loan Dataset (Kaggle, 2007-2018)
- Home Credit Default Risk (Kaggle)

## Summary Findings (Latest Sampled EDA Run)
- Run profile: `EDA_MAX_ROWS=50000`, `EDA_PLOT_SAMPLE=30000`
- Total rows loaded: 50,000
- Resolved rows used for supervised analysis: 44,006
- Unresolved rows excluded: 5,994 (11.99%)
- Resolved default rate: 20.52%
- Resolved pathway mix:
  - `pl_refinance`: 24,710
  - `cc_to_pl`: 10,988
  - `other`: 8,308
- Data quality checks passed for duplicate IDs, negative income, and rate range checks.
- Leakage exclusions are documented in `reports/tables/leakage_exclusions.csv`.
- Supporting planning/proposal documents are versioned in `project_docs/`.

## Primary Notebook for Review
- EDA and initial report notebook: `notebooks/exploratory_data_analysis.ipynb`
- End-to-end modeling notebook: `notebooks/end_to_end_pipeline.ipynb`
- EDA output artifacts:
  - `reports/tables/eda_summary.json`
  - `reports/tables/data_quality_checks.csv`
  - `reports/tables/data_cleaning_summary.csv`
  - `reports/tables/missingness_before_after.csv`
  - `reports/tables/outlier_analysis.csv`
  - `reports/figures/eda_overview.png`
  - `reports/figures/eda_feature_distributions.png`

## Quick Start
1. Install dependencies from `requirements.txt`.
2. Place raw datasets under `data/raw` (nested folders are supported).
3. Run `notebooks/exploratory_data_analysis.ipynb` to generate initial figures/tables.
4. Review outputs in `reports/figures` and `reports/tables`.

## Runtime Profiles
Use `scripts/run_pipeline_notebook.py` with environment knobs to control speed vs depth.

- `MODEL_MAX_ROWS`: modeling sample size cap
- `MODEL_RANDOM_SEARCH_ITERS`: randomized search iterations per model family
- `MODEL_GRID_TOP_K`: number of top models to refine via grid search
- `MODEL_CV_SPLITS`: temporal CV folds (`TimeSeriesSplit`)
- `MODEL_ENABLE_SVM`: `1` to include SVM/SVR models, `0` to skip for faster runs
- `MODEL_N_JOBS`: parallel workers for search jobs

If your machine supports multiprocessing, increase `MODEL_N_JOBS` (for example `4` or `8`) to speed up model search.  
If you see permission/process-spawn issues, set `MODEL_N_JOBS=1`.

### Fast iteration
```bash
MODEL_MAX_ROWS=20000 MODEL_RANDOM_SEARCH_ITERS=2 MODEL_GRID_TOP_K=0 MODEL_CV_SPLITS=2 MODEL_ENABLE_SVM=0 MODEL_N_JOBS=1 MPLCONFIGDIR=/tmp ./.venv/bin/python scripts/run_pipeline_notebook.py
```

### Balanced run
```bash
MODEL_MAX_ROWS=80000 MODEL_RANDOM_SEARCH_ITERS=5 MODEL_GRID_TOP_K=1 MODEL_CV_SPLITS=3 MODEL_ENABLE_SVM=0 MODEL_N_JOBS=4 MPLCONFIGDIR=/tmp ./.venv/bin/python scripts/run_pipeline_notebook.py
```

### Deep run
```bash
MODEL_MAX_ROWS=150000 MODEL_RANDOM_SEARCH_ITERS=8 MODEL_GRID_TOP_K=2 MODEL_CV_SPLITS=3 MODEL_ENABLE_SVM=1 MODEL_N_JOBS=8 MPLCONFIGDIR=/tmp ./.venv/bin/python scripts/run_pipeline_notebook.py
```

## Pipeline Notebook
- Main completed notebook: `notebooks/end_to_end_pipeline.ipynb`
- Consolidated metrics snapshot: `reports/tables/pipeline_summary.json`
- Supporting model and policy artifacts:
  - `reports/tables/risk_classification_*_metrics.csv`
  - `reports/tables/rate_regression_*_metrics.csv`
  - `reports/tables/risk_classification_model_search.csv`
  - `reports/tables/rate_regression_model_search.csv`
  - `reports/tables/feature_selection_summary.csv`
  - `reports/tables/risk_threshold_sweep.csv`
  - `reports/tables/policy_simulation_table.csv`

## Model Pipeline
- Feature engineering + feature selection (correlation filter, L1/Lasso signals)
- Classification layer with regularized linear models, Random Forest, and SVM variants
- Regression layer with Ridge/Lasso/ElasticNet, Random Forest, and SVR variants
- Time-series CV, randomized search, and grid refinement for model selection
- Calibration and threshold tuning for risk-policy alignment
- Recommendation layer with risk-adjusted savings and policy simulation outputs

## Data Wrangling Principles
- Supervised labels are built on resolved outcomes only.
- Unresolved loan statuses are excluded from risk-model training.
- Post-outcome leakage fields are explicitly excluded from modeling features.

## Data Wrangling Function Flow
Primary implementation: `src/data/wrangling.py`

1. `apply_core_wrangling(df)`
2. `resolved_modeling_frame(df)` (for supervised risk training only)
3. `summarize_outcomes(df)` (label mix and default rates)
4. `data_profile(df)` and `quality_checks(df)` (schema/quality diagnostics)
5. `leakage_exclusions()` (documented feature exclusion policy)

## Documentation
- `docs/README.md`
- `docs/PROJECT_GUIDE.md`
- `docs/DATA_WRANGLING_GUIDE.md`
- `docs/MODELING_AND_EVALUATION.md`
- `project_docs/README.md` (index for project context documents)

## Roadmap
Phase 1: Core intelligence pipeline with public data and explainability.  
Phase 2: Productization and controlled real-time integration into lending workflows.
