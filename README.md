# Credit Card Loan Recommendation Classification

This repository contains a focused binary classification study built on public LendingClub data. The business question is straightforward: when a borrower applies for a loan with the purpose `credit_card`, can borrower and loan attributes available at origination support an initial recommendation of `consider` or `hold`?

This is framed as a screening problem, not a full underwriting problem. The goal is to show how a simple model can help prioritize which credit-card debt-consolidation cases deserve further review.

Target definition used in this project:
- `consider` for fully paid loans
- `hold` for default-like outcomes

Because the public dataset does not contain an explicit historical recommendation field, `consider` and `hold` are treated as proxy labels derived from resolved loan outcomes.

Primary notebook:
- `notebooks/credit_card_loan_recommendation.ipynb`

What the notebook covers:
- problem statement and business context
- data source, structure, and filtering logic
- data cleaning, missing-value treatment, and outlier checks
- exploratory data analysis with labeled plots
- simple feature engineering
- Logistic Regression baseline
- Random Forest comparison
- results, interpretation, limitations, and next steps

Current run summary:
- filtered source universe: `295,354` resolved `credit_card` loans
- modeling sample used in the notebook: `30,000` rows
- label balance:
  - `consider`: `25,006` (`83.35%`)
  - `hold`: `4,994` (`16.65%`)
- test-set results:
  - Logistic Regression: `ROC-AUC 0.7081`, `PR-AUC 0.9172`, `precision 0.9071`, `recall 0.6365`, `F1 0.7481`
  - Random Forest: `ROC-AUC 0.7013`, `PR-AUC 0.9144`, `precision 0.8338`, `recall 1.0000`, `F1 0.9094`

Why Logistic Regression is the baseline:
- it is simpler to explain
- it has the strongest ROC-AUC in the current comparison
- its coefficient table makes feature effects easier to discuss

Repository contents:
- `notebooks/credit_card_loan_recommendation.ipynb`
- `reports/model_results.csv`
- `reports/logistic_coefficients.csv`
- `reports/label_summary.csv`
- `reports/figures/`

Data:
- download the LendingClub dataset from [Kaggle: Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- place `accepted_2007_to_2018Q4.csv.gz` under `data/raw/lending_club/`
- raw data is intentionally not committed
