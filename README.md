# Credit Card Loan Recommendation Classification

## Executive Summary: Project overview and goal

This project focuses on a loan-refinance screening problem. When a borrower applies for a loan with the purpose `credit_card`, can borrower and loan information available at origination help support an initial recommendation of `consider` or `hold`?

The business use case is credit-card debt consolidation. Borrowers often refinance revolving credit card debt into a structured installment loan, and lenders need a practical way to prioritize which applications deserve closer review. This project treats that as a binary classification problem using public LendingClub data.

Because the dataset does not contain an explicit historical recommendation field, the target is defined as a proxy:

- `consider`: loans that were fully paid
- `hold`: loans that ended in default-like outcomes

The goal is not to automate approval or denial. The goal is to build a transparent screening model that can help with first-pass triage.

### Findings

The final analysis compares three classification models:

- Logistic Regression
- Decision Tree
- Random Forest

Model selection was based on stratified cross-validation, small grid-search tuning, and final holdout test performance. Logistic Regression was selected as the best final model because it produced the strongest cross-validated ROC-AUC and remained the best overall model on the test set.

Execution summary:

- filtered source universe: `295,354` resolved `credit_card` loans
- modeling sample used in the notebook: `20,000` rows
- class balance:
  - `consider`: `16,658` (`83.29%`)
  - `hold`: `3,342` (`16.71%`)

Best model results:

- best model: Logistic Regression
- best cross-validated ROC-AUC: `0.7110`
- test ROC-AUC: `0.7032`
- test precision for `hold`: `0.2716`
- test recall for `hold`: `0.6916`
- test F1 for `hold`: `0.3900`

### Results and conclusion

The final results show that a simple, interpretable Logistic Regression model can provide useful signal for loan-refinance screening. In the current test set, the model catches about `69%` of the `hold` cases, which makes it useful as a first-pass review tool. Precision is lower, which means flagged `hold` cases should still be reviewed manually rather than being automatically rejected.

The strongest signals in the model come from expected credit-risk variables such as loan grade, sub-grade, interest rate, income, debt-to-income ratio, and average FICO. This makes the final model relatively easy to explain and defend in a business setting.

Overall, the project demonstrates that a focused and explainable machine learning workflow can support refinance screening while staying simple enough for academic review and straightforward business interpretation.

## Problem Statement

When a borrower applies for a loan with the purpose `credit_card`, can origination-time borrower and loan attributes be used to support an initial recommendation of `consider` or `hold`?

This project is intentionally framed as a screening and prioritization problem, not a full underwriting problem.

## Data Source

Dataset:

- source: [Kaggle: Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- file used: `accepted_2007_to_2018Q4.csv.gz`

Scope used in the notebook:

- accepted loans only
- `purpose == "credit_card"`
- resolved statuses only

Resolved statuses retained:

- `Fully Paid`
- `Charged Off`
- `Default`
- `Does not meet the credit policy. Status: Charged Off`

## Methodology

The project workflow includes:

- data filtering and cleaning
- duplicate checks
- missing-value treatment
- simple feature engineering
- exploratory data analysis and descriptive statistics
- multiple classification models
- `StratifiedKFold` cross-validation
- small `GridSearchCV` hyperparameter tuning
- final evaluation on a holdout test set

Primary evaluation metric:

- `ROC-AUC`

Supporting metrics:

- `precision`
- `recall`
- `F1`

The positive class for modeling is `hold`, because that makes the screening interpretation clearer: recall shows how many risky applications are correctly identified for closer review.

## Model Interpretation

The selected Logistic Regression model is the preferred final baseline for this project because it provides:

- the strongest overall ROC-AUC
- better interpretability than the tree-based alternatives
- coefficients that align with expected lending behavior

The most important coefficients are concentrated in grade and sub-grade features. Higher-risk grades push predictions toward `hold`, while stronger grades push predictions toward `consider`.

## Repository Outline

Main notebook:

- `notebooks/credit_card_loan_recommendation.ipynb`

Generated outputs:

- `reports/baseline_cv_results.csv`
- `reports/grid_search_summary.csv`
- `reports/final_test_results.csv`
- `reports/logistic_coefficients.csv`
- `reports/descriptive_summary_by_label.csv`
- `reports/figures/`

## Data Setup

Place the raw LendingClub file under:

- `data/raw/lending_club/accepted_2007_to_2018Q4.csv.gz`

Raw data is intentionally not committed to the repository.
