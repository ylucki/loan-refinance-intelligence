# Modeling and Evaluation

## Layer 1: Risk Suitability (Classification)
Goal:
- Estimate default risk for refinance candidacy.

Model families:
- Logistic Regression (L2, L1, ElasticNet)
- Random Forest Classifier
- Linear SVM (calibrated)
- RBF SVM

Metrics:
- PR-AUC (selection priority), ROC-AUC
- Log loss, Brier score (calibration quality)
- Precision, Recall, F1 (threshold-dependent policy quality)

## Layer 2: Rate/Savings (Regression)
Goal:
- Estimate achievable refinance rate and monthly savings.

Model families:
- Ridge (L2), Lasso (L1), ElasticNet
- Random Forest Regressor
- Linear SVR and RBF SVR

Metrics:
- RMSE (selection priority), MAE, MAPE, R2

## Layer 3: Recommendation Engine
Goal:
- Rank refinance actions using risk-adjusted savings.

Logic:
- Combine predicted risk + expected savings + pathway constraints
- Produce action labels (for example: strong recommend / consider / hold)
- Evaluate policy sensitivity using hold-threshold and min-savings grids

## Explainability and governance
- SHAP global + borrower-level explanations
- Optional LIME for borderline cases
- Fairness checks across major borrower segments
- Track experiments and model versions for reproducibility

## Split strategy
- Use temporal splitting (train/validation/test by issue period)
- Use `TimeSeriesSplit` for CV inside the training window
- Use randomized search + grid refinement for tuning
- Apply resolved-outcome filtering before supervised model training
