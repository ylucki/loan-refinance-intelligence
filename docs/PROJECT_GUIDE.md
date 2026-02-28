# Project Guide

## What this project solves
Loan Refinance Intelligence estimates whether a borrower should refinance and explains why.

It supports two pathways:
- `cc_to_pl`: credit card balance -> personal loan
- `pl_refinance`: personal loan -> better personal loan

## Core outputs
For each borrower profile, the system will produce:
1. Risk suitability score (is refinancing prudent)
2. Expected refinance rate and savings range
3. Recommendation strength and rationale

## System architecture
1. **Data wrangling layer**
   - Loads public lending datasets
   - Cleans fields and derives analysis-ready columns
   - Builds resolved-only labels for risk modeling
2. **Risk model layer (classification)**
   - Predicts default probability for refinance candidacy
3. **Savings model layer (regression)**
   - Predicts achievable rate and estimated monthly savings
4. **Recommendation layer (hybrid logic)**
   - Combines risk + savings + pathway rules into ranked actions
5. **Explainability layer**
   - SHAP/LIME explanations for global and borrower-level transparency

## Repository flow
- `data/raw`: immutable source files
- `notebooks`: iterative analysis and model development
- `src`: reusable wrangling/model/recommendation code
- `reports`: generated figures, tables, and summaries
- `app`: interactive demo interface

## Current progress
- EDA completed on LendingClub accepted loans
- Leakage exclusions documented
- Data quality checks implemented
- Resolved-only supervised labeling policy implemented
