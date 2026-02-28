# Non-Technical Report

## Executive Summary
This project builds an AI-driven refinancing intelligence system for two loan pathways:
- Credit Card to Personal Loan (`cc_to_pl`)
- Personal Loan refinancing (`pl_refinance`)

The system identifies borrowers who are likely to benefit from refinancing, estimates expected rate and monthly savings, and produces transparent recommendation labels.

In baseline testing, the model pipeline shows useful discrimination for risk screening and strong performance for rate estimation. This supports a practical pre-screening and prioritization workflow for lending teams.

## What the recommendation means
Each borrower receives a recommendation based on:
1. Predicted refinance risk suitability
2. Predicted refinance rate
3. Estimated monthly savings under a refinance scenario
4. Risk-adjusted decision logic

Current output labels:
- `consider`: borrower appears suitable with positive savings potential
- `hold`: borrower is either high-risk, low-benefit, or both

## How a lending company can use this model
### 1) Lead triage and prioritization
Use recommendation labels and risk-adjusted savings to rank inbound borrowers for agent outreach.

### 2) Pre-screen before manual underwriting
Use the risk model as a first-pass filter so underwriting teams focus on higher-likelihood candidates.

### 3) Personalized borrower messaging
Use predicted savings and pathway type to produce borrower-friendly value statements during sales calls.

### 4) Portfolio and campaign strategy
Aggregate model outputs by pathway, score bands, and savings bands to tune campaign targeting.

### 5) Compliance and governance support
Use explicit leakage policy, data quality checks, and recommendation rationale artifacts for auditability.

## Operational workflow (simple)
1. Collect borrower profile and current loan context.
2. Run risk suitability model.
3. Run rate/savings model.
4. Apply recommendation logic (`consider` / `hold`).
5. Route to sales/ops queue by priority.
6. Track outcomes and retrain periodically.

## Business KPIs this can improve
- Conversion rate from inquiry to refinance application
- Cost per funded refinance
- Average monthly savings delivered to borrowers
- Approval hit-rate of prioritized leads
- Analyst productivity (fewer low-quality manual reviews)

## Important limitations
- Training data is a public structural analog, not a live lender production book.
- Savings in this baseline are scenario-based, not final offer terms.
- Final production use should add calibration policy, fairness monitoring, and lender-specific constraints.

## Next step for production deployment
Run this model in shadow mode first (no automated decisions), compare against existing manual outcomes, then phase into assisted decisioning.
