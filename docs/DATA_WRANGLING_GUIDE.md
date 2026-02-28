# Data Wrangling Guide

## Datasets
Primary:
- LendingClub accepted loans (`accepted_2007_to_2018Q4.csv.gz`)

Optional secondary:
- Home Credit (`application_train.csv`) for later feature transfer checks

## Known dataset quirks
- LendingClub contains a large unresolved-status block in recent vintages; these rows are excluded from supervised risk labels.
- Home Credit uses sentinel values in some fields (for example `DAYS_EMPLOYED = 365243`), which should be remapped before modeling.

## Wrangling rules
Implemented in `src/data/wrangling.py`.

1. **Type normalization**
- Parse percent strings (for example `int_rate`, `revol_util`) to numeric values.
- Parse `issue_d` to a proper datetime and derive `issue_year`.

2. **Outcome construction**
- `is_default = 1` only for default outcomes (`charged off`, `default`, policy charged-off).
- `is_unresolved = 1` for unresolved statuses (`current`, grace period, late buckets).
- `is_resolved = 1 - is_unresolved`.

3. **Modeling frame policy**
- Supervised risk modeling uses **resolved-only** rows.
- Unresolved rows are excluded from label training to prevent target censoring bias.

4. **Pathway segmentation**
- `purpose=credit_card` -> `cc_to_pl`
- `purpose in {debt_consolidation, personal}` -> `pl_refinance`
- everything else -> `other`

## Leakage policy
Excluded from modeling feature sets:
- `total_pymnt`
- `last_pymnt_d`
- `last_credit_pull_d`

Reason: these are post-origination signals not available at refinance decision time.

## Quality checks
Generated in `reports/tables/data_quality_checks.csv`:
- duplicate loan ids
- negative incomes
- out-of-range rates/utilization
- unresolved status volume

## Output artifacts
EDA and wrangling outputs are written to:
- `reports/tables`
- `reports/figures`
