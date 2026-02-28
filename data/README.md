# Data Directory

Use this directory for all project datasets.

## Layout
- `raw/`: immutable source data files
- `interim/`: cleaned but not finalized data
- `processed/`: modeling-ready data artifacts

## Expected Primary Files
Place at least one primary LendingClub file anywhere under `raw/`:
- `lending_club.parquet`
- `lending_club.csv`
- `accepted_2007_to_2018Q4.csv`
- `accepted_2007_to_2018Q4.csv.gz`

Optional secondary dataset:
- `application_train.csv` (Home Credit)
- Can be in any nested folder under `raw/`

## Discovery Assumptions
- Files are discovered by scanning the current `raw/` tree recursively.
- Primary LendingClub file is selected by expected name/keyword match; if duplicates exist, the largest match is used.
- Home Credit is optional and identified by exact file name `application_train.csv`.

## Notes
- Do not commit private or sensitive customer data.
- Keep raw files unchanged and generate transformed outputs in `interim/` or `processed/`.
- Keep extracted archive folders in `raw/`; write cleaned versions to `interim/`.
