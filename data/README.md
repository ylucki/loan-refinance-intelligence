# Data

The `data/raw/` directory is intentionally empty in git. The LendingClub source file is large, so the repository keeps only the folder structure and download instructions.

Download source:
- [Kaggle: Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

File needed for this project:
- `accepted_2007_to_2018Q4.csv.gz`

Place the file here:
- `data/raw/lending_club/accepted_2007_to_2018Q4.csv.gz`

If you already have the unzipped Kaggle layout, this notebook also accepts:
- `data/raw/lending_club/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv`

How the notebook uses the data:
- each row is one accepted LendingClub loan
- the notebook filters to `purpose == "credit_card"`
- only resolved loan statuses are retained for modeling
- the target label is created from `loan_status`:
  - `Fully Paid` -> `consider`
  - default-like outcomes -> `hold`

Raw data files are ignored by git.
