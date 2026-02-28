"""Feature engineering helpers for refinance modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def parse_term_months(series: pd.Series) -> pd.Series:
    """Extract term months from strings like '36 months'."""

    months = pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    return months.fillna(36).clip(6, 120)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight derived features used by model pipelines."""

    out = df.copy()

    annual_inc = pd.to_numeric(out.get("annual_inc"), errors="coerce")
    revol_bal = pd.to_numeric(out.get("revol_bal"), errors="coerce")
    dti = pd.to_numeric(out.get("dti"), errors="coerce")
    revol_util = pd.to_numeric(out.get("revol_util_num"), errors="coerce")
    installment = pd.to_numeric(out.get("installment"), errors="coerce")
    fico_low = pd.to_numeric(out.get("fico_range_low"), errors="coerce")
    fico_high = pd.to_numeric(out.get("fico_range_high"), errors="coerce")
    int_rate = pd.to_numeric(out.get("int_rate_num"), errors="coerce")

    monthly_income = (annual_inc / 12.0).clip(lower=1.0)
    income_safe = annual_inc.clip(lower=1.0)

    out["term_months"] = parse_term_months(out.get("term", pd.Series(index=out.index, dtype="object")))
    out["annual_inc_log1p"] = np.log1p(annual_inc.clip(lower=0))
    out["revol_bal_log1p"] = np.log1p(revol_bal.clip(lower=0))
    out["fico_avg"] = (fico_low + fico_high) / 2.0
    out["fico_span"] = (fico_high - fico_low).clip(lower=0)
    out["installment_to_monthly_income"] = installment / monthly_income
    out["revol_bal_to_income"] = revol_bal / income_safe
    out["dti_x_revol_util"] = dti * revol_util
    out["rate_x_term"] = int_rate * out["term_months"]

    return out


def default_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return default numeric/categorical feature lists after engineering."""

    numeric_candidates = [
        "annual_inc",
        "annual_inc_log1p",
        "dti",
        "revol_util_num",
        "revol_bal",
        "revol_bal_log1p",
        "fico_range_low",
        "fico_range_high",
        "fico_avg",
        "fico_span",
        "delinq_2yrs",
        "open_acc",
        "installment",
        "term_months",
        "installment_to_monthly_income",
        "revol_bal_to_income",
        "dti_x_revol_util",
        "rate_x_term",
    ]
    categorical_candidates = ["term", "grade", "sub_grade", "refi_pathway"]

    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    categorical_cols = [c for c in categorical_candidates if c in df.columns]
    return numeric_cols, categorical_cols
