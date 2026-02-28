"""Core data-wrangling utilities for loan refinance modeling.

This module centralizes the transformations used by notebooks/tests:
1. Normalize raw fields (`to_numeric_percent`, `parse_issue_date`).
2. Build modeling labels/pathways (`add_outcome_flags`, `add_refi_pathway`).
3. Create analysis/training frames (`apply_core_wrangling`, `resolved_modeling_frame`).
4. Report dataset health and leakage policy (`summarize_outcomes`, `data_profile`,
   `quality_checks`, `leakage_exclusions`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


# --- Label/status configuration ---------------------------------------------

DEFAULT_STATUSES = {
    "charged off",
    "default",
    "does not meet the credit policy. status:charged off",
}

UNRESOLVED_STATUSES = {
    "current",
    "in grace period",
    "late (31-120 days)",
    "late (16-30 days)",
}


@dataclass(frozen=True)
class WranglingSummary:
    """Compact outcome summary used in reports and notebook checkpoints."""

    rows_total: int
    rows_resolved: int
    rows_unresolved: int
    unresolved_ratio: float
    default_rate_all: float
    default_rate_resolved: float


# --- Field normalization helpers --------------------------------------------

def to_numeric_percent(series: pd.Series) -> pd.Series:
    """Convert percent-like strings to numeric values.

    Examples:
    - "13.5%" -> 13.5
    - " 42 " -> 42.0
    - invalid values -> NaN
    """

    cleaned = series.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def parse_issue_date(series: pd.Series) -> pd.Series:
    """Parse issue-date strings into datetimes.

    Tries LendingClub format (`Jan-2018`) first, then falls back to pandas'
    generic parser when needed.
    """

    issue = pd.to_datetime(series, format="%b-%Y", errors="coerce")
    if issue.isna().all():
        issue = pd.to_datetime(series, errors="coerce")
    return issue


# --- Label/pathway builders --------------------------------------------------

def add_outcome_flags(df: pd.DataFrame, status_col: str = "loan_status") -> pd.DataFrame:
    """Add clean status text plus supervised-label helper flags.

    Adds:
    - `loan_status_clean`
    - `is_unresolved` (1 for unresolved/censored statuses)
    - `is_default` (1 for default outcomes)
    - `is_resolved` (1 for resolved outcomes)
    """

    out = df.copy()
    status = out.get(status_col, pd.Series(index=out.index, dtype="object")).astype(str).str.strip().str.lower()
    out["loan_status_clean"] = status
    out["is_unresolved"] = status.isin(UNRESOLVED_STATUSES).astype("Int64")
    out["is_default"] = status.isin(DEFAULT_STATUSES).astype("Int64")
    out["is_resolved"] = (1 - out["is_unresolved"]).astype("Int64")
    return out


def add_refi_pathway(df: pd.DataFrame, purpose_col: str = "purpose") -> pd.DataFrame:
    """Map loan purpose into refinance pathways used by recommendation logic."""

    out = df.copy()
    purpose = out.get(purpose_col, pd.Series(index=out.index, dtype="object")).astype(str).str.strip().str.lower()
    pathway = np.where(
        purpose.eq("credit_card"),
        "cc_to_pl",
        np.where(purpose.isin(["debt_consolidation", "personal"]), "pl_refinance", "other"),
    )
    out["refi_pathway"] = pd.Categorical(pathway, categories=["cc_to_pl", "pl_refinance", "other"])
    return out


# --- Pipeline entry points ---------------------------------------------------

def apply_core_wrangling(df: pd.DataFrame) -> pd.DataFrame:
    """Run the core enrichment pipeline on a raw dataframe.

    This is the main entry point and applies parsing, date features, outcome
    flags, and refinance pathway mapping in one pass.
    """

    out = df.copy()
    if "int_rate" in out.columns:
        out["int_rate_num"] = to_numeric_percent(out["int_rate"])
    if "revol_util" in out.columns:
        out["revol_util_num"] = to_numeric_percent(out["revol_util"])
    if "issue_d" in out.columns:
        out["issue_date"] = parse_issue_date(out["issue_d"])
        out["issue_year"] = out["issue_date"].dt.year
    out = add_outcome_flags(out)
    out = add_refi_pathway(out)
    return out


def resolved_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return only resolved rows for supervised risk modeling."""

    if "is_resolved" not in df.columns:
        return df.copy()
    return df[df["is_resolved"] == 1].copy()


# --- Reporting and diagnostics ----------------------------------------------

def summarize_outcomes(df: pd.DataFrame) -> WranglingSummary:
    """Summarize resolved/unresolved mix and observed default rates."""

    rows_total = int(len(df))
    unresolved = pd.to_numeric(df.get("is_unresolved"), errors="coerce")
    resolved = pd.to_numeric(df.get("is_resolved"), errors="coerce")
    default = pd.to_numeric(df.get("is_default"), errors="coerce")

    rows_unresolved = int(unresolved.fillna(0).sum()) if unresolved is not None else 0
    rows_resolved = int(resolved.fillna(0).sum()) if resolved is not None else 0

    default_rate_all = float(default.mean()) if default is not None and len(default) else float("nan")
    default_rate_resolved = float(default[resolved == 1].mean()) if rows_resolved > 0 else float("nan")

    unresolved_ratio = (rows_unresolved / rows_total) if rows_total else 0.0

    return WranglingSummary(
        rows_total=rows_total,
        rows_resolved=rows_resolved,
        rows_unresolved=rows_unresolved,
        unresolved_ratio=unresolved_ratio,
        default_rate_all=default_rate_all,
        default_rate_resolved=default_rate_resolved,
    )


def data_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Profile each column's type, missingness, and cardinality."""

    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_pct": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
            "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
        }
    ).sort_values("missing_pct", ascending=False)


def quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Run lightweight data quality checks and return a check result table."""

    checks: List[Dict[str, object]] = []

    if "id" in df.columns:
        dup_count = int(df["id"].duplicated().sum())
        checks.append(
            {
                "check": "duplicate_id_rows",
                "status": "pass" if dup_count == 0 else "warn",
                "value": dup_count,
                "details": "Expected unique loan id per row.",
            }
        )

    if "annual_inc" in df.columns:
        invalid = int((pd.to_numeric(df["annual_inc"], errors="coerce") < 0).sum())
        checks.append(
            {
                "check": "negative_annual_income_rows",
                "status": "pass" if invalid == 0 else "warn",
                "value": invalid,
                "details": "Annual income should not be negative.",
            }
        )

    if "int_rate_num" in df.columns:
        vals = pd.to_numeric(df["int_rate_num"], errors="coerce")
        invalid = int(((vals < 0) | (vals > 60)).sum())
        checks.append(
            {
                "check": "int_rate_out_of_range_rows",
                "status": "pass" if invalid == 0 else "warn",
                "value": invalid,
                "details": "Interest rate expected in [0, 60] percent.",
            }
        )

    if "revol_util_num" in df.columns:
        vals = pd.to_numeric(df["revol_util_num"], errors="coerce")
        invalid = int(((vals < 0) | (vals > 200)).sum())
        checks.append(
            {
                "check": "revol_util_out_of_range_rows",
                "status": "pass" if invalid == 0 else "warn",
                "value": invalid,
                "details": "Utilization expected in [0, 200] for raw data tolerance.",
            }
        )

    if "is_unresolved" in df.columns:
        unresolved = int(pd.to_numeric(df["is_unresolved"], errors="coerce").fillna(0).sum())
        checks.append(
            {
                "check": "unresolved_status_rows",
                "status": "info",
                "value": unresolved,
                "details": "Rows excluded from supervised risk labels.",
            }
        )

    return pd.DataFrame(checks)


def leakage_exclusions() -> pd.DataFrame:
    """List known post-origination fields excluded to prevent leakage."""

    rows = [
        {
            "column": "total_pymnt",
            "why_excluded": "Post-origination repayment outcome not available at decision time.",
        },
        {
            "column": "last_pymnt_d",
            "why_excluded": "Future repayment timing leaks target progression.",
        },
        {
            "column": "last_credit_pull_d",
            "why_excluded": "Post-origination servicing metadata may leak maturity/outcome state.",
        },
    ]
    return pd.DataFrame(rows)


# --- Validation helpers ------------------------------------------------------

def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    """Return required columns that are missing from the dataframe."""

    required_list = list(required)
    return [c for c in required_list if c not in df.columns]
