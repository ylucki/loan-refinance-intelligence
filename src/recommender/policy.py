"""Recommendation-policy helpers for refinance decisioning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineering import parse_term_months


def emi(principal: np.ndarray, annual_rate: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Compute EMI from principal/rate/term with zero-rate handling."""

    r = (annual_rate / 12.0) / 100.0
    n = np.maximum(months, 1.0)
    return np.where(
        r <= 0,
        principal / n,
        (principal * r * np.power(1 + r, n)) / np.maximum(np.power(1 + r, n) - 1, 1e-9),
    )


def scenario_current_rate(df: pd.DataFrame) -> np.ndarray:
    """Build a scenario current-rate baseline by refinance pathway."""

    int_rate = pd.to_numeric(df.get("int_rate_num"), errors="coerce").fillna(15.0).to_numpy()
    pathway = df.get("refi_pathway", pd.Series(index=df.index, dtype="object")).astype(str)
    return np.where(
        pathway == "cc_to_pl",
        36.0,
        np.clip(int_rate + 5.0, 8.0, 45.0),
    )


def apply_recommendation_policy(
    df: pd.DataFrame,
    pd_score: np.ndarray,
    pred_rate: np.ndarray,
    hold_threshold: float = 0.25,
    consider_min_savings: float = 0.0,
    strong_min_savings: float = 1500.0,
    strong_max_pd: float = 0.15,
) -> pd.DataFrame:
    """Create recommendation labels from risk and savings estimates."""

    out = df.copy()
    pd_arr = np.asarray(pd_score, dtype=float)
    rate_arr = np.clip(np.asarray(pred_rate, dtype=float), 5.0, 45.0)

    months = parse_term_months(out.get("term", pd.Series(index=out.index, dtype="object"))).to_numpy(dtype=float)
    installment = pd.to_numeric(out.get("installment"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    principal = np.clip(installment * months, 5000.0, None)
    current_rate = scenario_current_rate(out)

    current_emi = emi(principal, current_rate, months)
    new_emi = emi(principal, rate_arr, months)
    monthly_savings = current_emi - new_emi
    risk_adjusted_savings = monthly_savings * (1.0 - pd_arr)

    recommendation = np.where(
        pd_arr >= hold_threshold,
        "hold",
        np.where(
            (monthly_savings >= strong_min_savings) & (pd_arr <= strong_max_pd),
            "strong_recommend",
            np.where(monthly_savings > consider_min_savings, "consider", "hold"),
        ),
    )

    out["pd_score"] = pd_arr
    out["pred_rate"] = rate_arr
    out["scenario_current_rate"] = current_rate
    out["monthly_savings"] = monthly_savings
    out["risk_adjusted_savings"] = risk_adjusted_savings
    out["recommendation"] = recommendation
    return out


def policy_simulation_table(
    df: pd.DataFrame,
    hold_thresholds: list[float],
    min_savings_values: list[float],
) -> pd.DataFrame:
    """Simulate policy outcomes under threshold choices."""

    rows: list[dict] = []
    if df.empty:
        return pd.DataFrame(rows)

    for hold_t in hold_thresholds:
        for min_sav in min_savings_values:
            rec = np.where(
                df["pd_score"].to_numpy() >= hold_t,
                "hold",
                np.where(df["monthly_savings"].to_numpy() > min_sav, "consider", "hold"),
            )
            consider_mask = rec == "consider"
            rows.append(
                {
                    "hold_threshold": hold_t,
                    "min_savings": min_sav,
                    "consider_rate": float(consider_mask.mean()),
                    "consider_count": int(consider_mask.sum()),
                    "avg_pd_consider": float(df.loc[consider_mask, "pd_score"].mean()) if consider_mask.any() else np.nan,
                    "avg_savings_consider": float(df.loc[consider_mask, "monthly_savings"].mean()) if consider_mask.any() else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["hold_threshold", "min_savings"]).reset_index(drop=True)
