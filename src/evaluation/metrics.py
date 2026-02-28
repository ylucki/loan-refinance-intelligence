"""Evaluation helpers for classification/regression and threshold tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdSelection:
    """Container for threshold-selection output."""

    threshold: float
    objective: str
    objective_value: float


def classification_metrics_from_proba(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute a compact set of classification metrics from probabilities."""

    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(proba, dtype=float).clip(1e-6, 1.0 - 1e-6)
    pred = (p_arr >= float(threshold)).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_arr, p_arr)),
        "pr_auc": float(average_precision_score(y_arr, p_arr)),
        "log_loss": float(log_loss(y_arr, p_arr)),
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "precision": float(precision_score(y_arr, pred, zero_division=0)),
        "recall": float(recall_score(y_arr, pred, zero_division=0)),
        "f1": float(f1_score(y_arr, pred, zero_division=0)),
    }


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    pred: pd.Series | np.ndarray,
) -> dict:
    """Compute regression metrics used by the project."""

    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(pred, dtype=float)
    rmse = np.sqrt(mean_squared_error(y_arr, p_arr))
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_arr, p_arr)),
        "mape": float(mean_absolute_percentage_error(y_arr, p_arr)),
        "r2": float(r2_score(y_arr, p_arr)),
    }


def threshold_sweep(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Evaluate precision/recall/F1 across candidate thresholds."""

    rows: List[dict] = []
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(proba, dtype=float)

    for t in thresholds:
        pred = (p_arr >= float(t)).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y_arr, pred, zero_division=0)),
                "recall": float(recall_score(y_arr, pred, zero_division=0)),
                "f1": float(f1_score(y_arr, pred, zero_division=0)),
                "positive_rate": float(pred.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("threshold")


def select_threshold(
    table: pd.DataFrame,
    objective: str = "f1",
    min_precision: float | None = None,
) -> ThresholdSelection:
    """Select threshold by objective with optional minimum precision guardrail."""

    if table.empty:
        raise ValueError("Threshold table is empty.")
    if objective not in table.columns:
        raise ValueError(f"Objective column not found: {objective}")

    candidates = table.copy()
    if min_precision is not None:
        candidates = candidates[candidates["precision"] >= float(min_precision)]
        if candidates.empty:
            candidates = table.copy()

    # Use objective desc, then threshold asc for stable tie-breaking.
    best = candidates.sort_values([objective, "threshold"], ascending=[False, True]).iloc[0]
    return ThresholdSelection(
        threshold=float(best["threshold"]),
        objective=objective,
        objective_value=float(best[objective]),
    )
