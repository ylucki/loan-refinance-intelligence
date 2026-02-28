"""Feature-selection helpers (correlation filter + regularization signals)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Container for selected feature outputs."""

    selected_numeric: list[str]
    dropped_correlated: list[str]
    selection_table: pd.DataFrame


def _as_numeric_frame(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    cols_list = [c for c in cols if c in df.columns]
    out = pd.DataFrame(index=df.index)
    for c in cols_list:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def correlation_filter(
    df: pd.DataFrame,
    numeric_cols: list[str],
    threshold: float = 0.98,
) -> tuple[list[str], list[str]]:
    """Drop highly correlated numeric columns via upper-triangle scan."""

    if not numeric_cols:
        return [], []

    num_df = _as_numeric_frame(df, numeric_cols)
    if num_df.empty:
        return [], []

    imputer = SimpleImputer(strategy="median")
    imp = pd.DataFrame(imputer.fit_transform(num_df), columns=num_df.columns, index=num_df.index)
    corr = imp.corr().abs()

    to_drop: set[str] = set()
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] > threshold:
                to_drop.add(cols[j])

    kept = [c for c in numeric_cols if c not in to_drop]
    dropped = sorted(to_drop)
    return kept, dropped


def select_numeric_features_with_regularization(
    train_df: pd.DataFrame,
    target_cls: pd.Series,
    target_reg: pd.Series,
    numeric_cols: list[str],
    corr_threshold: float = 0.98,
    l1_c: float = 0.15,
    lasso_alpha: float = 0.0007,
    min_features: int = 8,
    random_state: int = 42,
) -> FeatureSelectionResult:
    """Select numeric features using correlation pruning + L1/Lasso signals."""

    if not numeric_cols:
        return FeatureSelectionResult([], [], pd.DataFrame(columns=["feature"]))

    kept_after_corr, dropped_correlated = correlation_filter(train_df, numeric_cols, threshold=corr_threshold)
    if not kept_after_corr:
        kept_after_corr = numeric_cols.copy()

    num_df = _as_numeric_frame(train_df, kept_after_corr)
    imputer = SimpleImputer(strategy="median")
    scaled = StandardScaler().fit_transform(imputer.fit_transform(num_df))

    y_cls = pd.to_numeric(target_cls, errors="coerce").fillna(0).astype(int).to_numpy()
    y_reg = pd.to_numeric(target_reg, errors="coerce").fillna(float(np.nanmean(pd.to_numeric(target_reg, errors="coerce")))).to_numpy()

    cls_model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=l1_c,
        class_weight="balanced",
        max_iter=4000,
        random_state=random_state,
    )
    cls_model.fit(scaled, y_cls)
    cls_abs = np.abs(cls_model.coef_[0])

    reg_model = Lasso(alpha=lasso_alpha, max_iter=8000, random_state=random_state)
    reg_model.fit(scaled, y_reg)
    reg_abs = np.abs(reg_model.coef_)

    selection_df = pd.DataFrame(
        {
            "feature": kept_after_corr,
            "logistic_l1_abs_coef": cls_abs,
            "lasso_abs_coef": reg_abs,
            "dropped_by_corr": False,
        }
    )

    selected_cls = set(selection_df.loc[selection_df["logistic_l1_abs_coef"] > 1e-8, "feature"])
    selected_reg = set(selection_df.loc[selection_df["lasso_abs_coef"] > 1e-8, "feature"])
    selected = selected_cls | selected_reg

    if len(selected) < min_features:
        # Build a simple blended score to guarantee a reasonable minimum set.
        cls_rank = selection_df["logistic_l1_abs_coef"].rank(method="dense", ascending=False)
        reg_rank = selection_df["lasso_abs_coef"].rank(method="dense", ascending=False)
        selection_df["blended_rank"] = cls_rank + reg_rank
        fallback = (
            selection_df.sort_values("blended_rank")
            .head(min_features)["feature"]
            .tolist()
        )
        selected |= set(fallback)
    else:
        selection_df["blended_rank"] = (
            selection_df["logistic_l1_abs_coef"].rank(method="dense", ascending=False)
            + selection_df["lasso_abs_coef"].rank(method="dense", ascending=False)
        )

    selection_df["selected"] = selection_df["feature"].isin(selected)

    if dropped_correlated:
        dropped_df = pd.DataFrame(
            {
                "feature": dropped_correlated,
                "logistic_l1_abs_coef": 0.0,
                "lasso_abs_coef": 0.0,
                "dropped_by_corr": True,
                "blended_rank": np.nan,
                "selected": False,
            }
        )
        selection_df = pd.concat([selection_df, dropped_df], ignore_index=True)

    selected_numeric = selection_df.loc[selection_df["selected"], "feature"].tolist()
    selected_numeric = [c for c in numeric_cols if c in selected_numeric]

    return FeatureSelectionResult(
        selected_numeric=selected_numeric,
        dropped_correlated=dropped_correlated,
        selection_table=selection_df.sort_values(["selected", "blended_rank"], ascending=[False, True]),
    )
