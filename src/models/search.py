"""Model search utilities with CV, randomized search, and grid refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR


@dataclass(frozen=True)
class SearchOutcome:
    """Search summary object used by the notebook orchestration."""

    leaderboard: pd.DataFrame
    best_estimators: dict[str, Any]
    champion_name: str
    primary_metric: str


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Build the shared tabular preprocessing graph."""

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def classification_candidates(enable_svm: bool = True) -> dict[str, Dict[str, Any]]:
    """Candidate model registry for classification."""

    candidates: dict[str, Dict[str, Any]] = {
        "logistic_l2": {
            "estimator": LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=2000,
            ),
            "random_params": {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
            "grid_params": {"model__C": [0.1, 0.3, 1.0, 3.0]},
        },
        "logistic_l1": {
            "estimator": LogisticRegression(
                penalty="l1",
                solver="saga",
                class_weight="balanced",
                max_iter=4000,
            ),
            "random_params": {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
            "grid_params": {"model__C": [0.05, 0.1, 0.3, 1.0]},
        },
        "logistic_elasticnet": {
            "estimator": LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                class_weight="balanced",
                max_iter=4000,
            ),
            "random_params": {
                "model__C": [0.05, 0.1, 0.3, 1.0, 3.0],
                "model__l1_ratio": [0.2, 0.5, 0.8],
            },
            "grid_params": {
                "model__C": [0.1, 0.3, 1.0],
                "model__l1_ratio": [0.3, 0.5, 0.7],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "random_params": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [8, 12, 16, None],
                "model__min_samples_leaf": [5, 10, 20, 40],
                "model__max_features": ["sqrt", 0.5, 0.8],
                "model__class_weight": ["balanced", "balanced_subsample"],
            },
            "grid_params": {
                "model__n_estimators": [300, 500],
                "model__max_depth": [10, 14, None],
                "model__min_samples_leaf": [10, 20],
            },
        },
        "linear_svm_calibrated": {
            "estimator": CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced", max_iter=6000, dual="auto"),
                method="sigmoid",
                cv=3,
            ),
            "random_params": {"model__estimator__C": [0.05, 0.1, 0.3, 1.0, 3.0]},
            "grid_params": {"model__estimator__C": [0.1, 0.3, 1.0]},
            "sample_cap": 70000,
        },
    }
    if enable_svm:
        candidates["svm_rbf"] = {
            "estimator": SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
            ),
            "random_params": {
                "model__C": [0.3, 1.0, 3.0, 10.0],
                "model__gamma": ["scale", 0.01, 0.03, 0.1],
            },
            "grid_params": {
                "model__C": [0.5, 1.0, 3.0],
                "model__gamma": ["scale", 0.01, 0.03],
            },
            "sample_cap": 35000,
        }
    return candidates


def regression_candidates(enable_svm: bool = True) -> dict[str, Dict[str, Any]]:
    """Candidate model registry for regression."""

    candidates: dict[str, Dict[str, Any]] = {
        "ridge": {
            "estimator": Ridge(random_state=42),
            "random_params": {"model__alpha": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]},
            "grid_params": {"model__alpha": [0.1, 0.3, 1.0, 3.0]},
        },
        "lasso": {
            "estimator": Lasso(random_state=42, max_iter=7000),
            "random_params": {"model__alpha": [0.0003, 0.0007, 0.001, 0.003, 0.01]},
            "grid_params": {"model__alpha": [0.0005, 0.001, 0.003]},
        },
        "elasticnet": {
            "estimator": ElasticNet(random_state=42, max_iter=7000),
            "random_params": {
                "model__alpha": [0.0003, 0.0007, 0.001, 0.003, 0.01],
                "model__l1_ratio": [0.2, 0.5, 0.8],
            },
            "grid_params": {
                "model__alpha": [0.0005, 0.001, 0.003],
                "model__l1_ratio": [0.3, 0.5, 0.7],
            },
        },
        "random_forest_regressor": {
            "estimator": RandomForestRegressor(random_state=42, n_jobs=-1),
            "random_params": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [8, 12, 16, None],
                "model__min_samples_leaf": [5, 10, 20, 40],
                "model__max_features": ["sqrt", 0.5, 0.8],
            },
            "grid_params": {
                "model__n_estimators": [300, 500],
                "model__max_depth": [10, 14, None],
                "model__min_samples_leaf": [10, 20],
            },
        },
        "linear_svr": {
            "estimator": LinearSVR(random_state=42, max_iter=7000),
            "random_params": {
                "model__C": [0.1, 0.3, 1.0, 3.0],
                "model__epsilon": [0.01, 0.03, 0.1, 0.2],
            },
            "grid_params": {
                "model__C": [0.3, 1.0, 3.0],
                "model__epsilon": [0.03, 0.1, 0.2],
            },
            "sample_cap": 80000,
        },
    }
    if enable_svm:
        candidates["svr_rbf"] = {
            "estimator": SVR(kernel="rbf"),
            "random_params": {
                "model__C": [1.0, 3.0, 10.0],
                "model__epsilon": [0.03, 0.1, 0.2],
                "model__gamma": ["scale", 0.01, 0.03],
            },
            "grid_params": {
                "model__C": [1.0, 3.0, 10.0],
                "model__epsilon": [0.03, 0.1],
                "model__gamma": ["scale", 0.01],
            },
            "sample_cap": 30000,
        }
    return candidates


def _parameter_space_size(params: Dict[str, Any]) -> int:
    total = 1
    for values in params.values():
        if isinstance(values, list):
            total *= max(len(values), 1)
    return total


def _subset_for_candidate(
    X: pd.DataFrame,
    y: pd.Series,
    issue_date: pd.Series,
    sample_cap: int | None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if sample_cap is None or len(X) <= sample_cap:
        return X, y, issue_date
    # Keep the latest segment to preserve time-order semantics.
    start = len(X) - int(sample_cap)
    return X.iloc[start:].copy(), y.iloc[start:].copy(), issue_date.iloc[start:].copy()


def _fit_search(
    estimator: Pipeline,
    params: Dict[str, Any],
    scoring: Dict[str, str],
    refit_metric: str,
    cv: TimeSeriesSplit,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int,
    n_jobs: int,
    random_state: int,
) -> RandomizedSearchCV:
    effective_iter = min(max(1, n_iter), max(1, _parameter_space_size(params)))
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_iter=effective_iter,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )
    search.fit(X, y)
    return search


def _fit_grid(
    estimator: Pipeline,
    param_grid: Dict[str, Any],
    scoring: Dict[str, str],
    refit_metric: str,
    cv: TimeSeriesSplit,
    X: pd.DataFrame,
    y: pd.Series,
    n_jobs: int,
) -> GridSearchCV:
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        n_jobs=n_jobs,
        verbose=0,
    )
    grid.fit(X, y)
    return grid


def run_search_suite(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    issue_date_train: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    candidates: dict[str, Dict[str, Any]],
    scoring: Dict[str, str],
    refit_metric: str,
    random_iter: int = 10,
    grid_top_k: int = 2,
    cv_splits: int = 3,
    n_jobs: int = -1,
    random_state: int = 42,
    metric_transform: Dict[str, str] | None = None,
) -> SearchOutcome:
    """Run randomized search for all models, then grid-search top candidates."""

    rows: list[dict] = []
    best_estimators: dict[str, Any] = {}
    best_raw_scores: dict[str, float] = {}
    candidate_data: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]] = {}

    for name, cfg in candidates.items():
        X_fit, y_fit, d_fit = _subset_for_candidate(
            X_train,
            y_train,
            issue_date_train,
            sample_cap=cfg.get("sample_cap"),
        )
        order = np.argsort(pd.to_datetime(d_fit, errors="coerce").fillna(pd.Timestamp("1970-01-01")).to_numpy())
        X_fit = X_fit.iloc[order].copy()
        y_fit = y_fit.iloc[order].copy()
        d_fit = d_fit.iloc[order].copy()
        candidate_data[name] = (X_fit, y_fit, d_fit)

        n_splits = max(2, min(cv_splits, len(X_fit) - 1))
        if len(X_fit) <= 10 or n_splits < 2:
            continue

        cv = TimeSeriesSplit(n_splits=n_splits)
        pipe = Pipeline(
            [
                ("prep", make_preprocessor(numeric_cols, categorical_cols)),
                ("model", cfg["estimator"]),
            ]
        )

        try:
            try:
                search = _fit_search(
                    estimator=pipe,
                    params=cfg["random_params"],
                    scoring=scoring,
                    refit_metric=refit_metric,
                    cv=cv,
                    X=X_fit,
                    y=y_fit,
                    n_iter=random_iter,
                    n_jobs=n_jobs,
                    random_state=random_state,
                )
            except PermissionError:
                # Some environments restrict process-based parallel backends.
                search = _fit_search(
                    estimator=pipe,
                    params=cfg["random_params"],
                    scoring=scoring,
                    refit_metric=refit_metric,
                    cv=cv,
                    X=X_fit,
                    y=y_fit,
                    n_iter=random_iter,
                    n_jobs=1,
                    random_state=random_state,
                )
            idx = search.best_index_
            row = {
                "model": name,
                "stage": "random_search",
                "rows_used": int(len(X_fit)),
                "cv_primary_raw": float(search.best_score_),
                "best_params": jsonable(search.best_params_),
            }
            for metric in scoring.keys():
                mean_key = f"mean_test_{metric}"
                if mean_key in search.cv_results_:
                    row[metric] = float(search.cv_results_[mean_key][idx])
            rows.append(row)
            best_estimators[name] = search.best_estimator_
            best_raw_scores[name] = float(search.best_score_)
        except Exception as exc:
            rows.append(
                {
                    "model": name,
                    "stage": "random_search_failed",
                    "rows_used": int(len(X_fit)),
                    "cv_primary_raw": float("-inf"),
                    "error": str(exc),
                }
            )

    # Grid-refine top candidates by primary raw score.
    top_for_grid = sorted(best_raw_scores.items(), key=lambda kv: kv[1], reverse=True)[: max(0, int(grid_top_k))]
    for name, _score in top_for_grid:
        cfg = candidates[name]
        if not cfg.get("grid_params"):
            continue
        X_fit, y_fit, _ = candidate_data[name]
        n_splits = max(2, min(cv_splits, len(X_fit) - 1))
        if len(X_fit) <= 10 or n_splits < 2:
            continue
        cv = TimeSeriesSplit(n_splits=n_splits)

        try:
            try:
                grid = _fit_grid(
                    estimator=best_estimators[name],
                    param_grid=cfg["grid_params"],
                    scoring=scoring,
                    refit_metric=refit_metric,
                    cv=cv,
                    X=X_fit,
                    y=y_fit,
                    n_jobs=n_jobs,
                )
            except PermissionError:
                grid = _fit_grid(
                    estimator=best_estimators[name],
                    param_grid=cfg["grid_params"],
                    scoring=scoring,
                    refit_metric=refit_metric,
                    cv=cv,
                    X=X_fit,
                    y=y_fit,
                    n_jobs=1,
                )
            idx = grid.best_index_
            row = {
                "model": name,
                "stage": "grid_search",
                "rows_used": int(len(X_fit)),
                "cv_primary_raw": float(grid.best_score_),
                "best_params": jsonable(grid.best_params_),
            }
            for metric in scoring.keys():
                mean_key = f"mean_test_{metric}"
                if mean_key in grid.cv_results_:
                    row[metric] = float(grid.cv_results_[mean_key][idx])
            rows.append(row)
            if float(grid.best_score_) > best_raw_scores.get(name, float("-inf")):
                best_estimators[name] = grid.best_estimator_
                best_raw_scores[name] = float(grid.best_score_)
        except Exception as exc:
            rows.append(
                {
                    "model": name,
                    "stage": "grid_search_failed",
                    "rows_used": int(len(X_fit)),
                    "cv_primary_raw": float("-inf"),
                    "error": str(exc),
                }
            )

    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty or not best_raw_scores:
        raise RuntimeError("No model searches succeeded.")

    # Human-friendly transformed columns (e.g., neg_rmse -> rmse).
    metric_transform = metric_transform or {}
    for k, how in metric_transform.items():
        if k not in leaderboard.columns:
            continue
        if how == "neg_to_pos":
            leaderboard[f"{k}_human"] = -leaderboard[k]

    leaderboard = leaderboard.sort_values("cv_primary_raw", ascending=False).reset_index(drop=True)
    champion_name = max(best_raw_scores.items(), key=lambda kv: kv[1])[0]

    return SearchOutcome(
        leaderboard=leaderboard,
        best_estimators=best_estimators,
        champion_name=champion_name,
        primary_metric=refit_metric,
    )


def jsonable(value: Any) -> Any:
    """Recursively convert values to JSON-serializable primitives."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)
