"""
Gradient-Boosted Tree Baselines — LightGBM / XGBoost on text+tabular features.

Strong non-deep-learning benchmark using sentence embeddings for text
and standard financial ratios for tabular features.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def build_gbm_features(
    tabular_features: np.ndarray,
    text_embeddings: np.ndarray | None = None,
    image_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """
    Concatenate available feature sets for GBM input.

    Args:
        tabular_features: (N, d_tab) standardized financials.
        text_embeddings: (N, d_text) sentence embeddings (optional).
        image_embeddings: (N, d_img) image embeddings (optional).

    Returns:
        (N, d_total) concatenated feature matrix.
    """
    parts = [tabular_features]
    if text_embeddings is not None:
        parts.append(text_embeddings)
    if image_embeddings is not None:
        parts.append(image_embeddings)
    return np.hstack(parts)


def run_lightgbm_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = "regression",
    params: dict | None = None,
) -> dict:
    """
    Train and evaluate a LightGBM model.

    Args:
        task: "regression" or "classification".
        params: LightGBM hyperparameters.
    """
    import lightgbm as lgb

    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    if task == "regression":
        model = lgb.LGBMRegressor(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        y_pred = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }
    else:
        model = lgb.LGBMClassifier(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        metrics = {
            "auc": float(roc_auc_score(y_test, y_pred_proba)),
            "accuracy": float((y_pred == y_test).mean()),
        }

    metrics["n_features"] = X_train.shape[1]
    metrics["best_iteration"] = model.best_iteration_ if hasattr(model, "best_iteration_") else -1

    # Feature importance
    importance = model.feature_importances_
    metrics["top_features"] = np.argsort(importance)[-10:][::-1].tolist()

    return metrics


def run_xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = "regression",
    params: dict | None = None,
) -> dict:
    """Train and evaluate an XGBoost model."""
    import xgboost as xgb

    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if params:
        default_params.update(params)

    if task == "regression":
        default_params["objective"] = "reg:squarederror"
        model = xgb.XGBRegressor(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        y_pred = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }
    else:
        default_params["objective"] = "binary:logistic"
        model = xgb.XGBClassifier(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        metrics = {
            "auc": float(roc_auc_score(y_test, y_pred_proba)),
            "accuracy": float((y_pred == y_test).mean()),
        }

    metrics["n_features"] = X_train.shape[1]
    return metrics


def run_all_gbm_baselines(
    X_train: np.ndarray,
    y_train_reg: np.ndarray,
    X_test: np.ndarray,
    y_test_reg: np.ndarray,
    y_train_cls: np.ndarray | None = None,
    y_test_cls: np.ndarray | None = None,
    output_dir: str | Path = "outputs/baselines",
) -> pd.DataFrame:
    """Run LightGBM and XGBoost for both regression and classification."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Regression (underpricing)
    results["lgbm_underpricing"] = run_lightgbm_baseline(
        X_train, y_train_reg, X_test, y_test_reg, task="regression"
    )
    results["xgb_underpricing"] = run_xgboost_baseline(
        X_train, y_train_reg, X_test, y_test_reg, task="regression"
    )

    # Classification (broken IPO)
    if y_train_cls is not None and y_test_cls is not None:
        results["lgbm_broken"] = run_lightgbm_baseline(
            X_train, y_train_cls, X_test, y_test_cls, task="classification"
        )
        results["xgb_broken"] = run_xgboost_baseline(
            X_train, y_train_cls, X_test, y_test_cls, task="classification"
        )

    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / "gbm_results.csv")
    logger.info("GBM baseline results:\n%s", results_df.to_string())

    return results_df
