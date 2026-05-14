"""
Evaluation Metrics — Regression, classification, and statistical tests.

Provides:
  - Standard regression metrics (MAE, RMSE, R², adjusted R²)
  - Classification metrics (AUC, F1, precision, recall)
  - Diebold-Mariano test for predictive accuracy comparison
  - Decile analysis for economic significance
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int | None = None,
) -> dict[str, float]:
    """Compute standard regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    result = {"mae": mae, "rmse": rmse, "r2": r2}

    # Adjusted R²
    if n_features is not None and len(y_true) > n_features + 1:
        n = len(y_true)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        result["adj_r2"] = adj_r2

    return result


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary classification metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    result = {
        "auc": roc_auc_score(y_true, y_pred_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": float(np.mean(y_true == y_pred)),
    }
    return result


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------

def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    loss: str = "squared",
    h: int = 1,
) -> dict[str, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[d_t] = 0, where d_t = L(e_1t) - L(e_2t).

    A negative DM statistic means model 1 is better (lower loss).

    Args:
        y_true: Actual values.
        y_pred_1: Predictions from model 1 (benchmark).
        y_pred_2: Predictions from model 2 (proposed).
        loss: "squared" or "absolute".
        h: Forecast horizon.

    Returns:
        Dict with dm_statistic, p_value, model_1_better (bool).
    """
    e1 = y_true - y_pred_1
    e2 = y_true - y_pred_2

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    n = len(d)
    d_bar = d.mean()

    # HAC variance estimate (Newey-West with bandwidth h-1)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += 2 * gamma_k

    var_d = (gamma_0 + gamma_sum) / n

    if var_d <= 0:
        return {"dm_statistic": 0.0, "p_value": 1.0, "model_2_better": False}

    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "model_2_better": bool(d_bar > 0),  # Positive d_bar means model 2 has lower loss
    }


# ---------------------------------------------------------------------------
# Decile analysis for economic significance
# ---------------------------------------------------------------------------

def decile_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """
    Construct decile portfolios based on predicted underpricing.

    For each decile of predicted values, compute:
      - Mean predicted underpricing
      - Mean realized underpricing
      - Spread (top - bottom decile)
    """
    df = pd.DataFrame({
        "predicted": y_pred,
        "realized": y_true,
    })
    df["decile"] = pd.qcut(df["predicted"], n_quantiles, labels=False, duplicates="drop")

    summary = df.groupby("decile").agg(
        mean_predicted=("predicted", "mean"),
        mean_realized=("realized", "mean"),
        std_realized=("realized", "std"),
        count=("realized", "count"),
    ).reset_index()

    # Long-short spread
    top = summary.iloc[-1]["mean_realized"]
    bottom = summary.iloc[0]["mean_realized"]
    spread = top - bottom

    logger.info(
        "Decile spread: top=%.4f, bottom=%.4f, spread=%.4f",
        top, bottom, spread,
    )

    return summary


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(
    results: dict[str, dict[str, float]],
    metric_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a formatted comparison table across models.

    Args:
        results: {model_name: {metric: value}}.
        metric_order: Order of metrics in columns.
    """
    df = pd.DataFrame(results).T

    if metric_order:
        available = [c for c in metric_order if c in df.columns]
        df = df[available]

    return df
