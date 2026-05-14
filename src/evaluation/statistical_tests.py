"""
Statistical Tests — Diebold-Mariano, nested model comparisons, and bootstrap tests.

Supplements the metrics module with formal inference tests for the paper.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def nested_f_test(
    y_true: np.ndarray,
    y_pred_restricted: np.ndarray,
    y_pred_full: np.ndarray,
    n_params_restricted: int,
    n_params_full: int,
) -> dict[str, float]:
    """
    F-test for nested model comparison.

    Tests whether the full model (with additional features) significantly
    improves upon the restricted model.

    H0: The additional features have no predictive power.
    """
    n = len(y_true)
    sse_r = np.sum((y_true - y_pred_restricted) ** 2)
    sse_f = np.sum((y_true - y_pred_full) ** 2)

    df_num = n_params_full - n_params_restricted
    df_den = n - n_params_full

    if df_den <= 0 or df_num <= 0 or sse_f <= 0:
        return {"f_statistic": 0.0, "p_value": 1.0, "significant": False}

    f_stat = ((sse_r - sse_f) / df_num) / (sse_f / df_den)
    p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "df_num": df_num,
        "df_den": df_den,
        "sse_restricted": float(sse_r),
        "sse_full": float(sse_f),
        "significant": p_value < 0.05,
    }


def bootstrap_r2_difference(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap test for the difference in R² between two models.

    Tests H0: R²(model_2) - R²(model_1) = 0.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        r2_1 = 1 - np.sum((y_true[idx] - y_pred_1[idx]) ** 2) / np.sum(
            (y_true[idx] - y_true[idx].mean()) ** 2
        )
        r2_2 = 1 - np.sum((y_true[idx] - y_pred_2[idx]) ** 2) / np.sum(
            (y_true[idx] - y_true[idx].mean()) ** 2
        )
        diffs.append(r2_2 - r2_1)

    diffs = np.array(diffs)
    alpha = 1 - confidence
    ci_low = np.percentile(diffs, 100 * alpha / 2)
    ci_high = np.percentile(diffs, 100 * (1 - alpha / 2))
    p_value = np.mean(diffs <= 0)  # One-sided: P(model_2 ≤ model_1)

    return {
        "mean_r2_diff": float(diffs.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "significant": ci_low > 0,  # Full CI above zero
    }


def run_all_comparisons(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    baseline_name: str = "text_tabular",
    output_dir: str = "outputs/analysis",
) -> pd.DataFrame:
    """
    Run all statistical comparisons against a baseline model.

    Args:
        predictions: {model_name: y_pred array}.
        baseline_name: Which model to compare against.
    """
    from src.evaluation.metrics import diebold_mariano_test

    results = []
    baseline_pred = predictions.get(baseline_name)
    if baseline_pred is None:
        logger.error("Baseline '%s' not found in predictions", baseline_name)
        return pd.DataFrame()

    for model_name, y_pred in predictions.items():
        if model_name == baseline_name:
            continue

        # DM test
        dm = diebold_mariano_test(y_true, baseline_pred, y_pred)

        # Bootstrap R² difference
        boot = bootstrap_r2_difference(y_true, baseline_pred, y_pred)

        results.append({
            "model": model_name,
            "vs_baseline": baseline_name,
            "dm_statistic": dm["dm_statistic"],
            "dm_p_value": dm["p_value"],
            "dm_model2_better": dm["model_2_better"],
            "r2_diff_mean": boot["mean_r2_diff"],
            "r2_diff_ci_low": boot["ci_low"],
            "r2_diff_ci_high": boot["ci_high"],
            "r2_diff_p_value": boot["p_value"],
            "r2_improvement_significant": boot["significant"],
        })

    results_df = pd.DataFrame(results)
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(Path(output_dir) / "statistical_comparisons.csv", index=False)
    logger.info("Statistical comparisons:\n%s", results_df.to_string())
    return results_df
