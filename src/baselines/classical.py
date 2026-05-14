"""
Classical Baselines — OLS, Lasso with LM sentiment dictionaries.

Implements the traditional IPO-literature baselines:
  - OLS with classic underpricing determinants
  - OLS + Loughran-McDonald sentiment scores
  - Lasso with all features
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# Loughran-McDonald sentiment word lists (abbreviated — full lists loaded from file)
LM_CATEGORIES = [
    "negative", "positive", "uncertainty", "litigious",
    "strong_modal", "weak_modal", "constraining",
]


def compute_lm_sentiment(
    text: str,
    lm_dict_path: str | Path | None = None,
) -> dict[str, float]:
    """
    Compute Loughran-McDonald dictionary-based sentiment scores.

    Returns proportions (count / total_words) for each category.
    """
    words = text.lower().split()
    total = len(words) if words else 1

    # Default minimal word lists (for testing; use full LM dictionary in production)
    lm_words = {
        "negative": {"loss", "losses", "decline", "declining", "adverse", "adversely",
                     "fail", "failure", "impair", "impairment", "risk", "litigation"},
        "positive": {"gain", "gains", "profit", "profitable", "benefit", "beneficial",
                     "growth", "improve", "improved", "strong", "strength", "opportunity"},
        "uncertainty": {"may", "might", "could", "possible", "possibly", "uncertain",
                       "uncertainty", "approximate", "approximately", "depend", "depends"},
        "litigious": {"arbitration", "claimant", "defendant", "deposition", "injunction",
                     "lawsuit", "litigation", "plaintiff", "settlement", "tribunal"},
        "constraining": {"commit", "commits", "committed", "binding", "bound",
                        "compel", "compelled", "comply", "obligation", "obligated"},
    }

    if lm_dict_path and Path(lm_dict_path).exists():
        import json
        lm_words = json.loads(Path(lm_dict_path).read_text())

    word_set = set(words)
    scores = {}
    for category, cat_words in lm_words.items():
        count = len(word_set & cat_words)
        scores[f"lm_{category}"] = count / total

    return scores


def build_classical_features(
    df: pd.DataFrame,
    text_dict: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Construct feature matrix with classic IPO determinants + LM sentiment.

    Args:
        df: IPO universe DataFrame.
        text_dict: Mapping from CIK to Risk Factors text.

    Returns:
        Feature DataFrame aligned with input.
    """
    features = pd.DataFrame(index=df.index)

    # Classic determinants
    for col in ["offer_size", "firm_age", "underwriter_rank", "vc_backed",
                "log_assets", "leverage", "rnd_intensity", "revenue_growth"]:
        if col in df.columns:
            features[col] = df[col].fillna(0)

    # LM sentiment features (if text available)
    if text_dict is not None:
        for idx, row in df.iterrows():
            cik = str(row.get("cik", ""))
            text = text_dict.get(cik, "")
            if text:
                scores = compute_lm_sentiment(text)
                for key, val in scores.items():
                    features.at[idx, key] = val

    features = features.fillna(0)
    return features


def run_classical_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "first_day_return",
    text_dict: dict[str, str] | None = None,
    output_dir: str | Path = "outputs/baselines",
) -> dict[str, dict]:
    """
    Run OLS and Lasso baselines and report metrics.

    Returns dict of {model_name: {mae, rmse, r2}}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build features
    X_train = build_classical_features(train_df, text_dict)
    X_test = build_classical_features(test_df, text_dict)

    y_train = train_df[target_col].fillna(0).values
    y_test = test_df[target_col].fillna(0).values

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- OLS (financials only) ---
    fin_cols = [c for c in X_train.columns if not c.startswith("lm_")]
    if fin_cols:
        fin_mask = [X_train.columns.get_loc(c) for c in fin_cols]
        ols_fin = LinearRegression()
        ols_fin.fit(X_train_scaled[:, fin_mask], y_train)
        y_pred = ols_fin.predict(X_test_scaled[:, fin_mask])
        results["ols_financials"] = _eval_metrics(y_test, y_pred)

    # --- OLS (financials + sentiment) ---
    ols_full = LinearRegression()
    ols_full.fit(X_train_scaled, y_train)
    y_pred = ols_full.predict(X_test_scaled)
    results["ols_full"] = _eval_metrics(y_test, y_pred)

    # --- Lasso with CV ---
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_test_scaled)
    results["lasso_cv"] = _eval_metrics(y_test, y_pred)
    results["lasso_cv"]["alpha"] = float(lasso.alpha_)
    results["lasso_cv"]["n_nonzero"] = int(np.sum(np.abs(lasso.coef_) > 1e-6))

    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / "classical_results.csv")
    logger.info("Classical baseline results:\n%s", results_df.to_string())

    return results


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
