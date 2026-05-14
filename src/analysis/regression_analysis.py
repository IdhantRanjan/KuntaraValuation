"""
Regression Analysis — Cross-sectional regressions of visual factors on IPO outcomes.

Tests:
  H1: Visual features predict first-day underpricing beyond text and financials.
  H2: Tangibility factors predict lower post-IPO volatility.
  H3: Visual factors have incremental R² over financial ratios alone.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.evaluation.statistical_tests import nested_f_test

logger = logging.getLogger(__name__)


DEFAULT_CONTROLS = [
    "log_assets", "leverage", "rnd_intensity", "revenue_growth",
    "firm_age", "underwriter_rank", "vc_backed",
]
TEXT_FACTORS = ["lm_negative", "lm_uncertainty"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_design(
    df: pd.DataFrame,
    factors_df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    factor_cols: list[str],
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Merge df with factors_df on a shared key and drop NA on the design."""
    extra_cols = extra_cols or []

    if "cik" in df.columns and "cik" in factors_df.columns:
        merged = df.merge(factors_df, on="cik", how="inner")
    elif "ticker" in df.columns and "ticker" in factors_df.columns:
        merged = df.merge(factors_df, on="ticker", how="inner")
    else:
        if len(df) == len(factors_df):
            merged = pd.concat(
                [df.reset_index(drop=True), factors_df.reset_index(drop=True)],
                axis=1,
            )
        else:
            raise ValueError("Cannot merge df and factors_df: no shared key")

    cols = [outcome] + controls + factor_cols + extra_cols
    cols = [c for c in cols if c in merged.columns]
    sub = merged[cols].apply(pd.to_numeric, errors="coerce").dropna()
    return sub


def _ols_fit(
    sub: pd.DataFrame, outcome: str, regressors: list[str],
) -> dict:
    """Fit OLS with HC3 robust SE, return a tidy dict."""
    if not regressors:
        return {}
    X = sm.add_constant(sub[regressors])
    y = sub[outcome]
    model = sm.OLS(y, X, missing="drop").fit(cov_type="HC3")
    coefs = model.params.to_dict()
    tstats = model.tvalues.to_dict()
    pvalues = model.pvalues.to_dict()
    return {
        "coefficients": {k: float(v) for k, v in coefs.items()},
        "t_stats": {k: float(v) for k, v in tstats.items()},
        "p_values": {k: float(v) for k, v in pvalues.items()},
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "n": int(model.nobs),
        "y_pred": model.fittedvalues.to_numpy(),
        "y_true": y.to_numpy(),
        "n_params": int(len(model.params)),
    }


# ---------------------------------------------------------------------------
# Three-spec OLS (Spec 1 / 2 / 3)
# ---------------------------------------------------------------------------

def run_ols_with_visual_factors(
    df: pd.DataFrame,
    factors_df: pd.DataFrame,
    outcome: str = "first_day_return",
    controls: list[str] | None = None,
    factor_prefix: str = "VF",
) -> dict:
    """Run nested OLS specifications: controls → +VF → +VF+text."""
    controls = [c for c in (controls or DEFAULT_CONTROLS) if c in df.columns]
    factor_cols = [c for c in factors_df.columns if c.startswith(factor_prefix)]
    text_cols = [c for c in TEXT_FACTORS if c in df.columns]

    extra = factor_cols + text_cols
    sub = _prepare_design(df, factors_df, outcome, controls, factor_cols, text_cols)

    if sub.empty:
        logger.warning("No rows after NA drop for outcome %s", outcome)
        return {}

    spec1 = _ols_fit(sub, outcome, controls)
    spec2 = _ols_fit(sub, outcome, controls + factor_cols)
    spec3 = _ols_fit(sub, outcome, controls + factor_cols + text_cols)

    return {
        "spec1_controls_only": spec1,
        "spec2_controls_plus_visual": spec2,
        "spec3_controls_visual_text": spec3,
        "outcome": outcome,
        "n_factors": len(factor_cols),
        "factor_cols": factor_cols,
    }


# ---------------------------------------------------------------------------
# Fama-MacBeth
# ---------------------------------------------------------------------------

def _newey_west_se(series: np.ndarray, lags: int = 4) -> float:
    """Newey-West (HAC) standard error for an i.i.d. mean estimator."""
    s = np.asarray(series, dtype=float)
    s = s - s.mean()
    n = len(s)
    if n < 2:
        return float("nan")
    g0 = float(np.dot(s, s) / n)
    var = g0
    for L in range(1, min(lags + 1, n)):
        w = 1.0 - L / (lags + 1.0)
        gL = float(np.dot(s[L:], s[:-L]) / n)
        var += 2.0 * w * gL
    return float(np.sqrt(max(var, 0.0) / n))


def run_fama_macbeth(
    df: pd.DataFrame,
    factors_df: pd.DataFrame,
    outcome: str = "first_day_return",
    factor_prefix: str = "VF",
    controls: list[str] | None = None,
    nw_lags: int = 4,
) -> dict:
    """Annual cross-sectional regressions, then Newey-West tests on the means."""
    controls = [c for c in (controls or DEFAULT_CONTROLS) if c in df.columns]

    if "cik" in df.columns and "cik" in factors_df.columns:
        merged = df.merge(factors_df, on="cik", how="inner")
    elif "ticker" in df.columns and "ticker" in factors_df.columns:
        merged = df.merge(factors_df, on="ticker", how="inner")
    elif len(df) == len(factors_df):
        merged = pd.concat(
            [df.reset_index(drop=True), factors_df.reset_index(drop=True)],
            axis=1,
        )
    else:
        raise ValueError("Cannot merge df and factors_df")

    if "ipo_date" not in merged.columns:
        raise ValueError("ipo_date column required for Fama-MacBeth")

    merged["ipo_year"] = pd.to_datetime(merged["ipo_date"]).dt.year
    factor_cols = [c for c in merged.columns if c.startswith(factor_prefix)]
    regs = controls + factor_cols

    yearly_coefs: list[pd.Series] = []
    for yr, group in merged.groupby("ipo_year"):
        sub = group[[outcome] + regs].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < max(len(regs) + 5, 10):
            continue
        try:
            X = sm.add_constant(sub[regs])
            y = sub[outcome]
            res = sm.OLS(y, X).fit()
            row = res.params.copy()
            row.name = yr
            yearly_coefs.append(row)
        except Exception as e:
            logger.debug("FM year %s skipped: %s", yr, e)
            continue

    if not yearly_coefs:
        logger.warning("No years had enough data for Fama-MacBeth")
        return {"summary": pd.DataFrame()}

    yearly_df = pd.DataFrame(yearly_coefs)
    rows = []
    for col in yearly_df.columns:
        s = yearly_df[col].dropna().to_numpy()
        if s.size == 0:
            continue
        mean = float(s.mean())
        se = _newey_west_se(s, lags=nw_lags)
        t = mean / se if se > 0 else float("nan")
        rows.append({
            "regressor": col,
            "fm_mean": mean,
            "nw_se": se,
            "nw_t": t,
        })
    summary = pd.DataFrame(rows)

    return {
        "summary": summary,
        "yearly_coefficients": yearly_df,
        "n_years": int(len(yearly_coefs)),
    }


# ---------------------------------------------------------------------------
# Incremental R²
# ---------------------------------------------------------------------------

def incremental_r2_test(
    df: pd.DataFrame,
    factors_df: pd.DataFrame,
    outcomes: list[str] | None = None,
    factor_prefix: str = "VF",
    controls: list[str] | None = None,
) -> pd.DataFrame:
    """Nested F-test of visual factors over a controls-only baseline."""
    outcomes = outcomes or [
        "first_day_return", "broken_ipo", "post_ipo_volatility_6m",
    ]
    outcomes = [o for o in outcomes if o in df.columns]
    controls = [c for c in (controls or DEFAULT_CONTROLS) if c in df.columns]
    factor_cols = [c for c in factors_df.columns if c.startswith(factor_prefix)]

    rows = []
    for outcome in outcomes:
        sub = _prepare_design(df, factors_df, outcome, controls, factor_cols)
        if sub.empty or len(sub) < len(controls) + len(factor_cols) + 5:
            continue
        restricted = _ols_fit(sub, outcome, controls)
        full = _ols_fit(sub, outcome, controls + factor_cols)
        if not restricted or not full:
            continue
        f = nested_f_test(
            restricted["y_true"], restricted["y_pred"], full["y_pred"],
            restricted["n_params"], full["n_params"],
        )
        rows.append({
            "outcome": outcome,
            "R2_restricted": restricted["r_squared"],
            "R2_full": full["r_squared"],
            "delta_R2": full["r_squared"] - restricted["r_squared"],
            "F_stat": f.get("f_statistic", float("nan")),
            "p_value": f.get("p_value", float("nan")),
            "n": int(restricted["n"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Run cross-sectional regressions")
    p.add_argument("--universe", type=str,
                   default="data/processed/ipo_sample/ipo_universe.parquet")
    p.add_argument("--factors", type=str,
                   default="outputs/visual_factors.csv")
    p.add_argument("--output-dir", type=str, default="outputs/regressions")
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    universe_path = Path(args.universe)
    if universe_path.suffix == ".parquet":
        df = pd.read_parquet(universe_path)
    else:
        df = pd.read_csv(universe_path, parse_dates=["ipo_date"])

    factors_df = pd.read_csv(args.factors)

    logger.info("Running OLS specs (outcome=first_day_return)...")
    ols_results = run_ols_with_visual_factors(df, factors_df, "first_day_return")
    if ols_results:
        with (out_dir / "ols_specs.json").open("w") as f:
            import json
            json.dump(
                {k: {kk: vv for kk, vv in v.items()
                     if kk not in {"y_pred", "y_true"}}
                 if isinstance(v, dict) else v
                 for k, v in ols_results.items()},
                f, indent=2, default=float,
            )

    try:
        logger.info("Running Fama-MacBeth...")
        fm = run_fama_macbeth(df, factors_df, "first_day_return")
        fm["summary"].to_csv(out_dir / "fama_macbeth_summary.csv", index=False)
    except Exception as e:
        logger.warning("Fama-MacBeth failed: %s", e)

    logger.info("Computing incremental R²...")
    inc = incremental_r2_test(df, factors_df)
    inc.to_csv(out_dir / "incremental_r2.csv", index=False)
    logger.info("\n%s", inc.to_string())

    logger.info("Saved → %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
