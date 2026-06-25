"""
Two additions K asked for:

(1) Rank ICs: per-cohort Spearman correlation between predicted and realized
    BHAR. Reports mean IC, SE across cohorts, t-stat, and p-value. This
    measures ranking ability directly, separate from level accuracy
    (which the negative R²s already tell us is bad).

(2) Bootstrap CIs on the calendar-time long-short Sharpes. Resamples the
    cohort-level spread series with replacement B times, recomputes Sharpe
    each draw, reports 2.5/97.5 percentiles. Tells us which Sharpes are
    distinguishable from zero given how few cohorts we have.

Both run on fold_predictions.csv (real out-of-sample test-fold predictions)
merged with ipo_date from the universe parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "bhar"

UNIQUE_RATIO_FLOOR = 0.05
FORMATION_FREQ = "Q"
FORMATIONS_PER_YEAR = {"Q": 4, "M": 12, "Y": 1, "2Q": 2}
MIN_FIRMS_PER_COHORT = {"full": 10, "multimodal": 5}
N_BOOTSTRAP = 5000
RNG_SEED = 42


def _load_ipo_dates() -> pd.DataFrame:
    uni = pd.read_parquet(ROOT / "data/processed/ipo_sample/ipo_universe_final.parquet")
    uni = uni[["cik", "ipo_date"]].copy()
    uni["cik"] = uni["cik"].astype(str)
    uni["ipo_date"] = pd.to_datetime(uni["ipo_date"])
    return uni.drop_duplicates(subset=["cik"])


def _is_signal_meaningful(preds: np.ndarray) -> bool:
    if len(preds) == 0:
        return False
    return pd.Series(preds).nunique() / len(preds) >= UNIQUE_RATIO_FLOOR


def _quintile_with_jitter(preds: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, 1e-9, size=preds.shape)
    try:
        return pd.qcut(preds + jitter, q=5, labels=False, duplicates="drop").astype(float)
    except Exception:
        return np.full_like(preds, np.nan, dtype=float)


def compute_rank_ic(grp: pd.DataFrame, sample: str, freq: str = FORMATION_FREQ) -> dict:
    """Per-cohort Spearman IC, then aggregate across cohorts."""
    preds = grp["y_pred"].to_numpy()
    if not _is_signal_meaningful(preds):
        return {
            "ic_n_cohorts": 0, "ic_mean": np.nan, "ic_se": np.nan,
            "ic_tstat": np.nan, "ic_pval": np.nan,
            "ic_pooled": np.nan,
        }

    min_firms = MIN_FIRMS_PER_COHORT.get(sample, 10)
    grp = grp.copy()
    grp["cohort"] = grp["ipo_date"].dt.to_period(freq).astype(str)

    ics = []
    for _, cohort in grp.groupby("cohort"):
        if len(cohort) < min_firms:
            continue
        rho, _ = sp_stats.spearmanr(cohort["y_pred"], cohort["y_actual"])
        if np.isfinite(rho):
            ics.append(rho)

    pooled, _ = sp_stats.spearmanr(grp["y_pred"], grp["y_actual"])

    if len(ics) < 4:
        return {
            "ic_n_cohorts": len(ics),
            "ic_mean": float(np.mean(ics)) if ics else np.nan,
            "ic_se": np.nan, "ic_tstat": np.nan, "ic_pval": np.nan,
            "ic_pooled": float(pooled) if np.isfinite(pooled) else np.nan,
        }

    arr = np.array(ics)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    tstat = mean / se if se > 0 else np.nan
    pval = float(2 * (1 - sp_stats.t.cdf(abs(tstat), df=len(arr) - 1))) if np.isfinite(tstat) else np.nan
    return {
        "ic_n_cohorts": len(ics),
        "ic_mean": mean,
        "ic_se": se,
        "ic_tstat": float(tstat) if np.isfinite(tstat) else np.nan,
        "ic_pval": pval,
        "ic_pooled": float(pooled) if np.isfinite(pooled) else np.nan,
    }


def _cohort_spreads(grp: pd.DataFrame, sample: str, freq: str = FORMATION_FREQ) -> np.ndarray:
    preds = grp["y_pred"].to_numpy()
    if not _is_signal_meaningful(preds):
        return np.array([])

    min_firms = MIN_FIRMS_PER_COHORT.get(sample, 10)
    grp = grp.copy()
    grp["cohort"] = grp["ipo_date"].dt.to_period(freq).astype(str)

    spreads = []
    for cohort_id, cohort in grp.groupby("cohort"):
        if len(cohort) < min_firms:
            continue
        c_preds = cohort["y_pred"].to_numpy()
        c_actuals = cohort["y_actual"].to_numpy()
        if not _is_signal_meaningful(c_preds):
            continue
        q = _quintile_with_jitter(c_preds, seed=hash(cohort_id) % (2**31))
        valid = ~np.isnan(q)
        if valid.sum() < 5 or len(np.unique(q[valid])) < 5:
            continue
        qi = q[valid].astype(int)
        a = c_actuals[valid]
        q5 = a[qi == 4].mean() if (qi == 4).any() else np.nan
        q1 = a[qi == 0].mean() if (qi == 0).any() else np.nan
        if np.isfinite(q5) and np.isfinite(q1):
            spreads.append(q5 - q1)
    return np.array(spreads)


def bootstrap_sharpe_ci(spreads: np.ndarray, freq: str, n_boot: int, rng) -> dict:
    if len(spreads) < 4:
        return {
            "sharpe_point": np.nan, "sharpe_ci_lo": np.nan, "sharpe_ci_hi": np.nan,
            "sharpe_pval_zero": np.nan,
        }
    ann = float(np.sqrt(FORMATIONS_PER_YEAR[freq]))

    def _sharpe(x):
        sd = x.std(ddof=1)
        return (x.mean() / sd) * ann if sd > 0 else np.nan

    point = _sharpe(spreads)
    n = len(spreads)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(spreads, size=n, replace=True)
        boots[b] = _sharpe(sample)
    boots = boots[np.isfinite(boots)]
    if len(boots) < n_boot // 2:
        return {
            "sharpe_point": float(point) if np.isfinite(point) else np.nan,
            "sharpe_ci_lo": np.nan, "sharpe_ci_hi": np.nan,
            "sharpe_pval_zero": np.nan,
        }
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pval = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {
        "sharpe_point": float(point),
        "sharpe_ci_lo": float(lo),
        "sharpe_ci_hi": float(hi),
        "sharpe_pval_zero": pval,
    }


def main() -> int:
    fp = pd.read_csv(OUT / "fold_predictions.csv")
    fp["cik"] = fp["cik"].astype(int).astype(str)
    dates = _load_ipo_dates()
    merged = fp.merge(dates, on="cik", how="left")
    missing = merged["ipo_date"].isna().sum()
    if missing:
        logger.warning("%d / %d rows missing ipo_date", missing, len(merged))

    rng = np.random.default_rng(RNG_SEED)

    rows = []
    for (sample, model, horizon), grp in merged.groupby(["sample", "model", "horizon_months"]):
        ic = compute_rank_ic(grp, sample)
        spreads = _cohort_spreads(grp, sample)
        boot = bootstrap_sharpe_ci(spreads, FORMATION_FREQ, N_BOOTSTRAP, rng)
        rows.append({
            "sample": sample, "model": model, "horizon_months": int(horizon),
            **ic, **boot,
        })

    out = pd.DataFrame(rows)
    out_path = OUT / "rank_ic_and_sharpe_ci.csv"
    out.to_csv(out_path, index=False)
    logger.info("Saved %s (%d rows)", out_path, len(out))

    show = ["sample", "model", "horizon_months",
            "ic_n_cohorts", "ic_mean", "ic_se", "ic_tstat", "ic_pval",
            "sharpe_point", "sharpe_ci_lo", "sharpe_ci_hi", "sharpe_pval_zero"]

    print("\n=== RANK ICs AND BOOTSTRAPPED SHARPE CIs ===")
    print("(IC = per-cohort Spearman corr(pred, actual), aggregated across cohorts)")
    print("(Sharpe CI = 2.5/97.5 percentile of bootstrap distribution, n_boot=%d)\n" % N_BOOTSTRAP)

    for sample in ["full", "multimodal"]:
        sub = out[out["sample"] == sample].sort_values(["horizon_months", "model"])
        if sub.empty:
            continue
        print(f"--- {sample} ---")
        print(sub[show].to_string(index=False, float_format=lambda x: f"{x:7.4f}" if isinstance(x, float) else str(x)))
        print()

    print("=== SHARPES DISTINGUISHABLE FROM ZERO (95% CI excludes 0) ===")
    sig = out[(out["sharpe_ci_lo"] > 0) | (out["sharpe_ci_hi"] < 0)]
    if sig.empty:
        print("  none")
    else:
        print(sig[["sample","model","horizon_months","sharpe_point","sharpe_ci_lo","sharpe_ci_hi"]].to_string(index=False))

    print("\n=== RANK ICs SIGNIFICANT AT p<0.05 ===")
    sig_ic = out[out["ic_pval"] < 0.05]
    if sig_ic.empty:
        print("  none")
    else:
        print(sig_ic[["sample","model","horizon_months","ic_mean","ic_tstat","ic_pval","ic_n_cohorts"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
