"""
Rebuild portfolio sorts with the two fixes K asked for.

(1) Tie handling. naive_mean predicts a single value per fold, so pooled
across folds it has only 5 unique values. qcut then partitions firms by
fold-membership rather than predicted firm quality, producing the fake
-0.15 to -0.28 Sharpes K flagged. Any prediction series whose unique
ratio is below UNIQUE_RATIO_FLOOR gets NaN.

(2) Calendar-time formation. Each formation period (quarterly) gets its
own quintile sort, producing a time series of cohort-level long-short
returns. Sharpe is computed on that series and annualized by
sqrt(formations_per_year). This is closer to a tradeable strategy than
the single-cross-section version, which is also kept as a column.

The full sample has 56 quarterly cohorts; the multimodal sample is
smaller so it gets a lower min-firms threshold (configurable per sample).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "bhar"

UNIQUE_RATIO_FLOOR = 0.05
FORMATION_FREQ = "Q"
FORMATIONS_PER_YEAR = {"Q": 4, "M": 12, "Y": 1, "2Q": 2}
MIN_FIRMS_PER_COHORT = {"full": 10, "multimodal": 5}


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


def cross_section_sort(grp: pd.DataFrame) -> dict:
    preds = grp["y_pred"].to_numpy()
    actuals = grp["y_actual"].to_numpy()

    if not _is_signal_meaningful(preds):
        return {
            "q1_mean": np.nan, "q2_mean": np.nan, "q3_mean": np.nan,
            "q4_mean": np.nan, "q5_mean": np.nan,
            "ls_spread": np.nan, "ls_xsec_sharpe": np.nan,
            "n_obs": len(grp), "constant_pred_flag": True,
        }

    q = _quintile_with_jitter(preds)
    valid = ~np.isnan(q)
    if valid.sum() == 0 or len(np.unique(q[valid])) < 5:
        return {
            "q1_mean": np.nan, "q2_mean": np.nan, "q3_mean": np.nan,
            "q4_mean": np.nan, "q5_mean": np.nan,
            "ls_spread": np.nan, "ls_xsec_sharpe": np.nan,
            "n_obs": len(grp), "constant_pred_flag": False,
        }

    qi = q[valid].astype(int)
    a = actuals[valid]
    means = [float(a[qi == i].mean()) if (qi == i).any() else np.nan for i in range(5)]
    spread = means[4] - means[0] if np.isfinite(means[0]) and np.isfinite(means[4]) else np.nan

    q5 = a[qi == 4]
    q1 = a[qi == 0]
    if len(q5) >= 2 and len(q1) >= 2:
        ls = np.concatenate([q5, -q1])
        xsec_sharpe = float(ls.mean() / ls.std(ddof=1)) if ls.std(ddof=1) > 0 else np.nan
    else:
        xsec_sharpe = np.nan

    return {
        "q1_mean": means[0], "q2_mean": means[1], "q3_mean": means[2],
        "q4_mean": means[3], "q5_mean": means[4],
        "ls_spread": spread, "ls_xsec_sharpe": xsec_sharpe,
        "n_obs": len(grp), "constant_pred_flag": False,
    }


def calendar_time_sort(grp: pd.DataFrame, sample: str, freq: str = FORMATION_FREQ) -> dict:
    preds = grp["y_pred"].to_numpy()
    if not _is_signal_meaningful(preds):
        return {
            "cal_n_cohorts": 0, "cal_avg_cohort_size": np.nan,
            "cal_ls_mean": np.nan, "cal_ls_std": np.nan,
            "cal_ls_sharpe_ann": np.nan,
        }

    min_firms = MIN_FIRMS_PER_COHORT.get(sample, 10)
    grp = grp.copy()
    grp["cohort"] = grp["ipo_date"].dt.to_period(freq).astype(str)

    spreads, cohort_sizes = [], []
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
            cohort_sizes.append(len(cohort))

    if len(spreads) < 4:
        return {
            "cal_n_cohorts": len(spreads),
            "cal_avg_cohort_size": float(np.mean(cohort_sizes)) if cohort_sizes else np.nan,
            "cal_ls_mean": np.nan, "cal_ls_std": np.nan,
            "cal_ls_sharpe_ann": np.nan,
        }

    s = np.array(spreads)
    mean, sd = float(s.mean()), float(s.std(ddof=1))
    ann = float(np.sqrt(FORMATIONS_PER_YEAR[freq]))
    sharpe = float((mean / sd) * ann) if sd > 0 else np.nan
    return {
        "cal_n_cohorts": len(spreads),
        "cal_avg_cohort_size": float(np.mean(cohort_sizes)),
        "cal_ls_mean": mean,
        "cal_ls_std": sd,
        "cal_ls_sharpe_ann": sharpe,
    }


def main() -> int:
    fp = pd.read_csv(OUT / "fold_predictions.csv")
    fp["cik"] = fp["cik"].astype(int).astype(str)

    dates = _load_ipo_dates()
    merged = fp.merge(dates, on="cik", how="left")
    missing = merged["ipo_date"].isna().sum()
    if missing:
        logger.warning("%d / %d rows missing ipo_date after merge", missing, len(merged))
    else:
        logger.info("All %d rows have ipo_date", len(merged))

    rows = []
    for (sample, model, horizon), grp in merged.groupby(["sample", "model", "horizon_months"]):
        xs = cross_section_sort(grp)
        cal = calendar_time_sort(grp, sample)
        rows.append({
            "sample": sample, "model": model, "horizon_months": int(horizon),
            **xs, **cal,
        })

    out = pd.DataFrame(rows)
    out_path = OUT / "portfolio_sorts.csv"
    out.to_csv(out_path, index=False)
    logger.info("Saved %s (%d rows)", out_path, len(out))

    print(f"\nFormation: {FORMATION_FREQ}  |  Sharpe annualization: sqrt({FORMATIONS_PER_YEAR[FORMATION_FREQ]})")
    print(f"Min firms per cohort: {MIN_FIRMS_PER_COHORT}")

    print("\n=== NAIVE_MEAN SANITY CHECK (should be all NaN with constant_pred_flag=True) ===")
    naive = out[out["model"] == "naive_mean"][
        ["sample", "horizon_months", "ls_spread", "ls_xsec_sharpe",
         "cal_n_cohorts", "cal_ls_sharpe_ann", "constant_pred_flag"]
    ]
    print(naive.to_string(index=False))

    print("\n=== CALENDAR-TIME SHARPES (real models) ===")
    real = out[out["model"] != "naive_mean"][
        ["sample", "model", "horizon_months", "cal_n_cohorts", "cal_avg_cohort_size",
         "cal_ls_mean", "cal_ls_sharpe_ann"]
    ].sort_values(["sample", "horizon_months", "cal_ls_sharpe_ann"], ascending=[True, True, False])
    print(real.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
