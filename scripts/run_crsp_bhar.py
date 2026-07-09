"""
End-to-end CRSP BHAR pipeline.

For each matched firm, keeps only CRSP daily returns dated STRICTLY AFTER
the offer date and requires the first available return to be within
POST_IPO_MAX_GAP_DAYS of the offer — otherwise the file's coverage
doesn't include this firm's post-IPO window and the firm is dropped.

Usage:
    python scripts/run_crsp_bhar.py --crsp "2024-2025 (1).gz"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bhar import HORIZONS_MONTHS, build_bhar_panel, horizon_sample_summary
from src.data.crsp_loader import (
    firm_returns_long,
    infer_delistings,
    load_crsp_daily,
    market_daily,
    match_universe_to_permno,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

POST_IPO_MAX_GAP_DAYS = 30


def restrict_to_post_ipo(
    firm: pd.DataFrame,
    xwalk: pd.DataFrame,
    max_gap_days: int = POST_IPO_MAX_GAP_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only returns dated strictly after each firm's offer date.
    Firms whose first available return is more than max_gap_days after
    the offer (i.e., the CRSP file starts long after their IPO) are dropped.
    """
    map_df = xwalk[xwalk["matched"]][["permno", "ipo_date"]].copy()
    map_df["permno"] = map_df["permno"].astype(int)
    map_df["ipo_date"] = pd.to_datetime(map_df["ipo_date"])

    firm = firm.copy()
    firm["permno"] = firm["permno"].astype(int)
    firm = firm.merge(map_df, on="permno", how="inner")
    firm = firm[firm["date"] > firm["ipo_date"]].copy()

    first = firm.groupby("permno").agg(first_ret=("date", "min"), ipo_date=("ipo_date", "first")).reset_index()
    first["gap_days"] = (first["first_ret"] - first["ipo_date"]).dt.days
    keep_permnos = first[first["gap_days"] <= max_gap_days]["permno"].tolist()
    dropped = first[first["gap_days"] > max_gap_days]

    firm_kept = firm[firm["permno"].isin(keep_permnos)][["permno", "date", "ret"]].reset_index(drop=True)
    return firm_kept, dropped


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--crsp", default="2024-2025 (1).gz")
    p.add_argument("--out-dir", default=str(ROOT / "outputs/bhar/crsp"))
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("loading CRSP file: %s", args.crsp)
    crsp = load_crsp_daily(args.crsp)
    logger.info(
        "CRSP: %d rows, %d PERMNOs, %d trading days, %s -> %s",
        len(crsp),
        crsp["PERMNO"].nunique(),
        crsp["DlyCalDt"].nunique(),
        crsp["DlyCalDt"].min().date(),
        crsp["DlyCalDt"].max().date(),
    )

    universe = pd.read_parquet(ROOT / "data/processed/ipo_sample/ipo_universe_final.parquet")
    xwalk = match_universe_to_permno(crsp, universe)
    xwalk.to_csv(out / "universe_permno_crosswalk.csv", index=False)
    matched = int(xwalk["matched"].sum())
    total = len(xwalk)
    logger.info("ticker-matched to PERMNO: %d / %d (%.1f%%)", matched, total, matched / total * 100)

    firm = firm_returns_long(crsp)
    market = market_daily(crsp, "vwretd")

    firm_scored, dropped = restrict_to_post_ipo(firm, xwalk, POST_IPO_MAX_GAP_DAYS)
    n_pre_filter = firm.merge(
        xwalk[xwalk["matched"]][["permno"]].assign(permno=lambda d: d["permno"].astype(int)),
        on="permno",
    )["permno"].nunique()
    logger.info(
        "post-IPO filter: kept %d PERMNOs, dropped %d (first CRSP return > %d days after IPO)",
        firm_scored["permno"].nunique(), len(dropped), POST_IPO_MAX_GAP_DAYS,
    )

    market.to_csv(out / "market_vwretd.csv", index=False)

    logger.info("computing BHAR panel (no delisting-events file — that's a separate WRDS pull)")
    panel = build_bhar_panel(firm_scored, market, delist_table=None,
                             horizons=HORIZONS_MONTHS, min_coverage=0.5)
    panel.to_parquet(out / "bhar_panel_crsp.parquet", index=False)

    summary = horizon_sample_summary(panel)
    summary.to_csv(out / "horizon_sample_summary_crsp.csv", index=False)

    print("\n" + "=" * 76)
    print("HONEST CRSP BHAR RESULT (post-IPO windows, gap filter <=30d)")
    print("=" * 76)
    print(f"\nCRSP file coverage:      {crsp['DlyCalDt'].min().date()} -> {crsp['DlyCalDt'].max().date()}")
    print(f"Universe IPOs (dates):   2010-01-21 -> 2024-12-31")
    print(f"Firms ticker-matched to CRSP PERMNO:   {matched} of {total}  ({matched/total*100:.1f}%)")
    print(f"Firms whose IPO was inside CRSP window: {firm_scored['permno'].nunique()}  (rest dropped)")

    print("\nHorizon-level sample sizes (post-IPO, real BHARs vs CRSP VW index):")
    cols = ["horizon_months", "n_total", "n_complete", "n_insufficient", "n_missing",
            "bhar_mean", "bhar_median", "bhar_std"]
    print(summary[cols].to_string(index=False))

    print("\n" + "-" * 76)
    print("Interpretation:")
    print("- The 2010-2023 IPOs (about 1,320 firms) cannot be scored — their post-IPO")
    print("  window is entirely outside the file's 2024-12-31 to 2025-12-31 coverage.")
    print("- The 2024 IPOs whose offer was in the last days of 2024 can be partially")
    print("  scored at short horizons. Almost none reach 12m or 24m.")
    print("- To reproduce the 1,500+ firm run against CRSP VW with delisting handling,")
    print("  the WRDS query needs to be rerun over 2010-01-01 to 2025-12-31 plus the")
    print("  delisting-events file.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
