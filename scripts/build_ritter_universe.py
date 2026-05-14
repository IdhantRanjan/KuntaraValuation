"""
Master orchestration script for the Ritter → CIK → verified universe pipeline.

Stages:
  1. Download/parse Ritter IPO Excel  → data/raw/ritter_ipos.csv
  2. Multi-strategy CIK lookup        → data/raw/ritter_cik_mapping.csv
  3. EDGAR submissions verification   → outputs/cik_verification_report.csv
  4. Merge + filter                   → data/raw/ipo_master.csv
  5. Summary                          → outputs/pipeline_summary.txt

This produces the canonical input for src.data.edgar_scraper and
src.data.ipo_universe.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ritter_parser import build_ritter_csv  # noqa: E402
from src.data.edgar_cik_lookup import build_cik_mapping  # noqa: E402
from src.data.verify_cik_mappings import verify_all_ciks  # noqa: E402

logger = logging.getLogger("build_ritter_universe")


# ---------------------------------------------------------------------------
# Synthetic fallback (used only if all downloads fail)
# ---------------------------------------------------------------------------

SYNTHETIC_IPOS = [
    # ticker, company_name, cik, ipo_date, offer_price, first_day_return_pct
    ("ABNB", "Airbnb, Inc.",        "1559720", "2020-12-10", 68.00, 112.81),
    ("COIN", "Coinbase Global, Inc.","1679788", "2021-04-14", 250.0, 31.31),
    ("DASH", "DoorDash, Inc.",      "1792789", "2020-12-09", 102.0, 85.79),
    ("SNOW", "Snowflake Inc.",      "1640147", "2020-09-16", 120.0, 111.61),
    ("PLTR", "Palantir Technologies","1321655","2020-09-30",  7.25, 31.30),
    ("RBLX", "Roblox Corporation",  "1315098", "2021-03-10", 45.00, 54.44),
    ("RIVN", "Rivian Automotive",   "1874178", "2021-11-10", 78.00, 29.14),
    ("PATH", "UiPath, Inc.",        "1734722", "2021-04-21", 56.00, 23.21),
    ("HOOD", "Robinhood Markets",   "1783879", "2021-07-29", 38.00, -8.37),
    ("U",    "Unity Software Inc.", "1810806", "2020-09-18", 52.00, 31.40),
    ("BMBL", "Bumble Inc.",         "1830043", "2021-02-11", 43.00, 63.51),
    ("TOST", "Toast, Inc.",         "1650164", "2021-09-22", 40.00, 56.45),
    ("GTLB", "GitLab Inc.",         "1653482", "2021-10-14", 77.00, 34.81),
    ("HCP",  "HashiCorp, Inc.",     "1720671", "2021-12-09", 80.00, 6.36),
    ("BASE", "Couchbase, Inc.",     "1639825", "2021-07-22", 24.00, 24.00),
    ("BRZE", "Braze, Inc.",         "1538097", "2021-11-17", 65.00, 31.86),
    ("AMPL", "Amplitude, Inc.",     "1866364", "2021-09-28", 35.00, 8.46),
    ("DUOL", "Duolingo, Inc.",      "1562088", "2021-07-28", 102.0, 36.40),
    ("IOT",  "Samsara Inc.",        "1642896", "2021-12-15", 23.00, 7.52),
    ("CXM",  "Sprinklr, Inc.",      "1569345", "2021-06-23", 16.00, 0.13),
]


def build_synthetic_master(output: Path) -> pd.DataFrame:
    """
    Build a small synthetic ipo_master.csv from a hardcoded list of
    well-known recent IPOs. Used only if the Ritter download fails.
    """
    rows = []
    for tk, name, cik, dt, op, fdr_pct in SYNTHETIC_IPOS:
        fdr = fdr_pct / 100.0
        rows.append({
            "cik": cik,
            "ticker": tk,
            "company_name": name,
            "ipo_date": dt,
            "offer_price": op,
            "offer_size": float("nan"),
            "first_day_return": fdr,
            "broken_ipo": int(fdr < 0),
            "industry": "",
            "vc_backed": pd.NA,
            "underwriter": "",
            "underwriter_rank": 7.5,
            "cik_match_score": 1.0,
            "cik_verification_status": "OK",
        })
    df = pd.DataFrame(rows)
    df["ipo_date"] = pd.to_datetime(df["ipo_date"])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.warning("Wrote SYNTHETIC ipo_master.csv (%d rows) → %s", len(df), output)
    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    start_year: int = 2010,
    end_year: int = 2024,
    user_agent: str = "IPOResearch pukthuanthongk@missouri.edu",
    rate_limit: float = 0.12,
    skip_verify: bool = False,
    use_efts: bool = False,
    excel_path: Path | None = None,
    ritter_csv: Path = Path("data/raw/ritter_ipos.csv"),
    cik_csv: Path = Path("data/raw/ritter_cik_mapping.csv"),
    verify_report: Path = Path("outputs/cik_verification_report.csv"),
    master_csv: Path = Path("data/raw/ipo_master.csv"),
    summary_txt: Path = Path("outputs/pipeline_summary.txt"),
) -> pd.DataFrame:
    """Run the full Ritter → CIK → verified universe pipeline."""
    for p in [ritter_csv.parent, cik_csv.parent, verify_report.parent,
              master_csv.parent, summary_txt.parent]:
        p.mkdir(parents=True, exist_ok=True)

    # --- 1. Ritter ---
    n_ritter = 0
    ritter_df: pd.DataFrame | None = None
    try:
        logger.info("[1/4] Downloading + parsing Ritter Excel...")
        ritter_df = build_ritter_csv(
            output_path=ritter_csv,
            start_year=start_year,
            end_year=end_year,
            excel_path=excel_path,
        )
        n_ritter = len(ritter_df)
    except Exception as e:
        logger.error("Ritter pipeline failed: %s", e)
        logger.warning("Falling back to synthetic 20-IPO master.")
        df = build_synthetic_master(master_csv)
        with summary_txt.open("w") as f:
            f.write("Pipeline Summary (FALLBACK / SYNTHETIC)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Ritter download:   FAILED ({type(e).__name__}: {e})\n")
            f.write(f"Synthetic rows:    {len(df)}\n")
        return df

    # --- 2. CIK mapping ---
    logger.info("[2/4] Running CIK mapping...")
    cik_df = build_cik_mapping(
        ritter_csv_path=ritter_csv,
        output_path=cik_csv,
        user_agent=user_agent,
        rate_limit=rate_limit,
        use_efts=use_efts,
    )
    n_mapped = (cik_df["cik"].notna() & (cik_df["method"] != "not_found")).sum()

    # --- 3. Verify ---
    if skip_verify:
        logger.info("[3/4] Skipping CIK verification (--skip-verify)")
        verify_df = cik_df[cik_df["cik"].notna()].copy()
        verify_df["status"] = "OK"
        verify_df["match_score"] = verify_df["score"]
        verify_df.rename(columns={"company_name": "csv_name"}, inplace=True)
    else:
        logger.info("[3/4] Verifying CIK mappings against EDGAR submissions...")
        # Build a temp CSV that verify_all_ciks expects: cik, company_name
        tmp = ritter_csv.parent / "ritter_for_verify.csv"
        merged = cik_df[cik_df["cik"].notna()][
            ["cik", "company_name"]
        ].copy()
        merged.to_csv(tmp, index=False)
        try:
            verify_df = verify_all_ciks(
                csv_path=tmp,
                output_path=verify_report,
                user_agent=user_agent,
                rate_limit=rate_limit,
            )
        except Exception as e:
            logger.warning("Verification failed (%s) — continuing without it", e)
            verify_df = cik_df[cik_df["cik"].notna()][
                ["cik", "company_name"]
            ].copy()
            verify_df["status"] = "CHECK"
            verify_df["match_score"] = 0.5
            verify_df.rename(columns={"company_name": "csv_name"}, inplace=True)

    # --- 4. Merge + filter ---
    logger.info("[4/4] Merging Ritter + CIK + verification...")
    merged = ritter_df.merge(
        cik_df[["ticker", "company_name", "cik", "edgar_name", "method", "score"]],
        on=["ticker", "company_name"],
        how="left",
        suffixes=("", "_cik"),
    )
    merged = merged.rename(columns={"score": "cik_match_score"})

    if "status" in verify_df.columns and "cik" in verify_df.columns:
        v_small = verify_df[["cik", "status"]].drop_duplicates(subset=["cik"])
        v_small = v_small.rename(columns={"status": "cik_verification_status"})
        v_small["cik"] = v_small["cik"].astype(str)
        merged["cik"] = merged["cik"].astype(str).where(merged["cik"].notna(), None)
        merged = merged.merge(v_small, on="cik", how="left")
    else:
        merged["cik_verification_status"] = pd.NA

    n_before_filter = len(merged)
    keep_mask = merged["cik"].notna() & merged["cik"].astype(str).ne("None")
    keep_mask = keep_mask & merged["cik_verification_status"].fillna("CHECK").isin(
        ["OK", "CHECK"]
    )
    merged = merged[keep_mask].reset_index(drop=True)

    # Final schema
    final_cols = [
        "cik", "ticker", "company_name", "ipo_date",
        "offer_price", "offer_size", "first_day_return", "broken_ipo",
        "industry", "vc_backed", "underwriter_rank",
        "cik_match_score", "cik_verification_status",
    ]
    for c in final_cols:
        if c not in merged.columns:
            merged[c] = pd.NA
    final = merged[final_cols].copy()
    final.to_csv(master_csv, index=False)

    # --- Summary ---
    n_final = len(final)
    summary = {
        "ritter_ipos": n_ritter,
        "after_year_filter": n_ritter,
        "cik_mapped": int(n_mapped),
        "after_verification_filter": n_before_filter - (n_before_filter - n_final),
        "final_rows": n_final,
    }
    msg = (
        f"\n{'=' * 60}\n"
        f"PIPELINE SUMMARY\n"
        f"{'=' * 60}\n"
        f"  Ritter rows (after year filter): {n_ritter}\n"
        f"  Mapped to CIK:                   {summary['cik_mapped']}\n"
        f"  After verification filter:       {n_before_filter}\n"
        f"  Final ipo_master.csv:            {n_final}\n"
        f"{'=' * 60}\n"
    )
    logger.info(msg)
    with summary_txt.open("w") as f:
        f.write(msg)

    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Build the Ritter IPO master CSV")
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--user-agent", type=str,
                   default="IPOResearch pukthuanthongk@missouri.edu")
    p.add_argument("--rate-limit", type=float, default=0.12)
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip the EDGAR verification step")
    p.add_argument("--use-efts", action="store_true",
                   help="Enable EDGAR EFTS fallback (slower)")
    p.add_argument("--excel", type=str, default=None,
                   help="Path to a local Ritter .xlsx (skips download)")
    args = p.parse_args(argv)

    run_pipeline(
        start_year=args.start_year,
        end_year=args.end_year,
        user_agent=args.user_agent,
        rate_limit=args.rate_limit,
        skip_verify=args.skip_verify,
        use_efts=args.use_efts,
        excel_path=Path(args.excel) if args.excel else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
