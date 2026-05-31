"""
Build the identifier crosswalk to hand to K for the CRSP pull.

Produces one row per IPO with the identifiers needed to match in WRDS/CRSP
(ticker, CIK, company name, offer date) plus per-horizon eligibility flags
driven by the data cutoff. Survivorship shrinkage is applied later, once the
delisting codes come back.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (3, 6, 12, 24)


def build(cutoff: str, out_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data/processed/ipo_sample/ipo_universe_final.parquet")
    df["ipo_date"] = pd.to_datetime(df["ipo_date"], errors="coerce")

    df = df[df["ticker"].notna() & df["ipo_date"].notna()].copy()
    df = df.drop_duplicates(subset=["cik"]).sort_values("ipo_date").reset_index(drop=True)

    cut = pd.Timestamp(cutoff)
    cols = {
        "cik": df["cik"].astype("Int64"),
        "ticker": df["ticker"].astype(str).str.upper().str.strip(),
        "company_name": df["company_name"],
        "ipo_date": df["ipo_date"].dt.strftime("%Y-%m-%d"),
        "offer_price": df["offer_price"],
        "industry": df.get("industry"),
        "has_s1_images": df.get("has_images"),
    }
    out = pd.DataFrame(cols)

    for h in HORIZONS:
        out[f"eligible_{h}m"] = (df["ipo_date"] <= cut - pd.DateOffset(months=h)).to_numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"crosswalk: {len(out)} unique firms -> {out_path}")
    print(f"data cutoff: {cutoff}\n")
    print("horizon eligibility (cutoff constraint only, before survivorship):")
    for h in HORIZONS:
        n = int(out[f"eligible_{h}m"].sum())
        print(f"  {h:>2}-month: {n:>4} firms priced >= {h} months before cutoff")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cutoff", default="2025-12-31",
                   help="last date CRSP price data is expected to cover")
    p.add_argument("--out", default=str(ROOT / "outputs/crsp_request/ipo_identifier_crosswalk.csv"))
    args = p.parse_args(argv)
    build(args.cutoff, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
