"""
Load CRSP daily-return extract and produce the three inputs
src.data.bhar.build_bhar_panel needs.

The WRDS extract is the joined firm-return + market-index panel:
  PERMNO, DlyCalDt, DlyRet, Ticker, CUSIP9, SICCD, vwretd, ewretd, sprtrn

Delisting returns are not embedded in this extract; they live in the CRSP
delisting-events file (Xn10_dsedelist), which we treat as optional. When
absent we infer "possibly delisted" from a permno whose last observation
is inside the file window but before the file's end date, but we do not
substitute a synthetic delist return.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_crsp_daily(path: Path | str, cache_parquet: bool = True) -> pd.DataFrame:
    """Read the raw .gz extract and cache as parquet for fast reuse."""
    path = Path(path)
    parq = path.with_suffix(".parquet")
    if cache_parquet and parq.exists():
        return pd.read_parquet(parq)

    df = pd.read_csv(path, compression="gzip", low_memory=False)
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.sort_values(["PERMNO", "DlyCalDt"]).reset_index(drop=True)
    if cache_parquet:
        df.to_parquet(parq, index=False)
    return df


def firm_returns_long(crsp: pd.DataFrame) -> pd.DataFrame:
    """[permno, date, ret] for src.data.bhar."""
    out = crsp[["PERMNO", "DlyCalDt", "DlyRet"]].dropna(subset=["DlyRet"]).copy()
    out.columns = ["permno", "date", "ret"]
    return out.sort_values(["permno", "date"]).reset_index(drop=True)


def market_daily(crsp: pd.DataFrame, index_col: str = "vwretd") -> pd.DataFrame:
    """[date, vwretd] time series — one row per trading day."""
    mkt = crsp[["DlyCalDt", index_col]].drop_duplicates(subset=["DlyCalDt"]).copy()
    mkt.columns = ["date", "vwretd"] if index_col == "vwretd" else ["date", index_col]
    return mkt.sort_values("date").reset_index(drop=True)


def infer_delistings(crsp: pd.DataFrame) -> pd.DataFrame:
    """
    Permnos whose last observation is strictly before the file's end date.
    Real delisting handling requires the delisting-events file with
    delisting returns; we return dates only so caller can decide policy.
    """
    end = crsp["DlyCalDt"].max()
    last = crsp.groupby("PERMNO")["DlyCalDt"].max().rename("delist_date")
    likely = last[last < end].reset_index()
    likely["permno"] = likely["PERMNO"]
    likely["delist_ret"] = float("nan")
    return likely[["permno", "delist_date", "delist_ret"]]


def match_universe_to_permno(
    crsp: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match our IPO universe (CIK + ticker) to CRSP PERMNO by ticker.

    Ticker matching is imperfect (some tickers change over time), so we
    also expose SICCD and issuer type from CRSP so a caller can spot-check.
    Returns [cik, ticker, ipo_date, permno, crsp_ticker, siccd, matched].
    """
    uni = universe[["cik", "ticker", "ipo_date"]].copy()
    uni["cik"] = uni["cik"].astype(str)
    uni["ticker_norm"] = uni["ticker"].astype(str).str.upper().str.strip()
    uni["ipo_date"] = pd.to_datetime(uni["ipo_date"])
    uni = uni.drop_duplicates(subset=["ticker_norm"])

    per_permno = (
        crsp.groupby("PERMNO")
        .agg(
            crsp_ticker=("Ticker", "first"),
            siccd=("SICCD", "first"),
            crsp_start=("DlyCalDt", "min"),
            crsp_end=("DlyCalDt", "max"),
        )
        .reset_index()
        .rename(columns={"PERMNO": "permno"})
    )
    per_permno["crsp_ticker"] = per_permno["crsp_ticker"].astype(str).str.upper().str.strip()

    merged = uni.merge(
        per_permno, left_on="ticker_norm", right_on="crsp_ticker", how="left"
    )
    merged["matched"] = merged["permno"].notna()
    return merged[
        ["cik", "ticker", "ipo_date", "permno", "crsp_ticker", "siccd",
         "crsp_start", "crsp_end", "matched"]
    ]
