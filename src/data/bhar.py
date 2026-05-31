"""
Long-run post-IPO abnormal return construction.

Buy-and-hold abnormal returns (BHAR) measured against the CRSP value-weighted
market index at 3, 6, 12, and 24 month horizons, with explicit delisting
handling so that failed firms do not silently leave the sample.

The actual price/return inputs come from CRSP (supplied separately). This
module is the pure-computation layer: feed it daily firm returns, daily market
returns, and delisting records, and it returns the horizon panel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_MONTH = 21
HORIZONS_MONTHS = (3, 6, 12, 24)


@dataclass
class HorizonResult:
    permno: int
    horizon_months: int
    bhar: float | None
    firm_bhr: float | None
    market_bhr: float | None
    n_days: int
    status: str  # complete | delisted | insufficient | missing


def _window_days(horizon_months: int) -> int:
    return horizon_months * TRADING_DAYS_PER_MONTH


def buy_and_hold(returns: np.ndarray) -> float:
    """Compound a sequence of simple daily returns into a holding-period return."""
    return float(np.prod(1.0 + returns) - 1.0)


def compute_firm_bhar(
    firm_daily: pd.DataFrame,
    market_daily: pd.DataFrame,
    delist: pd.Series | None,
    horizon_months: int,
    min_coverage: float = 0.5,
) -> HorizonResult:
    """
    BHAR for one firm at one horizon.

    firm_daily   : columns [date, ret], sorted, starting the first trading day
                   after the offer (we exclude the first-day pop so long-run
                   performance is not mechanically tied to the underpricing
                   outcome we already study).
    market_daily : columns [date, mktrf or vwretd] aligned by date.
    delist       : optional one-row series with [delist_date, delist_ret].

    Delisting handling: if the firm delists inside the window, the holding
    period ends at the delisting date and the CRSP delisting return is
    appended as the final day. The firm stays in the sample with its realized
    (often negative) return rather than being dropped.
    """
    permno = int(firm_daily["permno"].iloc[0]) if "permno" in firm_daily and len(firm_daily) else -1
    target_days = _window_days(horizon_months)

    if firm_daily.empty:
        return HorizonResult(permno, horizon_months, None, None, None, 0, "missing")

    fd = firm_daily.sort_values("date").reset_index(drop=True)
    window = fd.iloc[:target_days].copy()

    delisted_in_window = False
    if delist is not None and pd.notna(delist.get("delist_date")):
        dl_date = pd.Timestamp(delist["delist_date"])
        in_win = window["date"] <= dl_date
        if in_win.any() and dl_date <= window["date"].max():
            window = window[window["date"] <= dl_date].copy()
            dl_ret = delist.get("delist_ret")
            if pd.notna(dl_ret):
                window = pd.concat(
                    [window, pd.DataFrame([{"date": dl_date, "ret": float(dl_ret), "permno": permno}])],
                    ignore_index=True,
                )
            delisted_in_window = True

    coverage = len(window) / target_days if target_days else 0.0
    if not delisted_in_window and coverage < min_coverage:
        return HorizonResult(permno, horizon_months, None, None, None, len(window), "insufficient")

    mkt = market_daily[market_daily["date"].isin(window["date"])].sort_values("date")
    mkt_col = "vwretd" if "vwretd" in mkt.columns else "mktret"
    firm_bhr = buy_and_hold(window["ret"].to_numpy(dtype=float))
    market_bhr = buy_and_hold(mkt[mkt_col].to_numpy(dtype=float))
    bhar = firm_bhr - market_bhr

    return HorizonResult(
        permno=permno,
        horizon_months=horizon_months,
        bhar=bhar,
        firm_bhr=firm_bhr,
        market_bhr=market_bhr,
        n_days=len(window),
        status="delisted" if delisted_in_window else "complete",
    )


def build_bhar_panel(
    returns_long: pd.DataFrame,
    market_daily: pd.DataFrame,
    delist_table: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = HORIZONS_MONTHS,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """
    Build the firm x horizon BHAR panel.

    returns_long : [permno, date, ret] daily returns, post-offer, all firms.
    market_daily : [date, vwretd] CRSP value-weighted index daily returns.
    delist_table : optional [permno, delist_date, delist_ret].

    Returns one row per (permno, horizon) with bhar and a status flag.
    """
    if delist_table is not None:
        delist_idx = delist_table.set_index("permno")
    else:
        delist_idx = None

    rows: list[HorizonResult] = []
    for permno, firm in returns_long.groupby("permno"):
        firm = firm.assign(permno=permno)
        delist_row = None
        if delist_idx is not None and permno in delist_idx.index:
            delist_row = delist_idx.loc[permno]
        for h in horizons:
            rows.append(
                compute_firm_bhar(firm, market_daily, delist_row, h, min_coverage=min_coverage)
            )

    panel = pd.DataFrame([r.__dict__ for r in rows])
    return panel


def horizon_sample_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Count usable / delisted / dropped firms at each horizon, with BHAR moments."""
    out = []
    for h, grp in panel.groupby("horizon_months"):
        usable = grp[grp["status"].isin(["complete", "delisted"])]
        bhar = usable["bhar"].dropna()
        out.append({
            "horizon_months": h,
            "n_total": len(grp),
            "n_usable": len(usable),
            "n_complete": int((grp["status"] == "complete").sum()),
            "n_delisted": int((grp["status"] == "delisted").sum()),
            "n_insufficient": int((grp["status"] == "insufficient").sum()),
            "n_missing": int((grp["status"] == "missing").sum()),
            "bhar_mean": float(bhar.mean()) if len(bhar) else np.nan,
            "bhar_median": float(bhar.median()) if len(bhar) else np.nan,
            "bhar_std": float(bhar.std()) if len(bhar) else np.nan,
            "bhar_skew": float(bhar.skew()) if len(bhar) > 2 else np.nan,
            "pct_underperform": float((bhar < 0).mean()) if len(bhar) else np.nan,
        })
    return pd.DataFrame(out).sort_values("horizon_months").reset_index(drop=True)
