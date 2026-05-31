"""
Fetch post-IPO daily returns via yfinance and compute BHARs.

Market proxy: SPY (S&P 500 ETF) as a stand-in for the CRSP VW index.
Delisting note: yfinance silently drops tickers after delisting/acquisition,
so failed firms may be underrepresented in the long horizons. When CRSP
delisting codes arrive, pass them to src.data.bhar.build_bhar_panel directly
and skip this module.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

HORIZONS_MONTHS = (3, 6, 12, 24)
MARKET_TICKER = "SPY"


def fetch_market_returns(start: str, end: str) -> pd.DataFrame:
    raw = yf.download(MARKET_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].squeeze()
    ret = prices.pct_change().dropna()
    df = ret.reset_index()
    df.columns = ["date", "vwretd"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.sort_values("date").reset_index(drop=True)


def _download_batch(tickers: list[str], start: str, end: str, retries: int = 2) -> pd.DataFrame:
    for attempt in range(retries + 1):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                return pd.DataFrame()
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]].rename(columns={"Close": tickers[0]}) if len(tickers) == 1 else raw
            return close
        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                logger.warning("download failed: %s", e)
                return pd.DataFrame()


def fetch_firm_returns(
    universe: pd.DataFrame,
    data_dir: Path,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Pull post-offer daily returns for every ticker.

    Returns long-format [ticker, ipo_date, date, ret] starting the day after offer.
    """
    cache_path = data_dir / "firm_daily_returns.parquet"
    if cache_path.exists():
        logger.info("loading cached firm returns from %s", cache_path)
        return pd.read_parquet(cache_path)

    universe = universe[universe["ticker"].notna() & universe["ipo_date"].notna()].copy()
    universe["ipo_date"] = pd.to_datetime(universe["ipo_date"])
    universe = universe.drop_duplicates(subset=["ticker"]).sort_values("ipo_date")

    max_end = (pd.Timestamp.today() + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    market_start = universe["ipo_date"].min().strftime("%Y-%m-%d")

    logger.info("fetching SPY %s -> %s", market_start, max_end)
    market = fetch_market_returns(market_start, max_end)
    crsp_dir = data_dir / "crsp"
    crsp_dir.mkdir(parents=True, exist_ok=True)
    market.to_csv(crsp_dir / "market.csv", index=False)

    tickers = universe["ticker"].str.upper().str.strip().tolist()
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    ipo_map = universe.set_index("ticker")["ipo_date"].to_dict()

    rows = []
    for bi, batch in enumerate(batches):
        if bi % 5 == 0:
            logger.info("batch %d/%d", bi + 1, len(batches))
        close = _download_batch(batch, market_start, max_end)
        if close.empty:
            continue
        close.index = pd.to_datetime(close.index).tz_localize(None)
        for tkr in close.columns:
            ipo_dt = ipo_map.get(tkr)
            if ipo_dt is None:
                continue
            ipo_dt = pd.Timestamp(ipo_dt)
            ts = close[tkr].dropna()
            ts = ts[ts.index > ipo_dt]
            if ts.empty:
                continue
            ret = ts.pct_change().dropna()
            if ret.empty:
                continue
            df = ret.reset_index()
            df.columns = ["date", "ret"]
            df["ticker"] = tkr
            df["ipo_date"] = ipo_dt
            rows.append(df)
        time.sleep(0.3)

    if not rows:
        logger.error("no price data downloaded")
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)[["ticker", "ipo_date", "date", "ret"]]
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out.to_parquet(cache_path, index=False)
    logger.info("saved %d firm-day rows for %d tickers", len(out), out["ticker"].nunique())
    return out


def compute_bhar_yfinance(
    firm_daily: pd.DataFrame,
    market_daily: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS_MONTHS,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """Compute BHAR at each horizon for every firm."""
    trading_days = {h: h * 21 for h in horizons}
    mkt = market_daily.set_index("date")["vwretd"]

    rows = []
    for (tkr, ipo_dt), grp in firm_daily.groupby(["ticker", "ipo_date"]):
        grp = grp.sort_values("date")
        for h in horizons:
            target = trading_days[h]
            window = grp.iloc[:target]
            coverage = len(window) / target if target else 0.0

            if coverage < min_coverage:
                rows.append({
                    "ticker": tkr, "ipo_date": ipo_dt, "horizon_months": h,
                    "bhar": np.nan, "firm_bhr": np.nan, "market_bhr": np.nan,
                    "n_days": len(window), "status": "insufficient",
                })
                continue

            firm_bhr = float(np.prod(1.0 + window["ret"].to_numpy(dtype=float)) - 1.0)
            mkt_rets = mkt.reindex(window["date"]).fillna(0.0).to_numpy()
            market_bhr = float(np.prod(1.0 + mkt_rets) - 1.0)

            rows.append({
                "ticker": tkr, "ipo_date": ipo_dt, "horizon_months": h,
                "bhar": firm_bhr - market_bhr,
                "firm_bhr": firm_bhr,
                "market_bhr": market_bhr,
                "n_days": len(window),
                "status": "complete",
            })

    return pd.DataFrame(rows)
