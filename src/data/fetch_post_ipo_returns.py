"""
Post-IPO Return Downloader.

Downloads daily adjusted-close prices via yfinance, computes log returns,
and derives outcome variables: first-day return, broken IPO indicator,
post-IPO realized volatility (6m, 12m), and post-IPO market beta vs SPY.

Outputs cached CSVs at data/raw/returns/<ticker>.csv with columns
[date, close, log_return].
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# yfinance is imported lazily so that the module can be imported in test
# environments without the dependency.
# ---------------------------------------------------------------------------

def _yfinance():
    try:
        import yfinance as yf  # noqa: WPS433
        return yf
    except ImportError as e:
        raise ImportError(
            "yfinance is required for fetch_post_ipo_returns. "
            "Install with: pip install yfinance"
        ) from e


# ---------------------------------------------------------------------------
# Single-ticker downloader
# ---------------------------------------------------------------------------

def download_ticker_returns(
    ticker: str,
    ipo_date: str,
    end_date: str = "2025-01-01",
    output_dir: Path = Path("data/raw/returns"),
    overwrite: bool = False,
    retries: int = 3,
) -> pd.DataFrame | None:
    """
    Download daily prices for *ticker* from *ipo_date* to *end_date*.

    Returns a DataFrame with columns [date, close, log_return], or None
    on failure. Cached to {output_dir}/{ticker}.csv.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ticker}.csv"

    if out_path.exists() and not overwrite:
        try:
            cached = pd.read_csv(out_path, parse_dates=["date"])
            if not cached.empty:
                return cached
        except Exception:
            pass

    yf = _yfinance()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                start=ipo_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            break
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    else:
        logger.warning("yfinance download failed for %s: %s", ticker, last_err)
        return None

    if df is None or df.empty:
        logger.warning("No price data for %s in [%s, %s]",
                       ticker, ipo_date, end_date)
        return None

    # Yahoo sometimes returns a MultiIndex with the ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if "Close" not in df.columns:
        logger.warning("No Close column in yfinance output for %s", ticker)
        return None

    out = pd.DataFrame({
        "date": df.index,
        "close": df["Close"].astype(float).values,
    })
    out["log_return"] = np.log(out["close"]).diff()
    out.to_csv(out_path, index=False)
    return out


# ---------------------------------------------------------------------------
# SPY reference returns
# ---------------------------------------------------------------------------

def _ensure_spy_returns(
    start: str,
    end: str = "2025-01-01",
    output_dir: Path = Path("data/raw/returns"),
) -> pd.DataFrame | None:
    """Download SPY once and cache to {output_dir}/SPY.csv."""
    return download_ticker_returns("SPY", start, end, output_dir)


def _ols_beta(y: np.ndarray, x: np.ndarray) -> float | None:
    """Plain OLS beta of y on x (excluding intercept)."""
    if y.size < 30 or x.size < 30 or y.size != x.size:
        return None
    x = x - x.mean()
    y = y - y.mean()
    var_x = float(np.dot(x, x))
    if var_x <= 0:
        return None
    return float(np.dot(x, y) / var_x)


# ---------------------------------------------------------------------------
# Outcome computation
# ---------------------------------------------------------------------------

def compute_outcomes(
    ticker: str,
    ipo_date: str,
    offer_price: float,
    returns_dir: Path = Path("data/raw/returns"),
) -> dict:
    """
    Compute outcome variables for one ticker.

    Returns dict with keys: first_day_return, broken_ipo,
    post_ipo_volatility_6m, post_ipo_volatility_12m, post_ipo_beta_6m.
    """
    returns_dir = Path(returns_dir)
    out = {
        "first_day_return": np.nan,
        "broken_ipo": np.nan,
        "post_ipo_volatility_6m": np.nan,
        "post_ipo_volatility_12m": np.nan,
        "post_ipo_beta_6m": np.nan,
    }

    csv = returns_dir / f"{ticker}.csv"
    if not csv.exists():
        return out

    try:
        df = pd.read_csv(csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    except Exception as e:
        logger.warning("Failed to read %s: %s", csv, e)
        return out

    if df.empty or "close" not in df.columns:
        return out

    if offer_price and not pd.isna(offer_price) and offer_price > 0:
        first_close = float(df["close"].iloc[0])
        out["first_day_return"] = (first_close - float(offer_price)) / float(offer_price)
        out["broken_ipo"] = int(out["first_day_return"] < 0)

    # Volatility (annualized) over first N trading days
    if "log_return" in df.columns:
        rets = df["log_return"].dropna().reset_index(drop=True)
        if len(rets) >= 60:  # 6m proxy
            n6 = min(126, len(rets))
            out["post_ipo_volatility_6m"] = float(rets.iloc[:n6].std() * np.sqrt(252))
        if len(rets) >= 120:
            n12 = min(252, len(rets))
            out["post_ipo_volatility_12m"] = float(rets.iloc[:n12].std() * np.sqrt(252))

        # Beta vs SPY
        spy = _ensure_spy_returns(ipo_date, output_dir=returns_dir)
        if spy is not None and "log_return" in spy.columns:
            spy = spy.copy().sort_values("date").reset_index(drop=True)
            merged = df[["date", "log_return"]].merge(
                spy[["date", "log_return"]],
                on="date", how="inner", suffixes=("_x", "_spy"),
            )
            merged = merged.dropna()
            if len(merged) >= 60:
                n6 = min(126, len(merged))
                beta = _ols_beta(
                    merged["log_return_x"].iloc[:n6].to_numpy(),
                    merged["log_return_spy"].iloc[:n6].to_numpy(),
                )
                if beta is not None:
                    out["post_ipo_beta_6m"] = beta

    return out


# ---------------------------------------------------------------------------
# Universe enrichment
# ---------------------------------------------------------------------------

def enrich_universe_with_returns(
    universe_csv: Path,
    output_path: Path = Path(
        "data/processed/ipo_sample/ipo_universe_with_returns.csv"
    ),
    returns_dir: Path = Path("data/raw/returns"),
    end_date: str = "2025-01-01",
    sleep_between: float = 0.5,
) -> pd.DataFrame:
    """Download returns for every ticker in the universe and fill outcomes."""
    universe_csv = Path(universe_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    returns_dir = Path(returns_dir)
    returns_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(universe_csv, parse_dates=["ipo_date"])
    logger.info("Enriching %d rows with post-IPO returns", len(df))

    for col in [
        "first_day_return", "broken_ipo",
        "post_ipo_volatility_6m", "post_ipo_volatility_12m",
        "post_ipo_beta_6m",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    n_ok = 0
    for idx, row in df.iterrows():
        ticker = str(row.get("ticker", "") or "").strip().upper()
        if not ticker:
            continue
        ipo_date = pd.Timestamp(row["ipo_date"]).strftime("%Y-%m-%d")

        # 1) Download
        try:
            download_ticker_returns(
                ticker, ipo_date, end_date=end_date, output_dir=returns_dir,
            )
        except Exception as e:
            logger.warning("Download failed for %s: %s", ticker, e)
            continue

        # 2) Compute outcomes
        try:
            outcomes = compute_outcomes(
                ticker, ipo_date,
                float(row.get("offer_price", float("nan"))) if not pd.isna(row.get("offer_price")) else float("nan"),
                returns_dir=returns_dir,
            )
        except Exception as e:
            logger.warning("Outcome compute failed for %s: %s", ticker, e)
            outcomes = {}

        # Fill in any non-NaN outcomes (don't overwrite existing values)
        had_data = False
        for k, v in outcomes.items():
            if pd.isna(v):
                continue
            had_data = True
            if pd.isna(df.at[idx, k]):
                df.at[idx, k] = v
        if had_data:
            n_ok += 1

        time.sleep(sleep_between)
        if (idx + 1) % 25 == 0:
            logger.info("  ... %d / %d processed (%d filled)", idx + 1, len(df), n_ok)

    df.to_csv(output_path, index=False)
    logger.info("Saved → %s (%d / %d had outcome data)", output_path, n_ok, len(df))
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Download post-IPO returns")
    p.add_argument("--universe-csv", type=str, default="data/raw/ipo_master.csv")
    p.add_argument(
        "--output", type=str,
        default="data/processed/ipo_sample/ipo_universe_with_returns.csv",
    )
    p.add_argument("--returns-dir", type=str, default="data/raw/returns")
    p.add_argument("--end-date", type=str, default="2025-01-01")
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--sleep", type=float, default=0.5)
    args = p.parse_args(argv)

    df = pd.read_csv(args.universe_csv, parse_dates=["ipo_date"])
    yrs = df["ipo_date"].dt.year
    df = df[(yrs >= args.start_year) & (yrs <= args.end_year)]
    tmp = Path(args.universe_csv).parent / "_universe_year_filtered.csv"
    df.to_csv(tmp, index=False)

    enrich_universe_with_returns(
        universe_csv=tmp,
        output_path=Path(args.output),
        returns_dir=Path(args.returns_dir),
        end_date=args.end_date,
        sleep_between=args.sleep,
    )
    tmp.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
