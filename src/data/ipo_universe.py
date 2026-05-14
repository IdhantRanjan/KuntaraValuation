"""
IPO Universe Builder — Sample construction, outcome labels, and control variables.

Downloads and constructs the universe of U.S. IPOs (2010–2024) with:
  - First-day return (primary target)
  - Broken IPO indicator
  - Post-IPO volatility and beta

Uses Yahoo Finance as a free data source. For production-quality research, swap
in CRSP/SDC data via WRDS or Bloomberg.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IPORecord:
    """Single IPO observation with labels and controls."""

    cik: str
    ticker: str
    company_name: str
    ipo_date: str
    offer_price: float
    first_day_close: float | None = None
    first_day_return: float | None = None
    broken_ipo: int | None = None
    post_ipo_volatility_6m: float | None = None
    post_ipo_volatility_12m: float | None = None
    post_ipo_beta_6m: float | None = None
    # Controls
    offer_size: float | None = None
    firm_age: float | None = None
    underwriter_rank: float | None = None
    vc_backed: int | None = None
    industry: str | None = None
    log_assets: float | None = None
    leverage: float | None = None
    rnd_intensity: float | None = None
    revenue_growth: float | None = None


@dataclass
class IPOUniverse:
    """Collection of IPO records with utilities."""

    records: list[IPORecord] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to a pandas DataFrame."""
        import dataclasses
        return pd.DataFrame([dataclasses.asdict(r) for r in self.records])

    def __len__(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def load_ipo_list(
    csv_path: str | Path | None = None,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Load the IPO master list.

    If *csv_path* is provided, reads from a pre-built CSV with columns:
        cik, ticker, company_name, ipo_date, offer_price, offer_size,
        industry, vc_backed, underwriter_rank, firm_age, ...

    Otherwise returns an empty DataFrame with the expected schema so the
    pipeline can be tested without external data.
    """
    expected_cols = [
        "cik", "ticker", "company_name", "ipo_date", "offer_price",
        "offer_size", "industry", "vc_backed", "underwriter_rank",
        "firm_age", "log_assets", "leverage", "rnd_intensity", "revenue_growth",
    ]

    if csv_path is not None and Path(csv_path).exists():
        df = pd.read_csv(csv_path, parse_dates=["ipo_date"])
        logger.info("Loaded %d IPOs from %s", len(df), csv_path)
    else:
        logger.warning(
            "No IPO list CSV found at %s — returning empty scaffold. "
            "Populate data/raw/ipo_master.csv with CRSP/SDC data.",
            csv_path,
        )
        df = pd.DataFrame(columns=expected_cols)
        df["ipo_date"] = pd.to_datetime(df["ipo_date"])
        return df

    # Filter date range
    df = df[
        (df["ipo_date"] >= pd.Timestamp(start_date))
        & (df["ipo_date"] <= pd.Timestamp(end_date))
    ].copy()

    logger.info("After date filter [%s, %s]: %d IPOs", start_date, end_date, len(df))
    return df


def apply_sample_filters(
    df: pd.DataFrame,
    min_offer_size: float = 10_000_000,
    exclude_reits: bool = True,
    exclude_spacs: bool = True,
    exclude_adrs: bool = True,
) -> pd.DataFrame:
    """Apply standard IPO-research sample filters."""
    n_start = len(df)

    if "offer_size" in df.columns and min_offer_size > 0:
        # Keep rows where offer_size is missing OR meets the threshold;
        # downstream stages (e.g. fetch_post_ipo_returns) will fill missing values.
        offer = pd.to_numeric(df["offer_size"], errors="coerce")
        df = df[offer.isna() | (offer >= min_offer_size)]

    if exclude_reits and "industry" in df.columns:
        df = df[~df["industry"].astype(str).str.contains("REIT", case=False, na=False)]

    if exclude_spacs and "company_name" in df.columns:
        spac_mask = df["company_name"].astype(str).str.contains(
            r"SPAC|Blank Check|Acquisition Corp", case=False, na=False
        )
        df = df[~spac_mask]

    if exclude_adrs and "company_name" in df.columns:
        df = df[~df["company_name"].astype(str).str.contains("ADR|ADS", case=False, na=False)]

    logger.info("Sample filters: %d → %d IPOs", n_start, len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Outcome construction
# ---------------------------------------------------------------------------

def compute_first_day_return(df: pd.DataFrame) -> pd.DataFrame:
    """Compute first-day return = (close_1 - offer_price) / offer_price."""
    if "first_day_close" in df.columns and "offer_price" in df.columns:
        df["first_day_return"] = (
            (df["first_day_close"] - df["offer_price"]) / df["offer_price"]
        )
        df["broken_ipo"] = (df["first_day_close"] < df["offer_price"]).astype(int)
        logger.info(
            "Mean first-day return: %.2f%%, Broken IPO rate: %.1f%%",
            df["first_day_return"].mean() * 100,
            df["broken_ipo"].mean() * 100,
        )
    else:
        logger.warning("Missing first_day_close or offer_price — skipping return calc.")
    return df


def compute_post_ipo_volatility(
    df: pd.DataFrame,
    returns_dir: str | Path | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute post-IPO realized volatility over *windows* (in trading days).

    Expects daily return files at *returns_dir*/<ticker>.csv with columns
    [date, return]. If unavailable, columns are left as NaN.
    """
    if windows is None:
        windows = [126, 252]  # ~6 months, ~12 months

    if returns_dir is None or not Path(returns_dir).exists():
        for w in windows:
            months = w // 21
            df[f"post_ipo_volatility_{months}m"] = np.nan
        logger.warning("No returns directory — volatility columns set to NaN.")
        return df

    returns_dir = Path(returns_dir)
    for idx, row in df.iterrows():
        ticker = row.get("ticker")
        if ticker is None:
            continue
        ret_file = returns_dir / f"{ticker}.csv"
        if not ret_file.exists():
            continue
        rets = pd.read_csv(ret_file, parse_dates=["date"])
        rets = rets.sort_values("date").reset_index(drop=True)
        for w in windows:
            months = w // 21
            col = f"post_ipo_volatility_{months}m"
            if len(rets) >= w:
                df.at[idx, col] = rets["return"].iloc[:w].std() * np.sqrt(252)
            else:
                df.at[idx, col] = np.nan

    return df


# ---------------------------------------------------------------------------
# Time-based splitting
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
    train_end: str = "2018-12-31",
    val_start: str = "2019-01-01",
    val_end: str = "2020-12-31",
    test_start: str = "2021-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split IPO sample by listing date to avoid look-ahead bias."""
    train = df[df["ipo_date"] <= pd.Timestamp(train_end)].copy()
    val = df[
        (df["ipo_date"] >= pd.Timestamp(val_start))
        & (df["ipo_date"] <= pd.Timestamp(val_end))
    ].copy()
    test = df[df["ipo_date"] >= pd.Timestamp(test_start)].copy()
    logger.info(
        "Time split → train: %d | val: %d | test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Build full universe
# ---------------------------------------------------------------------------

def build_universe(
    csv_path: str | Path | None = None,
    output_dir: str | Path = "data/processed/ipo_sample",
    **filter_kwargs,
) -> pd.DataFrame:
    """End-to-end: load → filter → compute outcomes → save."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_ipo_list(csv_path)
    df = apply_sample_filters(df, **filter_kwargs)
    df = compute_first_day_return(df)
    df = compute_post_ipo_volatility(df)

    out_path = output_dir / "ipo_universe.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved IPO universe (%d rows) → %s", len(df), out_path)
    return df


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    """CLI entry for building the IPO universe."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Build the IPO universe sample")
    parser.add_argument("--csv", type=str, default="data/raw/ipo_master.csv")
    parser.add_argument("--output-dir", type=str, default="data/processed/ipo_sample")
    parser.add_argument("--min-offer-size", type=float, default=10_000_000)
    args = parser.parse_args()

    build_universe(
        csv_path=args.csv,
        output_dir=args.output_dir,
        min_offer_size=args.min_offer_size,
    )


if __name__ == "__main__":
    main()
