"""
S-1 Offer Price Extractor.

Parses the cover page of S-1 / S-1/A filings to extract the IPO offer
price (or price range midpoint).  This is the single most critical data
extraction task because first-day underpricing — our primary dependent
variable — is defined as:

    first_day_return = (day1_close - offer_price) / offer_price

Strategy:
  1. For each CIK, find all S-1 HTML files on disk.
  2. Sort by filing date (newest first) — the last S-1/A before the IPO
     is most likely to have the final offer price.
  3. Scan the first ~20 KB of plain text (the cover page) for price
     patterns:
       a. "$XX.XX per share" (most common)
       b. "Price to Public $XX.XX"
       c. "Initial public offering price of $XX.XX"
       d. Price range: "between $XX and $YY per share" → midpoint
  4. Take the price from the most recent filing that has one.
  5. Cross-reference with yfinance first-day close to compute the return.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Suppress XML-as-HTML warnings (some S-1s have XML preambles)
try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Price extraction patterns (ordered by specificity)
# ---------------------------------------------------------------------------

# Pattern 1: "Initial public offering price ... $XX.XX per share"
_PAT_IPO_PRICE = re.compile(
    r"(?:initial\s+public\s+offering\s+price|offering\s+price|"
    r"price\s+to\s+(?:the\s+)?public)"
    r"[\s:]*\$\s*(\d+(?:,\d{3})*\.?\d*)\s*per\s+(?:share|unit)",
    re.IGNORECASE,
)

# Pattern 2: Simple "$XX.XX per share" (very common on cover pages)
_PAT_DOLLAR_PER_SHARE = re.compile(
    r"\$\s*(\d+(?:,\d{3})*\.?\d*)\s+per\s+(?:share|unit)",
    re.IGNORECASE,
)

# Pattern 3: Price range "between $XX.XX and $YY.YY [per share]"
# Note: "per share" sometimes comes BEFORE the range, not after.
_PAT_PRICE_RANGE = re.compile(
    r"between\s+\$\s*(\d+(?:,\d{3})*\.?\d*)\s+and\s+\$\s*(\d+(?:,\d{3})*\.?\d*)",
    re.IGNORECASE,
)

# Pattern 4: "Price to Public" in a table row (without "per share")
_PAT_PRICE_TABLE = re.compile(
    r"Price\s+to\s+(?:the\s+)?Public\s+\$?\s*(\d+(?:,\d{3})*\.?\d*)",
    re.IGNORECASE,
)

# Pattern 5: Registration table row "$XX.XX" after "Per Share"
_PAT_REG_TABLE = re.compile(
    r"(?:Proposed\s+Maximum\s+)?(?:Aggregate\s+)?Offering\s+Price\s+Per\s+Share"
    r"[^$]*\$\s*(\d+(?:,\d{3})*\.?\d*)",
    re.IGNORECASE,
)


def _parse_price(s: str | None) -> float | None:
    """Convert a captured price string to float."""
    if s is None:
        return None
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError, AttributeError):
        return None


def extract_offer_price_from_html(html_path: Path) -> dict:
    """
    Extract the offer price from a single S-1 HTML filing.

    Returns dict with keys:
      - offer_price: float or None
      - price_low: float or None (if range found)
      - price_high: float or None (if range found)
      - method: str describing how price was found
      - source_file: str
    """
    result = {
        "offer_price": None,
        "price_low": None,
        "price_high": None,
        "method": "not_found",
        "source_file": str(html_path),
    }

    try:
        with html_path.open("r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
    except OSError as e:
        logger.debug("Cannot read %s: %s", html_path, e)
        return result

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    # Only inspect the cover page area (first ~20K characters).
    # The offer price is always on the front page of the prospectus.
    cover = text[:20_000]

    # --- Strategy 1: Specific "offering price" label ---
    m = _PAT_IPO_PRICE.search(cover)
    if m:
        price = _parse_price(m.group(1))
        if price and 0.5 < price < 500:
            result["offer_price"] = price
            result["method"] = "ipo_price_label"
            return result

    # --- Strategy 2: Price range (use midpoint) ---
    m = _PAT_PRICE_RANGE.search(cover)
    if m:
        low = _parse_price(m.group(1))
        high = _parse_price(m.group(2))
        if low and high and 0.5 < low < 500 and 0.5 < high < 500:
            result["price_low"] = low
            result["price_high"] = high
            result["offer_price"] = (low + high) / 2.0
            result["method"] = "price_range_midpoint"
            return result

    # --- Strategy 3: Generic "$XX per share" ---
    # Take the first occurrence that looks like a reasonable IPO price
    matches = _PAT_DOLLAR_PER_SHARE.findall(cover)
    for match_str in matches:
        price = _parse_price(match_str)
        if price and 1.0 < price < 500:
            result["offer_price"] = price
            result["method"] = "dollar_per_share"
            return result

    # --- Strategy 4: "Price to Public" table cell ---
    m = _PAT_PRICE_TABLE.search(cover)
    if m:
        price = _parse_price(m.group(1))
        if price and 1.0 < price < 500:
            result["offer_price"] = price
            result["method"] = "price_to_public_table"
            return result

    # --- Strategy 5: Registration table "Offering Price Per Share" ---
    m = _PAT_REG_TABLE.search(cover)
    if m:
        price = _parse_price(m.group(1))
        if price and 1.0 < price < 500:
            result["offer_price"] = price
            result["method"] = "reg_table_per_share"
            return result

    return result


# ---------------------------------------------------------------------------
# Per-CIK pipeline
# ---------------------------------------------------------------------------

def _find_s1_htmls(cik: str, edgar_dir: Path) -> list[Path]:
    """
    Find all S-1 HTML files for a CIK, sorted newest-first by filename
    (accession number encodes filing date).
    """
    cik_str = str(cik).lstrip("0") or str(cik)
    cik_padded = str(cik).zfill(10)

    candidates: list[Path] = []
    for variant in (cik_str, cik_padded, str(cik)):
        d = edgar_dir / variant
        if d.exists():
            for p in d.glob("*.htm*"):
                # Skip image files and non-filing artifacts
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".gif"):
                    continue
                candidates.append(p)

    # Sort by accession number (descending) — most recent first
    candidates.sort(key=lambda p: p.stem, reverse=True)
    return candidates


def extract_offer_price_for_cik(
    cik: str,
    edgar_dir: Path,
) -> dict:
    """
    Extract the offer price for a CIK by trying each S-1 file
    (newest first) until one yields a price.
    """
    htmls = _find_s1_htmls(cik, edgar_dir)

    for html_path in htmls:
        result = extract_offer_price_from_html(html_path)
        if result["offer_price"] is not None:
            return result

    return {
        "offer_price": None,
        "price_low": None,
        "price_high": None,
        "method": "not_found",
        "source_file": "",
    }


# ---------------------------------------------------------------------------
# 424B4 Final Prospectus Downloader
# ---------------------------------------------------------------------------
# The final offer price is set during the roadshow and appears in the 424B4
# filing (final prospectus), filed on the pricing date — 1 day before trading
# starts.  This is NOT a post-IPO filing, so there's no look-ahead bias.

EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"

# 424B filing types that contain the final offer price
PROSPECTUS_TYPES = frozenset({"424B4", "424B3", "424B1"})


def _download_424b_for_cik(
    cik: str,
    edgar_dir: Path,
    session: requests.Session | None = None,
    rate_limit: float = 0.15,
) -> Path | None:
    """
    Download the 424B4 (final prospectus) for a CIK if not already on disk.

    Returns the path to the downloaded HTML, or None if not found.
    """
    cik_str = str(cik).lstrip("0") or str(cik)
    out_dir = edgar_dir / cik_str

    # Check if we already have a 424B file
    if out_dir.exists():
        existing = [p for p in out_dir.glob("*_424B*.html")]
        if existing:
            return existing[0]

    if session is None:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "IPOValuationResearch pukthuanthongk@missouri.edu",
            "Accept-Encoding": "gzip, deflate",
        })

    cik_padded = cik_str.zfill(10)
    url = f"{EDGAR_SUBMISSIONS}/CIK{cik_padded}.json"

    time.sleep(rate_limit)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.debug("Cannot fetch submissions for CIK %s: %s", cik, e)
        return None

    # Search for 424B filings in recent filings
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for form, acc, doc in zip(forms, accessions, primary_docs):
        if form not in PROSPECTUS_TYPES:
            continue

        acc_formatted = acc.replace("-", "")
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik_str)}/{acc_formatted}/{doc}"
        )

        out_path = out_dir / f"{acc_formatted}_424B.html"
        if out_path.exists():
            return out_path

        out_path.parent.mkdir(parents=True, exist_ok=True)
        time.sleep(rate_limit)
        try:
            resp = session.get(filing_url, timeout=60)
            resp.raise_for_status()
            out_path.write_text(resp.text, encoding="utf-8")
            logger.debug("Downloaded 424B → %s", out_path)
            return out_path
        except requests.RequestException as e:
            logger.debug("Failed to download 424B for CIK %s: %s", cik, e)
            continue

    return None


def extract_final_offer_price(html_path: Path) -> dict:
    """
    Extract the final offer price from a 424B4 filing.

    The 424B4 cover page always states the exact offer price as
    "$XX.XX per share" — no range ambiguity.
    """
    result = {
        "offer_price": None,
        "method": "not_found",
        "source_file": str(html_path),
    }

    try:
        with html_path.open("r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
    except OSError:
        return result

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    cover = text[:20_000]

    # The 424B4 always has a definitive price. Try the most specific
    # patterns first.
    m = _PAT_IPO_PRICE.search(cover)
    if m:
        price = _parse_price(m.group(1))
        if price and 0.5 < price < 500:
            result["offer_price"] = price
            result["method"] = "424b_ipo_price"
            return result

    # Fall back to generic "$XX per share"
    matches = _PAT_DOLLAR_PER_SHARE.findall(cover)
    for match_str in matches:
        price = _parse_price(match_str)
        if price and 1.0 < price < 500:
            result["offer_price"] = price
            result["method"] = "424b_dollar_per_share"
            return result

    return result


# ---------------------------------------------------------------------------
# Combined price extraction: 424B4 preferred, S-1 fallback
# ---------------------------------------------------------------------------

def extract_best_offer_price(
    cik: str,
    edgar_dir: Path,
    session: requests.Session | None = None,
    download_424b: bool = True,
    rate_limit: float = 0.15,
) -> dict:
    """
    Get the best available offer price for a CIK:
      1. Try 424B4 (final prospectus) — exact offer price
      2. Fall back to S-1 cover page — range midpoint
    """
    # Try 424B4 first
    if download_424b:
        b4_path = _download_424b_for_cik(
            cik, edgar_dir, session=session, rate_limit=rate_limit,
        )
        if b4_path is not None:
            result = extract_final_offer_price(b4_path)
            if result["offer_price"] is not None:
                return result

    # Fall back to S-1
    return extract_offer_price_for_cik(cik, edgar_dir)


# ---------------------------------------------------------------------------
# First-day return computation
# ---------------------------------------------------------------------------

def _get_first_day_close(
    ticker: str,
    ipo_date: str | pd.Timestamp,
    returns_dir: Path,
) -> float | None:
    """
    Get the first trading day's closing price from the saved yfinance data.

    Includes sanity checks: reject prices that are clearly wrong
    (< $0.01 or > $10,000 per share — no US IPO has ever had a price
    outside this range).
    """
    ret_file = returns_dir / f"{ticker}.csv"
    if not ret_file.exists():
        return None

    try:
        df = pd.read_csv(ret_file, parse_dates=["date"])
    except Exception:
        return None

    if df.empty or "close" not in df.columns:
        return None

    ipo_ts = pd.Timestamp(ipo_date)

    # Find the first trading day on or after the IPO date
    df = df.sort_values("date").reset_index(drop=True)
    mask = df["date"] >= ipo_ts
    if not mask.any():
        return None

    close = float(df.loc[mask.idxmax(), "close"])

    # Sanity check: reject clearly wrong prices
    # (market cap, volume, or corrupted data from yfinance)
    if close < 0.01 or close > 10_000:
        logger.debug(
            "Rejected close price %.2f for %s on %s (likely corrupted)",
            close, ticker, ipo_date,
        )
        return None

    return close


# ---------------------------------------------------------------------------
# Universe enrichment
# ---------------------------------------------------------------------------

def enrich_universe_with_offer_prices(
    universe_path: Path,
    edgar_dir: Path = Path("data/raw/edgar"),
    returns_dir: Path = Path("data/raw/returns"),
    output_path: Path | None = None,
    download_424b: bool = True,
    user_agent: str = "IPOValuationResearch pukthuanthongk@missouri.edu",
    rate_limit: float = 0.15,
) -> pd.DataFrame:
    """
    Add offer_price, first_day_return, and broken_ipo to the universe.

    Strategy:
      1. Try 424B4 (final prospectus) for exact offer price
      2. Fall back to S-1 cover page for range midpoint

    Observations without a valid offer price are KEPT in the DataFrame
    but marked — the caller is responsible for filtering them out before
    modeling (per professor's instruction: "If you can't get offer prices
    for a given IPO, that observation has to be dropped from the analysis").
    """
    universe_path = Path(universe_path)
    if output_path is None:
        output_path = universe_path

    # Read the current universe
    if universe_path.suffix == ".parquet":
        df = pd.read_parquet(universe_path)
    else:
        df = pd.read_csv(universe_path, parse_dates=["ipo_date"])

    logger.info("Enriching %d rows with offer prices", len(df))

    # Set up HTTP session for 424B downloads
    session = None
    if download_424b:
        session = requests.Session()
        session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })

    n_found = 0
    n_fdr = 0
    methods: dict[str, int] = {}

    for idx, row in df.iterrows():
        cik = str(row.get("cik", "")).strip()
        if not cik or cik in {"nan", "None"}:
            continue

        result = extract_best_offer_price(
            cik, edgar_dir,
            session=session,
            download_424b=download_424b,
            rate_limit=rate_limit,
        )
        offer_price = result["offer_price"]

        if offer_price is not None:
            df.at[idx, "offer_price"] = offer_price
            df.at[idx, "offer_price_method"] = result["method"]
            n_found += 1

            method = result["method"]
            methods[method] = methods.get(method, 0) + 1

            # Compute first-day return
            ticker = str(row.get("ticker", "")).strip()
            ipo_date = row.get("ipo_date")

            if ticker and ipo_date is not None:
                close = _get_first_day_close(ticker, ipo_date, returns_dir)
                if close is not None and offer_price > 0:
                    fdr = (close - offer_price) / offer_price
                    # Sanity: reject FDR outside [-0.95, 5.0] (i.e. -95% to +500%)
                    # Beyond this range, either the offer price or close is wrong
                    # (likely a stock-split mismatch or corrupted yfinance data).
                    if -0.95 <= fdr <= 5.0:
                        df.at[idx, "first_day_return"] = fdr
                        df.at[idx, "first_day_close"] = close
                        df.at[idx, "broken_ipo"] = int(close < offer_price)
                        n_fdr += 1
                    else:
                        logger.debug(
                            "Rejected FDR %.2f for %s (offer=%.2f, close=%.2f)",
                            fdr, ticker, offer_price, close,
                        )

        if (idx + 1) % 50 == 0:
            logger.info("  ... %d / %d processed (%d prices found)", idx + 1, len(df), n_found)

    logger.info("=" * 60)
    logger.info("OFFER PRICE EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Total CIKs:           %d", len(df))
    logger.info("  Offer prices found:   %d (%.1f%%)", n_found, 100 * n_found / max(len(df), 1))
    logger.info("  First-day returns:    %d (%.1f%%)", n_fdr, 100 * n_fdr / max(len(df), 1))
    logger.info("  Methods:")
    for m, c in sorted(methods.items(), key=lambda x: -x[1]):
        logger.info("    %-25s %d", m, c)
    logger.info("=" * 60)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    logger.info("Saved → %s", output_path)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Extract offer prices from S-1 filings")
    p.add_argument(
        "--universe", type=str,
        default="data/processed/ipo_sample/ipo_universe.parquet",
        help="Path to ipo_universe.parquet or CSV",
    )
    p.add_argument("--edgar-dir", type=str, default="data/raw/edgar")
    p.add_argument("--returns-dir", type=str, default="data/raw/returns")
    p.add_argument(
        "--output", type=str, default=None,
        help="Output path (defaults to overwriting the universe file)",
    )
    p.add_argument(
        "--no-424b", action="store_true",
        help="Skip downloading 424B4 filings (use S-1 only)",
    )
    p.add_argument("--rate-limit", type=float, default=0.15)
    args = p.parse_args(argv)

    enrich_universe_with_offer_prices(
        universe_path=Path(args.universe),
        edgar_dir=Path(args.edgar_dir),
        returns_dir=Path(args.returns_dir),
        output_path=Path(args.output) if args.output else None,
        download_424b=not args.no_424b,
        rate_limit=args.rate_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
