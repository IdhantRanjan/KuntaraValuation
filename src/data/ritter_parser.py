"""
Ritter IPO Database Parser — Download and parse Jay Ritter's IPO dataset.

Source: https://site.warrington.ufl.edu/ritter/ipo-data/
Produces: data/raw/ritter_ipos.csv in the standard ipo_universe schema.

The Ritter database is the canonical academic IPO dataset, hand-curated since
1980 with quality controls on first-day returns, underwriter identification,
and venture-capital backing. We download the latest Excel workbook, normalize
heterogeneous column names across vintages, and emit a standardized CSV.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RITTER_PAGE_URL = "https://site.warrington.ufl.edu/ritter/ipo-data/"

# Ritter rotates the path each year. Try the most recent first, then fallbacks.
RITTER_DATA_URLS: list[str] = [
    "https://site.warrington.ufl.edu/ritter/files/2024/10/IPO-Statistics.xlsx",
    "https://site.warrington.ufl.edu/ritter/files/2024/10/IPOs2024-statistics.xlsx",
    "https://site.warrington.ufl.edu/ritter/files/2024/09/IPO-Statistics.xlsx",
    "https://site.warrington.ufl.edu/ritter/files/2023/11/IPO-Statistics.xlsx",
    "https://site.warrington.ufl.edu/ritter/files/2023/01/IPO-Statistics.xlsx",
    "https://site.warrington.ufl.edu/ritter/files/2022/02/IPO-Statistics.xlsx",
]

DEFAULT_USER_AGENT = (
    "IPOValuationResearch (pukthuanthongk@missouri.edu) "
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/537.36"
)

# ---------------------------------------------------------------------------
# Carter-Manaster (CM) underwriter rankings (tombstone-based, 0-9 scale).
# Values from Loughran & Ritter (2004) and updates in Ritter's online archive.
# ---------------------------------------------------------------------------

CARTER_MANASTER_RANKS: dict[str, float] = {
    # Bulge-bracket
    "goldman sachs": 9.0,
    "morgan stanley": 9.0,
    "jpmorgan": 9.0,
    "jp morgan": 9.0,
    "j.p. morgan": 9.0,
    "merrill lynch": 9.0,
    "bofa securities": 9.0,
    "bank of america": 9.0,
    "credit suisse": 8.1,
    "credit suisse first boston": 8.1,
    "csfb": 8.1,
    "deutsche bank": 8.0,
    "citigroup": 8.1,
    "salomon smith barney": 8.1,
    "ubs": 7.8,
    "ubs warburg": 7.8,
    "lehman brothers": 8.1,
    "bear stearns": 7.8,
    "barclays": 7.0,
    "barclays capital": 7.0,
    "wells fargo": 7.0,
    "rbc": 7.0,
    "rbc capital markets": 7.0,
    "hsbc": 7.0,
    "nomura": 7.0,
    "bnp paribas": 7.0,

    # Mid-tier
    "jefferies": 7.0,
    "william blair": 7.0,
    "robert w. baird": 6.5,
    "baird": 6.5,
    "raymond james": 6.0,
    "stifel": 6.0,
    "piper sandler": 6.0,
    "piper jaffray": 6.0,
    "cowen": 6.0,
    "needham": 6.0,
    "stephens": 6.0,
    "keefe bruyette": 6.0,
    "bmo": 6.5,
    "bmo capital markets": 6.5,
    "scotia": 6.0,
    "td securities": 6.5,

    # Lower-tier / boutique
    "canaccord": 5.0,
    "canaccord genuity": 5.0,
    "oppenheimer": 5.0,
    "guggenheim": 6.0,
    "evercore": 6.5,
    "lazard": 6.5,
    "moelis": 5.5,
    "houlihan lokey": 5.0,
    "roth capital": 4.0,
    "maxim group": 3.5,
    "aegis capital": 3.0,
    "ladenburg thalmann": 3.0,
    "boustead": 3.0,
    "ej obrien": 3.0,
}

DEFAULT_UW_RANK = 5.0  # Median CM rank for unmatched underwriters


# ---------------------------------------------------------------------------
# Column alias resolution
# ---------------------------------------------------------------------------

COLUMN_ALIASES: dict[str, list[str]] = {
    "ipo_year": [
        "year", "ipo year", "issue year", "offer year", "yr",
        "year of issue", "ipo_year",
    ],
    "ticker": [
        "ticker", "symbol", "tick", "trading symbol", "ticker symbol",
    ],
    "company_name": [
        "company name", "issuer", "name", "company", "issuer name", "firm",
    ],
    "offer_price": [
        "offer price", "price", "ipo price", "offering price", "offer_price",
    ],
    "offer_size_millions": [
        "proceeds ($mil)", "proceeds", "proceeds (millions)", "ipo proceeds",
        "offer size", "amount", "amount raised", "gross proceeds",
        "principal amount", "size",
    ],
    "first_day_return_pct": [
        "1st day return", "first day return", "initial return", "% return",
        "underpricing", "first-day return", "1st-day return",
        "first day percent return", "1st day %",
    ],
    "industry": [
        "sic", "sic code", "industry", "industry code", "fama-french",
    ],
    "vc_backed": [
        "vc", "vc-backed", "venture", "vc backed", "venture capital",
        "vc dummy",
    ],
    "underwriter": [
        "principal underwriter", "lead underwriter", "lead uw",
        "underwriter", "book runner", "bookrunner", "lead manager",
    ],
    "ipo_date": [
        "offer date", "ipo date", "issue date", "date", "offer dt",
    ],
}


def _norm_col(name: str) -> str:
    """Lowercase a column name and collapse whitespace/punctuation."""
    if not isinstance(name, str):
        name = str(name)
    name = name.lower().strip()
    name = re.sub(r"[ \t\r\n]+", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _find_column(df: pd.DataFrame, target: str) -> str | None:
    """Locate a DataFrame column by fuzzy alias match. Returns canonical name."""
    aliases = COLUMN_ALIASES.get(target, [target])
    norm_cols = {_norm_col(c): c for c in df.columns}

    # Exact alias match first
    for alias in aliases:
        if alias in norm_cols:
            return norm_cols[alias]

    # Partial substring match
    for alias in aliases:
        for nc, original in norm_cols.items():
            if alias in nc or nc in alias:
                return original

    return None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _scrape_xlsx_links_from_page(
    page_url: str = RITTER_PAGE_URL,
    timeout: float = 30.0,
) -> list[str]:
    """Scrape the Ritter IPO data page for any .xlsx links."""
    try:
        resp = requests.get(
            page_url,
            timeout=timeout,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to scrape Ritter page %s: %s", page_url, e)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".xlsx"):
            if href.startswith("/"):
                href = "https://site.warrington.ufl.edu" + href
            links.append(href)
    logger.info("Found %d .xlsx candidates on Ritter page", len(links))
    return links


def download_ritter_excel(
    output_path: Path = Path("data/raw/ritter_raw.xlsx"),
    timeout: float = 30.0,
    force: bool = False,
) -> Path:
    """
    Download Ritter's IPO Excel workbook.

    Tries known URLs in order, then falls back to scraping the page for
    .xlsx hrefs. Idempotent: if *output_path* already exists, returns it
    immediately unless *force* is True.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        logger.info("Ritter Excel already at %s — skipping download", output_path)
        return output_path

    candidates: list[str] = list(RITTER_DATA_URLS)
    candidates.extend(_scrape_xlsx_links_from_page())

    last_err: Exception | None = None
    for url in candidates:
        logger.info("Attempting Ritter download: %s", url)
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": DEFAULT_USER_AGENT},
                stream=True,
            )
            resp.raise_for_status()
            with output_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            if output_path.stat().st_size < 5_000:
                # Probably an HTML error page
                output_path.unlink(missing_ok=True)
                raise ValueError("Downloaded file too small — likely error page")
            logger.info("Downloaded Ritter Excel (%d bytes) → %s",
                        output_path.stat().st_size, output_path)
            return output_path
        except (requests.RequestException, ValueError) as e:
            logger.warning("  failed: %s", e)
            last_err = e
            continue

    raise RuntimeError(
        f"All Ritter download URLs failed. Last error: {last_err}. "
        f"Tried {len(candidates)} candidates."
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

CANDIDATE_SHEET_NAMES = (
    "All IPOs", "all ipos", "IPO data", "ipo data", "Data", "data",
    "IPOs", "ipos", "IPO", "ipo", "Underpricing", "Detail",
)


def _pick_data_sheet(xl: pd.ExcelFile) -> str:
    """Pick the most likely data sheet from an Excel workbook."""
    sheets = xl.sheet_names
    norm = {_norm_col(s): s for s in sheets}

    for cand in CANDIDATE_SHEET_NAMES:
        if cand in norm:
            return norm[cand]
        for ns, original in norm.items():
            if cand.lower() in ns:
                return original

    # Fall back to the largest sheet
    largest = None
    largest_n = -1
    for s in sheets:
        try:
            df = xl.parse(s, nrows=2000)
            if len(df) > largest_n:
                largest_n = len(df)
                largest = s
        except Exception:
            continue
    return largest or sheets[0]


def _coerce_first_day_return(value) -> float | None:
    """Coerce a first-day-return cell to a decimal return."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        value = value.replace("%", "").replace(",", "").strip()
        if not value:
            return None
        try:
            value = float(value)
        except ValueError:
            return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    # Heuristic: if abs > 3, assume the value is in percent (e.g. 12.5 -> 0.125)
    if abs(v) > 3.0:
        v = v / 100.0
    return v


def _coerce_offer_size_dollars(value) -> float | None:
    """Coerce an offer size cell (millions) to dollars."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").strip()
        try:
            value = float(value)
        except ValueError:
            return None
    try:
        return float(value) * 1_000_000.0
    except (TypeError, ValueError):
        return None


def _coerce_vc_backed(value) -> int | None:
    """Coerce a VC-backed cell to {0, 1}."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"yes", "y", "true", "vc", "1"}:
            return 1
        if s in {"no", "n", "false", "0", "non-vc"}:
            return 0
        return None
    try:
        return int(bool(int(float(value))))
    except (TypeError, ValueError):
        return None


def parse_ritter_excel(excel_path: Path) -> pd.DataFrame:
    """
    Parse a Ritter IPO Excel workbook into a normalized DataFrame.

    Returns a DataFrame with columns matching the ipo_universe schema:
        ticker, company_name, ipo_date, offer_price, offer_size,
        first_day_return, broken_ipo, industry, vc_backed, underwriter
    """
    excel_path = Path(excel_path)
    xl = pd.ExcelFile(excel_path)
    sheet = _pick_data_sheet(xl)
    logger.info("Parsing sheet '%s' from %s", sheet, excel_path.name)

    # Try several header rows — Ritter sometimes has a title row above headers.
    df = None
    for header_row in (0, 1, 2, 3):
        try:
            cand = xl.parse(sheet, header=header_row)
        except Exception:
            continue
        cand = cand.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if cand.empty:
            continue
        # Heuristic: choose the header row that yields the most aliased matches
        score = sum(
            1 for f in COLUMN_ALIASES if _find_column(cand, f) is not None
        )
        if df is None or score > df.attrs.get("_score", -1):
            df = cand
            df.attrs["_score"] = score
            df.attrs["_header_row"] = header_row
    if df is None or df.empty:
        raise ValueError(f"Could not parse any non-empty sheet from {excel_path}")

    logger.info("Parsed %d rows; matched %d field aliases (header row %d)",
                len(df), df.attrs.get("_score", 0), df.attrs.get("_header_row", 0))

    # Resolve canonical columns
    out = pd.DataFrame(index=df.index)

    year_col = _find_column(df, "ipo_year")
    date_col = _find_column(df, "ipo_date")
    ticker_col = _find_column(df, "ticker")
    name_col = _find_column(df, "company_name")
    price_col = _find_column(df, "offer_price")
    size_col = _find_column(df, "offer_size_millions")
    fdr_col = _find_column(df, "first_day_return_pct")
    ind_col = _find_column(df, "industry")
    vc_col = _find_column(df, "vc_backed")
    uw_col = _find_column(df, "underwriter")

    out["ticker"] = (
        df[ticker_col].astype(str).str.strip().str.upper() if ticker_col else ""
    )
    out["company_name"] = (
        df[name_col].astype(str).str.strip() if name_col else ""
    )

    if date_col is not None:
        date_series = df[date_col]
        # Ritter's "offer date" is sometimes a YYYYMMDD integer (e.g., 19750130)
        if pd.api.types.is_integer_dtype(date_series) or pd.api.types.is_float_dtype(date_series):
            sample = date_series.dropna().head(20)
            if not sample.empty and sample.between(19000101, 21001231).all():
                out["ipo_date"] = pd.to_datetime(
                    date_series.astype("Int64").astype(str),
                    format="%Y%m%d", errors="coerce",
                )
            else:
                out["ipo_date"] = pd.to_datetime(date_series, errors="coerce")
        else:
            out["ipo_date"] = pd.to_datetime(date_series, errors="coerce")
    elif year_col is not None:
        years = pd.to_numeric(df[year_col], errors="coerce")
        out["ipo_date"] = pd.to_datetime(
            years.astype("Int64").astype(str).where(years.notna()) + "-06-30",
            errors="coerce",
        )
    else:
        out["ipo_date"] = pd.NaT

    out["offer_price"] = (
        pd.to_numeric(df[price_col], errors="coerce") if price_col else float("nan")
    )

    if size_col is not None:
        out["offer_size"] = df[size_col].apply(_coerce_offer_size_dollars)
    else:
        out["offer_size"] = float("nan")

    if fdr_col is not None:
        out["first_day_return"] = df[fdr_col].apply(_coerce_first_day_return)
    else:
        out["first_day_return"] = float("nan")

    out["broken_ipo"] = (
        (out["first_day_return"] < 0).astype("Int64")
        .where(out["first_day_return"].notna())
    )

    out["industry"] = (
        df[ind_col].astype(str).str.strip() if ind_col else ""
    )

    if vc_col is not None:
        out["vc_backed"] = df[vc_col].apply(_coerce_vc_backed)
    else:
        out["vc_backed"] = pd.NA

    out["underwriter"] = (
        df[uw_col].astype(str).str.strip() if uw_col else ""
    )

    # Drop rows that are entirely empty in the key fields
    key_fields = ["ticker", "company_name"]
    mask = pd.Series(False, index=out.index)
    for f in key_fields:
        mask = mask | out[f].astype(str).str.strip().ne("").fillna(False)
    out = out[mask].reset_index(drop=True)

    logger.info("Parsed %d non-empty rows", len(out))
    return out


# ---------------------------------------------------------------------------
# Underwriter ranks
# ---------------------------------------------------------------------------

def _match_uw_rank(uw_name: str) -> float:
    """Match an underwriter name against the CM table."""
    if not isinstance(uw_name, str) or not uw_name.strip():
        return DEFAULT_UW_RANK
    norm = uw_name.lower().strip()
    norm = re.sub(r"[^a-z0-9 &]", "", norm)
    if norm in CARTER_MANASTER_RANKS:
        return CARTER_MANASTER_RANKS[norm]
    # Fallback: longest-substring match
    best = (None, 0)
    for key, rank in CARTER_MANASTER_RANKS.items():
        if key in norm and len(key) > best[1]:
            best = (rank, len(key))
    return best[0] if best[0] is not None else DEFAULT_UW_RANK


def assign_underwriter_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Assign Carter-Manaster underwriter ranks (0-9) using the lookup table."""
    df = df.copy()
    if "underwriter" not in df.columns:
        df["underwriter"] = ""
    df["underwriter_rank"] = df["underwriter"].apply(_match_uw_rank)
    n_matched = (df["underwriter_rank"] != DEFAULT_UW_RANK).sum()
    logger.info(
        "Underwriter ranks: %d matched in CM table, %d defaulted to %.1f",
        int(n_matched), len(df) - int(n_matched), DEFAULT_UW_RANK,
    )
    return df


# ---------------------------------------------------------------------------
# Year filter + final assembly
# ---------------------------------------------------------------------------

def _filter_by_year(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Keep IPOs with ipo_date.year in [start, end]."""
    if "ipo_date" not in df.columns:
        return df
    df = df.copy()
    years = df["ipo_date"].dt.year
    mask = (years >= start) & (years <= end)
    n_before = len(df)
    df = df[mask].reset_index(drop=True)
    logger.info("Year filter [%d, %d]: %d → %d", start, end, n_before, len(df))
    return df


def build_ritter_csv(
    output_path: Path = Path("data/raw/ritter_ipos.csv"),
    start_year: int = 2010,
    end_year: int = 2024,
    excel_path: Path | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: download → parse → filter by year → assign ranks → save CSV.

    If *excel_path* is provided, skip download and use that file. Otherwise
    download from the Ritter site and cache to data/raw/ritter_raw.xlsx.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if excel_path is None:
        excel_path = download_ritter_excel()

    df = parse_ritter_excel(excel_path)
    df = _filter_by_year(df, start_year, end_year)
    df = assign_underwriter_ranks(df)

    df.to_csv(output_path, index=False)
    logger.info("Saved %d Ritter IPOs → %s", len(df), output_path)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Build Ritter IPO CSV")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--output", type=str, default="data/raw/ritter_ipos.csv")
    parser.add_argument("--excel", type=str, default=None,
                        help="Path to a local Ritter .xlsx (skips download)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    excel_path = Path(args.excel) if args.excel else None
    build_ritter_csv(
        output_path=Path(args.output),
        start_year=args.start_year,
        end_year=args.end_year,
        excel_path=excel_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
