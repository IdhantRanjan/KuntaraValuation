"""
S-1 Financial Ratio Extractor.

Parses S-1 HTML filings (already downloaded by edgar_scraper) to extract
pre-IPO financial metrics: total assets, total debt, revenue, R&D expense,
and incorporation/founding year.

Outputs the canonical control variables expected by the dataset:
    log_assets, leverage, rnd_intensity, revenue_growth, firm_age
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cell parsing
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(
    r"^\(?\s*\$?\s*([0-9][0-9,]*\.?[0-9]*|\.[0-9]+)\s*\)?$"
)


def _to_number(cell) -> float | None:
    """Convert an Excel/HTML cell to a float; None if not numeric."""
    if cell is None:
        return None
    if isinstance(cell, (int, float, np.integer, np.floating)):
        if pd.isna(cell):
            return None
        return float(cell)
    s = str(cell).strip().replace("\xa0", " ")
    # Parentheses indicate negatives in financial statements
    neg = s.startswith("(") and s.endswith(")")
    s_clean = s.strip("() ")
    s_clean = s_clean.replace("$", "").replace(",", "").replace(" ", "")
    if not s_clean or s_clean in {"-", "—", "–"}:
        return None
    try:
        v = float(s_clean)
    except ValueError:
        m = _NUM_RE.match(s.strip())
        if not m:
            return None
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return -v if neg else v


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

def extract_financial_tables(html_path: Path) -> list[pd.DataFrame]:
    """Return all numeric-looking tables from an S-1 HTML filing."""
    import io
    html_path = Path(html_path)
    try:
        with html_path.open("r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
    except OSError as e:
        logger.warning("Cannot read %s: %s", html_path, e)
        return []

    try:
        # Wrap in StringIO — newer pandas treats raw strings as paths.
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except Exception:
        try:
            tables = pd.read_html(io.StringIO(html))
        except Exception as e:
            logger.debug("pd.read_html failed on %s: %s", html_path.name, e)
            return []

    out = []
    for t in tables:
        if t is None or t.empty:
            continue
        if t.shape[1] < 2 or t.shape[0] < 2:
            continue
        # Require at least 2 numeric values somewhere in the table.
        # S-1 tables often have many NaN spacer cells, so we count
        # non-empty numeric cells vs total non-empty cells.
        numeric_cells = 0
        nonempty_cells = 0
        for col in t.columns:
            for v in t[col].head(40):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    continue
                if isinstance(v, str) and not v.strip():
                    continue
                nonempty_cells += 1
                if _to_number(v) is not None:
                    numeric_cells += 1
        if numeric_cells < 2:
            continue
        if nonempty_cells > 0 and numeric_cells / nonempty_cells < 0.10:
            continue
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Statement detection
# ---------------------------------------------------------------------------

def _table_text(t: pd.DataFrame) -> str:
    """Concatenate all string content of a DataFrame (lowercase)."""
    return " ".join(t.astype(str).fillna("").values.flatten()).lower()


def find_balance_sheet(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Pick the table most likely to be the balance sheet."""
    best, best_score = None, 0
    for t in tables:
        text = _table_text(t)
        score = 0
        if "total assets" in text:
            score += 3
        if "total liabilities" in text:
            score += 3
        if "stockholders' equity" in text or "shareholders' equity" in text:
            score += 2
        if "long-term debt" in text or "long term debt" in text:
            score += 1
        if "cash and cash equivalents" in text:
            score += 1
        if score > best_score:
            best, best_score = t, score
    return best if best_score >= 3 else None


def find_income_statement(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Pick the table most likely to be the income statement."""
    best, best_score = None, 0
    for t in tables:
        text = _table_text(t)
        score = 0
        if "revenue" in text or "net sales" in text or "total revenue" in text:
            score += 3
        if "cost of revenue" in text or "cost of goods" in text:
            score += 2
        if "research and development" in text or "research & development" in text:
            score += 2
        if "operating expenses" in text:
            score += 1
        if "net loss" in text or "net income" in text:
            score += 1
        if score > best_score:
            best, best_score = t, score
    return best if best_score >= 3 else None


# ---------------------------------------------------------------------------
# Row lookup
# ---------------------------------------------------------------------------

def _find_row_value(
    table: pd.DataFrame,
    keywords: list[str],
    exclude: list[str] | None = None,
) -> tuple[float | None, float | None]:
    """
    Find the first row whose label cell contains any of *keywords*.
    Return (most_recent_period, prior_period) numeric values.

    Excludes rows whose label cell contains any of *exclude*.
    """
    exclude = exclude or []
    for _, row in table.iterrows():
        label = str(row.iloc[0]).lower() if len(row) > 0 else ""
        if not any(k in label for k in keywords):
            continue
        if any(x in label for x in exclude):
            continue
        # Collect numeric values from columns 1..N (left-to-right typically newest→older)
        nums: list[float] = []
        for cell in row.iloc[1:]:
            v = _to_number(cell)
            if v is not None:
                nums.append(v)
        if not nums:
            continue
        most_recent = nums[0]
        prior = nums[1] if len(nums) > 1 else None
        return most_recent, prior
    return None, None


# ---------------------------------------------------------------------------
# Founding year
# ---------------------------------------------------------------------------

_FOUNDED_PATTERNS = [
    re.compile(r"founded\s+in\s+(19|20)(\d{2})", re.IGNORECASE),
    re.compile(r"incorporated\s+in\s+\w+\s+in\s+(19|20)(\d{2})", re.IGNORECASE),
    re.compile(r"originally\s+incorporated.*?(19|20)(\d{2})", re.IGNORECASE),
    re.compile(r"founded\s+(19|20)(\d{2})", re.IGNORECASE),
    re.compile(r"established\s+in\s+(19|20)(\d{2})", re.IGNORECASE),
]


def extract_founding_year(html_path: Path) -> int | None:
    """Best-effort extraction of incorporation/founding year from S-1 text."""
    try:
        with html_path.open("r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
    except OSError:
        return None

    text = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
    # Inspect the first 250kB (founding history is typically up front)
    text = text[:250_000]
    for pat in _FOUNDED_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return int(m.group(1) + m.group(2))
            except (TypeError, ValueError):
                continue
    return None


# ---------------------------------------------------------------------------
# Per-CIK pipeline
# ---------------------------------------------------------------------------

def _find_s1_html(cik: str, edgar_dir: Path) -> Path | None:
    """Locate the most recent S-1 HTML file for a CIK in *edgar_dir*."""
    edgar_dir = Path(edgar_dir)
    cik_str = str(cik).lstrip("0") or str(cik)
    cik_padded = str(cik).zfill(10)

    candidates = []
    for cik_variant in (cik_str, cik_padded, str(cik)):
        d = edgar_dir / cik_variant
        if d.exists():
            candidates.extend(d.rglob("*.htm*"))
    # Also: flat-layout filings
    for fname in edgar_dir.rglob(f"*{cik_str}*.htm*"):
        candidates.append(fname)

    if not candidates:
        return None
    # Prefer files whose name suggests S-1
    s1_like = [
        p for p in candidates
        if any(kw in p.name.lower() for kw in ("s-1", "s1", "prospectus"))
    ]
    pool = s1_like or candidates
    pool.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pool[0]


def extract_key_metrics(cik: str, html_dir: Path) -> dict:
    """Extract raw financial metrics for a single CIK."""
    html_path = _find_s1_html(cik, html_dir)
    metrics = {
        "cik": str(cik),
        "html_path": str(html_path) if html_path else "",
        "total_assets": np.nan,
        "total_debt": np.nan,
        "revenue": np.nan,
        "revenue_prior": np.nan,
        "rnd_expense": np.nan,
        "founding_year": None,
    }
    if html_path is None:
        return metrics

    tables = extract_financial_tables(html_path)
    if not tables:
        logger.debug("No tables found in %s", html_path)
        return metrics

    bs = find_balance_sheet(tables)
    if bs is not None:
        ta, _ = _find_row_value(bs, ["total assets"])
        if ta is not None:
            metrics["total_assets"] = ta
        td, _ = _find_row_value(
            bs,
            ["long-term debt", "long term debt", "total debt", "notes payable"],
            exclude=["less"],
        )
        if td is not None:
            metrics["total_debt"] = td

    is_ = find_income_statement(tables)
    if is_ is not None:
        rev, rev_prior = _find_row_value(
            is_,
            ["total revenue", "net sales", "revenues", "revenue"],
            exclude=["cost", "deferred"],
        )
        if rev is not None:
            metrics["revenue"] = rev
        if rev_prior is not None:
            metrics["revenue_prior"] = rev_prior
        rnd, _ = _find_row_value(
            is_,
            ["research and development", "research & development"],
        )
        if rnd is not None:
            metrics["rnd_expense"] = rnd

    metrics["founding_year"] = extract_founding_year(html_path)

    return metrics


def compute_financial_ratios(metrics: dict, ipo_year: int | None) -> dict:
    """Compute log_assets, leverage, rnd_intensity, revenue_growth, firm_age."""
    out = {
        "log_assets": np.nan,
        "leverage": np.nan,
        "rnd_intensity": np.nan,
        "revenue_growth": np.nan,
        "firm_age": np.nan,
    }
    ta = metrics.get("total_assets")
    td = metrics.get("total_debt")
    rev = metrics.get("revenue")
    rev_prior = metrics.get("revenue_prior")
    rnd = metrics.get("rnd_expense")
    fy = metrics.get("founding_year")

    if ta is not None and not pd.isna(ta) and ta > 0:
        out["log_assets"] = float(np.log(ta))
        if td is not None and not pd.isna(td):
            out["leverage"] = float(td) / float(ta)

    if rev is not None and not pd.isna(rev) and rev > 0:
        if rnd is not None and not pd.isna(rnd):
            out["rnd_intensity"] = float(rnd) / float(rev)

    if (
        rev is not None and rev_prior is not None
        and not pd.isna(rev) and not pd.isna(rev_prior)
        and rev_prior != 0
    ):
        out["revenue_growth"] = (float(rev) - float(rev_prior)) / abs(float(rev_prior))

    if fy and ipo_year:
        try:
            out["firm_age"] = float(int(ipo_year) - int(fy))
        except (TypeError, ValueError):
            pass

    return out


# ---------------------------------------------------------------------------
# Universe enrichment
# ---------------------------------------------------------------------------

def enrich_universe_with_financials(
    universe_csv: Path,
    edgar_dir: Path = Path("data/raw/edgar"),
    output_path: Path = Path("data/processed/ipo_sample/ipo_universe_enriched.csv"),
) -> pd.DataFrame:
    """Add financial-ratio columns to the universe CSV."""
    universe_csv = Path(universe_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(universe_csv, parse_dates=["ipo_date"])
    logger.info("Enriching %d rows with S-1 financials", len(df))

    extracted = 0
    for col in ["log_assets", "leverage", "rnd_intensity",
                "revenue_growth", "firm_age"]:
        if col not in df.columns:
            df[col] = np.nan

    for idx, row in df.iterrows():
        cik = str(row.get("cik", "")).strip()
        if not cik or cik in {"nan", "None"}:
            continue
        ipo_year = None
        try:
            ipo_year = int(pd.Timestamp(row["ipo_date"]).year)
        except Exception:
            pass

        metrics = extract_key_metrics(cik, edgar_dir)
        ratios = compute_financial_ratios(metrics, ipo_year)
        for k, v in ratios.items():
            df.at[idx, k] = v
        if any(not pd.isna(v) for v in ratios.values()):
            extracted += 1

        if (idx + 1) % 25 == 0:
            logger.info("  ... %d / %d processed", idx + 1, len(df))

    df.to_csv(output_path, index=False)
    logger.info("Saved enriched universe → %s (%d / %d rows had ≥1 ratio)",
                output_path, extracted, len(df))
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Extract S-1 financial ratios")
    p.add_argument("--universe-csv", type=str, default="data/raw/ipo_master.csv")
    p.add_argument("--edgar-dir", type=str, default="data/raw/edgar")
    p.add_argument(
        "--output", type=str,
        default="data/processed/ipo_sample/ipo_universe_enriched.csv",
    )
    args = p.parse_args(argv)

    enrich_universe_with_financials(
        universe_csv=Path(args.universe_csv),
        edgar_dir=Path(args.edgar_dir),
        output_path=Path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
