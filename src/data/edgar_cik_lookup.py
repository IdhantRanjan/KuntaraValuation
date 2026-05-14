"""
EDGAR CIK Lookup — Map company names and tickers to SEC CIK numbers.

Multi-strategy lookup:
  1. Exact ticker match against EDGAR's company_tickers.json
  2. Fuzzy company-name match against the same DB
  3. EDGAR full-text search (EFTS) by ticker / name with S-1 form filter
  4. (Optional) manual override table for known hard cases

Produces a mapping CSV: ticker → cik, edgar_name, match_score, method.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.parse as urlparse
from pathlib import Path

import pandas as pd
import requests

from src.data.verify_cik_mappings import _normalize_name, _token_overlap_score

logger = logging.getLogger(__name__)


EDGAR_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
EDGAR_EFTS_SEARCH = "https://efts.sec.gov/LATEST/search-index?{query}"

DEFAULT_CACHE_PATH = Path("data/raw/edgar_company_tickers.json")

# Optional manual overrides for hard cases (ticker or normalized name → CIK).
MANUAL_OVERRIDES: dict[str, str] = {
    # ticker overrides
    "ABNB": "1559720",     # Airbnb (note: 1341439 is wrong; Airbnb is 1559720)
    "COIN": "1679788",     # Coinbase
    "DASH": "1792789",     # DoorDash
    "SNOW": "1640147",     # Snowflake
    "PLTR": "1321655",     # Palantir
    "RBLX": "1315098",     # Roblox
    "RIVN": "1874178",     # Rivian
    "PATH": "1734722",     # UiPath (Note: 1545654 is wrong)
    "HOOD": "1783879",     # Robinhood (Note: 1699855 is wrong)
    "U":    "1810806",     # Unity
    "BMBL": "1830043",     # Bumble
    "TOST": "1650164",     # Toast (Note: 1673139 is wrong)
    "GTLB": "1653482",     # GitLab
    "HCP":  "1720671",     # HashiCorp
    "BASE": "1639825",     # Couchbase
    "BRZE": "1538097",     # Braze
    "AMPL": "1866364",     # Amplitude
    "DUOL": "1562088",     # Duolingo
    "IOT":  "1642896",     # Samsara
    "CXM":  "1569345",     # Sprinklr
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_name_strict(name: str) -> str:
    """Normalize a name for ticker-DB lookup keys."""
    return _normalize_name(name)


def _name_variants(name: str) -> list[str]:
    """Generate alternative normalizations of a company name."""
    n0 = name.strip()
    variants = [n0]
    parts = n0.split()
    if len(parts) >= 2:
        variants.append(" ".join(parts[:2]))
    # No-suffix variant via _normalize_name
    nn = _normalize_name(n0)
    if nn:
        variants.append(nn)
    # First-token variant (only if reasonably distinctive)
    if parts and len(parts[0]) >= 4:
        variants.append(parts[0])
    # De-dup, preserve order
    seen = set()
    out = []
    for v in variants:
        v = v.strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# CIKLookup
# ---------------------------------------------------------------------------

class CIKLookup:
    """Multi-strategy CIK lookup against SEC EDGAR."""

    def __init__(
        self,
        user_agent: str,
        rate_limit: float = 0.12,
        cache_path: Path = DEFAULT_CACHE_PATH,
        use_efts: bool = False,
    ) -> None:
        self.user_agent = user_agent
        self.rate_limit = rate_limit
        self.cache_path = Path(cache_path)
        self.use_efts = use_efts
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json, text/html;q=0.9, */*;q=0.5",
        })

        self._tickers_db: dict[str, dict] | None = None
        self._name_index: dict[str, str] | None = None  # normalized_name -> cik

    # ---------- Tickers DB ----------
    def load_company_tickers_db(self) -> dict[str, dict]:
        """
        Download (or load cached) company_tickers.json.

        Returns a dict {ticker: {cik, title}, ...} and builds a parallel
        normalized-name → cik index in self._name_index.
        """
        if self._tickers_db is not None:
            return self._tickers_db

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict | None = None
        if self.cache_path.exists():
            try:
                with self.cache_path.open() as f:
                    data = json.load(f)
                logger.info("Loaded cached EDGAR company tickers from %s",
                            self.cache_path)
            except Exception as e:
                logger.warning("Cache read failed (%s) — re-downloading", e)
                data = None

        if data is None:
            logger.info("Downloading EDGAR company tickers...")
            try:
                resp = self.session.get(EDGAR_COMPANY_TICKERS, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                with self.cache_path.open("w") as f:
                    json.dump(data, f)
                logger.info("Cached %d entries → %s", len(data), self.cache_path)
            except requests.RequestException as e:
                logger.error("Failed to download company_tickers.json: %s", e)
                data = {}

        # Normalize: outer JSON is {"0": {"cik_str":..., "ticker":..., "title":...}, ...}
        tickers_db: dict[str, dict] = {}
        name_index: dict[str, str] = {}
        for _, entry in (data or {}).items():
            try:
                cik = str(entry["cik_str"]).strip()
                ticker = str(entry["ticker"]).strip().upper()
                title = str(entry["title"]).strip()
            except (KeyError, TypeError):
                continue
            tickers_db[ticker] = {"cik": cik, "title": title}
            norm = _norm_name_strict(title)
            if norm and norm not in name_index:
                name_index[norm] = cik

        self._tickers_db = tickers_db
        self._name_index = name_index
        logger.info("Indexed %d tickers / %d normalized names",
                    len(tickers_db), len(name_index))
        return tickers_db

    # ---------- Strategies ----------
    def lookup_by_ticker(
        self,
        ticker: str,
        ipo_year: int | None = None,
    ) -> dict | None:
        """Exact ticker match against the EDGAR tickers DB."""
        if not ticker:
            return None
        ticker = str(ticker).strip().upper()

        # Manual overrides first
        if ticker in MANUAL_OVERRIDES:
            return {
                "cik": MANUAL_OVERRIDES[ticker],
                "edgar_name": "",
                "method": "manual_override",
                "score": 1.0,
            }

        db = self.load_company_tickers_db()
        if ticker in db:
            entry = db[ticker]
            return {
                "cik": entry["cik"],
                "edgar_name": entry.get("title", ""),
                "method": "ticker_exact",
                "score": 1.0,
            }
        return None

    def lookup_by_name(
        self,
        company_name: str,
        ipo_year: int | None = None,
        min_score: float = 0.5,
    ) -> dict | None:
        """Fuzzy name match against the EDGAR tickers DB."""
        if not company_name:
            return None

        db = self.load_company_tickers_db()
        if not db:
            return None

        # Strict exact normalized-name hit
        for variant in _name_variants(company_name):
            norm = _norm_name_strict(variant)
            if not norm:
                continue
            if self._name_index and norm in self._name_index:
                cik = self._name_index[norm]
                # Find the title back
                edgar_name = ""
                for _, entry in db.items():
                    if entry["cik"] == cik:
                        edgar_name = entry.get("title", "")
                        break
                return {
                    "cik": cik,
                    "edgar_name": edgar_name,
                    "method": "name_exact",
                    "score": 1.0,
                }

        # Fuzzy: scan db, score by token overlap
        best: tuple[float, str, str] = (0.0, "", "")
        for _, entry in db.items():
            score = _token_overlap_score(company_name, entry.get("title", ""))
            if score > best[0]:
                best = (score, str(entry["cik"]), str(entry.get("title", "")))

        if best[0] >= min_score:
            return {
                "cik": best[1],
                "edgar_name": best[2],
                "method": "name_fuzzy",
                "score": round(best[0], 3),
            }
        return None

    def lookup_by_efts(
        self,
        ticker: str,
        company_name: str,
        ipo_year: int | None = None,
    ) -> dict | None:
        """
        EDGAR full-text search for S-1 filings matching ticker/name.

        Returns the highest-scoring CIK from EFTS hits.
        """
        # Construct query: prefer ticker quoted, fall back to name
        terms = []
        if ticker:
            terms.append(f'"{ticker}"')
        if company_name:
            # Drop suffixes for better recall
            simple = _normalize_name(company_name).strip()
            if simple:
                terms.append(f'"{simple}"')

        if not terms:
            return None

        query = " OR ".join(terms)
        params = {"q": query, "forms": "S-1"}
        if ipo_year:
            params["dateRange"] = "custom"
            params["startdt"] = f"{ipo_year - 2}-01-01"
            params["enddt"] = f"{ipo_year + 1}-12-31"

        url = EDGAR_EFTS_SEARCH.format(query=urlparse.urlencode(params))

        try:
            time.sleep(self.rate_limit)
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.debug("EFTS search failed for %s: %s", ticker or company_name, e)
            return None

        hits = (data.get("hits", {}) or {}).get("hits", []) or []
        best: tuple[float, str, str] = (0.0, "", "")
        for hit in hits[:25]:
            src = hit.get("_source", {}) or {}
            display_names = src.get("display_names") or []
            ciks_field = src.get("ciks") or []
            if not display_names or not ciks_field:
                continue
            display = display_names[0] if isinstance(display_names, list) else str(display_names)
            cik = ciks_field[0] if isinstance(ciks_field, list) else str(ciks_field)
            cik = str(cik).lstrip("0") or str(cik)
            score = _token_overlap_score(company_name or ticker, display)
            if score > best[0]:
                best = (score, cik, display)

        if best[0] >= 0.3:
            return {
                "cik": best[1],
                "edgar_name": best[2],
                "method": "efts",
                "score": round(best[0], 3),
            }
        return None

    # ---------- Top-level ----------
    def lookup(
        self,
        ticker: str,
        company_name: str,
        ipo_year: int | None = None,
    ) -> dict:
        """
        Run all strategies in order; return the first decisive result.
        """
        ticker = (ticker or "").strip().upper()
        company_name = (company_name or "").strip()

        # 1) Exact ticker
        result = self.lookup_by_ticker(ticker, ipo_year=ipo_year)
        if result:
            return result

        # 2) Name search
        result = self.lookup_by_name(company_name, ipo_year=ipo_year)
        if result:
            return result

        # 3) EFTS (slow — disabled by default for bulk jobs)
        if self.use_efts:
            result = self.lookup_by_efts(ticker, company_name, ipo_year=ipo_year)
            if result:
                return result

        return {
            "cik": None,
            "edgar_name": "",
            "method": "not_found",
            "score": 0.0,
        }


# ---------------------------------------------------------------------------
# Mapping builder
# ---------------------------------------------------------------------------

def build_cik_mapping(
    ritter_csv_path: Path,
    output_path: Path,
    user_agent: str = "IPOResearch pukthuanthongk@missouri.edu",
    rate_limit: float = 0.12,
    use_efts: bool = False,
) -> pd.DataFrame:
    """Run CIK lookup over every row in the Ritter CSV, save mapping CSV."""
    ritter_csv_path = Path(ritter_csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ritter_csv_path, parse_dates=["ipo_date"])
    logger.info("Loaded %d Ritter rows from %s", len(df), ritter_csv_path)

    lookup = CIKLookup(
        user_agent=user_agent, rate_limit=rate_limit, use_efts=use_efts,
    )
    lookup.load_company_tickers_db()

    rows = []
    for i, r in df.iterrows():
        ticker = str(r.get("ticker", "") or "").strip().upper()
        name = str(r.get("company_name", "") or "").strip()
        ipo_year = None
        if "ipo_date" in df.columns:
            try:
                ipo_year = int(pd.Timestamp(r["ipo_date"]).year)
            except Exception:
                ipo_year = None

        result = lookup.lookup(ticker, name, ipo_year=ipo_year)
        rows.append({
            "ticker": ticker,
            "company_name": name,
            "ipo_year": ipo_year,
            "cik": result["cik"],
            "edgar_name": result["edgar_name"],
            "method": result["method"],
            "score": result["score"],
        })
        if (i + 1) % 50 == 0:
            logger.info("  ... processed %d / %d", i + 1, len(df))

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)

    # Summary
    method_counts = out["method"].value_counts().to_dict()
    n_found = (out["cik"].notna() & (out["method"] != "not_found")).sum()
    logger.info("=" * 60)
    logger.info("CIK MAPPING SUMMARY")
    logger.info("=" * 60)
    logger.info("  Total: %d  Found: %d (%.1f%%)",
                len(out), int(n_found), 100 * n_found / max(len(out), 1))
    for m, c in method_counts.items():
        logger.info("    %-20s %d", m, c)
    logger.info("Saved → %s", output_path)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Build CIK mapping from Ritter CSV")
    p.add_argument("--ritter-csv", type=str, default="data/raw/ritter_ipos.csv")
    p.add_argument("--output", type=str, default="data/raw/ritter_cik_mapping.csv")
    p.add_argument("--user-agent", type=str,
                   default="IPOResearch pukthuanthongk@missouri.edu")
    p.add_argument("--rate-limit", type=float, default=0.12)
    p.add_argument("--use-efts", action="store_true",
                   help="Enable EFTS fallback (slower, more thorough)")
    args = p.parse_args(argv)

    build_cik_mapping(
        ritter_csv_path=Path(args.ritter_csv),
        output_path=Path(args.output),
        user_agent=args.user_agent,
        rate_limit=args.rate_limit,
        use_efts=args.use_efts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
