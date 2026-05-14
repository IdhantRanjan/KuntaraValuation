"""
SEC EDGAR S-1 Scraper — Download, parse, and extract sections from S-1 filings.

Handles:
  - Full-text index lookup via EDGAR full-text search API
  - S-1 / S-1/A filing download (HTML primary documents)
  - Section extraction: Risk Factors, MD&A, Business Overview
  - Text cleaning and normalization
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# SEC EDGAR rate limit: 10 requests / second
DEFAULT_RATE_LIMIT = 0.15  # seconds between requests
EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FILING_BASE = "https://www.sec.gov/Archives/edgar/data"
EDGAR_FULL_TEXT = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"

# Only S-1 (initial registration) and S-1/A (amendments) are in scope.
# Using post-IPO filings (10-K, 10-Q, annual reports, proxy statements,
# shareholder presentations) would introduce look-ahead bias — a fatal
# flaw that would invalidate any predictive model built from the data.
ALLOWED_FILING_TYPES = frozenset({"S-1", "S-1/A"})


class EdgarScraper:
    """Download and parse S-1 filings from SEC EDGAR."""

    def __init__(
        self,
        user_agent: str = "IPOValuationResearch pukthuanthongk@missouri.edu",
        rate_limit: float = DEFAULT_RATE_LIMIT,
        output_dir: str | Path = "data/raw/edgar",
    ):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self.rate_limit = rate_limit
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _throttle(self):
        """Enforce SEC rate limit."""
        time.sleep(self.rate_limit)

    # ------------------------------------------------------------------
    # Filing discovery
    # ------------------------------------------------------------------

    def get_filing_urls(
        self,
        cik: str,
        filing_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Query EDGAR submissions API for a CIK and return S-1 filing metadata.

        Only S-1 and S-1/A filings are allowed.  Passing any other filing
        type raises ValueError to prevent accidental look-ahead bias.

        Returns list of dicts with keys: accession_number, filing_date,
        primary_document, filing_url.
        """
        if filing_types is None:
            filing_types = ["S-1", "S-1/A"]

        # Enforce S-1-only scope
        disallowed = set(filing_types) - ALLOWED_FILING_TYPES
        if disallowed:
            raise ValueError(
                f"Filing types {disallowed} are not allowed. "
                f"Only {sorted(ALLOWED_FILING_TYPES)} are in scope. "
                f"Using post-IPO filings would introduce look-ahead bias."
            )

        cik_padded = cik.zfill(10)
        url = f"{EDGAR_SUBMISSIONS}/CIK{cik_padded}.json"

        self._throttle()
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to fetch submissions for CIK %s: %s", cik, e)
            return []

        data = resp.json()

        results: list[dict] = []

        def _ingest(block: dict) -> None:
            forms = block.get("form", [])
            accessions = block.get("accessionNumber", [])
            dates = block.get("filingDate", [])
            primary_docs = block.get("primaryDocument", [])
            for form, acc, date, doc in zip(forms, accessions, dates, primary_docs):
                if form not in filing_types:
                    continue
                acc_formatted = acc.replace("-", "")
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{acc_formatted}/{doc}"
                )
                results.append({
                    "cik": cik,
                    "accession_number": acc,
                    "filing_date": date,
                    "form_type": form,
                    "primary_document": doc,
                    "filing_url": filing_url,
                })

        # 1) Recent filings (last ~1000 entries kept in the main JSON)
        _ingest(data.get("filings", {}).get("recent", {}))

        # 2) Paginated history files (older S-1s roll off "recent" — common
        #    for IPOs >2-3 years old like ABNB/COIN/SNOW). Walk every paged
        #    submission file referenced in filings.files.
        history_files = data.get("filings", {}).get("files", []) or []
        for entry in history_files:
            sub_name = entry.get("name")
            if not sub_name:
                continue
            sub_url = f"https://data.sec.gov/submissions/{sub_name}"
            self._throttle()
            try:
                sub_resp = self.session.get(sub_url, timeout=30)
                sub_resp.raise_for_status()
                _ingest(sub_resp.json())
            except (requests.RequestException, ValueError) as e:
                logger.debug("History page %s failed: %s", sub_name, e)
                continue

        # De-duplicate by accession number (history can overlap "recent")
        seen: set[str] = set()
        unique: list[dict] = []
        for r in results:
            if r["accession_number"] in seen:
                continue
            seen.add(r["accession_number"])
            unique.append(r)

        logger.info("CIK %s: found %d S-1 filings (%d after dedup)",
                    cik, len(results), len(unique))
        return unique

    # ------------------------------------------------------------------
    # Filing download
    # ------------------------------------------------------------------

    def download_filing(self, filing_info: dict) -> Path | None:
        """Download a single filing HTML and save to disk."""
        cik = filing_info["cik"]
        acc = filing_info["accession_number"].replace("-", "")
        out_path = self.output_dir / cik / f"{acc}.html"

        if out_path.exists():
            logger.debug("Already downloaded: %s", out_path)
            return out_path

        out_path.parent.mkdir(parents=True, exist_ok=True)

        self._throttle()
        try:
            resp = self.session.get(filing_info["filing_url"], timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to download %s: %s", filing_info["filing_url"], e)
            return None

        out_path.write_text(resp.text, encoding="utf-8")
        logger.info("Downloaded → %s", out_path)
        return out_path

    def download_all_for_cik(
        self,
        cik: str,
        filing_types: list[str] | None = None,
    ) -> list[Path]:
        """Download all S-1 filings for a given CIK."""
        filings = self.get_filing_urls(cik, filing_types)
        if not filings:
            logger.warning(
                "CIK %s: NO S-1 filings found. This may indicate a wrong "
                "CIK mapping — verify with the CIK spot-check tool.", cik,
            )
        paths = []
        for f in filings:
            p = self.download_filing(f)
            if p is not None:
                paths.append(p)
        return paths

    # ------------------------------------------------------------------
    # Section extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_sections(
        html_path: str | Path,
        sections: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Extract named sections from an S-1 HTML filing.

        Uses heading-level regex to find section boundaries. Returns cleaned
        plain text for each section.
        """
        if sections is None:
            sections = ["Risk Factors", "Business", "Management's Discussion"]

        html_path = Path(html_path)
        if not html_path.exists():
            return {}

        raw_html = html_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw_html, "lxml")

        # Remove scripts, styles, and hidden elements
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()

        full_text = soup.get_text(separator="\n")

        extracted = {}
        for section_name in sections:
            # Build regex to find section heading
            pattern = re.compile(
                rf"(?:ITEM\s+\d+[A-Z]?\.?\s*)?{re.escape(section_name)}",
                re.IGNORECASE,
            )
            matches = list(pattern.finditer(full_text))
            if not matches:
                logger.debug("Section '%s' not found in %s", section_name, html_path.name)
                continue

            start = matches[0].end()

            # Find next section heading (any ITEM header or next known section)
            next_section = re.compile(
                r"\n\s*(?:ITEM\s+\d+[A-Z]?\.?|PART\s+[IVX]+)",
                re.IGNORECASE,
            )
            next_match = next_section.search(full_text, start)
            end = next_match.start() if next_match else start + 100_000

            raw_section = full_text[start:end]
            cleaned = _clean_text(raw_section)
            extracted[section_name] = cleaned

        return extracted

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_cik_list(
        self,
        ciks: list[str],
        sections: list[str] | None = None,
        text_output_dir: str | Path = "data/processed/text",
    ) -> None:
        """Download filings and extract text for a list of CIKs."""
        text_output_dir = Path(text_output_dir)
        text_output_dir.mkdir(parents=True, exist_ok=True)

        for i, cik in enumerate(ciks):
            logger.info("[%d/%d] Processing CIK %s", i + 1, len(ciks), cik)
            paths = self.download_all_for_cik(cik)

            for path in paths:
                sec_texts = self.extract_sections(path, sections)
                if not sec_texts:
                    continue

                out_file = text_output_dir / cik / f"{path.stem}_sections.json"
                out_file.parent.mkdir(parents=True, exist_ok=True)

                import json
                out_file.write_text(
                    json.dumps(sec_texts, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.info("  Saved sections → %s", out_file)


# ---------------------------------------------------------------------------
# Text cleaning utilities
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Clean extracted text for NLP processing."""
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove non-ASCII but keep common punctuation
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    # Remove leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Remove table artifacts (rows of dots, dashes, or underscores)
    text = re.sub(r"^[.\-_]{5,}$", "", text, flags=re.MULTILINE)
    return text.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry for EDGAR scraping."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Download and parse S-1 filings")
    parser.add_argument("--ciks", nargs="+", help="CIK numbers to process")
    parser.add_argument("--cik-file", type=str, help="File with one CIK per line")
    parser.add_argument("--user-agent", type=str, default="IPOResearch idhantran@gmail.com")
    parser.add_argument("--output-dir", type=str, default="data/raw/edgar")
    parser.add_argument("--text-output-dir", type=str, default="data/processed/text")
    args = parser.parse_args()

    ciks = args.ciks or []
    if args.cik_file:
        ciks = Path(args.cik_file).read_text().strip().split("\n")

    if not ciks:
        logger.error("No CIKs provided. Use --ciks or --cik-file.")
        return

    scraper = EdgarScraper(
        user_agent=args.user_agent,
        output_dir=args.output_dir,
    )
    scraper.process_cik_list(
        ciks=ciks,
        text_output_dir=args.text_output_dir,
    )


if __name__ == "__main__":
    main()
