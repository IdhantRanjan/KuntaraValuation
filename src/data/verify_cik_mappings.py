"""
CIK Mapping Verification — Spot-check CIK-to-company mappings against EDGAR.

Queries SEC EDGAR's submissions API for each CIK in the IPO universe CSV,
compares the returned entity name against our local company_name field, and
produces a verification report CSV.

Usage:
    python -m src.data.verify_cik_mappings \
        --csv data/raw/ipo_universe.csv \
        --output outputs/cik_verification_report.csv
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"


# ---------------------------------------------------------------------------
# Fuzzy name matching
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, and remove common suffixes."""
    name = name.lower()
    # Remove common legal suffixes
    for suffix in [
        "inc.", "inc", "corp.", "corp", "co.", "co",
        "ltd.", "ltd", "llc", "l.l.c.", "plc",
        "n.v.", "s.a.", "holdings", "group",
        "corporation", "incorporated", "company",
        "technologies", "technology",
    ]:
        name = name.replace(suffix, "")
    # Remove punctuation and extra whitespace
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _token_overlap_score(name_a: str, name_b: str) -> float:
    """
    Compute token-overlap similarity between two names.

    Returns a score in [0, 1] where 1 means all tokens match.
    """
    tokens_a = set(_normalize_name(name_a).split())
    tokens_b = set(_normalize_name(name_b).split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# EDGAR verification
# ---------------------------------------------------------------------------

def verify_single_cik(
    cik: str,
    expected_name: str,
    session: requests.Session,
    rate_limit: float = 0.15,
) -> dict:
    """
    Verify a single CIK mapping against EDGAR.

    Returns a dict with: cik, csv_name, edgar_name, match_score, status,
    has_s1_filings, s1_count, filing_types_found.
    """
    cik_padded = cik.zfill(10)
    url = f"{EDGAR_SUBMISSIONS}/CIK{cik_padded}.json"

    time.sleep(rate_limit)

    result = {
        "cik": cik,
        "csv_name": expected_name,
        "edgar_name": "",
        "match_score": 0.0,
        "status": "ERROR",
        "has_s1_filings": False,
        "s1_count": 0,
        "filing_types_found": "",
    }

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error("Failed to query EDGAR for CIK %s: %s", cik, e)
        result["status"] = "API_ERROR"
        return result

    # Extract entity name
    edgar_name = data.get("name", data.get("entityName", ""))
    result["edgar_name"] = edgar_name

    # Compute name similarity
    score = _token_overlap_score(expected_name, edgar_name)
    result["match_score"] = round(score, 3)

    if score >= 0.8:
        result["status"] = "OK"
    elif score >= 0.5:
        result["status"] = "CHECK"
    else:
        result["status"] = "MISMATCH"

    # Check for S-1 filings
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    s1_forms = [f for f in forms if f in ("S-1", "S-1/A")]
    result["s1_count"] = len(s1_forms)
    result["has_s1_filings"] = len(s1_forms) > 0

    # What filing types exist (for debugging wrong CIKs)
    unique_forms = sorted(set(forms))[:15]  # Cap to avoid huge lists
    result["filing_types_found"] = "; ".join(unique_forms)

    return result


def verify_all_ciks(
    csv_path: str | Path,
    output_path: str | Path,
    user_agent: str = "IPOValuationResearch pukthuanthongk@missouri.edu",
    rate_limit: float = 0.15,
) -> pd.DataFrame:
    """
    Verify all CIK mappings in the IPO universe CSV.

    Returns a DataFrame report and saves it to output_path.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d companies from %s", len(df), csv_path)

    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    })

    results = []
    for i, row in df.iterrows():
        cik = str(row["cik"]).strip()
        name = str(row["company_name"]).strip()
        logger.info(
            "[%d/%d] Verifying CIK %s (%s)",
            i + 1, len(df), cik, name,
        )
        result = verify_single_cik(cik, name, session, rate_limit)
        results.append(result)

    report = pd.DataFrame(results)

    # Summary statistics
    n_ok = (report["status"] == "OK").sum()
    n_check = (report["status"] == "CHECK").sum()
    n_mismatch = (report["status"] == "MISMATCH").sum()
    n_error = (report["status"].isin(["ERROR", "API_ERROR"])).sum()
    n_no_s1 = (~report["has_s1_filings"]).sum()

    logger.info("=" * 60)
    logger.info("CIK VERIFICATION REPORT")
    logger.info("=" * 60)
    logger.info("  Total companies:  %d", len(report))
    logger.info("  OK (score ≥ 0.8): %d", n_ok)
    logger.info("  CHECK (0.5-0.8):  %d", n_check)
    logger.info("  MISMATCH (< 0.5): %d", n_mismatch)
    logger.info("  API Errors:       %d", n_error)
    logger.info("  Missing S-1:      %d", n_no_s1)
    logger.info("=" * 60)

    # Print flagged entries
    flagged = report[report["status"].isin(["MISMATCH", "CHECK"])]
    if not flagged.empty:
        logger.warning("FLAGGED ENTRIES (require manual review):")
        for _, row in flagged.iterrows():
            logger.warning(
                "  CIK %s: CSV='%s' vs EDGAR='%s' (score=%.2f, status=%s)",
                row["cik"], row["csv_name"], row["edgar_name"],
                row["match_score"], row["status"],
            )

    no_s1 = report[~report["has_s1_filings"]]
    if not no_s1.empty:
        logger.warning("COMPANIES WITH NO S-1 FILINGS ON EDGAR:")
        for _, row in no_s1.iterrows():
            logger.warning(
                "  CIK %s (%s): forms found = %s",
                row["cik"], row["csv_name"], row["filing_types_found"],
            )

    report.to_csv(output_path, index=False)
    logger.info("Saved verification report → %s", output_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Verify CIK-to-company name mappings against EDGAR"
    )
    parser.add_argument(
        "--csv", type=str, default="data/raw/ipo_universe.csv",
        help="Path to the IPO universe CSV",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/cik_verification_report.csv",
        help="Output path for the verification report",
    )
    parser.add_argument(
        "--user-agent", type=str,
        default="IPOValuationResearch pukthuanthongk@missouri.edu",
        help="User-Agent for SEC EDGAR requests",
    )
    args = parser.parse_args()

    verify_all_ciks(
        csv_path=args.csv,
        output_path=args.output,
        user_agent=args.user_agent,
    )


if __name__ == "__main__":
    main()
