"""
Read data/raw/ipo_master.csv → write data/raw/cik_list.txt (one CIK per line).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=str, default="data/raw/ipo_master.csv")
    p.add_argument("--output", type=str, default="data/raw/cik_list.txt")
    args = p.parse_args(argv)

    df = pd.read_csv(args.master)
    if "cik" not in df.columns:
        raise SystemExit(f"No 'cik' column in {args.master}")

    ciks = (
        df["cik"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lstrip("0")
    )
    ciks = ciks[ciks != ""]
    ciks = ciks.drop_duplicates().sort_values()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ciks.tolist()) + "\n")
    logger.info("Wrote %d unique CIKs → %s", len(ciks), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
