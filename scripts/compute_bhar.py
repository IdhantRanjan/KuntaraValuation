"""
Compute the BHAR panel and horizon diagnostics once CRSP data is in hand.

Expects three inputs in data/raw/crsp/ (formats documented in
outputs/crsp_request/DATA_SPEC.md):
  firm_returns.csv  : permno, date, ret           (daily, post-offer)
  market.csv        : date, vwretd                 (CRSP value-weighted index)
  delisting.csv     : permno, delist_date, delist_ret   (optional)

Outputs the firm x horizon panel, the sample-size table, and one BHAR
histogram per horizon (raw + log-modulus) under outputs/bhar/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))
from src.data.bhar import HORIZONS_MONTHS, build_bhar_panel, horizon_sample_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _log_modulus(x: np.ndarray) -> np.ndarray:
    """Signed log transform that handles the negative BHAR tail: sign(x)*log(1+|x|)."""
    return np.sign(x) * np.log1p(np.abs(x))


def plot_horizon_histograms(panel: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted(panel["horizon_months"].unique())

    fig, axes = plt.subplots(len(horizons), 2, figsize=(13, 3.2 * len(horizons)))
    if len(horizons) == 1:
        axes = axes.reshape(1, 2)

    for i, h in enumerate(horizons):
        b = panel[(panel["horizon_months"] == h)]["bhar"].dropna().to_numpy()
        if b.size == 0:
            continue

        ax = axes[i, 0]
        ax.hist(b, bins=60, color="#4C72B0", edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.axvline(b.mean(), color="red", ls="--", lw=1.3, label=f"mean {b.mean():.2%}")
        ax.axvline(np.median(b), color="orange", ls="--", lw=1.3, label=f"median {np.median(b):.2%}")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(f"{h}-Month BHAR  (N={b.size})")
        ax.set_xlabel("BHAR vs CRSP VW index")
        ax.legend(fontsize=8)

        ax = axes[i, 1]
        lb = _log_modulus(b)
        ax.hist(lb, bins=60, color="#C44E52", edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(f"{h}-Month BHAR — signed log  (skew {pd.Series(b).skew():.2f} -> {pd.Series(lb).skew():.2f})")
        ax.set_xlabel("sign(BHAR) * log(1+|BHAR|)")

    fig.suptitle("Long-Run Post-IPO Abnormal Returns by Horizon", fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "bhar_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("wrote %s", out_dir / "bhar_histograms.png")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--crsp-dir", default=str(ROOT / "data/raw/crsp"))
    p.add_argument("--out-dir", default=str(ROOT / "outputs/bhar"))
    p.add_argument("--min-coverage", type=float, default=0.5)
    args = p.parse_args(argv)

    crsp = Path(args.crsp_dir)
    out_dir = Path(args.out_dir)

    fr_path = crsp / "firm_returns.csv"
    mk_path = crsp / "market.csv"
    dl_path = crsp / "delisting.csv"

    if not fr_path.exists() or not mk_path.exists():
        logger.error("CRSP inputs not found. Expected %s and %s", fr_path, mk_path)
        logger.error("See outputs/crsp_request/DATA_SPEC.md for the required format.")
        return 1

    firm = pd.read_csv(fr_path, parse_dates=["date"])
    market = pd.read_csv(mk_path, parse_dates=["date"])
    delist = pd.read_csv(dl_path, parse_dates=["delist_date"]) if dl_path.exists() else None

    logger.info("firm-day rows=%d  permnos=%d  market-days=%d  delist=%s",
                len(firm), firm["permno"].nunique(), len(market),
                "none" if delist is None else len(delist))

    panel = build_bhar_panel(firm, market, delist, HORIZONS_MONTHS, args.min_coverage)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_dir / "bhar_panel.csv", index=False)

    summary = horizon_sample_summary(panel)
    summary.to_csv(out_dir / "horizon_sample_summary.csv", index=False)
    print("\n=== BHAR sample size & distribution by horizon ===")
    print(summary.to_string(index=False))

    plot_horizon_histograms(panel, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
