"""
End-to-end BHAR pipeline:
  1. Fetch post-IPO daily returns via yfinance
  2. Compute BHARs at 3/6/12/24 months vs SPY
  3. Merge onto the universe (so new targets sit alongside old features)
  4. Generate BHAR histograms and sample-size tables
  5. Train the multimodal model on each horizon target

Run:
    python scripts/run_bhar_pipeline.py [--skip-fetch] [--skip-training] [--max-epochs N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.fetch_post_ipo_returns import (
    compute_bhar_yfinance,
    fetch_firm_returns,
    fetch_market_returns,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

HORIZONS = (3, 6, 12, 24)
OUT = ROOT / "outputs" / "bhar"


def stage1_fetch(universe: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=== Stage 1: fetch post-IPO returns ===")
    firm_daily = fetch_firm_returns(universe, data_dir)

    mkt_path = data_dir / "crsp" / "market.csv"
    market = pd.read_csv(mkt_path, parse_dates=["date"])
    return firm_daily, market


def stage2_compute_bhar(
    firm_daily: pd.DataFrame,
    market: pd.DataFrame,
    cache_path: Path,
) -> pd.DataFrame:
    logger.info("=== Stage 2: compute BHARs ===")
    if cache_path.exists():
        logger.info("loading cached BHAR panel from %s", cache_path)
        return pd.read_parquet(cache_path)

    panel = compute_bhar_yfinance(firm_daily, market, horizons=HORIZONS)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache_path, index=False)
    logger.info("saved BHAR panel: %d rows", len(panel))
    return panel


def stage3_sample_summary(panel: pd.DataFrame) -> pd.DataFrame:
    logger.info("=== Stage 3: sample summary by horizon ===")
    rows = []
    for h in HORIZONS:
        g = panel[panel["horizon_months"] == h]
        usable = g[g["status"] == "complete"]["bhar"].dropna()
        rows.append({
            "horizon_months": h,
            "n_total": len(g),
            "n_usable": int((g["status"] == "complete").sum()),
            "n_insufficient": int((g["status"] == "insufficient").sum()),
            "bhar_mean": float(usable.mean()) if len(usable) else np.nan,
            "bhar_median": float(usable.median()) if len(usable) else np.nan,
            "bhar_std": float(usable.std()) if len(usable) else np.nan,
            "bhar_skew": float(usable.skew()) if len(usable) > 2 else np.nan,
            "pct_below_zero": float((usable < 0).mean()) if len(usable) else np.nan,
            "pct_below_minus20": float((usable < -0.20).mean()) if len(usable) else np.nan,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "horizon_sample_summary.csv", index=False)

    print("\n=== BHAR SAMPLE SUMMARY ===")
    print(summary.to_string(index=False))
    return summary


def stage4_histograms(panel: pd.DataFrame) -> None:
    logger.info("=== Stage 4: BHAR histograms ===")
    fig, axes = plt.subplots(len(HORIZONS), 2, figsize=(13, 3.5 * len(HORIZONS)))

    for i, h in enumerate(HORIZONS):
        b = panel[(panel["horizon_months"] == h) & (panel["status"] == "complete")]["bhar"].dropna().to_numpy()
        if b.size == 0:
            continue
        log_b = np.sign(b) * np.log1p(np.abs(b))

        ax = axes[i, 0]
        ax.hist(b, bins=60, color="#4C72B0", edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.axvline(b.mean(), color="red", ls="--", lw=1.3, label=f"mean {b.mean():.2%}")
        ax.axvline(np.median(b), color="orange", ls="--", lw=1.3, label=f"median {np.median(b):.2%}")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(f"{h}-Month BHAR vs SPY  (N={b.size})")
        ax.set_xlabel("BHAR")
        ax.legend(fontsize=8)

        ax = axes[i, 1]
        ax.hist(log_b, bins=60, color="#C44E52", edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.axvline(0, color="black", lw=0.6)
        raw_skew = pd.Series(b).skew()
        log_skew = pd.Series(log_b).skew()
        ax.set_title(f"{h}-Month BHAR signed-log  (skew {raw_skew:.2f} → {log_skew:.2f})")
        ax.set_xlabel("sign(BHAR) × log(1+|BHAR|)")

    fig.suptitle("Long-Run Post-IPO Abnormal Returns vs SPY (yfinance)", fontsize=13, y=1.005)
    fig.tight_layout()
    out_path = OUT / "bhar_histograms.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("wrote %s", out_path)


def stage5_build_model_samples(
    universe: pd.DataFrame,
    panel: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    logger.info("=== Stage 5: merge BHAR targets onto universe ===")
    # Pivot to wide format: one column per horizon
    wide = panel[panel["status"] == "complete"].pivot_table(
        index="ticker", columns="horizon_months", values="bhar", aggfunc="first"
    ).reset_index()
    wide.columns = ["ticker"] + [f"bhar_{h}m" for h in wide.columns[1:]]
    wide.columns.name = None

    merged = universe.merge(wide, on="ticker", how="left")

    samples = {}
    bhar_cols = [f"bhar_{h}m" for h in HORIZONS]
    for col in bhar_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    full_path = OUT / "full_sample_bhar.parquet"
    merged.to_parquet(full_path, index=False)
    samples["full"] = merged

    mm_mask = merged["has_images"].fillna(False).astype(bool)
    mm = merged[mm_mask].reset_index(drop=True)
    mm_path = OUT / "multimodal_sample_bhar.parquet"
    mm.to_parquet(mm_path, index=False)
    samples["multimodal"] = mm

    for name, df in samples.items():
        print(f"\n{name} sample: {len(df)} firms")
        for col in bhar_cols:
            n = df[col].notna().sum()
            if n > 0:
                print(f"  {col}: {n} obs, mean={df[col].mean():.3f}, median={df[col].median():.3f}")

    return samples


def stage6_train_models(samples: dict[str, pd.DataFrame], max_epochs: int) -> None:
    logger.info("=== Stage 6: train multimodal models on BHAR targets ===")
    from torch.utils.data import DataLoader
    from src.data.dataset import IPOMultimodalDataset
    from src.training.ablations import run_ablations

    cfg = OmegaConf.load(ROOT / "configs/model/cross_attention.yaml")
    model_cfg = OmegaConf.to_container(cfg, resolve=True)

    for horizon in HORIZONS:
        bhar_col = f"bhar_{horizon}m"
        for sample_name, df in samples.items():
            usable = df[df[bhar_col].notna()].copy()
            usable = usable.rename(columns={bhar_col: "first_day_return"})
            if len(usable) < 30:
                logger.warning("skip h=%dm %s: only %d obs", horizon, sample_name, len(usable))
                continue

            # time split
            usable["ipo_date"] = pd.to_datetime(usable["ipo_date"])
            train = usable[usable["ipo_date"].dt.year <= 2018]
            val = usable[usable["ipo_date"].dt.year.isin([2019, 2020])]
            test = usable[usable["ipo_date"].dt.year >= 2021]
            if len(train) < 10 or len(val) < 5 or len(test) < 5:
                logger.warning("skip h=%dm %s: insufficient split sizes", horizon, sample_name)
                continue

            out_dir = OUT / f"models/{sample_name}_{horizon}m"
            out_dir.mkdir(parents=True, exist_ok=True)

            # save temp parquets for dataset loader
            train_p = out_dir / "train.parquet"
            val_p = out_dir / "val.parquet"
            test_p = out_dir / "test.parquet"
            train.to_parquet(train_p, index=False)
            val.to_parquet(val_p, index=False)
            test.to_parquet(test_p, index=False)

            try:
                train_ds = IPOMultimodalDataset(universe_path=train_p)
                val_ds = IPOMultimodalDataset(universe_path=val_p)
                test_ds = IPOMultimodalDataset(universe_path=test_p)

                train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
                test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

                model_cfg["encoders"]["tabular"]["input_dim"] = train_ds.get_tabular_dim()

                if sample_name == "multimodal":
                    configs = [
                        {"name": "naive_mean", "modalities": None},
                        {"name": "tabular_only", "modalities": ["tabular"]},
                        {"name": "text_only", "modalities": ["text"]},
                        {"name": "image_only", "modalities": ["image"]},
                        {"name": "text_tabular", "modalities": ["text", "tabular"]},
                        {"name": "full_multimodal", "modalities": ["image", "text", "tabular"]},
                    ]
                else:
                    configs = [
                        {"name": "naive_mean", "modalities": None},
                        {"name": "tabular_only", "modalities": ["tabular"]},
                        {"name": "text_only", "modalities": ["text"]},
                        {"name": "text_tabular", "modalities": ["text", "tabular"]},
                    ]

                results = run_ablations(
                    model_config=model_cfg,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    output_dir=str(out_dir),
                    max_epochs=max_epochs,
                    configs=configs,
                )
                results_path = out_dir / "results.csv"
                results.to_csv(results_path, index=False)
                logger.info("h=%dm %s: done -> %s", horizon, sample_name, results_path)
                print(f"\n=== {sample_name} {horizon}m results ===")
                print(results[["ablation", "test/underpricing_mae"]].to_string(index=False))

            except Exception as e:
                logger.error("h=%dm %s failed: %s", horizon, sample_name, e)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-fetch", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--max-epochs", type=int, default=50)
    args = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)

    universe = pd.read_parquet(ROOT / "data/processed/ipo_sample/ipo_universe_final.parquet")
    data_dir = ROOT / "data/raw"

    if args.skip_fetch:
        cache = data_dir / "firm_daily_returns.parquet"
        firm_daily = pd.read_parquet(cache)
        market = pd.read_csv(data_dir / "crsp/market.csv", parse_dates=["date"])
    else:
        firm_daily, market = stage1_fetch(universe, data_dir)

    panel = stage2_compute_bhar(firm_daily, market, OUT / "bhar_panel.parquet")
    stage3_sample_summary(panel)
    stage4_histograms(panel)
    samples = stage5_build_model_samples(universe, panel)

    if not args.skip_training:
        stage6_train_models(samples, args.max_epochs)

    logger.info("Pipeline complete. Outputs in %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
