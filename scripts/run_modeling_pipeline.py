"""
Master Modeling Pipeline — Executes the professor's directives in order:

  1. Extract offer prices (424B4 → S-1 fallback) and compute first-day returns
  2. Winsorize financial ratio outliers at 99th percentile
  3. Rebuild analysis-ready universe (drop obs without offer prices)
  4. Run ablation experiments with proper common-sample design:
     - Head-to-head comparisons on 247-firm multimodal subsample
     - Text-only robustness check on full 753-firm sample
  5. Run classical baselines (OLS, Lasso) on same samples

Usage:
    python scripts/run_modeling_pipeline.py [--skip-offer-price] [--skip-training]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("modeling_pipeline")


# ---------------------------------------------------------------------------
# Stage 1: Offer Price Extraction
# ---------------------------------------------------------------------------

def stage1_offer_prices(
    universe_path: Path,
    edgar_dir: Path,
    returns_dir: Path,
    output_path: Path,
    download_424b: bool = True,
) -> pd.DataFrame:
    """Extract offer prices and compute first-day returns."""
    from src.data.extract_offer_price import enrich_universe_with_offer_prices

    logger.info("=" * 60)
    logger.info("STAGE 1: Offer Price Extraction")
    logger.info("=" * 60)

    df = enrich_universe_with_offer_prices(
        universe_path=universe_path,
        edgar_dir=edgar_dir,
        returns_dir=returns_dir,
        output_path=output_path,
        download_424b=download_424b,
    )

    n_offer = df["offer_price"].notna().sum()
    n_fdr = df["first_day_return"].notna().sum()
    logger.info("Stage 1 done: %d offer prices, %d first-day returns", n_offer, n_fdr)
    return df


# ---------------------------------------------------------------------------
# Stage 2: Winsorization
# ---------------------------------------------------------------------------

def stage2_winsorize(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Winsorize financial ratio outliers at 99th percentile.

    Standard practice in IPO literature (per professor's approval).
    Applied to: leverage, rnd_intensity, revenue_growth, firm_age.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Winsorization (99th percentile)")
    logger.info("=" * 60)

    cols_to_winsorize = ["leverage", "rnd_intensity", "revenue_growth"]

    for col in cols_to_winsorize:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 10:
            continue

        p1 = vals.quantile(0.01)
        p99 = vals.quantile(0.99)
        n_clipped_low = (df[col] < p1).sum()
        n_clipped_high = (df[col] > p99).sum()

        df[col] = df[col].clip(lower=p1, upper=p99)
        logger.info(
            "  %s: clipped %d low (< %.4f) + %d high (> %.4f)",
            col, n_clipped_low, p1, n_clipped_high, p99,
        )

    # Also clip firm_age at reasonable bounds
    if "firm_age" in df.columns:
        # Negative firm ages are data errors; cap at 100 years
        df["firm_age"] = df["firm_age"].clip(lower=0, upper=100)
        logger.info("  firm_age: clipped to [0, 100]")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    logger.info("Saved winsorized universe → %s", output_path)
    return df


# ---------------------------------------------------------------------------
# Stage 3: Build analysis-ready samples
# ---------------------------------------------------------------------------

def stage3_build_samples(df: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Build the analysis-ready samples per professor's ablation design:

    1. full_sample: All firms with verified CIKs + offer prices + first-day returns
       (drop 332 MISMATCH, drop missing offer prices)
    2. multimodal_sample: Subset with images (for head-to-head ablation)
    3. Both use only OK/CHECK verified CIKs (professor: "Focus on the 1,235
       high-confidence matches")

    Per professor's instruction:
    "When you compare text-only to text+images, both need to be estimated
     on the same 247-firm subsample, not text on 753 and multimodal on 247."
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: Build Analysis-Ready Samples")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter: verified CIKs only (drop MISMATCH per professor's instruction)
    verified_mask = df["cik_verification_status"].isin(["OK", "CHECK"])
    logger.info("  Verification filter: %d → %d (dropped %d MISMATCH)",
                len(df), verified_mask.sum(), (~verified_mask).sum())

    # Filter: must have offer price + first-day return (professor's #1 priority)
    has_fdr = df["first_day_return"].notna()
    logger.info("  First-day return available: %d / %d", has_fdr.sum(), len(df))

    # Filter: must have S-1 filing
    has_s1 = df["edgar_has_s1"].astype(bool) if "edgar_has_s1" in df.columns else pd.Series(True, index=df.index)

    # Combined filter for full sample
    full_mask = verified_mask & has_fdr & has_s1
    full_sample = df[full_mask].reset_index(drop=True)

    # Multimodal subsample: also requires images
    has_images = df["has_images"].astype(bool) if "has_images" in df.columns else pd.Series(False, index=df.index)
    mm_mask = full_mask & has_images
    multimodal_sample = df[mm_mask].reset_index(drop=True)

    logger.info("  Full analysis-ready sample:       %d firms", len(full_sample))
    logger.info("  Multimodal (with images) sample:  %d firms", len(multimodal_sample))

    # Save both samples
    full_sample.to_parquet(output_dir / "full_sample.parquet", index=False)
    multimodal_sample.to_parquet(output_dir / "multimodal_sample.parquet", index=False)
    full_sample.to_csv(output_dir / "full_sample.csv", index=False)
    multimodal_sample.to_csv(output_dir / "multimodal_sample.csv", index=False)

    # Descriptive stats
    for name, sample in [("full", full_sample), ("multimodal", multimodal_sample)]:
        logger.info("\n--- %s sample descriptive stats ---", name)
        if "first_day_return" in sample.columns:
            fdr = sample["first_day_return"].dropna()
            logger.info("  First-day return: mean=%.3f, median=%.3f, std=%.3f, N=%d",
                        fdr.mean(), fdr.median(), fdr.std(), len(fdr))
            broken_rate = (fdr < 0).mean()
            logger.info("  Broken IPO rate: %.1f%%", broken_rate * 100)
        if "ipo_date" in sample.columns:
            years = pd.to_datetime(sample["ipo_date"]).dt.year.value_counts().sort_index()
            logger.info("  Year distribution:\n%s", years.to_string())

    return {
        "full": full_sample,
        "multimodal": multimodal_sample,
    }


# ---------------------------------------------------------------------------
# Stage 4: Run ablation experiments
# ---------------------------------------------------------------------------

def stage4_run_ablations(
    samples: dict[str, pd.DataFrame],
    output_dir: Path,
    max_epochs: int = 100,
) -> None:
    """
    Run ablation experiments with proper common-sample design.

    Experiment 1 (head-to-head, on multimodal_sample):
      - tabular_only
      - text_only
      - text_tabular
      - image_only
      - image_tabular
      - image_text
      - full_multimodal (image + text + tabular)

    Experiment 2 (robustness, on full_sample):
      - tabular_only
      - text_only
      - text_tabular
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: Ablation Experiments")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save samples as parquet for the dataset loader
    for name, sample in samples.items():
        path = output_dir / f"{name}_sample.parquet"
        sample.to_parquet(path, index=False)

    # --- Experiment 1: Head-to-head on multimodal subsample ---
    logger.info("\n--- Experiment 1: Head-to-head ablation (multimodal sample) ---")
    mm_path = output_dir / "multimodal_sample.parquet"

    if len(samples["multimodal"]) < 20:
        logger.warning("Multimodal sample too small (%d firms) — skipping training",
                       len(samples["multimodal"]))
        logger.info("Saving sample descriptive stats only.")
        _save_sample_stats(samples["multimodal"], output_dir / "multimodal_stats.json")
        return

    try:
        from src.training.ablations import run_ablations, ABLATION_CONFIGS
        from src.data.dataset import build_dataloaders
        from omegaconf import OmegaConf

        # Load model config
        cfg = OmegaConf.load(ROOT / "configs" / "model" / "cross_attention.yaml")
        model_cfg = OmegaConf.to_container(cfg, resolve=True)

        # Build dataloaders from multimodal sample
        loaders, datasets = build_dataloaders(
            universe_path=str(mm_path),
            batch_size=32,
            num_workers=0,
        )

        # Update tabular dim
        model_cfg["encoders"]["tabular"]["input_dim"] = datasets["train"].get_tabular_dim()

        # Run all 7 ablations on multimodal sample
        results_mm = run_ablations(
            model_config=model_cfg,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            test_loader=loaders["test"],
            output_dir=str(output_dir / "ablations_multimodal"),
            max_epochs=max_epochs,
        )
        logger.info("Multimodal ablation results:\n%s", results_mm.to_string())

    except Exception as e:
        logger.error("Ablation experiment 1 failed: %s", e, exc_info=True)

    # --- Experiment 2: Robustness on full sample (text-only configs) ---
    logger.info("\n--- Experiment 2: Robustness (full sample, text-only configs) ---")
    full_path = output_dir / "full_sample.parquet"

    robustness_configs = [
        {"name": "tabular_only", "modalities": ["tabular"]},
        {"name": "text_only", "modalities": ["text"]},
        {"name": "text_tabular", "modalities": ["text", "tabular"]},
    ]

    try:
        loaders_full, datasets_full = build_dataloaders(
            universe_path=str(full_path),
            batch_size=32,
            num_workers=0,
        )

        model_cfg["encoders"]["tabular"]["input_dim"] = datasets_full["train"].get_tabular_dim()

        results_full = run_ablations(
            model_config=model_cfg,
            train_loader=loaders_full["train"],
            val_loader=loaders_full["val"],
            test_loader=loaders_full["test"],
            output_dir=str(output_dir / "ablations_full_sample"),
            max_epochs=max_epochs,
            configs=robustness_configs,
        )
        logger.info("Full-sample robustness results:\n%s", results_full.to_string())

    except Exception as e:
        logger.error("Robustness experiment failed: %s", e, exc_info=True)


def _save_sample_stats(df: pd.DataFrame, path: Path) -> None:
    """Save descriptive statistics for a sample."""
    stats = {
        "n_firms": len(df),
        "year_range": f"{df['ipo_date'].min()} — {df['ipo_date'].max()}" if "ipo_date" in df.columns else "",
    }
    for col in ["first_day_return", "log_assets", "leverage", "firm_age",
                "post_ipo_volatility_6m", "rnd_intensity"]:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                stats[col] = {
                    "mean": float(vals.mean()),
                    "median": float(vals.median()),
                    "std": float(vals.std()),
                    "n": int(len(vals)),
                }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, default=str))


# ---------------------------------------------------------------------------
# Stage 5: Classical baselines
# ---------------------------------------------------------------------------

def stage5_baselines(
    samples: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Run OLS and Lasso baselines on both samples."""
    logger.info("=" * 60)
    logger.info("STAGE 5: Classical Baselines")
    logger.info("=" * 60)

    from src.data.ipo_universe import time_split
    from src.baselines.classical import run_classical_baselines

    for name, df in samples.items():
        if "first_day_return" not in df.columns or df["first_day_return"].isna().all():
            logger.warning("No first_day_return in %s sample — skipping baselines", name)
            continue

        try:
            train_df, val_df, test_df = time_split(df)
            # Merge val into train for baselines (no early stopping needed)
            train_full = pd.concat([train_df, val_df], ignore_index=True)

            if len(train_full) < 10 or len(test_df) < 5:
                logger.warning("Sample %s too small for baselines (train=%d, test=%d)",
                             name, len(train_full), len(test_df))
                continue

            results = run_classical_baselines(
                train_df=train_full,
                test_df=test_df,
                target_col="first_day_return",
                output_dir=str(output_dir / f"baselines_{name}"),
            )
            logger.info("Baselines (%s sample): %s", name, results)

        except Exception as e:
            logger.error("Baselines failed for %s: %s", name, e, exc_info=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    skip_offer_price: bool = False,
    skip_training: bool = False,
    download_424b: bool = True,
    max_epochs: int = 100,
) -> None:
    """Execute the full modeling pipeline."""
    universe_path = Path("data/processed/ipo_sample/ipo_universe.parquet")
    edgar_dir = Path("data/raw/edgar")
    returns_dir = Path("data/raw/returns")
    output_dir = Path("outputs/modeling")
    output_dir.mkdir(parents=True, exist_ok=True)

    enriched_path = Path("data/processed/ipo_sample/ipo_universe_with_prices.parquet")

    # --- Stage 1: Offer prices ---
    if skip_offer_price and enriched_path.exists():
        logger.info("Skipping offer price extraction (using existing file)")
        df = pd.read_parquet(enriched_path)
    else:
        df = stage1_offer_prices(
            universe_path=universe_path,
            edgar_dir=edgar_dir,
            returns_dir=returns_dir,
            output_path=enriched_path,
            download_424b=download_424b,
        )

    # --- Stage 2: Winsorize ---
    winsorized_path = Path("data/processed/ipo_sample/ipo_universe_final.parquet")
    df = stage2_winsorize(df, winsorized_path)

    # --- Stage 3: Build samples ---
    samples = stage3_build_samples(df, output_dir)

    # --- Stage 4: Ablations ---
    if not skip_training:
        stage4_run_ablations(samples, output_dir, max_epochs=max_epochs)

    # --- Stage 5: Baselines ---
    stage5_baselines(samples, output_dir)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Run the full modeling pipeline")
    p.add_argument("--skip-offer-price", action="store_true",
                   help="Skip offer price extraction (use existing file)")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip model training (data processing only)")
    p.add_argument("--no-424b", action="store_true",
                   help="Skip downloading 424B4 filings")
    p.add_argument("--max-epochs", type=int, default=100)
    args = p.parse_args(argv)

    run_pipeline(
        skip_offer_price=args.skip_offer_price,
        skip_training=args.skip_training,
        download_424b=not args.no_424b,
        max_epochs=args.max_epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
