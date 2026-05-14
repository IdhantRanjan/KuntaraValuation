"""
Reproduce the ablation results table from scratch.

Run:
    python notebooks/reproduce_ablation_table.py

Loads the analysis-ready samples, trains all ablation configs, and
outputs the results table. Requires the processed data files from
the pipeline (see scripts/run_modeling_pipeline.py).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

from src.data.dataset import build_dataloaders
from src.training.ablations import run_ablations
from src.data.ipo_universe import time_split


def main():
    mm_path = ROOT / "outputs" / "modeling" / "multimodal_sample.parquet"
    full_path = ROOT / "outputs" / "modeling" / "full_sample.parquet"

    if not mm_path.exists() or not full_path.exists():
        print("Run the modeling pipeline first:")
        print("  python scripts/run_modeling_pipeline.py --skip-offer-price")
        return

    cfg = OmegaConf.load(ROOT / "configs" / "model" / "cross_attention.yaml")
    model_cfg = OmegaConf.to_container(cfg, resolve=True)

    # experiment 1: head-to-head on multimodal subsample
    print("\n=== Experiment 1: Multimodal subsample (head-to-head) ===")
    loaders_mm, datasets_mm = build_dataloaders(str(mm_path), batch_size=32, num_workers=0)
    model_cfg["encoders"]["tabular"]["input_dim"] = datasets_mm["train"].get_tabular_dim()

    results_mm = run_ablations(
        model_config=model_cfg,
        train_loader=loaders_mm["train"],
        val_loader=loaders_mm["val"],
        test_loader=loaders_mm["test"],
        output_dir=str(ROOT / "outputs" / "reproduce" / "ablations_multimodal"),
        max_epochs=100,
    )

    # experiment 2: robustness on full sample (text configs only)
    print("\n=== Experiment 2: Full sample (robustness) ===")
    loaders_full, datasets_full = build_dataloaders(str(full_path), batch_size=32, num_workers=0)
    model_cfg["encoders"]["tabular"]["input_dim"] = datasets_full["train"].get_tabular_dim()

    results_full = run_ablations(
        model_config=model_cfg,
        train_loader=loaders_full["train"],
        val_loader=loaders_full["val"],
        test_loader=loaders_full["test"],
        output_dir=str(ROOT / "outputs" / "reproduce" / "ablations_full"),
        max_epochs=100,
        configs=[
            {"name": "tabular_only", "modalities": ["tabular"]},
            {"name": "text_only", "modalities": ["text"]},
            {"name": "text_tabular", "modalities": ["text", "tabular"]},
        ],
    )

    # classical baselines
    print("\n=== Classical Baselines ===")
    from src.baselines.classical import run_classical_baselines

    for name, path in [("multimodal", mm_path), ("full", full_path)]:
        df = pd.read_parquet(path)
        train_df, val_df, test_df = time_split(df)
        train_full = pd.concat([train_df, val_df])
        bl = run_classical_baselines(
            train_df=train_full,
            test_df=test_df,
            target_col="first_day_return",
            output_dir=str(ROOT / "outputs" / "reproduce" / f"baselines_{name}"),
        )
        print(f"\n{name} baselines:", bl)

    # naive mean
    for name, path in [("multimodal", mm_path), ("full", full_path)]:
        df = pd.read_parquet(path)
        train_df, val_df, test_df = time_split(df)
        mu = pd.concat([train_df, val_df])["first_day_return"].mean()
        y = test_df["first_day_return"].values
        print(f"\n{name} naive mean: MAE={np.mean(np.abs(y - mu)):.4f}")

    # combined table
    print("\n=== Combined Results ===")
    print("\nMultimodal sample:")
    print(results_mm[["ablation", "test/underpricing_mae"]].to_string(index=False))
    print("\nFull sample:")
    print(results_full[["ablation", "test/underpricing_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
