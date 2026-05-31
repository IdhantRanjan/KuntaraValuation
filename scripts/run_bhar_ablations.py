"""
Train ablation models on each BHAR horizon target.

Expects outputs/bhar/full_sample_bhar.parquet and multimodal_sample_bhar.parquet
to exist (run scripts/run_bhar_pipeline.py --skip-training first).

For each horizon (3/6/12/24 months) and each sample (full/multimodal),
trains the full ablation suite and saves results. Reports a combined table.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import IPOMultimodalDataset
from src.training.ablations import run_ablations

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

HORIZONS = (3, 6, 12, 24)
OUT = ROOT / "outputs" / "bhar"

MULTIMODAL_CONFIGS = [
    {"name": "naive_mean",       "modalities": None},
    {"name": "tabular_only",     "modalities": ["tabular"]},
    {"name": "text_only",        "modalities": ["text"]},
    {"name": "image_only",       "modalities": ["image"]},
    {"name": "text_tabular",     "modalities": ["text", "tabular"]},
    {"name": "full_multimodal",  "modalities": ["image", "text", "tabular"]},
]

FULL_CONFIGS = [
    {"name": "naive_mean",       "modalities": None},
    {"name": "tabular_only",     "modalities": ["tabular"]},
    {"name": "text_only",        "modalities": ["text"]},
    {"name": "text_tabular",     "modalities": ["text", "tabular"]},
]


def time_split(df: pd.DataFrame):
    df = df.copy()
    df["ipo_date"] = pd.to_datetime(df["ipo_date"])
    train = df[df["ipo_date"].dt.year <= 2018]
    val   = df[df["ipo_date"].dt.year.isin([2019, 2020])]
    test  = df[df["ipo_date"].dt.year >= 2021]
    return train, val, test


def run_horizon(
    df_orig: pd.DataFrame,
    horizon: int,
    sample_name: str,
    model_cfg: dict,
    max_epochs: int,
    configs: list[dict],
) -> pd.DataFrame | None:
    col = f"bhar_{horizon}m"
    if col not in df_orig.columns:
        logger.warning("no column %s in %s", col, sample_name)
        return None

    df = df_orig[df_orig[col].notna()].copy()
    # drop original first_day_return if present, then rename BHAR col to it
    if "first_day_return" in df.columns:
        df = df.drop(columns=["first_day_return"])
    df = df.rename(columns={col: "first_day_return"})

    train, val, test = time_split(df)
    if len(train) < 10 or len(val) < 3 or len(test) < 3:
        logger.warning("insufficient split: train=%d val=%d test=%d", len(train), len(val), len(test))
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        train.to_parquet(tp / "train.parquet", index=False)
        val.to_parquet(tp / "val.parquet",   index=False)
        test.to_parquet(tp / "test.parquet",  index=False)

        train_ds = IPOMultimodalDataset(universe_path=tp / "train.parquet")
        val_ds   = IPOMultimodalDataset(universe_path=tp / "val.parquet")
        test_ds  = IPOMultimodalDataset(universe_path=tp / "test.parquet")

        model_cfg["encoders"]["tabular"]["input_dim"] = train_ds.get_tabular_dim()

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

        out_dir = OUT / "models" / f"{sample_name}_{horizon}m"
        results = run_ablations(
            model_config=model_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            output_dir=str(out_dir),
            max_epochs=max_epochs,
            configs=configs,
        )
        results["horizon_months"] = horizon
        results["sample"] = sample_name
        results["n_train"] = len(train)
        results["n_test"]  = len(test)
        results.to_csv(out_dir / "results.csv", index=False)
        return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    p.add_argument("--samples", nargs="+", default=["multimodal", "full"])
    args = p.parse_args(argv)

    cfg = OmegaConf.load(ROOT / "configs/model/bhar_cross_attention.yaml")
    model_cfg = OmegaConf.to_container(cfg, resolve=True)

    samples = {}
    for name in args.samples:
        path = OUT / f"{name}_sample_bhar.parquet"
        if not path.exists():
            logger.error("missing %s — run run_bhar_pipeline.py first", path)
            return 1
        samples[name] = pd.read_parquet(path)

    all_results = []
    for name, df in samples.items():
        configs = MULTIMODAL_CONFIGS if name == "multimodal" else FULL_CONFIGS
        for h in args.horizons:
            logger.info("=== %s %dm ===", name, h)
            res = run_horizon(df, h, name, model_cfg, args.max_epochs, configs)
            if res is not None:
                all_results.append(res)

    if not all_results:
        logger.error("no results produced")
        return 1

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUT / "all_bhar_ablation_results.csv", index=False)

    print("\n=== BHAR ABLATION RESULTS ===")
    for h in args.horizons:
        for name in args.samples:
            sub = combined[(combined["horizon_months"] == h) & (combined["sample"] == name)]
            if sub.empty:
                continue
            print(f"\n--- {name} | {h}-month BHAR (N_train={sub['n_train'].iloc[0]}, N_test={sub['n_test'].iloc[0]}) ---")
            cols = ["ablation", "test/underpricing_mae"]
            cols = [c for c in cols if c in sub.columns]
            print(sub[cols].sort_values("test/underpricing_mae").to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
