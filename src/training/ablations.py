"""
Ablation Runner — Systematically run single-modality and pairwise ablations.

Runs all 7 ablation configurations (3 single, 3 pairwise, 1 full) and
collects results for the Figure 1 bar chart.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.models.multimodal import MultimodalIPOModel
from src.training.trainer import train

logger = logging.getLogger(__name__)

# All 7 ablation configurations
ABLATION_CONFIGS = [
    {"name": "tabular_only",     "modalities": ["tabular"]},
    {"name": "text_only",        "modalities": ["text"]},
    {"name": "image_only",       "modalities": ["image"]},
    {"name": "text_tabular",     "modalities": ["text", "tabular"]},
    {"name": "image_tabular",    "modalities": ["image", "tabular"]},
    {"name": "image_text",       "modalities": ["image", "text"]},
    {"name": "full_multimodal",  "modalities": ["image", "text", "tabular"]},
]


def run_ablations(
    model_config: dict,
    train_loader,
    val_loader,
    test_loader,
    output_dir: str | Path = "outputs/ablations",
    max_epochs: int = 100,
    lr: float = 1e-4,
    configs: list[dict] | None = None,
    **train_kwargs,
) -> pd.DataFrame:
    """
    Run all ablation experiments and collect results.

    Args:
        model_config: Base model configuration dict.
        train_loader, val_loader, test_loader: DataLoaders.
        output_dir: Directory for saving per-ablation results.
        configs: Optional custom ablation configurations.

    Returns:
        DataFrame with columns: ablation, modalities, mae, rmse, r2, auc_broken.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = configs or ABLATION_CONFIGS
    results = []

    for abl in configs:
        name = abl["name"]
        modalities = abl["modalities"]
        abl_dir = output_dir / name

        logger.info("=" * 60)
        logger.info("Running ablation: %s (modalities: %s)", name, modalities)
        logger.info("=" * 60)

        # Build model with restricted modalities + train
        try:
            # Naive mean: skip model entirely, predict training-set mean
            if name == "naive_mean" or modalities is None:
                import numpy as np
                y_train = []
                for batch in train_loader:
                    y_train.append(batch["targets"][:, 0].numpy())
                mu = float(np.concatenate(y_train).mean())
                y_test, preds = [], []
                for batch in test_loader:
                    y = batch["targets"][:, 0].numpy()
                    y_test.append(y)
                    preds.append(np.full_like(y, mu))
                y_test = np.concatenate(y_test)
                preds = np.concatenate(preds)
                mae = float(np.mean(np.abs(y_test - preds)))
                results.append({
                    "ablation": name,
                    "modalities": "none",
                    "n_modalities": 0,
                    "test/underpricing_mae": mae,
                })
                logger.info("naive_mean MAE=%.4f", mae)
                continue

            model = MultimodalIPOModel(
                image_config=model_config.get("encoders", {}).get("image", {}),
                text_config=model_config.get("encoders", {}).get("text", {}),
                tabular_config=model_config.get("encoders", {}).get("tabular", {}),
                fusion_config=model_config.get("fusion", {}),
                heads_config=model_config.get("heads", {}),
                modalities=modalities,
            )
            trainer = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=str(abl_dir),
                max_epochs=max_epochs,
                lr=lr,
                accelerator="auto",
                **train_kwargs,
            )

            # Evaluate on test set
            test_results = trainer.test(dataloaders=test_loader)

            result_row = {
                "ablation": name,
                "modalities": ",".join(modalities) if modalities else "none",
                "n_modalities": len(modalities) if modalities else 0,
            }
            if test_results:
                result_row.update(test_results[0])

            results.append(result_row)

        except Exception as e:
            logger.error("Ablation %s failed: %s", name, e)
            results.append({
                "ablation": name,
                "modalities": ",".join(modalities) if modalities else "none",
                "error": str(e),
            })

    # Compile results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "ablation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Ablation results saved → %s", results_path)

    # Also save as JSON
    results_df.to_json(output_dir / "ablation_results.json", orient="records", indent=2)

    return results_df


def main():
    """CLI for running ablations."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/ablations")
    parser.add_argument("--max-epochs", type=int, default=100)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    from pathlib import Path

    cfg_path = Path(args.config)
    cfg_base = OmegaConf.load(cfg_path)
    
    # Manually load and merge hydra defaults for simplicity
    cfg_dir = cfg_path.parent
    data_cfg = OmegaConf.load(cfg_dir / "data/default.yaml")
    model_cfg = OmegaConf.load(cfg_dir / "model/cross_attention.yaml") 
    train_cfg = OmegaConf.load(cfg_dir / "training/default.yaml")

    cfg = OmegaConf.merge(cfg_base, {"data": data_cfg, "model": model_cfg, "training": train_cfg})

    from src.data.dataset import build_dataloaders
    loaders, datasets = build_dataloaders(
        universe_path="data/processed/ipo_sample/ipo_universe.parquet",
        batch_size=cfg.training.training.batch_size,
    )

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["encoders"]["tabular"]["input_dim"] = datasets["train"].get_tabular_dim()

    run_ablations(
        model_config=model_cfg,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
