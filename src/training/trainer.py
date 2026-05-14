"""
PyTorch Lightning Trainer — Main training loop for the multimodal IPO model.

Features:
  - Time-based data splits
  - Mixed-precision training
  - Early stopping on validation MAE
  - W&B logging (optional)
  - Gradient accumulation
  - Checkpoint management
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.multimodal import MultimodalIPOModel

logger = logging.getLogger(__name__)


class IPOMultimodalLitModule(pl.LightningModule):
    """
    PyTorch Lightning module wrapping the multimodal IPO model.

    Handles training, validation, and test steps with multi-task loss
    computation and per-task metric logging.
    """

    def __init__(
        self,
        model: MultimodalIPOModel,
        lr: float = 1e-4,
        encoder_lr: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        min_lr: float = 1e-6,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.encoder_lr = encoder_lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.max_epochs = max_epochs
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        return self.model(batch)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        """Shared logic for train/val/test steps."""
        predictions = self.forward(batch)
        total_loss, loss_dict = self.model.compute_loss(batch, predictions)

        # Log all losses
        for name, value in loss_dict.items():
            self.log(
                f"{stage}/{name}_loss",
                value,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=(name == "total"),
                batch_size=batch["tabular"].shape[0],
            )

        # Log MAE for underpricing (primary metric)
        if "underpricing" in predictions:
            mae = (
                predictions["underpricing"].squeeze(-1) - batch["targets"][:, 0]
            ).abs().mean()
            self.log(
                f"{stage}/underpricing_mae",
                mae,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch["tabular"].shape[0],
            )

        return total_loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        """AdamW with differential LR and cosine annealing."""
        param_groups = self.model.get_parameter_groups(
            lr=self.lr, encoder_lr=self.encoder_lr
        )

        optimizer = AdamW(
            param_groups,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def predict_step(self, batch: dict, batch_idx: int) -> dict[str, torch.Tensor]:
        """Generate predictions for inference."""
        return self.forward(batch)


def train(
    model: MultimodalIPOModel,
    train_loader,
    val_loader,
    output_dir: str | Path = "outputs",
    max_epochs: int = 100,
    lr: float = 1e-4,
    encoder_lr: float = 1e-5,
    gradient_clip_val: float = 1.0,
    patience: int = 10,
    precision: str = "16-mixed",
    accelerator: str = "auto",
    **kwargs,
) -> pl.Trainer:
    """
    Train the multimodal model with PyTorch Lightning.

    Returns the trainer for downstream evaluation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MPS doesn't support fp16 AMP reliably — fall back to fp32
    import torch
    if accelerator == "auto" and torch.backends.mps.is_available():
        precision = "32-true"

    # Wrap model in Lightning module
    lit_model = IPOMultimodalLitModule(
        model=model,
        lr=lr,
        encoder_lr=encoder_lr,
        max_epochs=max_epochs,
        **kwargs,
    )

    # Callbacks
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val/underpricing_mae",
            patience=patience,
            mode="min",
            verbose=True,
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best-{epoch}-{val/underpricing_mae:.4f}",
            monitor="val/underpricing_mae",
            mode="min",
            save_top_k=3,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        pl.callbacks.RichProgressBar(),
    ]

    # Logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=output_dir / "logs",
        name="ipo_multimodal",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=kwargs.get("gradient_accumulation_steps", 2),
        callbacks=callbacks,
        logger=tb_logger,
        deterministic=True,
        enable_progress_bar=True,
    )

    logger.info("Starting training for %d epochs", max_epochs)
    trainer.fit(lit_model, train_loader, val_loader)

    return trainer


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    """CLI entry for training."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Train multimodal IPO model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    # Load config
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    # Build data loaders
    from src.data.dataset import build_dataloaders
    loaders, datasets = build_dataloaders(
        universe_path="data/processed/ipo_sample/ipo_universe.parquet",
        batch_size=cfg.training.training.batch_size,
    )

    # Update tabular input dim from data
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["encoders"]["tabular"]["input_dim"] = datasets["train"].get_tabular_dim()

    # Build model
    from src.models.multimodal import build_model_from_config
    model = build_model_from_config(model_cfg)

    # Train
    train(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        output_dir=args.output_dir,
        max_epochs=cfg.training.training.max_epochs,
        lr=cfg.training.optimizer.lr,
        precision="16-mixed" if cfg.training.training.mixed_precision else "32",
    )


if __name__ == "__main__":
    main()
