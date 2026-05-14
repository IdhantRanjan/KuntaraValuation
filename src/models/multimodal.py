"""
Full Multimodal Model — End-to-end image+text+tabular model for IPO valuation.

Combines:
  - CLIPImageEncoder (frozen ViT-L/14 + attention pooling)
  - FinBERTTextEncoder (optional fine-tuning)
  - TabularEncoder (learnable MLP)
  - Fusion module (late / gated / cross-attention)
  - MultiTaskHeads (underpricing, broken IPO, volatility)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from src.features.image_encoder import CLIPImageEncoder
from src.features.tabular_encoder import TabularEncoder
from src.features.text_encoder import FinBERTTextEncoder
from src.models.fusion import build_fusion
from src.models.heads import MultiTaskHeads

logger = logging.getLogger(__name__)


class MultimodalIPOModel(nn.Module):
    """
    Full multimodal architecture for IPO valuation prediction.

    Encodes images, text, and tabular data through modality-specific encoders,
    fuses them via a configurable fusion strategy, and produces multi-task
    predictions.

    Args:
        image_config: Dict with CLIPImageEncoder parameters.
        text_config: Dict with FinBERTTextEncoder parameters.
        tabular_config: Dict with TabularEncoder parameters.
        fusion_config: Dict with fusion strategy and parameters.
        heads_config: Dict with prediction head parameters.
        modalities: List of active modalities ("image", "text", "tabular").
    """

    def __init__(
        self,
        image_config: dict | None = None,
        text_config: dict | None = None,
        tabular_config: dict | None = None,
        fusion_config: dict | None = None,
        heads_config: dict | None = None,
        modalities: list[str] | None = None,
    ):
        super().__init__()

        self.active_modalities = modalities or ["image", "text", "tabular"]
        modality_dims = []

        # --- Image Encoder ---
        if "image" in self.active_modalities:
            cfg = image_config or {}
            self.image_encoder = CLIPImageEncoder(
                backbone=cfg.get("backbone", "ViT-L-14"),
                pretrained=cfg.get("pretrained", "openai"),
                freeze=cfg.get("freeze", True),
                proj_dim=cfg.get("proj_dim", 256),
                pool_method=cfg.get("pool_method", "attention"),
                attn_hidden=cfg.get("attn_hidden", 128),
            )
            modality_dims.append(self.image_encoder.get_output_dim())
        else:
            self.image_encoder = None

        # --- Text Encoder ---
        if "text" in self.active_modalities:
            cfg = text_config or {}
            self.text_encoder = FinBERTTextEncoder(
                model_name=cfg.get("backbone", "yiyanghkust/finbert-tone"),
                freeze=cfg.get("freeze", False),
                pool_strategy=cfg.get("pool_strategy", "cls"),
                proj_dim=cfg.get("proj_dim", 256),
                max_length=cfg.get("max_length", 512),
            )
            modality_dims.append(self.text_encoder.get_output_dim())
        else:
            self.text_encoder = None

        # --- Tabular Encoder ---
        if "tabular" in self.active_modalities:
            cfg = tabular_config or {}
            self.tabular_encoder = TabularEncoder(
                input_dim=cfg.get("input_dim", 8),
                hidden_dims=cfg.get("hidden_dims", [256, 256]),
                output_dim=cfg.get("output_dim", 256),
                activation=cfg.get("activation", "gelu"),
                dropout=cfg.get("dropout", 0.1),
            )
            modality_dims.append(self.tabular_encoder.get_output_dim())
        else:
            self.tabular_encoder = None

        # --- Fusion ---
        fcfg = fusion_config or {}
        strategy = fcfg.get("strategy", "cross_attention")

        # Extract strategy-specific kwargs
        if strategy == "cross_attention":
            transformer_cfg = fcfg.get("transformer", {})
            fusion_kwargs = {
                "d_model": transformer_cfg.get("d_model", 256),
                "n_layers": transformer_cfg.get("n_layers", 3),
                "n_heads": transformer_cfg.get("n_heads", 4),
                "d_ff": transformer_cfg.get("d_ff", 512),
                "dropout": transformer_cfg.get("dropout", 0.1),
                "use_fuse_token": transformer_cfg.get("use_fuse_token", True),
            }
        elif strategy == "gated":
            gated_cfg = fcfg.get("gated", {})
            fusion_kwargs = {
                "output_dim": gated_cfg.get("output_dim", 256),
                "gate_hidden": gated_cfg.get("gate_hidden", 64),
            }
        else:
            fusion_kwargs = {"output_dim": fcfg.get("output_dim", 256)}

        self.fusion = build_fusion(strategy, modality_dims, **fusion_kwargs)

        # --- Prediction Heads ---
        hcfg = heads_config or {}
        fusion_out_dim = getattr(self.fusion, "output_dim", 256)
        self.heads = MultiTaskHeads(
            input_dim=fusion_out_dim,
            underpricing_hidden=hcfg.get("underpricing", {}).get("hidden_dim", 128),
            broken_hidden=hcfg.get("broken_ipo", {}).get("hidden_dim", 128),
            volatility_hidden=hcfg.get("volatility", {}).get("hidden_dim", 128),
            dropout=hcfg.get("underpricing", {}).get("dropout", 0.2),
            underpricing_weight=hcfg.get("underpricing", {}).get("loss_weight", 1.0),
            broken_weight=hcfg.get("broken_ipo", {}).get("loss_weight", 0.5),
            volatility_weight=hcfg.get("volatility", {}).get("loss_weight", 0.3),
        )

        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "MultimodalIPOModel: %d params (%d trainable), modalities=%s, fusion=%s",
            n_params, n_trainable, self.active_modalities, strategy,
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            batch: Dict from IPOMultimodalDataset.__getitem__ with keys:
                   images, image_mask, text, tabular, targets.

        Returns:
            predictions: Dict with underpricing, broken_ipo, volatility tensors.
        """
        embeddings = []

        if self.image_encoder is not None:
            h_img = self.image_encoder(batch["images"], batch["image_mask"])
            embeddings.append(h_img)

        if self.text_encoder is not None:
            h_txt = self.text_encoder(texts=batch["text"])
            embeddings.append(h_txt)

        if self.tabular_encoder is not None:
            h_tab = self.tabular_encoder(batch["tabular"])
            embeddings.append(h_tab)

        # Fuse modalities
        z = self.fusion(embeddings)

        # Multi-task predictions
        predictions = self.heads(z)
        return predictions

    def compute_loss(
        self,
        batch: dict,
        predictions: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward + loss computation."""
        if predictions is None:
            predictions = self.forward(batch)
        return self.heads.compute_loss(predictions, batch["targets"])

    def get_parameter_groups(
        self,
        lr: float = 1e-4,
        encoder_lr: float = 1e-5,
    ) -> list[dict]:
        """
        Build parameter groups with differential learning rates.

        Frozen encoders get no LR; unfrozen encoders get a lower LR.
        """
        groups = []

        # Text encoder (potentially fine-tuned)
        if self.text_encoder is not None:
            groups.extend(self.text_encoder.get_fine_tune_params(encoder_lr, lr))

        # Image encoder projection (backbone is frozen)
        if self.image_encoder is not None:
            groups.append({
                "params": [p for p in self.image_encoder.parameters() if p.requires_grad],
                "lr": lr,
            })

        # Tabular, fusion, and heads
        other_params = []
        for module in [self.tabular_encoder, self.fusion, self.heads]:
            if module is not None:
                other_params.extend(module.parameters())
        groups.append({"params": other_params, "lr": lr})

        return groups


# ---------------------------------------------------------------------------
# Factory from config
# ---------------------------------------------------------------------------

def build_model_from_config(cfg: dict) -> MultimodalIPOModel:
    """Build the full model from a nested config dictionary."""
    encoders = cfg.get("encoders", {})
    fusion = cfg.get("fusion", {})
    heads = cfg.get("heads", {})
    pooling = cfg.get("image_pooling", {})

    # Merge image pooling into image encoder config
    image_cfg = {**encoders.get("image", {})}
    image_cfg["pool_method"] = pooling.get("method", "attention")
    image_cfg["attn_hidden"] = pooling.get("attn_hidden", 128)

    return MultimodalIPOModel(
        image_config=image_cfg,
        text_config=encoders.get("text", {}),
        tabular_config=encoders.get("tabular", {}),
        fusion_config=fusion,
        heads_config=heads,
    )
