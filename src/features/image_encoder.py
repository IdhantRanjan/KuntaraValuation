"""
CLIP ViT Image Encoder — Encode operational images into firm-level embeddings.

Pipeline:
  1. Per-image encoding with frozen CLIP ViT-L/14
  2. Linear projection to 256-d
  3. Attention-weighted mean pooling across firm images
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AttentionPooling(nn.Module):
    """
    Learned attention-weighted mean pooling over a set of image embeddings.

    For each firm, computes scalar attention weights over its N images and
    returns a single weighted-average embedding.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, max_images, embed_dim)
            mask: (batch, max_images) boolean mask of valid images

        Returns:
            pooled: (batch, embed_dim)
        """
        # Compute raw attention scores
        scores = self.attention(embeddings).squeeze(-1)  # (batch, max_images)

        # Mask invalid images with large negative value
        scores = scores.masked_fill(~mask, -1e9)

        # Softmax over valid images
        weights = torch.softmax(scores, dim=1)  # (batch, max_images)

        # Weighted sum
        pooled = (weights.unsqueeze(-1) * embeddings).sum(dim=1)  # (batch, embed_dim)

        return pooled


class CLIPImageEncoder(nn.Module):
    """
    CLIP ViT-L/14 image encoder with attention-weighted firm-level pooling.

    Args:
        backbone: CLIP model variant name (for open_clip).
        pretrained: Pretrained weights designation.
        freeze: Whether to freeze CLIP backbone weights.
        proj_dim: Dimension of projected embeddings.
        pool_method: "attention" | "mean" | "max".
        attn_hidden: Hidden dim for attention pooling MLP.
    """

    def __init__(
        self,
        backbone: str = "ViT-L-14",
        pretrained: str = "openai",
        freeze: bool = True,
        proj_dim: int = 256,
        pool_method: str = "attention",
        attn_hidden: int = 128,
    ):
        super().__init__()
        self.pool_method = pool_method

        # Load CLIP image encoder
        import open_clip
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            backbone, pretrained=pretrained
        )
        self.visual = clip_model.visual
        embed_dim = clip_model.visual.output_dim

        if freeze:
            for param in self.visual.parameters():
                param.requires_grad = False
            logger.info("CLIP ViT weights frozen")

        # Projection
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # Attention pooling
        if pool_method == "attention":
            self.pooler = AttentionPooling(proj_dim, attn_hidden)
        else:
            self.pooler = None

        logger.info(
            "CLIPImageEncoder: %s (%s) → %d-d (pool=%s, frozen=%s)",
            backbone, pretrained, proj_dim, pool_method, freeze,
        )

    def encode_single_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode individual images (not yet pooled across a firm).

        Args:
            images: (N, C, H, W) tensor of preprocessed images

        Returns:
            (N, proj_dim) embeddings
        """
        with torch.set_grad_enabled(self.visual.training):
            features = self.visual(images)  # (N, embed_dim)
        return self.projection(features)

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode and pool images per firm.

        Args:
            images: (batch, max_images, C, H, W) tensor
            mask: (batch, max_images) boolean mask of valid images

        Returns:
            firm_embeddings: (batch, proj_dim)
        """
        batch_size, max_images, C, H, W = images.shape

        # Flatten batch and image dimensions
        flat_images = images.view(batch_size * max_images, C, H, W)
        flat_mask = mask.view(batch_size * max_images)

        # Only encode valid images to save compute
        valid_indices = flat_mask.nonzero(as_tuple=True)[0]

        if len(valid_indices) == 0:
            # No valid images → return zeros
            proj_dim = self.projection[0].out_features
            return torch.zeros(batch_size, proj_dim, device=images.device)

        valid_images = flat_images[valid_indices]
        valid_embeddings = self.encode_single_images(valid_images)

        # Scatter back to full (batch * max_images) tensor
        proj_dim = valid_embeddings.shape[-1]
        all_embeddings = torch.zeros(
            batch_size * max_images, proj_dim,
            device=images.device, dtype=valid_embeddings.dtype,
        )
        all_embeddings[valid_indices] = valid_embeddings

        # Reshape to (batch, max_images, proj_dim)
        all_embeddings = all_embeddings.view(batch_size, max_images, proj_dim)

        # Pool across images per firm
        if self.pool_method == "attention":
            return self.pooler(all_embeddings, mask)
        elif self.pool_method == "mean":
            mask_f = mask.unsqueeze(-1).float()
            return (all_embeddings * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        elif self.pool_method == "max":
            all_embeddings[~mask.unsqueeze(-1).expand_as(all_embeddings)] = -1e9
            return all_embeddings.max(dim=1).values
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")

    def get_output_dim(self) -> int:
        return self.projection[0].out_features
