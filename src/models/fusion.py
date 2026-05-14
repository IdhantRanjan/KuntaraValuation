"""
Multimodal Fusion Strategies — Late, Gated, and Cross-Attention Fusion.

Implements three fusion approaches for combining image, text, and tabular
modality embeddings, following Tavakoli et al. and MMPFN architectures
adapted for the IPO valuation setting.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Late Fusion — Concatenation + MLP
# ---------------------------------------------------------------------------

class LateFusion(nn.Module):
    """
    Simple late fusion: concatenate modality embeddings and pass through MLP.

    This is the baseline fusion strategy. Effective when modality interactions
    are minimal or when data is limited.
    """

    def __init__(self, modality_dims: list[int], output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        total_dim = sum(modality_dims)
        self.fuse = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: List of (batch, dim_i) tensors, one per modality.

        Returns:
            (batch, output_dim) fused embedding.
        """
        concatenated = torch.cat(embeddings, dim=-1)
        return self.fuse(concatenated)


# ---------------------------------------------------------------------------
# Gated Additive Fusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """
    Gated additive fusion: learn per-modality gate weights and compute
    a weighted sum of modality embeddings.

    Each modality embedding is projected to a common dimension, then
    gated by a learnable scalar weight in [0, 1].
    """

    def __init__(
        self,
        modality_dims: list[int],
        output_dim: int = 256,
        gate_hidden: int = 64,
    ):
        super().__init__()
        self.n_modalities = len(modality_dims)
        self.output_dim = output_dim

        # Per-modality projection to common dim
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
            )
            for dim in modality_dims
        ])

        # Gate network: takes concatenated projected embeddings → per-modality weight
        self.gate = nn.Sequential(
            nn.Linear(output_dim * self.n_modalities, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, self.n_modalities),
            nn.Sigmoid(),  # Gate weights ∈ [0, 1]
        )

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: List of (batch, dim_i) tensors, one per modality.

        Returns:
            (batch, output_dim) fused embedding.
        """
        # Project each modality to common dimension
        projected = [proj(emb) for proj, emb in zip(self.projections, embeddings)]
        stacked = torch.stack(projected, dim=1)  # (batch, n_mod, output_dim)

        # Compute gate weights
        concatenated = torch.cat(projected, dim=-1)  # (batch, n_mod * output_dim)
        gates = self.gate(concatenated)  # (batch, n_mod)
        gates = gates.unsqueeze(-1)  # (batch, n_mod, 1)

        # Weighted sum
        fused = (gates * stacked).sum(dim=1)  # (batch, output_dim)
        return fused


# ---------------------------------------------------------------------------
# Cross-Attention Transformer Fusion
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Transformer-based cross-attention fusion with a learnable [FUSE] token.

    Each modality embedding is treated as a token with an added modality-type
    embedding. A small transformer encoder processes the sequence, and the
    [FUSE] token's output becomes the fused representation.

    This mirrors the Tavakoli et al. cross-attention approach and MMPFN's
    modality projector, adapted for the three-modality IPO setting.

    Args:
        modality_dims: Per-modality input dimensions.
        d_model: Transformer hidden dimension.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads.
        d_ff: Feed-forward dimension.
        dropout: Dropout rate.
        use_fuse_token: Use a dedicated [FUSE] token (True) or pool over
                        modality tokens (False).
    """

    def __init__(
        self,
        modality_dims: list[int],
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        use_fuse_token: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_modalities = len(modality_dims)
        self.use_fuse_token = use_fuse_token

        # Per-modality input projections
        self.input_projections = nn.ModuleList([
            nn.Linear(dim, d_model) for dim in modality_dims
        ])

        # Learnable modality-type embeddings
        n_tokens = self.n_modalities + (1 if use_fuse_token else 0)
        self.modality_embeddings = nn.Embedding(n_tokens, d_model)

        # Learnable [FUSE] token
        if use_fuse_token:
            self.fuse_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Final layer norm
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dim = d_model

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "CrossAttentionFusion: %d modalities → %d-d, %d layers, %d heads (%d params)",
            self.n_modalities, d_model, n_layers, n_heads, n_params,
        )

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: List of (batch, dim_i) tensors, one per modality.

        Returns:
            (batch, d_model) fused embedding from [FUSE] token.
        """
        batch_size = embeddings[0].shape[0]
        device = embeddings[0].device

        # Project each modality to d_model and add modality embeddings
        tokens = []
        for i, (proj, emb) in enumerate(zip(self.input_projections, embeddings)):
            projected = proj(emb)  # (batch, d_model)
            mod_emb = self.modality_embeddings(
                torch.tensor([i], device=device)
            )  # (1, d_model)
            tokens.append(projected + mod_emb)

        if self.use_fuse_token:
            # Add [FUSE] token
            fuse = self.fuse_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
            fuse_mod_emb = self.modality_embeddings(
                torch.tensor([self.n_modalities], device=device)
            )
            fuse = fuse + fuse_mod_emb
            tokens.insert(0, fuse.squeeze(1))  # Position 0

        # Stack into sequence: (batch, seq_len, d_model)
        sequence = torch.stack(tokens, dim=1)

        # Apply transformer
        output = self.transformer(sequence)

        # Extract fused representation
        if self.use_fuse_token:
            fused = output[:, 0, :]  # [FUSE] token at position 0
        else:
            fused = output.mean(dim=1)  # Mean pool over modality tokens

        return self.output_norm(fused)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_fusion(
    strategy: str,
    modality_dims: list[int],
    **kwargs,
) -> nn.Module:
    """
    Build a fusion module from config.

    Args:
        strategy: "late" | "gated" | "cross_attention"
        modality_dims: Per-modality embedding dimensions.
        **kwargs: Additional arguments for the specific fusion class.
    """
    strategies = {
        "late": LateFusion,
        "gated": GatedFusion,
        "cross_attention": CrossAttentionFusion,
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown fusion strategy: {strategy}. Options: {list(strategies.keys())}")

    logger.info("Building %s fusion with modality dims %s", strategy, modality_dims)
    return strategies[strategy](modality_dims=modality_dims, **kwargs)
