"""
Tabular Encoder — MLP for pre-IPO financial ratios and control variables.

Encodes standardized numeric features into a fixed-dimension embedding
suitable for fusion with text and image modalities.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TabularEncoder(nn.Module):
    """
    Multi-layer perceptron encoder for tabular financial features.

    Architecture: input → [Linear → LayerNorm → GELU → Dropout] × L → output

    Args:
        input_dim: Number of input features (financial ratios + controls).
        hidden_dims: List of hidden layer widths.
        output_dim: Final embedding dimension.
        activation: Activation function name ("gelu" | "relu" | "silu").
        dropout: Dropout probability.
        use_layernorm: Whether to apply layer normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        output_dim: int = 256,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        act_fn = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }[activation.lower()]

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final projection to output_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(output_dim))

        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "TabularEncoder: %d → %s → %d (%d params)",
            input_dim, hidden_dims, output_dim, n_params,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) standardized tabular features

        Returns:
            (batch_size, output_dim) embedding
        """
        return self.mlp(x)

    def get_output_dim(self) -> int:
        return self.output_dim
