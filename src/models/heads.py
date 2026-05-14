"""
Multi-Task Prediction Heads — Underpricing, Broken IPO, Volatility.

Each head is a small MLP that takes the fused multimodal embedding
and produces a task-specific prediction.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RegressionHead(nn.Module):
    """
    2-layer MLP for regression targets (first-day return, volatility).

    Architecture: input → Linear → GELU → Dropout → Linear → output (scalar)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, input_dim) → (batch, 1)"""
        return self.head(x)


class ClassificationHead(nn.Module):
    """
    2-layer MLP for binary classification (broken IPO indicator).

    Architecture: input → Linear → GELU → Dropout → Linear → Sigmoid
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, input_dim) → (batch, 1) probabilities"""
        return self.head(x)


class MultiTaskHeads(nn.Module):
    """
    Combined multi-task prediction module.

    Manages underpricing (regression), broken IPO (classification),
    and volatility (regression) heads with configurable loss weights.
    """

    def __init__(
        self,
        input_dim: int = 256,
        underpricing_hidden: int = 128,
        broken_hidden: int = 128,
        volatility_hidden: int = 128,
        dropout: float = 0.2,
        underpricing_weight: float = 1.0,
        broken_weight: float = 0.5,
        volatility_weight: float = 0.3,
    ):
        super().__init__()
        self.underpricing_head = RegressionHead(input_dim, underpricing_hidden, dropout)
        self.broken_head = ClassificationHead(input_dim, broken_hidden, dropout)
        self.volatility_head = RegressionHead(input_dim, volatility_hidden, dropout)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        # Task weights
        self.weights = {
            "underpricing": underpricing_weight,
            "broken_ipo": broken_weight,
            "volatility": volatility_weight,
        }

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Generate predictions for all tasks.

        Args:
            z: (batch, input_dim) fused multimodal embedding.

        Returns:
            Dict with keys: underpricing, broken_ipo, volatility.
            Each value is (batch, 1).
        """
        return {
            "underpricing": self.underpricing_head(z),
            "broken_ipo": self.broken_head(z),
            "volatility": self.volatility_head(z),
        }

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute weighted multi-task loss.

        Args:
            predictions: Dict from forward().
            targets: (batch, 3) tensor with columns:
                     [first_day_return, broken_ipo, volatility_6m].

        Returns:
            total_loss: Weighted sum of task losses.
            loss_dict: Per-task losses (for logging).
        """
        loss_dict = {}
        total = torch.tensor(0.0, device=targets.device, requires_grad=True)

        # Underpricing regression
        if targets.shape[1] > 0:
            l_under = self.mse_loss(
                predictions["underpricing"].squeeze(-1),
                targets[:, 0],
            )
            loss_dict["underpricing"] = l_under.item()
            total = total + self.weights["underpricing"] * l_under

        # Broken IPO classification
        if targets.shape[1] > 1:
            l_broken = self.bce_loss(
                predictions["broken_ipo"].squeeze(-1),
                targets[:, 1],
            )
            loss_dict["broken_ipo"] = l_broken.item()
            total = total + self.weights["broken_ipo"] * l_broken

        # Volatility regression
        if targets.shape[1] > 2:
            # Only compute loss for non-NaN targets
            vol_mask = ~torch.isnan(targets[:, 2])
            if vol_mask.any():
                l_vol = self.mse_loss(
                    predictions["volatility"].squeeze(-1)[vol_mask],
                    targets[:, 2][vol_mask],
                )
                loss_dict["volatility"] = l_vol.item()
                total = total + self.weights["volatility"] * l_vol

        loss_dict["total"] = total.item()
        return total, loss_dict
