"""
FinBERT Text Encoder — Encode S-1 Risk Factors into 256-d embeddings.

Supports:
  - Frozen FinBERT with learned projection
  - Optional fine-tuning with lower learning rate
  - CLS vs mean-pool strategies
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

logger = logging.getLogger(__name__)


class FinBERTTextEncoder(nn.Module):
    """
    FinBERT-based text encoder for financial filings.

    Takes raw text strings, tokenizes, encodes with FinBERT, and projects
    to a lower-dimensional embedding space.

    Args:
        model_name: HuggingFace model identifier.
        freeze: Whether to freeze FinBERT weights.
        pool_strategy: "cls" for [CLS] token, "mean" for mean pooling.
        proj_dim: Output projection dimension.
        max_length: Maximum token sequence length.
    """

    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone",
        freeze: bool = False,
        pool_strategy: Literal["cls", "mean"] = "cls",
        proj_dim: int = 256,
        max_length: int = 512,
    ):
        super().__init__()
        self.pool_strategy = pool_strategy
        self.max_length = max_length

        # Load FinBERT
        # yiyanghkust/finbert-tone is missing tokenizer_config.json and
        # model_type in config.json, breaking Auto* classes in
        # transformers ≥5.  Fall back to explicit Bert* classes.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except (ValueError, OSError):
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        try:
            self.encoder = AutoModel.from_pretrained(model_name)
        except (ValueError, OSError):
            self.encoder = BertModel.from_pretrained(model_name)
        embed_dim = self.encoder.config.hidden_size  # typically 768

        # Freeze if requested
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("FinBERT weights frozen")

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        logger.info(
            "FinBERTTextEncoder: %s → %d-d (pool=%s, frozen=%s)",
            model_name, proj_dim, pool_strategy, freeze,
        )

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a batch of texts."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(
        self,
        texts: list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode text to embeddings.

        Can accept either raw text strings (tokenized internally) or
        pre-tokenized input_ids + attention_mask.

        Returns:
            (batch_size, proj_dim) tensor
        """
        if texts is not None:
            tokens = self.tokenize(texts)
            input_ids = tokens["input_ids"].to(next(self.parameters()).device)
            attention_mask = tokens["attention_mask"].to(next(self.parameters()).device)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self.pool_strategy == "cls":
            pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        elif self.pool_strategy == "mean":
            # Mean pooling over non-padded tokens
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = (token_embeddings * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unknown pool strategy: {self.pool_strategy}")

        return self.projection(pooled)

    def get_output_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.projection[0].out_features

    def get_fine_tune_params(self, lr_encoder: float = 1e-5, lr_head: float = 1e-4):
        """Return parameter groups with differential learning rates."""
        return [
            {"params": self.encoder.parameters(), "lr": lr_encoder},
            {"params": self.projection.parameters(), "lr": lr_head},
        ]
