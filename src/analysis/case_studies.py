"""
Case Study Analysis — Attention-weight visualization for exemplar IPOs.

For high-profile IPOs (e.g., Airbnb, Coinbase, DoorDash), visualizes:
  - Which images received the highest attention weights
  - How text/image/tabular modalities interact
  - How predictions align with actual outcomes
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def extract_attention_weights(
    model,
    batch: dict,
) -> dict[str, np.ndarray]:
    """
    Extract attention weights from the model for interpretability.

    Returns:
        Dict with:
          - image_attention: (batch, max_images) weights from image pooling
          - fusion_attention: (batch, n_layers, n_heads, seq_len, seq_len)
                              (if cross-attention fusion)
    """
    model.eval()
    results = {}

    with torch.no_grad():
        # Image attention weights (from attention pooling)
        if model.image_encoder is not None:
            images = batch["images"]
            mask = batch["image_mask"]
            device = next(model.parameters()).device
            images = images.to(device)
            mask = mask.to(device)

            # Get per-image embeddings
            B, N, C, H, W = images.shape
            flat = images.view(B * N, C, H, W)
            flat_mask = mask.view(B * N)
            valid = flat_mask.nonzero(as_tuple=True)[0]

            if len(valid) > 0 and hasattr(model.image_encoder, 'pooler'):
                valid_imgs = flat[valid]
                valid_embs = model.image_encoder.encode_single_images(valid_imgs)

                d = valid_embs.shape[-1]
                all_embs = torch.zeros(B * N, d, device=device)
                all_embs[valid] = valid_embs
                all_embs = all_embs.view(B, N, d)

                # Get attention scores from pooler
                pooler = model.image_encoder.pooler
                if pooler is not None:
                    scores = pooler.attention(all_embs).squeeze(-1)
                    scores = scores.masked_fill(~mask.to(device), -1e9)
                    weights = torch.softmax(scores, dim=1)
                    results["image_attention"] = weights.cpu().numpy()

    return results


def visualize_case_study(
    cik: str,
    ticker: str,
    images: list[str],
    attention_weights: np.ndarray,
    prediction: float,
    actual: float,
    output_dir: str | Path = "outputs/case_studies",
) -> None:
    """
    Create a visual case study for a single IPO.

    Layout:
      - Top row: images ranked by attention weight
      - Bottom: prediction vs actual + key info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_images = min(len(images), 6)
    if n_images == 0:
        logger.warning("No images for %s (%s)", cik, ticker)
        return

    # Sort by attention weight
    sorted_idx = np.argsort(attention_weights[:n_images])[::-1]

    fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))
    if n_images == 1:
        axes = [axes]

    for i, idx in enumerate(sorted_idx[:n_images]):
        try:
            img = Image.open(images[idx])
            axes[i].imshow(img)
            axes[i].set_title(
                f"Attention: {attention_weights[idx]:.3f}",
                fontsize=10, fontweight="bold",
            )
        except Exception:
            axes[i].text(0.5, 0.5, "Image\nunavailable", ha="center", va="center")
        axes[i].axis("off")

    fig.suptitle(
        f"{ticker} (CIK: {cik})\n"
        f"Predicted: {prediction:.2%} | Actual: {actual:.2%}",
        fontsize=14, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_dir / f"{ticker}_case_study.pdf")
    plt.close(fig)
    logger.info("Case study saved for %s", ticker)


def run_case_studies(
    model,
    dataset,
    exemplar_tickers: list[str] | None = None,
    output_dir: str | Path = "outputs/case_studies",
) -> list[dict]:
    """
    Generate case studies for exemplar IPOs.

    Default exemplars: popular recent tech IPOs.
    """
    if exemplar_tickers is None:
        exemplar_tickers = [
            "ABNB",   # Airbnb
            "COIN",   # Coinbase
            "DASH",   # DoorDash
            "RIVN",   # Rivian
            "SNOW",   # Snowflake
            "PLTR",   # Palantir
            "RBLX",   # Roblox
            "PATH",   # UiPath
        ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(len(dataset)):
        sample = dataset[i]
        ticker = sample.get("ticker", "")

        if ticker not in exemplar_tickers:
            continue

        logger.info("Generating case study for %s", ticker)

        # Get attention weights
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]
                 for k, v in sample.items()}
        attention = extract_attention_weights(model, batch)

        img_attn = attention.get("image_attention", np.ones(16) / 16)[0]
        img_paths = dataset.image_manifest.get(sample["cik"], [])

        # Get prediction
        model.eval()
        with torch.no_grad():
            preds = model(batch)

        predicted = preds["underpricing"].item()
        actual = sample["targets"][0].item() if len(sample["targets"]) > 0 else 0.0

        # Visualize
        visualize_case_study(
            cik=sample["cik"],
            ticker=ticker,
            images=img_paths,
            attention_weights=img_attn,
            prediction=predicted,
            actual=actual,
            output_dir=output_dir,
        )

        results.append({
            "ticker": ticker,
            "cik": sample["cik"],
            "predicted": predicted,
            "actual": actual,
            "top_attention_image": img_paths[np.argmax(img_attn)] if img_paths else None,
        })

    # Save summary
    summary_path = output_dir / "case_study_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
