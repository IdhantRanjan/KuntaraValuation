"""
Attribution Analysis — SHAP and Integrated Gradients for modality contributions.

Quantifies:
  - Per-modality contribution to predictions
  - Feature-level attributions within each modality
  - Cross-modality interaction effects
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def compute_modality_attributions(
    model,
    batch: dict,
    modalities: list[str] = ["image", "text", "tabular"],
) -> dict[str, float]:
    """
    Compute leave-one-out modality attributions.

    For each modality, zero out its embedding and measure the change
    in prediction to quantify its contribution.

    Returns dict mapping modality_name → contribution score.
    """
    model.eval()
    with torch.no_grad():
        # Full prediction
        full_pred = model(batch)
        full_underpricing = full_pred["underpricing"].squeeze(-1).cpu().numpy()

        attributions = {}
        for mod in modalities:
            # Create ablated batch
            ablated_batch = {k: v for k, v in batch.items()}

            if mod == "image":
                ablated_batch["images"] = torch.zeros_like(batch["images"])
                ablated_batch["image_mask"] = torch.zeros_like(batch["image_mask"])
            elif mod == "text":
                ablated_batch["text"] = [""] * len(batch["text"])
            elif mod == "tabular":
                ablated_batch["tabular"] = torch.zeros_like(batch["tabular"])

            ablated_pred = model(ablated_batch)
            ablated_underpricing = ablated_pred["underpricing"].squeeze(-1).cpu().numpy()

            # Attribution = change in prediction when modality is removed
            contribution = np.mean(np.abs(full_underpricing - ablated_underpricing))
            attributions[mod] = float(contribution)

    # Normalize to sum to 1
    total = sum(attributions.values()) + 1e-8
    attributions = {k: v / total for k, v in attributions.items()}

    return attributions


def compute_shap_values(
    model,
    dataset,
    n_background: int = 50,
    n_explain: int = 100,
    output_dir: str | Path = "outputs/analysis",
) -> dict:
    """
    Compute SHAP values for the tabular features using KernelSHAP.

    For the deep model, uses the tabular input as the explanation target
    while treating image and text as fixed context.

    Returns:
        Dict with shap_values (np.array), feature_names (list), base_value.
    """
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Define prediction function for tabular features
    def predict_fn(tabular_array: np.ndarray) -> np.ndarray:
        """Predict underpricing from tabular features only."""
        with torch.no_grad():
            tab_tensor = torch.tensor(tabular_array, dtype=torch.float32)
            if hasattr(model, "tabular_encoder") and model.tabular_encoder is not None:
                h_tab = model.tabular_encoder(tab_tensor)
                # Simple prediction from tabular only
                dummy_embeddings = [h_tab]
                z = model.fusion(dummy_embeddings)
                pred = model.heads.underpricing_head(z)
                return pred.squeeze(-1).cpu().numpy()
        return np.zeros(len(tabular_array))

    # Get background data
    background_data = []
    for i in range(min(n_background, len(dataset))):
        sample = dataset[i]
        background_data.append(sample["tabular"].numpy())
    background = np.array(background_data)

    # Get explanation data
    explain_data = []
    for i in range(min(n_explain, len(dataset))):
        sample = dataset[i]
        explain_data.append(sample["tabular"].numpy())
    explain = np.array(explain_data)

    # Compute SHAP values
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explain, nsamples=100)

    # Feature names
    feature_names = dataset.tabular_cols if hasattr(dataset, "tabular_cols") else [
        f"feature_{i}" for i in range(explain.shape[1])
    ]

    result = {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "base_value": float(explainer.expected_value),
        "explain_data": explain,
    }

    # Save
    np.save(output_dir / "shap_values.npy", shap_values)
    logger.info("SHAP values computed for %d samples", len(explain))

    return result


def compute_integrated_gradients(
    model,
    batch: dict,
    target_head: str = "underpricing",
    n_steps: int = 50,
) -> dict[str, torch.Tensor]:
    """
    Compute integrated gradients for each modality's input.

    Returns per-modality attribution tensors.
    """
    try:
        from captum.attr import IntegratedGradients

        model.eval()

        # Simplified IG on tabular features
        def forward_fn(tabular_input):
            modified_batch = {k: v for k, v in batch.items()}
            modified_batch["tabular"] = tabular_input
            preds = model(modified_batch)
            return preds[target_head].squeeze(-1)

        ig = IntegratedGradients(forward_fn)
        baseline = torch.zeros_like(batch["tabular"])

        attributions = ig.attribute(
            batch["tabular"],
            baselines=baseline,
            n_steps=n_steps,
            return_convergence_delta=False,
        )

        return {"tabular_attributions": attributions}

    except ImportError:
        logger.warning("captum not installed — skipping integrated gradients")
        return {}
