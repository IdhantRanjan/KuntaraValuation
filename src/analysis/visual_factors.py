"""
Visual Factor Analysis — Extract interpretable latent factors from image embeddings.

Constructs 2–5 latent visual factors via PCA or supervised bottlenecks and
relates them to economic concepts:
  - Asset tangibility / capital intensity
  - Digital interface richness / SaaS-ness
  - Operational complexity / geographic scope
  - Product sophistication
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def extract_image_embeddings(
    model,
    dataset,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract firm-level image embeddings from the trained model.

    Returns:
        embeddings: (N, proj_dim) array
        ciks: List of CIK identifiers
    """
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    device = next(model.parameters()).device

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    all_ciks = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            mask = batch["image_mask"].to(device)

            if model.image_encoder is not None:
                emb = model.image_encoder(images, mask)
                all_embeddings.append(emb.cpu().numpy())

            all_ciks.extend(batch["cik"])

    if all_embeddings:
        embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        embeddings = np.zeros((len(all_ciks), 256))

    logger.info("Extracted image embeddings: %s", embeddings.shape)
    return embeddings, all_ciks


def compute_visual_factors(
    embeddings: np.ndarray,
    n_factors: int = 5,
    method: str = "pca",
) -> tuple[np.ndarray, PCA]:
    """
    Reduce image embeddings to interpretable latent factors.

    Args:
        embeddings: (N, d) array of firm-level image embeddings.
        n_factors: Number of factors to extract.
        method: "pca" (default).

    Returns:
        factors: (N, n_factors) factor scores
        decomposer: Fitted PCA object (for interpretation)
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    if method == "pca":
        pca = PCA(n_components=n_factors, random_state=42)
        factors = pca.fit_transform(scaled)

        logger.info(
            "PCA visual factors: %d components, explained variance = %.1f%%",
            n_factors,
            pca.explained_variance_ratio_.sum() * 100,
        )
        for i, ev in enumerate(pca.explained_variance_ratio_):
            logger.info("  Factor %d: %.1f%% variance", i + 1, ev * 100)

        return factors, pca
    else:
        raise ValueError(f"Unknown method: {method}")


def interpret_factors(
    factors: np.ndarray,
    df: pd.DataFrame,
    economic_vars: list[str] | None = None,
) -> pd.DataFrame:
    """
    Correlate visual factors with known economic variables.

    Tests hypotheses:
      H2: Tangibility factors predict lower volatility/beta.
      H3: Visual factors explain private valuations.

    Args:
        factors: (N, n_factors) factor scores.
        df: IPO universe DataFrame aligned with factors.
        economic_vars: Columns in df to correlate with.

    Returns:
        Correlation matrix with p-values.
    """
    from scipy import stats

    if economic_vars is None:
        economic_vars = [
            "first_day_return", "broken_ipo",
            "post_ipo_volatility_6m", "post_ipo_volatility_12m",
            "log_assets", "leverage", "rnd_intensity",
        ]

    available_vars = [v for v in economic_vars if v in df.columns]

    results = []
    for f_idx in range(factors.shape[1]):
        factor_scores = factors[:, f_idx]

        for var_name in available_vars:
            var_values = df[var_name].values

            # Remove NaN pairs
            valid = ~(np.isnan(factor_scores) | np.isnan(var_values))
            if valid.sum() < 10:
                continue

            corr, pval = stats.pearsonr(
                factor_scores[valid],
                var_values[valid],
            )

            results.append({
                "factor": f"VF{f_idx + 1}",
                "variable": var_name,
                "correlation": corr,
                "p_value": pval,
                "n_obs": int(valid.sum()),
                "significant": pval < 0.05,
            })

    results_df = pd.DataFrame(results)
    logger.info("Factor-variable correlations:\n%s", results_df.to_string())
    return results_df


def top_loading_images(
    pca: PCA,
    embeddings: np.ndarray,
    image_manifest: dict[str, list[str]],
    ciks: list[str],
    factor_idx: int = 0,
    top_k: int = 5,
) -> list[dict]:
    """
    Find images with the highest/lowest loadings on a given factor.

    Useful for interpreting what each visual factor captures.
    """
    factor_scores = pca.transform(
        StandardScaler().fit_transform(embeddings)
    )[:, factor_idx]

    # Top positive loadings
    top_pos = np.argsort(factor_scores)[-top_k:][::-1]
    # Top negative loadings
    top_neg = np.argsort(factor_scores)[:top_k]

    results = []
    for label, indices in [("high", top_pos), ("low", top_neg)]:
        for idx in indices:
            cik = ciks[idx]
            images = image_manifest.get(cik, [])
            results.append({
                "factor": f"VF{factor_idx + 1}",
                "direction": label,
                "cik": cik,
                "score": float(factor_scores[idx]),
                "images": images[:3],  # Show up to 3 images
            })

    return results


def run_visual_factor_analysis(
    model,
    dataset,
    df: pd.DataFrame,
    image_manifest: dict[str, list[str]],
    n_factors: int = 5,
    output_dir: str | Path = "outputs/analysis",
) -> dict:
    """Full visual factor analysis pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract embeddings
    embeddings, ciks = extract_image_embeddings(model, dataset)

    # Compute factors
    factors, pca = compute_visual_factors(embeddings, n_factors)

    # Save factors
    factors_df = pd.DataFrame(
        factors,
        columns=[f"VF{i+1}" for i in range(n_factors)],
    )
    factors_df["cik"] = ciks
    factors_df.to_csv(output_dir / "visual_factors.csv", index=False)

    # Correlate with economic variables
    aligned_df = df[df["cik"].astype(str).isin(ciks)].reset_index(drop=True)
    correlations = interpret_factors(factors, aligned_df)
    correlations.to_csv(output_dir / "factor_correlations.csv", index=False)

    # Top-loading images for each factor
    import json
    all_top_images = []
    for f in range(n_factors):
        top = top_loading_images(pca, embeddings, image_manifest, ciks, f)
        all_top_images.extend(top)

    with open(output_dir / "top_loading_images.json", "w") as f:
        json.dump(all_top_images, f, indent=2)

    return {
        "factors": factors,
        "pca": pca,
        "correlations": correlations,
        "top_images": all_top_images,
    }
