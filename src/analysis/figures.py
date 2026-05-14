"""
Figure Generation — Publication-quality figures for the paper.

Generates:
  - Figure 1: Modality ablation bar chart (R², MAE, RMSE across ablations)
  - Calibration plots (predicted vs actual underpricing by decile)
  - SHAP summary plots
  - Visual factor correlation heatmaps
  - Attention weight visualizations for case studies
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Modality colors
COLORS = {
    "tabular_only": "#636EFA",
    "text_only": "#EF553B",
    "image_only": "#00CC96",
    "text_tabular": "#AB63FA",
    "image_tabular": "#FFA15A",
    "image_text": "#19D3F3",
    "full_multimodal": "#FF6692",
}

LABELS = {
    "tabular_only": "Financials\nOnly",
    "text_only": "S-1 Text\nOnly",
    "image_only": "Images\nOnly",
    "text_tabular": "Text +\nFinancials",
    "image_tabular": "Images +\nFinancials",
    "image_text": "Images +\nText",
    "full_multimodal": "Full\nMultimodal",
}


def figure1_ablation_bars(
    results_df: pd.DataFrame,
    metric: str = "r2",
    output_path: str | Path = "outputs/figures/figure1_ablation.pdf",
    figsize: tuple = (10, 5),
) -> None:
    """
    Figure 1: Bar chart of out-of-sample performance across modality ablations.

    This is the "killer figure" showing incremental value of each modality.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Order ablations logically
    order = [
        "tabular_only", "text_only", "image_only",
        "text_tabular", "image_tabular", "image_text",
        "full_multimodal",
    ]
    available = [a for a in order if a in results_df["ablation"].values]

    fig, ax = plt.subplots(figsize=figsize)

    values = []
    colors = []
    labels = []
    for abl in available:
        row = results_df[results_df["ablation"] == abl].iloc[0]
        val = row.get(metric, row.get(f"test/{metric}", 0))
        values.append(val)
        colors.append(COLORS.get(abl, "#999999"))
        labels.append(LABELS.get(abl, abl))

    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate bars with values
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(metric.upper() if metric == "r2" else metric.upper())
    ax.set_title("Out-of-Sample Performance by Modality Configuration", fontweight="bold")

    # Highlight full multimodal bar
    if "full_multimodal" in available:
        idx = available.index("full_multimodal")
        bars[idx].set_edgecolor("black")
        bars[idx].set_linewidth(2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Figure 1 saved → %s", output_path)


def calibration_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    output_path: str | Path = "outputs/figures/calibration.pdf",
) -> None:
    """Predicted vs actual underpricing by decile."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"predicted": y_pred, "realized": y_true})
    df["decile"] = pd.qcut(df["predicted"], n_bins, labels=False, duplicates="drop")

    agg = df.groupby("decile").agg(
        mean_predicted=("predicted", "mean"),
        mean_realized=("realized", "mean"),
        se_realized=("realized", lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        agg["mean_predicted"], agg["mean_realized"],
        yerr=1.96 * agg["se_realized"],
        fmt="o-", capsize=3, color="#636EFA", markersize=8, label="Decile means",
    )

    # 45-degree line
    lims = [
        min(agg["mean_predicted"].min(), agg["mean_realized"].min()) - 0.02,
        max(agg["mean_predicted"].max(), agg["mean_realized"].max()) + 0.02,
    ]
    ax.plot(lims, lims, "k--", alpha=0.4, label="Perfect calibration")

    ax.set_xlabel("Mean Predicted First-Day Return")
    ax.set_ylabel("Mean Realized First-Day Return")
    ax.set_title("Calibration: Predicted vs Realized Underpricing", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Calibration plot saved → %s", output_path)


def shap_summary_plot(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_data: np.ndarray,
    output_path: str | Path = "outputs/figures/shap_summary.pdf",
) -> None:
    """SHAP beeswarm summary plot for tabular features."""
    import shap

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        feature_data,
        feature_names=feature_names,
        show=False,
    )
    fig = plt.gcf()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("SHAP summary plot saved → %s", output_path)


def factor_correlation_heatmap(
    correlations_df: pd.DataFrame,
    output_path: str | Path = "outputs/figures/factor_correlations.pdf",
) -> None:
    """Heatmap of visual factors vs economic variables."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = correlations_df.pivot(
        index="factor", columns="variable", values="correlation"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-0.5, vmax=0.5, ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_title("Visual Factor Correlations with Economic Variables", fontweight="bold")
    ax.set_ylabel("Visual Factor")
    ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Factor correlation heatmap saved → %s", output_path)


def modality_contribution_pie(
    attributions: dict[str, float],
    output_path: str | Path = "outputs/figures/modality_contributions.pdf",
) -> None:
    """Pie chart of modality contributions from leave-one-out analysis."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {"image": "#00CC96", "text": "#EF553B", "tabular": "#636EFA"}

    fig, ax = plt.subplots(figsize=(6, 6))
    labels = list(attributions.keys())
    sizes = list(attributions.values())
    pie_colors = [colors.get(l, "#999999") for l in labels]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=[l.capitalize() for l in labels],
        autopct="%1.1f%%", colors=pie_colors,
        textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")

    ax.set_title("Modality Contributions to IPO Underpricing Prediction", fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Modality contribution pie saved → %s", output_path)


def generate_all_figures(
    ablation_results_path: str | Path = "outputs/ablations/ablation_results.csv",
    output_dir: str | Path = "outputs/figures",
) -> None:
    """Generate all publication figures from saved results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Ablation bars
    if Path(ablation_results_path).exists():
        df = pd.read_csv(ablation_results_path)
        for metric in ["underpricing_mae", "underpricing_loss", "total_loss", "volatility_loss"]:
            if metric in df.columns or f"test/{metric}" in df.columns:
                figure1_ablation_bars(
                    df, metric,
                    output_path=output_dir / f"figure1_{metric}.pdf",
                )

    logger.info("All figures generated in %s", output_dir)


def main():
    """CLI for figure generation."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--ablation-results", default="outputs/ablations/ablation_results.csv")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    generate_all_figures(args.ablation_results, args.output_dir)


if __name__ == "__main__":
    main()
