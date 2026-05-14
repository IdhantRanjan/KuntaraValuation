"""
Full Evaluation Pipeline — Load checkpoint, run all metrics, generate figures.

Usage:
    python -m src.evaluation.run_eval \
        --checkpoint outputs/checkpoints/best.ckpt \
        --config configs/config.yaml \
        --output-dir outputs/eval

If --checkpoint is omitted, run_eval will look for any *.csv saved
predictions in outputs/predictions and synthesize a metric report from
those, which is useful for evaluating baselines without re-running the
multimodal model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import (
    classification_metrics,
    decile_analysis,
    diebold_mariano_test,
    regression_metrics,
)
from src.evaluation.statistical_tests import bootstrap_r2_difference

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _load_checkpoint_and_predict(
    checkpoint_path: Path,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load a Lightning checkpoint and run inference over the test loader.

    Returns (y_true_underpricing, y_pred_underpricing, y_pred_broken_proba).
    """
    import torch
    import pytorch_lightning as pl  # noqa: F401

    from src.training.trainer import IPOMultimodalLitModule
    from src.data.dataset import build_dataloaders

    data_cfg = config.get("data", {})
    universe_path = data_cfg.get(
        "universe_parquet",
        "data/processed/ipo_sample/ipo_universe.parquet",
    )

    loaders = build_dataloaders(
        universe_path=universe_path,
        train_end=data_cfg.get("train_end", "2018-12-31"),
        val_start=data_cfg.get("val_start", "2019-01-01"),
        val_end=data_cfg.get("val_end", "2020-12-31"),
        test_start=data_cfg.get("test_start", "2021-01-01"),
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=0,
    )
    test_loader = loaders["test"] if isinstance(loaders, dict) else loaders[2]

    lit = IPOMultimodalLitModule.load_from_checkpoint(str(checkpoint_path))
    lit.eval()

    y_true: list[np.ndarray] = []
    y_pred_under: list[np.ndarray] = []
    y_pred_broken: list[np.ndarray] = []

    with torch.no_grad():
        for batch in test_loader:
            preds = lit(batch)
            if "underpricing" in preds:
                y_pred_under.append(preds["underpricing"].squeeze(-1).cpu().numpy())
                y_true.append(batch["targets"][:, 0].cpu().numpy())
            if "broken_logits" in preds:
                logits = preds["broken_logits"].squeeze(-1)
                y_pred_broken.append(torch.sigmoid(logits).cpu().numpy())
            elif "broken" in preds:
                y_pred_broken.append(preds["broken"].squeeze(-1).cpu().numpy())

    y_true_arr = np.concatenate(y_true) if y_true else np.array([])
    y_pred_under_arr = np.concatenate(y_pred_under) if y_pred_under else np.array([])
    y_pred_broken_arr = (
        np.concatenate(y_pred_broken) if y_pred_broken else None
    )
    return y_true_arr, y_pred_under_arr, y_pred_broken_arr


# ---------------------------------------------------------------------------
# Predictions-CSV fallback
# ---------------------------------------------------------------------------

def _load_predictions_csv(predictions_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all *.csv prediction files, keyed by stem name."""
    out: dict[str, pd.DataFrame] = {}
    if not predictions_dir.exists():
        return out
    for f in sorted(predictions_dir.glob("*.csv")):
        try:
            df = pd.read_csv(f)
            if {"y_true", "y_pred"}.issubset(df.columns):
                out[f.stem] = df
        except Exception as e:
            logger.warning("Skipping %s: %s", f, e)
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_summary(name: str, metrics: dict) -> None:
    logger.info("--- %s ---", name)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info("  %-20s %.4f", k, v)
        else:
            logger.info("  %-20s %s", k, v)


def _save_summary(
    summary: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=float)

    rows = []
    for name, m in summary.items():
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                rows.append({"model": name, "metric": k, "value": float(v)})
    pd.DataFrame(rows).to_csv(output_dir / "metrics_summary.csv", index=False)


def run_eval(
    checkpoint: Path | None,
    config_path: Path,
    output_dir: Path = Path("outputs/eval"),
) -> dict:
    """
    Top-level evaluation entry. Returns a summary dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        with config_path.open() as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning("Config %s missing — using defaults", config_path)
        config = {}

    summary: dict = {}

    # ---- 1. Try checkpoint inference ----
    multimodal_pred: np.ndarray | None = None
    multimodal_true: np.ndarray | None = None
    broken_proba: np.ndarray | None = None
    if checkpoint and Path(checkpoint).exists():
        try:
            multimodal_true, multimodal_pred, broken_proba = (
                _load_checkpoint_and_predict(Path(checkpoint), config)
            )
            logger.info("Inference complete: %d test samples",
                        len(multimodal_true))
        except Exception as e:
            logger.error("Checkpoint inference failed: %s", e)
            multimodal_true = None
            multimodal_pred = None
            broken_proba = None

    if multimodal_pred is not None and len(multimodal_pred):
        reg = regression_metrics(multimodal_true, multimodal_pred)
        summary["multimodal_underpricing"] = reg
        _print_summary("Multimodal — Underpricing", reg)

        try:
            decile = decile_analysis(multimodal_true, multimodal_pred)
            decile.to_csv(output_dir / "decile_analysis.csv", index=False)
        except Exception as e:
            logger.warning("Decile analysis failed: %s", e)

        if broken_proba is not None and len(broken_proba) == len(multimodal_true):
            try:
                # Recover broken_ipo target from sign of underpricing as a fallback
                broken_true = (multimodal_true < 0).astype(int)
                cls = classification_metrics(broken_true, broken_proba)
                summary["multimodal_broken_ipo"] = cls
                _print_summary("Multimodal — Broken IPO", cls)
            except Exception as e:
                logger.warning("Broken-IPO classification failed: %s", e)

    # ---- 2. Predictions-CSV fallback ----
    pred_csvs = _load_predictions_csv(Path("outputs/predictions"))
    if pred_csvs:
        for name, df in pred_csvs.items():
            yt = df["y_true"].to_numpy()
            yp = df["y_pred"].to_numpy()
            reg = regression_metrics(yt, yp)
            summary[f"baseline_{name}"] = reg
            _print_summary(name, reg)

    # ---- 3. Statistical tests ----
    if multimodal_pred is not None and "tabular" in pred_csvs:
        df_tab = pred_csvs["tabular"]
        if len(df_tab) == len(multimodal_pred):
            yt = df_tab["y_true"].to_numpy()
            tab_pred = df_tab["y_pred"].to_numpy()
            try:
                dm = diebold_mariano_test(yt, tab_pred, multimodal_pred)
                summary["dm_test_vs_tabular"] = dm
                _print_summary("DM test vs tabular", dm)
            except Exception as e:
                logger.warning("DM test failed: %s", e)
            try:
                boot = bootstrap_r2_difference(yt, tab_pred, multimodal_pred)
                summary["bootstrap_r2_vs_tabular"] = boot
                _print_summary("Bootstrap R² vs tabular", boot)
            except Exception as e:
                logger.warning("Bootstrap R² failed: %s", e)

    # ---- 4. Modality attributions ----
    if checkpoint and Path(checkpoint).exists():
        try:
            from src.analysis.attributions import compute_modality_attributions
            attribs = compute_modality_attributions(
                checkpoint=str(checkpoint), config=config,
            )
            summary["modality_attributions"] = attribs
            _print_summary("Modality attributions", attribs)
        except Exception as e:
            logger.warning("Modality attribution failed: %s", e)

    # ---- 5. Figures ----
    try:
        from src.analysis.figures import (
            calibration_plot,
            factor_correlation_heatmap,
            figure1_ablation_bars,
            shap_summary_plot,
        )
        if multimodal_pred is not None and multimodal_true is not None and len(multimodal_pred):
            try:
                calibration_plot(
                    multimodal_true, multimodal_pred,
                    output_path=str(output_dir / "calibration.png"),
                )
            except Exception as e:
                logger.warning("calibration_plot failed: %s", e)

        ablation_csv = Path("outputs/ablation_results.csv")
        if ablation_csv.exists():
            try:
                figure1_ablation_bars(
                    str(ablation_csv),
                    output_path=str(output_dir / "figure1_ablation.png"),
                )
            except Exception as e:
                logger.warning("figure1_ablation_bars failed: %s", e)

        shap_csv = Path("outputs/shap_values.csv")
        if shap_csv.exists():
            try:
                shap_summary_plot(
                    str(shap_csv),
                    output_path=str(output_dir / "shap_summary.png"),
                )
            except Exception as e:
                logger.warning("shap_summary_plot failed: %s", e)

        factor_csv = Path("outputs/visual_factor_correlations.csv")
        if factor_csv.exists():
            try:
                factor_correlation_heatmap(
                    str(factor_csv),
                    output_path=str(output_dir / "factor_correlations.png"),
                )
            except Exception as e:
                logger.warning("factor_correlation_heatmap failed: %s", e)
    except ImportError as e:
        logger.warning("Figures module not available: %s", e)

    # ---- 6. Save ----
    _save_summary(summary, output_dir)

    # Pretty print final table
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE — summary:")
    logger.info("=" * 60)
    for k, v in summary.items():
        if isinstance(v, dict):
            line = ", ".join(
                f"{kk}={vv:.4f}" if isinstance(vv, (int, float)) else f"{kk}={vv}"
                for kk, vv in list(v.items())[:6]
            )
            logger.info("  %-30s %s", k, line)
    logger.info("=" * 60)
    logger.info("Saved metrics → %s", output_dir / "metrics_summary.json")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Run full IPO model evaluation")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a Lightning .ckpt file")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--output-dir", type=str, default="outputs/eval")
    args = p.parse_args(argv)

    run_eval(
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
