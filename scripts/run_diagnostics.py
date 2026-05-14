"""
Diagnostic analyses requested by K:
  - FDR histogram (raw + log-transformed)
  - Sample selection table (Ritter universe vs final samples)
  - Naive mean baseline
  - Industry fixed effects on image-only model
  - 5-fold CV with standard errors and Diebold-Mariano tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


# ── 1. FDR histogram ────────────────────────────────────────────────────

def plot_fdr_histogram(df: pd.DataFrame):
    fdr = df["first_day_return"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # raw
    ax = axes[0]
    ax.hist(fdr, bins=60, edgecolor="black", linewidth=0.4, alpha=0.8)
    ax.axvline(fdr.mean(), color="red", linestyle="--", label=f"Mean = {fdr.mean():.3f}")
    ax.axvline(fdr.median(), color="blue", linestyle="--", label=f"Median = {fdr.median():.3f}")
    ax.set_xlabel("First-Day Return")
    ax.set_ylabel("Count")
    ax.set_title(f"Raw First-Day Returns (N={len(fdr)})")
    ax.legend(fontsize=9)

    # log(1+r)
    log_fdr = np.log1p(fdr)
    ax = axes[1]
    ax.hist(log_fdr, bins=60, edgecolor="black", linewidth=0.4, alpha=0.8, color="steelblue")
    ax.axvline(log_fdr.mean(), color="red", linestyle="--", label=f"Mean = {log_fdr.mean():.3f}")
    ax.axvline(log_fdr.median(), color="blue", linestyle="--", label=f"Median = {log_fdr.median():.3f}")
    ax.set_xlabel("log(1 + First-Day Return)")
    ax.set_ylabel("Count")
    ax.set_title(f"Log-Transformed First-Day Returns (N={len(fdr)})")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fdr_histogram.png", dpi=150)
    plt.close(fig)

    print("FDR distribution:")
    print(f"  N           = {len(fdr)}")
    print(f"  Mean        = {fdr.mean():.4f}")
    print(f"  Median      = {fdr.median():.4f}")
    print(f"  Std         = {fdr.std():.4f}")
    print(f"  Skewness    = {fdr.skew():.4f}")
    print(f"  Kurtosis    = {fdr.kurtosis():.4f}")
    print(f"  Min / Max   = {fdr.min():.4f} / {fdr.max():.4f}")
    print(f"  % negative  = {(fdr < 0).mean()*100:.1f}%")
    print(f"  p5 / p95    = {fdr.quantile(.05):.4f} / {fdr.quantile(.95):.4f}")
    print()
    print("Log-transformed:")
    print(f"  Skewness    = {log_fdr.skew():.4f}")
    print(f"  Kurtosis    = {log_fdr.kurtosis():.4f}")

    # Shapiro-Wilk on log
    if len(log_fdr) <= 5000:
        w, p = stats.shapiro(log_fdr)
        print(f"  Shapiro-Wilk = {w:.4f}  (p = {p:.4e})")


# ── 2. Selection diagnostics ────────────────────────────────────────────

def selection_table(ritter: pd.DataFrame, full: pd.DataFrame, mm: pd.DataFrame):
    """Compare key variables across the Ritter universe and our subsamples."""

    compare_cols = [
        ("offer_price", "Offer Price ($)"),
        ("first_day_return", "First-Day Return"),
        ("log_assets", "Log Assets"),
        ("leverage", "Leverage"),
        ("rnd_intensity", "R&D Intensity"),
        ("revenue_growth", "Revenue Growth"),
        ("firm_age", "Firm Age (yrs)"),
        ("post_ipo_volatility_6m", "6-Month Volatility"),
        ("vc_backed", "VC-Backed (%)"),
    ]

    rows = []
    for col, label in compare_cols:
        row = {"Variable": label}
        for name, df in [("Ritter (N=4136)", ritter),
                         ("Full (N=621)", full),
                         ("Multimodal (N=209)", mm)]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if col == "vc_backed":
                    row[f"{name} mean"] = f"{vals.mean()*100:.1f}%"
                    row[f"{name} N"] = int(len(vals))
                else:
                    row[f"{name} mean"] = f"{vals.mean():.3f}"
                    row[f"{name} median"] = f"{vals.median():.3f}"
                    row[f"{name} N"] = int(len(vals))
            else:
                row[f"{name} mean"] = "—"
                row[f"{name} median"] = "—"
                row[f"{name} N"] = 0
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "selection_diagnostics.csv", index=False)
    print("\nSelection diagnostics:")
    print(table.to_string(index=False))
    return table


# ── 3. Naive mean baseline ──────────────────────────────────────────────

def naive_baselines(df: pd.DataFrame, label: str = ""):
    """MAE and MedAE of predicting the training-set mean/median."""
    from src.data.ipo_universe import time_split

    train, val, test = time_split(df)
    y_test = test["first_day_return"].values

    train_mean = pd.concat([train, val])["first_day_return"].mean()
    train_median = pd.concat([train, val])["first_day_return"].median()

    mae_mean = np.mean(np.abs(y_test - train_mean))
    mae_median = np.mean(np.abs(y_test - train_median))
    medae_mean = np.median(np.abs(y_test - train_mean))
    medae_median = np.median(np.abs(y_test - train_median))

    rmse_mean = np.sqrt(np.mean((y_test - train_mean) ** 2))
    ss_res = np.sum((y_test - train_mean) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"\nNaive baselines ({label}, test N={len(y_test)}):")
    print(f"  Train mean = {train_mean:.4f}, train median = {train_median:.4f}")
    print(f"  Predict-mean:   MAE={mae_mean:.4f}  MedAE={medae_mean:.4f}  RMSE={rmse_mean:.4f}  R²={r2:.4f}")
    print(f"  Predict-median: MAE={mae_median:.4f}  MedAE={medae_median:.4f}")

    return {
        "sample": label,
        "train_mean": train_mean,
        "mae_mean": mae_mean,
        "medae_mean": medae_mean,
        "rmse_mean": rmse_mean,
        "r2_mean": r2,
        "mae_median": mae_median,
        "medae_median": medae_median,
    }


# ── 4. Diebold-Mariano test ─────────────────────────────────────────────

def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1):
    """
    Two-sided Diebold-Mariano test comparing forecast accuracy.
    e1, e2: forecast errors (not absolute — we square them internally).
    Returns (DM statistic, p-value).
    """
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    d_bar = d.mean()
    # Newey-West variance with h-1 lags
    gamma_0 = np.mean((d - d_bar) ** 2)
    gamma_sum = gamma_0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2 * gamma_k
    var_d = gamma_sum / n
    if var_d <= 0:
        return 0.0, 1.0
    dm = d_bar / np.sqrt(var_d)
    pval = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(pval)


# ── 5. K-fold CV with standard errors ───────────────────────────────────

def kfold_cv_ablations(df: pd.DataFrame, n_folds: int = 5, max_epochs: int = 50):
    """
    Run k-fold CV for each ablation config.
    Returns per-fold MAE/MedAE/R² and standard errors.
    """
    from sklearn.model_selection import KFold
    from src.data.dataset import IPOMultimodalDataset
    from src.models.multimodal import MultimodalIPOModel
    from src.training.trainer import train
    from omegaconf import OmegaConf
    import torch
    from torch.utils.data import DataLoader

    cfg = OmegaConf.load(ROOT / "configs" / "model" / "cross_attention.yaml")
    model_cfg = OmegaConf.to_container(cfg, resolve=True)

    configs = [
        {"name": "tabular_only", "modalities": ["tabular"]},
        {"name": "text_only", "modalities": ["text"]},
        {"name": "image_only", "modalities": ["image"]},
        {"name": "text_tabular", "modalities": ["text", "tabular"]},
        {"name": "full_multimodal", "modalities": ["image", "text", "tabular"]},
    ]

    y = df["first_day_return"].values
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_results = []

    for abl in configs:
        name = abl["name"]
        mods = abl["modalities"]
        fold_maes, fold_medaes, fold_r2s = [], [], []
        fold_errors = {}

        print(f"\n{'='*50}")
        print(f"K-fold CV: {name}")
        print(f"{'='*50}")

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(df)):
            print(f"  Fold {fold_i+1}/{n_folds}...", end=" ", flush=True)
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            # split train into train/val (80/20)
            val_size = max(int(len(train_df) * 0.2), 1)
            val_df = train_df.iloc[-val_size:].reset_index(drop=True)
            train_sub = train_df.iloc[:-val_size].reset_index(drop=True)

            try:
                train_ds = IPOMultimodalDataset(train_sub, split="train")
                val_ds = IPOMultimodalDataset(val_df, split="val")
                test_ds = IPOMultimodalDataset(test_df, split="test")

                train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)
                test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

                tab_dim = train_ds.get_tabular_dim()
                model_cfg["encoders"]["tabular"]["input_dim"] = tab_dim

                model = MultimodalIPOModel(
                    image_config=model_cfg.get("encoders", {}).get("image", {}),
                    text_config=model_cfg.get("encoders", {}).get("text", {}),
                    tabular_config=model_cfg.get("encoders", {}).get("tabular", {}),
                    fusion_config=model_cfg.get("fusion", {}),
                    heads_config=model_cfg.get("heads", {}),
                    modalities=mods,
                )

                out_dir = OUT / "kfold" / name / f"fold_{fold_i}"
                trainer = train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    output_dir=str(out_dir),
                    max_epochs=max_epochs,
                    patience=8,
                    accelerator="auto",
                )

                # predict on test
                model.eval()
                device = next(model.parameters()).device
                preds, actuals = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        out = model(batch)
                        preds.append(out["underpricing"].cpu().numpy().flatten())
                        actuals.append(batch["targets"][:, 0].cpu().numpy().flatten())

                preds = np.concatenate(preds)
                actuals = np.concatenate(actuals)
                errors = actuals - preds

                mae = np.mean(np.abs(errors))
                medae = np.median(np.abs(errors))
                ss_res = np.sum(errors ** 2)
                ss_tot = np.sum((actuals - actuals.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

                fold_maes.append(mae)
                fold_medaes.append(medae)
                fold_r2s.append(r2)
                fold_errors[fold_i] = errors
                print(f"MAE={mae:.4f}, R²={r2:.4f}")

            except Exception as e:
                print(f"FAILED: {e}")
                fold_maes.append(np.nan)
                fold_medaes.append(np.nan)
                fold_r2s.append(np.nan)

        mae_arr = np.array([x for x in fold_maes if not np.isnan(x)])
        medae_arr = np.array([x for x in fold_medaes if not np.isnan(x)])
        r2_arr = np.array([x for x in fold_r2s if not np.isnan(x)])

        result = {
            "model": name,
            "modalities": ",".join(mods),
            "mae_mean": mae_arr.mean() if len(mae_arr) else np.nan,
            "mae_se": mae_arr.std() / np.sqrt(len(mae_arr)) if len(mae_arr) > 1 else np.nan,
            "medae_mean": medae_arr.mean() if len(medae_arr) else np.nan,
            "r2_mean": r2_arr.mean() if len(r2_arr) else np.nan,
            "r2_se": r2_arr.std() / np.sqrt(len(r2_arr)) if len(r2_arr) > 1 else np.nan,
            "n_folds": len(mae_arr),
        }
        all_results.append(result)
        print(f"  → {name}: MAE = {result['mae_mean']:.4f} ± {result['mae_se']:.4f}, R² = {result['r2_mean']:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT / "kfold_cv_results.csv", index=False)
    return results_df


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    ritter = pd.read_csv(ROOT / "data" / "raw" / "ritter_ipos.csv")
    universe = pd.read_parquet(ROOT / "data" / "processed" / "ipo_sample" / "ipo_universe_final.parquet")
    full = pd.read_parquet(ROOT / "outputs" / "modeling" / "full_sample.parquet")
    mm = pd.read_parquet(ROOT / "outputs" / "modeling" / "multimodal_sample.parquet")

    # merge SIC if available
    sic_path = ROOT / "data" / "raw" / "cik_sic_codes.csv"
    if sic_path.exists():
        sic = pd.read_csv(sic_path)
        for df in [full, mm, universe]:
            if "sic" not in df.columns and "cik" in df.columns:
                df_merged = df.merge(sic[["cik", "sic", "sic_description"]], on="cik", how="left")
                df.loc[:, "sic"] = df_merged["sic"].values
                df.loc[:, "sic_description"] = df_merged["sic_description"].values

    print("=" * 60)
    print("1. FDR HISTOGRAM")
    print("=" * 60)
    # use all obs with FDR, not just the filtered sample
    fdr_sample = universe[universe["first_day_return"].notna()]
    plot_fdr_histogram(fdr_sample)

    print("\n" + "=" * 60)
    print("2. SELECTION DIAGNOSTICS")
    print("=" * 60)
    selection_table(ritter, full, mm)

    print("\n" + "=" * 60)
    print("3. NAIVE BASELINES")
    print("=" * 60)
    nb_full = naive_baselines(full, "full (N=621)")
    nb_mm = naive_baselines(mm, "multimodal (N=209)")
    pd.DataFrame([nb_full, nb_mm]).to_csv(OUT / "naive_baselines.csv", index=False)

    print("\n" + "=" * 60)
    print("4. K-FOLD CV")
    print("=" * 60)
    print("Running on multimodal sample (N=209)...")
    cv_results = kfold_cv_ablations(mm, n_folds=5, max_epochs=50)
    print("\n\nK-fold CV results:")
    print(cv_results.to_string(index=False))


if __name__ == "__main__":
    main()
