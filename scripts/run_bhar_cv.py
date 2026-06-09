"""
5-fold cross-validated BHAR prediction evaluation.

Addresses the single-split bias identified in professor K's feedback:
- Within-fold naive mean (no stale-period anchor)
- MAE ± SE and R² ± SE across folds
- Diebold-Mariano tests vs naive_mean
- Industry FE (SIC2) check for image/multimodal
- Portfolio quintile sorts with long-short Sharpe

Architecture: pre-compute frozen backbone embeddings (CLIP ViT-L/14, FinBERT)
once, then per fold train only the lightweight projection+head layers. This is
equivalent to the full model under frozen backbones and is ~20x faster.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT = ROOT / "outputs" / "bhar"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = (3, 6, 12, 24)
N_FOLDS = 5
RANDOM_STATE = 42
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)

TABULAR_COLS = [
    "offer_size", "firm_age", "underwriter_rank", "vc_backed",
    "log_assets", "leverage", "rnd_intensity", "revenue_growth",
]

MULTIMODAL_CONFIGS = [
    "naive_mean", "tabular_only", "text_only", "image_only",
    "text_tabular", "full_multimodal",
]
FULL_CONFIGS = [
    "naive_mean", "tabular_only", "text_only", "text_tabular",
]

EMBED_CACHE = ROOT / "outputs" / "bhar" / "embedding_cache.npz"


# ---------------------------------------------------------------------------
# CRSP auto-switch (Step 4)
# ---------------------------------------------------------------------------

def check_crsp() -> bool:
    crsp_firm = ROOT / "data/raw/crsp/firm_returns.csv"
    crsp_mkt  = ROOT / "data/raw/crsp/market.csv"
    return crsp_firm.exists() and crsp_mkt.exists()


USING_CRSP = check_crsp()
if USING_CRSP:
    logger.info("CRSP data found — using CRSP VW index with delisting handling")
else:
    print(
        "\n" + "=" * 72
        + "\nWARNING: CRSP data not found — using yfinance/SPY proxy. "
        "Results are preliminary.\n"
        + "=" * 72 + "\n"
    )


# ---------------------------------------------------------------------------
# Backbone embedding pre-computation
# ---------------------------------------------------------------------------

def _load_text(cik: str) -> str:
    cik_dir = ROOT / "data/processed/text" / cik
    if not cik_dir.exists():
        return ""
    files = sorted(cik_dir.glob("*_sections.json"), reverse=True)
    if not files:
        return ""
    sections = json.loads(files[0].read_text())
    return sections.get("Risk Factors", "")


def _load_images_for_cik(cik: str, manifest: dict, max_images: int = 16) -> list:
    from PIL import Image
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    imgs = []
    for path in manifest.get(str(cik), [])[:max_images]:
        try:
            img = Image.open(path).convert("RGB")
            imgs.append(tf(img))
        except Exception:
            pass
    return imgs


def precompute_embeddings(
    ciks: list[str],
    force: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute and cache 768-d backbone embeddings for every CIK.
    Returns dict with keys: 'text_{cik}', 'img_{cik}' as numpy arrays.
    """
    cache_text = ROOT / "outputs/bhar/cache_text_emb.npz"
    cache_img  = ROOT / "outputs/bhar/cache_img_emb.npz"

    text_needed = force or not cache_text.exists()
    img_needed  = force or not cache_img.exists()

    # --- text embeddings ---
    if text_needed:
        logger.info("Pre-computing FinBERT CLS embeddings for %d CIKs ...", len(ciks))
        from transformers import BertTokenizer, BertModel
        try:
            tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            bert = BertModel.from_pretrained("yiyanghkust/finbert-tone").to(DEVICE)
        except Exception:
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            bert = AutoModel.from_pretrained("ProsusAI/finbert").to(DEVICE)
        bert.eval()

        text_embs = {}
        with torch.no_grad():
            for i, cik in enumerate(ciks):
                txt = _load_text(str(cik))
                if not txt:
                    text_embs[str(cik)] = np.zeros(768, dtype=np.float32)
                    continue
                toks = tokenizer(
                    txt, truncation=True, max_length=512,
                    return_tensors="pt", padding=True,
                )
                toks = {k: v.to(DEVICE) for k, v in toks.items()}
                out = bert(**toks)
                emb = out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
                text_embs[str(cik)] = emb
                if (i + 1) % 50 == 0:
                    logger.info("  text embeddings: %d/%d", i + 1, len(ciks))

        np.savez_compressed(cache_text, **text_embs)
        logger.info("Saved text embedding cache → %s", cache_text)
        del bert
    else:
        logger.info("Loading cached text embeddings from %s", cache_text)
        data = np.load(cache_text, allow_pickle=True)
        text_embs = {k: data[k] for k in data.files}

    # --- image embeddings ---
    if img_needed:
        logger.info("Pre-computing CLIP ViT-L/14 image embeddings for %d CIKs ...", len(ciks))
        import open_clip
        manifest_path = ROOT / "data/processed/images/image_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        clip_visual = clip_model.visual.to(DEVICE)
        clip_visual.eval()
        clip_embed_dim = clip_model.visual.output_dim

        img_embs = {}
        with torch.no_grad():
            for i, cik in enumerate(ciks):
                imgs = _load_images_for_cik(str(cik), manifest)
                if not imgs:
                    img_embs[str(cik)] = np.zeros(clip_embed_dim, dtype=np.float32)
                    continue
                batch = torch.stack(imgs).to(DEVICE)
                feats = clip_visual(batch).cpu().numpy()
                img_embs[str(cik)] = feats.mean(axis=0).astype(np.float32)
                if (i + 1) % 50 == 0:
                    logger.info("  image embeddings: %d/%d", i + 1, len(ciks))

        np.savez_compressed(cache_img, **img_embs)
        logger.info("Saved image embedding cache → %s", cache_img)
        del clip_visual, clip_model
    else:
        logger.info("Loading cached image embeddings from %s", cache_img)
        data = np.load(cache_img, allow_pickle=True)
        img_embs = {k: data[k] for k in data.files}

    return {"text": text_embs, "img": img_embs}


# ---------------------------------------------------------------------------
# Lightweight MLP for fold training
# ---------------------------------------------------------------------------

class FoldModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _build_X(
    ciks: list[str],
    tab_arr: np.ndarray,
    embeddings: dict,
    config: str,
) -> np.ndarray:
    parts = []
    if config in ("tabular_only", "text_tabular", "full_multimodal"):
        parts.append(tab_arr)
    if config in ("text_only", "text_tabular", "full_multimodal"):
        txt = np.stack([embeddings["text"].get(str(c), np.zeros(768)) for c in ciks])
        parts.append(txt.astype(np.float32))
    if config in ("image_only", "full_multimodal"):
        img = np.stack([embeddings["img"].get(str(c), np.zeros(1024)) for c in ciks])
        parts.append(img.astype(np.float32))
    if not parts:
        raise ValueError(f"Unknown config: {config}")
    return np.concatenate(parts, axis=1)


def train_fold_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_epochs: int = 40,
    lr: float = 5e-4,
    batch_size: int = 32,
) -> np.ndarray:
    model = FoldModel(X_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    model.train()
    n = len(X_t)
    for _ in range(n_epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            bi = idx[start: start + batch_size]
            xb, yb = X_t[bi], y_t[bi]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        X_v = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        preds = model(X_v).cpu().numpy()
    return preds


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    d_mean = d.mean()
    d_var = d.var(ddof=1)
    if d_var <= 0:
        return 0.0, 1.0
    dm_stat = d_mean / np.sqrt(d_var / n)
    p_val = float(2 * stats.norm.sf(abs(dm_stat)))
    return float(dm_stat), p_val


# ---------------------------------------------------------------------------
# Step 1: 5-fold CV
# ---------------------------------------------------------------------------

def run_cv(
    df: pd.DataFrame,
    embeddings: dict,
    sample_name: str,
    configs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        cv_results   - MAE/R2 mean±SE per model × horizon
        dm_tests     - Diebold-Mariano stats per model × horizon
        fold_preds   - Per-observation (cik, fold, horizon, y_actual, y_pred, model) records
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    img_embed_dim = next(iter(embeddings["img"].values())).shape[0] if embeddings["img"] else 1024

    cv_rows = []
    dm_rows = []
    pred_rows = []

    for horizon in HORIZONS:
        col = f"bhar_{horizon}m"
        sub = df[df[col].notna()].copy().reset_index(drop=True)
        if len(sub) < 20:
            logger.warning("skip h=%dm %s: only %d obs", horizon, sample_name, len(sub))
            continue

        logger.info("--- %s | horizon=%dm | n=%d ---", sample_name, horizon, len(sub))

        y_all = sub[col].values.astype(np.float32)
        ciks_all = sub["cik"].values.tolist()

        tab_cols_avail = [c for c in TABULAR_COLS if c in sub.columns]
        tab_raw = sub[tab_cols_avail].fillna(0).values.astype(np.float32)

        fold_data: dict[str, list] = {cfg: {"mae": [], "r2": [], "errors": [], "y_act": [], "y_pred": []} for cfg in configs}
        fold_naive_errors: list[np.ndarray] = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(sub)):
            logger.info("  fold %d/%d", fold_idx + 1, N_FOLDS)
            y_train, y_test = y_all[train_idx], y_all[test_idx]
            ciks_train = [ciks_all[i] for i in train_idx]
            ciks_test  = [ciks_all[i] for i in test_idx]

            tab_train = tab_raw[train_idx]
            tab_test  = tab_raw[test_idx]

            tab_mean = tab_train.mean(axis=0)
            tab_std  = tab_train.std(axis=0) + 1e-8
            tab_train_n = (tab_train - tab_mean) / tab_std
            tab_test_n  = (tab_test  - tab_mean) / tab_std

            naive_mu = float(y_train.mean())

            for cfg in configs:
                if cfg == "naive_mean":
                    preds = np.full(len(y_test), naive_mu, dtype=np.float32)
                else:
                    tab_for_build_train = tab_train_n
                    tab_for_build_test  = tab_test_n
                    try:
                        X_tr = _build_X(ciks_train, tab_for_build_train, embeddings, cfg)
                        X_te = _build_X(ciks_test,  tab_for_build_test,  embeddings, cfg)
                        preds = train_fold_model(X_tr, y_train, X_te)
                    except Exception as exc:
                        logger.error("    %s fold %d failed: %s", cfg, fold_idx, exc)
                        preds = np.full(len(y_test), naive_mu, dtype=np.float32)

                fold_mae = mae(y_test, preds)
                fold_r2  = r2(y_test, preds)
                errors   = y_test - preds

                fold_data[cfg]["mae"].append(fold_mae)
                fold_data[cfg]["r2"].append(fold_r2)
                fold_data[cfg]["errors"].append(errors)
                fold_data[cfg]["y_act"].append(y_test)
                fold_data[cfg]["y_pred"].append(preds)

                if cfg == "naive_mean":
                    fold_naive_errors.append(errors)

                for y_a, y_p, cik in zip(y_test, preds, ciks_test):
                    pred_rows.append({
                        "sample": sample_name,
                        "horizon_months": horizon,
                        "model": cfg,
                        "fold": fold_idx,
                        "cik": cik,
                        "y_actual": float(y_a),
                        "y_pred": float(y_p),
                    })

        naive_errors_pooled = np.concatenate(fold_naive_errors) if fold_naive_errors else np.array([])

        for cfg in configs:
            maes = np.array(fold_data[cfg]["mae"])
            r2s  = np.array(fold_data[cfg]["r2"])
            mae_mean = float(maes.mean())
            mae_se   = float(maes.std() / np.sqrt(N_FOLDS))
            r2_mean  = float(np.nanmean(r2s))
            r2_se    = float(np.nanstd(r2s) / np.sqrt(N_FOLDS))

            cv_rows.append({
                "sample": sample_name,
                "horizon_months": horizon,
                "model": cfg,
                "mae_mean": mae_mean,
                "mae_se": mae_se,
                "r2_mean": r2_mean,
                "r2_se": r2_se,
                "n_obs": len(sub),
            })

            if cfg != "naive_mean" and len(naive_errors_pooled) > 0:
                model_errors = np.concatenate(fold_data[cfg]["errors"])
                dm_stat, dm_p = diebold_mariano(naive_errors_pooled, model_errors)
                dm_rows.append({
                    "sample": sample_name,
                    "horizon_months": horizon,
                    "model": cfg,
                    "dm_stat": dm_stat,
                    "dm_pvalue": dm_p,
                })

    cv_results = pd.DataFrame(cv_rows)
    dm_tests   = pd.DataFrame(dm_rows)
    fold_preds = pd.DataFrame(pred_rows)
    return cv_results, dm_tests, fold_preds


# ---------------------------------------------------------------------------
# Step 2: SIC FE check
# ---------------------------------------------------------------------------

def run_sic_fe_check(fold_preds: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    sic_path = ROOT / "data/raw/cik_sic_codes.csv"
    if not sic_path.exists():
        logger.warning("No SIC codes file at %s — skipping SIC FE check", sic_path)
        return pd.DataFrame()

    sic_df = pd.read_csv(sic_path, dtype={"cik": str})
    sic_df["sic2"] = sic_df["sic"].astype(str).str[:2].str.zfill(2)
    sic_map = sic_df.set_index("cik")["sic2"].to_dict()

    target_models = ["image_only", "full_multimodal"]
    fe_rows = []

    for model in target_models:
        for horizon in HORIZONS:
            sub = fold_preds[
                (fold_preds["model"] == model)
                & (fold_preds["horizon_months"] == horizon)
            ].copy()
            if len(sub) < 10:
                continue

            sub["cik_str"] = sub["cik"].astype(str)
            sub["sic2"] = sub["cik_str"].map(sic_map)
            sub = sub.dropna(subset=["sic2"])
            if len(sub) < 10:
                logger.warning("SIC FE skip %s h=%dm: insufficient after SIC merge", model, horizon)
                continue

            try:
                m1 = smf.ols("y_actual ~ y_pred", data=sub).fit()
                coef_no_fe  = float(m1.params["y_pred"])
                tstat_no_fe = float(m1.tvalues["y_pred"])
                pval_no_fe  = float(m1.pvalues["y_pred"])
                r2_no_fe    = float(m1.rsquared)
            except Exception as e:
                logger.error("SIC FE OLS (no FE) failed for %s h=%dm: %s", model, horizon, e)
                coef_no_fe = tstat_no_fe = pval_no_fe = r2_no_fe = float("nan")

            try:
                unique_sics = sub["sic2"].nunique()
                if unique_sics < 2:
                    raise ValueError("only 1 SIC code — dummies redundant")
                m2 = smf.ols("y_actual ~ y_pred + C(sic2)", data=sub).fit()
                coef_fe  = float(m2.params["y_pred"])
                tstat_fe = float(m2.tvalues["y_pred"])
                pval_fe  = float(m2.pvalues["y_pred"])
                r2_fe    = float(m2.rsquared)
            except Exception as e:
                logger.warning("SIC FE OLS (with FE) failed for %s h=%dm: %s", model, horizon, e)
                coef_fe = tstat_fe = pval_fe = r2_fe = float("nan")

            fe_rows.append({
                "model": model,
                "horizon_months": horizon,
                "coef_no_fe": coef_no_fe,
                "tstat_no_fe": tstat_no_fe,
                "pval_no_fe": pval_no_fe,
                "r2_no_fe": r2_no_fe,
                "coef_with_fe": coef_fe,
                "tstat_with_fe": tstat_fe,
                "pval_with_fe": pval_fe,
                "r2_with_fe": r2_fe,
                "n_obs": len(sub),
                "n_sic2": int(sub["sic2"].nunique()),
            })

    return pd.DataFrame(fe_rows)


# ---------------------------------------------------------------------------
# Step 3: Portfolio sorts
# ---------------------------------------------------------------------------

def run_portfolio_sorts(fold_preds: pd.DataFrame) -> pd.DataFrame:
    port_rows = []
    for (sample, model, horizon), grp in fold_preds.groupby(
        ["sample", "model", "horizon_months"]
    ):
        if len(grp) < 10:
            continue
        grp = grp.copy()
        grp["quintile"] = pd.qcut(grp["y_pred"], q=5, labels=False, duplicates="drop")
        grp = grp.dropna(subset=["quintile"])
        if grp["quintile"].nunique() < 5:
            continue

        quintile_means = grp.groupby("quintile")["y_actual"].mean()
        q1_mean  = float(quintile_means.get(0, float("nan")))
        q2_mean  = float(quintile_means.get(1, float("nan")))
        q3_mean  = float(quintile_means.get(2, float("nan")))
        q4_mean  = float(quintile_means.get(3, float("nan")))
        q5_mean  = float(quintile_means.get(4, float("nan")))
        spread   = q5_mean - q1_mean if np.isfinite(q5_mean) and np.isfinite(q1_mean) else float("nan")

        q5_rets = grp[grp["quintile"] == 4]["y_actual"].values
        q1_rets = grp[grp["quintile"] == 0]["y_actual"].values
        ls_size = min(len(q5_rets), len(q1_rets))
        if ls_size > 1:
            ls_series = np.concatenate([q5_rets[:ls_size], -q1_rets[:ls_size]])
            ls_mean   = ls_series.mean()
            ls_std    = ls_series.std(ddof=1)
            ann_factor = np.sqrt(12.0 / horizon)
            sharpe = float((ls_mean / ls_std) * ann_factor) if ls_std > 0 else float("nan")
        else:
            sharpe = float("nan")

        port_rows.append({
            "sample": sample,
            "model": model,
            "horizon_months": horizon,
            "q1_mean": q1_mean,
            "q2_mean": q2_mean,
            "q3_mean": q3_mean,
            "q4_mean": q4_mean,
            "q5_mean": q5_mean,
            "ls_spread": spread,
            "ls_sharpe_annualized": sharpe,
            "n_obs": len(grp),
        })

    return pd.DataFrame(port_rows)


# ---------------------------------------------------------------------------
# Step 5: Regenerate histograms with correct labels
# ---------------------------------------------------------------------------

def regenerate_histograms() -> None:
    panel_path = OUT / "bhar_panel.parquet"
    if not panel_path.exists():
        logger.warning("bhar_panel.parquet not found — cannot regenerate histograms")
        return

    panel = pd.read_parquet(panel_path)
    benchmark = "CRSP VW Index" if USING_CRSP else "SPY (preliminary — yfinance proxy, survivorship-biased)"

    fig, axes = plt.subplots(len(HORIZONS), 2, figsize=(13, 3.5 * len(HORIZONS)))

    for i, h in enumerate(HORIZONS):
        b = panel[(panel["horizon_months"] == h) & (panel["status"] == "complete")]["bhar"].dropna().to_numpy()
        if b.size == 0:
            continue
        log_b = np.sign(b) * np.log1p(np.abs(b))

        ax = axes[i, 0]
        ax.hist(b, bins=60, color="#4C72B0", edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.axvline(b.mean(), color="red",    ls="--", lw=1.3, label=f"mean {b.mean():.2%}")
        ax.axvline(np.median(b), color="orange", ls="--", lw=1.3, label=f"median {np.median(b):.2%}")
        ax.axvline(0, color="black", lw=0.6)
        if USING_CRSP:
            title = f"{h}-Month BHAR vs CRSP VW Index  (N={b.size})"
        else:
            title = f"{h}-Month BHAR vs SPY (preliminary — yfinance proxy, survivorship-biased)  (N={b.size})"
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("BHAR")
        ax.legend(fontsize=7)

        ax = axes[i, 1]
        ax.hist(log_b, bins=60, color="#C44E52", edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.axvline(0, color="black", lw=0.6)
        raw_skew = pd.Series(b).skew()
        log_skew = pd.Series(log_b).skew()
        ax.set_title(f"{h}-Month BHAR signed-log  (skew {raw_skew:.2f} → {log_skew:.2f})", fontsize=8)
        ax.set_xlabel("sign(BHAR) × log(1+|BHAR|)")

    suptitle_label = f"CRSP VW Index" if USING_CRSP else "SPY proxy (yfinance, preliminary)"
    fig.suptitle(
        f"Long-Run Post-IPO Abnormal Returns vs {suptitle_label}",
        fontsize=12, y=1.005,
    )
    fig.tight_layout()
    out_path = OUT / "bhar_histograms.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Regenerated histograms → %s", out_path)


# ---------------------------------------------------------------------------
# Step 7: Verification
# ---------------------------------------------------------------------------

def run_verification(
    cv_results: pd.DataFrame,
    dm_tests: pd.DataFrame,
    sic_fe: pd.DataFrame,
    portfolio_sorts: pd.DataFrame,
) -> None:
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)

    def check(label: str, condition: bool) -> None:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}]  {label}")

    check(
        "cv_results.csv exists with MAE and R² ± SE for all model × horizon",
        (OUT / "cv_results.csv").exists()
        and "mae_mean" in cv_results.columns
        and "mae_se" in cv_results.columns
        and "r2_mean" in cv_results.columns
        and len(cv_results) > 0,
    )
    check(
        "dm_tests.csv exists with p-values for all model × horizon",
        (OUT / "dm_tests.csv").exists()
        and "dm_pvalue" in dm_tests.columns
        and len(dm_tests) > 0,
    )
    check(
        "sic_fe_check.csv exists with results for image_only and full_multimodal",
        (OUT / "sic_fe_check.csv").exists()
        and len(sic_fe) > 0,
    )
    check(
        "portfolio_sorts.csv exists with finite Sharpe values",
        (OUT / "portfolio_sorts.csv").exists()
        and len(portfolio_sorts) > 0
        and portfolio_sorts["ls_sharpe_annualized"].notna().any(),
    )

    if len(cv_results) > 0:
        naive = cv_results[cv_results["model"] == "naive_mean"]
        if len(naive) > 0:
            check(
                "naive_mean SE > 0 (confirming within-fold CV, not a single value)",
                float(naive["mae_se"].min()) > 0.0,
            )
        else:
            check("naive_mean SE > 0", False)

    if len(dm_tests) > 0:
        check(
            "DM tests have variation in p-values (not all identical)",
            dm_tests["dm_pvalue"].nunique() > 1,
        )
    else:
        check("DM tests have variation in p-values", False)

    # git push check
    import subprocess
    try:
        res = subprocess.run(
            ["git", "push", "--dry-run"],
            capture_output=True, text=True, cwd=ROOT,
        )
        push_ok = res.returncode == 0
    except Exception:
        push_ok = False
    check("Git push dry-run succeeded", push_ok)

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CRSP BHAR rebuild
# ---------------------------------------------------------------------------

def _load_crsp_crosswalk() -> pd.DataFrame | None:
    """
    Load a PERMNO→CIK crosswalk from one of two places:
      1. data/raw/crsp/cik_permno.csv  (columns: permno, cik)
      2. data/raw/crsp/firm_returns.csv already contains a 'cik' column

    Returns a DataFrame with columns [permno, cik], or None if neither source
    is available (caller should fall back to pre-computed parquets).
    """
    crosswalk_path = ROOT / "data/raw/crsp/cik_permno.csv"
    if crosswalk_path.exists():
        xw = pd.read_csv(crosswalk_path, dtype={"permno": int, "cik": str})
        xw["cik"] = xw["cik"].str.lstrip("0")
        logger.info("Crosswalk loaded from cik_permno.csv: %d rows", len(xw))
        return xw[["permno", "cik"]].drop_duplicates("permno")

    firm_path = ROOT / "data/raw/crsp/firm_returns.csv"
    if firm_path.exists():
        cols = pd.read_csv(firm_path, nrows=0).columns.tolist()
        if "cik" in cols:
            xw = pd.read_csv(firm_path, usecols=["permno", "cik"],
                             dtype={"permno": int, "cik": str})
            xw["cik"] = xw["cik"].str.lstrip("0")
            xw = xw.drop_duplicates("permno")
            logger.info("Crosswalk extracted from firm_returns.csv: %d unique PERMNOs", len(xw))
            return xw[["permno", "cik"]]

    logger.warning("No PERMNO→CIK crosswalk found (checked cik_permno.csv and firm_returns.csv 'cik' column)")
    return None


def _rebuild_bhar_from_crsp(
    base_mm: pd.DataFrame,
    base_full: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replace bhar_* columns in the universe DataFrames using CRSP inputs.

    Reads:
      data/raw/crsp/firm_returns.csv   [permno, date, ret]
      data/raw/crsp/market.csv         [date, vwretd]
      data/raw/crsp/delisting.csv      [permno, delist_date, delist_ret]  (optional)

    Falls back to the original DataFrames unchanged if the crosswalk cannot be
    resolved.  All changes are saved back to the parquet files so a subsequent
    run doesn't have to rebuild.
    """
    from src.data.bhar import build_bhar_panel

    firm_path = ROOT / "data/raw/crsp/firm_returns.csv"
    mkt_path  = ROOT / "data/raw/crsp/market.csv"
    delist_path = ROOT / "data/raw/crsp/delisting.csv"

    logger.info("Loading CRSP firm returns ...")
    firm_ret = pd.read_csv(firm_path, parse_dates=["date"],
                           dtype={"permno": int, "ret": float})
    firm_ret = firm_ret[["permno", "date", "ret"]].dropna(subset=["ret"])

    logger.info("Loading CRSP market returns ...")
    mkt = pd.read_csv(mkt_path, parse_dates=["date"])
    mkt_col = "vwretd" if "vwretd" in mkt.columns else mkt.columns[-1]
    mkt = mkt.rename(columns={mkt_col: "vwretd"})[["date", "vwretd"]].dropna()

    delist = None
    if delist_path.exists():
        logger.info("Loading CRSP delisting returns ...")
        delist = pd.read_csv(delist_path, parse_dates=["delist_date"],
                             dtype={"permno": int, "delist_ret": float})

    crosswalk = _load_crsp_crosswalk()
    if crosswalk is None:
        logger.error("Cannot rebuild BHAR from CRSP — no PERMNO→CIK crosswalk. Falling back to parquet BHARs.")
        return base_mm, base_full

    logger.info("Building BHAR panel for %d PERMNOs ...", firm_ret["permno"].nunique())
    panel = build_bhar_panel(firm_ret, mkt, delist)

    panel_wide = (
        panel[panel["status"].isin(["complete", "delisted"]) & panel["bhar"].notna()]
        .pivot(index="permno", columns="horizon_months", values="bhar")
        .rename(columns={3: "bhar_3m", 6: "bhar_6m", 12: "bhar_12m", 24: "bhar_24m"})
        .reset_index()
    )

    panel_wide = panel_wide.merge(crosswalk, on="permno", how="left")
    missing_cik = panel_wide["cik"].isna().sum()
    if missing_cik:
        logger.warning("%d PERMNOs have no CIK match — they will be excluded", missing_cik)
    panel_wide = panel_wide.dropna(subset=["cik"])
    panel_wide["cik"] = panel_wide["cik"].astype(str).str.lstrip("0")

    bhar_cols = ["bhar_3m", "bhar_6m", "bhar_12m", "bhar_24m"]

    def _replace_bhars(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["cik"] = df["cik"].astype(str).str.lstrip("0")
        df = df.drop(columns=[c for c in bhar_cols if c in df.columns])
        df = df.merge(panel_wide[["cik"] + bhar_cols], on="cik", how="left")
        n_matched = df[bhar_cols[0]].notna().sum()
        n_total = len(df)
        logger.info("CRSP BHARs merged: %d/%d firms matched", n_matched, n_total)
        return df

    mm_new   = _replace_bhars(base_mm)
    full_new = _replace_bhars(base_full)

    mm_new.to_parquet(OUT / "multimodal_sample_bhar.parquet", index=False)
    full_new.to_parquet(OUT / "full_sample_bhar.parquet", index=False)
    logger.info("Saved CRSP-backed BHAR parquet files (replaces yfinance/SPY proxy)")

    panel.to_parquet(OUT / "bhar_panel.parquet", index=False)
    logger.info("Saved full CRSP BHAR panel: %d rows", len(panel))

    return mm_new, full_new


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Loading BHAR parquet files ...")
    mm_df   = pd.read_parquet(OUT / "multimodal_sample_bhar.parquet")
    full_df = pd.read_parquet(OUT / "full_sample_bhar.parquet")

    if USING_CRSP:
        logger.info("CRSP files detected — rebuilding BHAR panels from raw CRSP data ...")
        mm_df, full_df = _rebuild_bhar_from_crsp(mm_df, full_df)
    else:
        logger.info("Using pre-computed BHAR values (yfinance/SPY proxy)")

    all_ciks = list(set(mm_df["cik"].astype(str).tolist() + full_df["cik"].astype(str).tolist()))
    logger.info("Total unique CIKs: %d", len(all_ciks))

    # Pre-compute embeddings
    embeddings = precompute_embeddings(all_ciks)

    # Verify image embed dim
    sample_img = next(iter(embeddings["img"].values()))
    img_dim = sample_img.shape[0]
    logger.info("Image embedding dim: %d, text embedding dim: %d", img_dim, 768)

    # Step 1: 5-fold CV
    logger.info("=== Step 1: 5-fold CV ===")
    mm_cv, mm_dm, mm_preds = run_cv(mm_df, embeddings, "multimodal", MULTIMODAL_CONFIGS)
    full_cv, full_dm, full_preds = run_cv(full_df, embeddings, "full", FULL_CONFIGS)

    cv_results   = pd.concat([mm_cv, full_cv], ignore_index=True)
    dm_tests     = pd.concat([mm_dm, full_dm], ignore_index=True)
    fold_preds   = pd.concat([mm_preds, full_preds], ignore_index=True)

    cv_results.to_csv(OUT / "cv_results.csv", index=False)
    dm_tests.to_csv(OUT / "dm_tests.csv", index=False)
    fold_preds.to_csv(OUT / "fold_predictions.csv", index=False)
    logger.info("Saved cv_results.csv (%d rows), dm_tests.csv (%d rows)", len(cv_results), len(dm_tests))

    # Step 2: SIC FE check
    logger.info("=== Step 2: SIC FE check ===")
    sic_fe = run_sic_fe_check(fold_preds)
    sic_fe.to_csv(OUT / "sic_fe_check.csv", index=False)
    logger.info("Saved sic_fe_check.csv (%d rows)", len(sic_fe))

    # Step 3: Portfolio sorts
    logger.info("=== Step 3: Portfolio sorts ===")
    portfolio_sorts = run_portfolio_sorts(fold_preds)
    portfolio_sorts.to_csv(OUT / "portfolio_sorts.csv", index=False)
    logger.info("Saved portfolio_sorts.csv (%d rows)", len(portfolio_sorts))

    # Step 5: Regenerate histograms
    logger.info("=== Step 5: Regenerate histograms ===")
    regenerate_histograms()

    # Print summary tables
    print("\n=== CV RESULTS (MAE ± SE) ===")
    for (sample, horizon), grp in cv_results.groupby(["sample", "horizon_months"]):
        print(f"\n  {sample} | {horizon}m BHAR:")
        for _, row in grp.sort_values("mae_mean").iterrows():
            print(f"    {row['model']:20s}  MAE={row['mae_mean']:.4f}±{row['mae_se']:.4f}  R²={row['r2_mean']:.4f}±{row['r2_se']:.4f}")

    print("\n=== DIEBOLD-MARIANO TESTS vs naive_mean ===")
    for (sample, horizon), grp in dm_tests.groupby(["sample", "horizon_months"]):
        print(f"\n  {sample} | {horizon}m:")
        for _, row in grp.iterrows():
            sig = "*" if row["dm_pvalue"] < 0.1 else ""
            sig = "**" if row["dm_pvalue"] < 0.05 else sig
            print(f"    {row['model']:20s}  DM={row['dm_stat']:+.3f}  p={row['dm_pvalue']:.3f}{sig}")

    if len(sic_fe) > 0:
        print("\n=== SIC FE CHECK ===")
        print(sic_fe[["model", "horizon_months", "coef_no_fe", "coef_with_fe", "pval_no_fe", "pval_with_fe", "r2_no_fe", "r2_with_fe"]].to_string(index=False))

    print("\n=== PORTFOLIO SORTS (long-short spreads) ===")
    for (sample, model, horizon), grp in portfolio_sorts.groupby(["sample", "model", "horizon_months"]):
        row = grp.iloc[0]
        print(f"  {sample:12s} {model:20s} {horizon}m  Q1={row['q1_mean']:.3f}  Q5={row['q5_mean']:.3f}  spread={row['ls_spread']:.3f}  Sharpe={row['ls_sharpe_annualized']:.3f}")

    # Step 7: Verification
    run_verification(cv_results, dm_tests, sic_fe, portfolio_sorts)


if __name__ == "__main__":
    main()
