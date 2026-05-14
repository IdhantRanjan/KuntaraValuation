# Multimodal Image–Text–Tabular Models for IPO Valuation

> **Research project**: Predicting IPO underpricing and post-IPO outcomes using
> operational images from S-1 prospectuses, risk-factor text, and pre-IPO financial ratios.

## Core Idea

IPO prospectuses (S-1 filings) contain **operational and product images** that
convey information about asset tangibility, capital intensity, and technological
sophistication—signals that are **not captured by text or financials alone**. This
project builds a multimodal deep-learning pipeline that fuses:

| Modality | Encoder | Source |
|----------|---------|--------|
| **Images** | CLIP ViT-L/14 (frozen) + attention pooling | Filtered operational images from S-1 filings |
| **Text** | FinBERT | Risk Factors section of S-1 |
| **Tabular** | MLP | Pre-IPO financial ratios & controls |

Three fusion strategies are compared: **late fusion**, **gated additive fusion**,
and **cross-attention transformer fusion**.

## Prediction Targets

| Target | Type | Definition |
|--------|------|------------|
| First-day return | Regression | (P_close − P_offer) / P_offer |
| Broken IPO | Classification | 1 if first-day close < offer price |
| Post-IPO volatility | Regression | 6-/12-month return volatility |

## Project Structure

```
KuntaraValuation/
├── configs/                 # Hydra / OmegaConf YAML configs
│   ├── config.yaml          # Master config
│   ├── data/                # Data-source configs
│   ├── model/               # Architecture configs
│   └── training/            # Training hyperparameters
├── src/
│   ├── data/                # Data collection & preprocessing
│   │   ├── ipo_universe.py  # Sample construction & labels
│   │   ├── edgar_scraper.py # SEC EDGAR S-1 downloader & parser
│   │   ├── image_pipeline.py# Image extraction & filtering
│   │   ├── private_firms.py # Private-firm web scraping (extension)
│   │   └── dataset.py       # PyTorch Dataset / DataLoader
│   ├── features/            # Modality-specific encoders
│   │   ├── text_encoder.py  # FinBERT encoder
│   │   ├── image_encoder.py # CLIP ViT encoder + attention pooling
│   │   └── tabular_encoder.py # MLP for financial ratios
│   ├── models/              # Fusion & prediction
│   │   ├── fusion.py        # Late, gated, cross-attention fusion
│   │   ├── heads.py         # Task-specific prediction heads
│   │   └── multimodal.py    # Full end-to-end model
│   ├── training/            # Training loop & ablations
│   │   ├── trainer.py       # PyTorch Lightning trainer
│   │   └── ablations.py     # Ablation runner
│   ├── baselines/           # Non-deep & single-modality baselines
│   │   ├── classical.py     # OLS / Lasso + LM dictionaries
│   │   └── gbm_baseline.py  # LightGBM / XGBoost
│   ├── evaluation/          # Metrics & statistical tests
│   │   ├── metrics.py       # MAE, RMSE, R², AUC, F1
│   │   └── statistical_tests.py # Diebold-Mariano, decile analysis
│   └── analysis/            # Interpretability & figures
│       ├── attributions.py  # SHAP & integrated gradients
│       ├── visual_factors.py# PCA on image embeddings
│       ├── case_studies.py  # Exemplar IPO attention viz
│       └── figures.py       # Publication-quality figures
├── notebooks/               # Exploratory analysis
├── tests/                   # Unit & integration tests
├── data/                    # Raw & processed data (gitignored)
├── outputs/                 # Model checkpoints, figures, tables
├── pyproject.toml
└── README.md
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Download S-1 filings
python -m src.data.edgar_scraper --config configs/data/edgar.yaml

# Extract & filter images
python -m src.data.image_pipeline --config configs/data/images.yaml

# Train full multimodal model
python -m src.training.trainer --config configs/config.yaml

# Run all ablations
python -m src.training.ablations --config configs/config.yaml

# Generate figures
python -m src.analysis.figures --config configs/config.yaml
```

## Key References

- Ghosh et al. (2024) — Multimodal Indian IPO prediction (text+numeric, images as OCR only)
- Pukthuanthong et al. — Image-based Firm Similarity (IFS)
- Sharpe (2022) — S-1 embeddings for IPO performance
- Tavakoli et al. (2023–25) — Multimodal credit rating with cross-attention
- Ben-Rephael et al. (2025) — Image informativeness in annual reports
