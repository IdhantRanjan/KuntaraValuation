#!/usr/bin/env bash
# =============================================================================
# Full IPO multimodal pipeline (data → train → evaluate → tables/figures).
# =============================================================================
set -euo pipefail

DRY_RUN="false"
PYTHON="${PYTHON:-.venv/bin/python}"
PIP="${PIP:-.venv/bin/pip}"

usage() {
    cat <<EOF
Usage: $0 [--dry-run]

Runs the end-to-end pipeline:
  1. Install package
  2. Build Ritter → CIK → ipo_master.csv
  3. Build IPO universe parquet
  4. Download post-IPO returns
  5. Scrape S-1 filings from EDGAR
  6. Run image pipeline (CLIP zero-shot filtering)
  7. Extract S-1 financial ratios
  8. Rebuild universe with financials
  9. Train multimodal model
 10. Run ablation studies
 11. Run evaluation
 12. Generate figures
 13. Generate LaTeX tables
EOF
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="true";;
        -h|--help) usage; exit 0;;
        *) echo "Unknown argument: $arg"; usage; exit 2;;
    esac
done

ts() { date +"%Y-%m-%dT%H:%M:%S"; }

run() {
    echo "[$(ts)] >>> $*"
    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    eval "$@"
}

run "$PIP install -e .[dev] --quiet"

run "$PYTHON scripts/build_ritter_universe.py"

run "$PYTHON scripts/generate_cik_list.py"

run "$PYTHON -m src.data.ipo_universe --csv data/raw/ipo_master.csv"

run "$PYTHON -m src.data.fetch_post_ipo_returns --universe-csv data/raw/ipo_master.csv"

run "$PYTHON -m src.data.edgar_scraper --cik-file data/raw/cik_list.txt"

run "$PYTHON -m src.data.image_pipeline --use-clip"

run "$PYTHON -m src.data.fetch_s1_financials --universe-csv data/raw/ipo_master.csv"

run "$PYTHON -m src.data.ipo_universe --csv data/raw/ipo_master.csv"

run "$PYTHON -m src.training.trainer --config configs/config.yaml"

run "$PYTHON -m src.training.ablations --config configs/config.yaml"

run "$PYTHON -m src.evaluation.run_eval --config configs/config.yaml"

run "$PYTHON -m src.analysis.figures"

run "$PYTHON -m src.analysis.tables"

echo "[$(ts)] === pipeline complete ==="
