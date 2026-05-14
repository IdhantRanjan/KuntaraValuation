"""
LaTeX Table Generator.

Produces publication-quality booktabs tables for the paper:
  Table 1 — Descriptive statistics
  Table 2 — Main results across ablation configurations
  Table 3 — Statistical tests (DM, bootstrap R²)
  Table 4 — Visual-factor correlations with economic outcomes
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

LATEX_HEADER = (
    "\\begin{{table}}[!htbp]\n"
    "\\centering\n"
    "\\caption{{{caption}}}\n"
    "\\label{{tab:{label}}}\n"
    "\\begin{{tabular}}{{{spec}}}\n"
    "\\toprule\n"
)

LATEX_FOOTER = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def _stars(p: float | None) -> str:
    """Return *, **, *** based on p-value."""
    if p is None or pd.isna(p):
        return ""
    if p < 0.01:
        return "$^{***}$"
    if p < 0.05:
        return "$^{**}$"
    if p < 0.10:
        return "$^{*}$"
    return ""


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (pd.isna(x) or np.isinf(x))):
        return "--"
    return f"{x:.{digits}f}"


def _escape_latex(s: str) -> str:
    return (
        str(s)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


# ---------------------------------------------------------------------------
# Table 1 — Descriptive statistics
# ---------------------------------------------------------------------------

DEFAULT_DESCRIPTIVE_VARS = [
    "first_day_return", "broken_ipo", "offer_size", "firm_age",
    "underwriter_rank", "vc_backed", "log_assets", "leverage",
    "rnd_intensity", "revenue_growth", "post_ipo_volatility_6m",
]


def table1_descriptive_stats(
    df: pd.DataFrame,
    output_path: Path,
    variables: list[str] | None = None,
) -> str:
    """N, mean, median, std, min, max, p25, p75 for the standard variable list."""
    variables = variables or DEFAULT_DESCRIPTIVE_VARS
    rows = []
    for v in variables:
        if v not in df.columns:
            continue
        s = pd.to_numeric(df[v], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "variable": v,
            "N": int(s.size),
            "Mean": float(s.mean()),
            "Median": float(s.median()),
            "SD": float(s.std()),
            "Min": float(s.min()),
            "P25": float(s.quantile(0.25)),
            "P75": float(s.quantile(0.75)),
            "Max": float(s.max()),
        })
    stats = pd.DataFrame(rows)

    spec = "lrrrrrrrr"
    out = LATEX_HEADER.format(
        caption="Descriptive Statistics for the IPO Sample",
        label="descriptive",
        spec=spec,
    )
    out += "Variable & N & Mean & Median & SD & Min & P25 & P75 & Max \\\\\n\\midrule\n"
    for _, row in stats.iterrows():
        out += (
            f"{_escape_latex(row['variable'])} & {row['N']} & "
            f"{_fmt(row['Mean'])} & {_fmt(row['Median'])} & "
            f"{_fmt(row['SD'])} & {_fmt(row['Min'])} & "
            f"{_fmt(row['P25'])} & {_fmt(row['P75'])} & {_fmt(row['Max'])} \\\\\n"
        )
    out += LATEX_FOOTER

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out)
    logger.info("Wrote Table 1 → %s", output_path)
    return out


# ---------------------------------------------------------------------------
# Table 2 — Main ablation results
# ---------------------------------------------------------------------------

def table2_main_results(
    results_dict: dict,
    output_path: Path,
) -> str:
    """
    Compare ablation configurations on MAE, RMSE, R², AUC, Vol-MAE.

    Args:
        results_dict: {config_name: {metric: value, ..., dm_p_value: optional}}.
    """
    metric_cols = ["mae", "rmse", "r2", "auc", "volatility_mae"]
    table = pd.DataFrame(results_dict).T

    # Bold the best per metric (min for mae/rmse/vol_mae, max for r2/auc)
    best = {}
    for m in metric_cols:
        if m not in table.columns:
            continue
        col = pd.to_numeric(table[m], errors="coerce")
        if m in {"mae", "rmse", "volatility_mae"}:
            best[m] = col.idxmin()
        else:
            best[m] = col.idxmax()

    spec = "l" + "r" * len(metric_cols)
    out = LATEX_HEADER.format(
        caption="Main Results: Multimodal Ablation Configurations",
        label="main-results",
        spec=spec,
    )
    out += "Configuration & MAE & RMSE & $R^2$ & AUC & Vol-MAE \\\\\n\\midrule\n"

    for cfg_name, metrics in results_dict.items():
        row_cells = [_escape_latex(cfg_name)]
        for m in metric_cols:
            v = metrics.get(m, None)
            cell = _fmt(v)
            if best.get(m) == cfg_name and cell != "--":
                cell = f"\\textbf{{{cell}}}"
            stars = ""
            if m == "mae" and "dm_p_value" in metrics:
                stars = _stars(metrics["dm_p_value"])
            cell = cell + stars
            row_cells.append(cell)
        out += " & ".join(row_cells) + " \\\\\n"

    out += LATEX_FOOTER
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out)
    logger.info("Wrote Table 2 → %s", output_path)
    return out


# ---------------------------------------------------------------------------
# Table 3 — Statistical tests
# ---------------------------------------------------------------------------

def table3_statistical_tests(
    comparisons_df: pd.DataFrame,
    output_path: Path,
) -> str:
    """DM + bootstrap R² comparisons relative to a reference model."""
    spec = "l" + "r" * 5
    out = LATEX_HEADER.format(
        caption="Statistical Comparison: Multimodal vs. Ablations",
        label="stat-tests",
        spec=spec,
    )
    out += (
        "Comparison & DM stat & DM $p$ & "
        "$\\Delta R^2$ & 95\\% CI & Significant \\\\\n\\midrule\n"
    )
    for _, row in comparisons_df.iterrows():
        out += (
            f"{_escape_latex(row.get('model', ''))} & "
            f"{_fmt(row.get('dm_statistic'))} & "
            f"{_fmt(row.get('dm_p_value'), 4)}{_stars(row.get('dm_p_value'))} & "
            f"{_fmt(row.get('r2_diff_mean'))} & "
            f"[{_fmt(row.get('r2_diff_ci_low'))}, "
            f"{_fmt(row.get('r2_diff_ci_high'))}] & "
            f"{'Yes' if row.get('r2_improvement_significant') else 'No'} \\\\\n"
        )
    out += LATEX_FOOTER
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out)
    logger.info("Wrote Table 3 → %s", output_path)
    return out


# ---------------------------------------------------------------------------
# Table 4 — Visual factor correlations
# ---------------------------------------------------------------------------

def table4_visual_factors(
    correlations_df: pd.DataFrame,
    output_path: Path,
) -> str:
    """
    Visual factor correlations table.
    Expects a DataFrame indexed by factor (VF1..VF5) with columns of the
    form '<var>' (Pearson r) and optional '<var>_p' (p-value).
    """
    df = correlations_df.copy()
    var_cols = [c for c in df.columns if not c.endswith("_p")]
    spec = "l" + "r" * len(var_cols)

    out = LATEX_HEADER.format(
        caption="Visual Factor Correlations with Economic Outcomes",
        label="visual-factors",
        spec=spec,
    )
    out += "Factor & " + " & ".join(_escape_latex(c) for c in var_cols)
    out += " \\\\\n\\midrule\n"

    for idx, row in df.iterrows():
        cells = [_escape_latex(str(idx))]
        for c in var_cols:
            r = row[c]
            p = row.get(f"{c}_p", None)
            cells.append(f"{_fmt(r)}{_stars(p)}")
        out += " & ".join(cells) + " \\\\\n"

    out += LATEX_FOOTER
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out)
    logger.info("Wrote Table 4 → %s", output_path)
    return out


# ---------------------------------------------------------------------------
# Generate-all driver
# ---------------------------------------------------------------------------

def generate_all_tables(output_dir: Path = Path("outputs/tables")) -> None:
    """Generate every table from CSV artifacts saved in outputs/."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Table 1
    desc_csv = Path("data/processed/ipo_sample/ipo_universe.parquet")
    if desc_csv.exists():
        try:
            df = pd.read_parquet(desc_csv)
            table1_descriptive_stats(df, output_dir / "table1_descriptive.tex")
        except Exception as e:
            logger.warning("Table 1 failed: %s", e)
    else:
        # Fall back to ipo_master.csv
        master = Path("data/raw/ipo_master.csv")
        if master.exists():
            df = pd.read_csv(master)
            table1_descriptive_stats(df, output_dir / "table1_descriptive.tex")

    # Table 2
    abl_csv = Path("outputs/ablation_results.csv")
    if abl_csv.exists():
        try:
            abl = pd.read_csv(abl_csv).set_index(abl_csv.columns[0])
            results_dict = abl.to_dict(orient="index")
            table2_main_results(results_dict, output_dir / "table2_main_results.tex")
        except Exception as e:
            logger.warning("Table 2 failed: %s", e)

    # Table 3
    comp_csv = Path("outputs/analysis/statistical_comparisons.csv")
    if comp_csv.exists():
        try:
            comp = pd.read_csv(comp_csv)
            table3_statistical_tests(comp, output_dir / "table3_stat_tests.tex")
        except Exception as e:
            logger.warning("Table 3 failed: %s", e)

    # Table 4
    fac_csv = Path("outputs/visual_factor_correlations.csv")
    if fac_csv.exists():
        try:
            fac = pd.read_csv(fac_csv, index_col=0)
            table4_visual_factors(fac, output_dir / "table4_visual_factors.tex")
        except Exception as e:
            logger.warning("Table 4 failed: %s", e)

    logger.info("All tables written to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="Generate LaTeX paper tables")
    p.add_argument("--output-dir", type=str, default="outputs/tables")
    args = p.parse_args(argv)
    generate_all_tables(Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
