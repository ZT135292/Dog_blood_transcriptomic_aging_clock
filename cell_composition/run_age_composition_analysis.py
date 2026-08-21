"""Associate CIBERSORTx leukocyte fractions / CLR with chronological age in Normal samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

mpl.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cell_composition_utils import (  # noqa: E402
    CELL_TYPES,
    CLR_COLS,
    bh_fdr,
    ensure_clr_columns,
    load_merged_composition,
    ols_table,
)

CELL_COLORS = {
    "Neutrophil": "#E76F51",
    "CD8/NK cell": "#2A9D8F",
    "Monocyte": "#E9C46A",
    "Granulocyte": "#F4A261",
    "CD4 T cell": "#264653",
    "B cell": "#5C7AEA",
    "DC": "#9B5DE5",
}

MPL_STYLE = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cell composition vs chronological age (no clock).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "cell_composition_analysis" / "age_composition",
    )
    return parser.parse_args()


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def prepare_normal(df: pd.DataFrame) -> pd.DataFrame:
    if "condition" in df.columns:
        out = df[df["condition"] == "Normal"].copy()
    else:
        out = df.copy()
    age_col = "Chronological_Age" if "Chronological_Age" in out.columns else "age"
    out["Chronological_Age"] = out[age_col].astype(float)
    out["sex"] = out["sex"].astype(str)
    out["batch"] = out["batch"].astype(str)
    return out.dropna(subset=["Chronological_Age"])


def cell_to_clr(cell_type: str) -> str:
    return CLR_COLS[CELL_TYPES.index(cell_type)]


def fit_one_cell_type(normal: pd.DataFrame, cell_type: str) -> dict:
    clr_col = cell_to_clr(cell_type)
    age = normal["Chronological_Age"]
    frac = normal[cell_type].astype(float)
    clr = normal[clr_col].astype(float)

    rho, sp = stats.spearmanr(age, frac, nan_policy="omit")
    raw_fit = ols_table(frac, pd.DataFrame({"Chronological_Age": age}), response_name="fraction")
    raw_row = raw_fit[raw_fit["term"] == "Chronological_Age"]

    adj_X = pd.get_dummies(
        normal[["Chronological_Age", "sex", "batch"]],
        drop_first=True,
        dtype=float,
    )
    adj_fit = ols_table(clr, adj_X, response_name="clr")
    adj_row = adj_fit[adj_fit["term"] == "Chronological_Age"]

    se = float(adj_row["std_error"].iloc[0]) if len(adj_row) else np.nan
    coef = float(adj_row["estimate"].iloc[0]) if len(adj_row) else np.nan
    ci_low = coef - 1.96 * se if pd.notna(se) and se > 0 else np.nan
    ci_high = coef + 1.96 * se if pd.notna(se) and se > 0 else np.nan

    return {
        "cell_type": cell_type,
        "n": int(len(normal)),
        "mean_fraction": float(frac.mean()),
        "median_fraction": float(frac.median()),
        "spearman_rho": float(rho),
        "spearman_p": float(sp),
        "raw_beta_fraction_per_year": float(raw_row["estimate"].iloc[0]) if len(raw_row) else np.nan,
        "raw_beta_p": float(raw_row["p_value"].iloc[0]) if len(raw_row) else np.nan,
        "adjusted_CLR_beta_per_year": coef,
        "adjusted_CLR_SE": se,
        "adjusted_CLR_CI_low": ci_low,
        "adjusted_CLR_CI_high": ci_high,
        "adjusted_CLR_p": float(adj_row["p_value"].iloc[0]) if len(adj_row) else np.nan,
    }


def add_fdr_columns(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["spearman_FDR"] = bh_fdr(out["spearman_p"])
    out["raw_beta_FDR"] = bh_fdr(out["raw_beta_p"])
    out["adjusted_CLR_FDR"] = bh_fdr(out["adjusted_CLR_p"])
    return out.sort_values("adjusted_CLR_p")


def fit_multivariate_models(normal: pd.DataFrame, clr_cols: list[str]) -> pd.DataFrame:
    rows = []
    specs = [
        ("age ~ CLR (7 types)", clr_cols, []),
        ("age ~ CLR + sex + batch", clr_cols, ["sex", "batch"]),
    ]
    for label, clr_part, cat_part in specs:
        parts = [normal[clr_part]]
        if cat_part:
            parts.append(pd.get_dummies(normal[cat_part], drop_first=True, dtype=float))
        X = pd.concat(parts, axis=1)
        fit = ols_table(normal["Chronological_Age"], X, response_name="age")
        rows.append({"model": label, "n": fit.attrs.get("n"), "r2": fit.attrs.get("r2")})
    return pd.DataFrame(rows)


def sensitivity_belgian_batch1(normal: pd.DataFrame) -> pd.DataFrame:
    if "breed" not in normal.columns:
        return pd.DataFrame()
    sub = normal[(normal["breed"] == "Belgian_Malinois") & (normal["batch"] == "batch1")].copy()
    if len(sub) < 20:
        return pd.DataFrame()
    rows = []
    for cell_type in CELL_TYPES:
        clr_col = cell_to_clr(cell_type)
        X = pd.get_dummies(
            sub[["Chronological_Age", "sex"]],
            drop_first=True,
            dtype=float,
        )
        fit = ols_table(sub[clr_col], X, response_name="clr")
        row = fit[fit["term"] == "Chronological_Age"]
        if row.empty:
            continue
        coef = float(row["estimate"].iloc[0])
        se = float(row["std_error"].iloc[0])
        rows.append(
            {
                "cell_type": cell_type,
                "n": len(sub),
                "adjusted_CLR_beta_per_year": coef,
                "adjusted_CLR_SE": se,
                "adjusted_CLR_CI_low": coef - 1.96 * se if se > 0 else np.nan,
                "adjusted_CLR_CI_high": coef + 1.96 * se if se > 0 else np.nan,
                "adjusted_CLR_p": float(row["p_value"].iloc[0]),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["adjusted_CLR_FDR"] = bh_fdr(out["adjusted_CLR_p"])
    return out.sort_values("adjusted_CLR_p")


def plot_composition_stack(normal: pd.DataFrame, out_stem: Path) -> None:
    work = normal.sort_values("Chronological_Age").reset_index(drop=True)
    work["sample_order"] = np.arange(len(work))
    bottom = np.zeros(len(work))
    fig, ax = plt.subplots(figsize=(11, 5))
    for ct in CELL_TYPES:
        vals = work[ct].astype(float).values
        ax.bar(work["sample_order"], vals, bottom=bottom, width=1.0, color=CELL_COLORS[ct], label=ct)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_xlabel("Normal samples ordered by chronological age")
    ax.set_ylabel("Inferred relative abundance")
    ax.set_title(f"Inferred leukocyte composition in Normal dogs (n={len(work)})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_stem)


def plot_fraction_scatter_grid(normal: pd.DataFrame, results: pd.DataFrame, out_stem: Path) -> None:
    ncols = 3
    nrows = int(np.ceil(len(CELL_TYPES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    res = results.set_index("cell_type")

    for idx, ct in enumerate(CELL_TYPES):
        ax = axes[idx // ncols][idx % ncols]
        x = normal["Chronological_Age"]
        y = normal[ct].astype(float)
        ax.scatter(x, y, s=28, alpha=0.75, color="#3B6EA8", edgecolors="white", lw=0.3)
        if len(x) >= 3:
            slope, intercept, _, _, _ = stats.linregress(x, y)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, intercept + slope * xx, color="#B23A48", lw=1.2)
        rho = res.loc[ct, "spearman_rho"]
        fdr = res.loc[ct, "spearman_FDR"]
        ax.set_title(f"{ct}\nSpearman ρ={rho:.2f}, FDR={fdr:.3g}")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Fraction")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    for idx in range(len(CELL_TYPES), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Leukocyte fractions vs chronological age (unadjusted linear trend)", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_stem)


def plot_clr_scatter_grid(normal: pd.DataFrame, results: pd.DataFrame, out_stem: Path) -> None:
    ncols = 3
    nrows = int(np.ceil(len(CELL_TYPES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    res = results.set_index("cell_type")

    for idx, ct in enumerate(CELL_TYPES):
        ax = axes[idx // ncols][idx % ncols]
        clr_col = cell_to_clr(ct)
        x = normal["Chronological_Age"]
        y = normal[clr_col].astype(float)
        ax.scatter(x, y, s=28, alpha=0.75, color="#2A9D8F", edgecolors="white", lw=0.3)
        if len(x) >= 3:
            slope, intercept, _, _, _ = stats.linregress(x, y)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, intercept + slope * xx, color="#B23A48", lw=1.2)
        beta = res.loc[ct, "adjusted_CLR_beta_per_year"]
        fdr = res.loc[ct, "adjusted_CLR_FDR"]
        ax.set_title(f"{ct}\nAdj. β={beta:.3f}/yr, FDR={fdr:.3g}")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("CLR")

    for idx in range(len(CELL_TYPES), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("CLR cell fractions vs age (scatter); title shows sex+batch-adjusted β", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_stem)


def plot_adjusted_forest(results: pd.DataFrame, out_stem: Path, subtitle: str) -> None:
    sub = results.sort_values("adjusted_CLR_beta_per_year")
    y = np.arange(len(sub))
    colors = ["#B23A48" if fdr < 0.05 else "0.55" for fdr in sub["adjusted_CLR_FDR"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (_, row) in enumerate(sub.iterrows()):
        left = row["adjusted_CLR_beta_per_year"] - row["adjusted_CLR_CI_low"]
        right = row["adjusted_CLR_CI_high"] - row["adjusted_CLR_beta_per_year"]
        ax.errorbar(
            row["adjusted_CLR_beta_per_year"],
            i,
            xerr=np.array([[left], [right]]),
            fmt="o",
            color=colors[i],
            capsize=3,
            markersize=6,
        )
    ax.axvline(0, color="0.45", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["cell_type"])
    ax.set_xlabel("Adjusted CLR coefficient per year (95% CI)")
    ax.set_title(f"Age associations with leukocyte composition\n{subtitle}")
    fig.tight_layout()
    save_figure(fig, out_stem)


def plot_spearman_bar(results: pd.DataFrame, out_stem: Path) -> None:
    sub = results.sort_values("spearman_rho")
    colors = ["#B23A48" if fdr < 0.05 else "#5C7AEA" for fdr in sub["spearman_FDR"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(sub["cell_type"], sub["spearman_rho"], color=colors)
    ax.axvline(0, color="0.4", lw=0.6)
    ax.set_xlabel("Spearman ρ (fraction vs age)")
    ax.set_title("Unadjusted rank correlation with chronological age")
    fig.tight_layout()
    save_figure(fig, out_stem)


def main() -> None:
    args = parse_args()
    mpl.rcParams.update(MPL_STYLE)
    out_dir = args.output_dir
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    df = ensure_clr_columns(load_merged_composition(SCRIPT_DIR))
    clr_cols = [c for c in CLR_COLS if c in df.columns]
    normal = prepare_normal(df)

    per_cell = [fit_one_cell_type(normal, ct) for ct in CELL_TYPES]
    results = add_fdr_columns(pd.DataFrame(per_cell))
    multiv = fit_multivariate_models(normal, clr_cols)
    sensitivity = sensitivity_belgian_batch1(normal)

    results.to_csv(table_dir / "Table_age_composition_per_celltype.csv", index=False)
    multiv.to_csv(table_dir / "Table_age_composition_multivariate.csv", index=False)
    if not sensitivity.empty:
        sensitivity.to_csv(table_dir / "Table_age_composition_sensitivity_BM_batch1.csv", index=False)

    plot_composition_stack(normal, fig_dir / "Figure_composition_by_age_order")
    plot_fraction_scatter_grid(normal, results, fig_dir / "Figure_fraction_vs_age_scatter")
    plot_clr_scatter_grid(normal, results, fig_dir / "Figure_CLR_vs_age_scatter")
    plot_adjusted_forest(
        results,
        fig_dir / "Figure_adjusted_CLR_forest",
        f"Normal dogs (n={len(normal)}); CLR ~ age + sex + batch",
    )
    plot_spearman_bar(results, fig_dir / "Figure_spearman_rho_bar")

    sig = results[results["adjusted_CLR_FDR"] < 0.05]
    report = [
        "=== Cell composition vs chronological age (no clock) ===",
        f"Normal samples: n={len(normal)}",
        f"Output: {out_dir}",
        "",
        "--- Multivariate models ---",
        multiv.to_string(index=False),
        "",
        "--- Per cell type (CLR ~ age + sex + batch) ---",
        results[
            [
                "cell_type",
                "spearman_rho",
                "spearman_FDR",
                "raw_beta_fraction_per_year",
                "raw_beta_FDR",
                "adjusted_CLR_beta_per_year",
                "adjusted_CLR_p",
                "adjusted_CLR_FDR",
            ]
        ].to_string(index=False),
        "",
        f"Significant after BH-FDR (adjusted CLR): {', '.join(sig['cell_type']) if not sig.empty else 'none'}",
    ]
    if not sensitivity.empty:
        report.extend(
            [
                "",
                "--- Sensitivity: Belgian Malinois batch1 ---",
                sensitivity.to_string(index=False),
            ]
        )
    (out_dir / "report_age_composition.txt").write_text("\n".join(report), encoding="utf-8")

    print(f"完成。表格: {table_dir}")
    print(f"      图片: {fig_dir}")
    print(f"      报告: {out_dir / 'report_age_composition.txt'}")


if __name__ == "__main__":
    main()
