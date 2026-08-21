"""Plot Top-K consensus (S31) and repeated-CV stability (S32) figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TOPK_DIR = SCRIPT_DIR / "top_k_consensus_results"
STAB_DIR = SCRIPT_DIR / "stability_repeated_cv_results"
OUT_DIR = SCRIPT_DIR / "results_figures"

MODEL_ORDER = ("ElasticNet", "LASSO", "MoE", "RF", "SVR", "XGBoost")
TIER_COLORS = {">=4/6": "#d95f02", ">=5/6": "#2e8b57", "6/6": "#1f4e79"}
CORE_GENES = ("PTPN13", "MYO6", "KLHL3", "GZMK", "RCAN2", "CLIC5", "COL6A5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 Results 配套 Figure S31/S32")
    parser.add_argument(
        "--figure",
        choices=("all", "s31", "s32"),
        default="all",
        help="生成哪张图",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--highlight-k", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--top-genes-panel-b", type=int, default=12)
    return parser.parse_args()


def save_figure(fig: plt.Figure, prefix: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = output_dir / f"{prefix}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"已保存: {path}")
    plt.close(fig)


def plot_s31_panel_a(ax: plt.Axes, summary: pd.DataFrame, highlight_k: int) -> None:
    k_values = sorted(summary["K"].unique())
    ax.plot(
        summary["K"],
        summary["strict_6_of_6"],
        marker="o",
        linewidth=2.2,
        markersize=7,
        color=TIER_COLORS["6/6"],
        label="6/6 (all models)",
    )
    ax.plot(
        summary["K"],
        summary["consensus_5_of_6"],
        marker="s",
        linewidth=2.2,
        markersize=7,
        color=TIER_COLORS[">=5/6"],
        label="≥5/6 models",
    )
    ax.plot(
        summary["K"],
        summary["consensus_4_of_6"],
        marker="^",
        linewidth=2.2,
        markersize=7,
        color=TIER_COLORS[">=4/6"],
        label="≥4/6 models",
    )
    ymax = max(
        summary["strict_6_of_6"].max(),
        summary["consensus_5_of_6"].max(),
        summary["consensus_4_of_6"].max(),
    )
    ax.set_ylim(0, ymax * 1.12)
    if highlight_k in k_values:
        ax.axvline(highlight_k, color="#666666", linestyle="--", linewidth=1.4, alpha=0.9)
        ax.text(highlight_k, ymax * 1.06, f"K = {highlight_k}", ha="center", fontsize=9)
    ax.set_xlabel("Top-K threshold")
    ax.set_ylabel("Number of consensus genes")
    ax.set_xticks(k_values)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("A", loc="left", fontweight="bold", fontsize=12)


def plot_s31_panel_b(ax: plt.Axes, consensus: pd.DataFrame, k: int, min_models: int = 4) -> None:
    shown = consensus.loc[consensus["n_models"] >= min_models].copy()
    shown = shown.sort_values(["n_models", "gene"], ascending=[False, True])
    matrix = shown[list(MODEL_ORDER)].to_numpy()
    ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(shown)))
    ax.set_yticklabels(shown["gene"], fontsize=8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=6)
    ax.set_xlabel("Model")
    ax.set_ylabel("Gene")
    ax.set_title(
        f"B  Top-{k}: genes with ≥{min_models}/6 model support",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )


def make_figure_s31(output_dir: Path, highlight_k: int) -> None:
    summary_path = TOPK_DIR / "threshold_sensitivity_summary.csv"
    consensus_path = TOPK_DIR / f"gene_consensus_top{highlight_k}.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 {summary_path}，请先运行 top_k_consensus_analysis.py")
    if not consensus_path.exists():
        raise FileNotFoundError(f"缺少 {consensus_path}")

    summary = pd.read_csv(summary_path)
    consensus = pd.read_csv(consensus_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.05, 1.15]})
    plot_s31_panel_a(axes[0], summary, highlight_k)
    plot_s31_panel_b(axes[1], consensus, highlight_k)
    fig.suptitle(
        "Cross-model consensus of high-contribution genes across Top-K thresholds",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "Figure_S31_topk_consensus", output_dir)


def plot_s32_panel_a(ax: plt.Axes, cons_resample: pd.DataFrame, top_k: int) -> None:
    sub = cons_resample.loc[cons_resample["K"] == top_k]
    cols = {
        "n_consensus_4of6": ">=4/6",
        "n_consensus_5of6": ">=5/6",
        "n_consensus_6of6": "6/6",
    }
    data = [sub[c].to_numpy() for c in cols]
    labels = list(cols.values())
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        whis=(0, 100),
        showfliers=True,
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.55},
        medianprops={"color": "black", "linewidth": 1.4},
    )
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(TIER_COLORS[label])
        patch.set_alpha(0.75)
    ax.set_ylabel("Number of consensus genes per resample")
    ax.set_xlabel("Consensus tier")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_title(f"A  K = {top_k}, n = {len(sub)} resamples", loc="left", fontweight="bold", fontsize=11)


def plot_s32_panel_b(ax: plt.Axes, cross: pd.DataFrame, top_k: int, n_genes: int) -> None:
    tiers_csv = [">=4/6", ">=5/6", ">=6/6"]
    tier_labels = [">=4/6", ">=5/6", "6/6"]
    base = cross.loc[
        (cross["K"] == top_k) & (cross["consensus_threshold"] == ">=4/6")
    ].sort_values("consensus_frequency", ascending=False)
    top_genes = base.head(n_genes)["gene"].astype(str).tolist()
    y = np.arange(len(top_genes))
    bar_h = 0.22
    offsets = {">=4/6": -bar_h, ">=5/6": 0.0, "6/6": bar_h}
    for tier_csv, tier_label in zip(tiers_csv, tier_labels):
        values = []
        for gene in top_genes:
            row = cross.loc[
                (cross["K"] == top_k)
                & (cross["consensus_threshold"] == tier_csv)
                & (cross["gene"].astype(str) == gene)
            ]
            values.append(float(row["consensus_frequency"].iloc[0]) if len(row) else 0.0)
        ax.barh(
            y + offsets[tier_label],
            values,
            height=bar_h,
            color=TIER_COLORS[tier_label],
            label=tier_label,
            alpha=0.9,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(top_genes)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Cross-model consensus frequency")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_title("B", loc="left", fontweight="bold", fontsize=12)


def plot_s32_panel_c(ax: plt.Axes, model_freq: pd.DataFrame, top_k: int, genes: tuple[str, ...]) -> None:
    matrix = np.zeros((len(genes), len(MODEL_ORDER)))
    for i, gene in enumerate(genes):
        for j, model in enumerate(MODEL_ORDER):
            row = model_freq.loc[
                (model_freq["model"] == model)
                & (model_freq["K"] == top_k)
                & (model_freq["gene"].astype(str) == gene)
            ]
            matrix[i, j] = float(row["selection_frequency"].iloc[0]) if len(row) else 0.0
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            val = matrix[row, col]
            color = "white" if val >= 0.55 else "black"
            ax.text(col, row, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Selection frequency")
    ax.set_title("C  Core genes: model-specific selection frequency", loc="left", fontweight="bold", fontsize=11)


def make_figure_s32(
    output_dir: Path,
    top_k: int,
    n_genes: int,
) -> None:
    cons_path = STAB_DIR / "consensus_counts_by_resample.csv"
    cross_path = STAB_DIR / "cross_model_consensus_frequencies.csv"
    freq_path = STAB_DIR / "model_gene_selection_frequencies.csv"
    for path in (cons_path, cross_path, freq_path):
        if not path.exists():
            raise FileNotFoundError(f"缺少 {path}，请先运行 stability_selection_analysis.py")

    cons_resample = pd.read_csv(cons_path)
    cross = pd.read_csv(cross_path)
    model_freq = pd.read_csv(freq_path)

    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    plot_s32_panel_a(ax_a, cons_resample, top_k)
    plot_s32_panel_b(ax_b, cross, top_k, n_genes)
    plot_s32_panel_c(ax_c, model_freq, top_k, CORE_GENES)

    fig.suptitle(
        "Stability of gene selection under repeated cross-validation",
        fontsize=13,
        y=0.98,
    )
    save_figure(fig, "Figure_S32_stability_repeated_cv", output_dir)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.figure in ("all", "s31"):
        make_figure_s31(output_dir, args.highlight_k)
    if args.figure in ("all", "s32"):
        make_figure_s32(output_dir, args.top_k, args.top_genes_panel_b)
    print(f"完成。输出目录: {output_dir}")


if __name__ == "__main__":
    main()
