"""Horizontal bars of cross-model consensus frequency (>=4/6, >=5/6, 6/6) from repeated CV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "stability_repeated_cv_results"
TIER_CSV = [">=4/6", ">=5/6", ">=6/6"]
TIER_LABELS = [">=4/6", ">=5/6", "6/6"]
TIER_COLORS = {">=4/6": "#d95f02", ">=5/6": "#2e8b57", "6/6": "#1f4e79"}
FONT_SIZE = 14
TITLE_SIZE = 14
ANNOTATION_SIZE = 10
LEGEND_SIZE = 11
BAR_HEIGHT = 0.30
FIG_HEIGHT_PER_GENE = 0.58
FIG_MIN_HEIGHT = 7.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stability repeated-CV Panel B (cross-model consensus frequency)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="含 cross_model_consensus_frequencies.csv 的目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="图与基因列表输出目录（默认与 --results-dir 相同）",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="按 >=4/6 共识频率取前 N 个基因（--genes 未指定时生效）",
    )
    parser.add_argument(
        "--min-freq-4of6",
        type=float,
        default=0.0,
        help=">=4/6 共识频率下限；在 top-n 之前过滤",
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        default=None,
        help="手动指定基因及 Y 轴顺序（覆盖 --top-n 自动筛选）",
    )
    parser.add_argument(
        "--highlight-genes",
        nargs="+",
        default=None,
        help="在自动 top-n 列表末尾追加、去重（仍按 >=4/6 频率排序的列表为基础）",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="Figure_X2_stability_repeated_cv_panel_b",
    )
    return parser.parse_args()


def load_cross_model(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "cross_model_consensus_frequencies.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}，请先运行 stability_selection_analysis.py")
    return pd.read_csv(path)


def select_genes(
    cross: pd.DataFrame,
    top_k: int,
    top_n: int,
    min_freq_4of6: float,
    manual_genes: list[str] | None,
    highlight_genes: list[str] | None,
) -> tuple[list[str], pd.DataFrame]:
    """返回基因列表及每个基因三档频率的明细表。"""
    base = cross.loc[
        (cross["K"] == top_k) & (cross["consensus_threshold"] == ">=4/6")
    ].copy()
    base["gene"] = base["gene"].astype(str)
    base = base.loc[base["consensus_frequency"] >= min_freq_4of6]
    base = base.sort_values(
        ["consensus_frequency", "gene"], ascending=[False, True]
    )

    if manual_genes:
        gene_list = [str(g) for g in manual_genes]
    else:
        gene_list = base.head(top_n)["gene"].tolist()
        if highlight_genes:
            for g in highlight_genes:
                g = str(g)
                if g not in gene_list:
                    gene_list.append(g)

    rows = []
    for rank, gene in enumerate(gene_list, start=1):
        freqs = {}
        for tier_csv, tier_label in zip(TIER_CSV, TIER_LABELS):
            sub = cross.loc[
                (cross["K"] == top_k)
                & (cross["consensus_threshold"] == tier_csv)
                & (cross["gene"].astype(str) == gene)
            ]
            freqs[tier_label] = (
                float(sub["consensus_frequency"].iloc[0]) if len(sub) else np.nan
            )
        rank_4of6_val = (
            int(base.reset_index(drop=True).index[base["gene"] == gene][0]) + 1
            if gene in base["gene"].values
            else np.nan
        )
        rows.append(
            {
                "plot_rank": rank,
                "gene": gene,
                "rank_by_ge4of6_freq": rank_4of6_val,
                "freq_ge4of6": freqs[">=4/6"],
                "freq_ge5of6": freqs[">=5/6"],
                "freq_6of6": freqs["6/6"],
                "selection_rule": "manual" if manual_genes else f"top_{top_n}_by_ge4of6_freq",
            }
        )
    detail = pd.DataFrame(rows)
    return gene_list, detail


def plot_panel_b(
    ax: plt.Axes,
    cross: pd.DataFrame,
    gene_list: list[str],
    top_k: int,
) -> None:
    y = np.arange(len(gene_list))
    bar_h = BAR_HEIGHT
    offsets = {">=4/6": -bar_h, ">=5/6": 0.0, "6/6": bar_h}

    for tier_csv, tier_label in zip(TIER_CSV, TIER_LABELS):
        values = []
        for gene in gene_list:
            sub = cross.loc[
                (cross["K"] == top_k)
                & (cross["consensus_threshold"] == tier_csv)
                & (cross["gene"].astype(str) == gene)
            ]
            values.append(float(sub["consensus_frequency"].iloc[0]) if len(sub) else 0.0)
        bars = ax.barh(
            y + offsets[tier_label],
            values,
            height=bar_h,
            color=TIER_COLORS[tier_label],
            label=tier_label,
            alpha=0.9,
        )
        for bar, value in zip(bars, values):
            if value <= 0:
                continue
            ax.text(
                value + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=ANNOTATION_SIZE,
                color="black",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(gene_list, fontsize=FONT_SIZE)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Cross-model consensus frequency", fontsize=FONT_SIZE)
    ax.set_ylabel("Gene", fontsize=FONT_SIZE)
    ax.tick_params(axis="x", labelsize=FONT_SIZE)
    ax.legend(loc="lower right", frameon=True, fontsize=LEGEND_SIZE)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_title(
        f"Cross-model consensus frequency (K = {top_k}, repeated CV)",
        fontsize=TITLE_SIZE,
    )


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cross = load_cross_model(results_dir)
    gene_list, detail = select_genes(
        cross,
        args.top_k,
        args.top_n,
        args.min_freq_4of6,
        args.genes,
        args.highlight_genes,
    )
    if not gene_list:
        raise ValueError("没有选出任何基因，请放宽 --min-freq-4of6 或检查数据")

    list_path = output_dir / f"{args.prefix}_gene_list.csv"
    detail.to_csv(list_path, index=False)

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    )

    fig, ax = plt.subplots(
        figsize=(8.5, max(FIG_MIN_HEIGHT, FIG_HEIGHT_PER_GENE * len(gene_list)))
    )
    plot_panel_b(ax, cross, gene_list, args.top_k)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = output_dir / f"{args.prefix}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"已保存: {out}")
    plt.close(fig)

    print(f"已保存基因列表: {list_path}")
    print("\n入选基因及筛选依据（>=4/6 排名 / 三档频率）:")
    print(
        detail.to_string(
            index=False,
            columns=[
                "plot_rank",
                "gene",
                "rank_by_ge4of6_freq",
                "freq_ge4of6",
                "freq_ge5of6",
                "freq_6of6",
                "selection_rule",
            ],
        )
    )


if __name__ == "__main__":
    main()
