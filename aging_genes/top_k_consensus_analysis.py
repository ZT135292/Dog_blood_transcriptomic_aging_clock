"""Top-K sensitivity and cross-model consensus from the six clocks' HPO gene rankings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_FILES = {
    "ElasticNet": (
        "aging_clock_results_Elastic_Net_pipeline_foldwise_TMM_logCPM/HPO/"
        "coefficients_HPO_all_normal.csv"
    ),
    "LASSO": (
        "aging_clock_results_LASSO_pipeline_foldwise_TMM_logCPM/HPO/"
        "coefficients_HPO_all_normal.csv"
    ),
    "MoE": (
        "aging_clock_results_MoE_pipeline_foldwise_TMM_logCPM/HPO/"
        "gene_importances_HPO_all_normal.csv"
    ),
    "RF": (
        "aging_clock_results_Random_Forest_pipeline_foldwise_TMM_logCPM/HPO/"
        "gene_importances_HPO_all_normal.csv"
    ),
    "SVR": (
        "aging_clock_results_SVR_pipeline_foldwise_TMM_logCPM/HPO/"
        "coefficients_HPO_all_normal.csv"
    ),
    "XGBoost": (
        "aging_clock_results_XGBoost_pipeline_foldwise_TMM_logCPM/HPO/"
        "gene_importances_HPO_all_normal.csv"
    ),
}

SCORE_COLUMNS = (
    "abs_coefficient",
    "coefficient",
    "importance",
    "importance_mean",
    "feature_importance",
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    results_root = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Top 50/100/150/200 交集及至少 4/6 模型共识分析"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=results_root,
        help="包含六个 aging_clock_results_* 目录的路径（默认：脚本上一级目录）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "top_k_consensus_results",
        help="输出目录（默认：脚本所在目录下的 top_k_consensus_results）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200],
        help="需要分析的 Top-K 阈值",
    )
    parser.add_argument(
        "--min-models",
        type=int,
        default=4,
        help="主要共识定义所需的最少模型数（默认：4）",
    )
    parser.add_argument(
        "--reference-genes",
        type=Path,
        default=results_root / "column_intersection_result.txt",
        help="可选的原核心基因文件，用于检查其在各阈值下的保留情况（默认：上一级目录）",
    )
    return parser.parse_args()


def load_ranked_genes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到模型重要性文件：{path}")
    df = pd.read_csv(path)
    if "gene" not in df.columns:
        raise ValueError(f"{path} 缺少 gene 列；现有列：{list(df.columns)}")

    score_col = next((column for column in SCORE_COLUMNS if column in df.columns), None)
    if score_col is None:
        if "rank" in df.columns:
            ranked = df.sort_values("rank", ascending=True).copy()
            ranked["_ranking_score"] = -pd.to_numeric(ranked["rank"], errors="coerce")
        else:
            raise ValueError(
                f"{path} 中找不到可识别的重要性列；支持：{SCORE_COLUMNS} 或 rank"
            )
    else:
        scores = pd.to_numeric(df[score_col], errors="coerce")
        if score_col == "coefficient":
            scores = scores.abs()
        ranked = df.assign(_ranking_score=scores).sort_values(
            "_ranking_score", ascending=False, na_position="last"
        )

    ranked["gene"] = ranked["gene"].astype(str).str.strip()
    ranked = ranked[
        ranked["gene"].ne("") & ranked["gene"].ne("nan")
    ].drop_duplicates("gene", keep="first")
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked[["gene", "rank", "_ranking_score"]]


def read_reference_genes(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if df.empty:
            return []
        column = "intersection" if "intersection" in df.columns else df.columns[0]
        return (
            df[column].dropna().astype(str).str.strip().loc[lambda x: x.ne("")].tolist()
        )
    except Exception as exc:
        print(f"警告：无法读取参考基因文件 {path}：{exc}")
        return []


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else np.nan


def analyse_threshold(
    k: int,
    ranked_by_model: dict[str, pd.DataFrame],
    min_models: int,
) -> tuple[dict[str, set[str]], pd.DataFrame, dict[str, object]]:
    selected = {
        model: set(df.head(min(k, len(df)))["gene"])
        for model, df in ranked_by_model.items()
    }
    all_genes = sorted(set().union(*selected.values()))
    rows = []
    for gene in all_genes:
        membership = {model: int(gene in genes) for model, genes in selected.items()}
        supporting = [model for model, present in membership.items() if present]
        rows.append(
            {
                "K": k,
                "gene": gene,
                "n_models": len(supporting),
                "supporting_models": ";".join(supporting),
                **membership,
            }
        )
    consensus = pd.DataFrame(rows).sort_values(
        ["n_models", "gene"], ascending=[False, True]
    )
    strict = set.intersection(*selected.values())
    min_consensus = set(
        consensus.loc[consensus["n_models"] >= min_models, "gene"]
    )
    five_of_six = set(consensus.loc[consensus["n_models"] >= 5, "gene"])
    summary = {
        "K": k,
        **{f"n_{model}": len(genes) for model, genes in selected.items()},
        "strict_6_of_6": len(strict),
        "consensus_5_of_6": len(five_of_six),
        f"consensus_{min_models}_of_6": len(min_consensus),
        "strict_genes": ";".join(sorted(strict)),
        "consensus_5_of_6_genes": ";".join(sorted(five_of_six)),
        f"consensus_{min_models}_of_6_genes": ";".join(sorted(min_consensus)),
    }
    return selected, consensus, summary


def plot_counts(summary: pd.DataFrame, min_models: int, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary["K"], summary["strict_6_of_6"], marker="o", label="6/6 strict")
    ax.plot(summary["K"], summary["consensus_5_of_6"], marker="o", label=">=5/6")
    ax.plot(
        summary["K"],
        summary[f"consensus_{min_models}_of_6"],
        marker="o",
        label=f">={min_models}/6",
    )
    ax.set_xlabel("Top-K threshold")
    ax.set_ylabel("Number of consensus genes")
    ax.set_xticks(summary["K"])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "consensus_gene_counts.png", dpi=300)
    fig.savefig(output_dir / "consensus_gene_counts.pdf")
    plt.close(fig)


def plot_membership_heatmap(
    consensus: pd.DataFrame, k: int, min_models: int, output_dir: Path
) -> None:
    model_columns = list(MODEL_FILES)
    shown = consensus.loc[consensus["n_models"] >= min_models].copy()
    if shown.empty:
        return
    shown = shown.sort_values(["n_models", "gene"], ascending=[False, True])
    height = max(4.0, min(20.0, 0.24 * len(shown)))
    fig, ax = plt.subplots(figsize=(8, height))
    matrix = shown[model_columns].to_numpy()
    ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(model_columns)), model_columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(shown)), shown["gene"])
    ax.set_title(f"Top-{k}: genes selected by >={min_models}/6 models")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / f"membership_heatmap_top{k}.png", dpi=300)
    fig.savefig(output_dir / f"membership_heatmap_top{k}.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    top_k = sorted(set(args.top_k))
    if any(k <= 0 for k in top_k):
        raise ValueError("--top-k 必须全部为正整数")
    if not 1 <= args.min_models <= len(MODEL_FILES):
        raise ValueError("--min-models 必须在 1 到 6 之间")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranked_by_model = {
        model: load_ranked_genes(args.results_root / relative_path)
        for model, relative_path in MODEL_FILES.items()
    }

    rank_rows = []
    for model, ranked in ranked_by_model.items():
        table = ranked.copy()
        table.insert(0, "model", model)
        rank_rows.append(table)
    pd.concat(rank_rows, ignore_index=True).to_csv(
        args.output_dir / "all_model_gene_rankings.csv", index=False
    )

    selections_by_k: dict[int, dict[str, set[str]]] = {}
    consensus_by_k: dict[int, pd.DataFrame] = {}
    summaries = []
    for k in top_k:
        selected, consensus, summary = analyse_threshold(
            k, ranked_by_model, args.min_models
        )
        selections_by_k[k] = selected
        consensus_by_k[k] = consensus
        summaries.append(summary)
        consensus.to_csv(args.output_dir / f"gene_consensus_top{k}.csv", index=False)
        consensus.loc[consensus["n_models"] == 6].to_csv(
            args.output_dir / f"strict_intersection_6of6_top{k}.csv", index=False
        )
        consensus.loc[consensus["n_models"] >= args.min_models].to_csv(
            args.output_dir / f"consensus_{args.min_models}of6_top{k}.csv", index=False
        )
        plot_membership_heatmap(consensus, k, args.min_models, args.output_dir)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_dir / "threshold_sensitivity_summary.csv", index=False)

    jaccard_rows = []
    for model in MODEL_FILES:
        for left_k, right_k in zip(top_k[:-1], top_k[1:]):
            jaccard_rows.append(
                {
                    "set": model,
                    "K_left": left_k,
                    "K_right": right_k,
                    "jaccard": jaccard(
                        selections_by_k[left_k][model],
                        selections_by_k[right_k][model],
                    ),
                }
            )
    for consensus_level in (args.min_models, 5, 6):
        for left_k, right_k in zip(top_k[:-1], top_k[1:]):
            left = set(
                consensus_by_k[left_k].loc[
                    consensus_by_k[left_k]["n_models"] >= consensus_level, "gene"
                ]
            )
            right = set(
                consensus_by_k[right_k].loc[
                    consensus_by_k[right_k]["n_models"] >= consensus_level, "gene"
                ]
            )
            jaccard_rows.append(
                {
                    "set": f">={consensus_level}/6 consensus",
                    "K_left": left_k,
                    "K_right": right_k,
                    "jaccard": jaccard(left, right),
                }
            )
    pd.DataFrame(jaccard_rows).to_csv(
        args.output_dir / "adjacent_threshold_jaccard.csv", index=False
    )

    reference_genes = read_reference_genes(args.reference_genes)
    if reference_genes:
        retention_rows = []
        for gene in reference_genes:
            for k in top_k:
                row = consensus_by_k[k]
                matched = row.loc[row["gene"] == gene]
                retention_rows.append(
                    {
                        "gene": gene,
                        "K": k,
                        "n_models": int(matched["n_models"].iloc[0])
                        if not matched.empty
                        else 0,
                        "in_strict_6of6": bool(
                            not matched.empty and matched["n_models"].iloc[0] == 6
                        ),
                        f"in_consensus_{args.min_models}of6": bool(
                            not matched.empty
                            and matched["n_models"].iloc[0] >= args.min_models
                        ),
                    }
                )
        pd.DataFrame(retention_rows).to_csv(
            args.output_dir / "reference_gene_retention.csv", index=False
        )

    plot_counts(summary_df, args.min_models, args.output_dir)
    print(f"分析完成。结果目录：{args.output_dir}")


if __name__ == "__main__":
    main()
