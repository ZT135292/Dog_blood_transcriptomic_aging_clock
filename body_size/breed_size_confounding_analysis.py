"""Summarize breed / size / study structure and related confounding checks."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "breed_size_confounding_results"

SIZE_ORDER = ["Small", "Medium", "Standard", "Large", "Giant"]
CORE_GENES = [
    "GZMK",
    "PLEKHA5",
    "PTPN13",
    "RCAN2",
    "GCNT4",
    "GSTM3",
    "MYO6",
    "SLC6A4",
]
TOP100_FILE = (
    SCRIPT_DIR.parent
    / "all_coefficients"
    / "coefficients_HPO_all_top100.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Breed/size/study confounding analysis for canine aging clocks."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=SCRIPT_DIR / "total_metadata_weight.txt",
    )
    parser.add_argument(
        "--expression",
        type=Path,
        default=SCRIPT_DIR.parent
        / "all_coefficients"
        / "total_count_sort_ortholog_TMM_logcpm_normal_ref.txt",
        help="样本×基因 logCPM 矩阵（行=样本，列=基因）。",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=SCRIPT_DIR
        / "aging_clock_results_size_augmented_comparison"
        / "ElasticNet"
        / "paired_oof_predictions.csv",
        help="Elastic Net Study-LOO OOF 预测（baseline 行用于混杂分析）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument(
        "--skip-pca-combat",
        action="store_true",
        help="跳过仅用于 PCA 的全样本 ComBat 及 raw vs ComBat 对比图。",
    )
    parser.add_argument(
        "--pca-combat-min-batch-n",
        type=int,
        default=2,
        help="ComBat（PCA 探索）中 batch 样本数低于该值时保留 raw 表达、不参与校正。",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def simplify_sex(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("female"):
        return "Female"
    if text.startswith("male"):
        return "Male"
    return "Unknown"


def load_top100_genes(path: Path) -> list[str]:
    if not path.exists():
        return []
    table = pd.read_csv(path)
    genes: set[str] = set()
    for column in table.columns:
        genes.update(table[column].dropna().astype(str).tolist())
    return sorted(genes)


def load_expression_matrix(path: Path, sample_ids: list[str]) -> pd.DataFrame:
    """读取 logCPM 矩阵并返回样本×基因 DataFrame。"""
    expr = pd.read_csv(path, sep="\t", index_col=0)
    expr.index = expr.index.astype(str)
    expr.columns = expr.columns.astype(str)
    missing = sorted(set(sample_ids) - set(expr.columns))
    if missing:
        raise ValueError(f"表达矩阵缺少样本: {missing[:5]} ... (共 {len(missing)} 个)")
    sample_matrix = expr[sample_ids].T.copy()
    sample_matrix.index = sample_matrix.index.astype(str)
    sample_matrix.columns = sample_matrix.columns.astype(str)
    return sample_matrix


def prepare_normal_metadata(metadata_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path, sep="\t")
    meta["sample_id"] = meta["sample_id"].astype(str)
    normal = meta[meta["condition"].eq("Normal")].copy()
    normal["sex_simple"] = normal["sex"].map(simplify_sex)

    if "size_class_DAP" in normal.columns:
        size = normal["size_class_DAP"].astype("string").str.strip()
        size = size.mask(size.isin(["", "None", "NA", "NaN", "nan"]))
        invalid = sorted(set(size.dropna()) - set(SIZE_ORDER))
        if invalid:
            raise ValueError(f"未识别的 size_class_DAP: {invalid}")
        normal["size_class"] = pd.Categorical(size, categories=SIZE_ORDER, ordered=True)
    else:
        normal["size_class"] = pd.Series(pd.NA, index=normal.index)

    normal["age_bin"] = pd.cut(
        normal["Chronological_Age"],
        bins=[0, 3, 6, 9, 12, 20],
        labels=["0-3", "3-6", "6-9", "9-12", "12+"],
        right=False,
    )
    normal["is_wolf_related"] = (
        normal["breed"].astype(str).str.contains("Wolf", case=False, na=False)
        | normal["treatment"].astype(str).str.contains("Wolf", case=False, na=False)
        | normal.get("group", pd.Series(dtype=str)).astype(str).str.contains(
            "Wolf", case=False, na=False
        )
    )
    return normal


def build_composition_tables(normal: pd.DataFrame, output_dir: Path) -> None:
    study_comp = (
        normal.groupby(["batch", "breed", "size_class_DAP"], observed=True)
        .agg(
            n=("sample_id", "count"),
            age_min=("Chronological_Age", "min"),
            age_median=("Chronological_Age", "median"),
            age_max=("Chronological_Age", "max"),
        )
        .reset_index()
        .sort_values(["batch", "breed"])
    )
    study_comp.to_csv(output_dir / "study_breed_size_composition.csv", index=False)

    age_size = pd.crosstab(
        normal["age_bin"],
        normal["size_class_DAP"],
        dropna=False,
    )
    age_size.to_csv(output_dir / "age_bin_by_size_class.csv")

    breed_size = pd.crosstab(normal["breed"], normal["size_class_DAP"], dropna=False)
    breed_size.to_csv(output_dir / "breed_by_size_class.csv")

    batch_summary = (
        normal.groupby("batch", observed=True)
        .agg(
            n=("sample_id", "count"),
            n_breeds=("breed", "nunique"),
            breeds=("breed", lambda x: ";".join(sorted(set(x.astype(str))))),
            n_sizes=("size_class_DAP", "nunique"),
            age_min=("Chronological_Age", "min"),
            age_max=("Chronological_Age", "max"),
        )
        .reset_index()
    )
    batch_summary.to_csv(output_dir / "study_summary.csv", index=False)


def fit_nested_r2(
    y: np.ndarray,
    formulas: list[str],
    data: pd.DataFrame,
) -> dict[str, float]:
    """对同一响应变量拟合嵌套 OLS，返回各模型的 R²。"""
    results: dict[str, float] = {}
    for formula in formulas:
        try:
            fitted = smf.ols(formula, data=data).fit()
            results[formula] = float(fitted.rsquared)
        except Exception:
            results[formula] = np.nan
    return results


def variance_partition_gene(
    gene_series: pd.Series,
    meta: pd.DataFrame,
    gene_name: str,
) -> dict:
    frame = meta.set_index("sample_id").join(
        gene_series.rename("expression"),
        how="inner",
    )
    frame = frame.reset_index()
    frame = frame.dropna(subset=["expression", "Chronological_Age", "batch", "breed"])

    formulas = [
        "expression ~ Chronological_Age",
        "expression ~ C(batch)",
        "expression ~ C(breed)",
        "expression ~ Chronological_Age + C(batch)",
        "expression ~ Chronological_Age + C(breed)",
        "expression ~ Chronological_Age + C(batch) + C(breed)",
    ]
    r2 = fit_nested_r2(frame["expression"].to_numpy(), formulas, frame)
    row = {"gene": gene_name}
    row["r2_age"] = r2.get("expression ~ Chronological_Age", np.nan)
    row["r2_batch"] = r2.get("expression ~ C(batch)", np.nan)
    row["r2_breed"] = r2.get("expression ~ C(breed)", np.nan)
    row["r2_age_batch"] = r2.get(
        "expression ~ Chronological_Age + C(batch)", np.nan
    )
    row["r2_age_breed"] = r2.get(
        "expression ~ Chronological_Age + C(breed)", np.nan
    )
    row["r2_age_batch_breed"] = r2.get(
        "expression ~ Chronological_Age + C(batch) + C(breed)", np.nan
    )
    row["delta_r2_age_after_batch"] = row["r2_age_batch"] - row["r2_batch"]
    return row


def run_gene_variance_partition(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    genes: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for gene in genes:
        if gene not in expr.columns:
            continue
        rows.append(variance_partition_gene(expr[gene], meta, gene))
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "gene_variance_partition.csv", index=False)

    summary = {
        "n_genes": int(len(result)),
        "mean_r2_age": float(result["r2_age"].mean()),
        "mean_r2_batch": float(result["r2_batch"].mean()),
        "mean_r2_breed": float(result["r2_breed"].mean()),
        "mean_r2_age_batch": float(result["r2_age_batch"].mean()),
        "mean_delta_r2_age_after_batch": float(
            result["delta_r2_age_after_batch"].mean()
        ),
    }
    pd.DataFrame([summary]).to_csv(
        output_dir / "gene_variance_partition_summary.csv", index=False
    )
    return result


def run_pca_analysis(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    genes: list[str],
    output_dir: Path,
    prefix: str,
) -> pd.DataFrame:
    use_genes = [gene for gene in genes if gene in expr.columns]
    if len(use_genes) < 2:
        raise ValueError(f"{prefix}: 可用于 PCA 的基因不足。")

    matrix = expr[use_genes].copy()
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaled = StandardScaler().fit_transform(matrix.values)
    pca = PCA(n_components=min(10, matrix.shape[0], matrix.shape[1]), random_state=66)
    pcs = pca.fit_transform(scaled)

    pc_cols = [f"PC{i + 1}" for i in range(pcs.shape[1])]
    pc_df = pd.DataFrame(pcs, columns=pc_cols, index=matrix.index)
    pc_df = pc_df.reset_index().rename(columns={"index": "sample_id"})
    meta_merge = meta.copy()
    if "sample_id" not in meta_merge.columns:
        meta_merge = meta_merge.reset_index()
    if meta_merge.index.name == "sample_id":
        meta_merge = meta_merge.reset_index()
    pc_df = pc_df.merge(
        meta_merge[
            [
                "sample_id",
                "Chronological_Age",
                "batch",
                "breed",
                "size_class_DAP",
                "sex_simple",
            ]
        ],
        on="sample_id",
        how="left",
    )
    pc_df.to_csv(output_dir / f"{prefix}_pca_coordinates.csv", index=False)

    explained = pd.DataFrame(
        {
            "PC": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    explained.to_csv(output_dir / f"{prefix}_pca_explained_variance.csv", index=False)

    age_r2 = np.nan
    if pcs.shape[1] > 0:
        aligned_age = meta.set_index("sample_id").loc[matrix.index, "Chronological_Age"].values
        age_r2 = float(pearsonr(aligned_age, pcs[:, 0]).statistic ** 2)
    pc1_pc2_age_r2 = np.nan
    if pcs.shape[1] >= 2:
        from sklearn.linear_model import LinearRegression

        aligned_age = meta.set_index("sample_id").loc[matrix.index, "Chronological_Age"].values
        lr = LinearRegression()
        lr.fit(pcs[:, :2], aligned_age)
        pc1_pc2_age_r2 = float(lr.score(pcs[:, :2], aligned_age))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_specs = [
        ("Chronological_Age", "Age (years)", "viridis"),
        ("batch", "Study (batch)", "tab10"),
        ("breed", "Breed", "tab20"),
        ("size_class_DAP", "Body-size class", "Set1"),
    ]
    for ax, (column, title, cmap) in zip(axes.flat, plot_specs):
        plot_data = pc_df.copy()
        if column == "Chronological_Age":
            sc = ax.scatter(
                plot_data["PC1"],
                plot_data["PC2"],
                c=plot_data[column],
                cmap=cmap,
                s=45,
                alpha=0.85,
            )
            plt.colorbar(sc, ax=ax, label=title)
        else:
            sns.scatterplot(
                data=plot_data,
                x="PC1",
                y="PC2",
                hue=column,
                palette=cmap,
                s=45,
                alpha=0.85,
                ax=ax,
                legend=False,
            )
        ax.set_title(f"Colored by {title}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    fig.suptitle(f"PCA on {prefix} (n={len(use_genes)} genes)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_pca_by_metadata.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{prefix}_pca_by_metadata.pdf", bbox_inches="tight")
    plt.close(fig)

    pc_partition_rows = []
    for pc_name in ["PC1", "PC2"]:
        frame = pc_df[["Chronological_Age", "batch", "breed", pc_name]].copy()
        formulas = [
            f"{pc_name} ~ Chronological_Age",
            f"{pc_name} ~ C(batch)",
            f"{pc_name} ~ C(breed)",
            f"{pc_name} ~ Chronological_Age + C(batch)",
        ]
        r2 = fit_nested_r2(frame[pc_name].to_numpy(), formulas, frame)
        pc_partition_rows.append(
            {
                "pc": pc_name,
                "r2_age": r2.get(f"{pc_name} ~ Chronological_Age", np.nan),
                "r2_batch": r2.get(f"{pc_name} ~ C(batch)", np.nan),
                "r2_breed": r2.get(f"{pc_name} ~ C(breed)", np.nan),
                "r2_age_batch": r2.get(
                    f"{pc_name} ~ Chronological_Age + C(batch)", np.nan
                ),
            }
        )
    pc_partition = pd.DataFrame(pc_partition_rows)
    pc_partition.to_csv(output_dir / f"{prefix}_pc_variance_partition.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "prefix": prefix,
                "n_genes": len(use_genes),
                "pc1_age_r2": age_r2,
                "pc1_pc2_age_r2": pc1_pc2_age_r2,
            }
        ]
    )
    summary.to_csv(output_dir / f"{prefix}_pca_summary.csv", index=False)
    return pc_df, pca


def apply_combat_for_exploratory_pca(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    min_batch_n: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """
    在全 Normal 样本上运行 ComBat（仅用于探索性 PCA，不用于时钟训练）。

    neuroCombat 输入为基因×样本；生物学协变量默认保护 age 与 sex。
    样本数 < min_batch_n 的 batch 不参与 ComBat，保留 TMM logCPM。
    """
    from neuroCombat import neuroCombat

    gene_columns = [str(c) for c in expr.columns]
    sample_ids = [str(s) for s in expr.index]
    meta_aligned = meta.set_index("sample_id").reindex(sample_ids)
    if meta_aligned["batch"].isna().any():
        missing = meta_aligned.index[meta_aligned["batch"].isna()].tolist()
        raise ValueError(f"ComBat PCA: 元数据缺少 batch: {missing[:5]}")

    batch_col = "batch"
    age_col = "Chronological_Age"
    sex_col = "sex_simple"

    batches = meta_aligned[batch_col].astype(str)
    batch_counts = batches.value_counts()
    thin_batches = batch_counts[batch_counts < min_batch_n].index.tolist()
    is_thin = batches.isin(thin_batches)

    corrected = expr.copy()
    corrected = corrected.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    eligible_idx = meta_aligned.index[~is_thin]
    thin_idx = meta_aligned.index[is_thin]
    unique_batches = sorted(batches.loc[eligible_idx].unique())

    info: dict = {
        "purpose": "exploratory_pca_only_not_for_clock_training",
        "n_samples_total": len(sample_ids),
        "n_genes": len(gene_columns),
        "min_batch_n": min_batch_n,
        "thin_batches_skipped": thin_batches,
        "n_samples_combat_corrected": int(len(eligible_idx)),
        "n_samples_kept_raw": int(len(thin_idx)),
        "batches_in_combat": unique_batches,
        "continuous_covariates": [],
        "categorical_covariates": [],
        "combat_success": False,
    }

    if len(unique_batches) < 2:
        info["error"] = "fewer_than_two_batches_for_combat"
        print(
            f"  - ComBat PCA: 可用于校正的 batch < 2（{unique_batches}），跳过 ComBat。"
        )
        return corrected, info

    eligible_meta = meta_aligned.loc[eligible_idx]
    eligible_expr = corrected.loc[eligible_idx, gene_columns]
    dat = eligible_expr.values.T.astype(float)

    cont_covs = [age_col] if age_col in eligible_meta.columns else []
    cat_covs = [sex_col] if sex_col in eligible_meta.columns else []

    cat_mapping: dict[str, dict[str, str]] = {}
    effective_cat: list[str] = []
    for col in cat_covs:
        vals = eligible_meta[col].astype(str)
        counts = vals.value_counts()
        rare = counts[counts < min_batch_n].index.tolist()
        mapping = {v: ("Other" if v in rare else v) for v in counts.index}
        mapped = vals.map(mapping)
        if mapped.nunique() < 2:
            print(f"  - ComBat PCA: 协变量 '{col}' 仅剩 1 个水平，跳过。")
            continue
        cat_mapping[col] = mapping
        effective_cat.append(col)

    def _make_covars_df(df: pd.DataFrame, use_cat: list[str], use_cont: list[str]) -> pd.DataFrame:
        cov = {batch_col: df[batch_col].astype(str).values}
        for col in use_cont:
            num = pd.to_numeric(df[col], errors="coerce")
            cov[col] = num.fillna(num.mean()).values
        for col in use_cat:
            mapped = df[col].astype(str).map(cat_mapping[col])
            cov[col] = mapped.fillna("Other").values
        return pd.DataFrame(cov)

    corrected_expr = None
    used_cont: list[str] = []
    used_cat: list[str] = []
    for attempt, (use_cat, use_cont) in enumerate([
        (effective_cat, cont_covs),
        ([], cont_covs),
        ([], []),
    ]):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                result = neuroCombat(
                    dat=dat,
                    covars=_make_covars_df(eligible_meta, use_cat, use_cont),
                    batch_col=batch_col,
                    continuous_cols=use_cont,
                    categorical_cols=use_cat,
                )
            corrected_expr = result["data"].T
            nan_mask = np.isnan(corrected_expr)
            if nan_mask.any():
                corrected_expr = np.where(
                    nan_mask, eligible_expr.values, corrected_expr
                )
            used_cont = list(use_cont)
            used_cat = list(use_cat)
            if attempt > 0:
                print(
                    f"    -> ComBat PCA 降级成功（分类={use_cat}，连续={use_cont}）"
                )
            break
        except Exception as exc:
            if attempt == 2:
                info["error"] = str(exc)
                print(f"  - ComBat PCA: 全部尝试失败（{exc}），保留 raw 表达。")
                return corrected, info
            print(f"    -> ComBat PCA 尝试 {attempt + 1} 失败（{exc}），降级重试...")

    corrected.loc[eligible_idx, gene_columns] = corrected_expr
    info["continuous_covariates"] = used_cont
    info["categorical_covariates"] = used_cat
    info["combat_success"] = True
    print(
        f"  - ComBat PCA: 校正 {len(eligible_idx)} 样本、"
        f"{len(unique_batches)} 个 batch；"
        f"保留 raw {len(thin_idx)} 样本（极小 batch: {thin_batches}）"
    )
    return corrected, info


def plot_pca_raw_vs_combat_comparison(
    raw_pc_df: pd.DataFrame,
    combat_pc_df: pd.DataFrame,
    raw_pca: PCA,
    combat_pca: PCA,
    output_dir: Path,
    n_genes: int,
) -> None:
    """2×4 面板：上行 raw、下行 ComBat；列分别为 age / batch / breed / size。"""
    plot_specs = [
        ("Chronological_Age", "Age (years)", "viridis"),
        ("batch", "Study (batch)", "tab10"),
        ("breed", "Breed", "tab20"),
        ("size_class_DAP", "Body-size class", "Set1"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    row_labels = ["Raw TMM logCPM", "ComBat-adjusted (PCA exploration only)"]
    pc_dfs = [raw_pc_df, combat_pc_df]
    pcas = [raw_pca, combat_pca]

    for row, (pc_df, pca, row_label) in enumerate(zip(pc_dfs, pcas, row_labels)):
        for col, (column, title, cmap) in enumerate(plot_specs):
            ax = axes[row, col]
            plot_data = pc_df.copy()
            if column == "Chronological_Age":
                sc = ax.scatter(
                    plot_data["PC1"],
                    plot_data["PC2"],
                    c=plot_data[column],
                    cmap=cmap,
                    s=40,
                    alpha=0.85,
                )
                plt.colorbar(sc, ax=ax, label=title, fraction=0.046, pad=0.04)
            else:
                sns.scatterplot(
                    data=plot_data,
                    x="PC1",
                    y="PC2",
                    hue=column,
                    palette=cmap,
                    s=40,
                    alpha=0.85,
                    ax=ax,
                    legend=False,
                )
            ax.set_title(f"{row_label} | {title}")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    fig.suptitle(
        f"PCA comparison: raw vs ComBat (all genes, n={n_genes}; clock not trained on ComBat data)",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    stem = output_dir / "all_genes_pca_raw_vs_combat_comparison"
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def run_pca_combat_comparison(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    genes: list[str],
    output_dir: Path,
    min_batch_n: int = 2,
    raw_pc_df: pd.DataFrame | None = None,
    raw_pca: PCA | None = None,
    raw_prefix: str = "all_genes",
) -> None:
    """全样本 ComBat + raw vs ComBat PCA 及 PC 方差分解对比表。"""
    print("PCA 探索性 ComBat（全 Normal，仅用于附图，不用于时钟）...")
    expr_combat, combat_info = apply_combat_for_exploratory_pca(
        expr, meta, min_batch_n=min_batch_n
    )
    (output_dir / "pca_combat_run_info.json").write_text(
        json.dumps(combat_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_genes = [g for g in genes if g in expr.columns]

    if raw_pc_df is None or raw_pca is None:
        print(f"  - Raw PCA（prefix={raw_prefix}）...")
        raw_pc_df, raw_pca = run_pca_analysis(
            expr, meta, all_genes, output_dir, prefix=raw_prefix
        )
    else:
        print(f"  - 复用已有 raw PCA 结果（prefix={raw_prefix}）")

    if not combat_info.get("combat_success"):
        print("  - ComBat 未成功，跳过 ComBat PCA 与对比图。")
        return

    print("  - ComBat PCA（prefix=all_genes_combat）...")
    combat_pc_df, combat_pca = run_pca_analysis(
        expr_combat, meta, all_genes, output_dir, prefix="all_genes_combat"
    )

    plot_pca_raw_vs_combat_comparison(
        raw_pc_df,
        combat_pc_df,
        raw_pca,
        combat_pca,
        output_dir,
        len(all_genes),
    )

    raw_part = pd.read_csv(output_dir / f"{raw_prefix}_pc_variance_partition.csv")
    combat_part = pd.read_csv(output_dir / "all_genes_combat_pc_variance_partition.csv")
    raw_part["matrix"] = "raw"
    combat_part["matrix"] = "combat_pca_only"
    comparison = pd.concat([raw_part, combat_part], ignore_index=True)
    comparison.to_csv(output_dir / "pca_raw_vs_combat_pc_partition.csv", index=False)

    raw_sum = pd.read_csv(output_dir / f"{raw_prefix}_pca_summary.csv")
    combat_sum = pd.read_csv(output_dir / "all_genes_combat_pca_summary.csv")
    raw_sum["matrix"] = "raw"
    combat_sum["matrix"] = "combat_pca_only"
    pd.concat([raw_sum, combat_sum], ignore_index=True).to_csv(
        output_dir / "pca_raw_vs_combat_summary.csv", index=False
    )
    print("  - 已保存 all_genes_pca_raw_vs_combat_comparison.png/.pdf")


def run_clock_confounding(
    predictions_path: Path,
    output_dir: Path,
    variant: str = "baseline",
) -> None:
    if not predictions_path.exists():
        print(f"  - 跳过时钟混杂分析：未找到 {predictions_path}")
        return

    pred = pd.read_csv(predictions_path)
    if "variant" in pred.columns:
        pred = pred[pred["variant"].eq(variant)].copy()
    pred["prediction_error"] = pred["predicted_age"] - pred["actual_age"]
    pred["sex_simple"] = pred["sex"].map(simplify_sex)
    if "age_bin" not in pred.columns:
        pred["age_bin"] = pd.cut(
            pred["actual_age"],
            bins=[0, 3, 6, 9, 12, 20],
            labels=["0-3", "3-6", "6-9", "9-12", "12+"],
            right=False,
        )

    formulas = [
        "predicted_age ~ actual_age",
        "predicted_age ~ actual_age + C(batch)",
        "predicted_age ~ actual_age + C(size_class)",
        "predicted_age ~ actual_age * C(size_class)",
        "predicted_age ~ actual_age + C(batch) + C(size_class)",
        "predicted_age ~ actual_age * C(size_class) + C(batch)",
    ]

    coef_rows = []
    summary_rows = []
    for formula in formulas:
        fitted = smf.ols(formula, data=pred).fit(cov_type="HC3")
        summary_rows.append(
            {
                "formula": formula,
                "n": int(fitted.nobs),
                "r_squared": float(fitted.rsquared),
                "adj_r_squared": float(fitted.rsquared_adj),
            }
        )
        conf = fitted.conf_int()
        for term in fitted.params.index:
            coef_rows.append(
                {
                    "formula": formula,
                    "term": term,
                    "estimate": float(fitted.params[term]),
                    "std_error_hc3": float(fitted.bse[term]),
                    "ci_low": float(conf.loc[term, 0]),
                    "ci_high": float(conf.loc[term, 1]),
                    "p_value": float(fitted.pvalues[term]),
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "clock_prediction_model_summary.csv", index=False
    )
    pd.DataFrame(coef_rows).to_csv(
        output_dir / "clock_prediction_model_coefficients.csv", index=False
    )

    stratified = (
        pred.groupby(["size_class", "age_bin"], observed=True)
        .agg(
            n=("sample_id", "count"),
            mae=("prediction_error", lambda x: np.mean(np.abs(x))),
            mean_error=("prediction_error", "mean"),
        )
        .reset_index()
    )
    stratified.to_csv(output_dir / "clock_mae_by_size_and_age_bin.csv", index=False)

    if pred["size_class"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=pred.dropna(subset=["size_class"]),
            x="size_class",
            y="prediction_error",
            order=[s for s in SIZE_ORDER if s in pred["size_class"].unique()],
            ax=ax,
        )
        sns.stripplot(
            data=pred.dropna(subset=["size_class"]),
            x="size_class",
            y="prediction_error",
            order=[s for s in SIZE_ORDER if s in pred["size_class"].unique()],
            color="black",
            alpha=0.5,
            ax=ax,
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Predicted age − chronological age (years)")
        ax.set_xlabel("Body-size class")
        ax.set_title("Elastic Net OOF prediction error by size class")
        fig.tight_layout()
        fig.savefig(output_dir / "clock_prediction_error_by_size.png", dpi=300)
        plt.close(fig)


def run_core_gene_models(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    present_sizes = [
        size
        for size in SIZE_ORDER
        if meta["size_class"].eq(size).any()
    ]
    if len(present_sizes) < 2:
        print("  - 跳过核心基因分析：体型类别不足。")
        return pd.DataFrame()

    ref = present_sizes[0]
    size_term = f"C(size_class, Treatment(reference='{ref}'))"

    for gene in CORE_GENES:
        if gene not in expr.columns:
            continue
        frame = meta.set_index("sample_id").join(
            expr[gene].rename("expression"),
            how="inner",
        )
        frame = frame.reset_index()
        frame = frame.dropna(subset=["expression", "Chronological_Age", "batch"])
        frame["size_class"] = pd.Categorical(
            frame["size_class"], categories=present_sizes, ordered=True
        )

        formulas = [
            f"expression ~ Chronological_Age",
            f"expression ~ Chronological_Age + {size_term}",
            f"expression ~ Chronological_Age + {size_term} + C(batch)",
            f"expression ~ Chronological_Age + {size_term} + C(batch) + C(sex_simple)",
        ]
        for formula in formulas:
            fitted = smf.ols(formula, data=frame).fit(cov_type="HC3")
            conf = fitted.conf_int()
            for term in fitted.params.index:
                rows.append(
                    {
                        "gene": gene,
                        "formula": formula,
                        "term": term,
                        "estimate": float(fitted.params[term]),
                        "std_error_hc3": float(fitted.bse[term]),
                        "ci_low": float(conf.loc[term, 0]),
                        "ci_high": float(conf.loc[term, 1]),
                        "p_value": float(fitted.pvalues[term]),
                        "model_r2": float(fitted.rsquared),
                        "n": int(fitted.nobs),
                        "reference_size": ref,
                    }
                )

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "core_gene_expression_models.csv", index=False)

    # Small vs Large（二分类）补充分析
    binary_rows = []
    frame_bin = meta.copy()
    frame_bin = frame_bin[frame_bin["size_class"].isin(["Small", "Large"])].copy()
    frame_bin["size_binary"] = frame_bin["size_class"].map(
        {"Small": "Small", "Large": "Large"}
    )
    if frame_bin["size_binary"].nunique() == 2:
        for gene in CORE_GENES:
            if gene not in expr.columns:
                continue
            sub = frame_bin.copy()
            sub = sub.set_index("sample_id")
            sub["expression"] = expr[gene]
            sub = sub.reset_index()
            formula = (
                "expression ~ Chronological_Age + C(size_binary) + C(batch) + C(sex_simple)"
            )
            fitted = smf.ols(formula, data=sub).fit(cov_type="HC3")
            conf = fitted.conf_int()
            for term in fitted.params.index:
                binary_rows.append(
                    {
                        "gene": gene,
                        "formula": formula,
                        "term": term,
                        "estimate": float(fitted.params[term]),
                        "p_value": float(fitted.pvalues[term]),
                        "model_r2": float(fitted.rsquared),
                        "n": int(fitted.nobs),
                    }
                )
        pd.DataFrame(binary_rows).to_csv(
            output_dir / "core_gene_small_vs_large_models.csv", index=False
        )

        plot_genes = [g for g in CORE_GENES if g in expr.columns]
        if plot_genes:
            plot_df = frame_bin.copy()
            for gene in plot_genes:
                plot_df = plot_df.merge(
                    expr[[gene]].rename(columns={gene: gene}),
                    left_on="sample_id",
                    right_index=True,
                    how="left",
                )
            melted = plot_df.melt(
                id_vars=["sample_id", "size_binary", "Chronological_Age", "batch"],
                value_vars=plot_genes,
                var_name="gene",
                value_name="expression",
            )
            # 仅 6–9 岁窗口（batch6 中 Small/Large 重叠较多）
            window = melted[
                melted["Chronological_Age"].between(6, 9, inclusive="both")
            ]
            if not window.empty:
                g = sns.catplot(
                    data=window,
                    x="size_binary",
                    y="expression",
                    col="gene",
                    col_wrap=4,
                    sharey=False,
                    height=3,
                    aspect=1.1,
                )
                g.set_axis_labels("Size class", "logCPM")
                g.fig.subplots_adjust(top=0.9)
                g.fig.suptitle("Core genes (ages 6–9 years): Small vs Large")
                g.savefig(
                    output_dir / "core_gene_small_vs_large_age6_9.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(g.fig)

    return result


def main() -> None:
    args = parse_args()
    metadata_path = resolve_path(args.metadata)
    expression_path = resolve_path(args.expression)
    predictions_path = resolve_path(args.predictions)
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载元数据与表达矩阵 ...")
    normal = prepare_normal_metadata(metadata_path)
    expr = load_expression_matrix(expression_path, normal["sample_id"].tolist())

    build_composition_tables(normal, output_dir)
    print(f"  - 组成表已写入 {output_dir}")

    top100_genes = load_top100_genes(TOP100_FILE)
    if not top100_genes:
        top100_genes = [col for col in expr.columns if col in CORE_GENES]
    top100_present = [gene for gene in top100_genes if gene in expr.columns]
    if len(top100_present) < 2:
        raise ValueError(
            f"top100 基因在表达矩阵中可用数量不足（n={len(top100_present)}）。"
        )

    all_genes = [str(g) for g in expr.columns]
    print(f"PCA（全基因 raw n={len(all_genes)}；top100 n={len(top100_present)}）...")
    raw_pc_df, raw_pca = run_pca_analysis(
        expr, normal, all_genes, output_dir, prefix="all_genes"
    )
    print(f"  - top100 高贡献基因 PCA（n={len(top100_present)}）...")
    run_pca_analysis(
        expr, normal, top100_present, output_dir, prefix="top100"
    )

    if not args.skip_pca_combat:
        run_pca_combat_comparison(
            expr,
            normal,
            all_genes,
            output_dir,
            min_batch_n=args.pca_combat_min_batch_n,
            raw_pc_df=raw_pc_df,
            raw_pca=raw_pca,
            raw_prefix="all_genes",
        )
    else:
        print("  - 已跳过 PCA ComBat（--skip-pca-combat）")

    run_gene_variance_partition(expr, normal, top100_genes, output_dir)

    print("时钟预测混杂分析 ...")
    run_clock_confounding(predictions_path, output_dir)

    print("核心基因 × 体型回归 ...")
    run_core_gene_models(expr, normal, output_dir)

    run_info = {
        "metadata": str(metadata_path),
        "expression": str(expression_path),
        "predictions": str(predictions_path),
        "n_normal": int(len(normal)),
        "n_breeds": int(normal["breed"].nunique()),
        "size_counts": {
            str(k): int(v)
            for k, v in normal["size_class_DAP"].value_counts(dropna=False).items()
        },
        "core_genes": CORE_GENES,
        "pca_all_genes": len(all_genes),
        "pca_top100_genes": len(top100_present),
        "pca_combat_exploration": not args.skip_pca_combat,
        "top100_source": str(TOP100_FILE),
        "seed": args.seed,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"完成。结果目录: {output_dir}")


if __name__ == "__main__":
    main()
