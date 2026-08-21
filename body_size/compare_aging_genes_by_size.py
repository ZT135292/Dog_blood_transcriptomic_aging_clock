"""Age–expression trajectories and age × size tests for consensus aging genes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import kruskal

from breed_size_confounding_analysis import (
    SIZE_ORDER,
    load_expression_matrix,
    prepare_normal_metadata,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GENES = ["GZMK", "PLEKHA5", "PTPN13", "RCAN2", "GCNT4", "MYO6"]
PALETTE = dict(zip(SIZE_ORDER, sns.color_palette("Set1", len(SIZE_ORDER))))
PLOT_SIZE_ORDER = ["Small", "Medium", "Standard", "Large"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare aging-related gene expression across dog size classes."
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
    )
    parser.add_argument("--genes", nargs="+", default=DEFAULT_GENES)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "aging_gene_expression_by_size_results",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def prepare_long_data(
    metadata_path: Path,
    expression_path: Path,
    requested_genes: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    meta = prepare_normal_metadata(metadata_path)
    expr = load_expression_matrix(expression_path, meta["sample_id"].tolist())

    genes = [gene for gene in requested_genes if gene in expr.columns]
    missing = sorted(set(requested_genes) - set(genes))
    if missing:
        print(f"Warning: expression matrix does not contain: {', '.join(missing)}")
    if not genes:
        raise ValueError("None of the requested genes were found in the expression matrix.")

    frame = meta[
        [
            "sample_id",
            "Chronological_Age",
            "batch",
            "breed",
            "size_class",
            "sex_simple",
        ]
    ].copy()
    frame = frame.dropna(subset=["size_class"]).set_index("sample_id")
    frame = frame.join(expr[genes], how="inner")
    frame.index.name = "sample_id"
    frame = frame.reset_index()
    frame["size_class"] = pd.Categorical(
        frame["size_class"], categories=SIZE_ORDER, ordered=True
    )
    frame["age_centered"] = (
        frame["Chronological_Age"] - frame["Chronological_Age"].mean()
    )

    long = frame.melt(
        id_vars=[
            "sample_id",
            "Chronological_Age",
            "age_centered",
            "batch",
            "breed",
            "size_class",
            "sex_simple",
        ],
        value_vars=genes,
        var_name="gene",
        value_name="expression",
    )
    return long, genes


def joint_wald_test(fitted, terms: list[str]) -> tuple[float, int, float]:
    """Joint Wald test for selected coefficient names."""
    terms = [term for term in terms if term in fitted.params.index]
    if not terms:
        return np.nan, 0, np.nan
    restriction = np.zeros((len(terms), len(fitted.params)))
    for row, term in enumerate(terms):
        restriction[row, fitted.params.index.get_loc(term)] = 1.0
    test = fitted.wald_test(restriction, scalar=True)
    return float(test.statistic), len(terms), float(test.pvalue)


def run_all_size_models(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_rows: list[dict] = []
    test_rows: list[dict] = []
    long = long[long["size_class"].isin(PLOT_SIZE_ORDER)].copy()
    long["size_class"] = long["size_class"].cat.remove_unused_categories()

    for gene, data in long.groupby("gene", observed=True):
        data = data.dropna(
            subset=[
                "expression",
                "age_centered",
                "size_class",
                "batch",
                "sex_simple",
            ]
        ).copy()
        formula = (
            "expression ~ age_centered * "
            "C(size_class, Treatment(reference='Small')) "
            "+ C(batch)"
        )
        fitted = smf.ols(formula, data=data).fit(cov_type="HC3")
        confidence = fitted.conf_int()

        for term in fitted.params.index:
            coefficient_rows.append(
                {
                    "gene": gene,
                    "model": "all_size_classes",
                    "term": term,
                    "estimate": float(fitted.params[term]),
                    "std_error_hc3": float(fitted.bse[term]),
                    "ci_low": float(confidence.loc[term, 0]),
                    "ci_high": float(confidence.loc[term, 1]),
                    "p_value": float(fitted.pvalues[term]),
                    "n": int(fitted.nobs),
                    "r_squared": float(fitted.rsquared),
                }
            )

        size_terms = [
            term
            for term in fitted.params.index
            if term.startswith("C(size_class") and ":age_centered" not in term
        ]
        interaction_terms = [
            term
            for term in fitted.params.index
            if "age_centered:C(size_class" in term
            or "C(size_class" in term
            and ":age_centered" in term
        ]
        for test_name, terms in [
            ("size_at_mean_age", size_terms),
            ("age_by_size_interaction", interaction_terms),
        ]:
            statistic, df, p_value = joint_wald_test(fitted, terms)
            test_rows.append(
                {
                    "gene": gene,
                    "model": "all_size_classes",
                    "test": test_name,
                    "wald_chi2": statistic,
                    "df": df,
                    "p_value": p_value,
                    "n": int(fitted.nobs),
                }
            )

    return pd.DataFrame(coefficient_rows), pd.DataFrame(test_rows)


def run_unadjusted_size_tests(long: pd.DataFrame) -> pd.DataFrame:
    """Kruskal–Wallis test of raw TMM-logCPM expression across four sizes."""
    rows = []
    data = long[long["size_class"].isin(PLOT_SIZE_ORDER)].copy()
    for gene, gene_data in data.groupby("gene", observed=True):
        groups = [
            gene_data.loc[gene_data["size_class"].eq(size), "expression"].dropna()
            for size in PLOT_SIZE_ORDER
        ]
        test = kruskal(*groups)
        rows.append(
            {
                "gene": gene,
                "test": "Kruskal-Wallis_unadjusted_four_sizes",
                "statistic": float(test.statistic),
                "df": len(PLOT_SIZE_ORDER) - 1,
                "p_value": float(test.pvalue),
                "n": int(sum(len(group) for group in groups)),
            }
        )
    return pd.DataFrame(rows)


def format_p_value(p_value: float) -> str:
    if np.isnan(p_value):
        return "NA"
    if p_value < 0.001:
        return f"{p_value:.1e}"
    return f"{p_value:.3f}"


def significance_label(p_value: float) -> str:
    if np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def run_small_large_models(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    subset = long[long["size_class"].isin(["Small", "Large"])].copy()
    subset["size_binary"] = pd.Categorical(
        subset["size_class"].astype(str),
        categories=["Large", "Small"],
        ordered=True,
    )

    for gene, data in subset.groupby("gene", observed=True):
        formula = (
            "expression ~ age_centered * "
            "C(size_binary, Treatment(reference='Large')) "
            "+ C(batch)"
        )
        fitted = smf.ols(formula, data=data).fit(cov_type="HC3")
        confidence = fitted.conf_int()
        target_terms = [
            term
            for term in fitted.params.index
            if "C(size_binary" in term
        ]
        for term in target_terms:
            comparison = (
                "Small_minus_Large_at_mean_age"
                if ":age_centered" not in term and not term.startswith("age_centered:")
                else "Small_minus_Large_age_slope"
            )
            rows.append(
                {
                    "gene": gene,
                    "comparison": comparison,
                    "term": term,
                    "estimate": float(fitted.params[term]),
                    "std_error_hc3": float(fitted.bse[term]),
                    "ci_low": float(confidence.loc[term, 0]),
                    "ci_high": float(confidence.loc[term, 1]),
                    "p_value": float(fitted.pvalues[term]),
                    "n_small": int(data["size_binary"].eq("Small").sum()),
                    "n_large": int(data["size_binary"].eq("Large").sum()),
                    "r_squared": float(fitted.rsquared),
                }
            )
    return pd.DataFrame(rows)


def add_adjusted_expression(long: pd.DataFrame) -> pd.DataFrame:
    """Residualize expression for study batch only for descriptive plots."""
    adjusted_parts = []
    for gene, data in long.groupby("gene", observed=True):
        data = data.copy()
        fitted = smf.ols("expression ~ C(batch)", data=data).fit()
        data["adjusted_expression"] = fitted.resid + data["expression"].mean()
        adjusted_parts.append(data)
    return pd.concat(adjusted_parts, ignore_index=True)


def plot_age_trajectories(
    long: pd.DataFrame, genes: list[str], output_dir: Path
) -> None:
    long = long[long["size_class"].isin(PLOT_SIZE_ORDER)].copy()
    long["size_class"] = long["size_class"].cat.remove_unused_categories()
    ncols = 2
    nrows = int(np.ceil(len(genes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows), squeeze=False)

    for ax, gene in zip(axes.flat, genes):
        data = long[long["gene"].eq(gene)]
        sns.scatterplot(
            data=data,
            x="Chronological_Age",
            y="expression",
            hue="size_class",
            hue_order=PLOT_SIZE_ORDER,
            palette=PALETTE,
            s=42,
            alpha=0.7,
            ax=ax,
            legend=False,
        )
        for size in PLOT_SIZE_ORDER:
            group = data[data["size_class"].eq(size)]
            if len(group) >= 3 and group["Chronological_Age"].nunique() >= 2:
                sns.regplot(
                    data=group,
                    x="Chronological_Age",
                    y="expression",
                    scatter=False,
                    ci=None,
                    color=PALETTE[size],
                    line_kws={"linewidth": 1.8},
                    ax=ax,
                )
        ax.set_title(gene)
        ax.set_xlabel("Chronological age (years)")
        ax.set_ylabel("Expression (TMM logCPM)")

    for ax in axes.flat[len(genes) :]:
        ax.set_visible(False)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=PALETTE[size],
            label=size,
        )
        for size in PLOT_SIZE_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(PLOT_SIZE_ORDER))
    fig.suptitle(
        "Aging-related gene expression by body-size class (descriptive)",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "gene_expression_age_trajectories_by_size.png", dpi=300)
    fig.savefig(output_dir / "gene_expression_age_trajectories_by_size.pdf")
    plt.close(fig)


def plot_expression_four_sizes(
    data: pd.DataFrame,
    genes: list[str],
    output_dir: Path,
    value_column: str,
    y_label: str,
    title: str,
    filename: str,
    annotations: dict[str, str],
) -> None:
    data = data[data["size_class"].isin(PLOT_SIZE_ORDER)].copy()
    data["size_class"] = data["size_class"].cat.remove_unused_categories()
    ncols = 3
    nrows = int(np.ceil(len(genes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows), squeeze=False)

    for ax, gene in zip(axes.flat, genes):
        gene_data = data[data["gene"].eq(gene)]
        sns.boxplot(
            data=gene_data,
            x="size_class",
            y=value_column,
            order=PLOT_SIZE_ORDER,
            color="white",
            width=0.55,
            ax=ax,
        )
        sns.stripplot(
            data=gene_data,
            x="size_class",
            y=value_column,
            order=PLOT_SIZE_ORDER,
            hue_order=PLOT_SIZE_ORDER,
            palette=PALETTE,
            hue="size_class",
            legend=False,
            jitter=0.15,
            alpha=0.8,
            ax=ax,
        )
        ax.set_title(gene)
        ax.set_xlabel("")
        ax.set_ylabel(y_label)
        annotation = annotations.get(gene)
        if annotation:
            ax.text(
                0.02,
                0.98,
                annotation,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "0.75",
                },
            )

    for ax in axes.flat[len(genes) :]:
        ax.set_visible(False)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.png", dpi=300)
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    long, genes = prepare_long_data(
        resolve_path(args.metadata),
        resolve_path(args.expression),
        args.genes,
    )
    long.to_csv(output_dir / "aging_gene_expression_long.csv", index=False)

    coefficients, omnibus_tests = run_all_size_models(long)
    coefficients.to_csv(output_dir / "all_size_model_coefficients.csv", index=False)
    omnibus_tests.to_csv(output_dir / "all_size_omnibus_tests.csv", index=False)

    unadjusted_tests = run_unadjusted_size_tests(long)
    unadjusted_tests.to_csv(
        output_dir / "unadjusted_four_size_tests.csv", index=False
    )

    small_large = run_small_large_models(long)
    small_large.to_csv(output_dir / "small_vs_large_adjusted_tests.csv", index=False)

    adjusted = add_adjusted_expression(long)
    adjusted.to_csv(output_dir / "adjusted_expression_values.csv", index=False)
    plot_age_trajectories(long, genes, output_dir)

    unadjusted_annotations = {
        row["gene"]: (
            f"Kruskal–Wallis P = {format_p_value(row['p_value'])} "
            f"({significance_label(row['p_value'])})"
        )
        for _, row in unadjusted_tests.iterrows()
    }
    adjusted_annotations = {}
    for gene in genes:
        gene_tests = omnibus_tests[omnibus_tests["gene"].eq(gene)]
        size_row = gene_tests[gene_tests["test"].eq("size_at_mean_age")]
        interaction_row = gene_tests[
            gene_tests["test"].eq("age_by_size_interaction")
        ]
        if not size_row.empty and not interaction_row.empty:
            p_size = float(size_row.iloc[0]["p_value"])
            p_interaction = float(interaction_row.iloc[0]["p_value"])
            adjusted_annotations[gene] = (
                f"Size P = {format_p_value(p_size)} "
                f"({significance_label(p_size)})\n"
                f"Age×size P = {format_p_value(p_interaction)} "
                f"({significance_label(p_interaction)})"
            )

    plot_expression_four_sizes(
        data=long,
        genes=genes,
        output_dir=output_dir,
        value_column="expression",
        y_label="Expression (TMM logCPM)",
        title="Aging-related gene expression across four body-size classes "
        "(unadjusted)",
        filename="unadjusted_expression_four_sizes",
        annotations=unadjusted_annotations,
    )
    plot_expression_four_sizes(
        data=adjusted,
        genes=genes,
        output_dir=output_dir,
        value_column="adjusted_expression",
        y_label="Adjusted expression",
        title="Aging-related gene expression across four body-size classes "
        "(adjusted for study batch)",
        filename="adjusted_expression_four_sizes",
        annotations=adjusted_annotations,
    )

    size_counts = (
        long.drop_duplicates("sample_id")["size_class"]
        .value_counts()
        .reindex(SIZE_ORDER)
        .fillna(0)
        .astype(int)
    )
    size_counts.rename("n").to_csv(output_dir / "size_class_counts.csv")

    print(f"Completed. Results: {output_dir}")
    print(f"Genes analyzed: {', '.join(genes)}")
    print("Small-vs-Large adjusted tests:")
    print(
        small_large[
            ["gene", "comparison", "estimate", "p_value", "n_small", "n_large"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
