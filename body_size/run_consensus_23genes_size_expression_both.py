"""Run consensus 23-gene body-size expression analysis for both expression matrices."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
COMPARE_SCRIPT = SCRIPT_DIR / "compare_aging_genes_by_size.py"
GENE_CSV = (
    SCRIPT_DIR.parent
    / "all_coefficients"
    / "top_k_consensus_results"
    / "consensus_4of6_top100.csv"
)
METADATA = SCRIPT_DIR / "total_metadata_weight.txt"
EXPR_DIR = SCRIPT_DIR.parent / "all_coefficients"

RUNS = [
    {
        "expression": EXPR_DIR / "total_count_sort_ortholog_TMM_logcpm_normal_ref.txt",
        "expression_tag": "TMM-logCPM",
        "output": SCRIPT_DIR / "consensus_23genes_expression_by_size_results",
    },
    {
        "expression": EXPR_DIR
        / "total_count_sort_ortholog_TMM_logcpm_normal_ref_combat.txt",
        "expression_tag": "TMM-logCPM + ComBat",
        "output": SCRIPT_DIR / "consensus_23genes_expression_by_size_combat_results",
    },
]


def run_one(run: dict) -> None:
    cmd = [
        PYTHON,
        str(COMPARE_SCRIPT),
        "--gene-csv",
        str(GENE_CSV),
        "--metadata",
        str(METADATA),
        "--expression",
        str(run["expression"]),
        "--expression-tag",
        run["expression_tag"],
        "--output",
        str(run["output"]),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_comparison(output_dir: Path) -> None:
    rows = []
    for run in RUNS:
        unadj = pd.read_csv(run["output"] / "unadjusted_four_size_tests.csv")
        omnibus = pd.read_csv(run["output"] / "all_size_omnibus_tests.csv")
        for _, row in unadj.iterrows():
            rows.append(
                {
                    "expression_matrix": run["expression_tag"],
                    "gene": row["gene"],
                    "test": "Kruskal-Wallis_unadjusted_four_sizes",
                    "statistic": row["statistic"],
                    "p_value": row["p_value"],
                }
            )
        for _, row in omnibus.iterrows():
            rows.append(
                {
                    "expression_matrix": run["expression_tag"],
                    "gene": row["gene"],
                    "test": row["test"],
                    "statistic": row["wald_chi2"],
                    "p_value": row["p_value"],
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "comparison_expression_matrix_tests.csv", index=False)

    wide = summary.pivot_table(
        index=["gene", "test"],
        columns="expression_matrix",
        values="p_value",
    ).reset_index()
    wide.to_csv(
        output_dir / "comparison_expression_matrix_pvalues_wide.csv", index=False
    )


def main() -> None:
    for run in RUNS:
        run_one(run)
    build_comparison(SCRIPT_DIR / "consensus_23genes_expression_by_size_comparison")
    print("Done. Comparison tables:")
    print(
        SCRIPT_DIR
        / "consensus_23genes_expression_by_size_comparison"
        / "comparison_expression_matrix_pvalues_wide.csv"
    )


if __name__ == "__main__":
    main()
