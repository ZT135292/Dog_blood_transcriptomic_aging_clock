"""Age distributions across DAP body-size classes (normal samples)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
SIZE_ORDER = ["Small", "Medium", "Standard", "Large", "Giant"]
AGE_BINS = [0, 3, 6, 9, 12, 20]
AGE_LABELS = ["0–3", "3–6", "6–9", "9–12", "12+"]


def save_figure(fig: plt.Figure, stem_path: Path) -> tuple[Path, Path]:
    """Save figure as PNG (300 dpi) and PDF (vector)."""
    png_path = stem_path.with_suffix(".png")
    pdf_path = stem_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot age distribution by body-size class.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=SCRIPT_DIR / "total_metadata_weight.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "breed_size_confounding_results",
    )
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=SCRIPT_DIR / "Figure_Table",
        help="Optional: copy main figure to revision folder.",
    )
    return parser.parse_args()


def load_normal_with_size(metadata_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path, sep="\t")
    normal = meta[meta["condition"].eq("Normal")].copy()
    size = normal["size_class_DAP"].astype("string").str.strip()
    size = size.mask(size.isin(["", "None", "NA", "NaN", "nan"]))
    normal["size_class"] = pd.Categorical(size, categories=SIZE_ORDER, ordered=True)
    normal["Chronological_Age"] = pd.to_numeric(normal["Chronological_Age"], errors="coerce")
    normal = normal.dropna(subset=["Chronological_Age"])
    return normal


def build_age_bin_table(df: pd.DataFrame) -> pd.DataFrame:
    with_size = df.dropna(subset=["size_class"]).copy()
    with_size["age_bin"] = pd.cut(
        with_size["Chronological_Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    )
    counts = (
        with_size.groupby(["size_class", "age_bin"], observed=True)
        .size()
        .reset_index(name="n")
    )
    pivot = counts.pivot(index="age_bin", columns="size_class", values="n").fillna(0)
    for col in SIZE_ORDER:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[SIZE_ORDER]
    pivot = pivot.reindex(AGE_LABELS)
    return pivot.astype(int)


def plot_stacked_bar(pivot: pd.DataFrame, out_stem: Path, n_total: int) -> tuple[Path, Path]:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index))
    colors = sns.color_palette("Set2", n_colors=len(SIZE_ORDER))

    for i, size in enumerate(SIZE_ORDER):
        values = pivot[size].values
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=f"{size} (n={int(pivot[size].sum())})",
            color=colors[i],
            edgecolor="white",
            linewidth=0.6,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("Number of normal samples")
    ax.set_title(
        f"Age distribution across body-size classes (DAP)\n"
        f"Normal samples with size annotation, n = {n_total}"
    )
    ax.legend(title="Body-size class", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, max(bottom) * 1.12 if len(bottom) else 1)

    fig.tight_layout()
    paths = save_figure(fig, out_stem)
    plt.close(fig)
    return paths


def plot_grouped_bar(pivot: pd.DataFrame, out_stem: Path, n_total: int) -> tuple[Path, Path]:
    sns.set_theme(style="whitegrid", context="talk")
    long = pivot.reset_index().melt(id_vars="age_bin", var_name="size_class", value_name="n")
    long["size_class"] = pd.Categorical(long["size_class"], categories=SIZE_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=long,
        x="age_bin",
        y="n",
        hue="size_class",
        hue_order=SIZE_ORDER,
        palette="Set2",
        edgecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("Number of normal samples")
    ax.set_title(
        f"Age distribution by body-size class (normal, n = {n_total})"
    )
    ax.legend(title="Body-size class", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    paths = save_figure(fig, out_stem)
    plt.close(fig)
    return paths


def plot_violin_box(df: pd.DataFrame, out_stem: Path) -> tuple[Path, Path] | None:
    plot_df = df.dropna(subset=["size_class"]).copy()
    if plot_df.empty:
        return None

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    order = [s for s in SIZE_ORDER if s in plot_df["size_class"].astype(str).unique()]
    sns.violinplot(
        data=plot_df,
        x="size_class",
        y="Chronological_Age",
        order=order,
        hue="size_class",
        palette="Set2",
        legend=False,
        inner=None,
        cut=0,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="size_class",
        y="Chronological_Age",
        order=order,
        color="black",
        alpha=0.55,
        size=4,
        jitter=0.25,
        ax=ax,
    )
    for i, size in enumerate(order):
        n = plot_df["size_class"].eq(size).sum()
        ax.text(i, ax.get_ylim()[1] * 0.98, f"n={n}", ha="center", va="top", fontsize=10)

    ax.set_xlabel("Body-size class (DAP)")
    ax.set_ylabel("Chronological age (years)")
    ax.set_title("Chronological age distribution within each body-size class")
    fig.tight_layout()
    paths = save_figure(fig, out_stem)
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_normal_with_size(args.metadata)
    with_size = df.dropna(subset=["size_class"])
    pivot = build_age_bin_table(df)

    pivot.to_csv(output_dir / "age_by_size_class_counts.csv")
    print("Age bin × size class counts:")
    print(pivot.to_string())
    print(f"\nNormal with size: n={len(with_size)}; without size: n={len(df) - len(with_size)}")

    plot_stacked_bar(
        pivot,
        output_dir / "Figure_SX_age_by_size_stacked_bar",
        len(with_size),
    )
    plot_grouped_bar(
        pivot,
        output_dir / "Figure_SX_age_by_size_grouped_bar",
        len(with_size),
    )
    plot_violin_box(with_size, output_dir / "Figure_SX_age_by_size_violin")

    _, main_pdf = plot_stacked_bar(
        pivot,
        output_dir / "Figure_SX_age_distribution_by_size_class",
        len(with_size),
    )

    if args.copy_to:
        copy_dir = args.copy_to.resolve()
        copy_dir.mkdir(parents=True, exist_ok=True)
        import shutil

        main_png = main_pdf.with_suffix(".png")
        shutil.copy2(main_png, copy_dir / "Supplementary_Figure_SX_age_distribution_by_size_class.png")
        shutil.copy2(main_pdf, copy_dir / "Supplementary_Figure_SX_age_distribution_by_size_class.pdf")
        pivot.to_csv(copy_dir / "Supplementary_Table_age_by_size_class_counts.csv")
        print(f"Copied main figure (PNG + PDF) to {copy_dir}")

    print(f"Figures saved under: {output_dir}")


if __name__ == "__main__":
    main()
