"""Plot paired MAE and ΔMAE for baseline vs size-augmented clocks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "paired_delta_mae_tests.csv"
MODEL_ORDER = ["LASSO", "ElasticNet", "RandomForest", "SVR", "XGBoost", "MoE"]
BASELINE_COLOR = "#3B6FB6"
AUGMENTED_COLOR = "#D95F4C"
IMPROVEMENT_COLOR = "#2E8B57"
WORSENING_COLOR = "#C23B33"
FONT_SIZE = 12
TITLE_SIZE = 14
SUPTITLE_SIZE = 16
ANNOTATION_SIZE = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paired MAE and delta-MAE results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="paired_delta_mae_tests.csv 文件路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出目录；默认在 CSV 同级目录创建 paired_delta_mae_figures。",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def validate_and_load(csv_path: Path) -> pd.DataFrame:
    required = {
        "model",
        "n_paired",
        "mae_baseline",
        "mae_size_augmented",
        "wilcoxon_p_value",
        "delta_mae_augmented_minus_baseline",
        "delta_mae_ci_low",
        "delta_mae_ci_high",
    }
    data = pd.read_csv(csv_path)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列：{sorted(missing)}")

    unknown_models = sorted(set(data["model"]) - set(MODEL_ORDER))
    order = MODEL_ORDER + unknown_models
    data["model"] = pd.Categorical(data["model"], categories=order, ordered=True)
    return data.sort_values("model").reset_index(drop=True)


def p_label(p_value: float) -> str:
    if pd.isna(p_value):
        return "p = NA"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def significance_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def draw_mae_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    y = np.arange(len(data))

    for index, row in data.iterrows():
        baseline = row["mae_baseline"]
        augmented = row["mae_size_augmented"]
        line_color = IMPROVEMENT_COLOR if augmented < baseline else WORSENING_COLOR
        ax.plot(
            [baseline, augmented],
            [index, index],
            color=line_color,
            linewidth=2.2,
            alpha=0.75,
            zorder=1,
        )

    ax.scatter(
        data["mae_baseline"],
        y,
        s=72,
        color=BASELINE_COLOR,
        edgecolor="white",
        linewidth=0.7,
        label="Expression only",
        zorder=3,
    )
    ax.scatter(
        data["mae_size_augmented"],
        y,
        s=72,
        color=AUGMENTED_COLOR,
        edgecolor="white",
        linewidth=0.7,
        label="Expression + size",
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(data["model"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("MAE (years; lower is better)")
    ax.set_title("Paired overall MAE")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)


def draw_delta_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    y = np.arange(len(data))
    delta = data["delta_mae_augmented_minus_baseline"].to_numpy(dtype=float)
    ci_low = data["delta_mae_ci_low"].to_numpy(dtype=float)
    ci_high = data["delta_mae_ci_high"].to_numpy(dtype=float)
    colors = [
        IMPROVEMENT_COLOR if value < 0 else WORSENING_COLOR for value in delta
    ]

    ax.hlines(y, ci_low, ci_high, color="#555555", linewidth=2.2, zorder=1)
    ax.scatter(
        delta,
        y,
        s=82,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1.1)

    all_x = np.concatenate([ci_low, ci_high, delta])
    span = float(np.nanmax(all_x) - np.nanmin(all_x))
    annotation_gap = max(span * 0.045, 0.012)
    right_limit = float(np.nanmax(ci_high) + max(span * 0.55, 0.18))
    left_limit = float(np.nanmin(ci_low) - max(span * 0.10, 0.04))

    for index, row in data.iterrows():
        p_value = float(row["wilcoxon_p_value"])
        annotation = (
            f"{row['delta_mae_augmented_minus_baseline']:+.3f}  "
            f"[{row['delta_mae_ci_low']:+.3f}, "
            f"{row['delta_mae_ci_high']:+.3f}]\n"
            f"{p_label(p_value)} {significance_stars(p_value)}"
        )
        ax.text(
            row["delta_mae_ci_high"] + annotation_gap,
            index,
            annotation,
            ha="left",
            va="center",
            fontsize=ANNOTATION_SIZE,
        )

    ax.set_xlim(left_limit, right_limit)
    ax.set_yticks(y)
    ax.set_yticklabels(data["model"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("Delta MAE (expression + size minus expression only, years)")
    ax.set_title("Paired delta-MAE with 95% bootstrap CI")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)


def save_figure(
    fig: plt.Figure, output_dir: Path, filename: str, dpi: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        output_path = output_dir / f"{filename}.{suffix}"
        fig.savefig(
            output_path,
            format=suffix,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"已保存：{output_path}")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else input_path.parent / "paired_delta_mae_figures"
    )

    data = validate_and_load(input_path)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "SimHei"],
            "axes.unicode_minus": False,
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6.2),
        gridspec_kw={"width_ratios": [0.85, 1.35], "wspace": 0.28},
    )
    draw_mae_panel(axes[0], data)
    draw_delta_panel(axes[1], data)
    n_paired = int(data["n_paired"].min())
    fig.suptitle(
        f"Effect of adding body-size class to aging-clock models "
        f"(paired LOSO predictions, n = {n_paired})",
        fontsize=SUPTITLE_SIZE,
        y=1.02,
    )
    save_figure(fig, output_dir, "paired_delta_mae_summary", args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    draw_delta_panel(ax, data)
    fig.suptitle(
        "Effect of body-size covariate on prediction error",
        fontsize=SUPTITLE_SIZE,
        y=1.02,
    )
    save_figure(fig, output_dir, "paired_delta_mae_forest", args.dpi)
    plt.close(fig)

    print("绘图完成。")


if __name__ == "__main__":
    main()
