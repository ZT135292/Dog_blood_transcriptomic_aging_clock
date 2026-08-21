"""Plot full post-hoc model R² and Predicted_Age coefficients."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER = ["ElasticNet", "LASSO", "SVR", "RandomForest", "XGBoost", "MoE"]
FULL_FORMULA = "Age ~ Predicted_Age + size_class + CLR + sex + batch"
PRED_ONLY_FORMULA = "Age ~ Predicted_Age"

MPL_STYLE = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot full post-hoc covariate model table.")
    parser.add_argument(
        "--table",
        type=Path,
        default=here / "Table_posthoc_full_covariate_model.csv",
    )
    parser.add_argument(
        "--spec-table",
        type=Path,
        default=here / "Table_posthoc_all_model_specifications.csv",
        help="用于 Pred-only vs Full 系数对比（可选）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=here.parent / "figures",
    )
    return parser.parse_args()


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_and_order(table_path: Path) -> pd.DataFrame:
    df = pd.read_csv(table_path)
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    df["_ord"] = df["model"].map(order)
    df = df.sort_values("_ord").drop(columns="_ord")
    return df


def sig_label(p: float) -> str:
    if pd.isna(p):
        return "n.e."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def format_p(p: float) -> str:
    if pd.isna(p):
        return "P=n.e."
    p = float(p)
    if p < 0.001:
        return "P<0.001"
    return f"P={p:.3f}"


def annotate_bar_p_values(ax, bars, p_values) -> None:
    for bar, p in zip(bars, p_values):
        height = bar.get_height()
        offset = 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        y = height + offset if height >= 0 else height - offset
        va = "bottom" if height >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            format_p(p),
            ha="center",
            va=va,
            fontsize=7,
            color="0.25",
        )


def plot_r2_bar(df: pd.DataFrame, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(df))
    bars = ax.bar(x, df["r2"], color="#2A9D8F", width=0.55, edgecolor="white")
    for bar, v in zip(bars, df["r2"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=22, ha="right")
    ax.set_ylim(0, min(1.0, float(df["r2"].max()) * 1.18))
    ax.set_ylabel("Model R²")
    n = int(df["n"].iloc[0]) if "n" in df.columns else 89
    ax.set_title(f"Full post-hoc model R² (group-LOO validation, n={n})")
    ax.set_xlabel("")
    fig.tight_layout()
    save_figure(fig, out_stem)


def plot_coef_forest(df: pd.DataFrame, out_stem: Path) -> None:
    """Predicted_Age 系数 ± SE（无 SE 的模型只画点）。"""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    y = np.arange(len(df))
    coefs = df["predicted_age_coef"].astype(float).values
    ses = df["predicted_age_se"].astype(float).values if "predicted_age_se" in df.columns else np.full(len(df), np.nan)

    for i, (b, se, p) in enumerate(zip(coefs, ses, df["predicted_age_p"])):
        if not np.isnan(se) and se > 0:
            ax.errorbar(
                b, i, xerr=1.96 * se, fmt="o", color="#5C7AEA", capsize=4, markersize=7, elinewidth=1.2
            )
        else:
            ax.plot(b, i, "o", color="#DD8452", markersize=8)
        x_text = b + (0.08 * (max(coefs) - min(coefs) + 0.5))
        ax.text(x_text, i, sig_label(p), va="center", fontsize=9, color="0.35")

    ax.axvline(0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("Coefficient of Predicted_Age (β)")
    ax.set_title("Predicted age in full covariate-adjusted model\n(size + CLR + sex + batch)")
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, out_stem)


def plot_r2_coef_panel(df: pd.DataFrame, out_stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n = int(df["n"].iloc[0]) if "n" in df.columns else 89

    ax = axes[0]
    x = np.arange(len(df))
    ax.bar(x, df["r2"], color="#2A9D8F", width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=22, ha="right")
    ax.set_ylabel("R²")
    ax.set_ylim(0, min(1.0, float(df["r2"].max()) * 1.15))
    ax.set_title("Full model R²")

    ax = axes[1]
    colors = ["#5C7AEA" if not pd.isna(p) and p < 0.05 else "#B0B0B0" for p in df["predicted_age_p"]]
    ax.bar(x, df["predicted_age_coef"], color=colors, width=0.55)
    ax.axhline(0, color="0.4", linewidth=0.6)
    for i, (v, p) in enumerate(zip(df["predicted_age_coef"], df["predicted_age_p"])):
        ax.text(i, v + 0.05, sig_label(p), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=22, ha="right")
    ax.set_ylabel("β (Predicted_Age)")
    ax.set_title("Predicted-age coefficient (blue: P<0.05)")

    fig.suptitle(
        f"Post-hoc OLS: Age ~ Pred + size + CLR + sex + batch (n={n})",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, out_stem)


def plot_coef_unadj_vs_full(spec_path: Path, full_df: pd.DataFrame, out_stem: Path) -> None:
    if not spec_path.is_file():
        return
    spec = pd.read_csv(spec_path)
    base = spec[spec["formula"] == PRED_ONLY_FORMULA].set_index("model")
    full = full_df.set_index("model")
    models = [m for m in MODEL_ORDER if m in base.index and m in full.index]
    if not models:
        return

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    c0 = [float(base.loc[m, "predicted_age_coef"]) for m in models]
    c1 = [float(full.loc[m, "predicted_age_coef"]) for m in models]
    p0 = [base.loc[m, "predicted_age_p"] for m in models]
    p1 = [full.loc[m, "predicted_age_p"] for m in models]
    bars0 = ax.bar(x - w / 2, c0, w, label="Pred only", color="#A9CCE3")
    bars1 = ax.bar(x + w / 2, c1, w, label="Full model", color="#5C7AEA")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=22, ha="right")
    ax.set_ylabel("Coefficient of Predicted_Age")
    ax.set_title("Predicted-age coefficient before vs after full adjustment")
    ax.legend(frameon=False)
    ax.axhline(0, color="0.4", lw=0.6)
    ymax = max(max(c0), max(c1), 0.0)
    ymin = min(min(c0), min(c1), 0.0)
    pad = 0.12 * (ymax - ymin + 1e-6)
    ax.set_ylim(ymin - pad, ymax + pad * 2.2)
    annotate_bar_p_values(ax, bars0, p0)
    annotate_bar_p_values(ax, bars1, p1)
    fig.tight_layout()
    save_figure(fig, out_stem)


def plot_summary_heatmap(df: pd.DataFrame, out_stem: Path) -> None:
    """R² 与 β 热图（便于补充材料一览）。"""
    metrics = df.set_index("model").reindex([m for m in MODEL_ORDER if m in df["model"].values])
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.8))
    for ax, col, title, cmap in zip(
        axes,
        ["r2", "predicted_age_coef"],
        ["R²", "β (Predicted_Age)"],
        ["YlGn", "RdBu_r"],
    ):
        vals = metrics[[col]].T.values
        im = ax.imshow(vals, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics.index, rotation=22, ha="right")
        ax.set_yticks([0])
        ax.set_yticklabels([title])
        for j in range(len(metrics)):
            v = metrics[col].iloc[j]
            ax.text(j, 0, f"{v:.2f}", ha="center", va="center", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    fig.suptitle("Full post-hoc model summary (validation)", fontsize=12, y=1.02)
    fig.tight_layout()
    save_figure(fig, out_stem)


def main() -> None:
    plt.rcParams.update(MPL_STYLE)
    args = parse_args()
    df = load_and_order(args.table)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    plot_r2_bar(df, out / "Figure_posthoc_full_model_r2_bar")
    plot_coef_forest(df, out / "Figure_posthoc_full_model_coef_forest")
    plot_r2_coef_panel(df, out / "Figure_posthoc_full_model_r2_coef_panel")
    plot_summary_heatmap(df, out / "Figure_posthoc_full_model_heatmap")
    plot_coef_unadj_vs_full(args.spec_table, df, out / "Figure_posthoc_predicted_age_coef_unadj_vs_full")

    print(f"读取: {args.table.resolve()}")
    print(f"图片输出: {out.resolve()}")
    print("生成文件:")
    for name in [
        "Figure_posthoc_full_model_r2_bar",
        "Figure_posthoc_full_model_coef_forest",
        "Figure_posthoc_full_model_r2_coef_panel",
        "Figure_posthoc_full_model_heatmap",
        "Figure_posthoc_predicted_age_coef_unadj_vs_full",
    ]:
        print(f"  - {name}.png / .pdf")


if __name__ == "__main__":
    main()
