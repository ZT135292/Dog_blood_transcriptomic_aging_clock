"""Post-hoc OLS of chronological age on OOF predicted age plus batch, size, CLR, and sex."""

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
NO_GROUP = SCRIPT_DIR.parent
CELL_TYPE_DIR = NO_GROUP / "cell_type"
if str(CELL_TYPE_DIR) not in sys.path:
    sys.path.insert(0, str(CELL_TYPE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cell_composition_utils import (  # noqa: E402
    CLR_COLS,
    MODEL_RESULT_DIRS,
    discover_prediction_files,
    ensure_clr_columns,
    load_merged_composition,
    load_predictions,
    merge_clock_and_composition,
    ols_table,
    script_paths,
)
from manuscript_sections import write_manuscript_sections  # noqa: E402

MODEL_ORDER = list(MODEL_RESULT_DIRS.keys())
BATCH_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

MPL_STYLE = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def parse_args() -> argparse.Namespace:
    _, parent = script_paths()
    parser = argparse.ArgumentParser(description="Post-hoc covariate + within-study centered analysis.")
    parser.add_argument("--parent-dir", type=Path, default=parent)
    parser.add_argument("--cell-dir", type=Path, default=CELL_TYPE_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--cv-scheme", default="group_loo", choices=["group_loo", "DataSet_loo"])
    return parser.parse_args()


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def simplify_sex(value: object) -> str:
    text = str(value).lower()
    if "female" in text:
        return "Female"
    if "male" in text:
        return "Male"
    return "Unknown"


def subset_normal_validation(df: pd.DataFrame) -> pd.DataFrame:
    if "data_type" in df.columns:
        val = df[df["data_type"] == "validation"].copy()
        if not val.empty:
            return val
    if "condition" in df.columns:
        return df[df["condition"] == "Normal"].copy()
    return df.copy()


def prepare_validation_frame(
    parent_dir: Path,
    cell_dir: Path,
    cv_scheme: str,
) -> tuple[pd.DataFrame, list[str], dict[str, pd.DataFrame]]:
    """返回合并后的长表 + clr_cols + 各模型 validation 子集。"""
    comp = ensure_clr_columns(load_merged_composition(cell_dir))
    clr_cols = [c for c in CLR_COLS if c in comp.columns]
    pred_files = discover_prediction_files(parent_dir, cv_scheme)
    if not pred_files:
        raise FileNotFoundError(f"未找到 {cv_scheme} 预测文件，parent_dir={parent_dir}")

    per_model: dict[str, pd.DataFrame] = {}
    chunks: list[pd.DataFrame] = []

    for model, path in pred_files.items():
        pred = load_predictions(path)
        merged = merge_clock_and_composition(pred, comp)
        val = subset_normal_validation(merged)
        if val.empty:
            continue
        val = val.copy()
        val["model"] = model
        val["sex_simple"] = val["sex"].map(simplify_sex)
        if "size_class" not in val.columns and "size_class_DAP" in val.columns:
            val["size_class"] = val["size_class_DAP"]
        val["size_class"] = val["size_class"].fillna("Unknown").astype(str)
        per_model[model] = val
        chunks.append(val)

    if not chunks:
        raise ValueError("无 Normal validation 样本与预测合并结果")

    long_df = pd.concat(chunks, ignore_index=True)
    return long_df, clr_cols, per_model


def within_study_centered_correlation(
    df: pd.DataFrame, model: str, cv_scheme: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_batch_rows = []
    for batch, g in df.groupby("batch", observed=True):
        if len(g) < 3:
            continue
        pred_c = g["Predicted_Age"] - g["Predicted_Age"].mean()
        age_c = g["actual_age"] - g["actual_age"].mean()
        if pred_c.std() == 0 or age_c.std() == 0:
            continue
        r, p = stats.pearsonr(pred_c, age_c)
        rho, rho_p = stats.spearmanr(pred_c, age_c)
        per_batch_rows.append(
            {
                "model": model,
                "cv_scheme": cv_scheme,
                "batch": batch,
                "n": len(g),
                "pearson_r": float(r),
                "pearson_p": float(p),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
            }
        )

    pred_c = df["Predicted_Age"] - df.groupby("batch")["Predicted_Age"].transform("mean")
    age_c = df["actual_age"] - df.groupby("batch")["actual_age"].transform("mean")
    r_all, p_all = stats.pearsonr(pred_c, age_c)
    rho_all, rho_p_all = stats.spearmanr(pred_c, age_c)

    summary = pd.DataFrame(
        [
            {
                "model": model,
                "cv_scheme": cv_scheme,
                "n": len(df),
                "n_batches": int(df["batch"].nunique()),
                "pearson_r_pooled": float(r_all),
                "pearson_p": float(p_all),
                "spearman_rho_pooled": float(rho_all),
                "spearman_p": float(rho_p_all),
            }
        ]
    )
    return pd.DataFrame(per_batch_rows), summary


def plot_within_study_scatter_grid(
    per_model: dict[str, pd.DataFrame],
    out_stem: Path,
    cv_scheme: str,
) -> None:
    models = [m for m in MODEL_ORDER if m in per_model]
    n = len(models)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        work = per_model[model].copy()
        work["pred_c"] = work["Predicted_Age"] - work.groupby("batch")["Predicted_Age"].transform(
            "mean"
        )
        work["age_c"] = work["actual_age"] - work.groupby("batch")["actual_age"].transform("mean")

        batches = sorted(work["batch"].astype(str).unique())
        for i, batch in enumerate(batches):
            g = work[work["batch"].astype(str) == batch]
            ax.scatter(
                g["age_c"],
                g["pred_c"],
                s=32,
                alpha=0.85,
                color=BATCH_COLORS[i % len(BATCH_COLORS)],
                edgecolors="white",
                linewidths=0.3,
                label=batch,
            )
        lim = max(work["age_c"].abs().max(), work["pred_c"].abs().max()) * 1.12
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.45)
        r, p = stats.pearsonr(work["pred_c"], work["age_c"])
        ax.set_title(f"{model}\nr={r:.3f}, P={p:.1e}")
        ax.set_xlabel("Age (within-batch centered)")
        ax.set_ylabel("Predicted age (centered)")

    for idx in range(len(models), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(8, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"Within-study centered: predicted vs chronological age ({cv_scheme}, n=89 Normal)",
        y=1.06,
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, out_stem)


def plot_within_study_correlation_bars(summary_df: pd.DataFrame, out_stem: Path) -> None:
    sub = summary_df.sort_values("pearson_r_pooled", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(sub))
    ax.barh(y, sub["pearson_r_pooled"], color="#5C7AEA", height=0.6)
    for i, (_, row) in enumerate(sub.iterrows()):
        sig = "***" if row["pearson_p"] < 0.001 else ("**" if row["pearson_p"] < 0.01 else "*")
        ax.text(row["pearson_r_pooled"] + 0.02, i, f"{row['pearson_r_pooled']:.3f} {sig}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["model"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Pearson r (within-batch centered, pooled)")
    ax.set_title("Age–prediction association after removing study-level offsets")
    ax.axvline(0, color="0.3", lw=0.5)
    fig.tight_layout()
    save_figure(fig, out_stem)


FULL_MODEL_LABEL = "Age ~ Predicted_Age + size_class + CLR + sex + batch"
INCREMENTAL_FORMULA_ORDER = [
    "Age ~ Predicted_Age",
    "Age ~ Predicted_Age + batch",
    "Age ~ Predicted_Age + batch + size_class",
    "Age ~ Predicted_Age + batch + size_class + CLR",
    "Age ~ Predicted_Age + batch + size_class + CLR + sex",
]
INCREMENTAL_SHORT_LABELS = ["Pred", "+batch", "+size", "+CLR", "+sex"]


def full_covariate_columns(clr_cols: list[str]) -> list[str]:
    cols = ["Predicted_Age", "size_class"]
    if clr_cols:
        cols.extend(clr_cols)
    cols.extend(["sex_simple", "batch"])
    return cols


def posthoc_model_specs(clr_cols: list[str]) -> list[tuple[str, list[str]]]:
    specs: list[tuple[str, list[str]]] = [
        ("Age ~ Predicted_Age", ["Predicted_Age"]),
        ("Age ~ Predicted_Age + batch", ["Predicted_Age", "batch"]),
        ("Age ~ Predicted_Age + batch + size_class", ["Predicted_Age", "batch", "size_class"]),
        ("Age ~ Predicted_Age + size_class", ["Predicted_Age", "size_class"]),
        ("Age ~ Predicted_Age + sex", ["Predicted_Age", "sex_simple"]),
    ]
    if clr_cols:
        specs.extend(
            [
                ("Age ~ Predicted_Age + CLR", ["Predicted_Age"] + clr_cols),
                (
                    "Age ~ Predicted_Age + batch + size_class + CLR",
                    ["Predicted_Age", "batch", "size_class"] + clr_cols,
                ),
                (
                    "Age ~ Predicted_Age + size_class + CLR",
                    ["Predicted_Age", "size_class"] + clr_cols,
                ),
                (
                    "Age ~ Predicted_Age + size_class + CLR + sex",
                    ["Predicted_Age", "size_class"] + clr_cols + ["sex_simple"],
                ),
                (
                    "Age ~ Predicted_Age + batch + size_class + CLR + sex",
                    ["Predicted_Age", "batch", "size_class"] + clr_cols + ["sex_simple"],
                ),
            ]
        )
    else:
        specs.append(
            (
                "Age ~ Predicted_Age + batch + size_class + sex",
                ["Predicted_Age", "batch", "size_class", "sex_simple"],
            )
        )
    specs.append((FULL_MODEL_LABEL, full_covariate_columns(clr_cols)))
    return specs


def single_covariate_specs(clr_cols: list[str]) -> list[tuple[str, str, list[str]]]:
    """(adjustment_label, formula_name, columns) — 每次只加一个协变量。"""
    specs: list[tuple[str, str, list[str]]] = [
        ("none", "Age ~ Predicted_Age", ["Predicted_Age"]),
        ("size_class", "Age ~ Predicted_Age + size_class", ["Predicted_Age", "size_class"]),
        ("sex", "Age ~ Predicted_Age + sex", ["Predicted_Age", "sex_simple"]),
        ("batch", "Age ~ Predicted_Age + batch", ["Predicted_Age", "batch"]),
    ]
    if clr_cols:
        specs.insert(2, ("CLR", "Age ~ Predicted_Age + CLR", ["Predicted_Age"] + clr_cols))
    return specs


def batch_plus_other_specs(clr_cols: list[str]) -> list[tuple[str, str, list[str]]]:
    """(adjustment_label, formula_name, columns) — batch 固定，轮流加入其它协变量。"""
    specs: list[tuple[str, str, list[str]]] = [
        (
            "batch+size_class",
            "Age ~ Predicted_Age + batch + size_class",
            ["Predicted_Age", "batch", "size_class"],
        ),
        (
            "batch+sex",
            "Age ~ Predicted_Age + batch + sex",
            ["Predicted_Age", "batch", "sex_simple"],
        ),
    ]
    if clr_cols:
        specs.insert(
            1,
            (
                "batch+CLR",
                "Age ~ Predicted_Age + batch + CLR",
                ["Predicted_Age", "batch"] + clr_cols,
            ),
        )
    return specs


def _fit_one_posthoc_spec(
    work: pd.DataFrame,
    model: str,
    cv_scheme: str,
    formula_name: str,
    cols: list[str],
    spec_group: str = "nested",
    adjustment_label: str = "",
) -> dict | None:
    use_cols = [c for c in cols if c in work.columns]
    if "Predicted_Age" not in use_cols:
        return None
    X = pd.get_dummies(work[use_cols], drop_first=True, dtype=float)
    fit = ols_table(work["actual_age"], X, response_name="actual_age")
    if fit.empty:
        return None
    pred_row = fit[fit["term"] == "Predicted_Age"]
    return {
        "model": model,
        "cv_scheme": cv_scheme,
        "spec_group": spec_group,
        "adjustment_label": adjustment_label or formula_name,
        "formula": formula_name,
        "n": fit.attrs.get("n"),
        "r2": fit.attrs.get("r2"),
        "predicted_age_coef": pred_row["estimate"].iloc[0] if len(pred_row) else np.nan,
        "predicted_age_se": pred_row["std_error"].iloc[0] if len(pred_row) else np.nan,
        "predicted_age_p": pred_row["p_value"].iloc[0] if len(pred_row) else np.nan,
    }


def fit_posthoc_sensitivity_models(
    df: pd.DataFrame, model: str, cv_scheme: str, clr_cols: list[str]
) -> pd.DataFrame:
    work = df.dropna(subset=["actual_age", "Predicted_Age", "batch", "sex_simple"]).copy()
    rows: list[dict] = []
    for label, formula_name, cols in single_covariate_specs(clr_cols):
        row = _fit_one_posthoc_spec(
            work, model, cv_scheme, formula_name, cols, "single_covariate", label
        )
        if row:
            rows.append(row)
    for label, formula_name, cols in batch_plus_other_specs(clr_cols):
        row = _fit_one_posthoc_spec(
            work, model, cv_scheme, formula_name, cols, "batch_plus_other", label
        )
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def fit_posthoc_full_coefficients(
    df: pd.DataFrame, model: str, cv_scheme: str, clr_cols: list[str]
) -> pd.DataFrame:
    """导出全后验模型全部回归系数。"""
    work = df.dropna(subset=["actual_age", "Predicted_Age", "batch", "sex_simple"]).copy()
    specs = posthoc_model_specs(clr_cols)
    full_cols = None
    for formula_name, cols in specs:
        if formula_name == FULL_MODEL_LABEL:
            full_cols = cols
            break
    if full_cols is None:
        return pd.DataFrame()

    use_cols = [c for c in full_cols if c in work.columns]
    X = pd.get_dummies(work[use_cols], drop_first=True, dtype=float)
    fit = ols_table(work["actual_age"], X, response_name="actual_age")
    if fit.empty:
        return pd.DataFrame()
    out = fit.copy()
    out.insert(0, "clock_model", model)
    out.insert(1, "cv_scheme", cv_scheme)
    out.insert(2, "formula", FULL_MODEL_LABEL)
    out.insert(3, "n", fit.attrs.get("n"))
    out.insert(4, "r2", fit.attrs.get("r2"))
    return out


def fit_posthoc_models(
    df: pd.DataFrame, model: str, cv_scheme: str, clr_cols: list[str]
) -> pd.DataFrame:
    work = df.dropna(subset=["actual_age", "Predicted_Age", "batch", "sex_simple"]).copy()
    rows = []
    for formula_name, cols in posthoc_model_specs(clr_cols):
        row = _fit_one_posthoc_spec(work, model, cv_scheme, formula_name, cols, "nested")
        if row:
            rows.append(row)
    return pd.DataFrame(rows)


def _sig_stars(p: float) -> str:
    if pd.isna(p):
        return "n.e."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _format_p(p: float) -> str:
    if pd.isna(p):
        return "P=n.e."
    p = float(p)
    if p < 0.001:
        return "P<0.001"
    return f"P={p:.3f}"


def _annotate_bar_p_values(ax, bars, p_values, fontsize: float = 7) -> None:
    for bar, p in zip(bars, p_values):
        height = bar.get_height()
        offset = 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        y = height + offset if height >= 0 else height - offset
        va = "bottom" if height >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            _format_p(p),
            ha="center",
            va=va,
            fontsize=fontsize,
            color="0.25",
        )


def plot_sensitivity_coef_heatmap(
    sensitivity_df: pd.DataFrame,
    spec_group: str,
    out_stem: Path,
    title: str,
) -> None:
    sub = sensitivity_df[sensitivity_df["spec_group"] == spec_group].copy()
    if sub.empty:
        return

    label_order_single = ["none", "size_class", "CLR", "sex", "batch"]
    label_order_batch = ["batch+size_class", "batch+CLR", "batch+sex"]
    label_order = label_order_single if spec_group == "single_covariate" else label_order_batch
    label_order = [x for x in label_order if x in sub["adjustment_label"].unique()]

    models = [m for m in MODEL_ORDER if m in sub["model"].unique()]
    coef_mat = np.full((len(models), len(label_order)), np.nan)
    annot = [["" for _ in label_order] for _ in models]

    for i, model in enumerate(models):
        for j, label in enumerate(label_order):
            row = sub[(sub["model"] == model) & (sub["adjustment_label"] == label)]
            if row.empty:
                continue
            r = row.iloc[0]
            coef_mat[i, j] = r["predicted_age_coef"]
            annot[i][j] = f"{r['predicted_age_coef']:.2f}\n({_sig_stars(r['predicted_age_p'])})"

    fig, ax = plt.subplots(figsize=(1.6 * len(label_order) + 2, 0.55 * len(models) + 2))
    vmax = np.nanmax(np.abs(coef_mat)) if np.any(~np.isnan(coef_mat)) else 1.0
    im = ax.imshow(coef_mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels(label_order, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(label_order)):
            if annot[i][j]:
                ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=8, color="0.15")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="β (Predicted_Age)")
    fig.tight_layout()
    save_figure(fig, out_stem)


def plot_sensitivity_r2_bars(
    sensitivity_df: pd.DataFrame,
    spec_group: str,
    out_stem: Path,
    title: str,
) -> None:
    sub = sensitivity_df[sensitivity_df["spec_group"] == spec_group].copy()
    if sub.empty:
        return

    label_order_single = ["none", "size_class", "CLR", "sex", "batch"]
    label_order_batch = ["batch+size_class", "batch+CLR", "batch+sex"]
    label_order = label_order_single if spec_group == "single_covariate" else label_order_batch
    label_order = [x for x in label_order if x in sub["adjustment_label"].unique()]
    models = [m for m in MODEL_ORDER if m in sub["model"].unique()]

    n_models = len(models)
    n_labels = len(label_order)
    x = np.arange(n_models)
    width = 0.8 / max(n_labels, 1)
    colors = plt.cm.Set2(np.linspace(0, 1, n_labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, label in enumerate(label_order):
        vals = []
        for model in models:
            row = sub[(sub["model"] == model) & (sub["adjustment_label"] == label)]
            vals.append(float(row.iloc[0]["r2"]) if not row.empty else np.nan)
        offset = (j - (n_labels - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=label, color=colors[j])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Model R²")
    ax.set_ylim(0, min(1.0, np.nanmax(sub["r2"]) * 1.12))
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, out_stem)


def plot_posthoc_coef_comparison(posthoc_df: pd.DataFrame, out_stem: Path) -> None:
    base = posthoc_df[posthoc_df["formula"] == "Age ~ Predicted_Age"].set_index("model")
    full = posthoc_df[posthoc_df["formula"] == FULL_MODEL_LABEL].set_index("model")
    models = [m for m in MODEL_ORDER if m in base.index and m in full.index]
    if not models:
        return

    x = np.arange(len(models))
    w = 0.35
    label_fs = 13
    tick_fs = 12
    title_fs = 14
    legend_fs = 12
    annot_fs = 11
    fig, ax = plt.subplots(figsize=(11, 5.5))
    c0 = [base.loc[m, "predicted_age_coef"] for m in models]
    c1 = [full.loc[m, "predicted_age_coef"] for m in models]
    p0 = [base.loc[m, "predicted_age_p"] for m in models]
    p1 = [full.loc[m, "predicted_age_p"] for m in models]
    bars0 = ax.bar(x - w / 2, c0, w, label="Unadjusted", color="#A9CCE3")
    bars1 = ax.bar(x + w / 2, c1, w, label="Full adjustment", color="#5C7AEA")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=tick_fs)
    ax.set_ylabel("Coefficient of Predicted_Age", fontsize=label_fs)
    ax.set_title(
        "Predicted age association with chronological age\nbefore vs after covariate adjustment",
        fontsize=title_fs,
    )
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.legend(frameon=False, fontsize=legend_fs)
    ax.axhline(0, color="0.4", lw=0.8, zorder=0)
    ymax = max(max(c0), max(c1), 0.0)
    top_pad = max(0.3, ymax * 0.14)
    ax.set_ylim(0, ymax + top_pad)
    _annotate_bar_p_values(ax, bars0, p0, fontsize=annot_fs)
    _annotate_bar_p_values(ax, bars1, p1, fontsize=annot_fs)
    fig.tight_layout()
    save_figure(fig, out_stem)


def _incremental_r2_for_model(posthoc_df: pd.DataFrame, model: str) -> tuple[list[str], list[float]]:
    sub = posthoc_df[posthoc_df["model"] == model].set_index("formula")
    labels = [f for f in INCREMENTAL_FORMULA_ORDER if f in sub.index]
    if not labels:
        return [], []
    r2 = [float(sub.loc[f, "r2"]) for f in labels]
    return labels, r2


def plot_posthoc_r2_progression(posthoc_df: pd.DataFrame, out_stem: Path, model: str = "ElasticNet") -> None:
    labels, r2 = _incremental_r2_for_model(posthoc_df, model)
    if not labels:
        return
    short = INCREMENTAL_SHORT_LABELS[: len(labels)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(labels)), r2, "o-", color="#E07A5F", lw=2, markersize=8)
    for i, v in enumerate(r2):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short)
    ax.set_ylabel("Model R²")
    ax.set_ylim(0, min(1.0, max(r2) * 1.15))
    ax.set_title(f"Incremental R² with covariates ({model}, group-LOO)")
    fig.tight_layout()
    save_figure(fig, out_stem)


def plot_posthoc_r2_progression_grid(
    posthoc_df: pd.DataFrame,
    out_stem: Path,
    cv_scheme: str,
) -> None:
    models = [m for m in MODEL_ORDER if m in posthoc_df["model"].unique()]
    if not models:
        return

    series = {model: _incremental_r2_for_model(posthoc_df, model) for model in models}
    all_r2 = [v for _, r2 in series.values() for v in r2]
    if not all_r2:
        return
    ymax = min(1.0, max(all_r2) * 1.15)

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    fig.suptitle(
        f"Incremental R² with covariates (group-LOO: {cv_scheme})",
        fontsize=13,
        y=1.02,
    )

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        labels, r2 = series[model]
        if not labels:
            ax.set_visible(False)
            continue
        short = INCREMENTAL_SHORT_LABELS[: len(labels)]
        ax.plot(range(len(labels)), r2, "o-", color="#E07A5F", lw=2, markersize=7)
        for i, v in enumerate(r2):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(short, fontsize=9)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("R²")
        ax.set_title(model, fontsize=11)

    for idx in range(len(models), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    save_figure(fig, out_stem)


def plot_posthoc_r2_all_models(posthoc_df: pd.DataFrame, out_stem: Path) -> None:
    sub = posthoc_df[posthoc_df["formula"] == FULL_MODEL_LABEL].copy()
    sub = sub.set_index("model")
    models = [m for m in MODEL_ORDER if m in sub.index]
    if not models:
        return
    sub = sub.reindex(models)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(sub))
    ax.bar(x, sub["r2"], color="#2A9D8F", width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(sub.index, rotation=25, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Full post-hoc model R²\nAge ~ Pred + size + CLR + sex + batch")
    for i, v in enumerate(sub["r2"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
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

    long_df, clr_cols, per_model = prepare_validation_frame(
        args.parent_dir, args.cell_dir, args.cv_scheme
    )
    _ = long_df  # merged long table (optional downstream use)

    within_batch_parts = []
    within_summary_parts = []
    for model, val in per_model.items():
        per_b, summ = within_study_centered_correlation(val, model, args.cv_scheme)
        within_batch_parts.append(per_b)
        within_summary_parts.append(summ)

    within_batch_df = pd.concat(within_batch_parts, ignore_index=True)
    within_summary_df = pd.concat(within_summary_parts, ignore_index=True)

    within_batch_df.to_csv(table_dir / "Table_within_study_correlation_by_batch.csv", index=False)
    within_summary_df.to_csv(table_dir / "Table_within_study_correlation_summary.csv", index=False)

    plot_within_study_scatter_grid(per_model, fig_dir / "Figure_within_study_centered_all_models", args.cv_scheme)
    plot_within_study_correlation_bars(within_summary_df, fig_dir / "Figure_within_study_pearson_r_by_model")

    posthoc_parts = []
    coef_parts = []
    sensitivity_parts = []
    for model, val in per_model.items():
        posthoc_parts.append(fit_posthoc_models(val, model, args.cv_scheme, clr_cols))
        coef_parts.append(fit_posthoc_full_coefficients(val, model, args.cv_scheme, clr_cols))
        sensitivity_parts.append(fit_posthoc_sensitivity_models(val, model, args.cv_scheme, clr_cols))
    posthoc_df = pd.concat(posthoc_parts, ignore_index=True)
    coef_df = pd.concat(coef_parts, ignore_index=True)
    sensitivity_df = pd.concat(sensitivity_parts, ignore_index=True)

    posthoc_df.to_csv(table_dir / "Table_posthoc_all_model_specifications.csv", index=False)
    full_only = posthoc_df[posthoc_df["formula"] == FULL_MODEL_LABEL].copy()
    full_only.to_csv(table_dir / "Table_posthoc_full_covariate_model.csv", index=False)
    if not coef_df.empty:
        coef_df.to_csv(table_dir / "Table_posthoc_full_model_all_coefficients.csv", index=False)

    if not sensitivity_df.empty:
        sensitivity_df.to_csv(table_dir / "Table_posthoc_covariate_sensitivity.csv", index=False)
        single_df = sensitivity_df[sensitivity_df["spec_group"] == "single_covariate"].copy()
        batch_plus_df = sensitivity_df[sensitivity_df["spec_group"] == "batch_plus_other"].copy()
        single_df.to_csv(table_dir / "Table_posthoc_single_covariate_adjustment.csv", index=False)
        batch_plus_df.to_csv(table_dir / "Table_posthoc_batch_plus_covariate.csv", index=False)

    plot_posthoc_coef_comparison(posthoc_df, fig_dir / "Figure_posthoc_predicted_age_coef_unadj_vs_full")
    plot_posthoc_r2_progression_grid(
        posthoc_df, fig_dir / "Figure_posthoc_r2_incremental_all_models", args.cv_scheme
    )
    plot_posthoc_r2_all_models(posthoc_df, fig_dir / "Figure_posthoc_full_model_r2_all_models")

    if not sensitivity_df.empty:
        plot_sensitivity_coef_heatmap(
            sensitivity_df,
            "single_covariate",
            fig_dir / "Figure_posthoc_single_covariate_coef_heatmap",
            "Predicted-age coefficient: single-covariate adjustment",
        )
        plot_sensitivity_coef_heatmap(
            sensitivity_df,
            "batch_plus_other",
            fig_dir / "Figure_posthoc_batch_plus_covariate_coef_heatmap",
            "Predicted-age coefficient: batch + one additional covariate",
        )
        plot_sensitivity_r2_bars(
            sensitivity_df,
            "single_covariate",
            fig_dir / "Figure_posthoc_single_covariate_r2_bars",
            "Model R²: single-covariate adjustment (group-LOO validation)",
        )
        plot_sensitivity_r2_bars(
            sensitivity_df,
            "batch_plus_other",
            fig_dir / "Figure_posthoc_batch_plus_covariate_r2_bars",
            "Model R²: batch + one additional covariate (group-LOO validation)",
        )

    manuscript_dir = out_dir / "manuscript"
    write_manuscript_sections(table_dir, manuscript_dir, args.cv_scheme, len(clr_cols))

    lines = [
        "=== Covariate post-hoc & within-study centered analysis ===",
        f"CV scheme: {args.cv_scheme}",
        f"CLR columns: {len(clr_cols)}",
        f"Output: {out_dir}",
        "",
        "--- (2) Within-study centered correlation (pooled Pearson r) ---",
        within_summary_df.to_string(index=False),
        "",
        "--- (1) Full post-hoc model: Predicted_Age coefficient ---",
        full_only[["model", "n", "r2", "predicted_age_coef", "predicted_age_p"]].to_string(index=False)
        if not full_only.empty
        else "(empty)",
        "",
        "--- (1b) Single-covariate adjustment ---",
        sensitivity_df[sensitivity_df["spec_group"] == "single_covariate"][
            ["model", "adjustment_label", "r2", "predicted_age_coef", "predicted_age_p"]
        ].to_string(index=False)
        if not sensitivity_df.empty
        else "(empty)",
        "",
        "--- (1c) Batch + one other covariate ---",
        sensitivity_df[sensitivity_df["spec_group"] == "batch_plus_other"][
            ["model", "adjustment_label", "r2", "predicted_age_coef", "predicted_age_p"]
        ].to_string(index=False)
        if not sensitivity_df.empty
        else "(empty)",
    ]
    (out_dir / "report_covariate_analysis.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"完成。表格: {table_dir}")
    print(f"      图片: {fig_dir}")
    print(f"      报告: {out_dir / 'report_covariate_analysis.txt'}")
    print(f"      文稿: {manuscript_dir}")


if __name__ == "__main__":
    main()
