"""Within-group Welch tests before vs after post-hoc covariate residualization."""

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
REPO_ROOT = SCRIPT_DIR.parent
CELL_COMPOSITION_DIR = REPO_ROOT / "cell_composition"
if str(CELL_COMPOSITION_DIR) not in sys.path:
    sys.path.insert(0, str(CELL_COMPOSITION_DIR))

from cell_composition_utils import (  # noqa: E402
    CLR_COLS,
    MODEL_RESULT_DIRS,
    bh_fdr,
    discover_prediction_files,
    ensure_clr_columns,
    load_merged_composition,
    load_predictions,
    merge_clock_and_composition,
    ols_table,
    subset_normal_validation,
)

FOCUS_GROUPS = [
    "CPP_Se",
    "Lymphoma",
    "MSC",
    "NMN",
    "Rapa",
    "Wolf_Dog",
    "arthritis",
    "autoimmune_hemolytic_anemia",
]
CALIBRATION_REFERENCE_GROUP = "Baseline"
CONTROL_KEYWORDS = ["Control", "Normal", "chinese_village_dog"]
OTHER_DISEASE_TREATMENTS = {
    "Macronychia",
    "Hyperadrenocorticism",
    "Heartworm_infection",
    "Epilepsy",
}
MODEL_ORDER = list(MODEL_RESULT_DIRS.keys())
OUTCOMES = ("Predicted_Age", "age_acceleration_residual")
POSTHOC_FULL_MODEL_LABEL = "Age ~ Predicted_Age + size_class + CLR + sex + batch"
REFERENCE_FORMULA = (
    "Y ~ Chronological_Age + size_class + CLR + sex + batch  "
    "[fit on Normal validation; symmetric to posthoc full model]"
)

MPL_STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Within-group Welch before/after post-hoc covariate adjustment."
    )
    parser.add_argument("--parent-dir", type=Path, default=REPO_ROOT / "model_training")
    parser.add_argument("--cell-dir", type=Path, default=CELL_COMPOSITION_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results_within_group_posthoc",
    )
    parser.add_argument("--cv-scheme", default="group_loo", choices=["group_loo", "DataSet_loo"])
    return parser.parse_args()


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def identify_control_treatment(treatments: list[str]) -> str | None:
    for keyword in CONTROL_KEYWORDS:
        kw = keyword.lower()
        for treatment in treatments:
            if kw in str(treatment).lower():
                return treatment
    return treatments[0] if treatments else None


def merge_other_disease_treatments(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    out = df.copy()
    if group_name != "Lymphoma":
        return out
    mask = (out["group"].astype(str) == group_name) & out["treatment"].isin(OTHER_DISEASE_TREATMENTS)
    out.loc[mask, "treatment"] = "other_disease"
    return out


def add_baseline_residual(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cal = out[out["group"].astype(str) == CALIBRATION_REFERENCE_GROUP]
    cal = cal[["actual_age", "Predicted_Age"]].dropna()
    if len(cal) < 2:
        out["age_acceleration_residual"] = np.nan
        return out
    slope, intercept = np.polyfit(cal["actual_age"].values, cal["Predicted_Age"].values, 1)
    expected = intercept + slope * out["actual_age"]
    out["age_acceleration_residual"] = out["Predicted_Age"] - expected
    return out


def simplify_sex(value: object) -> str:
    text = str(value).lower()
    if "female" in text:
        return "Female"
    if "male" in text:
        return "Male"
    return "Unknown"


def full_covariate_columns(clr_cols: list[str]) -> list[str]:
    """与 run_posthoc_covariate_analysis.py 一致，但用 Chronological_Age 替代 Predicted_Age。"""
    cols = ["actual_age", "size_class"]
    if clr_cols:
        cols.extend(clr_cols)
    cols.extend(["sex_simple", "batch"])
    return cols


def prepare_work_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sex" in out.columns:
        out["sex_simple"] = out["sex"].map(simplify_sex)
    else:
        out["sex_simple"] = "Unknown"
    if "size_class" not in out.columns and "size_class_DAP" in out.columns:
        out["size_class"] = out["size_class_DAP"]
    out["size_class"] = out.get("size_class", pd.Series(dtype=str)).fillna("Unknown").astype(str)
    if "batch" in out.columns:
        out["batch"] = out["batch"].fillna("Unknown").astype(str)
    return out


def analysis_workframe(df: pd.DataFrame) -> pd.DataFrame:
    """与 posthoc 相同：拟合参考模型时要求 batch / sex 非缺失。"""
    return df.dropna(subset=["actual_age", "batch", "sex_simple"]).copy()


def build_covariate_design(df: pd.DataFrame, clr_cols: list[str]) -> pd.DataFrame:
    """与 posthoc 相同 dummy 编码方式（get_dummies on all spec columns）。"""
    use_cols = [c for c in full_covariate_columns(clr_cols) if c in df.columns]
    work = df[use_cols].copy()
    work = work.rename(columns={"actual_age": "Chronological_Age"})
    return pd.get_dummies(work, drop_first=True, dtype=float)


def align_design_columns(X: pd.DataFrame, reference_columns: list[str]) -> pd.DataFrame:
    return X.reindex(columns=reference_columns, fill_value=0.0)


def predict_from_fit(fit: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    beta = fit.set_index("term")["estimate"]
    intercept = float(beta.get("Intercept", 0.0))
    coefs = beta.reindex(X.columns).fillna(0.0)
    return intercept + X.values @ coefs.values


def fit_reference_and_adjust(
    df: pd.DataFrame,
    normal_mask: pd.Series,
    y_col: str,
    clr_cols: list[str],
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """
    在 Normal validation 上拟合 y ~ 全协变量（与 posthoc 对称），对全部样本外推残差（+ Normal 均值）。
    返回: 校正后序列, 系数表, 设计矩阵列名
    """
    work = analysis_workframe(df.dropna(subset=[y_col]))
    if work.empty:
        return pd.Series(index=df.index, dtype=float), pd.DataFrame(), []

    X_all = build_covariate_design(work, clr_cols)
    train = work.loc[normal_mask.reindex(work.index, fill_value=False)]
    train = train.dropna(subset=[y_col])
    if len(train) < X_all.shape[1] + 3:
        fallback = df[y_col].copy() if y_col in df.columns else pd.Series(index=df.index, dtype=float)
        return fallback, pd.DataFrame(), list(X_all.columns)

    X_train = align_design_columns(build_covariate_design(train, clr_cols), list(X_all.columns))
    fit = ols_table(train[y_col], X_train, response_name=y_col)
    if fit.empty:
        fallback = df[y_col].copy() if y_col in df.columns else pd.Series(index=df.index, dtype=float)
        return fallback, pd.DataFrame(), list(X_all.columns)

    X_pred = align_design_columns(X_all, list(X_train.columns))
    fitted = np.asarray(predict_from_fit(fit, X_pred), dtype=float)
    y_mean = float(train[y_col].mean())
    adjusted = work[y_col].copy()
    adjusted.loc[work.index] = work[y_col].values - fitted + y_mean

    out = adjusted.reindex(df.index)
    out.loc[out.index.difference(work.index)] = df.loc[out.index.difference(work.index), y_col]
    fit["outcome"] = y_col
    fit["posthoc_symmetric_formula"] = REFERENCE_FORMULA
    fit["posthoc_full_model"] = POSTHOC_FULL_MODEL_LABEL
    return out, fit, list(X_train.columns)


def prepare_group_frame(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    use = df[df["group"].astype(str) == group_name].copy()
    use = merge_other_disease_treatments(use, group_name)
    treatments = use["treatment"].dropna().astype(str).unique().tolist()
    control = identify_control_treatment(treatments)
    if control is None:
        return pd.DataFrame()
    use["is_control"] = use["treatment"].astype(str) == control
    return use


def list_within_comparisons(df_group: pd.DataFrame) -> list[dict]:
    control = df_group.loc[df_group["is_control"], "treatment"].iloc[0]
    rows = []
    for treatment in sorted(df_group.loc[~df_group["is_control"], "treatment"].astype(str).unique()):
        mask = df_group["is_control"] | (df_group["treatment"].astype(str) == treatment)
        sub = df_group.loc[mask].copy()
        n_ctrl = int(sub["is_control"].sum())
        n_trt = int((~sub["is_control"]).sum())
        if n_ctrl < 2 or n_trt < 2:
            continue
        rows.append(
            {
                "comparison": f"Control_vs_{treatment}",
                "control_label": control,
                "treatment_label": treatment,
                "subset": sub,
                "n_control": n_ctrl,
                "n_treatment": n_trt,
            }
        )
    return rows


def welch_comparison(sub: pd.DataFrame, value_col: str) -> dict:
    ctrl = sub.loc[sub["is_control"], value_col].astype(float).dropna()
    trt = sub.loc[~sub["is_control"], value_col].astype(float).dropna()
    if len(ctrl) < 2 or len(trt) < 2:
        return {
            "mean_control": np.nan,
            "mean_treatment": np.nan,
            "mean_diff_treatment_minus_control": np.nan,
            "welch_t": np.nan,
            "welch_p": np.nan,
        }
    t_stat, p_val = stats.ttest_ind(trt, ctrl, equal_var=False)
    return {
        "mean_control": float(ctrl.mean()),
        "mean_treatment": float(trt.mean()),
        "mean_diff_treatment_minus_control": float(trt.mean() - ctrl.mean()),
        "welch_t": float(t_stat),
        "welch_p": float(p_val),
    }


def direction_label(diff: float) -> str:
    if pd.isna(diff):
        return "NA"
    if diff < 0:
        return "lower_in_treatment"
    if diff > 0:
        return "higher_in_treatment"
    return "equal"


def compare_before_after(
    sub: pd.DataFrame,
    outcome: str,
    unadj_col: str,
    adj_col: str,
) -> dict:
    before = welch_comparison(sub, unadj_col)
    after = welch_comparison(sub, adj_col)
    dir_before = direction_label(before["mean_diff_treatment_minus_control"])
    dir_after = direction_label(after["mean_diff_treatment_minus_control"])
    return {
        **{f"unadj_{k}": v for k, v in before.items()},
        **{f"adj_{k}": v for k, v in after.items()},
        "direction_before": dir_before,
        "direction_after": dir_after,
        "direction_changed": dir_before != dir_after and dir_before != "NA" and dir_after != "NA",
        "mean_diff_change": (
            after["mean_diff_treatment_minus_control"] - before["mean_diff_treatment_minus_control"]
            if not pd.isna(after["mean_diff_treatment_minus_control"])
            and not pd.isna(before["mean_diff_treatment_minus_control"])
            else np.nan
        ),
        "p_value_change": (
            after["welch_p"] - before["welch_p"]
            if not pd.isna(after["welch_p"]) and not pd.isna(before["welch_p"])
            else np.nan
        ),
        "outcome": outcome,
    }


def plot_mean_diff_before_after(results: pd.DataFrame, outcome: str, out_stem: Path) -> None:
    sub = results[results["outcome"] == outcome].copy()
    if sub.empty:
        return
    sub = sub[sub["group"].isin(FOCUS_GROUPS)].sort_values(["group", "clock_model"])
    if sub.empty:
        return

    groups = sorted(sub["group"].unique())
    models = [m for m in MODEL_ORDER if m in sub["clock_model"].unique()]
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(2.8 * n_models, 4.5), sharey=True, squeeze=False)
    for j, model in enumerate(models):
        ax = axes[0][j]
        mdf = sub[sub["clock_model"] == model]
        msub = (
            mdf.groupby("group", as_index=True)[
                ["unadj_mean_diff_treatment_minus_control", "adj_mean_diff_treatment_minus_control"]
            ]
            .mean()
            .reindex(groups)
        )
        x = np.arange(len(groups))
        w = 0.35
        ax.bar(x - w / 2, msub["unadj_mean_diff_treatment_minus_control"], w, label="Unadjusted", color="#5C7AEA")
        ax.bar(x + w / 2, msub["adj_mean_diff_treatment_minus_control"], w, label="Post-hoc adj.", color="#E07A5F")
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_title(model, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=7)
        if j == 0:
            ax.set_ylabel("Treatment − Control (years)")
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Within-group mean difference: before vs after post-hoc adjustment\n{outcome}", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_stem)


def write_report(out_path: Path, results: pd.DataFrame, ref_models: pd.DataFrame) -> None:
    lines = [
        "=== Within-group Welch: unadjusted vs post-hoc covariate-adjusted ===",
        f"Posthoc full model (Normal validation): {POSTHOC_FULL_MODEL_LABEL}",
        f"Symmetric extrapolation: {REFERENCE_FORMULA}",
        "Comparison: control vs treatment within each experimental/disease group.",
        "",
        "--- Reference model R² (Normal validation) ---",
    ]
    if not ref_models.empty:
        r2 = ref_models.groupby(["clock_model", "outcome"])["r2"].first().reset_index()
        lines.append(r2.to_string(index=False))
    else:
        lines.append("(none)")
    lines.extend(["", "--- Direction changed after adjustment ---"])
    changed = results[results["direction_changed"]].sort_values(["group", "outcome", "clock_model"])
    lines.append(changed.to_string(index=False) if not changed.empty else "(none)")
    lines.extend(["", "--- NMN / Rapa ---"])
    focus = results[results["group"].isin(["NMN", "Rapa"])].sort_values(
        ["group", "outcome", "clock_model"]
    )
    cols = [
        "clock_model",
        "group",
        "outcome",
        "unadj_mean_diff_treatment_minus_control",
        "unadj_welch_p",
        "adj_mean_diff_treatment_minus_control",
        "adj_welch_p",
        "direction_changed",
        "unadj_welch_p_FDR",
        "adj_welch_p_FDR",
    ]
    cols = [c for c in cols if c in focus.columns]
    lines.append(focus[cols].to_string(index=False) if not focus.empty else "(none)")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    mpl.rcParams.update(MPL_STYLE)
    table_dir = args.output_dir / "tables"
    fig_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    comp = ensure_clr_columns(load_merged_composition(args.cell_dir))
    clr_cols = [c for c in CLR_COLS if c in comp.columns]
    pred_files = discover_prediction_files(args.parent_dir, args.cv_scheme)
    if not pred_files:
        raise FileNotFoundError(f"No predictions for {args.cv_scheme}")

    comparison_rows: list[dict] = []
    ref_model_rows: list[pd.DataFrame] = []
    sample_rows: list[pd.DataFrame] = []

    for clock_model, path in pred_files.items():
        pred = load_predictions(path)
        merged = merge_clock_and_composition(pred, comp)
        work = prepare_work_frame(merged)
        work = add_baseline_residual(work)

        normal_val = subset_normal_validation(work)
        normal_mask = work.index.isin(normal_val.index)

        adj_predicted, fit_pred, _ = fit_reference_and_adjust(
            work, pd.Series(normal_mask, index=work.index), "Predicted_Age", clr_cols
        )
        adj_accel, fit_accel, _ = fit_reference_and_adjust(
            work,
            pd.Series(normal_mask, index=work.index),
            "age_acceleration_residual",
            clr_cols,
        )
        work["Predicted_Age_posthoc_adj"] = adj_predicted
        work["age_acceleration_residual_posthoc_adj"] = adj_accel

        for fit_df, outcome in [(fit_pred, "Predicted_Age"), (fit_accel, "age_acceleration_residual")]:
            if fit_df is not None and not fit_df.empty:
                f = fit_df.copy()
                f["r2"] = fit_df.attrs.get("r2", np.nan)
                f["n_fit"] = fit_df.attrs.get("n", np.nan)
                f.insert(0, "clock_model", clock_model)
                f.insert(1, "cv_scheme", args.cv_scheme)
                ref_model_rows.append(f)

        per_sample = work[
            [
                "sample_id",
                "group",
                "treatment",
                "actual_age",
                "Predicted_Age",
                "Predicted_Age_posthoc_adj",
                "age_acceleration_residual",
                "age_acceleration_residual_posthoc_adj",
            ]
        ].copy()
        per_sample["clock_model"] = clock_model
        per_sample["cv_scheme"] = args.cv_scheme
        sample_rows.append(per_sample)

        for group_name in FOCUS_GROUPS:
            df_group = prepare_group_frame(work, group_name)
            if df_group.empty:
                continue
            for comp_info in list_within_comparisons(df_group):
                sub = comp_info["subset"]
                for outcome, unadj_col, adj_col in [
                    ("Predicted_Age", "Predicted_Age", "Predicted_Age_posthoc_adj"),
                    (
                        "age_acceleration_residual",
                        "age_acceleration_residual",
                        "age_acceleration_residual_posthoc_adj",
                    ),
                ]:
                    cmp = compare_before_after(sub, outcome, unadj_col, adj_col)
                    comparison_rows.append(
                        {
                            "clock_model": clock_model,
                            "cv_scheme": args.cv_scheme,
                            "group": group_name,
                            "comparison": comp_info["comparison"],
                            "control_label": comp_info["control_label"],
                            "treatment_label": comp_info["treatment_label"],
                            "n_control": comp_info["n_control"],
                            "n_treatment": comp_info["n_treatment"],
                            "posthoc_full_model": POSTHOC_FULL_MODEL_LABEL,
                            "reference_formula": REFERENCE_FORMULA,
                            **cmp,
                        }
                    )

    results = pd.DataFrame(comparison_rows)
    if results.empty:
        print("No within-group comparisons produced.")
        return

    for outcome in OUTCOMES:
        mask = results["outcome"] == outcome
        results.loc[mask, "unadj_welch_p_FDR"] = bh_fdr(results.loc[mask, "unadj_welch_p"])
        results.loc[mask, "adj_welch_p_FDR"] = bh_fdr(results.loc[mask, "adj_welch_p"])

    results.to_csv(table_dir / "Table_within_group_posthoc_before_after.csv", index=False)

    ref_models = pd.concat(ref_model_rows, ignore_index=True) if ref_model_rows else pd.DataFrame()
    if not ref_models.empty:
        ref_models.to_csv(table_dir / "Table_posthoc_reference_model_coefficients.csv", index=False)

    samples = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()
    samples.to_csv(table_dir / "Table_per_sample_posthoc_adjusted_values.csv", index=False)

    summary = (
        results.groupby(["group", "outcome"], as_index=False)
        .agg(
            n_comparisons=("comparison", "count"),
            n_direction_changed=("direction_changed", "sum"),
            n_unadj_sig=("unadj_welch_p", lambda s: int((s < 0.05).sum())),
            n_adj_sig=("adj_welch_p", lambda s: int((s < 0.05).sum())),
        )
    )
    summary.to_csv(table_dir / "Table_summary_direction_and_significance.csv", index=False)

    for outcome in OUTCOMES:
        plot_mean_diff_before_after(
            results,
            outcome,
            fig_dir / f"Figure_mean_diff_before_after_{outcome}",
        )

    write_report(args.output_dir / "report_within_group_posthoc_comparison.txt", results, ref_models)

    print(f"Done. Tables: {table_dir}")
    print(f"      Figures: {fig_dir}")
    print(f"      Report: {args.output_dir / 'report_within_group_posthoc_comparison.txt'}")
    n_changed = int(results["direction_changed"].sum())
    print(f"      Direction changed: {n_changed}/{len(results)} comparisons")


if __name__ == "__main__":
    main()
