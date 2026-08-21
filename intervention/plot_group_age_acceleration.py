import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns
import scipy.stats as stats

"""Compute age acceleration from group-LOO predictions (delta and residual vs Baseline calibration)."""

script_dir = os.path.dirname(os.path.abspath(__file__))

grouping_column = "group"
CONTROL_KEYWORDS = ["Control", "Normal"]
AGE_ACCELERATION_MODES = ["delta", "residual"]
CALIBRATION_REFERENCE_GROUP = "Baseline"

MODE_YLABELS = {
    "delta": "Age Acceleration (Predicted − Chronological)",
    "residual": "Age Acceleration (residual vs Baseline calibration)",
}
MODE_SUPTITLES = {
    "delta": "Uncorrected age acceleration (Predicted − Chronological Age)",
    "residual": (
        f"Baseline-calibrated age acceleration "
        f"(Predicted − expected from Chronological Age; calibration = {CALIBRATION_REFERENCE_GROUP})"
    ),
}
COMBINED_FIGURE_BASENAMES = {
    "delta": "all_models_age_acceleration_boxplot_2x3_delta",
    "residual": "all_models_age_acceleration_boxplot_2x3_residual",
}

OTHER_DISEASE_TREATMENTS = {
    "Macronychia",
    "Hyperadrenocorticism",
    "Heartworm_infection",
    "Epilepsy"
}
DISPLAY_LABEL_MAP = {
    "autoimmune_hemolytic_anemia": "IMHA"
}
GROUP_ORDER = [
    "CPP_Se",
    "Lymphoma",
    "MSC",
    "NMN",
    "Rapa",
    "Wolf_Dog",
    "arthritis",
    "autoimmune_hemolytic_anemia"
]
TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 15
TICK_LABEL_FONT_SIZE = 13
SUPTITLE_FONT_SIZE = 14

combined_output_directory = os.path.join(script_dir, "model_age_acceleration_boxplots")
stats_output_directory = os.path.join(script_dir, "age_acceleration_stats")
os.makedirs(combined_output_directory, exist_ok=True)
os.makedirs(stats_output_directory, exist_ok=True)


def sanitize_filename(text):
    """将字符串转换为安全文件名。"""
    return (
        str(text)
        .replace(" ", "_")
        .replace(".", "")
        .replace(":", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )


def identify_control_group(treatments):
    """按关键词优先级自动识别当前 group 中的对照组。"""
    for keyword in CONTROL_KEYWORDS:
        keyword_lower = keyword.lower()
        for treatment in treatments:
            treatment_lower = str(treatment).lower()
            if keyword_lower in treatment_lower:
                return treatment

    if len(treatments) > 0:
        return treatments[0]
    return None


def is_control_like_treatment(value):
    """判断某个 treatment 是否属于对照/normal 类。"""
    value_lower = str(value).strip().lower()
    return any(keyword.lower() in value_lower for keyword in CONTROL_KEYWORDS)


def format_p_value_stars(p_val):
    """将 p 值转换为显著性标记。"""
    if pd.isna(p_val):
        return "NA"
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return "ns"


def bh_fdr(p_values):
    """Benjamini–Hochberg FDR correction across all within-cohort comparisons."""
    p = pd.Series(p_values, dtype=float).fillna(1.0).values
    n = len(p)
    if n == 0:
        return pd.Series(dtype=float)
    order = np.argsort(p)
    ranked = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, p[order[i]] * n / rank)
        ranked[order[i]] = val
        prev = val
    return pd.Series(ranked)


def add_fdr_columns(comparison_df):
    """对全部组内 Welch t-test 的 p 值做全局 BH-FDR 校正。"""
    df = comparison_df.copy()
    df["p_value_FDR"] = bh_fdr(df["p_value"])
    df["significance_FDR"] = df["p_value_FDR"].apply(format_p_value_stars)
    df["significant_FDR_0.05"] = df["p_value_FDR"] < 0.05
    return df


def get_model_display_name(folder_name):
    """根据文件夹名生成模型展示名。"""
    name = folder_name
    if name.startswith("aging_clock_results"):
        name = name[len("aging_clock_results"):]
    if name.endswith("_pipeline"):
        name = name[:-len("_pipeline")]
    return name.strip("_- ")


def get_control_label(group_name):
    """生成某个 group 的 control 标签。"""
    return f"{group_name}__Control"


def get_display_label(label):
    """将内部标签转换为图上显示标签。"""
    label = str(label)
    if label.endswith("__Control"):
        group_name = label[:-len("__Control")]
        group_display = DISPLAY_LABEL_MAP.get(group_name, group_name)
        return f"{group_display} Control"
    return DISPLAY_LABEL_MAP.get(label, label)


def get_group_key_from_label(label):
    """将绘图标签映射回所属 group，用于统一配色。"""
    label = str(label)
    if label.endswith("__Control"):
        return label[:-len("__Control")]
    if label == "other_disease":
        return "Lymphoma"
    if label == "Wolf":
        return "Wolf_Dog"
    return label


def get_calibration_mask(df):
    """返回用于 residual 校准的 Baseline 参考样本掩码。"""
    return df[grouping_column].astype(str).str.strip() == CALIBRATION_REFERENCE_GROUP


def _fit_calibration_r2(chrono, predicted, slope, intercept):
    """计算校准集上 Predicted ~ Chronological 线性拟合的 R²。"""
    if len(chrono) < 2:
        return np.nan
    fitted = intercept + slope * chrono
    ss_res = np.sum((predicted - fitted) ** 2)
    ss_tot = np.sum((predicted - predicted.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def build_age_acceleration(df, mode="delta"):
    """
    构建年龄加速度列。

    参数:
        mode: "delta" 或 "residual"

    返回:
        df: 增加 AgeAcceleration、Expected_Predicted_Age 列的数据框
        fit_info: dict, 记录年龄加速度来源与校准信息
    """
    df = df.copy()

    if mode == "delta":
        df["AgeAcceleration"] = df["Predicted_Age"] - df["Chronological_Age"]
        df["Expected_Predicted_Age"] = df["Chronological_Age"]
        fit_info = {
            "mode": "delta",
            "fit_slope": np.nan,
            "fit_intercept": np.nan,
            "fit_n": 0,
            "fit_r2": np.nan,
            "fit_source": "Predicted_Age_minus_Chronological_Age",
            "calibration_group": np.nan,
        }
        return df, fit_info

    if mode == "residual":
        cal_mask = get_calibration_mask(df)
        cal_df = df.loc[cal_mask, ["Chronological_Age", "Predicted_Age"]].dropna()
        if len(cal_df) < 2:
            raise ValueError(
                f"residual 模式需要至少 2 个 {CALIBRATION_REFERENCE_GROUP} 校准样本，"
                f"当前仅 {len(cal_df)} 个。"
            )

        slope, intercept = np.polyfit(
            cal_df["Chronological_Age"].values,
            cal_df["Predicted_Age"].values,
            1,
        )
        expected = intercept + slope * df["Chronological_Age"]
        df["Expected_Predicted_Age"] = expected
        df["AgeAcceleration"] = df["Predicted_Age"] - expected
        fit_info = {
            "mode": "residual",
            "fit_slope": float(slope),
            "fit_intercept": float(intercept),
            "fit_n": int(len(cal_df)),
            "fit_r2": float(
                _fit_calibration_r2(
                    cal_df["Chronological_Age"].values,
                    cal_df["Predicted_Age"].values,
                    slope,
                    intercept,
                )
            ),
            "fit_source": (
                f"OLS_on_{CALIBRATION_REFERENCE_GROUP}: "
                "Predicted_Age ~ intercept + slope * Chronological_Age"
            ),
            "calibration_group": CALIBRATION_REFERENCE_GROUP,
        }
        return df, fit_info

    raise ValueError(f"未知年龄加速度模式: {mode!r}，请使用 'delta' 或 'residual'。")


def cohen_d(x, y):
    """计算两组样本的 Cohen's d。"""
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()

    if len(x) < 2 or len(y) < 2:
        return np.nan

    x_var = x.var(ddof=1)
    y_var = y.var(ddof=1)
    pooled_denominator = len(x) + len(y) - 2
    if pooled_denominator <= 0:
        return np.nan

    pooled_sd = math.sqrt(((len(x) - 1) * x_var + (len(y) - 1) * y_var) / pooled_denominator)
    if pooled_sd == 0:
        return np.nan

    return (y.mean() - x.mean()) / pooled_sd


def load_and_preprocess_model(model_dir_path, mode="delta"):
    """读取并预处理单个模型的 group_loo 结果。"""
    csv_path = os.path.join(model_dir_path, "group_loo", "summary_all_predictions_cv.csv")
    if not os.path.exists(csv_path):
        print(f"警告: '{model_dir_path}' 下未找到 group_loo/summary_all_predictions_cv.csv，跳过。")
        return None

    try:
        df_model = pd.read_csv(csv_path)
    except Exception as e:
        print(f"警告: 读取 '{csv_path}' 失败: {e}")
        return None

    df_model.columns = df_model.columns.str.strip()

    if "Chronological_Age" not in df_model.columns and "actual_age" in df_model.columns:
        df_model["Chronological_Age"] = df_model["actual_age"]
    if "Predicted_Age" not in df_model.columns and "predicted_age" in df_model.columns:
        df_model["Predicted_Age"] = df_model["predicted_age"]

    required_columns = ["Chronological_Age", "Predicted_Age", "treatment", grouping_column]
    missing_columns = [col for col in required_columns if col not in df_model.columns]
    if missing_columns:
        print(f"警告: '{csv_path}' 缺少必要列 {missing_columns}，跳过。")
        return None

    for col in [grouping_column, "treatment", "condition", "sample_id"]:
        if col in df_model.columns:
            df_model[col] = df_model[col].astype(str).str.strip()

    df_model["Chronological_Age"] = pd.to_numeric(df_model["Chronological_Age"], errors="coerce")
    df_model["Predicted_Age"] = pd.to_numeric(df_model["Predicted_Age"], errors="coerce")
    df_model = df_model.dropna(subset=["Chronological_Age", "Predicted_Age", "treatment", grouping_column]).copy()

    if df_model.empty:
        print(f"警告: '{csv_path}' 预处理后为空，跳过。")
        return None

    df_model, fit_info = build_age_acceleration(df_model, mode=mode)
    df_model = df_model.dropna(subset=["AgeAcceleration"]).copy()

    all_groups = sorted(df_model[grouping_column].dropna().unique().tolist())
    print(f"\n模型 '{os.path.basename(model_dir_path)}' 检测到的 group: {all_groups}")
    print(f"  年龄加速度模式: {fit_info['mode']} | 来源: {fit_info['fit_source']}")
    if fit_info["mode"] == "residual":
        print(
            f"  校准: {fit_info['calibration_group']} n={fit_info['fit_n']}, "
            f"slope={fit_info['fit_slope']:.4f}, intercept={fit_info['fit_intercept']:.4f}, "
            f"R2={fit_info['fit_r2']:.4f}"
        )

    return df_model, all_groups, fit_info


def prepare_model_plot_data(df_model):
    """
    为单个模型构建聚合绘图数据。

    规则：
    1. 去除 Baseline
    2. 每个 group 的 control 单独命名为 <group>__Control
    3. 每个 group 的 control 放在该组 treatment 左侧
    4. 指定零散疾病 -> other_disease
    """
    df_model = df_model.copy()
    df_model["plot_treatment"] = df_model["treatment"].astype(str).str.strip()

    plot_parts = []
    order = []

    all_groups = df_model[grouping_column].astype(str).str.strip().unique().tolist()
    ordered_groups = [g for g in GROUP_ORDER if g in all_groups] + sorted([g for g in all_groups if g not in GROUP_ORDER])

    for group_name in ordered_groups:
        if group_name == "Baseline":
            continue

        group_df = df_model[df_model[grouping_column].astype(str).str.strip() == group_name].copy()
        if group_df.empty:
            continue

        control_mask = group_df["plot_treatment"].apply(is_control_like_treatment)
        control_df = group_df[control_mask].copy()
        treatment_df = group_df[~control_mask].copy()

        control_label = get_control_label(group_name)
        if not control_df.empty:
            control_df["plot_treatment"] = control_label
            plot_parts.append(control_df)
            order.append(control_label)

        if treatment_df.empty:
            continue

        treatment_df.loc[treatment_df["plot_treatment"].isin(OTHER_DISEASE_TREATMENTS), "plot_treatment"] = "other_disease"

        if group_name == "Lymphoma":
            preferred_treatments = ["Lymphoma", "other_disease"]
        elif group_name == "Wolf_Dog":
            preferred_treatments = ["Wolf"]
        else:
            preferred_treatments = [group_name]

        treatment_order = [t for t in preferred_treatments if t in treatment_df["plot_treatment"].unique()]
        treatment_order += sorted([t for t in treatment_df["plot_treatment"].unique() if t not in treatment_order])

        for treatment_name in treatment_order:
            current_treatment_df = treatment_df[treatment_df["plot_treatment"] == treatment_name].copy()
            if current_treatment_df.empty:
                continue
            plot_parts.append(current_treatment_df)
            order.append(treatment_name)

    if plot_parts:
        plot_df = pd.concat(plot_parts, ignore_index=True)
    else:
        plot_df = pd.DataFrame(columns=df_model.columns.tolist() + ["plot_treatment"])

    plot_df = plot_df.dropna(subset=["AgeAcceleration", "plot_treatment"]).copy()
    return plot_df, order


def summarize_model_statistics(df_plot, model_name):
    """计算单个模型聚合箱线图的描述统计与对比检验结果。"""
    summary_rows = []
    comparison_rows = []

    for treatment_name, sub_df in df_plot.groupby("plot_treatment"):
        values = sub_df["AgeAcceleration"].dropna()
        summary_rows.append({
            "model": model_name,
            "group": (
                treatment_name[:-len("__Control")] if str(treatment_name).endswith("__Control")
                else next((g for g in GROUP_ORDER if (treatment_name == g or (g == "Wolf_Dog" and treatment_name == "Wolf"))), "Lymphoma" if treatment_name == "other_disease" else np.nan)
            ),
            "treatment": treatment_name,
            "n": len(values),
            "mean_age_acceleration": values.mean() if len(values) > 0 else np.nan,
            "sd_age_acceleration": values.std(ddof=1) if len(values) > 1 else np.nan,
            "median_age_acceleration": values.median() if len(values) > 0 else np.nan,
            "min_age_acceleration": values.min() if len(values) > 0 else np.nan,
            "max_age_acceleration": values.max() if len(values) > 0 else np.nan
        })

    for group_name in GROUP_ORDER:
        control_label = get_control_label(group_name)
        control_values = df_plot.loc[df_plot["plot_treatment"] == control_label, "AgeAcceleration"].dropna()
        if len(control_values) == 0:
            continue

        if group_name == "Lymphoma":
            group_treatments = [t for t in ["Lymphoma", "other_disease"] if t in df_plot["plot_treatment"].unique()]
        elif group_name == "Wolf_Dog":
            group_treatments = [t for t in ["Wolf"] if t in df_plot["plot_treatment"].unique()]
        else:
            group_treatments = [t for t in [group_name] if t in df_plot["plot_treatment"].unique()]

        for treatment in group_treatments:
            treatment_values = df_plot.loc[df_plot["plot_treatment"] == treatment, "AgeAcceleration"].dropna()

            t_stat = np.nan
            p_val = np.nan
            if len(control_values) >= 2 and len(treatment_values) >= 2:
                t_stat, p_val = stats.ttest_ind(control_values, treatment_values, equal_var=False)

            comparison_rows.append({
                "model": model_name,
                "group": group_name,
                "comparison": f"{control_label}_vs_{treatment}",
                "control_name": control_label,
                "treatment_name": treatment,
                "n_control": len(control_values),
                "n_treatment": len(treatment_values),
                "mean_control": control_values.mean() if len(control_values) > 0 else np.nan,
                "mean_treatment": treatment_values.mean() if len(treatment_values) > 0 else np.nan,
                "mean_diff_treatment_minus_control": (
                    treatment_values.mean() - control_values.mean()
                    if len(control_values) > 0 and len(treatment_values) > 0 else np.nan
                ),
                "t_statistic": t_stat,
                "p_value": p_val,
                "significance": format_p_value_stars(p_val),
                "cohens_d": cohen_d(control_values, treatment_values)
            })

    return summary_rows, comparison_rows


def build_palette(order):
    """构建聚合箱线图配色，同组 control 和 treatment 使用同色。"""
    palette = {}
    ordered_group_keys = []
    for treatment in order:
        group_key = get_group_key_from_label(treatment)
        if group_key not in ordered_group_keys:
            ordered_group_keys.append(group_key)

    base_colors = sns.color_palette("tab10", n_colors=max(3, len(ordered_group_keys)))
    group_color_map = {}

    for idx, group_key in enumerate(ordered_group_keys):
        group_color_map[group_key] = base_colors[idx % len(base_colors)]

    for treatment in order:
        palette[treatment] = group_color_map[get_group_key_from_label(treatment)]

    return palette


def add_significance_annotations(ax, df_plot, order, comparison_rows):
    """将显著性检验结果标注到图上。"""
    significant_rows = []
    for row in comparison_rows:
        p_val = row.get("p_value", np.nan)
        if pd.isna(p_val) or p_val >= 0.05:
            continue

        control_name = row.get("control_name")
        treatment_name = row.get("treatment_name")
        if control_name not in order or treatment_name not in order:
            continue

        significant_rows.append({
            "x1": order.index(control_name),
            "x2": order.index(treatment_name),
            "label": format_p_value_stars(p_val),
            "p_value": p_val
        })

    if not significant_rows:
        return

    y_values = df_plot["AgeAcceleration"].dropna()
    if y_values.empty:
        return

    y_min = y_values.min()
    y_max = y_values.max()
    y_range = y_max - y_min
    if y_range <= 0:
        y_range = max(abs(y_max), 1.0)

    bracket_height = y_range * 0.03
    level_step = y_range * 0.08
    base_y = y_max + y_range * 0.05

    significant_rows = sorted(significant_rows, key=lambda item: (item["x2"] - item["x1"], item["x1"]))
    used_levels = []

    for row in significant_rows:
        x1, x2 = sorted([row["x1"], row["x2"]])
        level = 0
        while True:
            overlap = False
            for prev_x1, prev_x2, prev_level in used_levels:
                if prev_level == level and not (x2 < prev_x1 or x1 > prev_x2):
                    overlap = True
                    break
            if not overlap:
                break
            level += 1

        y = base_y + level * level_step
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + bracket_height, y + bracket_height, y],
            lw=1.2,
            c="black"
        )
        ax.text(
            (x1 + x2) / 2,
            y + bracket_height + y_range * 0.01,
            row["label"],
            ha="center",
            va="bottom",
            fontsize=TICK_LABEL_FONT_SIZE
        )
        used_levels.append((x1, x2, level))

    max_level = max(level for _, _, level in used_levels) if used_levels else 0
    ax.set_ylim(y_min - y_range * 0.05, base_y + (max_level + 1) * level_step + bracket_height + y_range * 0.08)


def plot_model_age_acceleration(ax, df_plot, order, model_display_name, comparison_rows, mode="delta"):
    """绘制单个模型的聚合年龄加速度箱线图。"""
    palette = build_palette(order)

    sns.boxplot(
        data=df_plot,
        x="plot_treatment",
        y="AgeAcceleration",
        hue="plot_treatment",
        order=order,
        palette=palette,
        ax=ax,
        width=0.55,
        fliersize=0,
        linewidth=1,
        dodge=False,
        legend=False
    )

    sns.stripplot(
        data=df_plot,
        x="plot_treatment",
        y="AgeAcceleration",
        hue="plot_treatment",
        order=order,
        palette=palette,
        ax=ax,
        dodge=False,
        size=6,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
        legend=False
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    ax.set_title(model_display_name, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Treatment", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(MODE_YLABELS.get(mode, "Age Acceleration"), fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="x", rotation=25, labelsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.xaxis.set_major_locator(FixedLocator(np.arange(len(order))))
    ax.set_xticklabels(
        [get_display_label(label) for label in order],
        fontsize=TICK_LABEL_FONT_SIZE,
        rotation=25,
        ha="right",
        rotation_mode="anchor"
    )
    ax.grid(False)
    add_significance_annotations(ax, df_plot, order, comparison_rows)
    ax.text(
        0.02,
        0.98,
        "Welch t-test: Control VS Intervention/Disease",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=max(TICK_LABEL_FONT_SIZE - 2, 9),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="lightgray", alpha=0.8)
    )


def output_suffix_for_mode(mode):
    """生成统计表/图文件名后缀。"""
    return f"_{mode}"


def combined_figure_basename(mode):
    """返回该模式对应的 2x3 组合图文件名（不含扩展名）。"""
    return COMBINED_FIGURE_BASENAMES.get(
        mode,
        f"all_models_age_acceleration_boxplot_2x3{output_suffix_for_mode(mode)}",
    )


def save_combined_figure(fig, mode):
    """保存单种年龄加速度定义下的 2x3 组合图（PDF + PNG）。"""
    basename = combined_figure_basename(mode)
    saved_paths = []
    for ext, kwargs in (("pdf", {"format": "pdf"}), ("png", {"format": "png", "dpi": 300})):
        out_path = os.path.join(combined_output_directory, f"{basename}.{ext}")
        try:
            fig.savefig(out_path, bbox_inches="tight", **kwargs)
            print(f"成功保存: {out_path}")
            saved_paths.append(out_path)
        except Exception as e:
            print(f"保存 {ext.upper()} 失败 ({out_path}): {e}")
    return saved_paths


def run_analysis_for_mode(mode):
    """对单一年龄加速度模式运行完整分析与出图流程。"""
    print(f"\n{'=' * 60}")
    print(f"年龄加速度模式: {mode}")
    if mode == "delta":
        print("定义: Predicted_Age - Chronological_Age")
    else:
        print(
            f"定义: Predicted_Age - (intercept + slope * Chronological_Age), "
            f"校准集 = {CALIBRATION_REFERENCE_GROUP}"
        )

    model_infos = []
    for name in sorted(os.listdir(script_dir)):
        model_dir = os.path.join(script_dir, name)
        if os.path.isdir(model_dir) and name.startswith("aging_clock_results"):
            print(f"\n发现模型文件夹: {name}")
            result = load_and_preprocess_model(model_dir, mode=mode)
            if result is None:
                continue

            df_model, groups_model, fit_info = result
            model_infos.append({
                "folder_name": name,
                "display_name": get_model_display_name(name) or name,
                "dir": model_dir,
                "df": df_model,
                "groups": groups_model,
                "fit_info": fit_info,
            })

    if not model_infos:
        print(f"错误: 模式 '{mode}' 没有读取到任何有效模型结果。")
        return

    all_summary_rows = []
    all_comparison_rows = []
    fit_rows = []
    per_sample_rows = []

    for info in model_infos:
        fit_info = info["fit_info"]
        fit_rows.append({
            "model": info["display_name"],
            "mode": fit_info["mode"],
            "fit_source": fit_info["fit_source"],
            "calibration_group": fit_info.get("calibration_group", np.nan),
            "fit_slope": fit_info["fit_slope"],
            "fit_intercept": fit_info["fit_intercept"],
            "fit_n": fit_info["fit_n"],
            "fit_r2": fit_info.get("fit_r2", np.nan),
        })

        sample_export = info["df"][
            [
                col for col in [
                    "sample_id", grouping_column, "treatment",
                    "Chronological_Age", "Predicted_Age",
                    "Expected_Predicted_Age", "AgeAcceleration",
                ]
                if col in info["df"].columns
            ]
        ].copy()
        sample_export.insert(0, "model", info["display_name"])
        sample_export["acceleration_mode"] = mode
        per_sample_rows.append(sample_export)

    max_plots = 6
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8.8, rows * 6.8))
    axes = axes.flatten()

    for idx, info in enumerate(model_infos[:max_plots]):
        print(f"\n========== [{mode}] 模型 '{info['display_name']}' 箱线图 ==========")
        df_plot, order = prepare_model_plot_data(info["df"])
        ax = axes[idx]

        if df_plot.empty or len(order) == 0:
            print(f"警告: 模型 '{info['display_name']}' 没有可绘图数据，跳过。")
            ax.set_title(f"{info['display_name']}\n(无数据)")
            ax.axis("off")
            continue

        summary_rows, comparison_rows = summarize_model_statistics(
            df_plot=df_plot,
            model_name=info["display_name"],
        )
        all_summary_rows.extend(summary_rows)
        all_comparison_rows.extend(comparison_rows)

        plot_model_age_acceleration(
            ax=ax,
            df_plot=df_plot,
            order=order,
            model_display_name=info["display_name"],
            comparison_rows=comparison_rows,
            mode=mode,
        )

    for j in range(len(model_infos), max_plots):
        axes[j].axis("off")

    fig.suptitle(
        MODE_SUPTITLES.get(mode, f"Age acceleration ({mode})"),
        fontsize=SUPTITLE_FONT_SIZE,
        y=1.01,
    )

    plt.tight_layout()
    save_combined_figure(fig, mode)
    plt.close(fig)

    mode_tag = mode

    if all_summary_rows:
        summary_path = os.path.join(
            stats_output_directory,
            f"age_acceleration_summary_{mode_tag}.csv",
        )
        pd.DataFrame(all_summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n已保存描述统计表: {summary_path}")

    if all_comparison_rows:
        comparison_df = add_fdr_columns(pd.DataFrame(all_comparison_rows))
        comparison_path = os.path.join(
            stats_output_directory,
            f"age_acceleration_control_vs_treatment_stats_{mode_tag}.csv",
        )
        comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
        n_fdr_sig = int(comparison_df["significant_FDR_0.05"].sum())
        print(f"已保存统计检验表: {comparison_path}")
        print(
            f"  BH-FDR 校正: {len(comparison_df)} 次组内比较, "
            f"FDR < 0.05 共 {n_fdr_sig} 项"
        )

    fit_path = os.path.join(
        stats_output_directory,
        f"age_acceleration_fit_info_{mode_tag}.csv",
    )
    pd.DataFrame(fit_rows).to_csv(fit_path, index=False, encoding="utf-8-sig")
    print(f"已保存拟合信息表: {fit_path}")

    if per_sample_rows:
        sample_path = os.path.join(
            stats_output_directory,
            f"age_acceleration_per_sample_{mode_tag}.csv",
        )
        pd.concat(per_sample_rows, ignore_index=True).to_csv(
            sample_path, index=False, encoding="utf-8-sig"
        )
        print(f"已保存样本级表: {sample_path}")


def main():
    print(f"脚本目录: {script_dir}")
    print(f"将依次生成两张组合图: delta(校正前) 与 residual(校正后)")
    print(f"输出目录: {combined_output_directory}")

    for mode in AGE_ACCELERATION_MODES:
        run_analysis_for_mode(mode)

    print("\n全部年龄加速度分析完成。")
    print("组合图输出:")
    for mode in AGE_ACCELERATION_MODES:
        basename = combined_figure_basename(mode)
        print(f"  - {mode}: {basename}.pdf / .png")


if __name__ == "__main__":
    main()
