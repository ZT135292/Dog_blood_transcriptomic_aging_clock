import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


script_dir = os.path.dirname(os.path.abspath(__file__))

grouping_column = 'group'
CONTROL_KEYWORDS = ['Control', 'Normal', 'chinese_village_dog']
combined_output_directory = os.path.join(script_dir, 'combined_group_plots')
os.makedirs(combined_output_directory, exist_ok=True)
stats_output_directory = os.path.join(script_dir, 'group_prediction_stats')
os.makedirs(stats_output_directory, exist_ok=True)

OTHER_DISEASE_TREATMENTS = {
    "Macronychia",
    "Hyperadrenocorticism",
    "Heartworm_infection",
    "Epilepsy",
}

TITLE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 12
TICK_LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 11
ANNOTATION_FONT_SIZE = 10
SUPTITLE_FONT_SIZE = 12


def identify_control_group(treatments):
    """Return the control treatment by CONTROL_KEYWORDS priority, else the first label."""
    for keyword in CONTROL_KEYWORDS:
        keyword_lower = keyword.lower()
        for treatment in treatments:
            treatment_lower = str(treatment).lower()
            if keyword_lower in treatment_lower:
                return treatment
    if len(treatments) > 0:
        return treatments[0]
    return None

def format_p_value_stars(p_val):
    """格式化p值为显著性星号"""
    if p_val is None or (isinstance(p_val, float) and np.isnan(p_val)):
        return ""
    if p_val < 0.001: 
        return "***"
    elif p_val < 0.01: 
        return "**"
    elif p_val < 0.05: 
        return "*"
    return "ns"


def cohen_d(x, y):
    """计算两组样本的 Cohen's d（y - x，合并标准差）。"""
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


def format_p_value_display(p_val, prefix="p"):
    """将 p 值格式化为图注/控制台展示字符串。"""
    if p_val is None or (isinstance(p_val, float) and np.isnan(p_val)):
        return f"{prefix} = NA"
    if p_val < 0.001:
        return f"{prefix} < 0.001"
    return f"{prefix} = {p_val:.3f}"


def build_fdr_lookup(comparison_df):
    """构建 (model, group, treatment) -> FDR 结果 的查找表。"""
    lookup = {}
    for _, row in comparison_df.iterrows():
        key = (row["model"], row["group"], row["treatment_name"])
        lookup[key] = {
            "p_value_FDR": row["p_value_FDR"],
            "significance_FDR": row["significance_FDR"],
        }
    return lookup


def enrich_test_results_with_fdr(test_results, fdr_lookup, model_name, group_name):
    """为组内 t-test 结果附加全局 BH-FDR 校正后的 p 值。"""
    enriched = []
    for result in test_results:
        row = result.copy()
        fdr_info = fdr_lookup.get((model_name, group_name, result["treatment"]), {})
        p_fdr = fdr_info.get("p_value_FDR", np.nan)
        row["p_value_FDR"] = p_fdr
        row["stars_FDR"] = fdr_info.get("significance_FDR", format_p_value_stars(p_fdr))
        row["p_fdr_str"] = format_p_value_display(p_fdr, prefix="FDR")
        enriched.append(row)
    return enriched


def comparison_row_from_result(model_name, group_name, result):
    """将单次 t-test 结果转为汇总表行。"""
    return {
        "model": model_name,
        "group": group_name,
        "comparison": f"Control_vs_{result['treatment']}",
        "control_name": "Control",
        "treatment_name": result["treatment"],
        "n_control": result["n_control"],
        "n_treatment": result["n_treatment"],
        "mean_control": result["mean_control"],
        "mean_treatment": result["mean_treatment"],
        "mean_diff_treatment_minus_control": result["mean_diff_treatment_minus_control"],
        "t_statistic": result["t_statistic"],
        "p_value": result["p_value"],
        "significance": result["stars"],
        "cohens_d": result["cohens_d"],
    }


def collect_all_comparison_rows(model_infos, common_groups, max_plots=6):
    """预先收集全部组内 Welch t-test 结果，用于全局 FDR 校正。"""
    rows = []
    for exp_group_name in common_groups:
        for info in model_infos[:max_plots]:
            df_for_plot, _ = prepare_group_plot_data(info["df"], exp_group_name)
            if df_for_plot is None or df_for_plot.empty:
                continue
            display_name = info["display_name"] or info["folder_name"]
            test_results = compute_t_test_results(df_for_plot, exp_group_name)
            for result in test_results:
                rows.append(comparison_row_from_result(display_name, exp_group_name, result))
    return rows

def compute_t_test_results(data_frame, current_group):
    """
    对当前 group 的 Control 和每个 treatment 进行 Welch t-test，返回结果列表。
    """
    if current_group is None:
        return []

    current_group_data = data_frame[data_frame[grouping_column] == current_group].copy()
    if current_group_data.empty:
        return []

    control_data = current_group_data[current_group_data['treatment'] == 'Control']
    treatment_list = [t for t in current_group_data['treatment'].unique() if t != 'Control']

    if control_data.empty or len(treatment_list) == 0:
        return []

    test_results = []
    control_predicted = control_data['Predicted_Age'].dropna()

    if len(control_predicted) < 2:
        return []

    for treatment in treatment_list:
        treatment_data = current_group_data[current_group_data['treatment'] == treatment]
        treatment_predicted = treatment_data['Predicted_Age'].dropna()

        if len(treatment_predicted) < 2:
            continue

        t_stat, p_val = stats.ttest_ind(control_predicted, treatment_predicted, equal_var=False)

        mean_control = control_predicted.mean()
        mean_treatment = treatment_predicted.mean()

        if p_val < 0.001:
            p_str = "p < 0.001"
        else:
            p_str = f"p = {p_val:.3f}"

        stars = format_p_value_stars(p_val)
        d = cohen_d(control_predicted, treatment_predicted)

        test_results.append({
            'treatment': treatment,
            't_statistic': t_stat,
            'p_value': p_val,
            'p_str': p_str,
            'stars': stars,
            'mean_control': mean_control,
            'mean_treatment': mean_treatment,
            'mean_diff_treatment_minus_control': mean_treatment - mean_control,
            'cohens_d': d,
            'n_control': len(control_predicted),
            'n_treatment': len(treatment_predicted)
        })

    return test_results


def annotate_t_test_results(ax, test_results):
    """在图上添加 t-test 统计结果文本框（含未校正 p 与全局 BH-FDR），并打印到控制台。"""
    if not test_results:
        return

    text_lines = ["Welch t-test (Predicted Age):"]
    for result in test_results:
        line = f"Control vs {result['treatment']}:"
        line += f"\n  {result['p_str']} {result['stars']}"
        line += f" | {result.get('p_fdr_str', 'FDR = NA')} {result.get('stars_FDR', '')}"
        line += f"\n  (n={result['n_control']} vs {result['n_treatment']})"
        text_lines.append(line)

    text_content = "\n".join(text_lines)

    ax.text(
        0.02,
        0.98,
        text_content,
        transform=ax.transAxes,
        fontsize=ANNOTATION_FONT_SIZE,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.9,
            edgecolor="black",
            linewidth=1,
        ),
        family="monospace",
    )

    print(f"\n--- Predicted Age 统计检验结果 (Control vs Treatments) ---")
    for result in test_results:
        print(f"Control vs {result['treatment']}:")
        print(f"  uncorrected p-value: {result['p_value']:.4f} ({result['stars']})")
        p_fdr = result.get("p_value_FDR", np.nan)
        if pd.isna(p_fdr):
            print("  FDR p-value: NA")
        else:
            print(f"  FDR p-value: {p_fdr:.4f} ({result.get('stars_FDR', '')})")
        print(f"  Mean Control: {result['mean_control']:.2f} (n={result['n_control']})")
        print(f"  Mean {result['treatment']}: {result['mean_treatment']:.2f} (n={result['n_treatment']})")
        d = result.get("cohens_d", np.nan)
        if pd.isna(d):
            print("  Cohen's d: NA")
        else:
            print(f"  Cohen's d: {d:.3f}")
        print()


def compute_descriptive_stats(data_frame, current_group, model_name):
    """计算当前 group 各 treatment 的 Predicted Age 描述统计（排除 Baseline）。"""
    if current_group is None:
        return []

    current_group_data = data_frame[
        (data_frame[grouping_column] == current_group) &
        (~data_frame['treatment'].astype(str).str.contains('(Baseline)', regex=False))
    ].copy()

    summary_rows = []
    for treatment_name, sub_df in current_group_data.groupby('treatment'):
        predicted = sub_df['Predicted_Age'].dropna()
        chronological = sub_df['Chronological_Age'].dropna()

        summary_rows.append({
            'model': model_name,
            'group': current_group,
            'treatment': treatment_name,
            'n': len(predicted),
            'mean_predicted_age': predicted.mean() if len(predicted) > 0 else np.nan,
            'sd_predicted_age': predicted.std(ddof=1) if len(predicted) > 1 else np.nan,
            'median_predicted_age': predicted.median() if len(predicted) > 0 else np.nan,
            'mean_chronological_age': chronological.mean() if len(chronological) > 0 else np.nan,
            'sd_chronological_age': chronological.std(ddof=1) if len(chronological) > 1 else np.nan,
        })

    return summary_rows

def assign_treatment_colors(data_frame, current_group):
    """Grey = Baseline, blue = Control, red = other treatments in the current group."""
    palette = {}
    all_treatments = data_frame['treatment'].unique()

    for treatment in all_treatments:
        if '(Baseline)' in str(treatment):
            palette[treatment] = 'gray'

    if current_group:
        current_group_data = data_frame[data_frame[grouping_column] == current_group]
        treatments = current_group_data['treatment'].unique().tolist()

        control_treatment = None
        for treatment in treatments:
            if '(Baseline)' not in str(treatment) and 'control' in str(treatment).lower():
                control_treatment = treatment
                palette[treatment] = 'blue'
                break

        for treatment in treatments:
            if '(Baseline)' not in str(treatment) and treatment != control_treatment and treatment not in palette:
                palette[treatment] = 'red'

    return palette


def plot_age_comparison(
    data_frame,
    english_plot_title,
    output_dir,
    current_group,
    baseline_fit_data=None,
    ax=None,
    save_pdf=True,
    show_plot=True,
    filename_suffix="",
    fdr_lookup=None,
    model_display_name=None,
):
    if data_frame.empty:
        print(f"\n警告: 为 '{english_plot_title}' 绘图时，没有找到任何有效数据。跳过此图。")
        return [], []

    actual_treatments = data_frame['treatment'].unique()
    actual_groups = data_frame[grouping_column].unique()
    print(f"\n--- 正在绘制: {english_plot_title} ---")
    print(f"参与绘图的组: {list(actual_groups)}")
    print(f"参与绘图的treatment: {list(actual_treatments)}")
    print(f"数据点数量: {len(data_frame)}")

    treatment_palette = assign_treatment_colors(data_frame, current_group)
    
    print(f"\n--- 颜色分配信息 ---")
    if current_group:
        current_group_data = data_frame[data_frame[grouping_column] == current_group]
        current_treatments = current_group_data['treatment'].unique()
        for treatment in current_treatments:
            if '(Baseline)' not in str(treatment):
                if treatment in treatment_palette:
                    if treatment_palette[treatment] == 'blue':
                        print(f"✓ 当前group '{current_group}' 的 treatment='{treatment}' → Control (蓝色)")
                    elif treatment_palette[treatment] == 'red':
                        print(f"  - 当前group '{current_group}' 的 treatment='{treatment}' → 红色")
    
    baseline_treatments = [t for t in data_frame['treatment'].unique() if '(Baseline)' in str(t)]
    if baseline_treatments:
        baseline_data_in_plot = data_frame[data_frame['treatment'].isin(baseline_treatments)]
        baseline_count = len(baseline_data_in_plot)
        
        if baseline_fit_data is not None and not baseline_fit_data.empty:
            baseline_source_groups = sorted(baseline_fit_data[grouping_column].unique().tolist())
            print(f"✓ Baseline样本 (灰色): {baseline_count} 个样本")
            print(f"  Baseline来源的group: {baseline_source_groups}")
            print(f"  ⚠️  注意: 当前group '{current_group}' 的Normal样本不在baseline中")
            print(f"  Baseline包含的group数量: {len(baseline_source_groups)} 个")
        else:
            print(f"✓ Baseline样本 (灰色): {baseline_count} 个样本")
    
    print(f"\n完整Treatment颜色配置: {treatment_palette}")
    
    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 8))
        created_figure = True
    else:
        fig = ax.figure
    
    sns.scatterplot(
        data=data_frame,
        x='Chronological_Age',
        y='Predicted_Age',
        hue='treatment',
        style='treatment',
        s=100,
        palette=treatment_palette,
        ax=ax
    )

    ax.plot([0, 14], [0, 14], color='gray', linestyle='--', label='y=x (Reference)')

    if baseline_fit_data is not None and not baseline_fit_data.empty and len(baseline_fit_data) > 1:
        m, b = np.polyfit(baseline_fit_data['Chronological_Age'], baseline_fit_data['Predicted_Age'], 1)
        x_fit = np.array([0, 14])
        y_fit = m * x_fit + b
        ax.plot(x_fit, y_fit, linestyle='-', color='black', linewidth=1.5, label=f'Baseline Group Fit')

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.set_title(english_plot_title, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel('Chronological Age (years)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('Predicted Age (years)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(False)
    ax.legend(
        title='Treatment',
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
        loc='lower right',
        bbox_to_anchor=(1.0, 0),
        frameon=True,
        framealpha=0.6
    )
    
    test_results = compute_t_test_results(data_frame, current_group)
    if fdr_lookup and model_display_name and current_group:
        test_results = enrich_test_results_with_fdr(
            test_results, fdr_lookup, model_display_name, current_group
        )
    annotate_t_test_results(ax, test_results)
    
    if save_pdf:
        safe_title = english_plot_title.replace(' ', '_').replace('.', '').replace(':', '').replace('/', '_').replace('(', '').replace(')', '')
        if filename_suffix:
            safe_title = f"{safe_title}_{filename_suffix}"
        pdf_filename = os.path.join(output_dir, f"{safe_title}.pdf")
        
        try:
            fig.savefig(pdf_filename, format='pdf', bbox_inches='tight')
            print(f"成功保存图表为: {pdf_filename}")
        except Exception as e:
            print(f"保存 PDF 失败: {e}")
    
    if show_plot:
        plt.show()
    elif created_figure:
        plt.close(fig)

    summary_rows = compute_descriptive_stats(data_frame, current_group, english_plot_title)
    return test_results, summary_rows


def load_and_preprocess_model(model_dir_path):
    """读取并预处理单个模型的 summary_all_predictions_cv.csv。"""
    csv_path = os.path.join(model_dir_path, 'group_loo', 'summary_all_predictions_cv.csv')
    if not os.path.exists(csv_path):
        print(f"警告: 模型目录 '{model_dir_path}' 下未找到 'group_loo/summary_all_predictions_cv.csv'，跳过该模型。")
        return None
    
    df_model = pd.read_csv(csv_path)
    
    df_model.columns = df_model.columns.str.strip()

    if grouping_column not in df_model.columns:
        print(f"警告: 文件 '{csv_path}' 中缺少分组列 '{grouping_column}'，跳过该模型。")
        return None
    if 'treatment' not in df_model.columns:
        print(f"警告: 文件 '{csv_path}' 中缺少 'treatment' 列，跳过该模型。")
        return None
    
    for col in ['group', 'treatment', grouping_column]:
        if col in df_model.columns and df_model[col].dtype == 'object':
            df_model[col] = df_model[col].str.strip()
    
    df_model['Chronological_Age'] = pd.to_numeric(df_model['Chronological_Age'], errors='coerce')
    df_model['Predicted_Age'] = pd.to_numeric(df_model['Predicted_Age'], errors='coerce')
    df_model.dropna(subset=['Chronological_Age', 'Predicted_Age', 'treatment', grouping_column], inplace=True)
    
    if df_model.empty:
        print(f"警告: 模型目录 '{model_dir_path}' 中预处理后数据为空，跳过该模型。")
        return None
    
    all_groups_model = sorted(df_model[grouping_column].dropna().unique().tolist())
    print(f"\n模型 '{model_dir_path}' 中检测到的 group: {all_groups_model}")
    
    return df_model, all_groups_model


def get_model_display_name(folder_name):
    """根据文件夹名生成模型在小图中的标题。"""
    name = folder_name
    if name.startswith('aging_clock_results'):
        name = name[len('aging_clock_results'):]
    if name.endswith('_pipeline'):
        name = name[:-len('_pipeline')]
    return name.strip('_- ')


def merge_other_disease_treatments(df, group_name):
    """将 Lymphoma 组内指定零散疾病 treatment 合并为 other_disease。"""
    if group_name != "Lymphoma":
        return df

    df = df.copy()
    merge_mask = (
        (df[grouping_column] == group_name) &
        (df["treatment"].isin(OTHER_DISEASE_TREATMENTS))
    )
    df.loc[merge_mask, "treatment"] = "other_disease"
    return df


def prepare_group_plot_data(df_model, exp_group_name):
    """Build plot table for one group; other-group controls are labeled as Baseline."""
    exp_group_data = df_model[df_model[grouping_column] == exp_group_name]
    if exp_group_data.empty:
        return None, None

    all_groups_model = sorted(df_model[grouping_column].dropna().unique().tolist())
    other_groups = [g for g in all_groups_model if g != exp_group_name]
    baseline_data = None

    if other_groups:
        other_groups_mask = df_model[grouping_column].isin(other_groups)
        normal_mask = pd.Series(False, index=df_model.index)
        for keyword in CONTROL_KEYWORDS:
            keyword_mask = df_model['treatment'].str.contains(keyword, case=False, na=False)
            normal_mask = normal_mask | keyword_mask

        baseline_mask = other_groups_mask & normal_mask
        baseline_data = df_model[baseline_mask].copy()

    baseline_data_for_plot = None
    if baseline_data is not None and not baseline_data.empty:
        baseline_data_for_plot = baseline_data.copy()
        baseline_data_for_plot['treatment'] = baseline_data_for_plot['treatment'].astype(str) + ' (Baseline)'

    exp_group_data_for_plot = exp_group_data.copy()
    treatments_in_group = exp_group_data_for_plot['treatment'].unique().tolist()
    identified_control = identify_control_group(treatments_in_group)
    if identified_control is not None:
        control_mask = exp_group_data_for_plot['treatment'] == identified_control
        exp_group_data_for_plot.loc[control_mask, 'treatment'] = 'Control'

    exp_group_data_for_plot = merge_other_disease_treatments(exp_group_data_for_plot, exp_group_name)

    if baseline_data_for_plot is not None and not baseline_data_for_plot.empty:
        df_for_plot = pd.concat([baseline_data_for_plot, exp_group_data_for_plot])
    else:
        df_for_plot = exp_group_data_for_plot

    return df_for_plot, baseline_data

def main():
    model_infos = []
    for name in sorted(os.listdir(script_dir)):
        model_dir = os.path.join(script_dir, name)
        if os.path.isdir(model_dir) and name.startswith('aging_clock_results'):
            print(f"\n发现模型文件夹: {name}")
            result = load_and_preprocess_model(model_dir)
            if result is None:
                continue
            df_model, groups_model = result
            display_name = get_model_display_name(name)
            model_infos.append({
                'folder_name': name,
                'display_name': display_name,
                'dir': model_dir,
                'df': df_model,
                'groups': groups_model
            })
    
    if not model_infos:
        print("错误: 未在当前路径下找到任何以 'aging_clock_results' 开头且包含有效数据的模型文件夹。")
        return
    
    common_groups = set(model_infos[0]['groups'])
    for info in model_infos[1:]:
        common_groups = common_groups.intersection(info['groups'])
    
    common_groups = sorted(list(common_groups))
    if not common_groups:
        print("错误: 不同模型之间没有共同的 group，无法生成组合图。")
        return

    print(f"\n所有模型共同的 group 列表: {common_groups}")
    print(f"将对每个共同的 group 生成一张 2x3 布局的大图。")

    max_plots = 6  # 2x3
    rows, cols = 2, 3

    all_comparison_rows = collect_all_comparison_rows(model_infos, common_groups, max_plots)
    if all_comparison_rows:
        comparison_df = add_fdr_columns(pd.DataFrame(all_comparison_rows))
        fdr_lookup = build_fdr_lookup(comparison_df)
        n_fdr_sig = int(comparison_df["significant_FDR_0.05"].sum())
        print(
            f"\n全局 BH-FDR 校正: {len(comparison_df)} 次组内比较, "
            f"FDR < 0.05 共 {n_fdr_sig} 项"
        )
    else:
        comparison_df = pd.DataFrame()
        fdr_lookup = {}

    all_summary_rows = []

    plt.rcParams.update(
        {
            "font.size": TICK_LABEL_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_FONT_SIZE,
            "ytick.labelsize": TICK_LABEL_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
        }
    )
    
    for exp_group_name in common_groups:
        print(f"\n========== 正在生成 group = '{exp_group_name}' 的组合图 ==========")
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 5.8))
        axes = axes.flatten()
        
        for idx, info in enumerate(model_infos[:max_plots]):
            ax = axes[idx]
            df_model = info['df']
            display_name = info['display_name'] or info['folder_name']
            
            df_for_plot, baseline_data = prepare_group_plot_data(df_model, exp_group_name)
            if df_for_plot is None or df_for_plot.empty:
                ax.set_title(f"{display_name}\n(无数据)", fontsize=TITLE_FONT_SIZE)
                ax.axis('off')
                continue
            
            subplot_title = display_name
            
            test_results, summary_rows = plot_age_comparison(
                data_frame=df_for_plot,
                english_plot_title=subplot_title,
                output_dir=combined_output_directory,
                current_group=exp_group_name,
                baseline_fit_data=baseline_data,
                ax=ax,
                save_pdf=False,
                show_plot=False,
                fdr_lookup=fdr_lookup,
                model_display_name=display_name,
            )

            all_summary_rows.extend(summary_rows)
        
        for j in range(len(model_infos), max_plots):
            axes[j].axis("off")

        fig.suptitle(
            "BH-FDR correction applied globally across all within-cohort model-by-condition tests",
            fontsize=SUPTITLE_FONT_SIZE,
            y=0.01,
        )
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        
        safe_group = str(exp_group_name).replace(' ', '_').replace('.', '').replace(':', '').replace('/', '_').replace('(', '').replace(')', '')
        pdf_filename = os.path.join(combined_output_directory, f"group_{safe_group}_all_models_2x3.pdf")
        try:
            fig.savefig(pdf_filename, format='pdf', bbox_inches='tight')
            print(f"成功保存 group '{exp_group_name}' 的组合图: {pdf_filename}")
        except Exception as e:
            print(f"保存 group '{exp_group_name}' 组合图失败: {e}")
        plt.close(fig)
    
    if all_summary_rows:
        summary_df = pd.DataFrame(all_summary_rows)
        summary_path = os.path.join(stats_output_directory, 'predicted_age_summary_by_group.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n已保存描述统计表: {summary_path}")

    if not comparison_df.empty:
        comparison_path = os.path.join(stats_output_directory, 'predicted_age_control_vs_treatment_stats.csv')
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        print(f"已保存统计检验表: {comparison_path}")

    print("\n所有组合绘图处理完成。")


if __name__ == "__main__":
    main()