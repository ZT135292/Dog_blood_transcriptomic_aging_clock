import os
import json
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score

# ===================================================================
#                       1. 配置中心 (CONFIGURATION)
# ===================================================================
CONFIG = {
    # --- 文件与路径 ---
    "metadata_file": "total_metadata.txt",
    "expression_file": "total_count_sort_ortholog_TMM_batched.txt",

    # 总输出根目录（下面会在里面再建 HPO / 5fold / DataSet_loo / group_loo / group_best_model 五个子目录）
    "output_root_dir": "aging_clock_results_lasso_pipeline",

    # --- 数据列名 ---
    "sample_id_col": "sample_id",
    "group_col": "condition",         # Normal / Intervention
    "subgroup_col": "group",          # 细分 group（来自 total_metadata.txt 的 group 列），用于作图和最后 group 统计，以及 Group-LOO 交叉验证
    "age_col": "Chronological_Age",
    "batch_col": "batch",             # DataSet-LOO 用到的批次列

    # --- 标签 ---
    "normal_group_label": "Normal",

    # --- 异常值检测与去除 ---
    "outlier_detection": {
        "enabled": True,
        "method": "iqr",  # "iqr" 或 "zscore"
        "iqr_factor": 1.5,  # IQR方法的倍数因子
        "zscore_threshold": 3.0,  # Z-score方法的阈值
        "check_age": False,  # 是否检查年龄异常值
        "age_iqr_factor": 3.0,  # 年龄异常值检测的IQR因子（更宽松）
    },

    # --- LASSO 内部 CV (LassoCV) 参数 ---
    "cv_folds_internal": 5,
    "max_iter": 10000,
    "random_state": 66,

    # --- 外层评估参数 ---
    "k_folds": 5,                     # KFold 折数
}


# ===================================================================
#                       2. 公共工具函数
# ===================================================================

def resolve_path(path: str) -> str:
    """把相对路径解析为脚本所在目录下的绝对路径。"""
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def detect_and_remove_outliers(data, gene_columns, age_col, config):
    """
    检测并去除异常值样本。
    
    参数:
        data: DataFrame，包含基因表达数据和元数据
        gene_columns: 基因列名列表
        age_col: 年龄列名
        config: 配置字典，包含异常值检测参数
    
    返回:
        cleaned_data: 去除异常值后的数据
        outlier_indices: 被标记为异常值的样本索引
    """
    outlier_cfg = config.get("outlier_detection", {})
    if not outlier_cfg.get("enabled", False):
        print("  - 异常值检测已禁用")
        return data, []
    
    print("  - 开始异常值检测...")
    original_count = len(data)
    outlier_indices = set()
    
    method = outlier_cfg.get("method", "iqr")
    
    # 方法1: 基于表达数据的异常值检测
    if method == "iqr":
        # IQR方法：对每个样本计算所有基因表达值的统计量
        iqr_factor = outlier_cfg.get("iqr_factor", 1.5)
        print(f"    - 使用IQR方法 (因子={iqr_factor})")
        
        # 计算每个样本的表达值总和（或均值）作为样本特征
        expr_data = data[gene_columns]
        sample_stats = expr_data.sum(axis=1)  # 或使用 mean()
        
        Q1 = sample_stats.quantile(0.25)
        Q3 = sample_stats.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        
        expr_outliers = sample_stats[(sample_stats < lower_bound) | (sample_stats > upper_bound)].index
        outlier_indices.update(expr_outliers)
        print(f"    - 表达数据异常值: {len(expr_outliers)} 个样本")
        
    elif method == "zscore":
        # Z-score方法
        zscore_threshold = outlier_cfg.get("zscore_threshold", 3.0)
        print(f"    - 使用Z-score方法 (阈值={zscore_threshold})")
        
        expr_data = data[gene_columns]
        sample_stats = expr_data.sum(axis=1)
        z_scores = np.abs((sample_stats - sample_stats.mean()) / sample_stats.std())
        expr_outliers = sample_stats[z_scores > zscore_threshold].index
        outlier_indices.update(expr_outliers)
        print(f"    - 表达数据异常值: {len(expr_outliers)} 个样本")
    
    # 方法2: 基于年龄的异常值检测
    if outlier_cfg.get("check_age", True) and age_col in data.columns:
        age_iqr_factor = outlier_cfg.get("age_iqr_factor", 3.0)
        print(f"    - 检查年龄异常值 (IQR因子={age_iqr_factor})")
        
        ages = data[age_col]
        Q1_age = ages.quantile(0.25)
        Q3_age = ages.quantile(0.75)
        IQR_age = Q3_age - Q1_age
        lower_bound_age = Q1_age - age_iqr_factor * IQR_age
        upper_bound_age = Q3_age + age_iqr_factor * IQR_age
        
        age_outliers = ages[(ages < lower_bound_age) | (ages > upper_bound_age)].index
        outlier_indices.update(age_outliers)
        print(f"    - 年龄异常值: {len(age_outliers)} 个样本")
    
    outlier_indices = list(outlier_indices)
    cleaned_data = data.drop(index=outlier_indices)
    
    print(f"  - 异常值检测完成: 原始样本数={original_count}, 异常值={len(outlier_indices)}, 保留样本数={len(cleaned_data)}")
    if outlier_indices:
        print(f"  - 被移除的样本ID: {outlier_indices[:10]}{'...' if len(outlier_indices) > 10 else ''}")
    
    return cleaned_data, outlier_indices


def load_full_data(config):
    """
    读取元数据和表达矩阵，并对齐：
      - 返回 normal_data, intervention_data, all_data, gene_columns, metadata
    """
    print("步骤 0: 加载和预处理数据...")

    metadata_path = resolve_path(config["metadata_file"])
    expr_path = resolve_path(config["expression_file"])
    print(f"  - 读取元数据: {metadata_path}")
    print(f"  - 读取表达矩阵: {expr_path}")

    metadata = pd.read_csv(metadata_path, sep="\t")
    expr_matrix = pd.read_csv(expr_path, sep="\t", index_col=0)

    expr_T = expr_matrix.T
    full_data = pd.merge(
        metadata.set_index(config["sample_id_col"]),
        expr_T,
        left_index=True,
        right_index=True,
    )

    normal_data = full_data[full_data[config["group_col"]] == config["normal_group_label"]].copy()
    intervention_data = full_data[full_data[config["group_col"]] != config["normal_group_label"]].copy()

    gene_columns = expr_matrix.index.tolist()

    print(f"  - Normal 样本数: {len(normal_data)}")
    print(f"  - Intervention 样本数: {len(intervention_data)}")
    
    # 异常值检测与去除（仅在Normal数据上进行，保持Intervention数据不变）
    normal_data, normal_outliers = detect_and_remove_outliers(
        normal_data, gene_columns, config["age_col"], config
    )
    
    # 更新full_data（移除Normal中的异常值）
    if normal_outliers:
        full_data = full_data.drop(index=normal_outliers)
    
    print(f"  - 异常值去除后 Normal 样本数: {len(normal_data)}")
    print(f"  - Intervention 样本数: {len(intervention_data)} (保持不变)")

    return {
        "normal_data": normal_data,
        "intervention_data": intervention_data,
        "all_data": full_data,
        "gene_columns": gene_columns,
        "metadata": metadata,
    }


def train_with_lassocv(X_raw, y, config):
    """
    使用 LassoCV + StandardScaler：
      - 所有 Normal 上做 HPO（alpha）
      - 返回已训练好的 model_cv、scaler 以及 best_params
    """
    print("\n步骤 1: 使用所有 Normal 样本做 HPO (LassoCV)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    model_cv = LassoCV(
        cv=config["cv_folds_internal"],
        max_iter=config["max_iter"],
        random_state=config["random_state"],
        n_jobs=-1,
    )
    model_cv.fit(X_scaled, y)

    best_params = {
        "alpha": float(model_cv.alpha_),
        "cv_folds_internal": config["cv_folds_internal"],
        "max_iter": config["max_iter"],
    }
    print(f"  - HPO 完成，最佳 alpha = {best_params['alpha']:.6f}")
    return model_cv, scaler, best_params


def train_fixed_lasso(X_raw, y, alpha, config):
    """
    在给定 alpha 下训练 LASSO（每个 fold 都重新训练自己的 scaler 和模型）。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    model = Lasso(
        alpha=alpha,
        max_iter=config["max_iter"],
        random_state=config["random_state"],
    )
    model.fit(X_scaled, y)
    return model, scaler


def evaluate_on_splits(model, scaler, splits_dict, gene_columns, metadata, config):
    """
    对多个子集（train/validation/intervention）生成预测并计算 age_acceleration。
    splits_dict: {name -> DataFrame}
    """
    frames = []
    for name, df in splits_dict.items():
        if df is None or len(df) == 0:
            continue
        X_raw = df[gene_columns]
        y_true = df[config["age_col"]]
        X_scaled = scaler.transform(X_raw)
        y_pred = model.predict(X_scaled)

        res = pd.DataFrame(
            {"actual_age": y_true, "predicted_age": y_pred, "data_type": name},
            index=df.index,
        )
        frames.append(res)

    all_res = pd.concat(frames)
    all_res = pd.merge(
        all_res.reset_index().rename(columns={"index": "sample_id"}),
        metadata,
        on="sample_id",
        how="left",
    )
    all_res["age_acceleration"] = all_res["predicted_age"] - all_res["actual_age"]
    return all_res


def save_coef_and_results(model, gene_columns, all_results, out_dir, prefix):
    """
    保存 LASSO 系数和预测结果。
    """
    os.makedirs(out_dir, exist_ok=True)

    coef_df = pd.DataFrame({"gene": gene_columns, "coefficient": model.coef_})
    coef_df = coef_df[coef_df["coefficient"] != 0].sort_values(
        "coefficient", key=abs, ascending=False
    )
    coef_path = os.path.join(out_dir, f"coefficients_{prefix}.csv")
    coef_df.to_csv(coef_path, index=False)

    pred_path = os.path.join(out_dir, f"predictions_{prefix}.csv")
    all_results.to_csv(pred_path, index=False)
    return coef_df, coef_path, pred_path


def plot_performance(train_df, val_df, metrics, out_path, title_prefix=""):
    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 2, 1)
    sns.regplot(x="actual_age", y="predicted_age", data=train_df, ax=ax1, scatter_kws={"alpha": 0.6})
    ax1.set_title(
        f"{title_prefix} Train (n={len(train_df)})\n"
        f"MAE={metrics['train_mae']:.2f}, R²={metrics['train_r2']:.2f}"
    )
    ax1.set_xlabel("Actual Age")
    ax1.set_ylabel("Predicted Age")

    ax2 = plt.subplot(1, 2, 2)
    sns.regplot(
        x="actual_age",
        y="predicted_age",
        data=val_df,
        ax=ax2,
        color="green",
        scatter_kws={"alpha": 0.6},
    )
    ax2.set_title(
        f"{title_prefix} Validation (n={len(val_df)})\n"
        f"MAE={metrics['val_mae']:.2f}, R²={metrics['val_r2']:.2f}"
    )
    ax2.set_xlabel("Actual Age")
    ax2.set_ylabel("Predicted Age")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_age_acceleration(all_results, config, out_path, title="Age Acceleration"):
    plt.figure(figsize=(12, 7))
    order = sorted(all_results[config["subgroup_col"]].unique())
    sns.boxplot(
        x=config["subgroup_col"],
        y="age_acceleration",
        data=all_results,
        palette="viridis",
        order=order,
    )
    sns.stripplot(
        x=config["subgroup_col"],
        y="age_acceleration",
        data=all_results,
        color="black",
        alpha=0.4,
        order=order,
    )
    plt.axhline(0, color="red", linestyle="--")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ===================================================================
#                       3. 各阶段流程
# ===================================================================

def stage_hpo_and_save_lasso(normal_data, gene_columns, metadata, config, root_dir):
    """
    阶段 1：
      - 所有 Normal 样本 + LassoCV 找到最佳 alpha
      - 保存 best_hyperparameters.json、模型、scaler、系数和预测
    """
    out_dir = os.path.join(root_dir, "HPO_LASSO")
    os.makedirs(out_dir, exist_ok=True)

    X_normal = normal_data[gene_columns]
    y_normal = normal_data[config["age_col"]]

    model_cv, scaler, best_params = train_with_lassocv(X_normal, y_normal, config)

    # 保存模型和 scaler
    joblib.dump(model_cv, os.path.join(out_dir, "lasso_best_model.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))

    # 保存最优超参数
    best_path = os.path.join(out_dir, "best_hyperparameters.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    print(f"  - 最优超参数已保存到: {best_path}")

    # 在 Normal 样本上生成预测及报告
    all_results = evaluate_on_splits(
        model_cv,
        scaler,
        {"train": normal_data},
        gene_columns,
        metadata,
        config,
    )
    coef_df, _, _ = save_coef_and_results(
        model_cv, gene_columns, all_results, out_dir, "HPO_all_normal"
    )

    train_df = all_results[all_results["data_type"] == "train"]
    train_mae = mean_absolute_error(train_df["actual_age"], train_df["predicted_age"])
    train_r2 = r2_score(train_df["actual_age"], train_df["predicted_age"])

    perf_plot_path = os.path.join(out_dir, "performance_train_only.png")
    plot_performance(
        train_df,
        train_df,
        {
            "train_mae": train_mae,
            "train_r2": train_r2,
            "val_mae": train_mae,
            "val_r2": train_r2,
        },
        perf_plot_path,
        title_prefix="HPO Best LASSO",
    )

    report_path = os.path.join(out_dir, "report_hpo_lasso.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Aging Clock HPO Report (LASSO) ===\n\n")
        f.write("Model: LassoCV\n")
        f.write(f"Best alpha: {best_params['alpha']:.6f}\n")
        f.write(f"Internal CV folds: {best_params['cv_folds_internal']}\n")
        f.write(f"Max iter: {best_params['max_iter']}\n\n")
        f.write("--- Training Performance (on all Normal) ---\n")
        f.write(f"  - MAE: {train_mae:.3f} years\n")
        f.write(f"  - R2 : {train_r2:.3f}\n\n")
        f.write(f"Number of selected (non-zero) features: {len(coef_df)}\n")
        f.write("\nTop 10 genes:\n")
        f.write(coef_df.head(10).to_string(index=False))

    return best_params, model_cv, scaler


def stage_kfold_cv_lasso(data_pkg, best_params, config, root_dir):
    """
    阶段 2：使用最佳 alpha 做 5-fold KFold 交叉验证评估 LASSO。
    """
    out_dir = os.path.join(root_dir, "5fold_cv_LASSO")
    os.makedirs(out_dir, exist_ok=True)

    normal_data = data_pkg["normal_data"]
    intervention_data = data_pkg["intervention_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]

    kf = KFold(
        n_splits=config["k_folds"],
        shuffle=True,
        random_state=config["random_state"],
    )

    all_fold_metrics = []
    cv_normal_list = []
    intervention_by_fold_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(normal_data)):
        fold_name = f"fold_{fold_idx + 1}"
        print("\n" + "=" * 50)
        print(f"阶段 2 - 5-fold LASSO CV: {fold_name} "
              f"({fold_idx + 1}/{config['k_folds']})")
        print("=" * 50)

        train_df = normal_data.iloc[train_idx].copy()
        val_df = normal_data.iloc[val_idx].copy()
        print(f"  - 训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}")

        model, scaler = train_fixed_lasso(
            train_df[gene_columns],
            train_df[config["age_col"]],
            alpha=best_params["alpha"],
            config=config,
        )

        fold_results = evaluate_on_splits(
            model,
            scaler,
            {
                "train": train_df,
                "validation": val_df,
                "intervention": intervention_data,
            },
            gene_columns,
            metadata,
            config,
        )

        fold_dir = os.path.join(out_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        joblib.dump(model, os.path.join(fold_dir, f"lasso_model_{fold_name}.pkl"))
        joblib.dump(scaler, os.path.join(fold_dir, f"scaler_{fold_name}.pkl"))

        coef_df, _, _ = save_coef_and_results(
            model, gene_columns, fold_results, fold_dir, fold_name
        )

        train_res = fold_results[fold_results["data_type"] == "train"]
        val_res = fold_results[fold_results["data_type"] == "validation"]
        intervention_res = fold_results[fold_results["data_type"] == "intervention"]

        metrics = {
            "fold": fold_name,
            "train_mae": mean_absolute_error(train_res["actual_age"], train_res["predicted_age"]),
            "train_r2": r2_score(train_res["actual_age"], train_res["predicted_age"]),
            "val_mae": mean_absolute_error(val_res["actual_age"], val_res["predicted_age"]),
            "val_r2": r2_score(val_res["actual_age"], val_res["predicted_age"]),
            "n_train": len(train_res),
            "n_val": len(val_res),
        }
        all_fold_metrics.append(metrics)

        perf_plot_path = os.path.join(fold_dir, f"performance_{fold_name}.png")
        plot_performance(train_res, val_res, metrics, perf_plot_path, title_prefix=f"LASSO {fold_name}")

        accel_plot_path = os.path.join(fold_dir, f"age_acceleration_{fold_name}.png")
        plot_age_acceleration(
            fold_results,
            config,
            accel_plot_path,
            title=f"Age Acceleration (LASSO {fold_name})",
        )

        with open(os.path.join(fold_dir, f"report_{fold_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== Report for {fold_name} (LASSO 5-fold CV) ===\n\n")
            f.write(f"alpha = {best_params['alpha']:.6f}\n")
            f.write(f"Number of non-zero features: {len(coef_df)}\n\n")
            f.write(f"Train MAE = {metrics['train_mae']:.3f}, R2 = {metrics['train_r2']:.3f}\n")
            f.write(f"Val   MAE = {metrics['val_mae']:.3f}, R2 = {metrics['val_r2']:.3f}\n")

        # 只保存当前 fold 的 validation 样本，并添加 fold 标识
        val_res_with_fold = val_res.copy()
        val_res_with_fold["fold"] = fold_name
        cv_normal_list.append(val_res_with_fold)
        
        # 只保存与当前 fold 的 validation 样本相同 group 的 intervention 样本，并添加 fold 标识
        if not intervention_res.empty:
            # 获取当前 fold validation 样本的所有 group
            group_col = config["subgroup_col"]
            if group_col in val_res.columns:
                val_groups = set(val_res[group_col].unique())
                # 过滤出与 validation 样本相同 group 的 intervention 样本
                if group_col in intervention_res.columns:
                    intervention_res_filtered = intervention_res[intervention_res[group_col].isin(val_groups)].copy()
                    if len(intervention_res_filtered) != len(intervention_res):
                        print(f"  信息: intervention 样本已过滤，只保留与 validation 相同 group 的样本。原始: {len(intervention_res)}, 过滤后: {len(intervention_res_filtered)}")
                else:
                    print(f"  警告: intervention 结果中找不到 '{group_col}' 列，无法按 group 过滤。")
                    intervention_res_filtered = intervention_res.copy()
            else:
                print(f"  警告: validation 结果中找不到 '{group_col}' 列，无法按 group 过滤 intervention。")
                intervention_res_filtered = intervention_res.copy()
            
            if not intervention_res_filtered.empty:
                intervention_res_with_fold = intervention_res_filtered.copy()
                intervention_res_with_fold["fold"] = fold_name
                intervention_by_fold_list.append(intervention_res_with_fold)

    cv_normal_all = pd.concat(cv_normal_list, ignore_index=True)
    
    # 合并所有 fold 的 intervention 结果（每个 fold 的预测结果不同）
    if intervention_by_fold_list:
        intervention_all = pd.concat(intervention_by_fold_list, ignore_index=True)
    else:
        intervention_all = pd.DataFrame()

    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(os.path.join(out_dir, "summary_metrics_by_fold.csv"), index=False)

    overall_mae = mean_absolute_error(
        cv_normal_all["actual_age"], cv_normal_all["predicted_age"]
    )
    overall_r2 = r2_score(cv_normal_all["actual_age"], cv_normal_all["predicted_age"])

    plt.figure(figsize=(8, 7))
    sns.regplot(
        x="actual_age",
        y="predicted_age",
        data=cv_normal_all,
        scatter_kws={"alpha": 0.7},
    )
    plt.title(
        f"Overall LASSO {config['k_folds']}-Fold CV (Normal)\n"
        f"MAE={overall_mae:.2f}, R²={overall_r2:.2f}",
        fontsize=14,
    )
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age (Out-of-sample)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_performance_cv_normal.png"), dpi=300)
    plt.close()

    # 合并所有 fold 的结果（validation + intervention），所有结果都包含 fold 标识
    if not intervention_all.empty:
        combined = pd.concat([cv_normal_all, intervention_all], ignore_index=True)
    else:
        combined = cv_normal_all.copy()
    combined.to_csv(os.path.join(out_dir, "summary_all_predictions_cv.csv"), index=False)

    plot_age_acceleration(
        combined,
        config,
        os.path.join(out_dir, "summary_age_acceleration.png"),
        title="Overall Age Acceleration (LASSO 5-fold CV)",
    )

    with open(os.path.join(out_dir, "summary_report_5fold_lasso_cv.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== Final LASSO 5-Fold CV Summary ({config['k_folds']} folds) ===\n\n")
        f.write(f"Overall Normal (out-of-sample) performance:\n")
        f.write(f"  - MAE: {overall_mae:.3f} years\n")
        f.write(f"  - R2 : {overall_r2:.3f}\n\n")
        f.write("--- Metrics by fold ---\n")
        f.write(metrics_df.to_string(index=False))


def stage_dataset_loo_lasso(data_pkg, best_params, config, root_dir):
    """
    阶段 3：按 batch 列进行 LASSO 的 Leave-One-Batch-Out (DataSet-LOO) CV。
    """
    out_dir = os.path.join(root_dir, "DataSet_loo_LASSO")
    os.makedirs(out_dir, exist_ok=True)

    normal_data = data_pkg["normal_data"]
    intervention_data = data_pkg["intervention_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]

    if config["batch_col"] not in normal_data.columns:
        raise ValueError(f"在 normal_data 中找不到批次列 '{config['batch_col']}'")

    unique_batches = sorted(normal_data[config["batch_col"]].unique())
    print("\n步骤 3: LASSO DataSet-LOO (按 batch leave-one-batch-out)...")
    print(f"  - 共发现 {len(unique_batches)} 个 batch: {unique_batches}")

    all_fold_metrics = []
    cv_normal_list = []
    intervention_by_fold_list = []

    for idx, val_batch in enumerate(unique_batches):
        fold_name = f"fold_{idx + 1}_val_{val_batch}"
        print("\n" + "=" * 50)
        print(f"LASSO DataSet-LOO: {fold_name}")
        print("=" * 50)

        train_df = normal_data[normal_data[config["batch_col"]] != val_batch].copy()
        val_df = normal_data[normal_data[config["batch_col"]] == val_batch].copy()
        print(f"  - 训练集大小: {len(train_df)}, 验证集(batch={val_batch})大小: {len(val_df)}")

        model, scaler = train_fixed_lasso(
            train_df[gene_columns],
            train_df[config["age_col"]],
            alpha=best_params["alpha"],
            config=config,
        )

        fold_results = evaluate_on_splits(
            model,
            scaler,
            {
                "train": train_df,
                "validation": val_df,
                "intervention": intervention_data,
            },
            gene_columns,
            metadata,
            config,
        )

        fold_dir = os.path.join(out_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        joblib.dump(model, os.path.join(fold_dir, f"lasso_model_{fold_name}.pkl"))
        joblib.dump(scaler, os.path.join(fold_dir, f"scaler_{fold_name}.pkl"))
        coef_df, _, _ = save_coef_and_results(
            model, gene_columns, fold_results, fold_dir, fold_name
        )

        train_res = fold_results[fold_results["data_type"] == "train"]
        val_res = fold_results[fold_results["data_type"] == "validation"]
        intervention_res = fold_results[fold_results["data_type"] == "intervention"]

        # 确保 validation 样本只包含当前被留出的 batch（双重检查）
        val_res_filtered = val_res[val_res[config["batch_col"]] == val_batch].copy()
        if len(val_res_filtered) != len(val_res):
            print(f"  警告: validation 样本的 batch 不匹配，已过滤。原始: {len(val_res)}, 过滤后: {len(val_res_filtered)}")

        metrics = {
            "fold": fold_name,
            "val_batch": val_batch,
            "train_mae": mean_absolute_error(train_res["actual_age"], train_res["predicted_age"]),
            "train_r2": r2_score(train_res["actual_age"], train_res["predicted_age"]),
            "val_mae": mean_absolute_error(val_res_filtered["actual_age"], val_res_filtered["predicted_age"]),
            "val_r2": r2_score(val_res_filtered["actual_age"], val_res_filtered["predicted_age"]),
            "n_train": len(train_res),
            "n_val": len(val_res_filtered),
        }
        all_fold_metrics.append(metrics)

        perf_plot_path = os.path.join(fold_dir, f"performance_{fold_name}.png")
        plot_performance(train_res, val_res_filtered, metrics, perf_plot_path, title_prefix=f"LASSO {fold_name}")

        accel_plot_path = os.path.join(fold_dir, f"age_acceleration_{fold_name}.png")
        plot_age_acceleration(
            fold_results,
            config,
            accel_plot_path,
            title=f"Age Acceleration (LASSO {fold_name})",
        )

        with open(os.path.join(fold_dir, f"report_{fold_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== Report for {fold_name} (LASSO DataSet-LOO) ===\n\n")
            f.write(f"Validation batch: {val_batch}\n")
            f.write(f"alpha = {best_params['alpha']:.6f}\n")
            f.write(f"Number of non-zero features: {len(coef_df)}\n\n")
            f.write(f"Train MAE = {metrics['train_mae']:.3f}, R2 = {metrics['train_r2']:.3f}\n")
            f.write(f"Val   MAE = {metrics['val_mae']:.3f}, R2 = {metrics['val_r2']:.3f}\n")

        # 只保存当前 fold 对应的 validation batch 的样本，并添加 fold 标识
        val_res_with_fold = val_res_filtered.copy()
        val_res_with_fold["fold"] = fold_name
        cv_normal_list.append(val_res_with_fold)
        
        # 只保存当前 fold 对应的 batch 的 intervention 样本，并添加 fold 标识
        if not intervention_res.empty:
            # 过滤出与当前被留出的 batch 相同的 intervention 样本
            if config["batch_col"] in intervention_res.columns:
                intervention_res_filtered = intervention_res[intervention_res[config["batch_col"]] == val_batch].copy()
                if len(intervention_res_filtered) != len(intervention_res):
                    print(f"  信息: intervention 样本已过滤，只保留 batch={val_batch} 的样本。原始: {len(intervention_res)}, 过滤后: {len(intervention_res_filtered)}")
            else:
                print(f"  警告: intervention 结果中找不到 '{config['batch_col']}' 列，无法按 batch 过滤。")
                intervention_res_filtered = intervention_res.copy()
            
            if not intervention_res_filtered.empty:
                intervention_res_with_fold = intervention_res_filtered.copy()
                intervention_res_with_fold["fold"] = fold_name
                intervention_by_fold_list.append(intervention_res_with_fold)

    cv_normal_all = pd.concat(cv_normal_list, ignore_index=True)
    
    # 合并所有 fold 的 intervention 结果（每个 fold 的预测结果不同）
    if intervention_by_fold_list:
        intervention_all = pd.concat(intervention_by_fold_list, ignore_index=True)
    else:
        intervention_all = pd.DataFrame()

    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(os.path.join(out_dir, "summary_metrics_by_batch.csv"), index=False)

    overall_mae = mean_absolute_error(
        cv_normal_all["actual_age"], cv_normal_all["predicted_age"]
    )
    overall_r2 = r2_score(cv_normal_all["actual_age"], cv_normal_all["predicted_age"])

    plt.figure(figsize=(8, 7))
    sns.regplot(
        x="actual_age",
        y="predicted_age",
        data=cv_normal_all,
        scatter_kws={"alpha": 0.7},
    )
    plt.title(
        "Overall LASSO DataSet-LOO (Normal, leave-one-batch-out)\n"
        f"MAE={overall_mae:.2f}, R²={overall_r2:.2f}",
        fontsize=14,
    )
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age (Out-of-sample)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_performance_cv_normal.png"), dpi=300)
    plt.close()

    # 合并所有 fold 的结果（validation + intervention），所有结果都包含 fold 标识
    if not intervention_all.empty:
        combined = pd.concat([cv_normal_all, intervention_all], ignore_index=True)
    else:
        combined = cv_normal_all.copy()
    combined.to_csv(os.path.join(out_dir, "summary_all_predictions_cv.csv"), index=False)

    plot_age_acceleration(
        combined,
        config,
        os.path.join(out_dir, "summary_age_acceleration.png"),
        title="Overall Age Acceleration (LASSO DataSet-LOO)",
    )

    with open(os.path.join(out_dir, "summary_report_dataset_loo_lasso.txt"), "w", encoding="utf-8") as f:
        f.write("=== Final LASSO DataSet-LOO Summary (Leave-One-Batch-Out) ===\n\n")
        f.write("Overall Normal (out-of-sample) performance:\n")
        f.write(f"  - MAE: {overall_mae:.3f} years\n")
        f.write(f"  - R2 : {overall_r2:.3f}\n\n")
        f.write("--- Metrics by batch ---\n")
        f.write(metrics_df.to_string(index=False))


def stage_group_loo(data_pkg, best_params, config, root_dir):
    """
    阶段 4：按 subgroup（config['subgroup_col']，即 total_metadata.txt 中的 group 列）进行 Leave-One-Group-Out CV。
    仅在 Normal 样本上做 group-LOO，Intervention 作为额外测试集。
    使用 sklearn 的 LeaveOneGroupOut 实现标准的按组留一法交叉验证。
    每次留出一个 group 作为验证集，其余 groups 作为训练集。
    """
    out_dir = os.path.join(root_dir, "group_loo")
    os.makedirs(out_dir, exist_ok=True)

    normal_data = data_pkg["normal_data"]
    intervention_data = data_pkg["intervention_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]

    group_col = config["subgroup_col"]  # 使用 total_metadata.txt 中的 "group" 列
    if group_col not in normal_data.columns:
        raise ValueError(f"在 normal_data 中找不到 group 列 '{group_col}'（应来自 total_metadata.txt）")

    unique_groups = sorted(normal_data[group_col].unique())
    print("\n步骤 4: Group-LOO (按 subgroup leave-one-group-out)...")
    print(f"  - 使用列: {group_col}（来自 total_metadata.txt）")
    print(f"  - 共发现 {len(unique_groups)} 个 group: {unique_groups}")

    # 使用 LeaveOneGroupOut 进行分组（基于 total_metadata.txt 中的 group 列）
    groups = normal_data[group_col].values
    logo = LeaveOneGroupOut()
    
    all_fold_metrics = []
    cv_normal_list = []
    intervention_by_fold_list = []

    for idx, (train_idx, val_idx) in enumerate(logo.split(normal_data, normal_data[config["age_col"]], groups)):
        val_group = normal_data.iloc[val_idx][group_col].iloc[0]  # 获取验证集的 group
        fold_name = f"fold_{idx + 1}_val_{val_group}"
        print("\n" + "=" * 50)
        print(f"Group-LOO: {fold_name}")
        print("=" * 50)

        train_df = normal_data.iloc[train_idx].copy()
        val_df = normal_data.iloc[val_idx].copy()
        print(f"  - 训练集大小: {len(train_df)}, 验证集({group_col}={val_group})大小: {len(val_df)}")

        model, scaler = train_fixed_lasso(
            train_df[gene_columns],
            train_df[config["age_col"]],
            alpha=best_params["alpha"],
            config=config,
        )

        fold_results = evaluate_on_splits(
            model,
            scaler,
            {
                "train": train_df,
                "validation": val_df,
                "intervention": intervention_data,
            },
            gene_columns,
            metadata,
            config,
        )

        fold_dir = os.path.join(out_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        joblib.dump(model, os.path.join(fold_dir, f"lasso_model_{fold_name}.pkl"))
        joblib.dump(scaler, os.path.join(fold_dir, f"scaler_{fold_name}.pkl"))
        coef_df, _, _ = save_coef_and_results(
            model, gene_columns, fold_results, fold_dir, fold_name
        )

        train_res = fold_results[fold_results["data_type"] == "train"]
        val_res = fold_results[fold_results["data_type"] == "validation"]
        intervention_res = fold_results[fold_results["data_type"] == "intervention"]

        # 确保 validation 样本只包含当前被留出的 group（双重检查）
        val_res_filtered = val_res[val_res[group_col] == val_group].copy()
        if len(val_res_filtered) != len(val_res):
            print(f"  警告: validation 样本的 group 不匹配，已过滤。原始: {len(val_res)}, 过滤后: {len(val_res_filtered)}")

        metrics = {
            "fold": fold_name,
            "val_group": val_group,
            "train_mae": mean_absolute_error(train_res["actual_age"], train_res["predicted_age"]),
            "train_r2": r2_score(train_res["actual_age"], train_res["predicted_age"]),
            "val_mae": mean_absolute_error(val_res_filtered["actual_age"], val_res_filtered["predicted_age"]),
            "val_r2": r2_score(val_res_filtered["actual_age"], val_res_filtered["predicted_age"]),
            "n_train": len(train_res),
            "n_val": len(val_res_filtered),
        }
        all_fold_metrics.append(metrics)

        perf_plot_path = os.path.join(fold_dir, f"performance_{fold_name}.png")
        plot_performance(train_res, val_res_filtered, metrics, perf_plot_path, title_prefix=f"LASSO {fold_name}")

        accel_plot_path = os.path.join(fold_dir, f"age_acceleration_{fold_name}.png")
        plot_age_acceleration(
            fold_results,
            config,
            accel_plot_path,
            title=f"Age Acceleration (LASSO {fold_name})",
        )

        with open(os.path.join(fold_dir, f"report_{fold_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== Report for {fold_name} (LASSO Group-LOO) ===\n\n")
            f.write(f"Validation group ({group_col}): {val_group}\n")
            f.write(f"alpha = {best_params['alpha']:.6f}\n")
            f.write(f"Number of non-zero features: {len(coef_df)}\n\n")
            f.write(f"Train MAE = {metrics['train_mae']:.3f}, R2 = {metrics['train_r2']:.3f}\n")
            f.write(f"Val   MAE = {metrics['val_mae']:.3f}, R2 = {metrics['val_r2']:.3f}\n")

        # 只保存当前 fold 对应的 validation group 的样本，并添加 fold 标识
        val_res_with_fold = val_res_filtered.copy()
        val_res_with_fold["fold"] = fold_name
        cv_normal_list.append(val_res_with_fold)
        
        # 只保存当前 fold 对应的 group 的 intervention 样本，并添加 fold 标识
        if not intervention_res.empty:
            # 过滤出与当前被留出的 group 相同的 intervention 样本
            if group_col in intervention_res.columns:
                intervention_res_filtered = intervention_res[intervention_res[group_col] == val_group].copy()
                if len(intervention_res_filtered) != len(intervention_res):
                    print(f"  信息: intervention 样本已过滤，只保留 group={val_group} 的样本。原始: {len(intervention_res)}, 过滤后: {len(intervention_res_filtered)}")
            else:
                print(f"  警告: intervention 结果中找不到 '{group_col}' 列，无法按 group 过滤。")
                intervention_res_filtered = intervention_res.copy()
            
            if not intervention_res_filtered.empty:
                intervention_res_with_fold = intervention_res_filtered.copy()
                intervention_res_with_fold["fold"] = fold_name
                intervention_by_fold_list.append(intervention_res_with_fold)

    cv_normal_all = pd.concat(cv_normal_list, ignore_index=True)
    
    # 合并所有 fold 的 intervention 结果（每个 fold 的预测结果不同）
    if intervention_by_fold_list:
        intervention_all = pd.concat(intervention_by_fold_list, ignore_index=True)
    else:
        intervention_all = pd.DataFrame()

    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(os.path.join(out_dir, "summary_metrics_by_group.csv"), index=False)

    overall_mae = mean_absolute_error(
        cv_normal_all["actual_age"], cv_normal_all["predicted_age"]
    )
    overall_r2 = r2_score(cv_normal_all["actual_age"], cv_normal_all["predicted_age"])

    plt.figure(figsize=(8, 7))
    sns.regplot(
        x="actual_age",
        y="predicted_age",
        data=cv_normal_all,
        scatter_kws={"alpha": 0.7},
    )
    plt.title(
        "Overall LASSO Group-LOO Performance (Normal, leave-one-group-out)\n"
        f"MAE={overall_mae:.2f}, R²={overall_r2:.2f}",
        fontsize=14,
    )
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age (Out-of-sample)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_performance_cv_normal.png"), dpi=300)
    plt.close()

    # 合并所有 fold 的结果（validation + intervention），所有结果都包含 fold 标识
    if not intervention_all.empty:
        combined = pd.concat([cv_normal_all, intervention_all], ignore_index=True)
    else:
        combined = cv_normal_all.copy()
    combined.to_csv(os.path.join(out_dir, "summary_all_predictions_cv.csv"), index=False)

    plot_age_acceleration(
        combined,
        config,
        os.path.join(out_dir, "summary_age_acceleration.png"),
        title="Overall Age Acceleration (LASSO Group-LOO)",
    )

    with open(os.path.join(out_dir, "summary_report_group_loo.txt"), "w", encoding="utf-8") as f:
        f.write("=== Final LASSO Group-LOO Summary (Leave-One-Group-Out by subgroup) ===\n\n")
        f.write("Overall Normal (out-of-sample) performance:\n")
        f.write(f"  - MAE: {overall_mae:.3f} years\n")
        f.write(f"  - R2 : {overall_r2:.3f}\n\n")
        f.write("--- Metrics by group ---\n")
        f.write(metrics_df.to_string(index=False))


def stage_group_prediction_with_best_lasso(data_pkg, best_model, best_scaler, config, root_dir):
    """
    阶段 5：用"最优模型"（HPO 阶段的 LassoCV 模型 + scaler）
    在所有样本 (Normal + Intervention) 上预测年龄，并统计每个 group 的年龄。

    注意：这里直接用 HPO 时训练在所有 Normal 上的 model_cv，不再重新训练。
    """
    out_dir = os.path.join(root_dir, "group_best_model")
    os.makedirs(out_dir, exist_ok=True)

    all_data = data_pkg["all_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]

    print("\n步骤 5: 使用 HPO 得到的最优模型，对所有样本预测，并按 group 汇总...")

    X_all = all_data[gene_columns]
    y_all = all_data[config["age_col"]]

    X_all_scaled = best_scaler.transform(X_all)
    y_pred_all = best_model.predict(X_all_scaled)

    res = pd.DataFrame(
        {
            "actual_age": y_all,
            "predicted_age": y_pred_all,
        },
        index=all_data.index,
    )
    res = pd.merge(
        res.reset_index().rename(columns={"index": "sample_id"}),
        metadata,
        on="sample_id",
        how="left",
    )
    res["age_acceleration"] = res["predicted_age"] - res["actual_age"]

    # 保存所有样本预测
    all_pred_path = os.path.join(out_dir, "all_samples_predictions_best_model.csv")
    res.to_csv(all_pred_path, index=False)

    # 按 group 汇总年龄（就是你说的"每个 group 的年龄"）
    group_col = config["subgroup_col"]
    group_summary = (
        res.groupby(group_col)
        .agg(
            n_samples=("sample_id", "count"),
            mean_actual_age=("actual_age", "mean"),
            mean_predicted_age=("predicted_age", "mean"),
            mean_age_acceleration=("age_acceleration", "mean"),
        )
        .reset_index()
    )
    group_summary_path = os.path.join(out_dir, "group_age_summary.csv")
    group_summary.to_csv(group_summary_path, index=False)

    # 每个 group 的实际 vs 预测年龄条形图
    plt.figure(figsize=(12, 7))
    x = np.arange(len(group_summary))
    width = 0.35
    plt.bar(x - width / 2, group_summary["mean_actual_age"], width, label="Actual")
    plt.bar(x + width / 2, group_summary["mean_predicted_age"], width, label="Predicted")
    plt.xticks(x, group_summary[group_col], rotation=45, ha="right")
    plt.ylabel("Age (years)")
    plt.title("Mean Actual & Predicted Age per Group (Best Model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "group_mean_age_barplot.png"), dpi=300)
    plt.close()

    # 画 age_acceleration 的 boxplot
    plot_age_acceleration(
        res,
        config,
        os.path.join(out_dir, "age_acceleration_by_group_best_model.png"),
        title="Age Acceleration by Group (Best Model on All Data)",
    )

    # 简要报告
    with open(os.path.join(out_dir, "report_group_best_model.txt"), "w", encoding="utf-8") as f:
        f.write("=== Group-level Predictions with Best Model ===\n\n")
        f.write("该模型来自 HPO 阶段，在所有 Normal 样本上用 LassoCV 训练，"
                "然后用于全体样本的预测。\n\n")
        f.write("--- Group-wise summary ---\n")
        f.write(group_summary.to_string(index=False))


# ===================================================================
#                       4. 主入口
# ===================================================================

if __name__ == "__main__":
    # 解析并创建根输出目录
    root_dir = resolve_path(CONFIG["output_root_dir"])
    os.makedirs(root_dir, exist_ok=True)
    print(f"输出根目录: {root_dir}")

    # 读取完整数据
    data_pkg = load_full_data(CONFIG)

    # 阶段 1: HPO + 最优 LASSO 模型
    best_params, best_model, best_scaler = stage_hpo_and_save_lasso(
        data_pkg["normal_data"],
        data_pkg["gene_columns"],
        data_pkg["metadata"],
        CONFIG,
        root_dir,
    )

    # 阶段 2: 5-fold CV 测试预测能力
    stage_kfold_cv_lasso(data_pkg, best_params, CONFIG, root_dir)

    # 阶段 3: DataSet-LOO (按 batch leave-one-batch-out)
    stage_dataset_loo_lasso(data_pkg, best_params, CONFIG, root_dir)

    # 阶段 4: Group-LOO (按 subgroup leave-one-group-out)
    stage_group_loo(data_pkg, best_params, CONFIG, root_dir)

    # 阶段 5: 用 HPO 得到的最优模型，预测所有样本，并按 group 汇总年龄
    stage_group_prediction_with_best_lasso(data_pkg, best_model, best_scaler, CONFIG, root_dir)

    print("\n✅ 全部流程完成！结果请查看输出根目录下的各个子文件夹。")