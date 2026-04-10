import os
import json
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================================================================
#                       1. 配置中心 (CONFIGURATION)
#        与 MoE pipeline 相同的总体流程：HPO -> 5fold -> DataSet_loo -> group_loo -> group_best_model
# ===================================================================
CONFIG = {
    # --- 文件与路径（默认认为和本脚本在同一目录下） ---
    "metadata_file": "total_metadata.txt",
    "expression_file": "total_count_sort_ortholog_TMM_batched.txt",
    "aging_genes_file": "aging_gene_list.csv",
    "output_root_dir": "aging_clock_results_DeepQA_pipeline",

    # --- 数据列名 ---
    "sample_id_col": "sample_id",
    "group_col": "condition",
    "subgroup_col": "group",
    "age_col": "Chronological_Age",
    "batch_col": "batch",

    # --- 数据划分 ---
    "normal_group_label": "Normal",
    "test_split_size": 0.2,
    "seed": 66,

    # --- 异常值检测与去除 ---
    "outlier_detection": {
        "enabled": True,
        "method": "iqr",
        "iqr_factor": 1.5,
        "zscore_threshold": 3.0,
        "check_age": False,
        "age_iqr_factor": 3.0,
    },

    # --- 外层评估 ---
    "k_folds": 5,

    # --- 模型与训练超参数 ---
    "num_experts": 2,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "batch_size": 32,
    "lambda_pred_loss": 30.0,
    "gamma_ahk_loss": 80.0,

    # --- 早停与学习率调度 ---
    "early_stopping": {
        "enabled": True,
        "patience": 15,
        "min_delta": 0.0,
        "monitor_interval": 1,
        "restore_best": True
    },
    "lr_scheduler": {
        "enabled": True,
        "mode": "min",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-6
    },

    # --- Optuna 搜索配置 ---
    "optuna": {
        "enabled": True,
        "trials": 25,
        "epochs": 60
    }
}

# ===================================================================
#                       2. 公共函数与模型定义
# ===================================================================


def set_seed(seed):
    """设置所有相关的随机种子以确保结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_path(path: str) -> str:
    """把相对路径解析为脚本所在目录下的绝对路径。"""
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def detect_and_remove_outliers(data, gene_columns, age_col, config):
    """检测并去除异常值样本。"""
    outlier_cfg = config.get("outlier_detection", {})
    if not outlier_cfg.get("enabled", False):
        print("  - 异常值检测已禁用")
        return data, []

    print("  - 开始异常值检测...")
    original_count = len(data)
    outlier_indices = set()
    method = outlier_cfg.get("method", "iqr")

    if method == "iqr":
        iqr_factor = outlier_cfg.get("iqr_factor", 1.5)
        print(f"    - 使用IQR方法 (因子={iqr_factor})")
        expr_data = data[gene_columns]
        sample_stats = expr_data.sum(axis=1)
        Q1 = sample_stats.quantile(0.25)
        Q3 = sample_stats.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        expr_outliers = sample_stats[(sample_stats < lower_bound) | (sample_stats > upper_bound)].index
        outlier_indices.update(expr_outliers)
        print(f"    - 表达数据异常值: {len(expr_outliers)} 个样本")
    elif method == "zscore":
        zscore_threshold = outlier_cfg.get("zscore_threshold", 3.0)
        print(f"    - 使用Z-score方法 (阈值={zscore_threshold})")
        expr_data = data[gene_columns]
        sample_stats = expr_data.sum(axis=1)
        z_scores = np.abs((sample_stats - sample_stats.mean()) / sample_stats.std())
        expr_outliers = sample_stats[z_scores > zscore_threshold].index
        outlier_indices.update(expr_outliers)
        print(f"    - 表达数据异常值: {len(expr_outliers)} 个样本")

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


class Expert(nn.Module):
    def __init__(self, input_dim):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 640), nn.BatchNorm1d(640), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(640, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 640), nn.BatchNorm1d(640), nn.ReLU(),
            nn.Linear(640, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, num_experts))
    def forward(self, x):
        return F.softmax(self.net(x), dim=1)


class DeepQA_MoE(nn.Module):
    def __init__(self, input_dim, num_experts=2):
        super(DeepQA_MoE, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim) for _ in range(num_experts)])
        self.gating = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        gating_weights = self.gating(x)
        expert_outputs = torch.cat([expert(x) for expert in self.experts], dim=1)
        final_output = torch.sum(gating_weights * expert_outputs, dim=1, keepdim=True)
        return final_output, gating_weights, expert_outputs


def ahk_loss(model_output, model_input, aging_gene_indices_tensor):
    if model_input.grad is not None:
        model_input.grad.zero_()
    saliency_map = torch.autograd.grad(outputs=model_output.sum(), inputs=model_input, create_graph=True)[0]
    gene_importance = saliency_map.abs().sum(dim=0)
    total_importance = torch.sum(gene_importance) + 1e-10
    sum_aging_importance = torch.sum(gene_importance[aging_gene_indices_tensor])
    return -(sum_aging_importance / total_importance)


# ===================================================================
#                       3. 数据加载与基础训练函数
# ===================================================================


def load_full_data(config):
    """
    读取元数据、表达矩阵和衰老基因列表，并对齐；对 Normal 做异常值去除。
    返回 data_pkg: normal_data, intervention_data, all_data, gene_columns, metadata, aging_gene_indices, aging_gene_pool
    """
    print("步骤 0: 加载和预处理数据...")
    metadata_path = resolve_path(config["metadata_file"])
    expr_path = resolve_path(config["expression_file"])
    aging_genes_path = resolve_path(config["aging_genes_file"])
    print(f"  - 读取元数据: {metadata_path}")
    print(f"  - 读取表达矩阵: {expr_path}")
    print(f"  - 读取衰老基因列表: {aging_genes_path}")

    metadata = pd.read_csv(metadata_path, sep="\t")
    expr_matrix = pd.read_csv(expr_path, sep="\t", index_col=0)
    aging_genes_df = pd.read_csv(aging_genes_path)

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

    aging_gene_pool = set(aging_genes_df.iloc[:, 0])
    aging_gene_indices = np.array([i for i, gene in enumerate(gene_columns) if gene in aging_gene_pool])
    print(f"  - 在 {len(gene_columns)} 个基因中，匹配到 {len(aging_gene_indices)} 个衰老基因。")
    print(f"  - Normal 样本数: {len(normal_data)}")
    print(f"  - Intervention 样本数: {len(intervention_data)}")

    normal_data, normal_outliers = detect_and_remove_outliers(
        normal_data, gene_columns, config["age_col"], config
    )
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
        "aging_gene_indices": aging_gene_indices,
        "aging_gene_pool": aging_gene_pool,
    }


def train_deepqa_clock(data_splits, config):
    """标准化数据并训练 DeepQA MoE 模型（含早停与可选 LR 调度，监控验证集 MAE）。"""
    print("\n步骤: 数据标准化并训练 DeepQA MoE 模型...")
    X_train_raw, y_train = data_splits["X_train_raw"], data_splits["y_train"]
    X_val_raw, y_val = data_splits["X_val_raw"], data_splits["y_val"]
    aging_gene_indices = data_splits["aging_gene_indices"]

    scaler_X = StandardScaler().fit(X_train_raw)
    scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))
    X_train = scaler_X.transform(X_train_raw)
    y_train_scaled = scaler_y.transform(y_train.values.reshape(-1, 1))

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = DeepQA_MoE(X_train.shape[1], config["num_experts"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = None
    if config.get("lr_scheduler", {}).get("enabled", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config["lr_scheduler"].get("mode", "min"),
            factor=config["lr_scheduler"].get("factor", 0.5),
            patience=config["lr_scheduler"].get("patience", 5),
            min_lr=config["lr_scheduler"].get("min_lr", 1e-6),
        )
    aging_gene_indices_t = torch.tensor(aging_gene_indices, dtype=torch.long).to(device)

    best_mae = float("inf")
    best_state = None
    no_improve_epochs = 0
    es_cfg = config.get("early_stopping", {})
    es_enabled = es_cfg.get("enabled", False)
    monitor_interval = max(1, int(es_cfg.get("monitor_interval", 1)))

    for epoch in range(config["epochs"]):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X.requires_grad_(True)
            optimizer.zero_grad()
            y_pred, _, _ = model(batch_X)
            pred_loss = F.l1_loss(y_pred, batch_y)
            ahk = ahk_loss(y_pred, batch_X, aging_gene_indices_t)
            total_loss = config["lambda_pred_loss"] * pred_loss + config["gamma_ahk_loss"] * ahk
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if (epoch + 1) % monitor_interval == 0:
            val_mae, val_r2 = evaluate_on_validation(
                model, (scaler_X, scaler_y), X_val_raw, y_val,
            )
            if scheduler is not None:
                scheduler.step(val_mae)
            print(
                f"Epoch {epoch+1}/{config['epochs']}, "
                f"TrainLoss: {total_loss.item():.4f}, ValMAE: {val_mae:.4f}, ValR2: {val_r2:.4f}"
            )
            improved = (best_mae - val_mae) > es_cfg.get("min_delta", 0.0)
            if improved:
                best_mae = val_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            if es_enabled and no_improve_epochs >= es_cfg.get("patience", 15):
                print(f"触发早停: 连续 {no_improve_epochs} 个评估周期无提升。最佳验证MAE={best_mae:.4f}")
                break
    print("模型训练完成。")
    if es_enabled and es_cfg.get("restore_best", True) and best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler_X, scaler_y


def evaluate_on_validation(model, scalers, X_val_raw, y_val):
    """使用验证集计算 MAE 与 R²（以原始年龄单位）。"""
    scaler_X, scaler_y = scalers
    device = next(model.parameters()).device
    X_val_scaled = scaler_X.transform(X_val_raw)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred_scaled, _, _ = model(X_val_t)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.cpu().numpy()).flatten()
    mae = mean_absolute_error(y_val.values, y_pred)
    r2 = r2_score(y_val.values, y_pred)
    return mae, r2


def optuna_search_hpo(data_splits, base_config):
    """使用 Optuna 进行超参数优化；若未安装则返回空。"""
    try:
        import optuna
    except Exception:
        warnings.warn("未检测到 Optuna，将跳过超参数搜索。安装: pip install optuna")
        return {}

    def objective(trial):
        lr = 10 ** trial.suggest_float("learning_rate_log10", -5, -3)
        wd = 10 ** trial.suggest_float("weight_decay_log10", -6, -3)
        num_experts = trial.suggest_categorical("num_experts", [1, 2, 3, 4])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        lambda_pred = trial.suggest_categorical("lambda_pred_loss", [10.0, 20.0, 30.0, 50.0, 80.0, 100.0])
        gamma_ahk = trial.suggest_categorical("gamma_ahk_loss", [10.0, 30.0, 50.0, 100.0, 200.0, 300.0])
        cfg = dict(base_config)
        cfg.update({
            "learning_rate": lr,
            "weight_decay": wd,
            "num_experts": num_experts,
            "batch_size": batch_size,
            "lambda_pred_loss": lambda_pred,
            "gamma_ahk_loss": gamma_ahk,
            "epochs": int(base_config.get("optuna", {}).get("epochs", base_config["epochs"])),
        })
        model, scaler_X, scaler_y = train_deepqa_clock(data_splits, cfg)
        mae, _ = evaluate_on_validation(
            model, (scaler_X, scaler_y),
            data_splits["X_val_raw"],
            data_splits["y_val"],
        )
        return mae

    n_trials = int(base_config.get("optuna", {}).get("trials", 0))
    if n_trials <= 0:
        return {}
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    out = {
        "learning_rate": 10 ** best.get("learning_rate_log10", -4),
        "weight_decay": 10 ** best.get("weight_decay_log10", -4),
        "num_experts": best.get("num_experts", base_config["num_experts"]),
        "batch_size": best.get("batch_size", base_config["batch_size"]),
        "lambda_pred_loss": best.get("lambda_pred_loss", base_config["lambda_pred_loss"]),
        "gamma_ahk_loss": best.get("gamma_ahk_loss", base_config["gamma_ahk_loss"]),
    }
    print("Optuna 最优参数:", out)
    return out


def evaluate_on_splits(model, scalers, splits_dict, gene_columns, metadata, config):
    """对给定的多个数据子集生成预测并计算 age_acceleration。"""
    scaler_X, scaler_y = scalers
    device = next(model.parameters()).device
    model.eval()
    frames = []
    for name, df in splits_dict.items():
        if df is None or len(df) == 0:
            continue
        X_raw = df[gene_columns]
        y_true = df[config["age_col"]]
        X_scaled = scaler_X.transform(X_raw)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_scaled, _, _ = model(X_t)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.cpu().numpy()).flatten()
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


def compute_gene_importance(model, scaler_X, X_raw, gene_columns):
    """使用梯度方法计算基因重要性，返回按重要性排序的 DataFrame。"""
    device = next(model.parameters()).device
    X_scaled = scaler_X.transform(X_raw)
    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device, requires_grad=True)
    model.eval()
    y_pred, _, _ = model(X_t)
    saliency_scores = torch.autograd.grad(outputs=y_pred.sum(), inputs=X_t, retain_graph=False)[0]
    gene_importance_scores = saliency_scores.abs().mean(dim=0).cpu().detach().numpy()
    importance_df = pd.DataFrame({"gene": gene_columns, "importance": gene_importance_scores})
    importance_df = importance_df.sort_values("importance", ascending=False)
    return importance_df


def save_gene_importance_and_results(importance_df, all_results, out_dir, prefix):
    """保存基因重要性和预测结果。"""
    os.makedirs(out_dir, exist_ok=True)
    importance_path = os.path.join(out_dir, f"gene_importances_{prefix}.csv")
    importance_df.to_csv(importance_path, index=False)
    pred_path = os.path.join(out_dir, f"predictions_{prefix}.csv")
    all_results.to_csv(pred_path, index=False)
    return importance_df, importance_path, pred_path


def create_visuals_and_report(*args, **kwargs):
    """兼容旧接口的占位函数。"""
    warnings.warn(
        "create_visuals_and_report 已被新的分阶段 DeepQA pipeline 取代，"
        "请使用阶段性输出结果（HPO / 5fold_cv / DataSet_loo / group_loo / group_best_model）。"
    )


# ===================================================================
#                       4. 公共绘图函数
# ===================================================================


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
    sns.regplot(x="actual_age", y="predicted_age", data=val_df, ax=ax2, color="green", scatter_kws={"alpha": 0.6})
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
    order = sorted(all_results[config["subgroup_col"]].dropna().unique())
    sns.boxplot(x=config["subgroup_col"], y="age_acceleration", data=all_results, palette="viridis", order=order)
    sns.stripplot(x=config["subgroup_col"], y="age_acceleration", data=all_results, color="black", alpha=0.4, order=order)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ===================================================================
#                       5. 各阶段流程 (DeepQA 版，与 MoE 一致)
# ===================================================================


def stage_hpo_and_save(normal_data, gene_columns, metadata, aging_gene_indices, config, root_dir):
    """阶段 1：Normal 上 train/val 划分，Optuna HPO，保存最优模型与报告。"""
    out_dir = os.path.join(root_dir, "HPO")
    os.makedirs(out_dir, exist_ok=True)
    X_normal = normal_data[gene_columns]
    y_normal = normal_data[config["age_col"]]
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_normal, y_normal,
        test_size=config["test_split_size"],
        random_state=config["seed"],
    )
    print(f"  - HPO 划分 Normal: 训练集 {len(X_train_raw)}, 验证集 {len(X_val_raw)} (test_size={config['test_split_size']})")
    data_splits = {
        "X_train_raw": X_train_raw,
        "y_train": y_train,
        "X_val_raw": X_val_raw,
        "y_val": y_val,
        "X_intervention_raw": X_val_raw.copy(),
        "y_intervention": y_val.copy(),
        "gene_columns": gene_columns,
        "metadata": metadata,
        "aging_gene_indices": aging_gene_indices,
    }
    best_params = {}
    if config.get("optuna", {}).get("enabled", False):
        best_params = optuna_search_hpo(data_splits, config)
    train_cfg = dict(config)
    if best_params:
        train_cfg.update(best_params)
    best_model, scaler_X, scaler_y = train_deepqa_clock(data_splits, train_cfg)

    torch.save(best_model.state_dict(), os.path.join(out_dir, "DeepQA_best_model.pth"))
    joblib.dump(scaler_X, os.path.join(out_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(out_dir, "scaler_y.pkl"))
    if best_params:
        best_path = os.path.join(out_dir, "best_hyperparameters.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)
        print(f"  - 最优超参数已保存到: {best_path}")

    train_df_data = normal_data.loc[y_train.index].copy()
    val_df_data = normal_data.loc[y_val.index].copy()
    all_results = evaluate_on_splits(
        best_model, (scaler_X, scaler_y),
        {"train": train_df_data, "validation": val_df_data},
        gene_columns, metadata, config,
    )
    importance_df = compute_gene_importance(
        best_model, scaler_X, normal_data[gene_columns], gene_columns
    )
    importance_df, _, _ = save_gene_importance_and_results(
        importance_df, all_results, out_dir, "HPO_all_normal"
    )
    train_df = all_results[all_results["data_type"] == "train"]
    val_df = all_results[all_results["data_type"] == "validation"]
    train_mae = mean_absolute_error(train_df["actual_age"], train_df["predicted_age"])
    train_r2 = r2_score(train_df["actual_age"], train_df["predicted_age"])
    val_mae = mean_absolute_error(val_df["actual_age"], val_df["predicted_age"])
    val_r2 = r2_score(val_df["actual_age"], val_df["predicted_age"])
    perf_plot_path = os.path.join(out_dir, "performance_train_val.png")
    plot_performance(
        train_df, val_df,
        {"train_mae": train_mae, "train_r2": train_r2, "val_mae": val_mae, "val_r2": val_r2},
        perf_plot_path,
        title_prefix="DeepQA HPO Best Model",
    )
    report_path = os.path.join(out_dir, "report_hpo.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Aging Clock HPO Report (PyTorch DeepQA) ===\n\n")
        f.write("Model: PyTorch DeepQA Mixture-of-Experts\n")
        if best_params:
            f.write("--- Best Hyperparameters (from Optuna) ---\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
        f.write("--- Performance (Normal train/val split) ---\n")
        f.write(f"Train MAE: {train_mae:.3f} years, R2: {train_r2:.3f}\n")
        f.write(f"Val   MAE: {val_mae:.3f} years, R2: {val_r2:.3f}\n\n")
        f.write(f"Number of genes with non-zero importance: {len(importance_df[importance_df['importance'] > 0])}\n")
        f.write("\nTop 10 genes by importance:\n")
        f.write(importance_df.head(10).to_string(index=False))
    return best_params, best_model, scaler_X, scaler_y


def stage_kfold_cv(data_pkg, best_params, config, root_dir):
    """阶段 2：5-fold KFold 交叉验证（Normal 上外层 CV）。"""
    out_dir = os.path.join(root_dir, "5fold_cv")
    os.makedirs(out_dir, exist_ok=True)
    normal_data = data_pkg["normal_data"]
    intervention_data = data_pkg["intervention_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]
    aging_gene_indices = data_pkg["aging_gene_indices"]

    kf = KFold(n_splits=config["k_folds"], shuffle=True, random_state=config["seed"])
    all_fold_metrics = []
    cv_normal_list = []
    intervention_by_fold_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(normal_data)):
        fold_name = f"fold_{fold_idx + 1}"
        print("\n" + "=" * 50)
        print(f"阶段 2 - 5fold CV (DeepQA): {fold_name} ({fold_idx + 1}/{config['k_folds']})")
        print("=" * 50)
        train_df = normal_data.iloc[train_idx].copy()
        val_df = normal_data.iloc[val_idx].copy()
        print(f"  - 训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}")
        X_train_raw = train_df[gene_columns]
        y_train = train_df[config["age_col"]]
        X_val_raw = val_df[gene_columns]
        y_val = val_df[config["age_col"]]
        data_splits = {
            "X_train_raw": X_train_raw, "y_train": y_train,
            "X_val_raw": X_val_raw, "y_val": y_val,
            "X_intervention_raw": intervention_data[gene_columns],
            "y_intervention": intervention_data[config["age_col"]],
            "gene_columns": gene_columns, "metadata": metadata,
            "aging_gene_indices": aging_gene_indices,
        }
        train_cfg = dict(config)
        if best_params:
            train_cfg.update(best_params)
        model, scaler_X, scaler_y = train_deepqa_clock(data_splits, train_cfg)
        fold_results = evaluate_on_splits(
            model, (scaler_X, scaler_y),
            {"train": train_df, "validation": val_df, "intervention": intervention_data},
            gene_columns, metadata, config,
        )
        importance_df = compute_gene_importance(model, scaler_X, train_df[gene_columns], gene_columns)
        fold_dir = os.path.join(out_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(fold_dir, f"DeepQA_model_{fold_name}.pth"))
        joblib.dump(scaler_X, os.path.join(fold_dir, f"scaler_X_{fold_name}.pkl"))
        joblib.dump(scaler_y, os.path.join(fold_dir, f"scaler_Y_{fold_name}.pkl"))
        importance_df, _, _ = save_gene_importance_and_results(
            importance_df, fold_results, fold_dir, fold_name
        )
        train_res = fold_results[fold_results["data_type"] == "train"]
        val_res = fold_results[fold_results["data_type"] == "validation"]
        interv_res = fold_results[fold_results["data_type"] == "intervention"]
        metrics = {
            "fold": fold_name,
            "train_mae": mean_absolute_error(train_res["actual_age"], train_res["predicted_age"]),
            "train_r2": r2_score(train_res["actual_age"], train_res["predicted_age"]),
            "val_mae": mean_absolute_error(val_res["actual_age"], val_res["predicted_age"]),
            "val_r2": r2_score(val_res["actual_age"], val_res["predicted_age"]),
            "n_train": len(train_res), "n_val": len(val_res),
        }
        all_fold_metrics.append(metrics)
        perf_plot_path = os.path.join(fold_dir, f"performance_{fold_name}.png")
        plot_performance(train_res, val_res, metrics, perf_plot_path, title_prefix=fold_name)
        accel_plot_path = os.path.join(fold_dir, f"age_acceleration_{fold_name}.png")
        plot_age_acceleration(fold_results, config, accel_plot_path, title=f"Age Acceleration (DeepQA {fold_name})")
        val_res_with_fold = val_res.copy()
        val_res_with_fold["fold"] = fold_name
        cv_normal_list.append(val_res_with_fold)
        if not interv_res.empty:
            group_col = config["subgroup_col"]
            if group_col in val_res.columns and group_col in interv_res.columns:
                val_groups = set(val_res[group_col].unique())
                intervention_res_filtered = interv_res[interv_res[group_col].isin(val_groups)].copy()
            else:
                intervention_res_filtered = interv_res.copy()
            if not intervention_res_filtered.empty:
                intervention_res_filtered = intervention_res_filtered.copy()
                intervention_res_filtered["fold"] = fold_name
                intervention_by_fold_list.append(intervention_res_filtered)
        with open(os.path.join(fold_dir, f"report_{fold_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== Report for {fold_name} (DeepQA 5-fold CV) ===\n\n")
            if best_params:
                f.write("--- Hyperparameters used ---\n")
                for k, v in best_params.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")
            f.write(f"Number of genes with non-zero importance: {len(importance_df[importance_df['importance'] > 0])}\n\n")
            f.write(f"Train MAE = {metrics['train_mae']:.3f}, R2 = {metrics['train_r2']:.3f}\n")
            f.write(f"Val   MAE = {metrics['val_mae']:.3f}, R2 = {metrics['val_r2']:.3f}\n")

    cv_normal_all = pd.concat(cv_normal_list, ignore_index=True)
    intervention_all = pd.concat(intervention_by_fold_list, ignore_index=True) if intervention_by_fold_list else pd.DataFrame()
    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(os.path.join(out_dir, "summary_metrics_by_fold.csv"), index=False)
    overall_mae = mean_absolute_error(cv_normal_all["actual_age"], cv_normal_all["predicted_age"])
    overall_r2 = r2_score(cv_normal_all["actual_age"], cv_normal_all["predicted_age"])
    plt.figure(figsize=(8, 7))
    sns.regplot(x="actual_age", y="predicted_age", data=cv_normal_all, scatter_kws={"alpha": 0.7})
    plt.title(f"Overall {config['k_folds']}-Fold CV (DeepQA, Normal)\nMAE={overall_mae:.2f}, R²={overall_r2:.2f}", fontsize=14)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age (Out-of-sample)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_performance_cv_normal.png"), dpi=300)
    plt.close()
    combined = pd.concat([cv_normal_all, intervention_all], ignore_index=True) if not intervention_all.empty else cv_normal_all.copy()
    combined.to_csv(os.path.join(out_dir, "summary_all_predictions_cv.csv"), index=False)
    plot_age_acceleration(combined, config, os.path.join(out_dir, "summary_age_acceleration.png"), title="Overall Age Acceleration (DeepQA 5-fold CV)")
    with open(os.path.join(out_dir, "summary_report_5fold_cv.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== Final 5-Fold CV Summary (DeepQA, {config['k_folds']} folds) ===\n\n")
        f.write("Overall Normal (out-of-sample) performance:\n")
        f.write(f"  - MAE: {overall_mae:.3f} years\n")
        f.write(f"  - R2 : {overall_r2:.3f}\n\n")
        f.write("--- Metrics by fold ---\n")
        f.write(metrics_df.to_string(index=False))


def stage_dataset_loo(data_pkg, best_params, config, root_dir):
    """阶段 3：按 batch 进行 Leave-One-Batch-Out CV。"""
    out_dir = os.path.join(root_dir, "DataSet_loo")
    os.makedirs(out_dir, exist_ok=True)
    normal_data = data_pkg["normal_data"]
    intervention_data = data_pkg["intervention_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]
    aging_gene_indices = data_pkg["aging_gene_indices"]
    if config["batch_col"] not in normal_data.columns:
        raise ValueError(f"在 normal_data 中找不到 batch 列 '{config['batch_col']}'")
    unique_batches = sorted(normal_data[config["batch_col"]].unique())
    print("\n步骤 3: DataSet-LOO (DeepQA, 按 batch leave-one-batch-out)...")
    print(f"  - 共发现 {len(unique_batches)} 个 batch: {unique_batches}")
    all_fold_metrics = []
    cv_normal_list = []
    intervention_by_fold_list = []
    for idx, val_batch in enumerate(unique_batches):
        fold_name = f"fold_{idx + 1}_val_{val_batch}"
        print("\n" + "=" * 50)
        print(f"DataSet-LOO (DeepQA): {fold_name}")
        print("=" * 50)
        train_df = normal_data[normal_data[config["batch_col"]] != val_batch].copy()
        val_df = normal_data[normal_data[config["batch_col"]] == val_batch].copy()
        print(f"  - 训练集大小: {len(train_df)}, 验证集(batch={val_batch})大小: {len(val_df)}")
        X_train_raw = train_df[gene_columns]
        y_train = train_df[config["age_col"]]
        X_val_raw = val_df[gene_columns]
        y_val = val_df[config["age_col"]]
        data_splits = {
            "X_train_raw": X_train_raw, "y_train": y_train,
            "X_val_raw": X_val_raw, "y_val": y_val,
            "X_intervention_raw": intervention_data[gene_columns],
            "y_intervention": intervention_data[config["age_col"]],
            "gene_columns": gene_columns, "metadata": metadata,
            "aging_gene_indices": aging_gene_indices,
        }
        train_cfg = dict(config)
        if best_params:
            train_cfg.update(best_params)
        model, scaler_X, scaler_y = train_deepqa_clock(data_splits, train_cfg)
        fold_results = evaluate_on_splits(
            model, (scaler_X, scaler_y),
            {"train": train_df, "validation": val_df, "intervention": intervention_data},
            gene_columns, metadata, config,
        )
        importance_df = compute_gene_importance(model, scaler_X, train_df[gene_columns], gene_columns)
        fold_dir = os.path.join(out_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(fold_dir, f"DeepQA_model_{fold_name}.pth"))
        joblib.dump(scaler_X, os.path.join(fold_dir, f"scaler_X_{fold_name}.pkl"))
        joblib.dump(scaler_y, os.path.join(fold_dir, f"scaler_Y_{fold_name}.pkl"))
        importance_df, _, _ = save_gene_importance_and_results(importance_df, fold_results, fold_dir, fold_name)
        train_res = fold_results[fold_results["data_type"] == "train"]
        val_res = fold_results[fold_results["data_type"] == "validation"]
        interv_res = fold_results[fold_results["data_type"] == "intervention"]
        val_res_filtered = val_res[val_res[config["batch_col"]] == val_batch].copy()
        metrics = {
            "fold": fold_name, "val_batch": val_batch,
            "train_mae": mean_absolute_error(train_res["actual_age"], train_res["predicted_age"]),
            "train_r2": r2_score(train_res["actual_age"], train_res["predicted_age"]),
            "val_mae": mean_absolute_error(val_res_filtered["actual_age"], val_res_filtered["predicted_age"]),
            "val_r2": r2_score(val_res_filtered["actual_age"], val_res_filtered["predicted_age"]),
            "n_train": len(train_res), "n_val": len(val_res_filtered),
        }
        all_fold_metrics.append(metrics)
        plot_performance(train_res, val_res_filtered, metrics, os.path.join(fold_dir, f"performance_{fold_name}.png"), title_prefix=fold_name)
        plot_age_acceleration(fold_results, config, os.path.join(fold_dir, f"age_acceleration_{fold_name}.png"), title=f"Age Acceleration (DeepQA {fold_name})")
        with open(os.path.join(fold_dir, f"report_{fold_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== Report for {fold_name} (DeepQA DataSet-LOO) ===\n\n")
            f.write(f"Validation batch: {val_batch}\n")
            if best_params:
                for k, v in best_params.items():
                    f.write(f"{k}: {v}\n")
            f.write(f"Number of genes with non-zero importance: {len(importance_df[importance_df['importance'] > 0])}\n\n")
            f.write(f"Train MAE = {metrics['train_mae']:.3f}, R2 = {metrics['train_r2']:.3f}\n")
            f.write(f"Val   MAE = {metrics['val_mae']:.3f}, R2 = {metrics['val_r2']:.3f}\n")
        val_res_with_fold = val_res_filtered.copy()
        val_res_with_fold["fold"] = fold_name
        cv_normal_list.append(val_res_with_fold)
        if not interv_res.empty and config["batch_col"] in interv_res.columns:
            intervention_res_filtered = interv_res[interv_res[config["batch_col"]] == val_batch].copy()
            if not intervention_res_filtered.empty:
                intervention_res_filtered = intervention_res_filtered.copy()
                intervention_res_filtered["fold"] = fold_name
                intervention_by_fold_list.append(intervention_res_filtered)
    cv_normal_all = pd.concat(cv_normal_list, ignore_index=True)
    intervention_all = pd.concat(intervention_by_fold_list, ignore_index=True) if intervention_by_fold_list else pd.DataFrame()
    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(os.path.join(out_dir, "summary_metrics_by_batch.csv"), index=False)
    overall_mae = mean_absolute_error(cv_normal_all["actual_age"], cv_normal_all["predicted_age"])
    overall_r2 = r2_score(cv_normal_all["actual_age"], cv_normal_all["predicted_age"])
    plt.figure(figsize=(8, 7))
    sns.regplot(x="actual_age", y="predicted_age", data=cv_normal_all, scatter_kws={"alpha": 0.7})
    plt.title("Overall DataSet-LOO Performance (DeepQA, Normal, leave-one-batch-out)\n" f"MAE={overall_mae:.2f}, R²={overall_r2:.2f}", fontsize=14)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age (Out-of-sample)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_performance_cv_normal.png"), dpi=300)
    plt.close()
    combined = pd.concat([cv_normal_all, intervention_all], ignore_index=True) if not intervention_all.empty else cv_normal_all.copy()
    combined.to_csv(os.path.join(out_dir, "summary_all_predictions_cv.csv"), index=False)
    plot_age_acceleration(combined, config, os.path.join(out_dir, "summary_age_acceleration.png"), title="Overall Age Acceleration (DeepQA DataSet-LOO)")
    with open(os.path.join(out_dir, "summary_report_dataset_loo.txt"), "w", encoding="utf-8") as f:
        f.write("=== Final DataSet-LOO Summary (DeepQA Leave-One-Batch-Out) ===\n\n")
        f.write("Overall Normal (out-of-sample) performance:\n")
        f.write(f"  - MAE: {overall_mae:.3f} years\n")
        f.write(f"  - R2 : {overall_r2:.3f}\n\n")
        f.write("--- Metrics by batch ---\n")
        f.write(metrics_df.to_string(index=False))


def stage_group_loo(data_pkg, best_params, config, root_dir):
    """阶段 4：按 subgroup 进行 Leave-One-Group-Out CV。"""
    out_dir = os.path.join(root_dir, "group_loo")
    os.makedirs(out_dir, exist_ok=True)
    normal_data = data_pkg["normal_data"]
    intervention_data = data_pkg["intervention_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]
    aging_gene_indices = data_pkg["aging_gene_indices"]
    group_col = config["subgroup_col"]
    if group_col not in normal_data.columns:
        raise ValueError(f"在 normal_data 中找不到 group 列 '{group_col}'")
    unique_groups = sorted(normal_data[group_col].unique())
    print("\n步骤 4: Group-LOO (DeepQA, 按 subgroup leave-one-group-out)...")
    print(f"  - 使用列: {group_col}")
    print(f"  - 共发现 {len(unique_groups)} 个 group: {unique_groups}")
    groups = normal_data[group_col].values
    logo = LeaveOneGroupOut()
    all_fold_metrics = []
    cv_normal_list = []
    intervention_by_fold_list = []
    for idx, (train_idx, val_idx) in enumerate(logo.split(normal_data, normal_data[config["age_col"]], groups)):
        val_group = normal_data.iloc[val_idx][group_col].iloc[0]
        fold_name = f"fold_{idx + 1}_val_{val_group}"
        print("\n" + "=" * 50)
        print(f"Group-LOO (DeepQA): {fold_name}")
        print("=" * 50)
        train_df = normal_data.iloc[train_idx].copy()
        val_df = normal_data.iloc[val_idx].copy()
        print(f"  - 训练集大小: {len(train_df)}, 验证集({group_col}={val_group})大小: {len(val_df)}")
        X_train_raw = train_df[gene_columns]
        y_train = train_df[config["age_col"]]
        X_val_raw = val_df[gene_columns]
        y_val = val_df[config["age_col"]]
        data_splits = {
            "X_train_raw": X_train_raw, "y_train": y_train,
            "X_val_raw": X_val_raw, "y_val": y_val,
            "X_intervention_raw": intervention_data[gene_columns],
            "y_intervention": intervention_data[config["age_col"]],
            "gene_columns": gene_columns, "metadata": metadata,
            "aging_gene_indices": aging_gene_indices,
        }
        train_cfg = dict(config)
        if best_params:
            train_cfg.update(best_params)
        model, scaler_X, scaler_y = train_deepqa_clock(data_splits, train_cfg)
        fold_results = evaluate_on_splits(
            model, (scaler_X, scaler_y),
            {"train": train_df, "validation": val_df, "intervention": intervention_data},
            gene_columns, metadata, config,
        )
        importance_df = compute_gene_importance(model, scaler_X, train_df[gene_columns], gene_columns)
        fold_dir = os.path.join(out_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(fold_dir, f"DeepQA_model_{fold_name}.pth"))
        joblib.dump(scaler_X, os.path.join(fold_dir, f"scaler_X_{fold_name}.pkl"))
        joblib.dump(scaler_y, os.path.join(fold_dir, f"scaler_Y_{fold_name}.pkl"))
        importance_df, _, _ = save_gene_importance_and_results(importance_df, fold_results, fold_dir, fold_name)
        train_res = fold_results[fold_results["data_type"] == "train"]
        val_res = fold_results[fold_results["data_type"] == "validation"]
        interv_res = fold_results[fold_results["data_type"] == "intervention"]
        val_res_filtered = val_res[val_res[group_col] == val_group].copy()
        metrics = {
            "fold": fold_name, "val_group": val_group,
            "train_mae": mean_absolute_error(train_res["actual_age"], train_res["predicted_age"]),
            "train_r2": r2_score(train_res["actual_age"], train_res["predicted_age"]),
            "val_mae": mean_absolute_error(val_res_filtered["actual_age"], val_res_filtered["predicted_age"]),
            "val_r2": r2_score(val_res_filtered["actual_age"], val_res_filtered["predicted_age"]),
            "n_train": len(train_res), "n_val": len(val_res_filtered),
        }
        all_fold_metrics.append(metrics)
        plot_performance(train_res, val_res_filtered, metrics, os.path.join(fold_dir, f"performance_{fold_name}.png"), title_prefix=f"DeepQA {fold_name}")
        plot_age_acceleration(fold_results, config, os.path.join(fold_dir, f"age_acceleration_{fold_name}.png"), title=f"Age Acceleration (DeepQA {fold_name})")
        with open(os.path.join(fold_dir, f"report_{fold_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== Report for {fold_name} (DeepQA Group-LOO) ===\n\n")
            f.write(f"Validation group ({group_col}): {val_group}\n")
            if best_params:
                for k, v in best_params.items():
                    f.write(f"{k}: {v}\n")
            f.write(f"Number of genes with non-zero importance: {len(importance_df[importance_df['importance'] > 0])}\n\n")
            f.write(f"Train MAE = {metrics['train_mae']:.3f}, R2 = {metrics['train_r2']:.3f}\n")
            f.write(f"Val   MAE = {metrics['val_mae']:.3f}, R2 = {metrics['val_r2']:.3f}\n")
        val_res_with_fold = val_res_filtered.copy()
        val_res_with_fold["fold"] = fold_name
        cv_normal_list.append(val_res_with_fold)
        if not interv_res.empty and group_col in interv_res.columns:
            intervention_res_filtered = interv_res[interv_res[group_col] == val_group].copy()
            if not intervention_res_filtered.empty:
                intervention_res_filtered = intervention_res_filtered.copy()
                intervention_res_filtered["fold"] = fold_name
                intervention_by_fold_list.append(intervention_res_filtered)
    cv_normal_all = pd.concat(cv_normal_list, ignore_index=True)
    intervention_all = pd.concat(intervention_by_fold_list, ignore_index=True) if intervention_by_fold_list else pd.DataFrame()
    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(os.path.join(out_dir, "summary_metrics_by_group.csv"), index=False)
    overall_mae = mean_absolute_error(cv_normal_all["actual_age"], cv_normal_all["predicted_age"])
    overall_r2 = r2_score(cv_normal_all["actual_age"], cv_normal_all["predicted_age"])
    plt.figure(figsize=(8, 7))
    sns.regplot(x="actual_age", y="predicted_age", data=cv_normal_all, scatter_kws={"alpha": 0.7})
    plt.title("Overall Group-LOO Performance (DeepQA, Normal, leave-one-group-out)\n" f"MAE={overall_mae:.2f}, R²={overall_r2:.2f}", fontsize=14)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age (Out-of-sample)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_performance_cv_normal.png"), dpi=300)
    plt.close()
    combined = pd.concat([cv_normal_all, intervention_all], ignore_index=True) if not intervention_all.empty else cv_normal_all.copy()
    combined.to_csv(os.path.join(out_dir, "summary_all_predictions_cv.csv"), index=False)
    plot_age_acceleration(combined, config, os.path.join(out_dir, "summary_age_acceleration.png"), title="Overall Age Acceleration (DeepQA Group-LOO)")
    with open(os.path.join(out_dir, "summary_report_group_loo.txt"), "w", encoding="utf-8") as f:
        f.write("=== Final Group-LOO Summary (DeepQA Leave-One-Group-Out by subgroup) ===\n\n")
        f.write("Overall Normal (out-of-sample) performance:\n")
        f.write(f"  - MAE: {overall_mae:.3f} years\n")
        f.write(f"  - R2 : {overall_r2:.3f}\n\n")
        f.write("--- Metrics by group ---\n")
        f.write(metrics_df.to_string(index=False))


def stage_group_prediction_with_best_model(data_pkg, best_model, best_scaler_X, best_scaler_y, config, root_dir):
    """阶段 5：用 HPO 得到的最优 DeepQA 模型对所有样本预测，并按 group 汇总。"""
    out_dir = os.path.join(root_dir, "group_best_model")
    os.makedirs(out_dir, exist_ok=True)
    all_data = data_pkg["all_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]
    normal_data = data_pkg["normal_data"]
    print("\n步骤 5: 使用 HPO 得到的最优模型，对所有样本预测，并按 group 汇总...")
    X_all = all_data[gene_columns]
    y_all = all_data[config["age_col"]]
    device = next(best_model.parameters()).device
    X_all_scaled = best_scaler_X.transform(X_all)
    X_all_t = torch.tensor(X_all_scaled, dtype=torch.float32, device=device)
    best_model.eval()
    with torch.no_grad():
        y_pred_scaled, _, _ = best_model(X_all_t)
    y_pred = best_scaler_y.inverse_transform(y_pred_scaled.cpu().numpy()).flatten()
    res = pd.DataFrame({"actual_age": y_all, "predicted_age": y_pred}, index=all_data.index)
    res = pd.merge(res.reset_index().rename(columns={"index": "sample_id"}), metadata, on="sample_id", how="left")
    res["age_acceleration"] = res["predicted_age"] - res["actual_age"]
    importance_df = compute_gene_importance(best_model, best_scaler_X, normal_data[gene_columns], gene_columns)
    importance_path = os.path.join(out_dir, "gene_importances_best_model.csv")
    importance_df.to_csv(importance_path, index=False)
    all_pred_path = os.path.join(out_dir, "all_samples_predictions_best_DeepQA_model.csv")
    res.to_csv(all_pred_path, index=False)
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
    x = np.arange(len(group_summary))
    width = 0.35
    plt.figure(figsize=(12, 7))
    plt.bar(x - width / 2, group_summary["mean_actual_age"], width, label="Actual")
    plt.bar(x + width / 2, group_summary["mean_predicted_age"], width, label="Predicted")
    plt.xticks(x, group_summary[group_col], rotation=45, ha="right")
    plt.ylabel("Age (years)")
    plt.title("Mean Actual & Predicted Age per Group (Best DeepQA Model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "group_mean_age_barplot.png"), dpi=300)
    plt.close()
    plot_age_acceleration(
        res, config,
        os.path.join(out_dir, "age_acceleration_by_group_best_model.png"),
        title="Age Acceleration by Group (Best DeepQA Model on All Data)",
    )
    with open(os.path.join(out_dir, "report_group_best_model.txt"), "w", encoding="utf-8") as f:
        f.write("=== Group-level Predictions with Best DeepQA Model ===\n\n")
        f.write("该模型来自 HPO 阶段，在所有 Normal 样本上训练得到，然后用于全体样本的预测。\n\n")
        f.write(f"Number of genes with non-zero importance: {len(importance_df[importance_df['importance'] > 0])}\n")
        f.write("\nTop 10 genes by importance:\n")
        f.write(importance_df.head(10).to_string(index=False))
        f.write("\n\n--- Group-wise summary ---\n")
        f.write(group_summary.to_string(index=False))


# ===================================================================
#                       6. 主入口
# ===================================================================


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    print(f"✅ 所有随机种子已设置为: {CONFIG['seed']}")
    root_dir = resolve_path(CONFIG["output_root_dir"])
    os.makedirs(root_dir, exist_ok=True)
    print(f"输出根目录: {root_dir}")
    data_pkg = load_full_data(CONFIG)

    best_params, best_model, best_scaler_X, best_scaler_Y = stage_hpo_and_save(
        data_pkg["normal_data"],
        data_pkg["gene_columns"],
        data_pkg["metadata"],
        data_pkg["aging_gene_indices"],
        CONFIG,
        root_dir,
    )
    stage_kfold_cv(data_pkg, best_params, CONFIG, root_dir)
    stage_dataset_loo(data_pkg, best_params, CONFIG, root_dir)
    stage_group_loo(data_pkg, best_params, CONFIG, root_dir)
    stage_group_prediction_with_best_model(
        data_pkg, best_model, best_scaler_X, best_scaler_Y, CONFIG, root_dir
    )
    print("\n✅ 全部流程完成！结果请查看输出根目录下的各个子文件夹。")
