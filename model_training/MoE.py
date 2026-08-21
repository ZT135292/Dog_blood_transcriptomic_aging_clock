import os
import json
import random
import shutil
import subprocess
import tempfile
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

CONFIG = {
    "metadata_file": "total_metadata.txt",
    "raw_count_file": "total_count_sort_ortholog.txt",
    "aging_genes_file": "aging_gene_list.csv",
    # 使用新目录，避免无泄露重跑结果与旧版结果混合。
    "output_root_dir": "aging_clock_results_MoE_pipeline_foldwise_TMM_logCPM",

    "foldwise_tmm_logcpm": {
        "enabled": True,
        "rscript": "Rscript",
        "prior_count": 1,
        "filter_low_expression": True,
    },

    "combat": {
        "enabled": False,
        "covariate_continuous": ["age"],
        "covariate_categorical": [],
        "min_samples_per_cat_level": 2,
        "min_samples_per_batch": 2,
    },

    "sample_id_col": "sample_id",
    "group_col": "condition",
    "subgroup_col": "group",
    "age_col": "Chronological_Age",
    "batch_col": "batch",

    "normal_group_label": "Normal",
    "test_split_size": 0.2,
    "seed": 66,

    "outlier_detection": {
        "enabled": False,
        "method": "iqr",
        "iqr_factor": 1.5,
        "zscore_threshold": 3.0,
        "check_age": False,
        "age_iqr_factor": 3.0,
    },

    "k_folds": 5,
    # 每个外层训练折内部划分出的模型选择集；外层验证折不参与 HPO/早停。
    "nested_validation_size": 0.2,

    "num_experts": 2,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "batch_size": 32,
    "lambda_pred_loss": 30.0,
    "gamma_ahk_loss": 80.0,

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

    "optuna": {
        "enabled": True,
        "trials": 25,
        "epochs": 60
    }
}


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


def is_foldwise_tmm_enabled(config):
    return config.get("foldwise_tmm_logcpm", {}).get("enabled", False)


def run_edger_tmm_logcpm(counts_df, out_dir, prefix, config, keep_genes=None):
    """使用 edgeR 对一个 counts 矩阵执行 TMM + logCPM。"""
    tmm_cfg = config.get("foldwise_tmm_logcpm", {})
    rscript = tmm_cfg.get("rscript", "Rscript")
    prior_count = tmm_cfg.get("prior_count", 1)
    filter_low_expression = bool(tmm_cfg.get("filter_low_expression", False))

    if shutil.which(rscript) is None:
        raise RuntimeError(
            f"未找到 Rscript: {rscript}. 请确认 R 已安装且 Rscript 在 PATH 中，"
            "或在 CONFIG['foldwise_tmm_logcpm']['rscript'] 中填写 Rscript.exe 的绝对路径。"
        )

    counts_df = counts_df.copy()
    counts_df.index = counts_df.index.astype(str)
    counts_df.columns = counts_df.columns.astype(str)

    if keep_genes is not None:
        keep_genes = [gene for gene in keep_genes if gene in counts_df.index]
        counts_df = counts_df.loc[keep_genes]

    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
        counts_path = os.path.join(tmpdir, f"{prefix}_counts.tsv")
        out_path = os.path.join(tmpdir, f"{prefix}_logcpm.tsv")
        genes_path = os.path.join(tmpdir, f"{prefix}_genes.tsv")
        r_path = os.path.join(tmpdir, f"{prefix}_run_edger.R")

        counts_df.to_csv(counts_path, sep="\t")

        r_code = f"""
suppressPackageStartupMessages(library(edgeR))
counts <- read.table({json.dumps(counts_path)}, header=TRUE, row.names=1, sep="\\t", check.names=FALSE)
counts <- as.matrix(counts)
y <- DGEList(counts=counts)
if ({str(filter_low_expression).upper()}) {{
  keep <- filterByExpr(y)
  y <- y[keep, , keep.lib.sizes=FALSE]
}}
if (ncol(y) >= 2) {{
  y <- calcNormFactors(y, method="TMM")
}} else {{
  y$samples$norm.factors <- 1
}}
logcpm <- cpm(y, normalized.lib.sizes=TRUE, log=TRUE, prior.count={float(prior_count)})
write.table(logcpm, file={json.dumps(out_path)}, sep="\\t", quote=FALSE, row.names=TRUE, col.names=NA)
write.table(data.frame(gene=rownames(logcpm)), file={json.dumps(genes_path)}, sep="\\t", quote=FALSE, row.names=FALSE)
"""
        with open(r_path, "w", encoding="utf-8") as f:
            f.write(r_code)

        completed = subprocess.run([rscript, r_path], capture_output=True, text=True, cwd=out_dir)
        if completed.returncode != 0:
            raise RuntimeError(
                "edgeR TMM/logCPM 运行失败。\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        logcpm = pd.read_csv(out_path, sep="\t", index_col=0)
        kept = pd.read_csv(genes_path, sep="\t")["gene"].astype(str).tolist()

    return logcpm, kept


def run_edger_logcpm_with_train_ref(
    counts_df,
    train_ref_counts_df,
    out_dir,
    prefix,
    config,
    keep_genes=None,
):
    """
    沿用训练集 TMM norm factor，对验证/测试 counts 计算 logCPM。

    按 edgeR 推荐：训练集 counts 与新样本 counts 合并后，以训练集第一个样本为
    reference 估计 TMM，仅输出新样本列的 logCPM。
    """
    tmm_cfg = config.get("foldwise_tmm_logcpm", {})
    rscript = tmm_cfg.get("rscript", "Rscript")
    prior_count = tmm_cfg.get("prior_count", 1)

    if shutil.which(rscript) is None:
        raise RuntimeError(
            f"未找到 Rscript: {rscript}. 请确认 R 已安装且 Rscript 在 PATH 中，"
            "或在 CONFIG['foldwise_tmm_logcpm']['rscript'] 中填写 Rscript.exe 的绝对路径。"
        )

    counts_df = counts_df.copy()
    train_ref_counts_df = train_ref_counts_df.copy()
    counts_df.index = counts_df.index.astype(str)
    counts_df.columns = counts_df.columns.astype(str)
    train_ref_counts_df.index = train_ref_counts_df.index.astype(str)
    train_ref_counts_df.columns = train_ref_counts_df.columns.astype(str)

    if keep_genes is not None:
        keep_genes = [
            gene for gene in keep_genes
            if gene in counts_df.index and gene in train_ref_counts_df.index
        ]
        counts_df = counts_df.loc[keep_genes]
        train_ref_counts_df = train_ref_counts_df.loc[keep_genes]

    common_genes = counts_df.index.intersection(train_ref_counts_df.index)
    counts_df = counts_df.loc[common_genes]
    train_ref_counts_df = train_ref_counts_df.loc[common_genes]

    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
        train_counts_path = os.path.join(tmpdir, f"{prefix}_train_ref_counts.tsv")
        val_counts_path = os.path.join(tmpdir, f"{prefix}_counts.tsv")
        out_path = os.path.join(tmpdir, f"{prefix}_logcpm.tsv")
        r_path = os.path.join(tmpdir, f"{prefix}_run_edger_train_ref.R")

        train_ref_counts_df.to_csv(train_counts_path, sep="\t")
        counts_df.to_csv(val_counts_path, sep="\t")

        r_code = f"""
suppressPackageStartupMessages(library(edgeR))
train_counts <- read.table({json.dumps(train_counts_path)}, header=TRUE, row.names=1, sep="\\t", check.names=FALSE)
val_counts <- read.table({json.dumps(val_counts_path)}, header=TRUE, row.names=1, sep="\\t", check.names=FALSE)
train_counts <- as.matrix(train_counts)
val_counts <- as.matrix(val_counts)
combined <- cbind(train_counts, val_counts)
y <- DGEList(counts=combined)
if (ncol(y) >= 2) {{
  y <- calcNormFactors(y, method="TMM", ref=1)
}} else {{
  y$samples$norm.factors <- 1
}}
logcpm_all <- cpm(y, normalized.lib.sizes=TRUE, log=TRUE, prior.count={float(prior_count)})
n_train <- ncol(train_counts)
logcpm <- logcpm_all[, (n_train + 1):ncol(logcpm_all), drop=FALSE]
colnames(logcpm) <- colnames(val_counts)
write.table(logcpm, file={json.dumps(out_path)}, sep="\\t", quote=FALSE, row.names=TRUE, col.names=NA)
"""
        with open(r_path, "w", encoding="utf-8") as f:
            f.write(r_code)

        completed = subprocess.run(
            [rscript, r_path],
            capture_output=True,
            text=True,
            cwd=out_dir,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "edgeR logCPM（沿用训练 norm factor）运行失败。\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        logcpm = pd.read_csv(out_path, sep="\t", index_col=0)
        logcpm.columns = logcpm.columns.astype(str)

    return logcpm


def attach_expression(metadata_df, logcpm_df):
    """将 gene x sample 的 logCPM 矩阵转为 sample x gene，并与对应 metadata 合并。"""
    expr_t = logcpm_df.T
    expr_t.index = expr_t.index.astype(str)
    merged = pd.merge(
        metadata_df.copy(),
        expr_t,
        left_index=True,
        right_index=True,
        how="inner",
    )
    return merged


def normalize_fold_splits_tmm_logcpm(train_df, splits_dict, gene_columns, config, out_dir, prefix):
    """在一个 fold 内执行 TMM + logCPM；验证/干预集沿用训练 norm factor。"""
    if not is_foldwise_tmm_enabled(config):
        return train_df, splits_dict, gene_columns

    meta_cols = [col for col in train_df.columns if col not in gene_columns]

    train_counts = train_df[gene_columns].T
    train_logcpm, kept_genes = run_edger_tmm_logcpm(
        train_counts, out_dir=out_dir, prefix=f"{prefix}_train",
        config=config, keep_genes=None,
    )
    train_norm = attach_expression(train_df[meta_cols], train_logcpm)
    train_counts_kept = train_counts.loc[
        [gene for gene in kept_genes if gene in train_counts.index]
    ]
    train_ids = set(train_df.index.astype(str))

    norm_splits = {}
    for split_name, split_df in splits_dict.items():
        if split_df is None or len(split_df) == 0:
            norm_splits[split_name] = split_df
            continue
        if split_name == "train" and split_df.index.equals(train_df.index):
            norm_splits[split_name] = train_norm
            continue

        split_counts = split_df[gene_columns].T
        non_train_cols = [col for col in split_counts.columns if str(col) not in train_ids]
        train_cols = [col for col in split_counts.columns if str(col) in train_ids]

        logcpm_parts = []
        if train_cols:
            available = [str(col) for col in train_cols if str(col) in train_logcpm.columns]
            if available:
                logcpm_parts.append(train_logcpm[available])
        if non_train_cols:
            non_train_logcpm = run_edger_logcpm_with_train_ref(
                split_counts[non_train_cols],
                train_ref_counts_df=train_counts_kept,
                out_dir=out_dir, prefix=f"{prefix}_{split_name}",
                config=config, keep_genes=kept_genes,
            )
            logcpm_parts.append(non_train_logcpm)

        if not logcpm_parts:
            norm_splits[split_name] = split_df
            continue

        split_logcpm = pd.concat(logcpm_parts, axis=1)
        split_logcpm = split_logcpm.reindex(columns=[str(col) for col in split_counts.columns])
        norm_splits[split_name] = attach_expression(split_df[meta_cols], split_logcpm)

    return train_norm, norm_splits, kept_genes


def is_combat_enabled(config):
    return config.get("combat", {}).get("enabled", False)


def run_combat_on_fold(train_df, splits_dict, gene_columns, config, fold_label=""):
    """在单个 fold 内执行 fold-wise 批次效应校正（Python neuroCombat）。"""
    from neuroCombat import neuroCombat, neuroCombatFromTraining

    tag = f" [{fold_label}]" if fold_label else ""
    batch_col    = config["batch_col"]
    combat_cfg   = config.get("combat", {})
    cont_covs    = [c for c in (combat_cfg.get("covariate_continuous",  []) or []) if c]
    cat_covs     = [c for c in (combat_cfg.get("covariate_categorical",  []) or []) if c]
    min_cat_n    = int(combat_cfg.get("min_samples_per_cat_level", 2))
    min_batch_n  = int(combat_cfg.get("min_samples_per_batch", 2))

    if batch_col not in train_df.columns:
        print(f"  - ComBat{tag}: 训练集中找不到 batch 列 '{batch_col}'，跳过校正。")
        return train_df, splits_dict

    train_batches = train_df[batch_col].astype(str).values
    batch_counts  = pd.Series(train_batches).value_counts()
    thin_batches  = batch_counts[batch_counts < min_batch_n].index.tolist()
    if thin_batches:
        for tb in thin_batches:
            print(f"  - ComBat{tag}: batch '{tb}' 只有 {batch_counts[tb]} 个样本"
                  f"（< {min_batch_n}），该 batch 样本跳过 ComBat，保留 TMM 结果。")

    is_thin        = train_df[batch_col].astype(str).isin(thin_batches)
    train_for_cb   = train_df[~is_thin]
    train_thin     = train_df[is_thin]
    unique_cb_batches = sorted(train_for_cb[batch_col].astype(str).unique())

    if len(unique_cb_batches) < 2:
        print(f"  - ComBat{tag}: 可用于 ComBat 的 batch 数 < 2（{unique_cb_batches}），跳过所有校正。")
        return train_df, splits_dict

    cont_covs = [c for c in cont_covs if c in train_df.columns]
    cat_covs  = [c for c in cat_covs  if c in train_df.columns]

    cat_mapping        = {}
    effective_cat_covs = []
    for col in cat_covs:
        vals   = train_for_cb[col].astype(str)
        counts = vals.value_counts()
        rare   = counts[counts < min_cat_n].index.tolist()
        mapping = {v: ("Other" if v in rare else v) for v in counts.index}
        if rare:
            print(f"  - ComBat{tag}: 协变量 '{col}' 中 {len(rare)} 个水平样本数 < {min_cat_n}，合并为 'Other'")
        mapped = vals.map(mapping)
        if mapped.nunique() < 2:
            print(f"  - ComBat{tag}: 协变量 '{col}' 合并后仅剩 1 个水平，跳过该协变量。")
            continue
        cat_mapping[col] = mapping
        effective_cat_covs.append(col)

    meta_cols = [c for c in train_df.columns if c not in gene_columns]

    def _make_covars_df(df, use_cat, use_cont):
        d = {batch_col: df[batch_col].astype(str).values}
        for col in use_cont:
            num = pd.to_numeric(df[col], errors="coerce")
            d[col] = num.fillna(num.mean()).values
        for col in use_cat:
            mapped = df[col].astype(str).map(cat_mapping[col])
            d[col] = mapped.fillna("Other").values
        return pd.DataFrame(d)

    train_expr = train_for_cb[gene_columns].values.T.astype(float)
    print(f"  - ComBat{tag}: 在训练集上拟合 neuroCombat "
          f"({len(unique_cb_batches)} 个 batch，n={len(train_for_cb)})，"
          f"连续协变量={cont_covs}，分类协变量={effective_cat_covs} ...")

    estimates            = None
    train_corrected_expr = None
    for attempt, (use_cat, use_cont) in enumerate([
        (effective_cat_covs, cont_covs),
        ([],                 cont_covs),
        ([],                 []),
    ]):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                result = neuroCombat(
                    dat=train_expr,
                    covars=_make_covars_df(train_for_cb, use_cat, use_cont),
                    batch_col=batch_col,
                    continuous_cols=use_cont,
                    categorical_cols=use_cat,
                )
            estimates            = result["estimates"]
            train_corrected_expr = result["data"].T
            nan_mask = np.isnan(train_corrected_expr)
            if nan_mask.any():
                train_corrected_expr = np.where(nan_mask, train_for_cb[gene_columns].values, train_corrected_expr)
            if attempt > 0:
                print(f"    -> 降级成功（分类协变量={use_cat}，连续协变量={use_cont}）")
            break
        except Exception as exc:
            if attempt == 2:
                print(f"  - ComBat{tag}: 训练集校正全部尝试失败（{exc}），跳过所有校正。")
                return train_df, splits_dict
            print(f"    -> 尝试 {attempt + 1} 失败（{exc}），降级重试...")

    corrected_cb_df = pd.concat([
        train_for_cb[meta_cols],
        pd.DataFrame(train_corrected_expr, index=train_for_cb.index, columns=gene_columns),
    ], axis=1)
    if not train_thin.empty:
        corrected_train = pd.concat(
            [corrected_cb_df, train_thin[meta_cols + gene_columns]], axis=0
        ).loc[train_df.index]
    else:
        corrected_train = corrected_cb_df
    print(f"    -> 训练集 neuroCombat 校正完成 (n_corrected={len(train_for_cb)}，n_skipped_singleton={len(train_thin)})")

    corrected_splits = {}
    for split_name, split_df in splits_dict.items():
        if split_df is None or len(split_df) == 0:
            corrected_splits[split_name] = split_df
            continue
        if split_name == "train":
            corrected_splits[split_name] = corrected_train
            continue
        if batch_col not in split_df.columns:
            corrected_splits[split_name] = split_df
            continue

        split_batches = split_df[batch_col].astype(str).values
        seen_mask = pd.Series(split_batches).isin(unique_cb_batches).values
        unseen    = sorted(set(split_batches[~seen_mask]))
        if unseen:
            print(f"  - ComBat{tag}: split '{split_name}' 中存在未参与 ComBat 训练的 batch {unseen}，这些样本保留 TMM 结果。")
        if not seen_mask.any():
            corrected_splits[split_name] = split_df
            continue

        seen_df        = split_df.loc[split_df.index[seen_mask]]
        seen_expr      = seen_df[gene_columns].values.T.astype(float)
        seen_batch_arr = split_batches[seen_mask]

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                val_result = neuroCombatFromTraining(
                    dat=seen_expr, batch=seen_batch_arr, estimates=estimates,
                )
            seen_corrected = val_result["data"].T
            nan_mask_val = np.isnan(seen_corrected)
            if nan_mask_val.any():
                seen_corrected = np.where(nan_mask_val, seen_df[gene_columns].values, seen_corrected)
            split_gene = split_df[gene_columns].copy()
            split_gene.loc[seen_df.index] = seen_corrected
            corrected_splits[split_name] = pd.concat([split_df[meta_cols], split_gene], axis=1)
            print(f"    -> split '{split_name}': {seen_mask.sum()} 个样本已用训练集参数校正，"
                  f"{(~seen_mask).sum()} 个样本保留 TMM 结果。")
        except Exception as exc:
            print(f"  - ComBat{tag}: split '{split_name}' 应用训练参数失败（{exc}），保留 TMM 结果。")
            corrected_splits[split_name] = split_df

    return corrected_train, corrected_splits


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


class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=2):
        super(MoE, self).__init__()
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


def compute_aging_gene_indices(gene_columns, aging_gene_pool):
    """根据当前基因列表和衰老基因 pool，计算衰老基因在当前列表中的索引。"""
    return np.array([i for i, gene in enumerate(gene_columns) if gene in aging_gene_pool])


def load_full_data(config):
    """
    读取元数据、raw counts 矩阵和衰老基因列表，并对齐；对 Normal 做异常值去除。
    返回 data_pkg: normal_data, intervention_data, all_data, gene_columns, metadata, aging_gene_indices, aging_gene_pool
    """
    print("步骤 0: 加载和预处理数据...")
    metadata_path = resolve_path(config["metadata_file"])
    expr_path = resolve_path(config["raw_count_file"])
    aging_genes_path = resolve_path(config["aging_genes_file"])
    print(f"  - 读取元数据: {metadata_path}")
    print(f"  - 读取 raw counts: {expr_path}")
    print(f"  - 读取衰老基因列表: {aging_genes_path}")

    if not os.path.exists(expr_path):
        raise FileNotFoundError(
            f"找不到 raw_count_file: {expr_path}. "
            "请将文件放到脚本同目录，或在 CONFIG['raw_count_file'] 中填写绝对路径。"
        )

    metadata = pd.read_csv(metadata_path, sep="\t")
    expr_matrix = pd.read_csv(expr_path, sep="\t", index_col=0)
    expr_matrix.index = expr_matrix.index.astype(str)
    expr_matrix.columns = expr_matrix.columns.astype(str)
    metadata[config["sample_id_col"]] = metadata[config["sample_id_col"]].astype(str)
    aging_genes_df = pd.read_csv(aging_genes_path)

    missing_samples = sorted(set(metadata[config["sample_id_col"]]) - set(expr_matrix.columns))
    if missing_samples:
        raise ValueError(f"表达矩阵缺少 metadata 中的样本: {missing_samples[:10]}")
    expr_matrix = expr_matrix.loc[:, metadata[config["sample_id_col"]]]

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

    aging_gene_pool = set(aging_genes_df.iloc[:, 0].astype(str))
    aging_gene_indices = compute_aging_gene_indices(gene_columns, aging_gene_pool)
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


def train_moe_clock(data_splits, config):
    """
    标准化数据并训练 MoE。

    当 data_splits 含验证集时，可用其早停并返回最佳 epoch；不含验证集时，
    按 config["epochs"] 固定轮数训练。外层 CV 的最终 refit 必须使用后一模式，
    避免外层验证折参与任何训练决策。
    """
    print("\n步骤: 数据标准化并训练  MoE 模型...")
    X_train_raw, y_train = data_splits["X_train_raw"], data_splits["y_train"]
    X_val_raw = data_splits.get("X_val_raw")
    y_val = data_splits.get("y_val")
    has_validation = X_val_raw is not None and y_val is not None and len(y_val) > 0
    aging_gene_indices = data_splits["aging_gene_indices"]

    scaler_X = StandardScaler().fit(X_train_raw)
    scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))
    X_train = scaler_X.transform(X_train_raw)
    y_train_scaled = scaler_y.transform(y_train.values.reshape(-1, 1))

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    effective_batch_size = min(int(config["batch_size"]), len(train_dataset))
    if effective_batch_size < 2:
        raise ValueError("MoE 至少需要 2 个训练样本（BatchNorm 无法处理单样本训练）。")
    # 仅当最后一个 batch 会剩 1 个样本时丢弃，避免 BatchNorm 报错。
    drop_last = len(train_dataset) > effective_batch_size and len(train_dataset) % effective_batch_size == 1
    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch_size, shuffle=True, drop_last=drop_last
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = MoE(X_train.shape[1], config["num_experts"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = None
    if has_validation and config.get("lr_scheduler", {}).get("enabled", False):
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
    best_epoch = int(config["epochs"])
    no_improve_epochs = 0
    es_cfg = config.get("early_stopping", {})
    es_enabled = has_validation and es_cfg.get("enabled", False)
    monitor_interval = max(1, int(es_cfg.get("monitor_interval", 1)))
    epochs_trained = 0

    for epoch in range(config["epochs"]):
        epochs_trained = epoch + 1
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

        if has_validation and (epoch + 1) % monitor_interval == 0:
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
                best_epoch = epoch + 1
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
    training_info = {
        "best_epoch": int(best_epoch),
        "epochs_trained": int(epochs_trained),
        "best_val_mae": float(best_mae) if has_validation else None,
    }
    return model, scaler_X, scaler_y, training_info


def evaluate_on_validation(model, scalers, X_val_raw, y_val):
    """使用验证集计算 MAE 与 R²（以原始年龄单位）。"""
    scaler_X, scaler_y = scalers
    device = next(model.parameters()).device
    X_val_scaled = scaler_X.transform(X_val_raw)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        y_pred_scaled, _, _ = model(X_val_t)
    if was_training:
        model.train()
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
        model, scaler_X, scaler_y, _ = train_moe_clock(data_splits, cfg)
        mae, _ = evaluate_on_validation(
            model, (scaler_X, scaler_y),
            data_splits["X_val_raw"],
            data_splits["y_val"],
        )
        return mae

    n_trials = int(base_config.get("optuna", {}).get("trials", 0))
    if n_trials <= 0:
        return {}
    sampler = optuna.samplers.TPESampler(seed=int(base_config.get("seed", 66)))
    study = optuna.create_study(direction="minimize", sampler=sampler)
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


def tune_and_refit_moe_on_outer_train(
    X_outer_train,
    y_outer_train,
    aging_gene_indices,
    config,
    fold_label,
):
    """
    在单个外层训练折内完成 HPO、早停和最终 refit。

    外层验证数据不会传入本函数：
      1. 外层训练折再拆为 inner-train / inner-validation；
      2. 仅在 inner-validation 上进行 Optuna 和早停；
      3. 记录最佳 epoch；
      4. 用整个外层训练折固定 epoch 重训，供外层验证一次性评估。
    """
    if len(X_outer_train) < 10:
        raise ValueError(
            f"{fold_label}: 外层训练样本仅 {len(X_outer_train)} 个，"
            "不足以稳定拆分 inner validation。"
        )

    inner_test_size = float(config.get("nested_validation_size", config["test_split_size"]))
    inner_indices = np.arange(len(X_outer_train))
    inner_train_idx, inner_val_idx = train_test_split(
        inner_indices,
        test_size=inner_test_size,
        random_state=int(config["seed"]),
        shuffle=True,
    )
    inner_splits = {
        "X_train_raw": X_outer_train.iloc[inner_train_idx],
        "y_train": y_outer_train.iloc[inner_train_idx],
        "X_val_raw": X_outer_train.iloc[inner_val_idx],
        "y_val": y_outer_train.iloc[inner_val_idx],
        "aging_gene_indices": aging_gene_indices,
    }
    print(
        f"  - Nested selection [{fold_label}]: inner-train={len(inner_train_idx)}, "
        f"inner-validation={len(inner_val_idx)}"
    )

    fold_best_params = {}
    if config.get("optuna", {}).get("enabled", False):
        fold_best_params = optuna_search_hpo(inner_splits, config)

    selection_cfg = dict(config)
    selection_cfg.update(fold_best_params)
    _, _, _, selection_info = train_moe_clock(inner_splits, selection_cfg)
    selected_epochs = max(1, int(selection_info["best_epoch"]))

    # 固定 epoch 在整个外层训练折重训；不传外层验证集，也不再早停或调度。
    refit_cfg = dict(config)
    refit_cfg.update(fold_best_params)
    refit_cfg["epochs"] = selected_epochs
    refit_cfg["early_stopping"] = {
        **config.get("early_stopping", {}),
        "enabled": False,
    }
    refit_cfg["lr_scheduler"] = {
        **config.get("lr_scheduler", {}),
        "enabled": False,
    }
    refit_splits = {
        "X_train_raw": X_outer_train,
        "y_train": y_outer_train,
        "aging_gene_indices": aging_gene_indices,
    }
    print(f"  - Nested refit [{fold_label}]: 使用全部外层训练样本，固定训练 {selected_epochs} epochs")
    model, scaler_X, scaler_y, _ = train_moe_clock(refit_splits, refit_cfg)
    return model, scaler_X, scaler_y, fold_best_params, selected_epochs


def evaluate_on_splits(model, scalers, splits_dict, gene_columns, metadata, config):
    """对给定的多个数据子集生成预测并计算 age_acceleration。"""
    scaler_X, scaler_y = scalers
    device = next(model.parameters()).device
    model.eval()
    frames = []
    for name, df in splits_dict.items():
        if df is None or len(df) == 0:
            continue
        X_raw = df[gene_columns].copy()
        if X_raw.isnull().any().any():
            X_raw = X_raw.fillna(X_raw.mean())
        y_true = df[config["age_col"]]
        X_scaled = scaler_X.transform(X_raw)
        if np.isnan(X_scaled).any():
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_scaled, _, _ = model(X_t)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.cpu().numpy()).flatten()
        res = pd.DataFrame(
            {"actual_age": y_true, "Predicted_Age": y_pred, "data_type": name},
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
    all_res["age_acceleration"] = all_res["Predicted_Age"] - all_res["actual_age"]
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
        "create_visuals_and_report 已被新的分阶段  pipeline 取代，"
        "请使用阶段性输出结果（HPO / 5fold_cv / DataSet_loo / group_loo / group_best_model）。"
    )


def plot_performance(train_df, val_df, metrics, out_path, title_prefix=""):
    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    sns.regplot(x="actual_age", y="Predicted_Age", data=train_df, ax=ax1, scatter_kws={"alpha": 0.6})
    ax1.set_title(
        f"{title_prefix} Train (n={len(train_df)})\n"
        f"MAE={metrics['train_mae']:.2f}, R²={metrics['train_r2']:.2f}"
    )
    ax1.set_xlabel("Actual Age")
    ax1.set_ylabel("Predicted Age")
    ax2 = plt.subplot(1, 2, 2)
    sns.regplot(x="actual_age", y="Predicted_Age", data=val_df, ax=ax2, color="green", scatter_kws={"alpha": 0.6})
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
    subgroup = config["subgroup_col"]
    sns.boxplot(
        x=subgroup, y="age_acceleration", data=all_results,
        hue=subgroup, palette="viridis", order=order, legend=False,
    )
    sns.stripplot(x=subgroup, y="age_acceleration", data=all_results, color="black", alpha=0.4, order=order)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def stage_hpo_and_save(normal_data, gene_columns, metadata, aging_gene_indices, config, root_dir,
                       aging_gene_pool=None):
    """
    阶段 1：Normal 上做 TMM+logCPM，再 train/val 划分，Optuna HPO，保存最优模型与报告。
    返回 best_params, best_model, scaler_X, scaler_y, hpo_gene_columns
    """
    out_dir = os.path.join(root_dir, "HPO")
    os.makedirs(out_dir, exist_ok=True)

    if is_foldwise_tmm_enabled(config):
        normal_data, norm_splits, gene_columns = normalize_fold_splits_tmm_logcpm(
            train_df=normal_data,
            splits_dict={"train": normal_data},
            gene_columns=gene_columns,
            config=config,
            out_dir=os.path.join(out_dir, "foldwise_tmm_logcpm"),
            prefix="HPO_all_normal",
        )
        normal_data = norm_splits["train"]
        # 重新计算 aging_gene_indices（基因数量可能因 TMM 过滤而减少）
        if aging_gene_pool is not None:
            aging_gene_indices = compute_aging_gene_indices(gene_columns, aging_gene_pool)
            print(f"  - TMM 过滤后基因数: {len(gene_columns)}，其中衰老基因: {len(aging_gene_indices)}")

    hpo_gene_columns = gene_columns  # 记录 HPO 阶段使用的基因列表

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
    selection_model, selection_scaler_X, selection_scaler_y, selection_info = train_moe_clock(
        data_splits, train_cfg
    )

    if best_params:
        best_path = os.path.join(out_dir, "best_hyperparameters.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(
                {**best_params, "selected_epochs": int(selection_info["best_epoch"])},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  - 最优超参数已保存到: {best_path}")

    train_df_data = normal_data.loc[y_train.index].copy()
    val_df_data = normal_data.loc[y_val.index].copy()
    all_results = evaluate_on_splits(
        selection_model, (selection_scaler_X, selection_scaler_y),
        {"train": train_df_data, "validation": val_df_data},
        gene_columns, metadata, config,
    )

    # HPO/早停完成后，用全部 Normal 样本固定 epoch 重训最终部署模型。
    # HPO holdout 的指标仅作为模型选择诊断，不作为最终 OOF 性能。
    final_cfg = dict(config)
    final_cfg.update(best_params)
    final_cfg["epochs"] = max(1, int(selection_info["best_epoch"]))
    final_cfg["early_stopping"] = {
        **config.get("early_stopping", {}),
        "enabled": False,
    }
    final_cfg["lr_scheduler"] = {
        **config.get("lr_scheduler", {}),
        "enabled": False,
    }
    final_splits = {
        "X_train_raw": X_normal,
        "y_train": y_normal,
        "aging_gene_indices": aging_gene_indices,
    }
    best_model, scaler_X, scaler_y, _ = train_moe_clock(final_splits, final_cfg)
    torch.save(best_model.state_dict(), os.path.join(out_dir, "_best_model.pth"))
    joblib.dump(scaler_X, os.path.join(out_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(out_dir, "scaler_y.pkl"))

    importance_df = compute_gene_importance(
        best_model, scaler_X, normal_data[gene_columns], gene_columns
    )
    importance_df, _, _ = save_gene_importance_and_results(
        importance_df, all_results, out_dir, "HPO_all_normal"
    )
    train_df = all_results[all_results["data_type"] == "train"]
    val_df = all_results[all_results["data_type"] == "validation"]
    train_mae = mean_absolute_error(train_df["actual_age"], train_df["Predicted_Age"])
    train_r2 = r2_score(train_df["actual_age"], train_df["Predicted_Age"])
    val_mae = mean_absolute_error(val_df["actual_age"], val_df["Predicted_Age"])
    val_r2 = r2_score(val_df["actual_age"], val_df["Predicted_Age"])
    perf_plot_path = os.path.join(out_dir, "performance_train_val.png")
    plot_performance(
        train_df, val_df,
        {"train_mae": train_mae, "train_r2": train_r2, "val_mae": val_mae, "val_r2": val_r2},
        perf_plot_path,
        title_prefix="MoE HPO Best Model",
    )
    report_path = os.path.join(out_dir, "report_hpo.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Aging Clock HPO Report (PyTorch MoE) ===\n\n")
        f.write("Model: PyTorch Mixture-of-Experts\n")
        f.write(f"Fold-wise TMM+logCPM: {'enabled' if is_foldwise_tmm_enabled(config) else 'disabled'}\n")
        f.write(f"ComBat: {'enabled' if is_combat_enabled(config) else 'disabled'}\n")
        f.write(f"Total genes used (after TMM filtering): {len(gene_columns)}\n\n")
        if best_params:
            f.write("--- Best Hyperparameters (from Optuna) ---\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
        f.write("--- Performance (Normal train/val split) ---\n")
        f.write("These metrics are model-selection diagnostics, not outer-CV performance.\n")
        f.write(f"Train MAE: {train_mae:.3f} years, R2: {train_r2:.3f}\n")
        f.write(f"Val   MAE: {val_mae:.3f} years, R2: {val_r2:.3f}\n\n")
        f.write(f"Selected epochs: {int(selection_info['best_epoch'])}\n")
        f.write("Final deployment model was refit on all Normal samples with fixed epochs.\n\n")
        f.write(f"Number of genes with non-zero importance: {len(importance_df[importance_df['importance'] > 0])}\n")
        f.write("\nTop 10 genes by importance:\n")
        f.write(importance_df.head(10).to_string(index=False))
    return best_params, best_model, scaler_X, scaler_y, hpo_gene_columns


def stage_kfold_cv(data_pkg, config, root_dir):
    """
    阶段 2：5-fold KFold 交叉验证（Normal 上外层 CV）。
    每个 fold 应用 fold-wise TMM+ComBat，并仅在外层训练折内部完成 HPO 与早停。
    """
    out_dir = os.path.join(root_dir, "5fold_cv")
    os.makedirs(out_dir, exist_ok=True)
    normal_data = data_pkg["normal_data"]
    intervention_data = data_pkg["intervention_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]
    aging_gene_pool = data_pkg.get("aging_gene_pool", set())

    kf = KFold(n_splits=config["k_folds"], shuffle=True, random_state=config["seed"])
    all_fold_metrics = []
    cv_normal_list = []
    intervention_by_fold_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(normal_data)):
        fold_name = f"fold_{fold_idx + 1}"
        print("\n" + "=" * 50)
        print(f"阶段 2 - 5fold CV (MoE): {fold_name} ({fold_idx + 1}/{config['k_folds']})")
        print("=" * 50)
        train_df = normal_data.iloc[train_idx].copy()
        val_df = normal_data.iloc[val_idx].copy()
        print(f"  - 训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}")

        fold_dir = os.path.join(out_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        current_gene_columns = gene_columns
        split_dict = {"train": train_df, "validation": val_df, "intervention": intervention_data}

        if is_foldwise_tmm_enabled(config):
            train_df, split_dict, current_gene_columns = normalize_fold_splits_tmm_logcpm(
                train_df=train_df, splits_dict=split_dict, gene_columns=gene_columns,
                config=config, out_dir=os.path.join(fold_dir, "foldwise_tmm_logcpm"), prefix=fold_name,
            )
            val_df = split_dict["validation"]

        if is_combat_enabled(config):
            train_df, split_dict = run_combat_on_fold(
                train_df, split_dict, current_gene_columns, config, fold_label=fold_name
            )
            val_df = split_dict["validation"]

        fold_aging_gene_indices = compute_aging_gene_indices(current_gene_columns, aging_gene_pool)

        X_train_raw = train_df[current_gene_columns]
        y_train = train_df[config["age_col"]]
        model, scaler_X, scaler_y, fold_best_params, selected_epochs = (
            tune_and_refit_moe_on_outer_train(
                X_train_raw,
                y_train,
                fold_aging_gene_indices,
                config,
                fold_name,
            )
        )
        with open(os.path.join(fold_dir, f"best_params_{fold_name}.json"), "w", encoding="utf-8") as f:
            json.dump(
                {**fold_best_params, "selected_epochs": selected_epochs},
                f,
                ensure_ascii=False,
                indent=2,
            )

        fold_intervention = split_dict.get("intervention")
        fold_results = evaluate_on_splits(
            model, (scaler_X, scaler_y),
            {"train": train_df, "validation": val_df, "intervention": fold_intervention},
            current_gene_columns, metadata, config,
        )
        importance_df = compute_gene_importance(model, scaler_X, train_df[current_gene_columns], current_gene_columns)
        torch.save(model.state_dict(), os.path.join(fold_dir, f"MoE_model_{fold_name}.pth"))
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
            "train_mae": mean_absolute_error(train_res["actual_age"], train_res["Predicted_Age"]),
            "train_r2": r2_score(train_res["actual_age"], train_res["Predicted_Age"]),
            "val_mae": mean_absolute_error(val_res["actual_age"], val_res["Predicted_Age"]),
            "val_r2": r2_score(val_res["actual_age"], val_res["Predicted_Age"]),
            "selected_epochs": selected_epochs,
            "n_train": len(train_res), "n_val": len(val_res),
        }
        all_fold_metrics.append(metrics)
        perf_plot_path = os.path.join(fold_dir, f"performance_{fold_name}.png")
        plot_performance(train_res, val_res, metrics, perf_plot_path, title_prefix=f"MoE {fold_name}")
        accel_plot_path = os.path.join(fold_dir, f"age_acceleration_{fold_name}.png")
        plot_age_acceleration(fold_results, config, accel_plot_path, title=f"Age Acceleration (MoE {fold_name})")
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
        combat_note = "ComBat+" if is_combat_enabled(config) else ""
        with open(os.path.join(fold_dir, f"report_{fold_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== Report for {fold_name} (MoE 5-fold CV, {combat_note}fold-wise TMM) ===\n\n")
            f.write(f"Total genes used (after TMM filtering): {len(current_gene_columns)}\n")
            f.write(f"Aging genes in this fold: {len(fold_aging_gene_indices)}\n")
            if fold_best_params:
                f.write("--- Hyperparameters selected inside the outer training fold ---\n")
                for k, v in fold_best_params.items():
                    f.write(f"{k}: {v}\n")
            f.write(f"Selected epochs from inner validation: {selected_epochs}\n")
            f.write("Outer validation was used only once for final evaluation.\n\n")
            f.write(f"Number of genes with non-zero importance: {len(importance_df[importance_df['importance'] > 0])}\n\n")
            f.write(f"Train MAE = {metrics['train_mae']:.3f}, R2 = {metrics['train_r2']:.3f}\n")
            f.write(f"Val   MAE = {metrics['val_mae']:.3f}, R2 = {metrics['val_r2']:.3f}\n")

    cv_normal_all = pd.concat(cv_normal_list, ignore_index=True)
    intervention_all = pd.concat(intervention_by_fold_list, ignore_index=True) if intervention_by_fold_list else pd.DataFrame()
    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(os.path.join(out_dir, "summary_metrics_by_fold.csv"), index=False)
    overall_mae = mean_absolute_error(cv_normal_all["actual_age"], cv_normal_all["Predicted_Age"])
    overall_r2 = r2_score(cv_normal_all["actual_age"], cv_normal_all["Predicted_Age"])
    plt.figure(figsize=(8, 7))
    sns.regplot(x="actual_age", y="Predicted_Age", data=cv_normal_all, scatter_kws={"alpha": 0.7})
    plt.title(f"Overall {config['k_folds']}-Fold CV (, Normal)\nMAE={overall_mae:.2f}, R²={overall_r2:.2f}", fontsize=14)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age (Out-of-sample)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_performance_cv_normal.png"), dpi=300)
    plt.close()
    combined = pd.concat([cv_normal_all, intervention_all], ignore_index=True) if not intervention_all.empty else cv_normal_all.copy()
    combined.to_csv(os.path.join(out_dir, "summary_all_predictions_cv.csv"), index=False)
    plot_age_acceleration(combined, config, os.path.join(out_dir, "summary_age_acceleration.png"), title="Overall Age Acceleration ( 5-fold CV)")
    with open(os.path.join(out_dir, "summary_report_5fold_cv.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== Final 5-Fold CV Summary (, {config['k_folds']} folds) ===\n\n")
        f.write("Overall Normal (out-of-sample) performance:\n")
        f.write(f"  - MAE: {overall_mae:.3f} years\n")
        f.write(f"  - R2 : {overall_r2:.3f}\n\n")
        f.write("--- Metrics by fold ---\n")
        f.write(metrics_df.to_string(index=False))


def stage_dataset_loo(data_pkg, config, root_dir):
    """
    阶段 3：按 batch 进行 Leave-One-Batch-Out CV。

    HPO 与早停仅使用当前外层训练折；被留出的 batch 只用于最终评估。
    """
    out_dir = os.path.join(root_dir, "DataSet_loo")
    os.makedirs(out_dir, exist_ok=True)
    normal_data = data_pkg["normal_data"]
    intervention_data = data_pkg["intervention_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]
    aging_gene_pool = data_pkg.get("aging_gene_pool", set())
    if config["batch_col"] not in normal_data.columns:
        raise ValueError(f"在 normal_data 中找不到 batch 列 '{config['batch_col']}'")
    unique_batches = sorted(normal_data[config["batch_col"]].unique())
    print("\n步骤 3: DataSet-LOO (MoE, 按 batch leave-one-batch-out)...")
    print(f"  - 共发现 {len(unique_batches)} 个 batch: {unique_batches}")
    all_fold_metrics = []
    cv_normal_list = []
    intervention_by_fold_list = []
    for idx, val_batch in enumerate(unique_batches):
        fold_name = f"fold_{idx + 1}_val_{val_batch}"
        print("\n" + "=" * 50)
        print(f"DataSet-LOO (MoE): {fold_name}")
        print("=" * 50)
        train_df = normal_data[normal_data[config["batch_col"]] != val_batch].copy()
        val_df = normal_data[normal_data[config["batch_col"]] == val_batch].copy()
        print(f"  - 训练集大小: {len(train_df)}, 验证集(batch={val_batch})大小: {len(val_df)}")

        fold_dir = os.path.join(out_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        current_gene_columns = gene_columns
        split_dict = {"train": train_df, "validation": val_df, "intervention": intervention_data}

        if is_foldwise_tmm_enabled(config):
            train_df, split_dict, current_gene_columns = normalize_fold_splits_tmm_logcpm(
                train_df=train_df, splits_dict=split_dict, gene_columns=gene_columns,
                config=config, out_dir=os.path.join(fold_dir, "foldwise_tmm_logcpm"), prefix=fold_name,
            )
            val_df = split_dict["validation"]

        if is_combat_enabled(config):
            train_df, split_dict = run_combat_on_fold(
                train_df, split_dict, current_gene_columns, config, fold_label=fold_name
            )
            val_df = split_dict["validation"]

        fold_aging_gene_indices = compute_aging_gene_indices(current_gene_columns, aging_gene_pool)
        fold_intervention = split_dict.get("intervention")

        X_train_raw = train_df[current_gene_columns]
        y_train = train_df[config["age_col"]]
        model, scaler_X, scaler_y, fold_best_params, selected_epochs = (
            tune_and_refit_moe_on_outer_train(
                X_train_raw,
                y_train,
                fold_aging_gene_indices,
                config,
                fold_name,
            )
        )
        with open(os.path.join(fold_dir, f"best_params_{fold_name}.json"), "w", encoding="utf-8") as f:
            json.dump(
                {**fold_best_params, "selected_epochs": selected_epochs},
                f,
                ensure_ascii=False,
                indent=2,
            )
        fold_results = evaluate_on_splits(
            model, (scaler_X, scaler_y),
            {"train": train_df, "validation": val_df, "intervention": fold_intervention},
            current_gene_columns, metadata, config,
        )
        importance_df = compute_gene_importance(model, scaler_X, train_df[current_gene_columns], current_gene_columns)
        torch.save(model.state_dict(), os.path.join(fold_dir, f"MoE_model_{fold_name}.pth"))
        joblib.dump(scaler_X, os.path.join(fold_dir, f"scaler_X_{fold_name}.pkl"))
        joblib.dump(scaler_y, os.path.join(fold_dir, f"scaler_Y_{fold_name}.pkl"))
        importance_df, _, _ = save_gene_importance_and_results(importance_df, fold_results, fold_dir, fold_name)
        train_res = fold_results[fold_results["data_type"] == "train"]
        val_res = fold_results[fold_results["data_type"] == "validation"]
        interv_res = fold_results[fold_results["data_type"] == "intervention"]
        val_res_filtered = val_res[val_res[config["batch_col"]] == val_batch].copy()
        metrics = {
            "fold": fold_name, "val_batch": val_batch,
            "train_mae": mean_absolute_error(train_res["actual_age"], train_res["Predicted_Age"]),
            "train_r2": r2_score(train_res["actual_age"], train_res["Predicted_Age"]),
            "val_mae": mean_absolute_error(val_res_filtered["actual_age"], val_res_filtered["Predicted_Age"]),
            "val_r2": r2_score(val_res_filtered["actual_age"], val_res_filtered["Predicted_Age"]),
            "selected_epochs": selected_epochs,
            "n_train": len(train_res), "n_val": len(val_res_filtered),
        }
        all_fold_metrics.append(metrics)
        plot_performance(train_res, val_res_filtered, metrics, os.path.join(fold_dir, f"performance_{fold_name}.png"), title_prefix=f"MoE {fold_name}")
        plot_age_acceleration(fold_results, config, os.path.join(fold_dir, f"age_acceleration_{fold_name}.png"), title=f"Age Acceleration (MoE {fold_name})")
        combat_note = "ComBat+" if is_combat_enabled(config) else ""
        with open(os.path.join(fold_dir, f"report_{fold_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== Report for {fold_name} (MoE DataSet-LOO, {combat_note}fold-wise TMM) ===\n\n")
            f.write(f"Validation batch: {val_batch}\n")
            f.write(f"Total genes used (after TMM filtering): {len(current_gene_columns)}\n")
            if fold_best_params:
                f.write("--- Hyperparameters selected inside the outer training fold ---\n")
                for k, v in fold_best_params.items():
                    f.write(f"{k}: {v}\n")
            f.write(f"Selected epochs from inner validation: {selected_epochs}\n")
            f.write("Outer validation batch was used only once for final evaluation.\n\n")
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
    overall_mae = mean_absolute_error(cv_normal_all["actual_age"], cv_normal_all["Predicted_Age"])
    overall_r2 = r2_score(cv_normal_all["actual_age"], cv_normal_all["Predicted_Age"])
    plt.figure(figsize=(8, 7))
    sns.regplot(x="actual_age", y="Predicted_Age", data=cv_normal_all, scatter_kws={"alpha": 0.7})
    plt.title("Overall DataSet-LOO Performance (MoE, Normal, leave-one-batch-out)\n" f"MAE={overall_mae:.2f}, R²={overall_r2:.2f}", fontsize=14)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age (Out-of-sample)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_performance_cv_normal.png"), dpi=300)
    plt.close()
    combined = pd.concat([cv_normal_all, intervention_all], ignore_index=True) if not intervention_all.empty else cv_normal_all.copy()
    combined.to_csv(os.path.join(out_dir, "summary_all_predictions_cv.csv"), index=False)
    plot_age_acceleration(combined, config, os.path.join(out_dir, "summary_age_acceleration.png"), title="Overall Age Acceleration (MoE DataSet-LOO)")
    with open(os.path.join(out_dir, "summary_report_dataset_loo.txt"), "w", encoding="utf-8") as f:
        f.write("=== Final DataSet-LOO Summary (MoE Leave-One-Batch-Out) ===\n\n")
        f.write("Overall Normal (out-of-sample) performance:\n")
        f.write(f"  - MAE: {overall_mae:.3f} years\n")
        f.write(f"  - R2 : {overall_r2:.3f}\n\n")
        f.write("--- Metrics by batch ---\n")
        f.write(metrics_df.to_string(index=False))


def stage_group_loo(data_pkg, config, root_dir):
    """
    阶段 4：按 subgroup 进行 Leave-One-Group-Out CV。

    HPO 与早停仅使用当前外层训练折；被留出的 subgroup 只用于最终评估。
    """
    out_dir = os.path.join(root_dir, "group_loo")
    os.makedirs(out_dir, exist_ok=True)
    normal_data = data_pkg["normal_data"]
    intervention_data = data_pkg["intervention_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]
    aging_gene_pool = data_pkg.get("aging_gene_pool", set())
    group_col = config["subgroup_col"]
    if group_col not in normal_data.columns:
        raise ValueError(f"在 normal_data 中找不到 group 列 '{group_col}'")
    unique_groups = sorted(normal_data[group_col].unique())
    print("\n步骤 4: Group-LOO (MoE, 按 subgroup leave-one-group-out)...")
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
        print(f"Group-LOO (MoE): {fold_name}")
        print("=" * 50)
        train_df = normal_data.iloc[train_idx].copy()
        val_df = normal_data.iloc[val_idx].copy()
        print(f"  - 训练集大小: {len(train_df)}, 验证集({group_col}={val_group})大小: {len(val_df)}")

        fold_dir = os.path.join(out_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        current_gene_columns = gene_columns
        split_dict = {"train": train_df, "validation": val_df, "intervention": intervention_data}

        if is_foldwise_tmm_enabled(config):
            train_df, split_dict, current_gene_columns = normalize_fold_splits_tmm_logcpm(
                train_df=train_df, splits_dict=split_dict, gene_columns=gene_columns,
                config=config, out_dir=os.path.join(fold_dir, "foldwise_tmm_logcpm"), prefix=fold_name,
            )
            val_df = split_dict["validation"]

        if is_combat_enabled(config):
            train_df, split_dict = run_combat_on_fold(
                train_df, split_dict, current_gene_columns, config, fold_label=fold_name
            )
            val_df = split_dict["validation"]

        fold_aging_gene_indices = compute_aging_gene_indices(current_gene_columns, aging_gene_pool)
        fold_intervention = split_dict.get("intervention")

        X_train_raw = train_df[current_gene_columns]
        y_train = train_df[config["age_col"]]
        model, scaler_X, scaler_y, fold_best_params, selected_epochs = (
            tune_and_refit_moe_on_outer_train(
                X_train_raw,
                y_train,
                fold_aging_gene_indices,
                config,
                fold_name,
            )
        )
        with open(os.path.join(fold_dir, f"best_params_{fold_name}.json"), "w", encoding="utf-8") as f:
            json.dump(
                {**fold_best_params, "selected_epochs": selected_epochs},
                f,
                ensure_ascii=False,
                indent=2,
            )
        fold_results = evaluate_on_splits(
            model, (scaler_X, scaler_y),
            {"train": train_df, "validation": val_df, "intervention": fold_intervention},
            current_gene_columns, metadata, config,
        )
        importance_df = compute_gene_importance(model, scaler_X, train_df[current_gene_columns], current_gene_columns)
        torch.save(model.state_dict(), os.path.join(fold_dir, f"MoE_model_{fold_name}.pth"))
        joblib.dump(scaler_X, os.path.join(fold_dir, f"scaler_X_{fold_name}.pkl"))
        joblib.dump(scaler_y, os.path.join(fold_dir, f"scaler_Y_{fold_name}.pkl"))
        importance_df, _, _ = save_gene_importance_and_results(importance_df, fold_results, fold_dir, fold_name)
        train_res = fold_results[fold_results["data_type"] == "train"]
        val_res = fold_results[fold_results["data_type"] == "validation"]
        interv_res = fold_results[fold_results["data_type"] == "intervention"]
        val_res_filtered = val_res[val_res[group_col] == val_group].copy()
        metrics = {
            "fold": fold_name, "val_group": val_group,
            "train_mae": mean_absolute_error(train_res["actual_age"], train_res["Predicted_Age"]),
            "train_r2": r2_score(train_res["actual_age"], train_res["Predicted_Age"]),
            "val_mae": mean_absolute_error(val_res_filtered["actual_age"], val_res_filtered["Predicted_Age"]),
            "val_r2": r2_score(val_res_filtered["actual_age"], val_res_filtered["Predicted_Age"]),
            "selected_epochs": selected_epochs,
            "n_train": len(train_res), "n_val": len(val_res_filtered),
        }
        all_fold_metrics.append(metrics)
        plot_performance(train_res, val_res_filtered, metrics, os.path.join(fold_dir, f"performance_{fold_name}.png"), title_prefix=f"MoE {fold_name}")
        plot_age_acceleration(fold_results, config, os.path.join(fold_dir, f"age_acceleration_{fold_name}.png"), title=f"Age Acceleration (MoE {fold_name})")
        combat_note = "ComBat+" if is_combat_enabled(config) else ""
        with open(os.path.join(fold_dir, f"report_{fold_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== Report for {fold_name} (MoE Group-LOO, {combat_note}fold-wise TMM) ===\n\n")
            f.write(f"Validation group ({group_col}): {val_group}\n")
            f.write(f"Total genes used (after TMM filtering): {len(current_gene_columns)}\n")
            if fold_best_params:
                f.write("--- Hyperparameters selected inside the outer training fold ---\n")
                for k, v in fold_best_params.items():
                    f.write(f"{k}: {v}\n")
            f.write(f"Selected epochs from inner validation: {selected_epochs}\n")
            f.write("Outer validation group was used only once for final evaluation.\n\n")
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
    overall_mae = mean_absolute_error(cv_normal_all["actual_age"], cv_normal_all["Predicted_Age"])
    overall_r2 = r2_score(cv_normal_all["actual_age"], cv_normal_all["Predicted_Age"])
    plt.figure(figsize=(8, 7))
    sns.regplot(x="actual_age", y="Predicted_Age", data=cv_normal_all, scatter_kws={"alpha": 0.7})
    plt.title("Overall Group-LOO Performance (MoE, Normal, leave-one-group-out)\n" f"MAE={overall_mae:.2f}, R²={overall_r2:.2f}", fontsize=14)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age (Out-of-sample)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_performance_cv_normal.png"), dpi=300)
    plt.close()
    combined = pd.concat([cv_normal_all, intervention_all], ignore_index=True) if not intervention_all.empty else cv_normal_all.copy()
    combined.to_csv(os.path.join(out_dir, "summary_all_predictions_cv.csv"), index=False)
    plot_age_acceleration(combined, config, os.path.join(out_dir, "summary_age_acceleration.png"), title="Overall Age Acceleration (MoE Group-LOO)")
    with open(os.path.join(out_dir, "summary_report_group_loo.txt"), "w", encoding="utf-8") as f:
        f.write("=== Final Group-LOO Summary (MoE Leave-One-Group-Out by subgroup) ===\n\n")
        f.write("Overall Normal (out-of-sample) performance:\n")
        f.write(f"  - MAE: {overall_mae:.3f} years\n")
        f.write(f"  - R2 : {overall_r2:.3f}\n\n")
        f.write("--- Metrics by group ---\n")
        f.write(metrics_df.to_string(index=False))


def stage_group_prediction_with_best_model(data_pkg, best_model, best_scaler_X, best_scaler_y,
                                           best_gene_columns, config, root_dir):
    """阶段 5：用 HPO 得到的最优 MoE 模型对所有样本预测，并按 group 汇总。"""
    out_dir = os.path.join(root_dir, "group_best_model")
    os.makedirs(out_dir, exist_ok=True)
    all_data = data_pkg["all_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]
    normal_data = data_pkg["normal_data"]
    print("\n步骤 5: 使用 HPO 得到的最优模型，对所有样本预测，并按 group 汇总...")

    if is_foldwise_tmm_enabled(config):
        _, norm_splits, current_gene_columns = normalize_fold_splits_tmm_logcpm(
            train_df=normal_data,
            splits_dict={"all": all_data},
            gene_columns=gene_columns,
            config=config,
            out_dir=os.path.join(out_dir, "foldwise_tmm_logcpm"),
            prefix="best_model_all_samples",
        )
        all_data = norm_splits["all"]
        normal_data_norm = norm_splits.get("train", normal_data)
        missing = [gene for gene in best_gene_columns if gene not in all_data.columns]
        if missing:
            raise ValueError(f"best-model 预测数据缺少 HPO 阶段使用的基因: {missing[:10]}")
        gene_columns = best_gene_columns
    else:
        gene_columns = best_gene_columns
        normal_data_norm = normal_data

    X_all = all_data[gene_columns].copy()
    if X_all.isnull().any().any():
        X_all = X_all.fillna(X_all.mean())
    y_all = all_data[config["age_col"]]
    device = next(best_model.parameters()).device
    X_all_scaled = best_scaler_X.transform(X_all)
    if np.isnan(X_all_scaled).any():
        X_all_scaled = np.nan_to_num(X_all_scaled, nan=0.0)
    X_all_t = torch.tensor(X_all_scaled, dtype=torch.float32, device=device)
    best_model.eval()
    with torch.no_grad():
        y_pred_scaled, _, _ = best_model(X_all_t)
    y_pred = best_scaler_y.inverse_transform(y_pred_scaled.cpu().numpy()).flatten()
    res = pd.DataFrame({"actual_age": y_all, "Predicted_Age": y_pred}, index=all_data.index)
    res = pd.merge(res.reset_index().rename(columns={"index": "sample_id"}), metadata, on="sample_id", how="left")
    res["age_acceleration"] = res["Predicted_Age"] - res["actual_age"]
    norm_for_imp = normal_data_norm if is_foldwise_tmm_enabled(config) else normal_data
    X_norm_imp = norm_for_imp[gene_columns].copy()
    if X_norm_imp.isnull().any().any():
        X_norm_imp = X_norm_imp.fillna(X_norm_imp.mean())
    importance_df = compute_gene_importance(best_model, best_scaler_X, X_norm_imp, gene_columns)
    importance_path = os.path.join(out_dir, "gene_importances_best_model.csv")
    importance_df.to_csv(importance_path, index=False)
    all_pred_path = os.path.join(out_dir, "all_samples_predictions_best_MoE_model.csv")
    res.to_csv(all_pred_path, index=False)
    group_col = config["subgroup_col"]
    group_summary = (
        res.groupby(group_col)
        .agg(
            n_samples=("sample_id", "count"),
            mean_actual_age=("actual_age", "mean"),
            mean_Predicted_Age=("Predicted_Age", "mean"),
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
    plt.bar(x + width / 2, group_summary["mean_Predicted_Age"], width, label="Predicted")
    plt.xticks(x, group_summary[group_col], rotation=45, ha="right")
    plt.ylabel("Age (years)")
    plt.title("Mean Actual & Predicted Age per Group (Best MoE Model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "group_mean_age_barplot.png"), dpi=300)
    plt.close()
    plot_age_acceleration(
        res, config,
        os.path.join(out_dir, "age_acceleration_by_group_best_model.png"),
        title="Age Acceleration by Group (Best MoE Model on All Data)",
    )
    with open(os.path.join(out_dir, "report_group_best_model.txt"), "w", encoding="utf-8") as f:
        f.write("=== Group-level Predictions with Best MoE Model ===\n\n")
        f.write("该模型来自 HPO 阶段，在所有 Normal 样本上训练得到，然后用于全体样本的预测。\n\n")
        f.write(f"Number of genes with non-zero importance: {len(importance_df[importance_df['importance'] > 0])}\n")
        f.write("\nTop 10 genes by importance:\n")
        f.write(importance_df.head(10).to_string(index=False))
        f.write("\n\n--- Group-wise summary ---\n")
        f.write(group_summary.to_string(index=False))


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    print(f"✅ 所有随机种子已设置为: {CONFIG['seed']}")
    root_dir = resolve_path(CONFIG["output_root_dir"])
    os.makedirs(root_dir, exist_ok=True)
    print(f"输出根目录: {root_dir}")
    data_pkg = load_full_data(CONFIG)

    _, best_model, best_scaler_X, best_scaler_Y, best_gene_columns = stage_hpo_and_save(
        data_pkg["normal_data"],
        data_pkg["gene_columns"],
        data_pkg["metadata"],
        data_pkg["aging_gene_indices"],
        CONFIG,
        root_dir,
        aging_gene_pool=data_pkg.get("aging_gene_pool"),
    )
    stage_kfold_cv(data_pkg, CONFIG, root_dir)
    stage_dataset_loo(data_pkg, CONFIG, root_dir)
    stage_group_loo(data_pkg, CONFIG, root_dir)
    stage_group_prediction_with_best_model(
        data_pkg, best_model, best_scaler_X, best_scaler_Y, best_gene_columns, CONFIG, root_dir
    )
    print("\n✅ 全部流程完成！结果请查看输出根目录下的各个子文件夹。")
