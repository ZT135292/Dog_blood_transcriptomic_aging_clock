import os
import json
import shutil
import subprocess
import tempfile
import warnings
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, RepeatedKFold, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score

CONFIG = {
    "metadata_file": "total_metadata.txt",
    "raw_count_file": "total_count_sort_ortholog.txt",

    "output_root_dir": "aging_clock_results_Random_Forest_pipeline_foldwise_TMM_logCPM",

    "foldwise_tmm_logcpm": {
        "enabled": True,
        "rscript": "Rscript",
        "prior_count": 1,
        "filter_low_expression": True,
    },

    "sample_id_col": "sample_id",
    "group_col": "condition",
    "subgroup_col": "group",
    "age_col": "Chronological_Age",
    "batch_col": "batch",

    "normal_group_label": "Normal",

    "outlier_detection": {
        "enabled": False,
        "method": "iqr",
        "iqr_factor": 1.5,
        "zscore_threshold": 3.0,
        "check_age": False,
        "age_iqr_factor": 3.0,
    },

    "combat": {
        "enabled": False,
        "covariate_continuous": ["age"],
        "covariate_categorical": [],
        "min_samples_per_cat_level": 2,
        "min_samples_per_batch": 2,
    },

    "k_folds": 5,
    "random_state": 66,

    "n_estimators": 100,
    "max_depth": None,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "bootstrap": True,

    "rf_hpo": {
        "enabled": True,
        # 全量 HPO 与所有外层 fold 使用同一搜索预算。
        "n_iter": 20,
        "cv_folds": 5,
        "repeats": 1,
        "random_state": 42,
    }
}


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


def load_full_data(config):
    """读取元数据和 raw counts 矩阵，并对齐。"""
    print("步骤 0: 加载和预处理数据...")

    metadata_path = resolve_path(config["metadata_file"])
    expr_path = resolve_path(config["raw_count_file"])
    print(f"  - 读取元数据: {metadata_path}")
    print(f"  - 读取 raw counts: {expr_path}")

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
    }


def _get_rf_param_dist():
    return {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [None, 8, 12, 16, 20, 30],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 4, 8, 16],
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        "bootstrap": [True],
    }


def train_with_rf_hpo(X_raw, y, config):
    """使用 RandomizedSearchCV 进行超参数优化（仅在 Normal 上做 HPO）。"""
    print("\n步骤 1: 使用所有 Normal 样本做 HPO (RandomizedSearchCV)...")
    hpo = config.get("rf_hpo", {})
    if not hpo.get("enabled", False):
        print("  - HPO 关闭，使用默认参数训练。")
        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            min_samples_split=config["min_samples_split"],
            max_features=config["max_features"],
            bootstrap=config["bootstrap"],
            random_state=config["random_state"],
            oob_score=True,
            n_jobs=-1,
        )
        model.fit(X_raw, y)
        best_params = {
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "min_samples_leaf": config["min_samples_leaf"],
            "min_samples_split": config["min_samples_split"],
            "max_features": config["max_features"],
            "bootstrap": config["bootstrap"],
        }
        return model, best_params

    base_model = RandomForestRegressor(random_state=config["random_state"], oob_score=False, n_jobs=-1)

    if hpo.get("repeats", 1) > 1:
        cv = RepeatedKFold(
            n_splits=hpo.get("cv_folds", 5),
            n_repeats=hpo.get("repeats", 1),
            random_state=hpo.get("random_state", 42),
        )
    else:
        cv = hpo.get("cv_folds", 5)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=_get_rf_param_dist(),
        n_iter=hpo.get("n_iter", 40),
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=hpo.get("random_state", 42),
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_raw, y)
    best_params = search.best_params_.copy()
    best_score = -search.best_score_
    print(f"  - HPO 完成，最佳验证(交叉验证) MAE: {best_score:.4f}")
    print(f"  - 最佳参数: {best_params}")
    return search.best_estimator_, best_params


def run_rf_hpo_on_fold(X_raw, y, config, fold_label=""):
    """在单个 fold 的训练集上运行 RandomizedSearchCV，独立搜索最优超参数。"""
    tag = f" [{fold_label}]" if fold_label else ""
    print(f"  - Fold-wise HPO{tag}: 在当前训练集上运行 RandomizedSearchCV ...")

    X_raw = X_raw.copy()
    if isinstance(X_raw, pd.DataFrame) and X_raw.isnull().any().any():
        print(f"    -> 警告：训练特征矩阵含 NaN，用列均值填充。")
        X_raw = X_raw.fillna(X_raw.mean())

    hpo = config.get("rf_hpo", {})
    n_iter = hpo.get("n_iter", 20)

    base_model = RandomForestRegressor(random_state=config["random_state"], oob_score=False, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=_get_rf_param_dist(),
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=hpo.get("cv_folds", 5),
        random_state=hpo.get("random_state", 42),
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_raw, y)
    best_params = search.best_params_.copy()
    print(f"    -> 最佳参数: {best_params}")
    return search.best_estimator_, best_params


def evaluate_on_splits(model, splits_dict, gene_columns, metadata, config):
    """对给定的多个数据子集生成预测并计算 age_acceleration（RF 不需要 scaler）。"""
    frames = []
    for name, df in splits_dict.items():
        if df is None or len(df) == 0:
            continue
        X_raw = df[gene_columns].copy()
        if X_raw.isnull().any().any():
            X_raw = X_raw.fillna(X_raw.mean())
        y_true = df[config["age_col"]]
        y_pred = model.predict(X_raw)

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


def save_importance_and_results(model, gene_columns, all_results, out_dir, prefix):
    """保存模型特征重要性和预测结果。"""
    os.makedirs(out_dir, exist_ok=True)

    importance_df = pd.DataFrame(
        {"gene": gene_columns, "importance": model.feature_importances_}
    )
    importance_df = importance_df[importance_df["importance"] > 0].sort_values(
        "importance", ascending=False
    )
    importance_path = os.path.join(out_dir, f"gene_importances_{prefix}.csv")
    importance_df.to_csv(importance_path, index=False)

    pred_path = os.path.join(out_dir, f"predictions_{prefix}.csv")
    all_results.to_csv(pred_path, index=False)

    return importance_df, importance_path, pred_path


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
    order = sorted(all_results[config["subgroup_col"]].unique())
    sns.boxplot(
        x=config["subgroup_col"], y="age_acceleration",
        hue=config["subgroup_col"], data=all_results,
        palette="viridis", order=order, legend=False,
    )
    sns.stripplot(
        x=config["subgroup_col"], y="age_acceleration",
        data=all_results, color="black", alpha=0.4, order=order,
    )
    plt.axhline(0, color="red", linestyle="--")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def stage_hpo_and_save(normal_data, gene_columns, metadata, config, root_dir):
    """
    阶段 1：
      - 所有 Normal 样本做 TMM+logCPM + RandomizedSearchCV 找到最佳超参数
      - 保存 best_hyperparameters.json、模型、特征重要性和预测
      - 返回 best_params, best_model, hpo_gene_columns
    """
    out_dir = os.path.join(root_dir, "HPO")
    os.makedirs(out_dir, exist_ok=True)

    if is_foldwise_tmm_enabled(config):
        normal_data, norm_splits, hpo_gene_columns = normalize_fold_splits_tmm_logcpm(
            train_df=normal_data,
            splits_dict={"train": normal_data},
            gene_columns=gene_columns,
            config=config,
            out_dir=out_dir,
            prefix="HPO_all_normal",
        )
        normal_data = norm_splits["train"]
        gene_columns = hpo_gene_columns

    X_normal = normal_data[gene_columns]
    y_normal = normal_data[config["age_col"]]

    model_cv, best_params = train_with_rf_hpo(X_normal, y_normal, config)

    joblib.dump(model_cv, os.path.join(out_dir, "random_forest_best_model.pkl"))

    best_path = os.path.join(out_dir, "best_hyperparameters.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    print(f"  - 最优超参数已保存到: {best_path}")

    all_results = evaluate_on_splits(model_cv, {"train": normal_data}, gene_columns, metadata, config)
    importance_df, importance_path, pred_path = save_importance_and_results(
        model_cv, gene_columns, all_results, out_dir, "HPO_all_normal"
    )

    train_df = all_results[all_results["data_type"] == "train"]
    train_mae = mean_absolute_error(train_df["actual_age"], train_df["Predicted_Age"])
    train_r2 = r2_score(train_df["actual_age"], train_df["Predicted_Age"])

    perf_plot_path = os.path.join(out_dir, "performance_train_only.png")
    plot_performance(
        train_df, train_df,
        {"train_mae": train_mae, "train_r2": train_r2, "val_mae": train_mae, "val_r2": train_r2},
        perf_plot_path, title_prefix="RF HPO Best Model",
    )

    report_path = os.path.join(out_dir, "report_hpo.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Aging Clock HPO Report ===\n\n")
        f.write("Model: RandomForestRegressor\n")
        f.write("Best parameters:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        oob_score = getattr(model_cv, 'oob_score_', None)
        if oob_score is not None:
            f.write(f"\nOut-of-Bag (OOB) R2 Score: {oob_score:.4f}\n\n")
        else:
            f.write("\nOut-of-Bag (OOB) R2 Score: N/A\n\n")
        f.write(f"Total genes used (after TMM filtering): {len(gene_columns)}\n\n")
        f.write("--- Training Performance (on all Normal) ---\n")
        f.write(f"  - MAE: {train_mae:.3f} years\n")
        f.write(f"  - R2 : {train_r2:.3f}\n\n")
        f.write(f"Number of selected (non-zero) features: {len(importance_df)}\n")
        f.write("\nTop 10 genes:\n")
        f.write(importance_df.head(10).to_string(index=False))

    return best_params, model_cv, gene_columns


def stage_kfold_cv(data_pkg, config, root_dir):
    """
    阶段 2：每个 fold 独立在训练集上运行 RandomizedSearchCV 搜索最优超参数。
    """
    out_dir = os.path.join(root_dir, "5fold_cv")
    os.makedirs(out_dir, exist_ok=True)

    normal_data = data_pkg["normal_data"]
    intervention_data = data_pkg["intervention_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]

    kf = KFold(n_splits=config["k_folds"], shuffle=True, random_state=config["random_state"])

    all_fold_metrics = []
    cv_normal_list = []
    intervention_by_fold_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(normal_data)):
        fold_name = f"fold_{fold_idx + 1}"
        print("\n" + "=" * 50)
        print(f"阶段 2 - 5fold CV: {fold_name} ({fold_idx + 1}/{config['k_folds']})")
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

        model, fold_best_params = run_rf_hpo_on_fold(
            train_df[current_gene_columns], train_df[config["age_col"]], config, fold_label=fold_name,
        )

        fold_results = evaluate_on_splits(model, split_dict, current_gene_columns, metadata, config)

        joblib.dump(model, os.path.join(fold_dir, f"model_{fold_name}.pkl"))

        with open(os.path.join(fold_dir, f"best_params_{fold_name}.json"), "w", encoding="utf-8") as f:
            json.dump(fold_best_params, f, ensure_ascii=False, indent=2)

        importance_df, _, _ = save_importance_and_results(model, current_gene_columns, fold_results, fold_dir, fold_name)

        train_res = fold_results[fold_results["data_type"] == "train"]
        val_res = fold_results[fold_results["data_type"] == "validation"]
        intervention_res = fold_results[fold_results["data_type"] == "intervention"]

        metrics = {
            "fold": fold_name,
            "train_mae": mean_absolute_error(train_res["actual_age"], train_res["Predicted_Age"]),
            "train_r2": r2_score(train_res["actual_age"], train_res["Predicted_Age"]),
            "val_mae": mean_absolute_error(val_res["actual_age"], val_res["Predicted_Age"]),
            "val_r2": r2_score(val_res["actual_age"], val_res["Predicted_Age"]),
            "n_train": len(train_res),
            "n_val": len(val_res),
        }
        all_fold_metrics.append(metrics)

        plot_performance(train_res, val_res, metrics, os.path.join(fold_dir, f"performance_{fold_name}.png"), title_prefix=fold_name)
        plot_age_acceleration(fold_results, config, os.path.join(fold_dir, f"age_acceleration_{fold_name}.png"), title=f"Age Acceleration ({fold_name})")

        val_res_with_fold = val_res.copy()
        val_res_with_fold["fold"] = fold_name
        cv_normal_list.append(val_res_with_fold)

        if not intervention_res.empty:
            group_col = config["subgroup_col"]
            if group_col in val_res.columns and group_col in intervention_res.columns:
                val_groups = set(val_res[group_col].unique())
                intervention_res_filtered = intervention_res[intervention_res[group_col].isin(val_groups)].copy()
                if len(intervention_res_filtered) != len(intervention_res):
                    print(f"  信息: intervention 样本已过滤，只保留与 validation 相同 group 的样本。原始: {len(intervention_res)}, 过滤后: {len(intervention_res_filtered)}")
            else:
                intervention_res_filtered = intervention_res.copy()
            if not intervention_res_filtered.empty:
                intervention_res_filtered = intervention_res_filtered.copy()
                intervention_res_filtered["fold"] = fold_name
                intervention_by_fold_list.append(intervention_res_filtered)

        combat_note = "ComBat+" if is_combat_enabled(config) else ""
        with open(os.path.join(fold_dir, f"report_{fold_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== Report for {fold_name} (5-fold CV, {combat_note}fold-wise HPO) ===\n\n")
            f.write("Fold-wise HPO best parameters:\n")
            for k, v in fold_best_params.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nComBat batch correction: {'enabled' if is_combat_enabled(config) else 'disabled'}\n")
            f.write(f"Total genes used (after TMM filtering): {len(current_gene_columns)}\n")
            f.write(f"Number of non-zero features: {len(importance_df)}\n\n")
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
    plt.title(f"Overall {config['k_folds']}-Fold CV (Normal)\nMAE={overall_mae:.2f}, R²={overall_r2:.2f}", fontsize=14)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age (Out-of-sample)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_performance_cv_normal.png"), dpi=300)
    plt.close()

    if not intervention_all.empty:
        combined = pd.concat([cv_normal_all, intervention_all], ignore_index=True)
    else:
        combined = cv_normal_all.copy()
    combined.to_csv(os.path.join(out_dir, "summary_all_predictions_cv.csv"), index=False)
    plot_age_acceleration(combined, config, os.path.join(out_dir, "summary_age_acceleration.png"), title="Overall Age Acceleration (5-fold CV)")

    with open(os.path.join(out_dir, "summary_report_5fold_cv.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== Final 5-Fold CV Summary ({config['k_folds']} folds) ===\n\n")
        f.write("Overall Normal (out-of-sample) performance:\n")
        f.write(f"  - MAE: {overall_mae:.3f} years\n")
        f.write(f"  - R2 : {overall_r2:.3f}\n\n")
        f.write("--- Metrics by fold ---\n")
        f.write(metrics_df.to_string(index=False))


def stage_dataset_loo(data_pkg, config, root_dir):
    """
    阶段 3：按 batch 进行 Leave-One-Batch-Out CV。
    每个 fold 独立在训练集上运行 RandomizedSearchCV 搜索最优超参数。
    """
    out_dir = os.path.join(root_dir, "DataSet_loo")
    os.makedirs(out_dir, exist_ok=True)

    normal_data = data_pkg["normal_data"]
    intervention_data = data_pkg["intervention_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]

    if config["batch_col"] not in normal_data.columns:
        raise ValueError(f"在 normal_data 中找不到 batch 列 '{config['batch_col']}'")

    unique_batches = sorted(normal_data[config["batch_col"]].unique())
    print("\n步骤 3: DataSet-LOO (按 batch leave-one-batch-out, fold-wise HPO)...")
    print(f"  - 共发现 {len(unique_batches)} 个 batch: {unique_batches}")

    all_fold_metrics = []
    cv_normal_list = []
    intervention_by_fold_list = []

    for idx, val_batch in enumerate(unique_batches):
        fold_name = f"fold_{idx + 1}_val_{val_batch}"
        print("\n" + "=" * 50)
        print(f"DataSet-LOO: {fold_name}")
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

        model, fold_best_params = run_rf_hpo_on_fold(
            train_df[current_gene_columns], train_df[config["age_col"]], config, fold_label=fold_name,
        )

        fold_results = evaluate_on_splits(model, split_dict, current_gene_columns, metadata, config)

        joblib.dump(model, os.path.join(fold_dir, f"model_{fold_name}.pkl"))

        with open(os.path.join(fold_dir, f"best_params_{fold_name}.json"), "w", encoding="utf-8") as f:
            json.dump(fold_best_params, f, ensure_ascii=False, indent=2)

        importance_df, _, _ = save_importance_and_results(model, current_gene_columns, fold_results, fold_dir, fold_name)

        train_res = fold_results[fold_results["data_type"] == "train"]
        val_res = fold_results[fold_results["data_type"] == "validation"]
        intervention_res = fold_results[fold_results["data_type"] == "intervention"]

        val_res_filtered = val_res[val_res[config["batch_col"]] == val_batch].copy()
        if len(val_res_filtered) != len(val_res):
            print(f"  警告: validation 样本的 batch 不匹配，已过滤。原始: {len(val_res)}, 过滤后: {len(val_res_filtered)}")

        metrics = {
            "fold": fold_name,
            "val_batch": val_batch,
            "train_mae": mean_absolute_error(train_res["actual_age"], train_res["Predicted_Age"]),
            "train_r2": r2_score(train_res["actual_age"], train_res["Predicted_Age"]),
            "val_mae": mean_absolute_error(val_res_filtered["actual_age"], val_res_filtered["Predicted_Age"]),
            "val_r2": r2_score(val_res_filtered["actual_age"], val_res_filtered["Predicted_Age"]),
            "n_train": len(train_res),
            "n_val": len(val_res_filtered),
        }
        all_fold_metrics.append(metrics)

        plot_performance(train_res, val_res_filtered, metrics, os.path.join(fold_dir, f"performance_{fold_name}.png"), title_prefix=fold_name)
        plot_age_acceleration(fold_results, config, os.path.join(fold_dir, f"age_acceleration_{fold_name}.png"), title=f"Age Acceleration ({fold_name})")

        combat_note = "ComBat+" if is_combat_enabled(config) else ""
        with open(os.path.join(fold_dir, f"report_{fold_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== Report for {fold_name} (DataSet-LOO, {combat_note}fold-wise HPO) ===\n\n")
            f.write(f"Validation batch: {val_batch}\n")
            f.write("Fold-wise HPO best parameters:\n")
            for k, v in fold_best_params.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nComBat batch correction: {'enabled' if is_combat_enabled(config) else 'disabled'}\n")
            f.write(f"Total genes used (after TMM filtering): {len(current_gene_columns)}\n")
            f.write(f"Number of non-zero features: {len(importance_df)}\n\n")
            f.write(f"Train MAE = {metrics['train_mae']:.3f}, R2 = {metrics['train_r2']:.3f}\n")
            f.write(f"Val   MAE = {metrics['val_mae']:.3f}, R2 = {metrics['val_r2']:.3f}\n")

        val_res_with_fold = val_res_filtered.copy()
        val_res_with_fold["fold"] = fold_name
        cv_normal_list.append(val_res_with_fold)

        if not intervention_res.empty:
            if config["batch_col"] in intervention_res.columns:
                intervention_res_filtered = intervention_res[intervention_res[config["batch_col"]] == val_batch].copy()
                if len(intervention_res_filtered) != len(intervention_res):
                    print(f"  信息: intervention 样本已过滤，只保留 batch={val_batch} 的样本。原始: {len(intervention_res)}, 过滤后: {len(intervention_res_filtered)}")
            else:
                print(f"  警告: intervention 结果中找不到 '{config['batch_col']}' 列，无法按 batch 过滤。")
                intervention_res_filtered = intervention_res.copy()
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
    plt.title("Overall DataSet-LOO Performance (Normal, leave-one-batch-out)\n" f"MAE={overall_mae:.2f}, R²={overall_r2:.2f}", fontsize=14)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age (Out-of-sample)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_performance_cv_normal.png"), dpi=300)
    plt.close()

    if not intervention_all.empty:
        combined = pd.concat([cv_normal_all, intervention_all], ignore_index=True)
    else:
        combined = cv_normal_all.copy()
    combined.to_csv(os.path.join(out_dir, "summary_all_predictions_cv.csv"), index=False)
    plot_age_acceleration(combined, config, os.path.join(out_dir, "summary_age_acceleration.png"), title="Overall Age Acceleration (DataSet-LOO)")

    with open(os.path.join(out_dir, "summary_report_dataset_loo.txt"), "w", encoding="utf-8") as f:
        f.write("=== Final DataSet-LOO Summary (Leave-One-Batch-Out) ===\n\n")
        f.write("Overall Normal (out-of-sample) performance:\n")
        f.write(f"  - MAE: {overall_mae:.3f} years\n")
        f.write(f"  - R2 : {overall_r2:.3f}\n\n")
        f.write("--- Metrics by batch ---\n")
        f.write(metrics_df.to_string(index=False))


def stage_group_loo(data_pkg, config, root_dir):
    """
    阶段 4：按 subgroup 进行 Leave-One-Group-Out CV。
    每个 fold 独立在训练集上运行 RandomizedSearchCV 搜索最优超参数。
    """
    out_dir = os.path.join(root_dir, "group_loo")
    os.makedirs(out_dir, exist_ok=True)

    normal_data = data_pkg["normal_data"]
    intervention_data = data_pkg["intervention_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]

    group_col = config["subgroup_col"]
    if group_col not in normal_data.columns:
        raise ValueError(f"在 normal_data 中找不到 group 列 '{group_col}'（应来自 total_metadata.txt）")

    unique_groups = sorted(normal_data[group_col].unique())
    print("\n步骤 4: Group-LOO (按 subgroup leave-one-group-out, fold-wise HPO)...")
    print(f"  - 使用列: {group_col}（来自 total_metadata.txt）")
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
        print(f"Group-LOO: {fold_name}")
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

        model, fold_best_params = run_rf_hpo_on_fold(
            train_df[current_gene_columns], train_df[config["age_col"]], config, fold_label=fold_name,
        )

        fold_results = evaluate_on_splits(model, split_dict, current_gene_columns, metadata, config)

        joblib.dump(model, os.path.join(fold_dir, f"model_{fold_name}.pkl"))

        with open(os.path.join(fold_dir, f"best_params_{fold_name}.json"), "w", encoding="utf-8") as f:
            json.dump(fold_best_params, f, ensure_ascii=False, indent=2)

        importance_df, _, _ = save_importance_and_results(model, current_gene_columns, fold_results, fold_dir, fold_name)

        train_res = fold_results[fold_results["data_type"] == "train"]
        val_res = fold_results[fold_results["data_type"] == "validation"]
        intervention_res = fold_results[fold_results["data_type"] == "intervention"]

        val_res_filtered = val_res[val_res[group_col] == val_group].copy()
        if len(val_res_filtered) != len(val_res):
            print(f"  警告: validation 样本的 group 不匹配，已过滤。原始: {len(val_res)}, 过滤后: {len(val_res_filtered)}")

        metrics = {
            "fold": fold_name,
            "val_group": val_group,
            "train_mae": mean_absolute_error(train_res["actual_age"], train_res["Predicted_Age"]),
            "train_r2": r2_score(train_res["actual_age"], train_res["Predicted_Age"]),
            "val_mae": mean_absolute_error(val_res_filtered["actual_age"], val_res_filtered["Predicted_Age"]),
            "val_r2": r2_score(val_res_filtered["actual_age"], val_res_filtered["Predicted_Age"]),
            "n_train": len(train_res),
            "n_val": len(val_res_filtered),
        }
        all_fold_metrics.append(metrics)

        plot_performance(train_res, val_res_filtered, metrics, os.path.join(fold_dir, f"performance_{fold_name}.png"), title_prefix=fold_name)
        plot_age_acceleration(fold_results, config, os.path.join(fold_dir, f"age_acceleration_{fold_name}.png"), title=f"Age Acceleration ({fold_name})")

        combat_note = "ComBat+" if is_combat_enabled(config) else ""
        with open(os.path.join(fold_dir, f"report_{fold_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"=== Report for {fold_name} (Group-LOO, {combat_note}fold-wise HPO) ===\n\n")
            f.write(f"Validation group ({group_col}): {val_group}\n")
            f.write("Fold-wise HPO best parameters:\n")
            for k, v in fold_best_params.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nComBat batch correction: {'enabled' if is_combat_enabled(config) else 'disabled'}\n")
            f.write(f"Total genes used (after TMM filtering): {len(current_gene_columns)}\n")
            f.write(f"Number of non-zero features: {len(importance_df)}\n\n")
            f.write(f"Train MAE = {metrics['train_mae']:.3f}, R2 = {metrics['train_r2']:.3f}\n")
            f.write(f"Val   MAE = {metrics['val_mae']:.3f}, R2 = {metrics['val_r2']:.3f}\n")

        val_res_with_fold = val_res_filtered.copy()
        val_res_with_fold["fold"] = fold_name
        cv_normal_list.append(val_res_with_fold)

        if not intervention_res.empty:
            if group_col in intervention_res.columns:
                intervention_res_filtered = intervention_res[intervention_res[group_col] == val_group].copy()
                if len(intervention_res_filtered) != len(intervention_res):
                    print(f"  信息: intervention 样本已过滤，只保留 group={val_group} 的样本。原始: {len(intervention_res)}, 过滤后: {len(intervention_res_filtered)}")
            else:
                print(f"  警告: intervention 结果中找不到 '{group_col}' 列，无法按 group 过滤。")
                intervention_res_filtered = intervention_res.copy()
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
    plt.title("Overall Group-LOO Performance (Normal, leave-one-group-out)\n" f"MAE={overall_mae:.2f}, R²={overall_r2:.2f}", fontsize=14)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age (Out-of-sample)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_performance_cv_normal.png"), dpi=300)
    plt.close()

    if not intervention_all.empty:
        combined = pd.concat([cv_normal_all, intervention_all], ignore_index=True)
    else:
        combined = cv_normal_all.copy()
    combined.to_csv(os.path.join(out_dir, "summary_all_predictions_cv.csv"), index=False)
    plot_age_acceleration(combined, config, os.path.join(out_dir, "summary_age_acceleration.png"), title="Overall Age Acceleration (Group-LOO)")

    with open(os.path.join(out_dir, "summary_report_group_loo.txt"), "w", encoding="utf-8") as f:
        f.write("=== Final Group-LOO Summary (Leave-One-Group-Out by subgroup) ===\n\n")
        f.write("Overall Normal (out-of-sample) performance:\n")
        f.write(f"  - MAE: {overall_mae:.3f} years\n")
        f.write(f"  - R2 : {overall_r2:.3f}\n\n")
        f.write("--- Metrics by group ---\n")
        f.write(metrics_df.to_string(index=False))


def stage_group_prediction_with_best_model(data_pkg, best_model, best_gene_columns, config, root_dir):
    """阶段 5：用 HPO 得到的最优 RF 模型对所有样本预测，并按 group 汇总。"""
    out_dir = os.path.join(root_dir, "group_best_model")
    os.makedirs(out_dir, exist_ok=True)

    all_data = data_pkg["all_data"]
    gene_columns = data_pkg["gene_columns"]
    metadata = data_pkg["metadata"]

    print("\n步骤 5: 使用 HPO 得到的最优模型，对所有样本预测，并按 group 汇总...")

    if is_foldwise_tmm_enabled(config):
        normal_data = data_pkg["normal_data"]
        _, norm_splits, current_gene_columns = normalize_fold_splits_tmm_logcpm(
            train_df=normal_data,
            splits_dict={"all": all_data},
            gene_columns=gene_columns,
            config=config,
            out_dir=os.path.join(out_dir, "foldwise_tmm_logcpm"),
            prefix="best_model_all_samples",
        )
        all_data = norm_splits["all"]
        missing = [gene for gene in best_gene_columns if gene not in all_data.columns]
        if missing:
            raise ValueError(f"best-model 预测数据缺少 HPO 阶段使用的基因: {missing[:10]}")
        gene_columns = best_gene_columns
    else:
        gene_columns = best_gene_columns

    X_all = all_data[gene_columns].copy()
    if X_all.isnull().any().any():
        X_all = X_all.fillna(X_all.mean())
    y_all = all_data[config["age_col"]]

    y_pred_all = best_model.predict(X_all)

    res = pd.DataFrame({"actual_age": y_all, "Predicted_Age": y_pred_all}, index=all_data.index)
    res = pd.merge(
        res.reset_index().rename(columns={"index": "sample_id"}),
        metadata, on="sample_id", how="left",
    )
    res["age_acceleration"] = res["Predicted_Age"] - res["actual_age"]

    res.to_csv(os.path.join(out_dir, "all_samples_predictions_best_model.csv"), index=False)

    importance_df = pd.DataFrame({"gene": gene_columns, "importance": best_model.feature_importances_})
    importance_df = importance_df[importance_df["importance"] > 0].sort_values("importance", ascending=False)
    importance_df.to_csv(os.path.join(out_dir, "gene_importances_best_model.csv"), index=False)

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
    group_summary.to_csv(os.path.join(out_dir, "group_age_summary.csv"), index=False)

    plt.figure(figsize=(12, 7))
    x = np.arange(len(group_summary))
    width = 0.35
    plt.bar(x - width / 2, group_summary["mean_actual_age"], width, label="Actual")
    plt.bar(x + width / 2, group_summary["mean_Predicted_Age"], width, label="Predicted")
    plt.xticks(x, group_summary[group_col], rotation=45, ha="right")
    plt.ylabel("Age (years)")
    plt.title("Mean Actual & Predicted Age per Group (Best RF Model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "group_mean_age_barplot.png"), dpi=300)
    plt.close()

    plot_age_acceleration(
        res, config,
        os.path.join(out_dir, "age_acceleration_by_group_best_model.png"),
        title="Age Acceleration by Group (Best RF Model on All Data)",
    )

    with open(os.path.join(out_dir, "report_group_best_model.txt"), "w", encoding="utf-8") as f:
        f.write("=== Group-level Predictions with Best RF Model ===\n\n")
        f.write("该模型来自 HPO 阶段，在所有 Normal 样本上用 RandomizedSearchCV 训练，然后用于全体样本的预测。\n\n")
        oob_score = getattr(best_model, 'oob_score_', None)
        if oob_score is not None:
            f.write(f"Out-of-Bag (OOB) R2 Score: {oob_score:.4f}\n")
        f.write(f"Number of selected (non-zero) features: {len(importance_df)}\n")
        f.write("\nTop 10 genes:\n")
        f.write(importance_df.head(10).to_string(index=False))
        f.write("\n\n--- Group-wise summary ---\n")
        f.write(group_summary.to_string(index=False))


if __name__ == "__main__":
    root_dir = resolve_path(CONFIG["output_root_dir"])
    os.makedirs(root_dir, exist_ok=True)
    print(f"输出根目录: {root_dir}")

    data_pkg = load_full_data(CONFIG)

    best_params, best_model, best_gene_columns = stage_hpo_and_save(
        data_pkg["normal_data"],
        data_pkg["gene_columns"],
        data_pkg["metadata"],
        CONFIG,
        root_dir,
    )

    stage_kfold_cv(data_pkg, CONFIG, root_dir)

    stage_dataset_loo(data_pkg, CONFIG, root_dir)

    stage_group_loo(data_pkg, CONFIG, root_dir)

    stage_group_prediction_with_best_model(data_pkg, best_model, best_gene_columns, CONFIG, root_dir)

    print("\n✅ 全部流程完成！结果请查看输出根目录下的各个子文件夹。")
