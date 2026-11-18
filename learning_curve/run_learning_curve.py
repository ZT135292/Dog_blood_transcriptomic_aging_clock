#!/usr/bin/env python3
"""
Learning-curve / bootstrap-subsampling analysis for the canine blood clocks.

Question:
  On the current 89 Normal samples, does held-out prediction error keep falling
  as n_train grows (more data would help), or has it already approached a
  plateau set by biological / technical heterogeneity?

Design (leak-free, matches the revised pipeline)
  * Only Normal samples are used (n = 89).
  * Each repeat: stratified hold-out of a fixed test set (~20%).
  * Nested, batch-stratified training subsets of increasing size.
  * For every (repeat, n_train): TMM + logCPM + filterByExpr are fit on that
    training subset only; the test set is transformed with the training
    reference. Hyperparameters are tuned only on the training subset.
  * The same normalised matrices are reused across models within a cell.

Default models: LASSO, Elastic Net, SVR. RF / XGBoost / MoE can be added with
--models (see run_learning_curve_remaining.py and run_lc_<model>.py).

Example
  python run_learning_curve.py \\
      --metadata ../model_training/total_metadata.txt \\
      --counts   ../model_training/total_count_sort_ortholog.txt \\
      --outdir   ./results \\
      --models   lasso,elasticnet,svr \\
      --n-repeats 10 \\
      --resume

  # after the run, or to rebuild figures from an existing CSV
  python run_learning_curve.py --plot-only --outdir ./results
"""

from __future__ import annotations

import argparse
import sys
import json
import os
import shutil
import subprocess
import tempfile
import traceback
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Defaults (same as the leak-free training scripts)
# ---------------------------------------------------------------------------

DEFAULT_METADATA = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "model_training", "total_metadata.txt")
)
DEFAULT_COUNTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "model_training", "total_count_sort_ortholog.txt")
)
DEFAULT_AGING_GENES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "model_training", "aging_gene_list.csv")
)

SAMPLE_ID_COL = "sample_id"
GROUP_COL = "condition"
AGE_COL = "Chronological_Age"
BATCH_COL = "batch"
NORMAL_LABEL = "Normal"

L1_RATIOS = [0.01, 0.05, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
EN_MAX_ITER = 100000
LASSO_MAX_ITER = 10000
RANDOM_STATE = 66

SVR_PARAM_GRID = {
    "C": [0.1, 1.0, 10.0],
    "kernel": ["linear"],
    "epsilon": [0.1, 0.5],
}

RF_PARAM_DIST = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [None, 8, 12, 16, 20, 30],
    "min_samples_leaf": [1, 2, 4, 8],
    "min_samples_split": [2, 4, 8, 16],
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "bootstrap": [True],
}

XGB_PARAM_DIST = {
    "n_estimators": [100, 200, 400, 600, 800],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],
    "gamma": [0, 0.1, 0.2, 0.5, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}

MODEL_ORDER = ["lasso", "elasticnet", "svr", "rf", "xgboost", "moe"]
MODEL_LABELS = {
    "lasso": "LASSO",
    "elasticnet": "Elastic Net",
    "svr": "SVR",
    "rf": "Random Forest",
    "xgboost": "XGBoost",
    "moe": "MoE",
}
MODEL_COLORS = {
    "lasso": "#4C72B0",
    "elasticnet": "#DD8452",
    "svr": "#55A868",
    "rf": "#C44E52",
    "xgboost": "#8172B3",
    "moe": "#937860",
}

RUN_CSV_NAME = "run_records.csv"
SUMMARY_CSV_NAME = "summary_by_size.csv"
CONFIG_JSON_NAME = "run_config.json"


# ===========================================================================
# Fold-wise TMM + logCPM (copied from the leak-free ElasticNet pipeline)
# ===========================================================================

def run_edger_tmm_logcpm(counts_df, out_dir, prefix, rscript, prior_count=1, filter_low_expression=True):
    if shutil.which(rscript) is None:
        raise RuntimeError(
            f"Cannot find Rscript: {rscript}. Put it on PATH or pass --rscript."
        )

    counts_df = counts_df.copy()
    counts_df.index = counts_df.index.astype(str)
    counts_df.columns = counts_df.columns.astype(str)

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
if ({str(bool(filter_low_expression)).upper()}) {{
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
        with open(r_path, "w", encoding="utf-8") as handle:
            handle.write(r_code)
        completed = subprocess.run(
            [rscript, r_path], capture_output=True, text=True, cwd=out_dir
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "edgeR TMM/logCPM failed.\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        logcpm = pd.read_csv(out_path, sep="\t", index_col=0)
        kept = pd.read_csv(genes_path, sep="\t")["gene"].astype(str).tolist()
    return logcpm, kept


def run_edger_logcpm_with_train_ref(
    counts_df, train_ref_counts_df, out_dir, prefix, rscript, prior_count=1, keep_genes=None
):
    counts_df = counts_df.copy()
    train_ref_counts_df = train_ref_counts_df.copy()
    counts_df.index = counts_df.index.astype(str)
    counts_df.columns = counts_df.columns.astype(str)
    train_ref_counts_df.index = train_ref_counts_df.index.astype(str)
    train_ref_counts_df.columns = train_ref_counts_df.columns.astype(str)

    if keep_genes is not None:
        keep_genes = [
            gene
            for gene in keep_genes
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
        with open(r_path, "w", encoding="utf-8") as handle:
            handle.write(r_code)
        completed = subprocess.run(
            [rscript, r_path], capture_output=True, text=True, cwd=out_dir
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "edgeR logCPM (training-reference TMM) failed.\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        logcpm = pd.read_csv(out_path, sep="\t", index_col=0)
        logcpm.columns = logcpm.columns.astype(str)
    return logcpm


def normalize_train_test(
    train_counts,
    test_counts,
    gene_columns,
    work_dir,
    prefix,
    rscript,
    prior_count=1,
    filter_low_expression=True,
):
    """Fit TMM + filterByExpr on train; transform test with the training reference."""
    train_mat = train_counts[gene_columns].T
    train_logcpm, kept_genes = run_edger_tmm_logcpm(
        train_mat,
        out_dir=work_dir,
        prefix=f"{prefix}_train",
        rscript=rscript,
        prior_count=prior_count,
        filter_low_expression=filter_low_expression,
    )
    train_counts_kept = train_mat.loc[[gene for gene in kept_genes if gene in train_mat.index]]
    test_mat = test_counts[gene_columns].T
    test_logcpm = run_edger_logcpm_with_train_ref(
        test_mat,
        train_ref_counts_df=train_counts_kept,
        out_dir=work_dir,
        prefix=f"{prefix}_test",
        rscript=rscript,
        prior_count=prior_count,
        keep_genes=kept_genes,
    )
    x_train = train_logcpm.T
    x_test = test_logcpm.T.reindex(columns=x_train.columns)
    return x_train, x_test, list(x_train.columns)


# ===========================================================================
# Data I/O and splitting
# ===========================================================================

def load_normal_data(metadata_path: str, counts_path: str) -> Tuple[pd.DataFrame, List[str]]:
    metadata = pd.read_csv(metadata_path, sep="\t")
    expr = pd.read_csv(counts_path, sep="\t", index_col=0)
    expr.index = expr.index.astype(str)
    expr.columns = expr.columns.astype(str)
    metadata[SAMPLE_ID_COL] = metadata[SAMPLE_ID_COL].astype(str)

    missing = sorted(set(metadata[SAMPLE_ID_COL]) - set(expr.columns))
    if missing:
        raise ValueError(f"Count matrix is missing metadata samples: {missing[:10]}")
    expr = expr.loc[:, metadata[SAMPLE_ID_COL]]

    full = pd.merge(
        metadata.set_index(SAMPLE_ID_COL),
        expr.T,
        left_index=True,
        right_index=True,
    )
    normal = full[full[GROUP_COL] == NORMAL_LABEL].copy()
    gene_columns = expr.index.tolist()
    print(f"Loaded Normal samples: n={len(normal)}, genes={len(gene_columns)}")
    print("Batch counts:")
    print(normal[BATCH_COL].astype(str).value_counts().sort_index().to_string())
    return normal, gene_columns


def age_tertile(ages: pd.Series) -> pd.Series:
    return pd.qcut(pd.to_numeric(ages, errors="coerce"), q=3, labels=False, duplicates="drop")


def make_strata(df: pd.DataFrame, min_batch_for_split: int = 5) -> pd.Series:
    """Batch + age tertile for large batches; 'small' + age tertile otherwise."""
    batch = df[BATCH_COL].astype(str)
    counts = batch.value_counts()
    batch_group = batch.where(batch.map(counts) >= min_batch_for_split, "small")
    bins = age_tertile(df[AGE_COL]).fillna(0).astype(int).astype(str)
    return batch_group.astype(str) + "|age" + bins


def stratified_holdout(
    index: pd.Index,
    strata: pd.Series,
    test_frac: float,
    rng: np.random.Generator,
    min_stratum_for_test: int = 5,
) -> Tuple[pd.Index, pd.Index]:
    """Hold out ~test_frac from strata that are large enough; keep tiny strata in train."""
    train_ids: List[str] = []
    test_ids: List[str] = []
    strata = strata.loc[index]
    for key, members in strata.groupby(strata, sort=True):
        ids = members.index.to_numpy()
        rng.shuffle(ids)
        if len(ids) < min_stratum_for_test:
            train_ids.extend(ids.tolist())
            continue
        n_test = max(1, int(round(len(ids) * test_frac)))
        n_test = min(n_test, len(ids) - 1)
        test_ids.extend(ids[:n_test].tolist())
        train_ids.extend(ids[n_test:].tolist())
    if len(test_ids) < 8:
        raise RuntimeError(
            f"Held-out test set is too small (n={len(test_ids)}). "
            "Check batch composition or lower --min-stratum-for-test."
        )
    return pd.Index(train_ids), pd.Index(test_ids)


def nested_stratified_subsets(
    train_index: pd.Index,
    strata: pd.Series,
    sizes: Sequence[int],
    rng: np.random.Generator,
) -> Dict[int, pd.Index]:
    """
    Nested subsets: each larger set contains the previous one.
    Allocation is proportional to stratum size, then trimmed/padded to exact n.
    """
    strata = strata.loc[train_index]
    shuffled: Dict[str, np.ndarray] = {}
    for key, members in strata.groupby(strata, sort=True):
        ids = members.index.to_numpy()
        rng.shuffle(ids)
        shuffled[str(key)] = ids

    n_pool = len(train_index)
    out: Dict[int, pd.Index] = {}
    for n in sizes:
        n = int(min(max(n, 1), n_pool))
        chosen: List[str] = []
        for key, ids in shuffled.items():
            take = int(round(n * len(ids) / n_pool))
            take = min(max(take, 0), len(ids))
            chosen.extend(ids[:take].tolist())
        # adjust to exact n, preferring leftover IDs in the shuffled order
        leftover = [i for ids in shuffled.values() for i in ids if i not in set(chosen)]
        if len(chosen) < n:
            chosen.extend(leftover[: n - len(chosen)])
        elif len(chosen) > n:
            chosen = chosen[:n]
        out[n] = pd.Index(chosen)
    return out


def bootstrap_indices(index: pd.Index, n: int, rng: np.random.Generator) -> pd.Index:
    drawn = rng.choice(index.to_numpy(), size=n, replace=True)
    # TMM cannot use duplicate columns; keep unique IDs in draw order, then
    # top up without replacement if collapse is too severe.
    unique = list(dict.fromkeys(drawn.tolist()))
    if len(unique) < max(10, int(0.6 * n)):
        leftover = [i for i in index if i not in unique]
        rng.shuffle(leftover)
        unique.extend(leftover[: max(0, n - len(unique))])
    return pd.Index(unique[:n] if len(unique) >= n else unique)


# ===========================================================================
# Model training
# ===========================================================================

def _fill_nan(x: pd.DataFrame) -> pd.DataFrame:
    if x.isnull().any().any():
        x = x.fillna(x.mean())
    return x


def _scale(train_x: pd.DataFrame, test_x: pd.DataFrame):
    scaler = StandardScaler()
    xtr = scaler.fit_transform(_fill_nan(train_x))
    xte = scaler.transform(_fill_nan(test_x))
    if np.isnan(xtr).any():
        xtr = np.nan_to_num(xtr, nan=0.0)
    if np.isnan(xte).any():
        xte = np.nan_to_num(xte, nan=0.0)
    return xtr, xte, scaler


def inner_cv_folds(n_train: int, requested: int) -> int:
    """Shrink inner CV when the training subset is small."""
    if n_train < 15:
        return 3
    if n_train < 25:
        return min(4, requested)
    return requested


def train_lasso(x_train, y_train, x_test, inner_cv, n_jobs, random_state, extra=None):
    xtr, xte, _ = _scale(x_train, x_test)
    model = LassoCV(
        cv=inner_cv,
        max_iter=LASSO_MAX_ITER,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(xtr, y_train)
    return model.predict(xtr), model.predict(xte), {"alpha": float(model.alpha_)}


def train_elasticnet(x_train, y_train, x_test, inner_cv, n_jobs, random_state, extra=None):
    xtr, xte, _ = _scale(x_train, x_test)
    model = ElasticNetCV(
        l1_ratio=L1_RATIOS,
        cv=inner_cv,
        max_iter=EN_MAX_ITER,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(xtr, y_train)
    return (
        model.predict(xtr),
        model.predict(xte),
        {"alpha": float(model.alpha_), "l1_ratio": float(model.l1_ratio_)},
    )


def train_svr(x_train, y_train, x_test, inner_cv, n_jobs, random_state, extra=None):
    del random_state, extra
    xtr, xte, _ = _scale(x_train, x_test)
    search = GridSearchCV(
        SVR(),
        param_grid=SVR_PARAM_GRID,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=n_jobs,
        verbose=0,
    )
    search.fit(xtr, y_train)
    return search.predict(xtr), search.predict(xte), dict(search.best_params_)


def train_rf(x_train, y_train, x_test, inner_cv, n_jobs, random_state, extra=None):
    from sklearn.ensemble import RandomForestRegressor

    xtr = _fill_nan(x_train).values
    xte = _fill_nan(x_test).reindex(columns=x_train.columns).values
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=random_state, n_jobs=n_jobs),
        param_distributions=RF_PARAM_DIST,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        random_state=42,
        n_jobs=n_jobs,
        verbose=0,
    )
    search.fit(xtr, y_train)
    return search.predict(xtr), search.predict(xte), dict(search.best_params_)


def train_xgboost(x_train, y_train, x_test, inner_cv, n_jobs, random_state, extra=None):
    import xgboost as xgb

    xtr = _fill_nan(x_train)
    xte = _fill_nan(x_test).reindex(columns=x_train.columns)
    base = xgb.XGBRegressor(
        tree_method="hist",
        device="cpu",
        verbosity=0,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=XGB_PARAM_DIST,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        random_state=42,
        n_jobs=n_jobs,
        verbose=0,
    )
    search.fit(xtr, y_train)
    return search.predict(xtr), search.predict(xte), dict(search.best_params_)


def train_moe(x_train, y_train, x_test, inner_cv, n_jobs, random_state, extra=None):
    del inner_cv, n_jobs, random_state
    if extra is None or "aging_gene_pool" not in extra or "moe_config" not in extra:
        raise ValueError("MoE trainer needs extra['aging_gene_pool'] and extra['moe_config'].")
    from moe_trainer import fit_predict_moe

    return fit_predict_moe(x_train, y_train, x_test, extra)


TRAINERS = {
    "lasso": train_lasso,
    "elasticnet": train_elasticnet,
    "svr": train_svr,
    "rf": train_rf,
    "xgboost": train_xgboost,
    "moe": train_moe,
}


def metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson_r": np.nan}
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson_r": pearson}


# ===========================================================================
# Checkpoint I/O
# ===========================================================================

def run_key(row: dict) -> Tuple:
    n_req = row.get("n_train_requested", row.get("n_train"))
    return (row["mode"], int(row["repeat"]), int(n_req), row["model"])


def load_completed(path: str) -> set:
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    if "test_mae" in df.columns:
        df = df[pd.to_numeric(df["test_mae"], errors="coerce").notna()]
    return {run_key(rec) for rec in df.to_dict(orient="records")}


def append_row(path: str, row: dict) -> None:
    frame = pd.DataFrame([row])
    header = not os.path.exists(path)
    frame.to_csv(path, mode="a", header=header, index=False)


# ===========================================================================
# Main analysis
# ===========================================================================

def parse_models(text: str) -> List[str]:
    models = [m.strip().lower() for m in text.split(",") if m.strip()]
    unknown = [m for m in models if m not in TRAINERS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Choose from {list(TRAINERS)}")
    return models


def load_aging_gene_pool(path: str) -> set:
    df = pd.read_csv(path)
    pool = set(df.iloc[:, 0].astype(str))
    print(f"Aging-gene list: {path} ({len(pool)} symbols)")
    return pool


def build_trainer_extra(args, models: Sequence[str]) -> dict:
    extra = {}
    if "moe" in models:
        from moe_trainer import default_moe_config

        extra["aging_gene_pool"] = load_aging_gene_pool(args.aging_genes)
        extra["moe_config"] = default_moe_config(args)
        print("MoE config:", json.dumps(extra["moe_config"], indent=2, default=str))
    return extra


def planned_sizes(train_pool_n: int, requested: Sequence[int]) -> List[int]:
    sizes = sorted({int(s) for s in requested if 1 <= int(s) < train_pool_n})
    if train_pool_n not in sizes:
        sizes.append(int(train_pool_n))
    return sizes


def run_learning_curve(args) -> pd.DataFrame:
    normal, gene_columns = load_normal_data(args.metadata, args.counts)
    strata = make_strata(normal, min_batch_for_split=args.min_batch_for_split)
    models = parse_models(args.models)
    trainer_extra = build_trainer_extra(args, models)

    os.makedirs(args.outdir, exist_ok=True)
    work_root = os.path.join(args.outdir, "_tmm_tmp")
    os.makedirs(work_root, exist_ok=True)

    run_csv = os.path.join(args.outdir, RUN_CSV_NAME)
    completed = load_completed(run_csv) if args.resume else set()
    if not args.resume and os.path.exists(run_csv):
        raise FileExistsError(
            f"{run_csv} already exists. Pass --resume to continue, or choose a new --outdir."
        )

    config = vars(args).copy()
    config["n_normal"] = int(len(normal))
    config["n_genes_raw"] = int(len(gene_columns))
    with open(os.path.join(args.outdir, CONFIG_JSON_NAME), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    master_rng = np.random.default_rng(args.seed)
    repeat_seeds = master_rng.integers(0, 10**9, size=args.n_repeats)

    for rep, seed in enumerate(repeat_seeds, start=1):
        rng = np.random.default_rng(int(seed))
        train_ids, test_ids = stratified_holdout(
            normal.index,
            strata,
            test_frac=args.test_size,
            rng=rng,
            min_stratum_for_test=args.min_stratum_for_test,
        )
        train_df = normal.loc[train_ids]
        test_df = normal.loc[test_ids]
        sizes = planned_sizes(len(train_ids), args.train_sizes)
        subsets = nested_stratified_subsets(train_ids, strata, sizes, rng)

        print("\n" + "=" * 72)
        print(
            f"Repeat {rep}/{args.n_repeats}  |  train pool={len(train_ids)}  "
            f"test={len(test_ids)}  sizes={sizes}"
        )
        print(
            "  test batches: "
            + ", ".join(
                f"{k}={v}"
                for k, v in test_df[BATCH_COL].astype(str).value_counts().sort_index().items()
            )
        )
        print("=" * 72)

        for n_train in sizes:
            sub_ids = subsets[n_train]
            # skip a whole (repeat, n) cell only if every model is already done
            pending = [
                m
                for m in models
                if (args.mode, rep, n_train, m) not in completed
            ]
            if not pending:
                print(f"  n={n_train}: all models already finished, skip.")
                continue

            print(f"\n  -- n_train={n_train} (pending: {pending}) --")
            prefix = f"rep{rep:02d}_n{n_train}"
            cell_dir = os.path.join(work_root, prefix)
            os.makedirs(cell_dir, exist_ok=True)
            try:
                x_train, x_test, kept = normalize_train_test(
                    train_counts=normal.loc[sub_ids],
                    test_counts=test_df,
                    gene_columns=gene_columns,
                    work_dir=cell_dir,
                    prefix=prefix,
                    rscript=args.rscript,
                    filter_low_expression=not args.no_filter,
                )
            except Exception as exc:
                print(f"  TMM failed for {prefix}: {exc}")
                traceback.print_exc()
                continue

            y_train = normal.loc[x_train.index, AGE_COL].astype(float)
            y_test = normal.loc[x_test.index, AGE_COL].astype(float)
            inner_cv = inner_cv_folds(len(x_train), args.inner_cv)
            print(f"     genes after filterByExpr: {len(kept)}; inner CV folds: {inner_cv}")

            for model_name in pending:
                print(f"     training {MODEL_LABELS[model_name]} ...", flush=True)
                trainer = TRAINERS[model_name]
                try:
                    pred_tr, pred_te, params = trainer(
                        x_train,
                        y_train,
                        x_test,
                        inner_cv=inner_cv,
                        n_jobs=args.n_jobs,
                        random_state=args.seed,
                        extra=trainer_extra,
                    )
                    tr = metrics(y_train.values, pred_tr)
                    te = metrics(y_test.values, pred_te)
                    size_label = "all" if n_train == len(train_ids) else str(int(n_train))
                    row = {
                        "mode": args.mode,
                        "repeat": rep,
                        "repeat_seed": int(seed),
                        "n_train": int(len(x_train)),
                        "n_test": int(len(x_test)),
                        "n_train_requested": int(n_train),
                        "size_label": size_label,
                        "n_genes": int(len(kept)),
                        "inner_cv": int(inner_cv),
                        "model": model_name,
                        "train_mae": tr["mae"],
                        "train_rmse": tr["rmse"],
                        "train_r2": tr["r2"],
                        "train_pearson_r": tr["pearson_r"],
                        "test_mae": te["mae"],
                        "test_rmse": te["rmse"],
                        "test_r2": te["r2"],
                        "test_pearson_r": te["pearson_r"],
                        "best_params": json.dumps(params, default=str),
                    }
                    append_row(run_csv, row)
                    completed.add(run_key(row))
                    print(
                        f"       test MAE={te['mae']:.3f}  R2={te['r2']:.3f}  "
                        f"train MAE={tr['mae']:.3f}"
                    )
                except Exception as exc:
                    print(f"       {model_name} failed: {exc}")
                    traceback.print_exc()
                    fail_row = {
                        "mode": args.mode,
                        "repeat": rep,
                        "repeat_seed": int(seed),
                        "n_train": int(len(x_train)),
                        "n_test": int(len(x_test)),
                        "n_train_requested": int(n_train),
                        "size_label": "all" if n_train == len(train_ids) else str(int(n_train)),
                        "n_genes": int(len(kept)),
                        "inner_cv": int(inner_cv),
                        "model": model_name,
                        "train_mae": np.nan,
                        "train_rmse": np.nan,
                        "train_r2": np.nan,
                        "train_pearson_r": np.nan,
                        "test_mae": np.nan,
                        "test_rmse": np.nan,
                        "test_r2": np.nan,
                        "test_pearson_r": np.nan,
                        "best_params": json.dumps({"error": str(exc)}),
                    }
                    append_row(run_csv, fail_row)

            shutil.rmtree(cell_dir, ignore_errors=True)

    if args.cleanup_tmp:
        shutil.rmtree(work_root, ignore_errors=True)
    return pd.read_csv(run_csv)


def run_bootstrap(args) -> pd.DataFrame:
    """
    Bootstrap / subsampling mode: for each size, draw a subset of the 89
    Normal samples (with replacement, then uniquified) and evaluate by an
    internal stratified hold-out of that subset. The default learning-curve
    mode is preferred.
    """
    normal, gene_columns = load_normal_data(args.metadata, args.counts)
    strata = make_strata(normal, min_batch_for_split=args.min_batch_for_split)
    models = parse_models(args.models)
    trainer_extra = build_trainer_extra(args, models)
    os.makedirs(args.outdir, exist_ok=True)
    work_root = os.path.join(args.outdir, "_tmm_tmp")
    os.makedirs(work_root, exist_ok=True)

    run_csv = os.path.join(args.outdir, RUN_CSV_NAME)
    completed = load_completed(run_csv) if args.resume else set()
    if not args.resume and os.path.exists(run_csv):
        raise FileExistsError(
            f"{run_csv} already exists. Pass --resume to continue, or choose a new --outdir."
        )

    sizes = sorted({int(s) for s in args.train_sizes if 15 <= int(s) <= len(normal)})
    if len(normal) not in sizes:
        sizes.append(int(len(normal)))

    master_rng = np.random.default_rng(args.seed)
    repeat_seeds = master_rng.integers(0, 10**9, size=args.n_repeats)
    for rep, seed in enumerate(repeat_seeds, start=1):
        rng = np.random.default_rng(int(seed))
        print(f"\n===== Bootstrap repeat {rep}/{args.n_repeats} =====")
        for n_train in sizes:
            drawn = bootstrap_indices(normal.index, n_train, rng)
            # hold out ~20% of the drawn unique IDs as test
            try:
                tr_ids, te_ids = stratified_holdout(
                    drawn, strata, args.test_size, rng, args.min_stratum_for_test
                )
            except RuntimeError:
                # fall back to a simple random split if strata collapse
                ids = drawn.to_numpy().copy()
                rng.shuffle(ids)
                n_te = max(8, int(round(len(ids) * args.test_size)))
                te_ids = pd.Index(ids[:n_te])
                tr_ids = pd.Index(ids[n_te:])

            pending = [
                m for m in models if (args.mode, rep, n_train, m) not in completed
            ]
            if not pending:
                continue

            prefix = f"boot_rep{rep:02d}_n{n_train}"
            cell_dir = os.path.join(work_root, prefix)
            os.makedirs(cell_dir, exist_ok=True)
            try:
                x_train, x_test, kept = normalize_train_test(
                    train_counts=normal.loc[tr_ids],
                    test_counts=normal.loc[te_ids],
                    gene_columns=gene_columns,
                    work_dir=cell_dir,
                    prefix=prefix,
                    rscript=args.rscript,
                    filter_low_expression=not args.no_filter,
                )
            except Exception as exc:
                print(f"  TMM failed for {prefix}: {exc}")
                traceback.print_exc()
                continue

            y_train = normal.loc[x_train.index, AGE_COL].astype(float)
            y_test = normal.loc[x_test.index, AGE_COL].astype(float)
            inner_cv = inner_cv_folds(len(x_train), args.inner_cv)
            for model_name in pending:
                print(f"  n={n_train}  {model_name} ...", flush=True)
                try:
                    pred_tr, pred_te, params = TRAINERS[model_name](
                        x_train,
                        y_train,
                        x_test,
                        inner_cv=inner_cv,
                        n_jobs=args.n_jobs,
                        random_state=args.seed,
                        extra=trainer_extra,
                    )
                    tr = metrics(y_train.values, pred_tr)
                    te = metrics(y_test.values, pred_te)
                    row = {
                        "mode": args.mode,
                        "repeat": rep,
                        "repeat_seed": int(seed),
                        "n_train": int(len(x_train)),
                        "n_test": int(len(x_test)),
                        "n_train_requested": int(n_train),
                        "size_label": str(int(n_train)),
                        "n_genes": int(len(kept)),
                        "inner_cv": int(inner_cv),
                        "model": model_name,
                        "train_mae": tr["mae"],
                        "train_rmse": tr["rmse"],
                        "train_r2": tr["r2"],
                        "train_pearson_r": tr["pearson_r"],
                        "test_mae": te["mae"],
                        "test_rmse": te["rmse"],
                        "test_r2": te["r2"],
                        "test_pearson_r": te["pearson_r"],
                        "best_params": json.dumps(params, default=str),
                    }
                    append_row(run_csv, row)
                    completed.add(run_key(row))
                    print(f"    test MAE={te['mae']:.3f}  R2={te['r2']:.3f}")
                except Exception as exc:
                    print(f"    {model_name} failed: {exc}")
                    traceback.print_exc()
            shutil.rmtree(cell_dir, ignore_errors=True)

    if args.cleanup_tmp:
        shutil.rmtree(work_root, ignore_errors=True)
    return pd.read_csv(run_csv)


# ===========================================================================
# Summaries, saturation fit, figures
# ===========================================================================

def summarise(run_df: pd.DataFrame) -> pd.DataFrame:
    clean = run_df.dropna(subset=["test_mae"]).copy()
    if "size_label" not in clean.columns:
        clean["size_label"] = clean["n_train"].astype(int).astype(str)
    rows = []
    for (model, size_label), sub in clean.groupby(["model", "size_label"], sort=False):
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "size_label": str(size_label),
                "n_train": float(sub["n_train"].mean()),
                "n_repeats": int(len(sub)),
                "n_genes_mean": float(sub["n_genes"].mean()),
                "test_mae_mean": float(sub["test_mae"].mean()),
                "test_mae_sd": float(sub["test_mae"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "test_mae_median": float(sub["test_mae"].median()),
                "test_r2_mean": float(sub["test_r2"].mean()),
                "test_r2_sd": float(sub["test_r2"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "test_pearson_mean": float(sub["test_pearson_r"].mean()),
                "train_mae_mean": float(sub["train_mae"].mean()),
                "train_r2_mean": float(sub["train_r2"].mean()),
                "gap_mae_mean": float((sub["test_mae"] - sub["train_mae"]).mean()),
            }
        )
    out = pd.DataFrame(rows)

    def _size_key(label: str) -> float:
        if label == "all":
            return 1e9
        try:
            return float(label)
        except ValueError:
            return 0.0

    out["_ord"] = out["size_label"].map(_size_key)
    return out.sort_values(["model", "_ord"]).drop(columns="_ord").reset_index(drop=True)


def fit_asymptote(summary: pd.DataFrame) -> pd.DataFrame:
    """Fit test MAE ≈ a + b / n for each model. a is the estimated error floor."""

    def _fit(n, mae):
        n = np.asarray(n, dtype=float)
        mae = np.asarray(mae, dtype=float)
        if len(n) < 3:
            return np.nan, np.nan, np.nan
        try:
            from scipy.optimize import curve_fit

            def f(x, a, b):
                return a + b / np.maximum(x, 1.0)

            popt, _ = curve_fit(
                f, n, mae, p0=(float(np.min(mae)), 20.0), bounds=([0.0, -50.0], [20.0, 200.0])
            )
            pred = f(n, *popt)
            ss_res = float(np.sum((mae - pred) ** 2))
            ss_tot = float(np.sum((mae - mae.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            return float(popt[0]), float(popt[1]), r2
        except Exception:
            # linear least squares on MAE vs 1/n
            x = 1.0 / n
            A = np.column_stack([np.ones_like(x), x])
            coef, _, _, _ = np.linalg.lstsq(A, mae, rcond=None)
            pred = A @ coef
            ss_res = float(np.sum((mae - pred) ** 2))
            ss_tot = float(np.sum((mae - mae.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            return float(coef[0]), float(coef[1]), r2

    rows = []
    for model, sub in summary.groupby("model", sort=True):
        a, b, r2 = _fit(sub["n_train"].values, sub["test_mae_mean"].values)
        n_max = int(sub["n_train"].max())
        mae_max = float(sub.loc[sub["n_train"].idxmax(), "test_mae_mean"])
        n_min = int(sub["n_train"].min())
        mae_min = float(sub.loc[sub["n_train"].idxmin(), "test_mae_mean"])
        drop = mae_min - mae_max
        # last step: second-largest n vs largest n
        ordered = sub.sort_values("n_train")
        if len(ordered) >= 2:
            last_drop = float(ordered["test_mae_mean"].iloc[-2] - ordered["test_mae_mean"].iloc[-1])
        else:
            last_drop = np.nan
        room_to_floor = mae_max - a if pd.notna(a) else np.nan
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "n_min": n_min,
                "n_max": n_max,
                "mae_at_n_min": mae_min,
                "mae_at_n_max": mae_max,
                "mae_drop_total": drop,
                "mae_drop_last_step": last_drop,
                "asymptote_a": a,
                "slope_b": b,
                "fit_r2": r2,
                "room_to_floor": room_to_floor,
                "near_ceiling": bool(pd.notna(room_to_floor) and room_to_floor < 0.15),
            }
        )
    return pd.DataFrame(rows)


def plot_curves(summary: pd.DataFrame, outdir: str, mode: str) -> None:
    if summary.empty:
        print("No summary rows to plot.")
        return

    def _one(metric_mean, metric_sd, ylabel, title, fname, hline=None):
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for model in MODEL_ORDER:
            sub = summary[summary["model"] == model]
            if sub.empty:
                continue
            ax.errorbar(
                sub["n_train"],
                sub[metric_mean],
                yerr=sub[metric_sd],
                marker="o",
                markersize=6,
                linewidth=1.8,
                capsize=3,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        if hline is not None:
            ax.axhline(hline, color="0.4", linestyle="--", linewidth=1, zorder=0)
        ax.set_xlabel("Number of training samples", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(frameon=False, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=300)
        fig.savefig(os.path.splitext(path)[0] + ".pdf")
        plt.close(fig)
        print(f"Wrote {path}")

    scheme = "Learning curve" if mode == "learning_curve" else "Bootstrap subsampling"
    _one(
        "test_mae_mean",
        "test_mae_sd",
        "Held-out MAE (years)",
        f"{scheme}: held-out MAE vs training size",
        "learning_curve_mae.png",
    )
    _one(
        "test_r2_mean",
        "test_r2_sd",
        "Held-out $R^2$",
        f"{scheme}: held-out $R^2$ vs training size",
        "learning_curve_r2.png",
    )

    # train vs test MAE (overfitting diagnostic)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for model in MODEL_ORDER:
        sub = summary[summary["model"] == model]
        if sub.empty:
            continue
        ax.plot(
            sub["n_train"],
            sub["train_mae_mean"],
            linestyle="--",
            marker="s",
            markersize=5,
            color=MODEL_COLORS[model],
            alpha=0.7,
            label=f"{MODEL_LABELS[model]} train",
        )
        ax.plot(
            sub["n_train"],
            sub["test_mae_mean"],
            linestyle="-",
            marker="o",
            markersize=6,
            color=MODEL_COLORS[model],
            label=f"{MODEL_LABELS[model]} test",
        )
    ax.set_xlabel("Number of training samples", fontsize=12)
    ax.set_ylabel("MAE (years)", fontsize=12)
    ax.set_title("Train vs held-out MAE", fontsize=13)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(outdir, "learning_curve_train_vs_test_mae.png")
    fig.savefig(path, dpi=300)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)
    print(f"Wrote {path}")


def write_report(summary: pd.DataFrame, fit: pd.DataFrame, outdir: str, mode: str) -> None:
    path = os.path.join(outdir, "report_learning_curve.txt")
    lines = []
    lines.append("Learning-curve / bootstrap-subsampling report")
    lines.append("=" * 64)
    lines.append(
        "Question: on the current 89 Normal samples, does held-out error keep "
        "falling as n_train grows, or has it already approached a ceiling?"
    )
    lines.append(f"Mode: {mode}")
    lines.append("")
    lines.append("Summary (mean ± SD across repeats)")
    lines.append(summary.to_string(index=False))
    lines.append("")
    lines.append("Saturation fit: test MAE ≈ a + b / n")
    lines.append("  a = estimated irreducible error floor (years)")
    lines.append("  room_to_floor = MAE(n_max) - a")
    lines.append("  near_ceiling = room_to_floor < 0.15 years")
    lines.append(fit.to_string(index=False))
    lines.append("")
    lines.append("Interpretation")
    lines.append("-" * 64)
    for _, row in fit.iterrows():
        label = row["model_label"]
        lines.append(
            f"- {label}: MAE fell from {row['mae_at_n_min']:.3f} y at n={int(row['n_min'])} "
            f"to {row['mae_at_n_max']:.3f} y at n={int(row['n_max'])} "
            f"(total drop {row['mae_drop_total']:.3f} y; last-step drop "
            f"{row['mae_drop_last_step']:.3f} y)."
        )
        if pd.notna(row["asymptote_a"]):
            lines.append(
                f"  Fitted floor a = {row['asymptote_a']:.3f} y; "
                f"room remaining = {row['room_to_floor']:.3f} y "
                f"({'near a ceiling' if row['near_ceiling'] else 'still some headroom'})."
            )
    lines.append("")
    lines.append(
        "A last-step drop << 0.1 years together with room_to_floor < 0.15 years "
        "supports the statement that additional samples drawn from the same "
        "multi-study mix are unlikely to cut error substantially, i.e. the "
        "present accuracy is closer to a heterogeneity ceiling than to a "
        "pure sample-size limit. A still-steep curve would argue the opposite."
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def finalise(outdir: str, mode: str) -> None:
    run_csv = os.path.join(outdir, RUN_CSV_NAME)
    if not os.path.exists(run_csv):
        raise FileNotFoundError(f"No run records at {run_csv}")
    run_df = pd.read_csv(run_csv)
    summary = summarise(run_df)
    summary_path = os.path.join(outdir, SUMMARY_CSV_NAME)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    fit = fit_asymptote(summary)
    fit.to_csv(os.path.join(outdir, "saturation_fit.csv"), index=False)
    write_report(summary, fit, outdir, mode)
    plot_curves(summary, outdir, mode)


# ===========================================================================
# CLI
# ===========================================================================

def parse_sizes(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Leak-free learning-curve / bootstrap-subsampling analysis for the canine clocks."
    )
    p.add_argument("--metadata", default=DEFAULT_METADATA, help="total_metadata.txt")
    p.add_argument("--counts", default=DEFAULT_COUNTS, help="raw count matrix (genes x samples)")
    p.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="output directory",
    )
    p.add_argument(
        "--models",
        default="lasso,elasticnet,svr",
        help="comma-separated: lasso,elasticnet,svr,rf,xgboost,moe",
    )
    p.add_argument(
        "--aging-genes",
        default=DEFAULT_AGING_GENES,
        help="aging_gene_list.csv (required only when moe is included)",
    )
    p.add_argument("--moe-trials", type=int, default=25, help="MoE Optuna trials")
    p.add_argument("--moe-hpo-epochs", type=int, default=60, help="MoE epochs per Optuna trial")
    p.add_argument("--moe-refit-epochs", type=int, default=100, help="MoE max epochs for the selection/refit stage")
    p.add_argument(
        "--moe-skip-optuna",
        action="store_true",
        help="skip MoE Optuna and use the paper default hyperparameters",
    )
    p.add_argument(
        "--mode",
        choices=("learning_curve", "bootstrap"),
        default="learning_curve",
        help="learning_curve (recommended) or bootstrap subsampling",
    )
    p.add_argument("--n-repeats", type=int, default=10, help="outer repeats (default 10)")
    p.add_argument(
        "--train-sizes",
        type=parse_sizes,
        default=[20, 30, 40, 50, 60],
        help="requested training sizes; the full train-pool size is always added",
    )
    p.add_argument("--test-size", type=float, default=0.20, help="held-out fraction (default 0.20)")
    p.add_argument("--inner-cv", type=int, default=5, help="inner HPO folds (auto-shrunk if n is small)")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--rscript", default="Rscript")
    p.add_argument("--no-filter", action="store_true", help="disable filterByExpr (not recommended)")
    p.add_argument("--min-batch-for-split", type=int, default=5)
    p.add_argument("--min-stratum-for-test", type=int, default=5)
    p.add_argument("--resume", action="store_true", help="skip (repeat, n, model) cells already in run_records.csv")
    p.add_argument("--plot-only", action="store_true", help="rebuild figures/report from existing CSV")
    p.add_argument("--cleanup-tmp", action="store_true", default=True)
    p.add_argument("--keep-tmp", action="store_true", help="keep per-cell TMM temp directories")
    return p


def run_from_args(args) -> None:
    if getattr(args, "keep_tmp", False):
        args.cleanup_tmp = False
    os.makedirs(args.outdir, exist_ok=True)
    print("Output directory:", os.path.abspath(args.outdir))
    print("Metadata:", args.metadata)
    print("Counts:", args.counts)
    print("Models:", args.models)
    print("Mode:", args.mode)

    if args.plot_only:
        finalise(args.outdir, args.mode)
        return

    warnings.filterwarnings("ignore", category=UserWarning)
    if args.mode == "learning_curve":
        run_learning_curve(args)
    else:
        run_bootstrap(args)
    finalise(args.outdir, args.mode)
    print("\nDone. See", os.path.abspath(args.outdir))


def main():
    run_from_args(build_parser().parse_args())


if __name__ == "__main__":
    main()
