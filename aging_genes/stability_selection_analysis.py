"""Repeated CV / bootstrap stability of gene rankings with fold-wise TMM+logCPM and fixed HPO params."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ALL_MODELS = ("ElasticNet", "LASSO", "MoE", "RF", "SVR", "XGBoost")
RESULT_DIRS = {
    "ElasticNet": "aging_clock_results_Elastic_Net_pipeline_foldwise_TMM_logCPM",
    "LASSO": "aging_clock_results_LASSO_pipeline_foldwise_TMM_logCPM",
    "MoE": "aging_clock_results_MoE_pipeline_foldwise_TMM_logCPM",
    "RF": "aging_clock_results_Random_Forest_pipeline_foldwise_TMM_logCPM",
    "SVR": "aging_clock_results_SVR_pipeline_foldwise_TMM_logCPM",
    "XGBoost": "aging_clock_results_XGBoost_pipeline_foldwise_TMM_logCPM",
}


def parse_args(forced_method: str | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    descriptions = {
        None: "六模型 repeated CV / 分层 bootstrap 基因选择稳定性分析",
        "repeated_cv": "六模型 10×5 repeated CV 基因选择稳定性分析",
        "bootstrap": "六模型500次分层 bootstrap 基因选择稳定性分析",
    }
    parser = argparse.ArgumentParser(
        description=descriptions[forced_method]
    )
    if forced_method is None:
        parser.add_argument(
            "--method",
            choices=("repeated_cv", "bootstrap"),
            default="repeated_cv",
        )
    else:
        parser.set_defaults(method=forced_method)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=script_dir,
        help="total_metadata.txt 和表达矩阵所在路径（默认：当前脚本目录）",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=script_dir,
        help="六个 aging_clock_results_* 结果文件夹所在路径（默认：当前脚本目录）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="默认写入 all_coefficients/stability_<method>_results",
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS)
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[50, 100, 150, 200])
    if forced_method != "bootstrap":
        parser.add_argument("--n-splits", type=int, default=5)
        parser.add_argument("--n-repeats", type=int, default=10)
    else:
        parser.set_defaults(n_splits=5, n_repeats=10)
    if forced_method != "repeated_cv":
        parser.add_argument("--n-bootstraps", type=int, default=500)
        parser.add_argument(
            "--strata-cols",
            nargs="+",
            default=["batch", "group"],
            help="bootstrap 分层列；默认同时按 batch 和 group 分层",
        )
    else:
        parser.set_defaults(n_bootstraps=500, strata_cols=["batch", "group"])
    parser.add_argument("--random-state", type=int, default=66)
    parser.add_argument("--moe-epochs", type=int, default=100)
    parser.add_argument(
        "--moe-importance-samples",
        type=int,
        default=64,
        help="每次用于计算 MoE 梯度重要性的最大训练样本数",
    )
    parser.add_argument(
        "--aging-genes-file",
        type=Path,
        default=None,
        help="MoE AHK 先验基因文件；未指定时自动搜索",
    )
    parser.add_argument(
        "--disable-moe-prior",
        action="store_true",
        help="令 MoE gamma_ahk_loss=0，用于无衰老基因先验的敏感性分析",
    )
    parser.add_argument(
        "--reference-genes",
        type=Path,
        default=script_dir / "column_intersection_result.txt",
        help="需要单独汇报选择频率的原核心基因文件",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="忽略 raw counts 缓存，重新加载并构建 Normal 样本矩阵",
    )
    parser.add_argument(
        "--rscript",
        type=Path,
        default=None,
        help="Rscript.exe 路径；默认先查 PATH，再搜索 Windows 常见 R 安装目录",
    )
    return parser.parse_args()


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"找不到 HPO 参数文件：{path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_aging_genes_file(data_root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    candidates = [
        data_root / "aging_gene_list.csv",
        data_root.parent.parent / "aging_gene_list.csv",
        data_root.parent.parent / "20260706" / "no_ComBat" / "aging_gene_list.csv",
        data_root / "body_size" / "aging_gene_list.csv",
        data_root / "bodysize_new" / "aging_gene_list.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def resolve_rscript(explicit: Path | None) -> str:
    if explicit is not None:
        explicit = explicit.expanduser().resolve()
        if not explicit.exists():
            raise FileNotFoundError(f"--rscript 指定的文件不存在：{explicit}")
        return str(explicit)
    from_path = shutil.which("Rscript")
    if from_path:
        return from_path
    candidates = []
    for base in (Path("C:/Program Files/R"), Path("C:/Program Files (x86)/R")):
        if base.exists():
            candidates.extend(base.glob("R-*/bin/Rscript.exe"))
            candidates.extend(base.glob("R-*/bin/x64/Rscript.exe"))
    if candidates:
        return str(sorted(candidates)[-1])
    raise RuntimeError(
        "未找到 Rscript。请安装 R 和 edgeR，并把 Rscript 加入 PATH，"
        "或使用 --rscript \"C:\\\\...\\\\Rscript.exe\" 指定绝对路径。"
    )


def build_tmm_config(rscript: Path | None) -> dict:
    rscript_command = resolve_rscript(rscript)
    return {
        "foldwise_tmm_logcpm": {
            "enabled": True,
            "rscript": rscript_command,
            "prior_count": 1,
            "filter_low_expression": True,
        }
    }


def load_normal_data(
    data_root: Path,
    output_dir: Path,
    force: bool,
) -> tuple[pd.DataFrame, list[str]]:
    cache_path = output_dir / "normal_raw_data_cache.pkl"
    metadata_path = data_root / "total_metadata.txt"
    counts_path = data_root / "total_count_sort_ortholog.txt"
    if not metadata_path.exists() or not counts_path.exists():
        raise FileNotFoundError(
            f"数据目录应包含 total_metadata.txt 和 total_count_sort_ortholog.txt：{data_root}"
        )

    if cache_path.exists() and not force:
        cached = joblib.load(cache_path)
        return cached["normal_data"], cached["gene_columns"]

    metadata = pd.read_csv(metadata_path, sep="\t")
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    counts = pd.read_csv(counts_path, sep="\t", index_col=0)
    counts.index = counts.index.astype(str)
    counts.columns = counts.columns.astype(str)

    missing = sorted(set(metadata["sample_id"]) - set(counts.columns))
    if missing:
        raise ValueError(f"raw counts 缺少 metadata 中的样本：{missing[:10]}")
    counts = counts.loc[:, metadata["sample_id"]]

    gene_columns = counts.index.astype(str).tolist()
    expr_t = counts.T
    expr_t.index = expr_t.index.astype(str)
    normal_data = pd.merge(
        metadata.set_index("sample_id"),
        expr_t,
        left_index=True,
        right_index=True,
        how="inner",
    )
    normal_data = normal_data.loc[normal_data["condition"] == "Normal"].copy()
    if normal_data.empty:
        raise ValueError("未找到 condition == 'Normal' 的样本")

    joblib.dump(
        {"normal_data": normal_data, "gene_columns": gene_columns},
        cache_path,
    )
    return normal_data, gene_columns


def normalize_resample_splits(
    normal_data: pd.DataFrame,
    gene_columns: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    tmm_config: dict,
    elastic_module,
    output_dir: Path,
    run_name: str,
    age_col: str = "Chronological_Age",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    train_df = normal_data.iloc[train_idx].copy()
    val_df = normal_data.iloc[val_idx].copy()
    split_dict = {"train": train_df, "validation": val_df}
    fold_out_dir = output_dir / "foldwise_tmm_logcpm" / run_name

    train_df, split_dict, current_gene_columns = (
        elastic_module.normalize_fold_splits_tmm_logcpm(
            train_df=train_df,
            splits_dict=split_dict,
            gene_columns=gene_columns,
            config=tmm_config,
            out_dir=str(fold_out_dir),
            prefix=run_name,
        )
    )
    val_df = split_dict["validation"]
    X_train = train_df[current_gene_columns]
    y_train = pd.to_numeric(train_df[age_col], errors="raise")
    X_val = val_df[current_gene_columns]
    y_val = pd.to_numeric(val_df[age_col], errors="raise")
    return X_train, y_train, X_val, y_val, current_gene_columns


def load_hpo_params(data_root: Path, models: list[str]) -> dict[str, dict]:
    params = {}
    for model in models:
        params[model] = read_json(
            data_root / RESULT_DIRS[model] / "HPO" / "best_hyperparameters.json"
        )
    return params


def rank_nonzero_coefficients(
    genes: np.ndarray, coefficients: np.ndarray, tolerance: float = 1e-12
) -> list[str]:
    absolute = np.abs(np.asarray(coefficients).ravel())
    order = np.argsort(-absolute, kind="stable")
    return [str(genes[i]) for i in order if absolute[i] > tolerance]


def rank_all_scores(genes: np.ndarray, scores: np.ndarray) -> list[str]:
    values = np.asarray(scores).ravel()
    values = np.nan_to_num(values, nan=-np.inf)
    order = np.argsort(-values, kind="stable")
    return [str(genes[i]) for i in order]


def rank_positive_scores(
    genes: np.ndarray, scores: np.ndarray, tolerance: float = 0.0
) -> list[str]:
    values = np.asarray(scores).ravel()
    values = np.nan_to_num(values, nan=-np.inf)
    order = np.argsort(-values, kind="stable")
    return [str(genes[i]) for i in order if values[i] > tolerance]


def fit_elasticnet(
    X: pd.DataFrame, y: pd.Series, params: dict, seed: int
) -> list[str]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    model = ElasticNet(
        alpha=params["alpha"],
        l1_ratio=params["l1_ratio"],
        max_iter=int(params.get("max_iter", 100000)),
        random_state=seed,
    )
    model.fit(scaled, y)
    return rank_nonzero_coefficients(X.columns.to_numpy(), model.coef_)


def fit_lasso(X: pd.DataFrame, y: pd.Series, params: dict, seed: int) -> list[str]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    model = Lasso(
        alpha=params["alpha"],
        max_iter=10000,
        random_state=seed,
    )
    model.fit(scaled, y)
    return rank_nonzero_coefficients(X.columns.to_numpy(), model.coef_)


def fit_rf(X: pd.DataFrame, y: pd.Series, params: dict, seed: int) -> list[str]:
    model = RandomForestRegressor(
        **params,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return rank_positive_scores(X.columns.to_numpy(), model.feature_importances_)


def fit_svr(X: pd.DataFrame, y: pd.Series, params: dict, seed: int) -> list[str]:
    del seed
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    model = SVR(**params)
    model.fit(scaled, y)
    if params.get("kernel") != "linear":
        raise ValueError(
            "当前稳定性脚本要求 SVR 的 HPO 最优 kernel=linear；"
            "RBF 模型没有直接系数，需要另行计算置换重要性。"
        )
    return rank_all_scores(X.columns.to_numpy(), np.abs(model.coef_.ravel()))


def fit_xgboost(
    X: pd.DataFrame, y: pd.Series, params: dict, seed: int
) -> list[str]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("运行 XGBoost 稳定性分析需要安装 xgboost") from exc
    model = xgb.XGBRegressor(
        **params,
        tree_method="hist",
        device="cpu",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)
    return rank_positive_scores(X.columns.to_numpy(), model.feature_importances_)


def make_moe_fitter(
    script_dir: Path,
    params: dict,
    aging_genes: set[str],
    disable_prior: bool,
    epochs: int,
    importance_samples: int,
) -> Callable[[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, int], list[str]]:
    moe_module = load_module("_stability_moe", script_dir / "MoE.py")

    def fit(
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        seed: int,
    ) -> list[str]:
        moe_module.set_seed(seed)
        config = dict(moe_module.CONFIG)
        config.update(params)
        config["epochs"] = epochs
        config["seed"] = seed
        if disable_prior:
            config["gamma_ahk_loss"] = 0.0

        gene_list = X.columns.astype(str).tolist()
        aging_indices = np.array(
            [i for i, gene in enumerate(gene_list) if gene in aging_genes], dtype=int
        )
        if not disable_prior and len(aging_indices) == 0:
            raise ValueError("MoE 启用了 AHK 先验，但候选特征中未匹配到衰老基因")
        data_splits = {
            "X_train_raw": X,
            "y_train": y,
            "X_val_raw": X_val,
            "y_val": y_val,
            "aging_gene_indices": aging_indices,
        }
        model, scaler_x, _, _ = moe_module.train_moe_clock(data_splits, config)
        if len(X) > importance_samples:
            importance_x = X.sample(n=importance_samples, random_state=seed)
        else:
            importance_x = X
        importance = moe_module.compute_gene_importance(
            model, scaler_x, importance_x, gene_list
        )
        return importance["gene"].astype(str).tolist()

    return fit


def stratified_bootstrap_indices(
    metadata: pd.DataFrame,
    strata_cols: list[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    missing = [column for column in strata_cols if column not in metadata.columns]
    if missing:
        raise ValueError(f"metadata 缺少 bootstrap 分层列：{missing}")
    positions = np.arange(len(metadata))
    strata_frame = metadata.reset_index(drop=True)
    sampled_parts = []
    grouped = strata_frame.groupby(strata_cols, dropna=False, sort=False).indices
    for group_positions in grouped.values():
        group_positions = np.asarray(group_positions, dtype=int)
        sampled_parts.append(
            rng.choice(group_positions, size=len(group_positions), replace=True)
        )
    sampled = np.concatenate(sampled_parts)
    rng.shuffle(sampled)
    oob = np.setdiff1d(positions, np.unique(sampled), assume_unique=False)
    return sampled, oob


def make_resamples(
    method: str,
    metadata: pd.DataFrame,
    n_splits: int,
    n_repeats: int,
    n_bootstraps: int,
    strata_cols: list[str],
    seed: int,
):
    if method == "repeated_cv":
        splitter = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        for run_id, (train_idx, val_idx) in enumerate(splitter.split(metadata), start=1):
            yield f"cv_{run_id:03d}", train_idx, val_idx
    else:
        rng = np.random.default_rng(seed)
        for run_id in range(1, n_bootstraps + 1):
            train_idx, val_idx = stratified_bootstrap_indices(
                metadata, strata_cols, rng
            )
            if len(val_idx) < 2:
                raise RuntimeError(
                    f"bootstrap_{run_id} 的 OOB 样本不足2个，无法作为 MoE 验证集"
                )
            yield f"bootstrap_{run_id:04d}", train_idx, val_idx


def update_selection_counts(
    model: str,
    ranking: list[str],
    top_k: list[int],
    model_counts: dict[tuple[str, int], Counter],
) -> dict[int, set[str]]:
    selected = {}
    for k in top_k:
        genes = set(ranking[: min(k, len(ranking))])
        selected[k] = genes
        model_counts[(model, k)].update(genes)
    return selected


def plot_reference_frequencies(
    frequency_df: pd.DataFrame,
    reference_genes: list[str],
    output_dir: Path,
) -> None:
    if not reference_genes:
        return
    shown = frequency_df.loc[
        frequency_df["gene"].isin(reference_genes)
        & frequency_df["K"].eq(100)
    ].copy()
    if shown.empty:
        return
    pivot = shown.pivot(index="gene", columns="model", values="selection_frequency")
    pivot = pivot.reindex(reference_genes).fillna(0)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("Top-100 selection frequency of reference genes")
    fig.colorbar(image, ax=ax, label="Selection frequency")
    fig.tight_layout()
    fig.savefig(output_dir / "reference_gene_top100_frequency.png", dpi=300)
    fig.savefig(output_dir / "reference_gene_top100_frequency.pdf")
    plt.close(fig)


def read_reference_genes(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path, sep=None, engine="python")
    if df.empty:
        return []
    column = "intersection" if "intersection" in df.columns else df.columns[0]
    return df[column].dropna().astype(str).str.strip().tolist()


def main(forced_method: str | None = None) -> None:
    args = parse_args(forced_method=forced_method)
    script_dir = Path(__file__).resolve().parent
    args.data_root = args.data_root.resolve()
    args.results_root = args.results_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else script_dir / f"stability_{args.method}_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k = sorted(set(args.top_k))
    if any(k <= 0 for k in top_k):
        raise ValueError("--top-k 必须全部为正整数")
    if len(args.models) < 4:
        print(
            "提示：当前选择的模型少于4个，将只输出逐模型频率；"
            ">=4/6、>=5/6、6/6 共识仅在六模型全部运行时有原定义。"
        )

    normal_data, gene_columns = load_normal_data(
        args.data_root,
        output_dir,
        args.force_preprocess,
    )
    elastic_module = load_module(
        "_stability_elastic_helpers", script_dir / "ElasticNet.py"
    )
    tmm_config = build_tmm_config(args.rscript)
    params = load_hpo_params(args.results_root, args.models)

    fitters: dict[str, Callable] = {
        "ElasticNet": fit_elasticnet,
        "LASSO": fit_lasso,
        "RF": fit_rf,
        "SVR": fit_svr,
        "XGBoost": fit_xgboost,
    }
    if "MoE" in args.models:
        aging_path = find_aging_genes_file(args.data_root, args.aging_genes_file)
        if not args.disable_moe_prior and (aging_path is None or not aging_path.exists()):
            raise FileNotFoundError(
                "MoE 启用了衰老基因先验，但未找到 aging_gene_list.csv；"
                "请使用 --aging-genes-file 指定，或使用 --disable-moe-prior。"
            )
        aging_genes = (
            set(pd.read_csv(aging_path).iloc[:, 0].dropna().astype(str))
            if aging_path is not None and aging_path.exists()
            else set()
        )
        fitters["MoE"] = make_moe_fitter(
            script_dir,
            params["MoE"],
            aging_genes,
            args.disable_moe_prior,
            args.moe_epochs,
            args.moe_importance_samples,
        )

    model_counts: dict[tuple[str, int], Counter] = defaultdict(Counter)
    consensus_counts: dict[tuple[int, int], Counter] = defaultdict(Counter)
    run_rows = []
    ranking_rows = []
    n_runs = 0

    resamples = make_resamples(
        args.method,
        normal_data,
        args.n_splits,
        args.n_repeats,
        args.n_bootstraps,
        args.strata_cols,
        args.random_state,
    )
    for n_runs, (run_name, train_idx, val_idx) in enumerate(resamples, start=1):
        seed = args.random_state + n_runs
        X_train, y_train, X_val, y_val, current_gene_columns = normalize_resample_splits(
            normal_data,
            gene_columns,
            train_idx,
            val_idx,
            tmm_config,
            elastic_module,
            output_dir,
            run_name,
        )
        print(
            f"\n[{run_name}] train={len(X_train)}, validation/OOB={len(X_val)}, "
            f"genes={len(current_gene_columns)}"
        )

        selected_by_model: dict[str, dict[int, set[str]]] = {}
        for model_name in args.models:
            print(f"  - 拟合 {model_name} ...")
            if model_name == "MoE":
                ranking = fitters[model_name](
                    X_train, y_train, X_val, y_val, seed
                )
            else:
                ranking = fitters[model_name](
                    X_train, y_train, params[model_name], seed
                )
            selected_by_model[model_name] = update_selection_counts(
                model_name, ranking, top_k, model_counts
            )
            for rank, gene in enumerate(ranking[: max(top_k)], start=1):
                ranking_rows.append(
                    {
                        "run": run_name,
                        "model": model_name,
                        "rank": rank,
                        "gene": gene,
                    }
                )

        for k in top_k:
            gene_support = Counter()
            for model_name in args.models:
                gene_support.update(selected_by_model[model_name][k])
            row = {
                "run": run_name,
                "K": k,
                "n_train": len(train_idx),
                "n_validation_or_oob": len(val_idx),
            }
            for threshold in (4, 5, 6):
                genes = {
                    gene for gene, count in gene_support.items() if count >= threshold
                }
                consensus_counts[(threshold, k)].update(genes)
                row[f"n_consensus_{threshold}of6"] = len(genes)
                row[f"genes_consensus_{threshold}of6"] = ";".join(sorted(genes))
            run_rows.append(row)

        # 长任务每10次重采样写一次检查点，避免每一轮反复重写不断增大的文件。
        if n_runs % 10 == 0:
            pd.DataFrame(ranking_rows).to_csv(
                output_dir / "top_rankings_by_resample.partial.csv", index=False
            )
            pd.DataFrame(run_rows).to_csv(
                output_dir / "consensus_counts_by_resample.partial.csv", index=False
            )

    if n_runs == 0:
        raise RuntimeError("没有产生任何重采样")

    pd.DataFrame(ranking_rows).to_csv(
        output_dir / "top_rankings_by_resample.csv", index=False
    )
    pd.DataFrame(run_rows).to_csv(
        output_dir / "consensus_counts_by_resample.csv", index=False
    )

    frequency_rows = []
    all_genes = gene_columns
    for model_name in args.models:
        for k in top_k:
            counts = model_counts[(model_name, k)]
            for gene in all_genes:
                count = counts.get(gene, 0)
                frequency_rows.append(
                    {
                        "model": model_name,
                        "K": k,
                        "gene": gene,
                        "selected_runs": count,
                        "total_runs": n_runs,
                        "selection_frequency": count / n_runs,
                    }
                )
    frequency_df = pd.DataFrame(frequency_rows)
    frequency_df.to_csv(output_dir / "model_gene_selection_frequencies.csv", index=False)

    consensus_frequency_rows = []
    for threshold in (4, 5, 6):
        for k in top_k:
            counts = consensus_counts[(threshold, k)]
            for gene in all_genes:
                count = counts.get(gene, 0)
                consensus_frequency_rows.append(
                    {
                        "consensus_threshold": f">={threshold}/6",
                        "K": k,
                        "gene": gene,
                        "consensus_runs": count,
                        "total_runs": n_runs,
                        "consensus_frequency": count / n_runs,
                    }
                )
    pd.DataFrame(consensus_frequency_rows).to_csv(
        output_dir / "cross_model_consensus_frequencies.csv", index=False
    )

    reference_genes = read_reference_genes(args.reference_genes)
    if reference_genes:
        reference_table = frequency_df.loc[
            frequency_df["gene"].isin(reference_genes)
        ].copy()
        reference_table.to_csv(
            output_dir / "reference_gene_selection_frequencies.csv", index=False
        )
        plot_reference_frequencies(
            frequency_df, reference_genes, output_dir
        )

    manifest = {
        "method": args.method,
        "models": args.models,
        "top_k": top_k,
        "total_resamples": n_runs,
        "n_splits": args.n_splits if args.method == "repeated_cv" else None,
        "n_repeats": args.n_repeats if args.method == "repeated_cv" else None,
        "n_bootstraps": args.n_bootstraps if args.method == "bootstrap" else None,
        "bootstrap_strata": args.strata_cols if args.method == "bootstrap" else None,
        "random_state": args.random_state,
        "hyperparameters": params,
        "preprocessing": (
            "Fold-wise TMM+logCPM within each resample: training counts determine "
            "gene filtering and TMM factors; validation/OOB samples normalized using "
            "the training reference; no age labels used in normalization"
        ),
        "stability_interpretation": "conditional on the original HPO hyperparameters",
        "moe_prior_disabled": args.disable_moe_prior,
    }
    with (output_dir / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    for partial in (
        output_dir / "top_rankings_by_resample.partial.csv",
        output_dir / "consensus_counts_by_resample.partial.csv",
    ):
        partial.unlink(missing_ok=True)
    print(f"\n稳定性分析完成，共 {n_runs} 次重采样。结果目录：{output_dir}")


if __name__ == "__main__":
    main()
