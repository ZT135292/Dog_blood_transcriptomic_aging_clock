"""Leave-one-size-out validation using DAP size_class from total_metadata_weight.txt."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import ElasticNet as elastic_module
import LASSO as lasso_module
import MoE as moe_module
import Random_Forest as rf_module
import SVR as svr_module
import XGBoost as xgb_module


SCRIPT_DIR = Path(__file__).resolve().parent
SIZE_ORDER = ["Small", "Medium", "Standard", "Large", "Giant"]
ALL_MODELS = ["LASSO", "ElasticNet", "RandomForest", "SVR", "XGBoost", "MoE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run six aging-clock models with leave-one-size-out validation."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
        help="需要运行的模型；默认全部六种。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "aging_clock_results_leave_one_size_out",
    )
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument(
        "--disable-moe-optuna",
        action="store_true",
        help="关闭每个外层fold内的MoE Optuna调参以缩短运行时间。",
    )
    parser.add_argument(
        "--moe-trials",
        type=int,
        default=None,
        help="覆盖MoE每个外层fold的Optuna trials数量。",
    )
    return parser.parse_args()


def simplify_sex(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("female"):
        return "Female"
    if text.startswith("male"):
        return "Male"
    return "Unknown"


def metric_values(group: pd.DataFrame) -> dict:
    actual = group["actual_age"].to_numpy(dtype=float)
    predicted = group["predicted_age"].to_numpy(dtype=float)
    error = predicted - actual
    n = len(group)
    result = {
        "n": n,
        "n_studies": int(group["batch"].nunique()),
        "n_breeds": int(group["breed"].nunique()),
        "age_min": float(np.min(actual)),
        "age_median": float(np.median(actual)),
        "age_max": float(np.max(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "median_ae": float(np.median(np.abs(error))),
        "mean_error": float(np.mean(error)),
    }
    if n >= 3 and np.std(actual) > 0 and np.std(predicted) > 0:
        result["r2"] = float(r2_score(actual, predicted))
        result["pearson_r"] = float(pearsonr(actual, predicted).statistic)
        result["spearman_rho"] = float(spearmanr(actual, predicted).statistic)
        slope, intercept = np.polyfit(actual, predicted, deg=1)
        result["calibration_intercept"] = float(intercept)
        result["calibration_slope"] = float(slope)
    else:
        result.update(
            {
                "r2": np.nan,
                "pearson_r": np.nan,
                "spearman_rho": np.nan,
                "calibration_intercept": np.nan,
                "calibration_slope": np.nan,
            }
        )
    return result


def fit_interaction_model(
    data: pd.DataFrame, adjusted: bool
) -> tuple[pd.DataFrame, dict, str]:
    model_data = data[
        ["predicted_age", "actual_age", "size_class", "batch"]
    ].dropna().copy()
    present_sizes = [
        size for size in SIZE_ORDER if model_data["size_class"].eq(size).any()
    ]
    model_data["size_class"] = pd.Categorical(
        model_data["size_class"], categories=present_sizes, ordered=True
    )
    if len(present_sizes) < 2:
        raise ValueError("至少需要两个体型组才能进行 age × size 检验。")

    reference = present_sizes[0]
    size_term = f"C(size_class, Treatment(reference='{reference}'))"
    formula = f"predicted_age ~ actual_age * {size_term}"
    if adjusted:
        if model_data["batch"].nunique() > 1:
            formula += " + C(batch)"

    fitted = smf.ols(formula, data=model_data).fit(cov_type="HC3")
    confidence = fitted.conf_int()
    coefficients = pd.DataFrame(
        {
            "term": fitted.params.index,
            "estimate": fitted.params.values,
            "std_error_hc3": fitted.bse.values,
            "ci_low": confidence.iloc[:, 0].values,
            "ci_high": confidence.iloc[:, 1].values,
            "p_value": fitted.pvalues.values,
        }
    )
    interaction_terms = [
        term
        for term in fitted.params.index
        if "actual_age:" in term or ":actual_age" in term
    ]
    if interaction_terms:
        restriction = np.zeros((len(interaction_terms), len(fitted.params)))
        for row_index, term in enumerate(interaction_terms):
            restriction[row_index, fitted.params.index.get_loc(term)] = 1.0
        test = fitted.wald_test(restriction, scalar=True)
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        statistic = np.nan
        p_value = np.nan

    summary = {
        "model_type": "adjusted" if adjusted else "unadjusted",
        "formula": formula,
        "reference_size": reference,
        "n": int(fitted.nobs),
        "n_studies": int(model_data["batch"].nunique()),
        "r_squared": float(fitted.rsquared),
        "adjusted_r_squared": float(fitted.rsquared_adj),
        "interaction_terms": "; ".join(interaction_terms),
        "interaction_wald_chi2": statistic,
        "interaction_df": len(interaction_terms),
        "interaction_p_value": p_value,
    }
    return coefficients, summary, fitted.summary().as_text()


def prepare_data() -> tuple[dict, pd.DataFrame]:
    package = lasso_module.load_full_data(lasso_module.CONFIG)
    normal = package["normal_data"].copy()
    if "size_class_DAP" not in normal.columns:
        raise ValueError("元数据缺少体型分类列 'size_class_DAP'")
    size = normal["size_class_DAP"].astype("string").str.strip()
    size = size.mask(size.isin(["", "None", "NA", "NaN", "nan"]))
    invalid = sorted(set(size.dropna()) - set(SIZE_ORDER))
    if invalid:
        raise ValueError(
            f"size_class_DAP 包含未识别类别: {invalid}；允许类别为 {SIZE_ORDER}"
        )
    normal["size_class"] = pd.Categorical(
        size, categories=SIZE_ORDER, ordered=True
    )

    missing_size = normal[normal["size_class"].isna()].copy()
    normal = normal.dropna(subset=["size_class"]).copy()
    present = normal["size_class"].dropna().unique().tolist()
    if len(present) < 2:
        raise ValueError(f"LOSO至少需要两个体型组；当前仅发现: {present}")

    package["normal_data"] = normal
    return package, missing_size


def normalize_outer_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gene_columns: list[str],
    fold_dir: Path,
    fold_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    config = copy.deepcopy(lasso_module.CONFIG)
    train_norm, splits, kept_genes = (
        lasso_module.normalize_fold_splits_tmm_logcpm(
            train_df=train_df,
            splits_dict={"train": train_df, "validation": test_df},
            gene_columns=gene_columns,
            config=config,
            out_dir=str(fold_dir / "foldwise_tmm_logcpm"),
            prefix=fold_name,
        )
    )
    return train_norm, splits["validation"], kept_genes


def prediction_frame(
    test_df: pd.DataFrame,
    predicted: np.ndarray,
    model_name: str,
    held_out_size: str,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "sample_id": test_df.index.astype(str),
            "actual_age": test_df["Chronological_Age"].to_numpy(dtype=float),
            "predicted_age": np.asarray(predicted, dtype=float),
        }
    )
    metadata_columns = [
        "sex",
        "breed",
        "treatment",
        "batch",
        "condition",
        "weight_kg_mid",
        "size_class",
    ]
    metadata_columns = [col for col in metadata_columns if col in test_df.columns]
    test_metadata = test_df[metadata_columns].copy().reset_index()
    test_metadata = test_metadata.rename(columns={test_metadata.columns[0]: "sample_id"})
    test_metadata["sample_id"] = test_metadata["sample_id"].astype(str)
    result = result.merge(
        test_metadata, on="sample_id", how="left", validate="one_to_one"
    )
    result["model"] = model_name
    result["held_out_size"] = held_out_size
    result["data_type"] = "leave_one_size_out"
    result["prediction_error"] = result["predicted_age"] - result["actual_age"]
    result["absolute_error"] = result["prediction_error"].abs()
    result["sex_simple"] = result["sex"].map(simplify_sex)
    result["size_class"] = pd.Categorical(
        result["size_class"], categories=SIZE_ORDER, ordered=True
    )
    return result


def save_fold_metrics(
    predictions: pd.DataFrame,
    model_name: str,
    held_out_size: str,
    n_train: int,
    train_sizes: str,
    fold_dir: Path,
) -> dict:
    metrics = metric_values(predictions)
    metrics.update(
        {
            "model": model_name,
            "held_out_size": held_out_size,
            "train_sizes": train_sizes,
            "n_train": int(n_train),
        }
    )
    pd.DataFrame([metrics]).to_csv(fold_dir / "metrics.csv", index=False)
    return metrics


def train_lasso(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    genes: list[str],
    fold_dir: Path,
    fold_name: str,
) -> np.ndarray:
    config = copy.deepcopy(lasso_module.CONFIG)
    model, scaler, params = lasso_module.run_lassocv_on_fold(
        train_df[genes],
        train_df[config["age_col"]],
        config,
        fold_label=fold_name,
    )
    joblib.dump(model, fold_dir / "model.pkl")
    joblib.dump(scaler, fold_dir / "scaler.pkl")
    (fold_dir / "best_params.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )
    return model.predict(scaler.transform(test_df[genes]))


def train_elasticnet(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    genes: list[str],
    fold_dir: Path,
    fold_name: str,
) -> np.ndarray:
    config = copy.deepcopy(elastic_module.CONFIG)
    model, scaler, params = elastic_module.run_elasticnet_cv_on_fold(
        train_df[genes],
        train_df[config["age_col"]],
        config,
        fold_label=fold_name,
    )
    joblib.dump(model, fold_dir / "model.pkl")
    joblib.dump(scaler, fold_dir / "scaler.pkl")
    (fold_dir / "best_params.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )
    return model.predict(scaler.transform(test_df[genes]))


def train_random_forest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    genes: list[str],
    fold_dir: Path,
    fold_name: str,
) -> np.ndarray:
    config = copy.deepcopy(rf_module.CONFIG)
    model, params = rf_module.run_rf_hpo_on_fold(
        train_df[genes],
        train_df[config["age_col"]],
        config,
        fold_label=fold_name,
    )
    joblib.dump(model, fold_dir / "model.pkl")
    (fold_dir / "best_params.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )
    return model.predict(test_df[genes])


def train_svr(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    genes: list[str],
    fold_dir: Path,
    fold_name: str,
) -> np.ndarray:
    config = copy.deepcopy(svr_module.CONFIG)
    model, scaler, params = svr_module.run_svr_hpo_on_fold(
        train_df[genes],
        train_df[config["age_col"]],
        config,
        fold_label=fold_name,
    )
    joblib.dump(model, fold_dir / "model.pkl")
    joblib.dump(scaler, fold_dir / "scaler.pkl")
    (fold_dir / "best_params.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )
    return model.predict(scaler.transform(test_df[genes]))


def train_xgboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    genes: list[str],
    fold_dir: Path,
    fold_name: str,
) -> np.ndarray:
    config = copy.deepcopy(xgb_module.CONFIG)
    model, params = xgb_module.run_xgb_hpo_on_fold(
        train_df[genes],
        train_df[config["age_col"]],
        config,
        fold_label=fold_name,
    )
    model.save_model(fold_dir / "model.json")
    (fold_dir / "best_params.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )
    return model.predict(test_df[genes])


def split_moe_inner_validation(
    train_df: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 外层训练集通常包含两个体型；若两组均有足够样本则保持体型比例。
    size_counts = train_df["size_class"].value_counts()
    stratify = (
        train_df["size_class"]
        if len(size_counts) >= 2 and int(size_counts.min()) >= 2
        else None
    )
    inner_train_index, inner_val_index = train_test_split(
        train_df.index,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )
    return train_df.loc[inner_train_index].copy(), train_df.loc[inner_val_index].copy()


def train_moe(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    genes: list[str],
    fold_dir: Path,
    fold_name: str,
    seed: int,
    enable_optuna: bool,
    moe_trials: int | None,
) -> np.ndarray:
    moe_module.set_seed(seed)
    config = copy.deepcopy(moe_module.CONFIG)
    config["seed"] = seed
    if moe_trials is not None:
        config["optuna"]["trials"] = int(moe_trials)

    inner_train, inner_val = split_moe_inner_validation(train_df, seed)
    aging_genes = pd.read_csv(SCRIPT_DIR / config["aging_genes_file"])
    aging_pool = set(aging_genes.iloc[:, 0].astype(str))
    aging_indices = moe_module.compute_aging_gene_indices(genes, aging_pool)
    data_splits = {
        "X_train_raw": inner_train[genes],
        "y_train": inner_train[config["age_col"]],
        "X_val_raw": inner_val[genes],
        "y_val": inner_val[config["age_col"]],
        "X_intervention_raw": test_df[genes],
        "y_intervention": test_df[config["age_col"]],
        "gene_columns": genes,
        "metadata": pd.DataFrame(),
        "aging_gene_indices": aging_indices,
    }

    best_params = {}
    if enable_optuna:
        best_params = moe_module.optuna_search_hpo(data_splits, config)
    train_config = copy.deepcopy(config)
    train_config.update(best_params)
    model, scaler_x, scaler_y, training_info = moe_module.train_moe_clock(
        data_splits, train_config
    )

    torch.save(model.state_dict(), fold_dir / "model.pth")
    joblib.dump(scaler_x, fold_dir / "scaler_X.pkl")
    joblib.dump(scaler_y, fold_dir / "scaler_y.pkl")
    (fold_dir / "best_params.json").write_text(
        json.dumps({**best_params, "training_info": training_info}, indent=2),
        encoding="utf-8",
    )
    (fold_dir / "inner_split.json").write_text(
        json.dumps(
            {
                "n_inner_train": len(inner_train),
                "n_inner_validation": len(inner_val),
                "inner_train_sizes": {
                    str(key): int(value)
                    for key, value in inner_train["size_class"]
                    .value_counts()
                    .items()
                },
                "inner_validation_sizes": {
                    str(key): int(value)
                    for key, value in inner_val["size_class"]
                    .value_counts()
                    .items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    x_test = scaler_x.transform(test_df[genes])
    x_test = np.nan_to_num(x_test, nan=0.0)
    x_test_tensor = torch.tensor(
        x_test, dtype=torch.float32, device=next(model.parameters()).device
    )
    model.eval()
    with torch.no_grad():
        predicted_scaled, _, _ = model(x_test_tensor)
    return scaler_y.inverse_transform(
        predicted_scaled.detach().cpu().numpy()
    ).ravel()


TRAINERS = {
    "LASSO": train_lasso,
    "ElasticNet": train_elasticnet,
    "RandomForest": train_random_forest,
    "SVR": train_svr,
    "XGBoost": train_xgboost,
}


def save_model_summary(
    predictions: pd.DataFrame,
    metrics: list[dict],
    model_name: str,
    model_dir: Path,
) -> None:
    predictions.to_csv(model_dir / "all_oof_predictions.csv", index=False)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(model_dir / "metrics_by_held_out_size.csv", index=False)

    overall = metric_values(predictions)
    overall.update({"model": model_name, "held_out_size": "Overall"})
    pd.DataFrame([overall]).to_csv(model_dir / "overall_metrics.csv", index=False)

    coefficient_tables = []
    interaction_summaries = []
    for adjusted in (False, True):
        coefficients, summary, text = fit_interaction_model(
            predictions, adjusted=adjusted
        )
        summary["model"] = model_name
        label = summary["model_type"]
        coefficients.insert(0, "model", model_name)
        coefficients.insert(1, "model_type", label)
        coefficient_tables.append(coefficients)
        interaction_summaries.append(summary)
        (model_dir / f"age_by_size_{label}_summary.txt").write_text(
            text, encoding="utf-8"
        )
    pd.concat(coefficient_tables, ignore_index=True).to_csv(
        model_dir / "age_by_size_coefficients.csv", index=False
    )
    pd.DataFrame(interaction_summaries).to_csv(
        model_dir / "age_by_size_interaction_tests.csv", index=False
    )

    sns.set_theme(style="whitegrid")
    grid = sns.lmplot(
        data=predictions,
        x="actual_age",
        y="predicted_age",
        hue="size_class",
        hue_order=SIZE_ORDER,
        palette="Set1",
        height=6,
        aspect=1.2,
        scatter_kws={"alpha": 0.75, "s": 45},
        ci=95,
    )
    lower = min(predictions["actual_age"].min(), predictions["predicted_age"].min())
    upper = max(predictions["actual_age"].max(), predictions["predicted_age"].max())
    grid.ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
    grid.ax.set_title(f"{model_name}: Leave-One-Size-Out predictions")
    grid.ax.set_xlabel("Chronological age (years)")
    grid.ax.set_ylabel("Predicted age (years)")
    grid.figure.tight_layout()
    grid.figure.savefig(
        model_dir / "predicted_vs_actual_by_held_out_size.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(grid.figure)

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=predictions,
        x="size_class",
        y="prediction_error",
        order=SIZE_ORDER,
        hue="size_class",
        palette="Set1",
        legend=False,
    )
    sns.stripplot(
        data=predictions,
        x="size_class",
        y="prediction_error",
        order=SIZE_ORDER,
        color="black",
        alpha=0.55,
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Held-out body-size class")
    plt.ylabel("Predicted age − chronological age (years)")
    plt.title(f"{model_name}: LOSO prediction error")
    plt.tight_layout()
    plt.savefig(model_dir / "prediction_error_by_size.png", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data_package, missing_size = prepare_data()
    normal = data_package["normal_data"]
    raw_genes = data_package["gene_columns"]

    composition = (
        normal.groupby(["size_class", "batch"], observed=True)
        .agg(
            n=("Chronological_Age", "size"),
            n_breeds=("breed", "nunique"),
            age_min=("Chronological_Age", "min"),
            age_median=("Chronological_Age", "median"),
            age_max=("Chronological_Age", "max"),
        )
        .reset_index()
    )
    composition.to_csv(output_root / "size_by_study_composition.csv", index=False)
    if not missing_size.empty:
        missing_size.reset_index().rename(
            columns={"index": "sample_id"}
        ).to_csv(output_root / "normal_samples_excluded_missing_weight.csv", index=False)

    all_predictions = []
    all_metrics = []
    for held_out_size in SIZE_ORDER:
        fold_name = f"leave_{held_out_size}_out"
        train_raw = normal[~normal["size_class"].eq(held_out_size)].copy()
        test_raw = normal[normal["size_class"].eq(held_out_size)].copy()
        train_sizes = "+".join(
            size for size in SIZE_ORDER if size != held_out_size
        )
        print(
            f"\n{fold_name}: train n={len(train_raw)} ({train_sizes}), "
            f"test n={len(test_raw)}"
        )

        preprocessing_dir = output_root / "shared_preprocessing" / fold_name
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        train_df, test_df, genes = normalize_outer_fold(
            train_raw,
            test_raw,
            raw_genes,
            preprocessing_dir,
            fold_name,
        )

        for model_index, model_name in enumerate(args.models):
            print(f"  - {model_name}: {fold_name}")
            model_dir = output_root / model_name
            fold_dir = model_dir / fold_name
            fold_dir.mkdir(parents=True, exist_ok=True)

            if model_name == "MoE":
                predicted = train_moe(
                    train_df=train_df,
                    test_df=test_df,
                    genes=genes,
                    fold_dir=fold_dir,
                    fold_name=fold_name,
                    seed=args.seed + model_index,
                    enable_optuna=not args.disable_moe_optuna,
                    moe_trials=args.moe_trials,
                )
            else:
                predicted = TRAINERS[model_name](
                    train_df, test_df, genes, fold_dir, fold_name
                )

            fold_predictions = prediction_frame(
                test_df, predicted, model_name, held_out_size
            )
            fold_predictions.to_csv(fold_dir / "predictions.csv", index=False)
            fold_metrics = save_fold_metrics(
                fold_predictions,
                model_name,
                held_out_size,
                n_train=len(train_df),
                train_sizes=train_sizes,
                fold_dir=fold_dir,
            )
            all_predictions.append(fold_predictions)
            all_metrics.append(fold_metrics)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_metrics_df = pd.DataFrame(all_metrics)
    all_predictions_df.to_csv(
        output_root / "all_models_oof_predictions.csv", index=False
    )
    all_metrics_df.to_csv(
        output_root / "all_models_metrics_by_held_out_size.csv", index=False
    )

    for model_name in args.models:
        model_predictions = all_predictions_df[
            all_predictions_df["model"].eq(model_name)
        ].copy()
        model_metrics = [
            row for row in all_metrics if row["model"] == model_name
        ]
        save_model_summary(
            model_predictions,
            model_metrics,
            model_name,
            output_root / model_name,
        )

    run_info = {
        "models": args.models,
        "size_source": "total_metadata_weight.txt:size_class_DAP",
        "size_categories": SIZE_ORDER,
        "normal_size_counts": {
            size: int(normal["size_class"].eq(size).sum())
            for size in SIZE_ORDER
        },
        "n_excluded_missing_weight": int(len(missing_size)),
        "moe_optuna_enabled": not args.disable_moe_optuna,
        "moe_trials_override": args.moe_trials,
        "seed": args.seed,
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nLeave-One-Size-Out完成: {output_root}")


if __name__ == "__main__":
    main()
