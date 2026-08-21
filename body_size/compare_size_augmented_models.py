"""Compare clocks with vs without DAP size-class dummies under matched DataSet-LOO folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import LASSO as lasso_module
from leave_one_size_out import (
    ALL_MODELS,
    SIZE_ORDER,
    TRAINERS,
    metric_values,
    normalize_outer_fold,
    simplify_sex,
    train_moe,
)


SCRIPT_DIR = Path(__file__).resolve().parent
VARIANTS = ["baseline", "size_augmented"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare expression-only and expression-plus-size clocks."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "aging_clock_results_size_augmented_comparison",
    )
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument(
        "--disable-moe-optuna",
        action="store_true",
        help="关闭MoE Optuna；默认与原始MoE.py一致，在每个训练fold中启用。",
    )
    parser.add_argument("--moe-trials", type=int, default=None)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help="配对Delta-MAE bootstrap次数（默认5000）。",
    )
    return parser.parse_args()


def prepare_normal_data() -> tuple[dict, pd.DataFrame]:
    package = lasso_module.load_full_data(lasso_module.CONFIG)
    normal = package["normal_data"].copy()
    if "size_class_DAP" not in normal.columns:
        raise ValueError("元数据缺少 'size_class_DAP' 列。")

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
    excluded = normal[normal["size_class"].isna()].copy()
    normal = normal.dropna(subset=["size_class"]).copy()
    package["normal_data"] = normal
    return package, excluded


def add_size_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train_df.copy()
    test = test_df.copy()
    feature_names = []
    # 固定以Small为参考，加入其余四个DAP类别。某训练fold未出现的类别列为全0；
    # 这不会从验证集估计任何编码参数。
    for size in SIZE_ORDER[1:]:
        feature = f"size_DAP_{size}"
        train[feature] = train["size_class"].eq(size).astype(float)
        test[feature] = test["size_class"].eq(size).astype(float)
        feature_names.append(feature)
    return train, test, feature_names


def make_prediction_frame(
    test_df: pd.DataFrame,
    predicted: np.ndarray,
    model_name: str,
    variant: str,
    held_out_study: str,
) -> pd.DataFrame:
    columns = [
        "sex",
        "breed",
        "treatment",
        "batch",
        "condition",
        "weight_kg_mid",
        "size_class_DAP",
        "size_class",
    ]
    columns = [column for column in columns if column in test_df.columns]
    metadata = test_df[columns].copy().reset_index()
    metadata = metadata.rename(columns={metadata.columns[0]: "sample_id"})
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    result = pd.DataFrame(
        {
            "sample_id": test_df.index.astype(str),
            "actual_age": test_df["Chronological_Age"].to_numpy(dtype=float),
            "predicted_age": np.asarray(predicted, dtype=float),
        }
    )
    result = result.merge(
        metadata, on="sample_id", how="left", validate="one_to_one"
    )
    result["model"] = model_name
    result["variant"] = variant
    result["held_out_study"] = str(held_out_study)
    result["prediction_error"] = result["predicted_age"] - result["actual_age"]
    result["absolute_error"] = result["prediction_error"].abs()
    result["sex_simple"] = result["sex"].map(simplify_sex)
    return result


def bootstrap_delta_mae(
    paired: pd.DataFrame, n_bootstrap: int, seed: int
) -> dict:
    baseline_error = paired["absolute_error_baseline"].to_numpy(dtype=float)
    augmented_error = paired["absolute_error_size_augmented"].to_numpy(dtype=float)
    observed = float(np.mean(augmented_error) - np.mean(baseline_error))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        selected = rng.integers(0, len(paired), size=len(paired))
        bootstrap[index] = float(
            np.mean(augmented_error[selected]) - np.mean(baseline_error[selected])
        )
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "delta_mae_augmented_minus_baseline": observed,
        "delta_mae_ci_low": float(low),
        "delta_mae_ci_high": float(high),
    }


def paired_comparison(
    model_predictions: pd.DataFrame,
    model_name: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    baseline = model_predictions[
        model_predictions["variant"].eq("baseline")
    ][["sample_id", "actual_age", "predicted_age", "absolute_error"]].rename(
        columns={
            "predicted_age": "predicted_age_baseline",
            "absolute_error": "absolute_error_baseline",
        }
    )
    augmented = model_predictions[
        model_predictions["variant"].eq("size_augmented")
    ][["sample_id", "actual_age", "predicted_age", "absolute_error"]].rename(
        columns={
            "actual_age": "actual_age_augmented",
            "predicted_age": "predicted_age_size_augmented",
            "absolute_error": "absolute_error_size_augmented",
        }
    )
    paired = baseline.merge(
        augmented, on="sample_id", how="inner", validate="one_to_one"
    )
    if len(paired) != len(baseline) or len(paired) != len(augmented):
        raise ValueError(f"{model_name} 的两种模型未产生完全配对的OOF预测。")

    paired["delta_absolute_error"] = (
        paired["absolute_error_size_augmented"]
        - paired["absolute_error_baseline"]
    )
    nonzero = paired["delta_absolute_error"].to_numpy(dtype=float)
    if len(nonzero) > 0 and not np.allclose(nonzero, 0):
        test = wilcoxon(
            paired["absolute_error_size_augmented"],
            paired["absolute_error_baseline"],
            alternative="two-sided",
            zero_method="wilcox",
        )
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        statistic = np.nan
        p_value = np.nan

    summary = {
        "model": model_name,
        "n_paired": len(paired),
        "mae_baseline": float(paired["absolute_error_baseline"].mean()),
        "mae_size_augmented": float(
            paired["absolute_error_size_augmented"].mean()
        ),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value": p_value,
    }
    summary.update(bootstrap_delta_mae(paired, n_bootstrap, seed))
    return paired, summary


def main() -> None:
    args = parse_args()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    package, excluded = prepare_normal_data()
    normal = package["normal_data"]
    raw_genes = package["gene_columns"]

    if not excluded.empty:
        excluded.reset_index().rename(
            columns={"index": "sample_id"}
        ).to_csv(
            output_root / "excluded_normal_samples_missing_size.csv", index=False
        )

    composition = (
        normal.groupby(["batch", "size_class"], observed=True)
        .agg(
            n=("Chronological_Age", "size"),
            n_breeds=("breed", "nunique"),
            age_min=("Chronological_Age", "min"),
            age_median=("Chronological_Age", "median"),
            age_max=("Chronological_Age", "max"),
        )
        .reset_index()
    )
    composition.to_csv(output_root / "study_by_size_composition.csv", index=False)

    predictions = []
    fold_metrics = []
    studies = sorted(normal["batch"].dropna().unique())
    for fold_index, held_out_study in enumerate(studies):
        fold_name = f"leave_study_{held_out_study}_out"
        train_raw = normal[~normal["batch"].eq(held_out_study)].copy()
        test_raw = normal[normal["batch"].eq(held_out_study)].copy()
        print(
            f"\n{fold_name}: train n={len(train_raw)}, test n={len(test_raw)}"
        )

        preprocessing_dir = output_root / "shared_preprocessing" / fold_name
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        train_expr, test_expr, genes = normalize_outer_fold(
            train_raw,
            test_raw,
            raw_genes,
            preprocessing_dir,
            fold_name,
        )
        train_augmented, test_augmented, size_features = add_size_features(
            train_expr, test_expr
        )

        for model_index, model_name in enumerate(args.models):
            for variant in VARIANTS:
                print(f"  - {model_name}: {variant}")
                model_fold_dir = (
                    output_root / model_name / variant / fold_name
                )
                model_fold_dir.mkdir(parents=True, exist_ok=True)
                if variant == "baseline":
                    train_df = train_expr
                    test_df = test_expr
                    features = genes
                else:
                    train_df = train_augmented
                    test_df = test_augmented
                    features = genes + size_features

                if model_name == "MoE":
                    predicted = train_moe(
                        train_df=train_df,
                        test_df=test_df,
                        genes=features,
                        fold_dir=model_fold_dir,
                        fold_name=f"{fold_name}_{variant}",
                        seed=args.seed + fold_index * 100 + model_index,
                        enable_optuna=not args.disable_moe_optuna,
                        moe_trials=args.moe_trials,
                    )
                else:
                    predicted = TRAINERS[model_name](
                        train_df,
                        test_df,
                        features,
                        model_fold_dir,
                        f"{fold_name}_{variant}",
                    )

                fold_prediction = make_prediction_frame(
                    test_df,
                    predicted,
                    model_name,
                    variant,
                    held_out_study,
                )
                fold_prediction.to_csv(
                    model_fold_dir / "predictions.csv", index=False
                )
                metrics = metric_values(fold_prediction)
                metrics.update(
                    {
                        "model": model_name,
                        "variant": variant,
                        "held_out_study": held_out_study,
                        "n_train": len(train_df),
                        "n_features": len(features),
                    }
                )
                pd.DataFrame([metrics]).to_csv(
                    model_fold_dir / "metrics.csv", index=False
                )
                predictions.append(fold_prediction)
                fold_metrics.append(metrics)

    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(
        output_root / "all_models_study_loo_predictions.csv", index=False
    )
    pd.DataFrame(fold_metrics).to_csv(
        output_root / "all_models_metrics_by_study.csv", index=False
    )

    overall_rows = []
    paired_rows = []
    for model_index, model_name in enumerate(args.models):
        model_predictions = all_predictions[
            all_predictions["model"].eq(model_name)
        ].copy()
        model_dir = output_root / model_name
        model_predictions.to_csv(
            model_dir / "paired_oof_predictions.csv", index=False
        )
        for variant in VARIANTS:
            subset = model_predictions[
                model_predictions["variant"].eq(variant)
            ]
            metrics = metric_values(subset)
            metrics.update({"model": model_name, "variant": variant})
            overall_rows.append(metrics)

        paired, summary = paired_comparison(
            model_predictions,
            model_name,
            n_bootstrap=args.bootstrap,
            seed=args.seed + model_index,
        )
        paired.to_csv(
            model_dir / "paired_sample_error_comparison.csv", index=False
        )
        paired_rows.append(summary)

    pd.DataFrame(overall_rows).to_csv(
        output_root / "overall_metrics_baseline_vs_size_augmented.csv",
        index=False,
    )
    pd.DataFrame(paired_rows).to_csv(
        output_root / "paired_delta_mae_tests.csv", index=False
    )
    run_metadata = {
        "models": args.models,
        "outer_validation": "Leave-One-Study-Out using batch",
        "variants": {
            "baseline": "gene expression only",
            "size_augmented": (
                "gene expression + one-hot(size_class_DAP), Small reference"
            ),
        },
        "size_categories": SIZE_ORDER,
        "normal_size_counts": {
            size: int(normal["size_class"].eq(size).sum())
            for size in SIZE_ORDER
        },
        "n_excluded_missing_size": int(len(excluded)),
        "moe_optuna_enabled": not args.disable_moe_optuna,
        "moe_trials_override": args.moe_trials,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n体型协变量配对比较完成: {output_root}")


if __name__ == "__main__":
    main()
