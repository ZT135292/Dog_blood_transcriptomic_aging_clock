from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "model_stability_complexity_analysis"
OUT_DIR.mkdir(exist_ok=True)

MODELS = ["Elastic_Net", "LASSO", "MoE", "Random_Forest", "SVR", "XGBoost"]
MODEL_LABELS = {
    "Elastic_Net": "Elastic Net",
    "LASSO": "LASSO",
    "MoE": "MoE",
    "Random_Forest": "Random Forest",
    "SVR": "SVR",
    "XGBoost": "XGBoost",
}
MODEL_DIRS = {
    "Elastic_Net": "aging_clock_results_elastic_net_pipeline_foldwise_TMM_logCPM",
    "LASSO": "aging_clock_results_lasso_pipeline_foldwise_TMM_logCPM",
    "MoE": "aging_clock_results_MoE_pipeline_foldwise_TMM_logCPM",
    "Random_Forest": "aging_clock_results_random_forest_pipeline_foldwise_TMM_logCPM",
    "SVR": "aging_clock_results_svr_pipeline_foldwise_TMM_logCPM",
    "XGBoost": "aging_clock_results_xgboost_pipeline_foldwise_TMM_logCPM",
}
EXPRESSION_FILE_CANDIDATES = [
    "total_count_sort_ortholog_TMM_batched.txt",
    "total_count_sort_ortholog.txt",
]
SCHEMES = {
    "5-fold CV": {
        "metrics": "5fold_cv/summary_metrics_by_fold.csv",
        "predictions": "5fold_cv/summary_all_predictions_cv.csv",
    },
    "Group-LOO": {
        "metrics": "group_loo/summary_metrics_by_group.csv",
        "predictions": "group_loo/summary_all_predictions_cv.csv",
    },
    "Dataset-LOO": {
        "metrics": "DataSet_loo/summary_metrics_by_batch.csv",
        "predictions": "DataSet_loo/summary_all_predictions_cv.csv",
    },
}


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def bootstrap_ci(y_true, y_pred, metric_func, n_boot=5000, seed=66):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n < 2:
        return np.nan, np.nan

    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if metric_func is r2 and np.all(y_true[idx] == y_true[idx][0]):
            continue
        values.append(metric_func(y_true[idx], y_pred[idx]))

    if not values:
        return np.nan, np.nan
    return tuple(np.percentile(values, [2.5, 97.5]))


def normalize_prediction_columns(df):
    df = df.copy()
    if "actual_age" not in df.columns and "Chronological_Age" in df.columns:
        df["actual_age"] = df["Chronological_Age"]
    if "predicted_age" not in df.columns and "Predicted_Age" in df.columns:
        df["predicted_age"] = df["Predicted_Age"]
    return df


def get_validation_predictions(pred_path):
    df = normalize_prediction_columns(pd.read_csv(pred_path))
    if "data_type" in df.columns:
        df = df[df["data_type"].astype(str).str.lower() == "validation"].copy()
    if "condition" in df.columns:
        df = df[df["condition"].astype(str) == "Normal"].copy()
    return df.dropna(subset=["actual_age", "predicted_age"]).copy()


def resolve_n_features():
    for filename in EXPRESSION_FILE_CANDIDATES:
        path = ROOT / filename
        if path.exists():
            return len(pd.read_csv(path, sep="\t", usecols=[0])), filename
    return np.nan, ""


def summarize_performance():
    rows = []
    fold_rows = []

    for model in MODELS:
        model_dir = ROOT / MODEL_DIRS[model]
        for scheme, paths in SCHEMES.items():
            metrics_path = model_dir / paths["metrics"]
            pred_path = model_dir / paths["predictions"]
            if not metrics_path.exists() or not pred_path.exists():
                continue

            metrics = pd.read_csv(metrics_path)
            preds = get_validation_predictions(pred_path)
            y_true = preds["actual_age"].astype(float)
            y_pred = preds["predicted_age"].astype(float)

            pooled_mae = mae(y_true, y_pred)
            pooled_r2 = r2(y_true, y_pred)
            mae_low, mae_high = bootstrap_ci(y_true, y_pred, mae)
            r2_low, r2_high = bootstrap_ci(y_true, y_pred, r2)

            val_mae = metrics["val_mae"].astype(float)
            val_r2 = metrics["val_r2"].astype(float)
            train_mae = metrics["train_mae"].astype(float)
            train_r2 = metrics["train_r2"].astype(float)

            rows.append({
                "model": model,
                "model_label": MODEL_LABELS[model],
                "validation_scheme": scheme,
                "n_validation_samples_pooled": len(preds),
                "n_folds_or_groups": len(metrics),
                "pooled_mae": pooled_mae,
                "pooled_mae_ci95_low": mae_low,
                "pooled_mae_ci95_high": mae_high,
                "pooled_r2": pooled_r2,
                "pooled_r2_ci95_low": r2_low,
                "pooled_r2_ci95_high": r2_high,
                "fold_val_mae_mean": val_mae.mean(),
                "fold_val_mae_sd": val_mae.std(ddof=1),
                "fold_val_mae_median": val_mae.median(),
                "fold_val_mae_q25": val_mae.quantile(0.25),
                "fold_val_mae_q75": val_mae.quantile(0.75),
                "fold_val_mae_iqr": val_mae.quantile(0.75) - val_mae.quantile(0.25),
                "fold_val_r2_mean": val_r2.mean(),
                "fold_val_r2_sd": val_r2.std(ddof=1),
                "fold_train_mae_mean": train_mae.mean(),
                "fold_train_mae_sd": train_mae.std(ddof=1),
                "fold_train_r2_mean": train_r2.mean(),
                "fold_train_r2_sd": train_r2.std(ddof=1),
                "train_val_mae_gap": val_mae.mean() - train_mae.mean(),
                "min_fold_n_val": metrics["n_val"].min() if "n_val" in metrics.columns else np.nan,
                "max_fold_n_val": metrics["n_val"].max() if "n_val" in metrics.columns else np.nan,
            })

            for _, row in metrics.iterrows():
                fold_rows.append({
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "validation_scheme": scheme,
                    "fold": row.get("fold", ""),
                    "train_mae": row["train_mae"],
                    "val_mae": row["val_mae"],
                    "train_r2": row["train_r2"],
                    "val_r2": row["val_r2"],
                    "n_val": row.get("n_val", np.nan),
                })

    summary = pd.DataFrame(rows)
    fold_level = pd.DataFrame(fold_rows)
    summary.to_csv(OUT_DIR / "model_performance_stability_summary.csv", index=False, encoding="utf-8-sig")
    fold_level.to_csv(OUT_DIR / "model_performance_fold_level_values.csv", index=False, encoding="utf-8-sig")
    return summary, fold_level


def parse_hpo_report(model_dir):
    out = {}
    pred_path = model_dir / "HPO" / "predictions_HPO_all_normal.csv"
    if pred_path.exists():
        pred_df = normalize_prediction_columns(pd.read_csv(pred_path))
        if "data_type" in pred_df.columns:
            for data_type, prefix in [("train", "hpo_train"), ("validation", "hpo_val")]:
                part = pred_df[pred_df["data_type"].astype(str).str.lower() == data_type].copy()
                if not part.empty:
                    out[f"{prefix}_mae"] = mae(part["actual_age"], part["predicted_age"])
                    out[f"{prefix}_r2"] = r2(part["actual_age"], part["predicted_age"])

    report_files = list((model_dir / "HPO").glob("report_hpo*.txt"))
    if not report_files:
        return out
    text = report_files[0].read_text(encoding="utf-8", errors="ignore")
    patterns = {
        "hpo_train_mae": r"Train MAE:\s*([0-9.]+)",
        "hpo_train_r2": r"Train MAE:[^\n]*R2:\s*([-0-9.]+)",
        "hpo_val_mae": r"Val\s+MAE:\s*([0-9.]+)",
        "hpo_val_r2": r"Val\s+MAE:[^\n]*R2:\s*([-0-9.]+)",
        "hpo_train_mae_report": r"Training Performance[\s\S]*?MAE:\s*([0-9.]+)",
        "hpo_train_r2_report": r"Training Performance[\s\S]*?R2\s*:\s*([-0-9.]+)",
        "n_genes_after_tmm": r"Total genes used \(after TMM filtering\):\s*(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            if key == "hpo_train_mae_report" and "hpo_train_mae" not in out:
                out["hpo_train_mae"] = float(match.group(1))
            elif key == "hpo_train_r2_report" and "hpo_train_r2" not in out:
                out["hpo_train_r2"] = float(match.group(1))
            elif key == "n_genes_after_tmm":
                out[key] = int(match.group(1))
            elif not key.endswith("_report"):
                out[key] = float(match.group(1))
    return out


def count_nonzero_feature_file(path):
    if not path.exists():
        return np.nan, np.nan
    df = pd.read_csv(path)
    if "coefficient" in df.columns:
        values = df["coefficient"].astype(float)
    elif "importance" in df.columns:
        values = df["importance"].astype(float)
    else:
        return len(df), np.nan
    return int((values.abs() > 0).sum()), len(df)


def linear(in_dim, out_dim):
    return in_dim * out_dim + out_dim


def batchnorm(dim):
    return 2 * dim


def resolve_moe_input_dim(total_ranked_features, hpo_report, n_features_raw):
    """MoE 实际输入维度：优先 HPO 过滤后基因数，而非原始表达矩阵基因数。"""
    if pd.notna(total_ranked_features):
        return int(total_ranked_features), "HPO feature file (after TMM filtering)"
    n_from_report = hpo_report.get("n_genes_after_tmm")
    if n_from_report is not None and not pd.isna(n_from_report):
        return int(n_from_report), "HPO report (after TMM filtering)"
    if pd.notna(n_features_raw):
        return int(n_features_raw), "raw expression matrix (fallback)"
    return np.nan, ""


def estimate_moe_parameters(input_dim, num_experts):
    # Mirrors the architecture in MoE.py:
    # input -> 640 -> 256 -> 128 -> 64 -> output, with BatchNorm after first 3 hidden layers.
    expert = (
        linear(input_dim, 640) + batchnorm(640)
        + linear(640, 256) + batchnorm(256)
        + linear(256, 128) + batchnorm(128)
        + linear(128, 64)
        + linear(64, 1)
    )
    gating = (
        linear(input_dim, 640) + batchnorm(640)
        + linear(640, 256) + batchnorm(256)
        + linear(256, 128) + batchnorm(128)
        + linear(128, 64)
        + linear(64, num_experts)
    )
    return expert * num_experts + gating


def summarize_complexity():
    n_features_raw, expression_source = resolve_n_features()
    metadata_path = ROOT / "total_metadata.txt"
    if metadata_path.exists():
        meta = pd.read_csv(metadata_path, sep="\t")
        n_normal_metadata = int((meta["condition"] == "Normal").sum()) if "condition" in meta.columns else np.nan
    else:
        n_normal_metadata = np.nan

    rows = []
    for model in MODELS:
        model_dir = ROOT / MODEL_DIRS[model]
        hpo_dir = model_dir / "HPO"
        hpo = parse_hpo_report(model_dir)

        best_params_path = hpo_dir / "best_hyperparameters.json"
        best_params = {}
        if best_params_path.exists():
            best_params = json.loads(best_params_path.read_text(encoding="utf-8"))

        feature_file_candidates = [
            hpo_dir / "coefficients_HPO_all_normal.csv",
            hpo_dir / "gene_importances_HPO_all_normal.csv",
        ]
        nonzero_features, total_ranked_features = np.nan, np.nan
        for candidate in feature_file_candidates:
            if candidate.exists():
                nonzero_features, total_ranked_features = count_nonzero_feature_file(candidate)
                break

        n_features_model_input = (
            total_ranked_features if pd.notna(total_ranked_features) else n_features_raw
        )

        estimated_trainable_parameters = np.nan
        complexity_note = ""
        if model == "MoE":
            num_experts = int(best_params.get("num_experts", 2))
            feature_dim, source_note = resolve_moe_input_dim(
                total_ranked_features, hpo, n_features_raw
            )
            n_features_model_input = feature_dim
            if pd.notna(feature_dim):
                estimated_trainable_parameters = estimate_moe_parameters(int(feature_dim), num_experts)
                complexity_note = (
                    f"Estimated from MoE.py architecture, num_experts={num_experts}, "
                    f"input_dim={int(feature_dim)} ({source_note})"
                )
            else:
                complexity_note = "Could not estimate MoE parameters: expression/feature count unavailable"
        elif model in {"Elastic_Net", "LASSO", "SVR"}:
            complexity_note = "Effective complexity summarized by non-zero linear coefficients"
        elif model == "Random_Forest":
            complexity_note = "Tree ensemble complexity summarized by HPO parameters and non-zero feature importances"
        elif model == "XGBoost":
            complexity_note = "Boosted-tree complexity summarized by HPO parameters and non-zero feature importances"

        rows.append({
            "model": model,
            "model_label": MODEL_LABELS[model],
            "n_normal_samples_in_metadata": n_normal_metadata,
            "n_features_raw": n_features_raw,
            "n_features_model_input": n_features_model_input,
            "expression_file_used": expression_source,
            "nonzero_features_or_importances": nonzero_features,
            "total_ranked_features": total_ranked_features,
            "estimated_trainable_parameters_moe_only": estimated_trainable_parameters,
            "hpo_train_mae": hpo.get("hpo_train_mae", np.nan),
            "hpo_val_mae": hpo.get("hpo_val_mae", np.nan),
            "hpo_train_r2": hpo.get("hpo_train_r2", np.nan),
            "hpo_val_r2": hpo.get("hpo_val_r2", np.nan),
            "hpo_train_val_mae_gap": hpo.get("hpo_val_mae", np.nan) - hpo.get("hpo_train_mae", np.nan),
            "best_hyperparameters_json": json.dumps(best_params, ensure_ascii=False),
            "complexity_note": complexity_note,
        })

    complexity = pd.DataFrame(rows)
    complexity.to_csv(OUT_DIR / "model_complexity_and_hpo_overfitting_summary.csv", index=False, encoding="utf-8-sig")
    return complexity


def save_figure(fig, basename):
    """Save each figure as both high-resolution PNG and vector PDF."""
    fig.savefig(OUT_DIR / f"{basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{basename}.pdf", bbox_inches="tight")


def plot_mae_ci(summary):
    schemes = list(SCHEMES.keys())
    fig, axes = plt.subplots(1, len(schemes), figsize=(18, 5), sharey=True)
    for ax, scheme in zip(axes, schemes):
        data = summary[summary["validation_scheme"] == scheme].copy()
        data["model_label"] = pd.Categorical(data["model_label"], [MODEL_LABELS[m] for m in MODELS])
        data = data.sort_values("model_label")
        x = np.arange(len(data))
        y = data["pooled_mae"].values
        yerr = np.vstack([
            y - data["pooled_mae_ci95_low"].values,
            data["pooled_mae_ci95_high"].values - y,
        ])
        ax.bar(x, y, color="#4472C4", alpha=0.85)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=4, linewidth=1)
        ax.set_title(scheme)
        ax.set_xticks(x)
        ax.set_xticklabels(data["model_label"], rotation=45, ha="right")
        ax.set_ylabel("MAE (years)" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Pooled Out-of-Sample MAE with Bootstrap 95% CI", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "pooled_mae_bootstrap_ci")
    plt.close(fig)


def plot_r2_ci(summary):
    schemes = list(SCHEMES.keys())
    fig, axes = plt.subplots(1, len(schemes), figsize=(18, 5), sharey=True)
    for ax, scheme in zip(axes, schemes):
        data = summary[summary["validation_scheme"] == scheme].copy()
        data["model_label"] = pd.Categorical(data["model_label"], [MODEL_LABELS[m] for m in MODELS])
        data = data.sort_values("model_label")
        x = np.arange(len(data))
        y = data["pooled_r2"].values
        yerr = np.vstack([
            y - data["pooled_r2_ci95_low"].values,
            data["pooled_r2_ci95_high"].values - y,
        ])
        ax.bar(x, y, color="#70AD47", alpha=0.85)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=4, linewidth=1)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_title(scheme)
        ax.set_xticks(x)
        ax.set_xticklabels(data["model_label"], rotation=45, ha="right")
        ax.set_ylabel("R²" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Pooled Out-of-Sample R² with Bootstrap 95% CI", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "pooled_r2_bootstrap_ci")
    plt.close(fig)


def plot_fold_mae_distribution(fold_level):
    schemes = list(SCHEMES.keys())
    fig, axes = plt.subplots(1, len(schemes), figsize=(18, 5), sharey=True)
    labels = [MODEL_LABELS[m] for m in MODELS]
    for ax, scheme in zip(axes, schemes):
        data = fold_level[fold_level["validation_scheme"] == scheme].copy()
        values = [data.loc[data["model"] == m, "val_mae"].astype(float).values for m in MODELS]
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        for i, vals in enumerate(values, 1):
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) else []
            ax.scatter(np.full(len(vals), i) + jitter, vals, color="#ED7D31", s=22, alpha=0.8, zorder=3)
        ax.set_title(scheme)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Fold/group validation MAE (years)" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Fold-Level Validation MAE Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fold_level_val_mae_distribution")
    plt.close(fig)


def plot_train_val_gap(summary):
    data = summary[summary["validation_scheme"].isin(["5-fold CV", "Group-LOO"])].copy()
    labels = [MODEL_LABELS[m] for m in MODELS]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, scheme in enumerate(["5-fold CV", "Group-LOO"]):
        vals = []
        for m in MODELS:
            row = data[(data["model"] == m) & (data["validation_scheme"] == scheme)]
            vals.append(row["train_val_mae_gap"].iloc[0] if not row.empty else np.nan)
        ax.bar(x + (j - 0.5) * width, vals, width=width, label=scheme)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean validation MAE - mean training MAE")
    ax.set_title("Train-Validation MAE Gap")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "train_validation_mae_gap")
    plt.close(fig)


def plot_hpo_train_val(complexity):
    labels = [MODEL_LABELS[m] for m in MODELS]
    x = np.arange(len(labels))
    width = 0.38
    data = complexity.set_index("model")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    train = [data.loc[m, "hpo_train_mae"] for m in MODELS]
    val = [data.loc[m, "hpo_val_mae"] for m in MODELS]
    ax.bar(x - width / 2, train, width=width, label="HPO train MAE", color="#A5A5A5")
    ax.bar(x + width / 2, val, width=width, label="HPO validation MAE", color="#4472C4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("MAE (years)")
    ax.set_title("HPO Train vs Validation MAE")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "hpo_train_vs_validation_mae")
    plt.close(fig)


def plot_complexity(complexity):
    labels = [MODEL_LABELS[m] for m in MODELS]
    data = complexity.set_index("model")
    feature_counts = [data.loc[m, "nonzero_features_or_importances"] for m in MODELS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(labels, feature_counts, color="#70AD47")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Non-zero coefficients/importances (log scale)")
    axes[0].set_title("Effective Feature Count")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.25)

    moe_params = data.loc["MoE", "estimated_trainable_parameters_moe_only"]
    n_samples = data.loc["MoE", "n_normal_samples_in_metadata"]
    n_features = data.loc["MoE", "n_features_model_input"]
    axes[1].bar(["Normal samples", "Genes (MoE input)", "MoE parameters"], [n_samples, n_features, moe_params],
                color=["#A5A5A5", "#ED7D31", "#C00000"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Count (log scale)")
    axes[1].set_title("MoE Complexity Relative to Data Size")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    save_figure(fig, "model_complexity_overview")
    plt.close(fig)


def main():
    summary, fold_level = summarize_performance()
    complexity = summarize_complexity()

    plot_mae_ci(summary)
    plot_r2_ci(summary)
    plot_fold_mae_distribution(fold_level)
    plot_train_val_gap(summary)
    plot_hpo_train_val(complexity)
    plot_complexity(complexity)

    print(f"Output directory: {OUT_DIR}")
    print("\nPerformance summary:")
    print(summary.round(3).to_string(index=False))
    print("\nComplexity/HPO overfitting summary:")
    display_cols = [
        "model_label", "n_normal_samples_in_metadata", "n_features_raw", "n_features_model_input",
        "nonzero_features_or_importances", "estimated_trainable_parameters_moe_only",
        "hpo_train_mae", "hpo_val_mae", "hpo_train_r2", "hpo_val_r2",
        "hpo_train_val_mae_gap",
    ]
    print(complexity[display_cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
