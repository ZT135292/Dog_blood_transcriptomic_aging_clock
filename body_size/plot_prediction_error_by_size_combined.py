"""Combine six models' LOSO prediction-error-by-size plots into one 3×2 figure."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = ["LASSO", "ElasticNet", "RandomForest", "SVR", "XGBoost", "MoE"]
SIZE_ORDER = ["Small", "Medium", "Standard", "Large", "Giant"]
OUTPUT_STEM = "prediction_error_by_size_combined"
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 20


def load_model_names(script_dir: Path) -> list[str]:
    metadata_path = script_dir / "run_metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        return sorted(metadata.get("models", DEFAULT_MODELS))
    return sorted(DEFAULT_MODELS)


def load_model_predictions(script_dir: Path, model_name: str) -> pd.DataFrame | None:
    csv_path = script_dir / model_name / "all_oof_predictions.csv"
    if not csv_path.exists():
        print(f"警告: 跳过 {model_name}，找不到文件: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        if "actual_age" not in df.columns and "Chronological_Age" in df.columns:
            df["actual_age"] = df["Chronological_Age"]
        if "predicted_age" not in df.columns and "Predicted_Age" in df.columns:
            df["predicted_age"] = df["Predicted_Age"]
        if "size_class" not in df.columns and "Size_Class" in df.columns:
            df["size_class"] = df["Size_Class"]

        required_cols = ["actual_age", "predicted_age", "condition", "size_class"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"警告: {model_name} 缺少必要列 {missing}，跳过")
            return None

        df["actual_age"] = pd.to_numeric(df["actual_age"], errors="coerce")
        df["predicted_age"] = pd.to_numeric(df["predicted_age"], errors="coerce")
        df["condition"] = df["condition"].astype(str).str.strip()
        df["size_class"] = df["size_class"].astype(str).str.strip()
        df = df.dropna(subset=["actual_age", "predicted_age", "condition", "size_class"])

        if "prediction_error" not in df.columns:
            df["prediction_error"] = df["predicted_age"] - df["actual_age"]

        normal_df = df[df["condition"] == "Normal"].copy()
        if normal_df.empty:
            print(f"警告: {model_name} 中没有 Normal 样本，跳过")
            return None

        present_sizes = [size for size in SIZE_ORDER if normal_df["size_class"].eq(size).any()]
        normal_df["size_class"] = pd.Categorical(
            normal_df["size_class"], categories=present_sizes, ordered=True
        )
        print(f"成功读取 {model_name}: {len(normal_df)} 个 Normal 样本")
        return normal_df

    except Exception as exc:
        print(f"警告: 读取 {model_name} 时出错: {exc}，跳过")
        return None


def plot_prediction_error_panel(ax: plt.Axes, model_name: str, data: pd.DataFrame) -> None:
    present_sizes = [size for size in SIZE_ORDER if data["size_class"].eq(size).any()]

    sns.boxplot(
        data=data,
        x="size_class",
        y="prediction_error",
        order=present_sizes,
        hue="size_class",
        palette="Set1",
        legend=False,
        ax=ax,
        linewidth=1.2,
    )
    sns.stripplot(
        data=data,
        x="size_class",
        y="prediction_error",
        order=present_sizes,
        color="black",
        alpha=0.55,
        size=4,
        ax=ax,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(model_name, fontsize=TITLE_SIZE)
    ax.set_xlabel("Held-out body-size class", fontsize=LABEL_SIZE)
    ax.set_ylabel("Predicted age − chronological age (years)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)


def main() -> None:
    model_names = load_model_names(SCRIPT_DIR)
    print(f"将绘制以下 {len(model_names)} 个模型: {model_names}")

    data_list: list[tuple[str, pd.DataFrame]] = []
    for model_name in model_names:
        predictions = load_model_predictions(SCRIPT_DIR, model_name)
        if predictions is not None:
            data_list.append((model_name, predictions))

    if not data_list:
        raise SystemExit("错误: 没有成功读取任何数据，无法作图。")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()

    for idx, (model_name, predictions) in enumerate(data_list[:6]):
        plot_prediction_error_panel(axes[idx], model_name, predictions)

    for idx in range(len(data_list), 6):
        axes[idx].set_visible(False)

    all_errors = pd.concat([df["prediction_error"] for _, df in data_list[:6]], ignore_index=True)
    y_limit = float(np.ceil(max(abs(all_errors.min()), abs(all_errors.max()))))
    y_limit = max(y_limit, 5.0)

    for idx in range(min(len(data_list), 6)):
        axes[idx].set_ylim(-y_limit, y_limit)

    plt.tight_layout()

    output_pdf = SCRIPT_DIR / f"{OUTPUT_STEM}.pdf"
    output_png = SCRIPT_DIR / f"{OUTPUT_STEM}.png"

    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(output_png, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"\n合并图已保存为: {output_pdf}")
    print(f"合并图已保存为: {output_png}")


if __name__ == "__main__":
    main()
