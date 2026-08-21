import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""Plot leave-one-size-out OOF predicted vs chronological age for six models."""

script_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODELS = ["LASSO", "ElasticNet", "RandomForest", "SVR", "XGBoost", "MoE"]
metadata_path = os.path.join(script_dir, "run_metadata.json")

if os.path.exists(metadata_path):
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    model_names = metadata.get("models", DEFAULT_MODELS)
else:
    model_names = DEFAULT_MODELS

model_names = sorted(model_names)
print(f"将绘制以下 {len(model_names)} 个模型的 size-LOO 结果（按首字母排序）: {model_names}")

data_list = []

for model_name in model_names:
    csv_file_path = os.path.join(script_dir, model_name, "all_oof_predictions.csv")

    if not os.path.exists(csv_file_path):
        print(f"警告: 跳过 {model_name}，找不到文件: {csv_file_path}")
        continue

    try:
        df = pd.read_csv(csv_file_path)
        df.columns = df.columns.str.strip()

        if "Chronological_Age" not in df.columns and "actual_age" in df.columns:
            df["Chronological_Age"] = df["actual_age"]

        if "Predicted_Age" not in df.columns and "predicted_age" in df.columns:
            df["Predicted_Age"] = df["predicted_age"]

        required_cols = ["Chronological_Age", "Predicted_Age", "condition"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"警告: {model_name} 的 CSV 中缺少必要列: {missing}，跳过")
            continue

        df["Chronological_Age"] = pd.to_numeric(df["Chronological_Age"], errors="coerce")
        df["Predicted_Age"] = pd.to_numeric(df["Predicted_Age"], errors="coerce")
        df["condition"] = df["condition"].astype(str).str.strip()
        df = df.dropna(subset=["Chronological_Age", "Predicted_Age", "condition"])

        normal_df = df[df["condition"] == "Normal"].copy()

        if normal_df.empty:
            print(f"警告: {model_name} 中没有找到 'Normal' 样本，跳过")
            continue

        data_list.append((model_name, normal_df))
        print(f"成功读取 {model_name}: {len(normal_df)} 个 Normal 样本")

    except Exception as e:
        print(f"警告: 读取 {model_name} 时出错: {e}，跳过")
        continue

if len(data_list) == 0:
    print("错误: 没有成功读取任何数据，无法作图。")
    raise SystemExit(1)

print(f"\n共准备绘制 {len(data_list)} 张子图")

sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(24, 16))
axes = axes.flatten()

for idx, (model_name, normal_df) in enumerate(data_list):
    if idx >= 6:
        print("警告: 数据超过 6 个，只绘制前 6 个")
        break

    ax = axes[idx]
    y_true = normal_df["Chronological_Age"].values
    y_pred = normal_df["Predicted_Age"].values

    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    sns.regplot(
        x="Chronological_Age",
        y="Predicted_Age",
        data=normal_df,
        scatter_kws={"alpha": 0.8, "s": 80},
        line_kws={"alpha": 0.9},
        color="#3182bd",
        ax=ax,
    )

    ax.set_xlabel("Chronological Age (years)", fontsize=20)
    ax.set_ylabel("Predicted Age (years)", fontsize=20)
    ax.set_title(f"{model_name}_size_loo", fontsize=22)

    min_age = float(normal_df["Chronological_Age"].min())
    max_age = float(normal_df["Chronological_Age"].max())
    ax.plot(
        [min_age, max_age],
        [min_age, max_age],
        linestyle="--",
        color="gray",
        label="y = x (Reference)",
    )

    if ax.lines:
        ax.lines[0].set_label(f"Fitted (MAE={mae:.2f}, R²={r2:.2f})")

    ax.legend(loc="upper left", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=18)

for idx in range(len(data_list), 6):
    axes[idx].set_visible(False)

plt.tight_layout()

output_pdf = os.path.join(script_dir, "normal_samples_age_scatter_combined_size_loo.pdf")
try:
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    print(f"\n合并图已保存为: {output_pdf}")
except Exception as e:
    print(f"保存 PDF 失败: {e}")

output_png = os.path.join(script_dir, "normal_samples_age_scatter_combined_size_loo.png")
try:
    plt.savefig(output_png, format="png", bbox_inches="tight", dpi=300)
    print(f"合并图已保存为: {output_png}")
except Exception as e:
    print(f"保存 PNG 失败: {e}")

plt.show()
