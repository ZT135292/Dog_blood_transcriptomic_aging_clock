"""细胞组成 × 衰老时钟分析：共享常量与工具函数。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CELL_TYPES = [
    "Neutrophil",
    "CD8/NK cell",
    "Monocyte",
    "Granulocyte",
    "CD4 T cell",
    "B cell",
    "DC",
]

CLR_COLS = [
    "CLR_Neutrophil",
    "CLR_CD8_NK_cell",
    "CLR_Monocyte",
    "CLR_Granulocyte",
    "CLR_CD4_T_cell",
    "CLR_B_cell",
    "CLR_DC",
]

PC_COLS = ["PC1", "PC2", "PC3"]

MODEL_RESULT_DIRS = {
    "ElasticNet": "aging_clock_results_Elastic_Net_pipeline_foldwise_TMM_logCPM",
    "LASSO": "aging_clock_results_LASSO_pipeline_foldwise_TMM_logCPM",
    "SVR": "aging_clock_results_SVR_pipeline_foldwise_TMM_logCPM",
    "RandomForest": "aging_clock_results_Random_Forest_pipeline_foldwise_TMM_logCPM",
    "XGBoost": "aging_clock_results_XGBoost_pipeline_foldwise_TMM_logCPM",
    "MoE": "aging_clock_results_MoE_pipeline_foldwise_TMM_logCPM",
}

CV_SCHEMES = ("group_loo", "DataSet_loo")

KNOWN_IMMUNE_MARKERS = {
    "CTSW": "Cytotoxic T / NK (granzyme)",
    "GZMK": "Cytotoxic T / NK (granzyme)",
    "GZMB": "Cytotoxic T / NK",
    "PRF1": "Cytotoxic lymphocyte",
    "KLRG1": "NK / activated T cell",
    "CD300E": "Myeloid / DC",
    "CD3D": "T cell",
    "CD3E": "T cell",
    "CD4": "CD4 T cell",
    "CD8A": "CD8 T cell",
    "CD8B": "CD8 T cell",
    "MS4A1": "B cell (CD20)",
    "CD19": "B cell",
    "CD14": "Monocyte",
    "LYZ": "Monocyte / myeloid",
    "FCGR3A": "Monocyte / NK",
    "TNF": "Inflammatory / myeloid",
    "CLEC12B": "Myeloid",
    "KLRF1": "NK cell",
    "CD69": "Activated T / NK",
    "CD80": "Antigen-presenting / B cell",
}


def script_paths() -> tuple[Path, Path]:
    cell_dir = Path(__file__).resolve().parent
    return cell_dir, cell_dir.parent


def compute_clr_from_fractions(df: pd.DataFrame) -> pd.DataFrame:
    """由 CIBERSORTx 细胞比例计算 CLR 列（缺失 CLR 文件时的回退方案）。"""
    out = df.copy()
    if not all(ct in out.columns for ct in CELL_TYPES):
        return out

    frac = out[CELL_TYPES].astype(float).clip(lower=1e-6)
    log_frac = np.log(frac)
    geo = log_frac.mean(axis=1)
    for ct, clr_col in zip(CELL_TYPES, CLR_COLS):
        out[clr_col] = log_frac[ct] - geo
    return out


def ensure_clr_columns(df: pd.DataFrame) -> pd.DataFrame:
    """确保存在 CLR_* 列；优先使用已有列，否则由细胞比例计算。"""
    if all(c in df.columns for c in CLR_COLS):
        return df
    return compute_clr_from_fractions(df)


def subset_normal_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Normal validation 子集；无 data_type 时回退到 condition == Normal。"""
    if "data_type" in df.columns:
        val = df[df["data_type"] == "validation"].copy()
        if not val.empty:
            return val
    if "condition" in df.columns:
        return df[df["condition"] == "Normal"].copy()
    return pd.DataFrame()


def safe_concat(rows: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [x for x in rows if x is not None and not x.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


def load_merged_composition(cell_dir: Path) -> pd.DataFrame:
    merged_path = cell_dir / "cell_composition_analysis" / "CIBERSORTx_merged_metadata.csv"
    if not merged_path.exists():
        raise FileNotFoundError(f"找不到 {merged_path}")

    df = pd.read_csv(merged_path)
    df["sample_id"] = df["sample_id"].astype(str)

    for fname in ("cell_composition_CLR_covariates.csv", "cell_composition_PC_covariates.csv"):
        aux_path = cell_dir / "cell_composition_analysis" / fname
        if aux_path.exists():
            aux = pd.read_csv(aux_path)
            aux["sample_id"] = aux["sample_id"].astype(str)
            extra = [c for c in aux.columns if c != "sample_id" and c not in df.columns]
            df = df.merge(aux[["sample_id"] + extra], on="sample_id", how="left")
    return ensure_clr_columns(df)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到预测文件: {path}")
    pred = pd.read_csv(path)
    pred["sample_id"] = pred["sample_id"].astype(str)
    if "Predicted_Age" not in pred.columns and "predicted_age" in pred.columns:
        pred["Predicted_Age"] = pred["predicted_age"]
    if "actual_age" not in pred.columns and "Chronological_Age" in pred.columns:
        pred["actual_age"] = pred["Chronological_Age"]
    if "age_acceleration" not in pred.columns:
        pred["age_acceleration"] = pred["Predicted_Age"] - pred["actual_age"]
    return pred


def merge_clock_and_composition(pred: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    comp_extra = [c for c in comp.columns if c != "sample_id" and c not in pred.columns]
    return pred.merge(comp[["sample_id"] + comp_extra], on="sample_id", how="inner")


def load_expression_matrix(path: Path) -> pd.DataFrame:
    """基因 × 样本 logCPM → 返回 样本 × 基因。"""
    mat = pd.read_csv(path, sep="\t", index_col=0)
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)
    return mat.T


def ols_table(y: pd.Series, X: pd.DataFrame, response_name: str = "y") -> pd.DataFrame:
    data = pd.concat([y.rename(response_name), X], axis=1).dropna()
    if len(data) < X.shape[1] + 2:
        return pd.DataFrame()

    y_vec = data[response_name].values.astype(float)
    x_mat = data[X.columns].values.astype(float)
    n, p = x_mat.shape
    x_design = np.column_stack([np.ones(n), x_mat])
    beta, _, _, _ = np.linalg.lstsq(x_design, y_vec, rcond=None)
    resid = y_vec - x_design @ beta
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y_vec - y_vec.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan

    if n > p + 1:
        sigma2 = rss / (n - p - 1)
        try:
            cov = sigma2 * np.linalg.inv(x_design.T @ x_design)
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            se = np.full(p + 1, np.nan)
    else:
        se = np.full(p + 1, np.nan)

    terms = ["Intercept"] + list(X.columns)
    rows = []
    for i, term in enumerate(terms):
        b = beta[i]
        s = se[i]
        t_stat = b / s if s and not np.isnan(s) and s > 0 else np.nan
        p_val = (
            2 * stats.t.sf(abs(t_stat), df=max(n - p - 1, 1))
            if not np.isnan(t_stat)
            else np.nan
        )
        rows.append(
            {
                "term": term,
                "estimate": b,
                "std_error": s,
                "t_stat": t_stat,
                "p_value": p_val,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["n"] = n
    out.attrs["r2"] = r2
    return out


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float).fillna(1.0).values
    n = len(p)
    if n == 0:
        return p_values
    order = np.argsort(p)
    ranked = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, p[order[i]] * n / rank)
        ranked[order[i]] = val
        prev = val
    return pd.Series(ranked, index=p_values.index)


def partial_correlation(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < Z.shape[1] + 3:
        return np.nan, np.nan
    z_design = np.column_stack([np.ones(n), Z])
    rx = x - z_design @ np.linalg.lstsq(z_design, x, rcond=None)[0]
    ry = y - z_design @ np.linalg.lstsq(z_design, y, rcond=None)[0]
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan, np.nan
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)


def composition_adjust_residuals(
    values: pd.Series,
    covariates: pd.DataFrame,
) -> pd.Series:
    """values ~ covariates，返回残差（保持原索引均值）。"""
    valid = pd.concat([values.rename("y"), covariates], axis=1).dropna()
    if valid.empty or len(valid) < covariates.shape[1] + 3:
        return values.copy()
    fit = ols_table(valid["y"], valid[covariates.columns])
    if fit.empty:
        return values.copy()
    beta = fit.set_index("term")["estimate"]
    intercept = beta.get("Intercept", 0.0)
    pred = intercept + valid[covariates.columns].values @ beta.reindex(covariates.columns).fillna(0).values
    resid = values.copy()
    resid.loc[valid.index] = valid["y"].values - pred + values.loc[valid.index].mean()
    return resid


def discover_prediction_files(parent_dir: Path, cv_scheme: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for model, result_dir in MODEL_RESULT_DIRS.items():
        path = parent_dir / result_dir / cv_scheme / "summary_all_predictions_cv.csv"
        if path.exists():
            out[model] = path
    return out


def make_group_dummies(groups: pd.Series, reference: str = "Baseline") -> pd.DataFrame:
    cats = sorted(groups.dropna().unique())
    if reference in cats:
        cats = [reference] + [c for c in cats if c != reference]
    dummies = pd.get_dummies(groups.astype(str), drop_first=True)
    # 确保 reference 被 drop
    if reference in groups.astype(str).unique() and reference in dummies.columns:
        dummies = dummies.drop(columns=[reference], errors="ignore")
    return dummies.astype(float)
