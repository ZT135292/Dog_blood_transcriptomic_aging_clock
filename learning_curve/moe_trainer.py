"""MoE trainer for the learning-curve analysis.

Nested protocol matches model_training/MoE.py:
  inner-train / inner-validation split -> optional Optuna + early stopping
  -> refit on the full outer training subset with a fixed epoch count.
The outer test fold is never used for HPO or early stopping.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class Expert(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)


class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim) for _ in range(num_experts)])
        self.gating = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        gating_weights = self.gating(x)
        expert_outputs = torch.cat([expert(x) for expert in self.experts], dim=1)
        final_output = torch.sum(gating_weights * expert_outputs, dim=1, keepdim=True)
        return final_output, gating_weights, expert_outputs


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_aging_gene_indices(gene_columns, aging_gene_pool):
    return np.array([i for i, gene in enumerate(gene_columns) if gene in aging_gene_pool])


def _predict_raw(model, scaler_x, scaler_y, x_raw: pd.DataFrame) -> np.ndarray:
    device = next(model.parameters()).device
    x = x_raw.copy()
    if x.isnull().any().any():
        x = x.fillna(x.mean())
    x_scaled = scaler_x.transform(x)
    if np.isnan(x_scaled).any():
        x_scaled = np.nan_to_num(x_scaled, nan=0.0)
    xt = torch.tensor(x_scaled, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        y_scaled, _, _ = model(xt)
    return scaler_y.inverse_transform(y_scaled.cpu().numpy()).reshape(-1)


def _evaluate_mae(model, scaler_x, scaler_y, x_val, y_val) -> float:
    from sklearn.metrics import mean_absolute_error

    pred = _predict_raw(model, scaler_x, scaler_y, x_val)
    return float(mean_absolute_error(np.asarray(y_val, dtype=float), pred))


def train_moe_clock(data_splits, config):
    x_train_raw = data_splits["X_train_raw"]
    y_train = data_splits["y_train"]
    x_val_raw = data_splits.get("X_val_raw")
    y_val = data_splits.get("y_val")
    has_validation = x_val_raw is not None and y_val is not None and len(y_val) > 0
    aging_gene_indices = data_splits["aging_gene_indices"]

    scaler_x = StandardScaler().fit(x_train_raw)
    scaler_y = StandardScaler().fit(np.asarray(y_train, dtype=float).reshape(-1, 1))
    x_train = scaler_x.transform(x_train_raw)
    y_train_scaled = scaler_y.transform(np.asarray(y_train, dtype=float).reshape(-1, 1))

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(x_train_t, y_train_t)
    effective_batch = min(int(config["batch_size"]), len(train_dataset))
    if effective_batch < 2:
        raise ValueError("MoE needs at least 2 training samples (BatchNorm).")
    drop_last = len(train_dataset) > effective_batch and len(train_dataset) % effective_batch == 1
    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch, shuffle=True, drop_last=drop_last
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoE(x_train.shape[1], int(config["num_experts"])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
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
    aging_idx_t = torch.tensor(aging_gene_indices, dtype=torch.long).to(device)

    best_mae = float("inf")
    best_state = None
    best_epoch = int(config["epochs"])
    no_improve = 0
    es_cfg = config.get("early_stopping", {})
    es_enabled = has_validation and es_cfg.get("enabled", False)
    monitor_interval = max(1, int(es_cfg.get("monitor_interval", 1)))
    epochs_trained = 0

    for epoch in range(int(config["epochs"])):
        epochs_trained = epoch + 1
        model.train()
        last_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x.requires_grad_(True)
            optimizer.zero_grad()
            y_pred, _, _ = model(batch_x)
            pred_loss = F.l1_loss(y_pred, batch_y)
            if batch_x.grad is not None:
                batch_x.grad.zero_()
            saliency = torch.autograd.grad(
                outputs=y_pred.sum(), inputs=batch_x, create_graph=True
            )[0]
            gene_imp = saliency.abs().sum(dim=0)
            ahk = -(gene_imp[aging_idx_t].sum() / (gene_imp.sum() + 1e-10))
            loss = float(config["lambda_pred_loss"]) * pred_loss + float(config["gamma_ahk_loss"]) * ahk
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            last_loss = float(loss.item())

        if has_validation and (epoch + 1) % monitor_interval == 0:
            val_mae = _evaluate_mae(model, scaler_x, scaler_y, x_val_raw, y_val)
            if scheduler is not None:
                scheduler.step(val_mae)
            improved = (best_mae - val_mae) > es_cfg.get("min_delta", 0.0)
            if improved:
                best_mae = val_mae
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if es_enabled and no_improve >= es_cfg.get("patience", 15):
                break

    if es_enabled and es_cfg.get("restore_best", True) and best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler_x, scaler_y, {
        "best_epoch": int(best_epoch),
        "epochs_trained": int(epochs_trained),
        "best_val_mae": float(best_mae) if has_validation else None,
        "last_loss": last_loss,
    }


def optuna_search_hpo(data_splits, base_config):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        warnings.warn("Optuna is not installed; MoE will use default hyperparameters.")
        return {}

    def objective(trial):
        cfg = copy.deepcopy(base_config)
        cfg.update(
            {
                "learning_rate": 10 ** trial.suggest_float("learning_rate_log10", -5, -3),
                "weight_decay": 10 ** trial.suggest_float("weight_decay_log10", -6, -3),
                "num_experts": trial.suggest_categorical("num_experts", [1, 2, 3, 4]),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                "lambda_pred_loss": trial.suggest_categorical(
                    "lambda_pred_loss", [10.0, 20.0, 30.0, 50.0, 80.0, 100.0]
                ),
                "gamma_ahk_loss": trial.suggest_categorical(
                    "gamma_ahk_loss", [10.0, 30.0, 50.0, 100.0, 200.0, 300.0]
                ),
                "epochs": int(base_config.get("optuna", {}).get("epochs", base_config["epochs"])),
            }
        )
        model, sx, sy, _ = train_moe_clock(data_splits, cfg)
        return _evaluate_mae(model, sx, sy, data_splits["X_val_raw"], data_splits["y_val"])

    n_trials = int(base_config.get("optuna", {}).get("trials", 0))
    if n_trials <= 0:
        return {}
    sampler = optuna.samplers.TPESampler(seed=int(base_config.get("seed", 66)))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    return {
        "learning_rate": 10 ** best.get("learning_rate_log10", -4),
        "weight_decay": 10 ** best.get("weight_decay_log10", -4),
        "num_experts": best.get("num_experts", base_config["num_experts"]),
        "batch_size": best.get("batch_size", base_config["batch_size"]),
        "lambda_pred_loss": best.get("lambda_pred_loss", base_config["lambda_pred_loss"]),
        "gamma_ahk_loss": best.get("gamma_ahk_loss", base_config["gamma_ahk_loss"]),
    }


def default_moe_config(args) -> dict:
    skip = bool(getattr(args, "moe_skip_optuna", False))
    return {
        "num_experts": 2,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "epochs": int(getattr(args, "moe_refit_epochs", 100)),
        "batch_size": 32,
        "lambda_pred_loss": 30.0,
        "gamma_ahk_loss": 80.0,
        "seed": int(getattr(args, "seed", 66)),
        "nested_validation_size": 0.2,
        "test_split_size": 0.2,
        "early_stopping": {
            "enabled": True,
            "patience": 15,
            "min_delta": 0.0,
            "monitor_interval": 1,
            "restore_best": True,
        },
        "lr_scheduler": {
            "enabled": True,
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6,
        },
        "optuna": {
            "enabled": not skip,
            "trials": int(getattr(args, "moe_trials", 25)),
            "epochs": int(getattr(args, "moe_hpo_epochs", 60)),
        },
    }


def fit_predict_moe(x_train, y_train, x_test, extra: dict):
    """Return train pred, test pred, and the selected hyperparameter dict."""
    aging_pool = extra["aging_gene_pool"]
    config = copy.deepcopy(extra["moe_config"])
    set_seed(int(config.get("seed", 66)))

    if isinstance(y_train, pd.Series):
        y_series = y_train.astype(float)
    else:
        y_series = pd.Series(np.asarray(y_train, dtype=float), index=x_train.index)

    aging_idx = compute_aging_gene_indices(list(x_train.columns), aging_pool)
    if len(aging_idx) == 0:
        aging_idx = np.array([0], dtype=int)
        warnings.warn("No aging-gene overlap after filterByExpr; AHK loss uses gene 0 as a fallback.")

    if len(x_train) < 10:
        raise ValueError(f"MoE outer train n={len(x_train)} is too small for an inner split.")

    inner_frac = float(config.get("nested_validation_size", 0.2))
    inner_train_idx, inner_val_idx = train_test_split(
        np.arange(len(x_train)),
        test_size=inner_frac,
        random_state=int(config["seed"]),
        shuffle=True,
    )
    inner_splits = {
        "X_train_raw": x_train.iloc[inner_train_idx],
        "y_train": y_series.iloc[inner_train_idx],
        "X_val_raw": x_train.iloc[inner_val_idx],
        "y_val": y_series.iloc[inner_val_idx],
        "aging_gene_indices": aging_idx,
    }

    best_params = {}
    if config.get("optuna", {}).get("enabled", False):
        print(
            f"       MoE Optuna: {config['optuna']['trials']} trials, "
            f"{config['optuna']['epochs']} epochs each"
        )
        best_params = optuna_search_hpo(inner_splits, config)

    selection_cfg = dict(config)
    selection_cfg.update(best_params)
    _, _, _, selection_info = train_moe_clock(inner_splits, selection_cfg)
    selected_epochs = max(1, int(selection_info["best_epoch"]))

    refit_cfg = dict(config)
    refit_cfg.update(best_params)
    refit_cfg["epochs"] = selected_epochs
    refit_cfg["early_stopping"] = {**config.get("early_stopping", {}), "enabled": False}
    refit_cfg["lr_scheduler"] = {**config.get("lr_scheduler", {}), "enabled": False}
    model, sx, sy, _ = train_moe_clock(
        {"X_train_raw": x_train, "y_train": y_series, "aging_gene_indices": aging_idx},
        refit_cfg,
    )
    pred_tr = _predict_raw(model, sx, sy, x_train)
    pred_te = _predict_raw(model, sx, sy, x_test)
    params = dict(best_params)
    params["selected_epochs"] = selected_epochs
    params["n_aging_genes"] = int(len(aging_idx))
    params["device"] = str(next(model.parameters()).device)
    return pred_tr, pred_te, params
