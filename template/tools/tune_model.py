{% if use_optuna %}
"""
tools/tune_model.py — Optimización de hiperparámetros con Optuna.

Ejecutar con:
    make tune

O directamente:
    uv run python -m tools.tune_model

Qué hace:
    1. Ejecuta un estudio Optuna por cada modelo activo.
    2. Guarda los mejores params en artifacts/best_params_<modelo>.joblib.
    3. train_models() los carga automáticamente en el siguiente make train.
    4. Guarda un resumen en reports/tuning_results.csv.

Configurar el número de trials en main.py:
    OPTUNA_TRIALS = 30
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import cross_val_score

from {{ project_slug }}.utils.paths import ARTIFACTS_DIR, REPORTS_DIR

{% if ml_type == "supervisado" or ml_type == "hibrido" %}
{% if task_type == "clasificacion" %}
from sklearn.metrics import f1_score
_SCORING  = "f1_weighted"
_MINIMIZE = False   # maximize F1
{% else %}
_SCORING  = "neg_root_mean_squared_error"
_MINIMIZE = True    # minimize RMSE
{% endif %}
{% endif %}


# ---------------------------------------------------------------------------
# Objetivos por modelo
# ---------------------------------------------------------------------------
{% if ml_type == "supervisado" or ml_type == "hibrido" %}

{% if model_type == "todos" or model_type == "RandomForest" %}
def _objective_rf(trial: optuna.Trial, X_train: Any, y_train: Any) -> float:
{% if task_type == "clasificacion" %}
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators     = trial.suggest_int("n_estimators",   50,  400, step=50),
        max_depth        = trial.suggest_int("max_depth",       3,   20),
        min_samples_leaf = trial.suggest_int("min_samples_leaf",1,   20),
        max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        class_weight     = "balanced",
        random_state     = 42, n_jobs=-1,
    )
{% else %}
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators     = trial.suggest_int("n_estimators",   50,  400, step=50),
        max_depth        = trial.suggest_int("max_depth",       3,   20),
        min_samples_leaf = trial.suggest_int("min_samples_leaf",1,   20),
        max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        random_state     = 42, n_jobs=-1,
    )
{% endif %}
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=_SCORING)
    return float(-scores.mean() if _MINIMIZE else scores.mean())
{% endif %}


{% if model_type == "todos" or model_type == "KNN" %}
def _objective_knn(trial: optuna.Trial, X_train: Any, y_train: Any) -> float:
{% if task_type == "clasificacion" %}
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(
        n_neighbors = trial.suggest_int("n_neighbors", 1, 30),
        weights     = trial.suggest_categorical("weights", ["uniform", "distance"]),
        metric      = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
    )
{% else %}
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(
        n_neighbors = trial.suggest_int("n_neighbors", 1, 30),
        weights     = trial.suggest_categorical("weights", ["uniform", "distance"]),
        metric      = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
    )
{% endif %}
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=_SCORING)
    return float(-scores.mean() if _MINIMIZE else scores.mean())
{% endif %}


{% if model_type == "todos" or model_type == "DecisionTree" %}
def _objective_dt(trial: optuna.Trial, X_train: Any, y_train: Any) -> float:
{% if task_type == "clasificacion" %}
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(
        max_depth        = trial.suggest_int("max_depth",        2,  20),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1,  30),
        min_samples_split= trial.suggest_int("min_samples_split",2,  20),
        criterion        = trial.suggest_categorical("criterion", ["gini", "entropy"]),
        class_weight     = "balanced",
        random_state     = 42,
    )
{% else %}
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(
        max_depth        = trial.suggest_int("max_depth",        2,  20),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1,  30),
        min_samples_split= trial.suggest_int("min_samples_split",2,  20),
        criterion        = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"]),
        random_state     = 42,
    )
{% endif %}
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=_SCORING)
    return float(-scores.mean() if _MINIMIZE else scores.mean())
{% endif %}


{% if model_type == "todos" or model_type == "LogisticRegression" %}
{% if task_type == "clasificacion" %}
def _objective_lr(trial: optuna.Trial, X_train: Any, y_train: Any) -> float:
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(
        C            = trial.suggest_float("C", 1e-3, 100.0, log=True),
        solver       = trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        class_weight = "balanced",
        max_iter     = 1000,
        random_state = 42,
    )
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=_SCORING)
    return float(scores.mean())
{% endif %}
{% endif %}


{% if use_xgboost or model_type == "XGBoost" %}
def _objective_xgb(trial: optuna.Trial, X_train: Any, y_train: Any) -> float:
{% if task_type == "clasificacion" %}
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators      = trial.suggest_int("n_estimators",    50,  500, step=50),
        max_depth         = trial.suggest_int("max_depth",        3,   10),
        learning_rate     = trial.suggest_float("learning_rate",  1e-3, 0.3, log=True),
        subsample         = trial.suggest_float("subsample",      0.5,  1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree",0.5, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha",      1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",     1e-4, 10.0, log=True),
        eval_metric       = "logloss",
        random_state      = 42, n_jobs=-1,
    )
{% else %}
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators      = trial.suggest_int("n_estimators",    50,  500, step=50),
        max_depth         = trial.suggest_int("max_depth",        3,   10),
        learning_rate     = trial.suggest_float("learning_rate",  1e-3, 0.3, log=True),
        subsample         = trial.suggest_float("subsample",      0.5,  1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree",0.5, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha",      1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",     1e-4, 10.0, log=True),
        eval_metric       = "rmse",
        random_state      = 42, n_jobs=-1,
    )
{% endif %}
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=_SCORING)
    return float(-scores.mean() if _MINIMIZE else scores.mean())
{% endif %}


{% if use_lightgbm or model_type == "LightGBM" %}
def _objective_lgbm(trial: optuna.Trial, X_train: Any, y_train: Any) -> float:{% if task_type == "clasificacion" %}
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(
        n_estimators      = trial.suggest_int("n_estimators",    50,  500, step=50),
        num_leaves        = trial.suggest_int("num_leaves",       15,  127),
        learning_rate     = trial.suggest_float("learning_rate",  1e-3, 0.3, log=True),
        subsample         = trial.suggest_float("subsample",      0.5,  1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree",0.5, 1.0),
        min_child_samples = trial.suggest_int("min_child_samples",5,   50),
        reg_alpha         = trial.suggest_float("reg_alpha",      1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",     1e-4, 10.0, log=True),
        class_weight      = "balanced",
        random_state      = 42, n_jobs=-1, verbose=-1,
    )
{% else %}
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(
        n_estimators      = trial.suggest_int("n_estimators",    50,  500, step=50),
        num_leaves        = trial.suggest_int("num_leaves",       15,  127),
        learning_rate     = trial.suggest_float("learning_rate",  1e-3, 0.3, log=True),
        subsample         = trial.suggest_float("subsample",      0.5,  1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree",0.5, 1.0),
        min_child_samples = trial.suggest_int("min_child_samples",5,   50),
        reg_alpha         = trial.suggest_float("reg_alpha",      1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",     1e-4, 10.0, log=True),
        random_state      = 42, n_jobs=-1, verbose=-1,
    )
{% endif %}
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=_SCORING)
    return float(-scores.mean() if _MINIMIZE else scores.mean())
{% endif %}


{% if use_catboost or model_type == "CatBoost" %}
def _objective_catboost(trial: optuna.Trial, X_train: Any, y_train: Any) -> float:
{% if task_type == "clasificacion" %}
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations     = trial.suggest_int("iterations",    50,  500, step=50),
        depth          = trial.suggest_int("depth",          3,   10),
        learning_rate  = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        l2_leaf_reg    = trial.suggest_float("l2_leaf_reg",   1e-2, 10.0, log=True),
        border_count   = trial.suggest_categorical("border_count", [32, 64, 128, 254]),
        loss_function  = "Logloss",
        eval_metric    = "Accuracy",
        random_seed    = 42,
        verbose        = 0,
    )
{% else %}
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(
        iterations     = trial.suggest_int("iterations",    50,  500, step=50),
        depth          = trial.suggest_int("depth",          3,   10),
        learning_rate  = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        l2_leaf_reg    = trial.suggest_float("l2_leaf_reg",   1e-2, 10.0, log=True),
        border_count   = trial.suggest_categorical("border_count", [32, 64, 128, 254]),
        loss_function  = "RMSE",
        random_seed    = 42,
        verbose        = 0,
    )
{% endif %}
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=_SCORING)
    return float(-scores.mean() if _MINIMIZE else scores.mean())
{% endif %}


# ---------------------------------------------------------------------------
# Mapa modelo → función objetivo
# ---------------------------------------------------------------------------
_OBJECTIVES: dict[str, Callable[[optuna.Trial, Any, Any], float]] = {}
{% if model_type == "todos" or model_type == "RandomForest" %}
_OBJECTIVES["RandomForest"] = _objective_rf
{% endif %}
{% if model_type == "todos" or model_type == "KNN" %}
_OBJECTIVES["KNN"] = _objective_knn
{% endif %}
{% if model_type == "todos" or model_type == "DecisionTree" %}
_OBJECTIVES["DecisionTree"] = _objective_dt
{% endif %}
{% if (model_type == "todos" or model_type == "LogisticRegression") and task_type == "clasificacion" %}
_OBJECTIVES["LogisticRegression"] = _objective_lr
{% endif %}
{% if use_xgboost or model_type == "XGBoost" %}
_OBJECTIVES["XGBoost"] = _objective_xgb
{% endif %}
{% if use_lightgbm or model_type == "LightGBM" %}
_OBJECTIVES["LightGBM"] = _objective_lgbm
{% endif %}
{% if use_catboost or model_type == "CatBoost" %}
_OBJECTIVES["CatBoost"] = _objective_catboost
{% endif %}
{% if model_type == "todos" or model_type == "SVM" %}


def _objective_svm(trial: optuna.Trial, X_train: Any, y_train: Any) -> float:
{% if task_type == "clasificacion" %}
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            C        = trial.suggest_float("C",     1e-2, 100.0, log=True),
            gamma    = trial.suggest_categorical("gamma", ["scale", "auto"]),
            kernel   = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
            class_weight = "balanced",
            probability  = True,
            random_state = 42,
        )),
    ])
{% else %}
    from sklearn.svm import SVR
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(
            C      = trial.suggest_float("C",      1e-2, 100.0, log=True),
            gamma  = trial.suggest_categorical("gamma", ["scale", "auto"]),
            kernel = trial.suggest_categorical("kernel", ["rbf", "poly"]),
            epsilon= trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
        )),
    ])
{% endif %}
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=_SCORING)
    return float(-scores.mean() if _MINIMIZE else scores.mean())


_OBJECTIVES["SVM"] = _objective_svm
{% endif %}

{% elif ml_type == "redes_neuronales" %}

def _objective_nn(trial, X_train, y_train, input_dim: int, output_dim: int):
    """Objetivo para la red neuronal — busca LR, batch_size y arquitectura."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    lr         = trial.suggest_float("lr",         1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    epochs     = 10  # epochs cortos para tuning rápido

{% if nn_model == "MLP" %}
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_dims = [
        trial.suggest_int(f"hidden_{i}", 32, 512, step=32) for i in range(n_layers)
    ]
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    from {{ project_slug }}.models.train_model import MLP
    model = MLP(input_dim=input_dim, output_dim=output_dim,
                hidden_dims=hidden_dims, dropout=dropout)
{% elif nn_model == "LSTM" %}
    hidden_dim  = trial.suggest_int("hidden_dim",  32, 256, step=32)
    num_layers  = trial.suggest_int("num_layers",   1,   4)
    dropout     = trial.suggest_float("dropout",    0.0, 0.5)
    from {{ project_slug }}.models.train_model import LSTMClassifier
    model = LSTMClassifier(input_dim=input_dim, output_dim=output_dim,
                           hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
{% elif nn_model == "GRU" %}
    hidden_dim  = trial.suggest_int("hidden_dim",  32, 256, step=32)
    num_layers  = trial.suggest_int("num_layers",   1,   4)
    dropout     = trial.suggest_float("dropout",    0.0, 0.5)
    from {{ project_slug }}.models.train_model import GRUClassifier
    model = GRUClassifier(input_dim=input_dim, output_dim=output_dim,
                          hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
{% elif nn_model == "CNN1D" %}
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    from {{ project_slug }}.models.train_model import CNN1D
    model = CNN1D(input_dim=input_dim, output_dim=output_dim, dropout=dropout)
{% elif nn_model == "Transformer" %}
    d_model  = trial.suggest_categorical("d_model",  [32, 64, 128])
    nhead    = trial.suggest_categorical("nhead",    [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout  = trial.suggest_float("dropout", 0.0, 0.3)
    from {{ project_slug }}.models.train_model import TransformerClassifier
    model = TransformerClassifier(input_dim=input_dim, output_dim=output_dim,
                                  d_model=d_model, nhead=nhead,
                                  num_layers=num_layers, dropout=dropout)
{% elif nn_model == "ResNet" %}
    hidden_dim  = trial.suggest_int("hidden_dim",  64, 256, step=32)
    num_blocks  = trial.suggest_int("num_blocks",   2,  10)
    dropout     = trial.suggest_float("dropout",    0.0, 0.4)
    from {{ project_slug }}.models.train_model import ResNet
    model = ResNet(input_dim=input_dim, output_dim=output_dim,
                   hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=dropout)
{% endif %}

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    model  = model.to(device)

    X_arr = X_train.values if hasattr(X_train, "values") else X_train
    y_arr = y_train.values if hasattr(y_train, "values") else y_train
    n_val = max(1, int(len(X_arr) * 0.1))
    X_tr, X_val = X_arr[:-n_val], X_arr[-n_val:]
    y_tr, y_val = y_arr[:-n_val], y_arr[-n_val:]

    loader = DataLoader(
{% if task_type == 'regresion' %}
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.float32)),
{% else %}
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.long)),
{% endif %}
        batch_size=batch_size, shuffle=True,
    )
    val_X = torch.tensor(X_val, dtype=torch.float32).to(device)
{% if task_type == 'regresion' %}
    val_y = torch.tensor(y_val, dtype=torch.float32).to(device)
{% else %}
    val_y = torch.tensor(y_val, dtype=torch.long).to(device)
{% endif %}

{% set _opt = optimizer_type if optimizer_type is defined else 'AdamW' %}
{% if _opt == 'AdamW' %}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
{% elif _opt == 'Adam' %}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
{% elif _opt == 'SGD' %}
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
{% elif _opt == 'RMSProp' %}
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
{% elif _opt == 'Adagrad' %}
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
{% else %}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
{% endif %}

{% set _loss = nn_loss_fn if nn_loss_fn is defined else 'Auto' %}
{% if _loss == 'Auto' %}
{%   if task_type == 'regresion' %}
    criterion = nn.MSELoss()
{%   else %}
    criterion = nn.CrossEntropyLoss()
{%   endif %}
{% elif _loss == 'MSELoss' %}
    criterion = nn.MSELoss()
{% elif _loss == 'L1Loss' %}
    criterion = nn.L1Loss()
{% elif _loss == 'BCEWithLogitsLoss' %}
    criterion = nn.BCEWithLogitsLoss()
{% else %}
    criterion = nn.CrossEntropyLoss()
{% endif %}

    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
{% if task_type == 'regresion' %}
            criterion(model(Xb).squeeze(), yb).backward()
{% else %}
            criterion(model(Xb), yb).backward()
{% endif %}
            optimizer.step()

        model.eval()
        with torch.no_grad():
{% if task_type == 'regresion' %}
            val_loss = criterion(model(val_X).squeeze(), val_y).item()
{% else %}
            val_loss = criterion(model(val_X), val_y).item()
{% endif %}

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss  # minimize


_OBJECTIVES = {"{{ nn_model }}": _objective_nn}

{% elif ml_type == "no_supervisado" %}
def _objective_kmeans(trial, X):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n_clusters = trial.suggest_int("n_clusters", 2, 15)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = model.fit_predict(X)
    if len(set(labels)) < 2:
        return -1.0
    return silhouette_score(X, labels)

_OBJECTIVES = {
{% if cluster_model == "todos" or cluster_model == "KMeans" %}
    "KMeans": _objective_kmeans,
{% endif %}
}

{% endif %}  {# end ml_type branches #}

# ---------------------------------------------------------------------------
# Motor principal de tuning
# ---------------------------------------------------------------------------
def tune_models(
    X_train: Any,
    y_train: Any = None,
    n_trials: int = 30,
    timeout: int | None = None,
{% if ml_type == "redes_neuronales" %}
    input_dim:  int = None,
    output_dim: int = None,
{% endif %}
    artifacts_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Optimiza hiperparámetros de todos los modelos activos con Optuna.

    Guarda los mejores params en artifacts/best_params_<modelo>.joblib.
    train_models() los carga automáticamente si existen.

    Parameters
    ----------
    X_train       : features de entrenamiento
    y_train       : target (None para no_supervisado)
    n_trials      : número de trials por modelo (default: 30)
    timeout       : segundos máximos por estudio (None = sin límite)
    artifacts_dir : directorio donde guardar los .joblib (default: ARTIFACTS_DIR)
    reports_dir   : directorio donde guardar tuning_results.csv (default: REPORTS_DIR)

    Returns
    -------
    dict[str, dict] : {nombre_modelo: mejores_params}
    """
    _artifacts = artifacts_dir or ARTIFACTS_DIR
    _reports   = reports_dir   or REPORTS_DIR
    _artifacts.mkdir(parents=True, exist_ok=True)
    _reports.mkdir(parents=True, exist_ok=True)

    X_arr = X_train.values if hasattr(X_train, "values") else X_train
    if y_train is not None:
        y_arr = y_train.values if hasattr(y_train, "values") else y_train
    else:
        y_arr = None

    results = []
    best_params_all: dict[str, dict[str, Any]] = {}

    print(f"\n{'='*60}")
    print(f"  Optuna — optimizando {len(_OBJECTIVES)} modelo(s), {n_trials} trials c/u")
    print(f"{'='*60}")

{% if ml_type == "redes_neuronales" %}
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    study   = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        lambda trial: _objective_nn(
            trial, X_train, y_train,
            input_dim=input_dim or X_arr.shape[1],
            output_dim=output_dim or (int(np.unique(y_arr).shape[0]) if y_arr is not None else 2),
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    model_name = "{{ nn_model }}"
    best = study.best_params
    best_val  = study.best_value
    print(f"\n  {model_name}: val_loss={best_val:.4f}")
    print(f"    Params: {best}")
    joblib.dump(best, _artifacts / f"best_params_{model_name}.joblib")
    best_params_all[model_name] = best
    results.append({"modelo": model_name, "best_value": best_val, **best})

{% else %}
    for model_name, objective_fn in _OBJECTIVES.items():
        print(f"\n  Optimizando {model_name}...")
        sampler = optuna.samplers.TPESampler(seed=42)
{% if ml_type == "no_supervisado" %}
        direction = "maximize"
        def _obj(trial): return objective_fn(trial, X_arr)
{% else %}
        direction = "maximize" if not _MINIMIZE else "minimize"
        def _obj(trial: optuna.Trial) -> float: return objective_fn(trial, X_arr, y_arr)
{% endif %}
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(_obj, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        best      = study.best_params
        best_val  = study.best_value
        print(f"    Mejor valor: {best_val:.4f}")
        print(f"    Params: {best}")

        path = _artifacts / f"best_params_{model_name}.joblib"
        joblib.dump(best, path)
        print(f"    Guardado → {path.name}")
        best_params_all[model_name] = best
        results.append({"modelo": model_name, "best_value": round(best_val, 4), **best})
{% endif %}

    df = pd.DataFrame(results)
    out_csv = _reports / "tuning_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Resumen guardado → {out_csv.name}")
    print(df.to_string(index=False))
    print(f"\n{'='*60}")
    print("  Ejecuta 'make train' para entrenar con los mejores params.")
    print(f"{'='*60}\n")

    return best_params_all


if __name__ == "__main__":
    print("Ejecuta 'make tune' para lanzar la optimización con tus datos procesados.")
    print("O importa tune_models() y pásale X_train e y_train directamente.")
{% endif %}
