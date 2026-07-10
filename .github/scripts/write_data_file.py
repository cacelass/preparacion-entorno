"""
write_data_file.py

Convierte la cadena "key=val key=val key=val" pasada como argumento
a un fichero YAML que copier leerá con --data-file.

También añade los campos obligatorios comunes para todos los proyectos CI.

Uso:
    python write_data_file.py "ml_type=supervisado task_type=clasificacion"
"""
import sys
from pathlib import Path

import yaml

# Campos comunes a todas las combinaciones
BASE = {
    "project_name": "test_project",
    "project_slug": "test_project",
    "project_author_name": "CI",
    "project_author_email": "ci@dskit.test",
    "project_description": "CI test project",
    "project_open_source_license": "MIT",
    "python_version": "3.12",
    "project_version": "0.1.0",
    # Flags opcionales — todos apagados por defecto en CI
    "model_type": "todos",
    "cluster_model": "todos",
    "nn_model": "MLP",
    "optimizer_type": "AdamW",
    "nn_loss_fn": "Auto",
    "use_mlflow": False,
    "use_optuna": False,
    "use_duckdb": False,
    "use_api": False,
    "use_docker": False,
    "use_shap": False,
    "use_xgboost": False,
    "use_lightgbm": False,
    "use_catboost": False,
    "use_monitoring": False,
    "graphify_mode": "no",
}

# Parsear argumentos de la forma "key=val key=val"
overrides: dict = {}
VALID_KEYS = {"ml_type", "task_type", "model_type", "cluster_model", "nn_model",
              "optimizer_type", "nn_loss_fn", "use_mlflow", "use_optuna", "use_duckdb",
              "use_api", "use_docker", "use_shap", "use_xgboost", "use_lightgbm",
               "use_catboost", "use_monitoring", "graphify_mode", "project_slug", "project_name",
              "project_author_name", "project_author_email", "project_description",
              "project_open_source_license", "python_version", "project_version"}

for token in sys.argv[1:]:
    if "=" in token:
        k, v = token.split("=", 1)
        if k not in VALID_KEYS:
            print(f"AVISO: clave desconocida '{k}' será ignorada")
            continue
        # Convertir booleanos
        if v.lower() in ("true", "yes", "on", "1"):
            v = True
        elif v.lower() in ("false", "no", "off", "0"):
            v = False
        overrides[k] = v

data = {**BASE, **overrides}

out = Path("/tmp/copier-data.yml")
out.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False))
print(f"copier-data.yml escrito con {len(data)} claves")
print(yaml.dump(data, allow_unicode=True, default_flow_style=False))
