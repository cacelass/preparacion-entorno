"""
write_data_file.py

Convierte la cadena "key=val key=val key=val" pasada como argumento
a un fichero YAML que copier leerá con --data-file.

También añade los campos obligatorios comunes para todos los proyectos CI.

Uso:
    python write_data_file.py "ml_type=supervisado task_type=clasificacion"
"""

import json
import sys
from pathlib import Path

import yaml


def _version_de_copier() -> str:
    """La version sale de copier.yml: escrita a mano se quedaba vieja."""
    doc = yaml.safe_load((Path(__file__).parent.parent.parent / "copier.yml").read_text())
    return doc.get("dskit_version", {}).get("default", "0.0.0")


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
    "dskit_version": _version_de_copier(),
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
    "use_calibration": False,
    "use_conformal": False,
    # Ojo: en copier `use_rag` viene en True por defecto. Se declara aqui
    # explicito para que la matriz pueda apagarlo y para que se vea que se
    # esta instalando chromadb en cada job.
    "use_rag": True,
    "use_sdd": False,
    "proyecto_perfil": "estandar",
    "graphify_mode": "no",
}

# Parsear argumentos de la forma "key=val key=val"
raw = " ".join(sys.argv[1:])
overrides: dict = {}
VALID_KEYS = {
    "ml_type",
    "task_type",
    "model_type",
    "cluster_model",
    "nn_model",
    "optimizer_type",
    "nn_loss_fn",
    "use_mlflow",
    "use_optuna",
    "use_duckdb",
    "use_api",
    "use_docker",
    "use_shap",
    "use_xgboost",
    "use_lightgbm",
    "use_catboost",
    "use_monitoring",
    "use_calibration",
    "use_conformal",
    "use_rag",
    "use_sdd",
    "use_mcp",
    "proyecto_perfil",
    "graphify_mode",
    "project_slug",
    "project_name",
    "project_author_name",
    "project_author_email",
    "project_description",
    "project_open_source_license",
    "python_version",
    "project_version",
    "dskit_version",
}

# La matriz de CI se genera y pasa JSON: "graphify + obsidian vault" lleva
# espacios y el formato "key=val key=val" lo parte por la mitad. Se mantiene
# el formato antiguo para invocaciones a mano.
try:
    _json = json.loads(raw)
except (json.JSONDecodeError, ValueError):
    _json = None

if isinstance(_json, dict):
    for k, v in _json.items():
        if k not in VALID_KEYS:
            print(f"AVISO: clave desconocida '{k}' sera ignorada")
            continue
        overrides[k] = v
    raw = ""

for token in raw.split():
    if "=" in token:
        k, v = token.split("=", 1)
        if k not in VALID_KEYS:
            print(f"AVISO: clave desconocida '{k}' será ignorada")
            continue
        # Convertir booleanos (excepto campos que son strings enum)
        if k in ("graphify_mode",):
            pass  # mantener como string
        elif v.lower() in ("true", "yes", "on", "1"):
            v = True
        elif v.lower() in ("false", "no", "off", "0"):
            v = False
        overrides[k] = v

data = {**BASE, **overrides}

out = Path("/tmp/copier-data.yml")
out.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False))
print(f"copier-data.yml escrito con {len(data)} claves")
print(yaml.dump(data, allow_unicode=True, default_flow_style=False))
