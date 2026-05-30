"""
check_copier.py — Valida que copier.yml es YAML válido y contiene las variables mínimas.
"""
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

try:
    cfg = yaml.safe_load((ROOT / "copier.yml").read_text())
except Exception as e:
    print(f"ERROR: copier.yml no es YAML válido: {e}")
    sys.exit(1)

REQUIRED = [
    "project_name", "project_slug", "project_author_name", "project_author_email",
    "ml_type", "task_type", "nn_model", "optimizer_type", "nn_loss_fn",
    "use_api", "use_optuna", "use_monitoring", "use_mlflow",
    "use_duckdb", "use_docker",
]

missing = [k for k in REQUIRED if k not in cfg]
if missing:
    print(f"ERROR: variables ausentes en copier.yml: {missing}")
    sys.exit(1)

print(f"copier.yml OK — {len(cfg)} variables definidas")
