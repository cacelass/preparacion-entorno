"""
validate_template.py — Fase 1 del CI.

Renderiza todas las combinaciones con Jinja2 StrictUndefined y verifica
que cada fichero .py generado pasa ast.parse() sin errores de sintaxis.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

try:
    from jinja2 import BaseLoader, Environment, StrictUndefined, TemplateSyntaxError, UndefinedError
except ImportError:
    sys.exit("ERROR: pip install jinja2")

BASE = Path(__file__).parent.parent.parent / "template"

if not BASE.exists():
    sys.exit(f"ERROR: template/ no encontrado en {BASE}. Verifica la estructura del repo.")

STD = dict(
    project_slug="test_project", project_name="Test Project",
    project_author_name="CI", project_author_email="ci@test.com",
    project_description="CI validation", project_open_source_license="MIT",
    python_version="3.12", project_version="0.1.0",
)

DEFAULTS = dict(
    model_type="todos", cluster_model="todos", nn_model="MLP",
    optimizer_type="AdamW", nn_loss_fn="Auto", use_mlflow=False,
    use_optuna=False, use_duckdb=False, use_api=False, use_docker=False,
    use_shap=False, use_xgboost=False, use_lightgbm=False,
    use_catboost=False, use_monitoring=False,
)

COMBOS: list[tuple[str, dict]] = [
    ("sup+clf",             dict(ml_type="supervisado",      task_type="clasificacion")),
    ("sup+reg",             dict(ml_type="supervisado",      task_type="regresion")),
    ("sup+clf+ALL",         dict(ml_type="supervisado",      task_type="clasificacion",
                                 model_type="RandomForest", use_mlflow=True, use_optuna=True,
                                 use_duckdb=True, use_api=True, use_docker=True, use_shap=True,
                                 use_xgboost=True, use_lightgbm=True, use_catboost=True,
                                 use_monitoring=True)),
    ("sup+reg+ALL",         dict(ml_type="supervisado",      task_type="regresion",
                                 use_mlflow=True, use_optuna=True, use_api=True, use_monitoring=True)),
    ("nosup",               dict(ml_type="no_supervisado",   task_type="clasificacion")),
    ("nosup+ALL",           dict(ml_type="no_supervisado",   task_type="clasificacion",
                                 cluster_model="KMeans", use_api=True, use_optuna=True,
                                 use_monitoring=True, use_docker=True)),
    ("nn+MLP+clf",          dict(ml_type="redes_neuronales", task_type="clasificacion",
                                 nn_model="MLP",         optimizer_type="AdamW",   nn_loss_fn="Auto")),
    ("nn+MLP+reg+SGD+MSE",  dict(ml_type="redes_neuronales", task_type="regresion",
                                 nn_model="MLP",         optimizer_type="SGD",     nn_loss_fn="MSELoss")),
    ("nn+CNN1D+clf+Adam+CE",dict(ml_type="redes_neuronales", task_type="clasificacion",
                                 nn_model="CNN1D",        optimizer_type="Adam",    nn_loss_fn="CrossEntropyLoss",
                                 use_mlflow=True, use_api=True)),
    ("nn+CNN1D+reg+L1",     dict(ml_type="redes_neuronales", task_type="regresion",
                                 nn_model="CNN1D",        optimizer_type="Adam",    nn_loss_fn="L1Loss")),
    ("nn+LSTM+clf+RMS+BCE", dict(ml_type="redes_neuronales", task_type="clasificacion",
                                 nn_model="LSTM",         optimizer_type="RMSProp", nn_loss_fn="BCEWithLogitsLoss",
                                 use_optuna=True)),
    ("nn+LSTM+reg+RMS",     dict(ml_type="redes_neuronales", task_type="regresion",
                                 nn_model="LSTM",         optimizer_type="RMSProp", nn_loss_fn="Auto")),
    ("nn+GRU+clf+SGD",      dict(ml_type="redes_neuronales", task_type="clasificacion",
                                 nn_model="GRU",          optimizer_type="SGD",     nn_loss_fn="Auto")),
    ("nn+GRU+reg+Adag+L1",  dict(ml_type="redes_neuronales", task_type="regresion",
                                 nn_model="GRU",          optimizer_type="Adagrad", nn_loss_fn="L1Loss")),
    ("nn+Transf+clf+ALL",   dict(ml_type="redes_neuronales", task_type="clasificacion",
                                 nn_model="Transformer",  optimizer_type="AdamW",   nn_loss_fn="Auto",
                                 use_mlflow=True, use_optuna=True, use_api=True,
                                 use_docker=True, use_monitoring=True)),
    ("nn+Transf+reg+MSE",   dict(ml_type="redes_neuronales", task_type="regresion",
                                 nn_model="Transformer",  optimizer_type="AdamW",   nn_loss_fn="MSELoss")),
    ("nn+MLP+reg+ALL",      dict(ml_type="redes_neuronales", task_type="regresion",
                                 nn_model="MLP",          optimizer_type="AdamW",   nn_loss_fn="Auto",
                                 use_mlflow=True, use_optuna=True, use_api=True, use_monitoring=True)),
    ("hibrido+clf",         dict(ml_type="hibrido",          task_type="clasificacion",
                                 use_shap=True, use_xgboost=True, use_lightgbm=True)),
    ("hibrido+reg",         dict(ml_type="hibrido",          task_type="regresion")),
    ("hibrido+reg+ALL",     dict(ml_type="hibrido",          task_type="regresion",
                                 use_mlflow=True, use_optuna=True, use_api=True, use_monitoring=True)),
]

ALL_FILES = sorted([
    str(f.relative_to(BASE))
    for f in BASE.rglob("*")
    if f.is_file()
    and "__pycache__" not in str(f)
    and ".DS_Store" not in str(f)
])

env = Environment(loader=BaseLoader(), undefined=StrictUndefined, keep_trailing_newline=True)
bugs: list[tuple[str, str, str]] = []

print(f"BASE         : {BASE}")
print(f"Combinaciones: {len(COMBOS)}")
print(f"Ficheros     : {len(ALL_FILES)}")
print(f"Checks totales: {len(COMBOS) * len(ALL_FILES)}")
print()

for label, combo in COMBOS:
    ctx = {**STD, **DEFAULTS, **combo}
    for rel in ALL_FILES:
        raw = (BASE / rel).read_text(errors="ignore")
        try:
            rendered = env.from_string(raw).render(**ctx)
        except UndefinedError as exc:
            bugs.append((label, rel, f"UNDEF: {exc}"))
            print(f"  ✗ UNDEF  [{label}] {rel}: {exc}")
            continue
        except TemplateSyntaxError as exc:
            bugs.append((label, rel, f"SYNTAX: {exc}"))
            print(f"  ✗ SYNTAX [{label}] {rel}: {exc}")
            continue
        except Exception as exc:
            bugs.append((label, rel, f"RENDER: {exc}"))
            print(f"  ✗ RENDER [{label}] {rel}: {exc}")
            continue

        if rel.endswith(".py"):
            try:
                ast.parse(rendered)
            except SyntaxError as exc:
                lines = rendered.splitlines()
                ln = exc.lineno or 1
                snippet = "\n".join(
                    f"    {ln+i-2}: {lines[max(0,ln+i-3)]}"
                    for i in range(6)
                    if 0 <= ln+i-3 < len(lines)
                )
                msg = f"PY_SYNTAX L{ln}: {exc.msg}\n{snippet}"
                bugs.append((label, rel, msg))
                print(f"  ✗ PY_SYN [{label}] {rel}:\n{snippet}")

print()
print(f"Bugs encontrados: {len(bugs)}")

if bugs:
    print("\n" + "="*60)
    print("RESUMEN DE BUGS:")
    for label, rel, msg in bugs:
        print(f"\n  Combo : {label}")
        print(f"  Fichero: {rel}")
        print(f"  Error  : {msg[:400]}")
    sys.exit(1)

print("✓ Plantilla válida — 0 bugs")