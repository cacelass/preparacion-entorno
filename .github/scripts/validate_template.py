"""
validate_template.py — Fase 1 del CI.

Renderiza todas las combinaciones con Jinja2 StrictUndefined y verifica
que cada fichero .py generado pasa ast.parse() sin errores de sintaxis.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

try:
    from jinja2 import BaseLoader, Environment, StrictUndefined, TemplateSyntaxError, UndefinedError
except ImportError:
    sys.exit("ERROR: pip install jinja2")

project_version_date = "2025-01-01"

BASE = Path(__file__).parent.parent.parent / "template"

if not BASE.exists():
    sys.exit(f"ERROR: template/ no encontrado en {BASE}. Verifica la estructura del repo.")

STD = dict(
    project_slug="test_project",
    project_name="Test Project",
    project_author_name="CI",
    project_author_email="ci@test.com",
    project_description="CI validation",
    project_open_source_license="MIT",
    python_version="3.12",
    project_version="0.1.0",
    dskit_version="1.9.0",
    project_version_date="2025-01-01",
)

DEFAULTS = dict(
    model_type="todos",
    cluster_model="todos",
    nn_model="MLP",
    optimizer_type="AdamW",
    nn_loss_fn="Auto",
    use_mlflow=False,
    use_optuna=False,
    use_duckdb=False,
    use_api=False,
    use_docker=False,
    use_shap=False,
    use_xgboost=False,
    use_lightgbm=False,
    use_catboost=False,
    use_monitoring=False,
    use_calibration=False,
    use_conformal=False,
    use_rag=False,
    graphify_mode="no",
)

COMBOS: list[tuple[str, dict]] = [
    ("sup+clf", dict(ml_type="supervisado", task_type="clasificacion")),
    ("sup+reg", dict(ml_type="supervisado", task_type="regresion")),
    (
        "sup+clf+ALL",
        dict(
            ml_type="supervisado",
            task_type="clasificacion",
            model_type="RandomForest",
            use_mlflow=True,
            use_optuna=True,
            use_duckdb=True,
            use_api=True,
            use_docker=True,
            use_shap=True,
            use_xgboost=True,
            use_lightgbm=True,
            use_catboost=True,
            use_monitoring=True,
            use_rag=True,
        ),
    ),
    (
        "sup+reg+ALL",
        dict(
            ml_type="supervisado",
            task_type="regresion",
            use_mlflow=True,
            use_optuna=True,
            use_api=True,
            use_monitoring=True,
            use_rag=True,
        ),
    ),
    ("nosup", dict(ml_type="no_supervisado", task_type="clasificacion")),
    (
        "nosup+ALL",
        dict(
            ml_type="no_supervisado",
            task_type="clasificacion",
            cluster_model="KMeans",
            use_api=True,
            use_optuna=True,
            use_monitoring=True,
            use_docker=True,
            use_rag=True,
        ),
    ),
    (
        "nn+MLP+clf",
        dict(
            ml_type="redes_neuronales",
            task_type="clasificacion",
            nn_model="MLP",
            optimizer_type="AdamW",
            nn_loss_fn="Auto",
        ),
    ),
    (
        "nn+MLP+reg+SGD+MSE",
        dict(
            ml_type="redes_neuronales",
            task_type="regresion",
            nn_model="MLP",
            optimizer_type="SGD",
            nn_loss_fn="MSELoss",
        ),
    ),
    (
        "nn+CNN1D+clf+Adam+CE",
        dict(
            ml_type="redes_neuronales",
            task_type="clasificacion",
            nn_model="CNN1D",
            optimizer_type="Adam",
            nn_loss_fn="CrossEntropyLoss",
            use_mlflow=True,
            use_api=True,
        ),
    ),
    (
        "nn+CNN1D+reg+L1",
        dict(
            ml_type="redes_neuronales",
            task_type="regresion",
            nn_model="CNN1D",
            optimizer_type="Adam",
            nn_loss_fn="L1Loss",
        ),
    ),
    (
        "nn+LSTM+clf+RMS+BCE",
        dict(
            ml_type="redes_neuronales",
            task_type="clasificacion",
            nn_model="LSTM",
            optimizer_type="RMSProp",
            nn_loss_fn="BCEWithLogitsLoss",
            use_optuna=True,
            use_calibration=True,
        ),
    ),
    (
        "nn+LSTM+reg+RMS",
        dict(
            ml_type="redes_neuronales",
            task_type="regresion",
            nn_model="LSTM",
            optimizer_type="RMSProp",
            nn_loss_fn="Auto",
        ),
    ),
    (
        "nn+GRU+clf+SGD",
        dict(
            ml_type="redes_neuronales",
            task_type="clasificacion",
            nn_model="GRU",
            optimizer_type="SGD",
            nn_loss_fn="Auto",
        ),
    ),
    (
        "nn+GRU+reg+Adag+L1",
        dict(
            ml_type="redes_neuronales",
            task_type="regresion",
            nn_model="GRU",
            optimizer_type="Adagrad",
            nn_loss_fn="L1Loss",
        ),
    ),
    (
        "nn+Transf+clf+ALL",
        dict(
            ml_type="redes_neuronales",
            task_type="clasificacion",
            nn_model="Transformer",
            optimizer_type="AdamW",
            nn_loss_fn="Auto",
            use_mlflow=True,
            use_optuna=True,
            use_api=True,
            use_docker=True,
            use_monitoring=True,
            use_rag=True,
        ),
    ),
    (
        "nn+Transf+reg+MSE",
        dict(
            ml_type="redes_neuronales",
            task_type="regresion",
            nn_model="Transformer",
            optimizer_type="AdamW",
            nn_loss_fn="MSELoss",
        ),
    ),
    (
        "nn+MLP+reg+ALL",
        dict(
            ml_type="redes_neuronales",
            task_type="regresion",
            nn_model="MLP",
            optimizer_type="AdamW",
            nn_loss_fn="Auto",
            use_mlflow=True,
            use_optuna=True,
            use_api=True,
            use_monitoring=True,
            use_rag=True,
        ),
    ),
    (
        "hibrido+clf",
        dict(
            ml_type="hibrido",
            task_type="clasificacion",
            use_shap=True,
            use_xgboost=True,
            use_lightgbm=True,
        ),
    ),
    ("hibrido+reg", dict(ml_type="hibrido", task_type="regresion")),
    (
        "hibrido+reg+ALL",
        dict(
            ml_type="hibrido",
            task_type="regresion",
            use_mlflow=True,
            use_optuna=True,
            use_api=True,
            use_monitoring=True,
            use_rag=True,
        ),
    ),
]

ALL_FILES = sorted(
    [
        str(f.relative_to(BASE))
        for f in BASE.rglob("*")
        if f.is_file() and "__pycache__" not in str(f) and ".DS_Store" not in str(f)
    ]
)

BACKLOG_STATUS = ("pending", "in_progress", "done", "blocked")
BACKLOG_REQUIRED = ("id", "title", "description", "acceptance_criteria", "status")


def validate_backlog(doc: object) -> list[str]:
    """
    Mismo contrato que verifica init.sh en el proyecto generado: si el backlog
    del template no lo cumple, el arnés nace bloqueado.
    """
    if not isinstance(doc, dict):
        return ["se esperaba un objeto JSON en la raíz"]
    features = doc.get("features")
    if not isinstance(features, list) or not features:
        return ["falta la clave 'features' con una lista no vacía"]

    problems: list[str] = []
    ids: set[str] = set()
    for i, feat in enumerate(features):
        if not isinstance(feat, dict):
            problems.append(f"feature #{i} no es un objeto")
            continue
        missing = [k for k in BACKLOG_REQUIRED if k not in feat]
        if missing:
            problems.append(f"feature #{i} sin campos: {', '.join(missing)}")
            continue
        if feat["id"] in ids:
            problems.append(f"id duplicado: {feat['id']}")
        ids.add(feat["id"])
        if feat["status"] not in BACKLOG_STATUS:
            problems.append(f"{feat['id']}: status '{feat['status']}' no válido")
        criteria = feat["acceptance_criteria"]
        if not isinstance(criteria, list) or not criteria:
            problems.append(f"{feat['id']}: acceptance_criteria vacío o no es lista")

    for feat in features:
        if not isinstance(feat, dict):
            continue
        for dep in feat.get("depends_on", []):
            if dep not in ids:
                problems.append(f"{feat.get('id')}: depends_on '{dep}' no existe en este combo")

    return problems


def validate_claude_mirror() -> list[str]:
    """
    Los agentes del arnés se escriben una vez en `.opencode/agents/` y viajan
    también en `.claude/agents/` con frontmatter, porque cada asistente los
    busca en su sitio. Son ficheros versionados, no generados al vuelo: si se
    desincronizan, Claude Code y opencode se comportan distinto sobre el mismo
    proyecto y nadie se entera. Este check lo impide.
    """
    problems: list[str] = []
    for nombre in ("lider", "explorer", "implementer", "reviewer"):
        fuente = BASE / ".opencode" / "agents" / f"{nombre}.md"
        espejo = BASE / ".claude" / "agents" / f"{nombre}.md"
        if not fuente.exists():
            problems.append(f"falta .opencode/agents/{nombre}.md")
            continue
        if not espejo.exists():
            problems.append(f"falta .claude/agents/{nombre}.md — ejecuta 'make assistants-sync'")
            continue
        cuerpo = fuente.read_text(errors="replace").strip()
        texto = espejo.read_text(errors="replace")
        if not texto.startswith("---"):
            problems.append(f".claude/agents/{nombre}.md sin frontmatter YAML")
        if cuerpo not in texto:
            problems.append(f".claude/agents/{nombre}.md desincronizado de su fuente en .opencode/")
    return problems


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
        raw = (BASE / rel).read_text(errors="replace")
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
                    f"    {ln + i - 1}: {lines[ln + i - 1]}"
                    for i in range(6)
                    if ln + i - 1 < len(lines)
                )
                msg = f"PY_SYNTAX L{ln}: {exc.msg}\n{snippet}"
                bugs.append((label, rel, msg))
                print(f"  ✗ PY_SYN [{label}] {rel}:\n{snippet}")

        # Los .json con condicionales Jinja2 se rompen fácil (comas colgantes).
        # .vscode/ queda fuera: son JSONC, VS Code acepta comentarios //.
        elif rel.endswith(".json") and not rel.startswith(".vscode/"):
            try:
                doc = json.loads(rendered)
            except json.JSONDecodeError as exc:
                lines = rendered.splitlines()
                ln = exc.lineno
                snippet = "\n".join(
                    f"    {i + 1}: {lines[i]}"
                    for i in range(max(ln - 3, 0), min(ln + 2, len(lines)))
                )
                msg = f"JSON L{ln}: {exc.msg}\n{snippet}"
                bugs.append((label, rel, msg))
                print(f"  ✗ JSON    [{label}] {rel}:\n{snippet}")
            else:
                if rel == "featureslist.json":
                    for problem in validate_backlog(doc):
                        bugs.append((label, rel, f"BACKLOG: {problem}"))
                        print(f"  ✗ BACKLOG [{label}] {rel}: {problem}")

for problem in validate_claude_mirror():
    bugs.append(("*", ".claude/agents/", problem))
    print(f"  ✗ MIRROR  {problem}")

print()
print(f"Bugs encontrados: {len(bugs)}")

if bugs:
    print("\n" + "=" * 60)
    print("RESUMEN DE BUGS:")
    for label, rel, msg in bugs:
        print(f"\n  Combo : {label}")
        print(f"  Fichero: {rel}")
        print(f"  Error  : {msg[:400]}")
    sys.exit(1)

print("✓ Plantilla válida — 0 bugs")
