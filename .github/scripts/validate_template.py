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
    proyecto_perfil="estandar",
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
    use_sdd=False,
    use_integration=False,
    use_mcp=False,
    use_demo=False,
    graphify_mode="no",
)

# Combos que ya rompieron algo alguna vez. All-pairs cubre interacciones de 2
# variables; estos van fijos porque fallaron por interacciones de 3+ y no hay
# garantia de que el generador los reproduzca.
PINNED: list[tuple[str, dict]] = [
    # Encontro 12 errores de lint que la matriz elegida a mano no veia.
    (
        "pinned:todo-activado",
        dict(
            ml_type="supervisado",
            task_type="clasificacion",
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
            use_sdd=True,
            use_integration=True,
            use_conformal=True,
            use_mcp=True,
            use_demo=True,
            # Todos los servidores a la vez: es el unico combo donde el
            # generador de .mcp.json y de opencode.json tiene que encadenar
            # varias entradas, que es donde se cuela una coma de mas.
            # Tupla y no lista: el informe de cobertura mete cada valor en un
            # set, y una lista no es hasheable.
            mcp_servers=(
                "filesystem (acotado a data/ y reports/)",
                "git (historial y diffs en solo lectura)",
                "fetch (descarga paginas web — CONTENIDO NO CONFIABLE)",
                "sqlite (consulta bases SQLite de data/)",
                "time (fecha y hora, zonas horarias)",
            ),
            graphify_mode="graphify + obsidian vault",
        ),
    ),
    # hibrido+regresion entrenaba clasificadores sobre un target continuo.
    ("pinned:hibrido-regresion", dict(ml_type="hibrido", task_type="regresion")),
    # Minimo absoluto: todo apagado.
    ("pinned:minimo", dict(ml_type="supervisado", task_type="clasificacion")),
]

#: Variables cuyas condiciones `when:` gobiernan al resto.
DRIVERS = ["ml_type"]

_SUP_HIB = {"supervisado", "hibrido"}
_SOLO_SUP_HIB = ("model_type", "use_shap", "use_xgboost", "use_lightgbm", "use_catboost")
_SOLO_NN = ("nn_model", "optimizer_type", "nn_loss_fn", "use_calibration")


def aplica(var: str, combo: dict) -> bool:
    """Traduce las condiciones `when:` de copier.yml a codigo."""
    ml = combo.get("ml_type")
    if ml is None:
        return True
    if var in _SOLO_SUP_HIB:
        return ml in _SUP_HIB
    if var == "cluster_model":
        return ml == "no_supervisado"
    if var in _SOLO_NN:
        return ml == "redes_neuronales"
    if var == "task_type":
        return ml != "no_supervisado"
    if var == "use_conformal":
        return ml in _SUP_HIB or ml == "redes_neuronales"
    return True


def variables_de_copier() -> tuple[dict[str, list], dict]:
    """
    Lee las opciones directamente de copier.yml.

    Leerlas en vez de copiarlas a mano es lo que impide la deriva: una opcion
    nueva entra sola en la matriz, y no puede pasar lo de `use_rag` — que
    existia desde hacia versiones y no aparecia en ninguna combinacion.

    `proyecto_perfil` no ramifica en la matriz: es un driver global cuyos 4
    valores solo cambian los DEFAULTS de los extras, y la logica del render
    depende de los valores de `use_*`/`graphify_mode`, no del perfil. Se deja
    fijado en `DEFAULTS` (estandar) y se excluye aqui.
    """
    import yaml

    doc = yaml.safe_load((Path(__file__).parent.parent.parent / "copier.yml").read_text())
    variables: dict[str, list] = {}
    defaults: dict = {}
    for nombre, spec in doc.items():
        if nombre.startswith("_") or not isinstance(spec, dict) or "type" not in spec:
            continue
        if nombre == "proyecto_perfil":
            continue
        if "choices" in spec:
            valores = list(spec["choices"])
        elif spec["type"] == "bool":
            valores = [True, False]
        else:
            continue  # los str libres no ramifican el render
        variables[nombre] = valores
        # Los defaults derivados del perfil son strings Jinja ("{{ ... }}"):
        # aqui se resuelven al valor que tendria el perfil estandar, porque la
        # matriz no puede renderizar un string Jinja como si fuera el valor.
        default = spec.get("default", valores[0])
        if isinstance(default, str) and default.startswith("{{"):
            default = DEFAULTS.get(nombre, valores[0])
        defaults[nombre] = default
    return variables, defaults


def construir_combos() -> list[tuple[str, dict]]:
    sys.path.insert(0, str(Path(__file__).parent))
    from pairwise import etiqueta, generate

    variables, defaults = variables_de_copier()
    generados = generate(variables, defaults, aplica, DRIVERS)
    claves = ["ml_type", "task_type", "graphify_mode"]
    combos = [(f"{i:03d}:{etiqueta(c, claves)}", c) for i, c in enumerate(generados)]
    return PINNED + combos


COMBOS: list[tuple[str, dict]] = construir_combos()

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
        touched = feat.get("touched_files")
        if touched is not None and (
            not isinstance(touched, list) or not all(isinstance(f, str) and f for f in touched)
        ):
            problems.append(f"{feat['id']}: touched_files debe ser una lista de rutas")

    for feat in features:
        if not isinstance(feat, dict):
            continue
        for dep in feat.get("depends_on", []):
            if dep not in ids:
                problems.append(f"{feat.get('id')}: depends_on '{dep}' no existe en este combo")

    reclamados: dict[str, list[str]] = {}
    for feat in features:
        if not isinstance(feat, dict):
            continue
        for f in feat.get("touched_files", []):
            reclamados.setdefault(f, []).append(feat["id"])
    for f, ids in reclamados.items():
        if len(ids) > 1:
            problems.append(f"{f} reclamado por varias features: {', '.join(ids)}")

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


def informe_cobertura() -> str:
    """Cuantos pares de (variable, valor) cubre la matriz. Si baja, se nota."""
    sys.path.insert(0, str(Path(__file__).parent))
    from pairwise import _pairs

    variables, _ = variables_de_copier()
    objetivo = _pairs(variables, aplica, list(variables), DRIVERS)
    cubiertos = set()
    claves = list(variables)
    for _, combo in COMBOS:
        for i, a in enumerate(claves):
            for b in claves[i + 1 :]:
                if a in combo and b in combo and aplica(a, combo) and aplica(b, combo):
                    cubiertos.add((a, combo[a], b, combo[b]))
    falta = objetivo - cubiertos
    pct = 100 * (1 - len(falta) / len(objetivo)) if objetivo else 100.0
    return f"{len(objetivo) - len(falta)}/{len(objetivo)} pares ({pct:.1f}%)"


print(f"BASE         : {BASE}")
print(f"Combinaciones: {len(COMBOS)}  ({len(PINNED)} fijas + all-pairs)")
print(f"Cobertura    : {informe_cobertura()}")
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
                if rel == "harness/featureslist.json":
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
