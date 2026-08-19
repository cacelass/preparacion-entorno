#!/usr/bin/env bash
#
# init.sh — Puerta de entrada del arnés de {{ project_name }}
#
# Decide si el proyecto está en un estado suficientemente bueno para que un
# agente de IA empiece a trabajar. Si algo falla, el agente PARA — no intenta
# arreglarlo por su cuenta ni sigue implementando encima de un proyecto roto.
#
# Uso:
#   ./init.sh              verificación completa (estructura + tests)
#   ./init.sh --quick      omite la suite de tests (solo checks estructurales)
#   ./init.sh --json       salida JSON para consumo por agentes
#
# Códigos de salida:
#   0  ENTORNO LISTO — puedes trabajar
#   1  ENTORNO BLOQUEADO — para y reporta el fallo al usuario
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

QUICK=0
JSON=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --json)  JSON=1 ;;
    -h|--help)
      sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "init.sh: opción desconocida '$arg'" >&2; exit 1 ;;
  esac
done

ERRORS=0
WARNINGS=0
REPORT=""

record() {  # record <status> <check> <detail>
  REPORT="${REPORT}${1}"$'\t'"${2}"$'\t'"${3}"$'\n'
}

ok()   { record "ok"   "$1" "$2"; [ "$JSON" -eq 1 ] || printf '  \033[32m✔\033[0m %-22s %s\n' "$1" "$2"; }
warn() { record "warn" "$1" "$2"; WARNINGS=$((WARNINGS + 1)); [ "$JSON" -eq 1 ] || printf '  \033[33m⚠\033[0m %-22s %s\n' "$1" "$2"; }
fail() { record "fail" "$1" "$2"; ERRORS=$((ERRORS + 1));   [ "$JSON" -eq 1 ] || printf '  \033[31m✘\033[0m %-22s %s\n' "$1" "$2"; }

section() { [ "$JSON" -eq 1 ] || printf '\n\033[1m%s\033[0m\n' "$1"; }

[ "$JSON" -eq 1 ] || printf '\n\033[1m━━ init.sh · arnés de {{ project_name }} ━━\033[0m\n'

# ─────────────────────────────────────────────────────────────────────────────
#  1. Entorno de ejecución
# ─────────────────────────────────────────────────────────────────────────────
section "Entorno"

PY=""
for candidate in "python{{ python_version }}" python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done

if [ -z "$PY" ]; then
  fail "python" "no se encontró ningún intérprete de Python en PATH"
else
  PY_VERSION="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
  if [ -z "$PY_VERSION" ]; then
    fail "python" "'$PY' existe pero no ejecuta código"
  elif [ "$PY_VERSION" = "{{ python_version }}" ]; then
    ok "python" "$PY $PY_VERSION"
  else
    warn "python" "$PY es $PY_VERSION, el proyecto declara {{ python_version }} — usa el intérprete de .venv"
  fi
fi

if command -v uv >/dev/null 2>&1; then
  ok "uv" "$(uv --version 2>/dev/null | head -1)"
  RUNNER="uv run"
else
  warn "uv" "no instalado — se usará '$PY' directamente (https://docs.astral.sh/uv/)"
  RUNNER="$PY -m"
fi

if [ -d ".venv" ]; then
  ok "venv" ".venv presente"
else
  warn "venv" "sin .venv — ejecuta 'make setup' antes de tocar código"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  2. Ficheros del arnés
# ─────────────────────────────────────────────────────────────────────────────
section "Arnés"

# Migración del layout viejo (featureslist.json y progress/ en la raíz).
# `copier update` trae los ficheros nuevos pero NO borra los viejos, así que
# un proyecto actualizado se queda con las dos copias y el agente escribiendo
# en una mientras alguien lee la otra. Eso no se avisa: se para, porque
# trabajar sobre un backlog duplicado es peor que no trabajar.
if [ -f "featureslist.json" ] || [ -d "progress" ]; then
  fail "layout" "quedan ficheros del arnés en la raíz junto a harness/. Mueve lo tuyo y borra los viejos:"
  printf '        git mv featureslist.json harness/featureslist.json   # si harness/ aún no lo tiene\n'
  printf '        git mv progress harness/progress\n'
  printf '        git mv memory.md harness/memory.md\n'
  printf '        # si harness/ ya trae los del template, quédate con TU version y borra la otra\n'
fi

for required in AGENTS.md harness/featureslist.json harness/progress/current.md harness/progress/history.md; do
  if [ -f "$required" ]; then
    ok "$(basename "$required")" "presente"
  else
    fail "$(basename "$required")" "FALTA — el arnés está incompleto ($required)"
  fi
done

MISSING_AGENTS=""
for agent_def in lider implementer reviewer explorer; do
  [ -f ".opencode/agents/${agent_def}.md" ] || MISSING_AGENTS="$MISSING_AGENTS $agent_def"
done
if [ -z "$MISSING_AGENTS" ]; then
  ok "subagentes" "lider, implementer, reviewer, explorer"
else
  fail "subagentes" "faltan definiciones en .opencode/agents/:$MISSING_AGENTS"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  3. Backlog — harness/featureslist.json bien formado
# ─────────────────────────────────────────────────────────────────────────────
section "Backlog"

if [ -n "$PY" ] && [ -f "harness/featureslist.json" ]; then
  BACKLOG_OUT="$("$PY" - <<'PYEOF'
import json
import sys

VALID_STATUS = ("pending", "in_progress", "done", "blocked")
REQUIRED = ("id", "title", "description", "acceptance_criteria", "status")

try:
    with open("harness/featureslist.json", encoding="utf-8") as fh:
        doc = json.load(fh)
except json.JSONDecodeError as exc:
    print("fail\tJSON inválido: %s" % exc)
    sys.exit(0)
except OSError as exc:
    print("fail\tno se pudo leer: %s" % exc)
    sys.exit(0)

features = doc.get("features") if isinstance(doc, dict) else None
if not isinstance(features, list):
    print("fail\tse esperaba un objeto con la clave 'features' (lista)")
    sys.exit(0)

problems = []
seen = set()
for i, feat in enumerate(features):
    if not isinstance(feat, dict):
        problems.append("feature #%d no es un objeto" % i)
        continue
    missing = [k for k in REQUIRED if k not in feat]
    if missing:
        problems.append("feature #%d sin campos: %s" % (i, ", ".join(missing)))
        continue
    fid = feat["id"]
    if fid in seen:
        problems.append("id duplicado: %s" % fid)
    seen.add(fid)
    if feat["status"] not in VALID_STATUS:
        problems.append("%s: status '%s' no válido (%s)" % (fid, feat["status"], "|".join(VALID_STATUS)))
    criteria = feat["acceptance_criteria"]
    if not isinstance(criteria, list) or not criteria:
        problems.append("%s: acceptance_criteria debe ser una lista no vacía" % fid)
    touched = feat.get("touched_files")
    if touched is not None and (not isinstance(touched, list) or not all(isinstance(f, str) and f for f in touched)):
        problems.append("%s: touched_files debe ser una lista de rutas" % fid)

for feat in features:
    if not isinstance(feat, dict):
        continue
    for dep in feat.get("depends_on", []):
        if dep not in seen:
            problems.append("%s: depends_on '%s' no existe en el backlog" % (feat.get("id"), dep))

for problem in problems:
    print("fail\t%s" % problem)

if problems:
    sys.exit(0)

counts = {status: 0 for status in VALID_STATUS}
for feat in features:
    counts[feat["status"]] += 1

running = [f["id"] for f in features if f["status"] == "in_progress"]
if len(running) > 1:
    print("warn\t%d features en in_progress a la vez: %s" % (len(running), ", ".join(running)))

reclamados = {}
for feat in features:
    for f in feat.get("touched_files", []):
        reclamados.setdefault(f, []).append(feat["id"])
for f, ids in reclamados.items():
    if len(ids) > 1:
        print("warn\t%s reclamado por varias features: %s" % (f, ", ".join(ids)))

print("ok\t%d features · %d pending · %d in_progress · %d done · %d blocked" % (
    len(features), counts["pending"], counts["in_progress"], counts["done"], counts["blocked"]))

nxt = next((f for f in features if f["status"] == "in_progress"), None)
nxt = nxt or next((f for f in features if f["status"] == "pending"), None)
if nxt is None:
    print("ok\tsin trabajo pendiente — backlog vacío")
else:
    print("next\t%s — %s [%s]" % (nxt["id"], nxt["title"], nxt["status"]))
PYEOF
)"
  while IFS=$'\t' read -r status detail; do
    [ -z "$status" ] && continue
    case "$status" in
      ok)   ok   "featureslist" "$detail" ;;
      warn) warn "featureslist" "$detail" ;;
      fail) fail "featureslist" "$detail" ;;
      next) ok   "siguiente tarea" "$detail" ;;
    esac
  done <<< "$BACKLOG_OUT"
else
  fail "featureslist" "no verificable (falta Python o harness/featureslist.json)"
fi
{% if use_sdd %}
# ─────────────────────────────────────────────────────────────────────────────
#  3b. Contrato Gherkin — features/*.feature bien formados
# ─────────────────────────────────────────────────────────────────────────────

if [ -d "features" ]; then
  FEATURE_COUNT="$(find features -name '*.feature' -type f 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$FEATURE_COUNT" -gt 0 ]; then
    FEATURE_BROKEN=""
    for feat in features/*.feature; do
      grep -q '^Feature:' "$feat" || FEATURE_BROKEN="$FEATURE_BROKEN $feat"
      grep -q 'Scenario:' "$feat" || FEATURE_BROKEN="$FEATURE_BROKEN $feat"
    done
    if [ -z "$FEATURE_BROKEN" ]; then
      ok "features" "$FEATURE_COUNT contrato(s) Gherkin válido(s)"
    else
      fail "features" "contratos sin Feature:/Scenario:$FEATURE_BROKEN"
    fi
  else
    ok "features" "directorio presente, sin contratos todavía"
  fi
else
  ok "features" "sin contratos (ejecuta 'run harness write_feature --id <ID>' cuando toque)"
fi
{% endif %}
# ─────────────────────────────────────────────────────────────────────────────
#  4. Código del proyecto
# ─────────────────────────────────────────────────────────────────────────────
section "Proyecto"

if [ -d "{{ project_slug }}" ]; then
  ok "paquete" "{{ project_slug }}/ presente"
else
  fail "paquete" "falta el paquete {{ project_slug }}/"
fi

if [ -f "pyproject.toml" ]; then
  ok "pyproject" "presente"
else
  fail "pyproject" "falta pyproject.toml"
fi

if [ -d "tests" ]; then
  TEST_COUNT="$(find tests -name 'test_*.py' -type f 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$TEST_COUNT" -gt 0 ]; then
    ok "tests" "$TEST_COUNT ficheros de test"
  else
    fail "tests" "tests/ existe pero no contiene ningún test_*.py"
  fi
else
  fail "tests" "falta el directorio tests/"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  5. Suite de tests — la prueba de que el proyecto no está roto
# ─────────────────────────────────────────────────────────────────────────────
section "Verificación"

if [ "$QUICK" -eq 1 ]; then
  warn "pytest" "omitido (--quick) — NO declares ninguna feature como done sin esto"
elif [ "$ERRORS" -gt 0 ]; then
  warn "pytest" "omitido — hay errores estructurales que arreglar antes"
else
  TEST_LOG="$(mktemp)"
  if $RUNNER pytest tests/ -q --no-header >"$TEST_LOG" 2>&1; then
    SUMMARY="$(grep -Eo '[0-9]+ passed[^=]*' "$TEST_LOG" | tail -1 | sed 's/[[:space:]]*$//')"
    ok "pytest" "${SUMMARY:-suite en verde}"
  else
    fail "pytest" "$(grep -E '^(FAILED|ERROR)' "$TEST_LOG" | head -3 | tr '\n' ' ' || echo 'la suite falla')"
    [ "$JSON" -eq 1 ] || tail -15 "$TEST_LOG" | sed 's/^/      /'
  fi
  rm -f "$TEST_LOG"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  Veredicto
# ─────────────────────────────────────────────────────────────────────────────
if [ "$JSON" -eq 1 ]; then
  REPORT="$REPORT" ERRORS="$ERRORS" WARNINGS="$WARNINGS" "$PY" - <<'PYEOF'
import json
import os

checks = []
for line in os.environ.get("REPORT", "").splitlines():
    if not line.strip():
        continue
    parts = line.split("\t")
    if len(parts) < 3:
        continue
    checks.append({"status": parts[0], "check": parts[1], "detail": parts[2]})

errors = int(os.environ.get("ERRORS", "0"))
print(json.dumps({
    "ready": errors == 0,
    "errors": errors,
    "warnings": int(os.environ.get("WARNINGS", "0")),
    "checks": checks,
}, indent=2, ensure_ascii=False))
PYEOF
elif [ "$ERRORS" -eq 0 ]; then
  printf '\n\033[1;32m━━ ENTORNO LISTO ━━\033[0m  %d aviso(s)\n' "$WARNINGS"
  printf 'Puedes trabajar. Siguiente paso: lee harness/progress/current.md y elige la primera feature pendiente.\n\n'
else
  printf '\n\033[1;31m━━ ENTORNO BLOQUEADO ━━\033[0m  %d error(es), %d aviso(s)\n' "$ERRORS" "$WARNINGS"
  printf 'NO empieces a implementar. Reporta estos fallos al usuario y para.\n\n'
fi

[ "$ERRORS" -eq 0 ] || exit 1
exit 0
