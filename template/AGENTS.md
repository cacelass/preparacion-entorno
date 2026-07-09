# Sistema de Agentes — dskit

Este proyecto incluye un sistema de agentes autónomos que automatizan todo el
ciclo de desarrollo: desde análisis de datos hasta release, pasando por
revisión de código, tests, dependencias y despliegue.

## Filosofía

- **No es un chatbot.** Cada agente ejecuta tareas reales (git, docker, tests...),
  no conversa. El ruteo por lenguaje natural es una capa fina sobre acciones
  deterministas.
- **Agnóstico de proveedor de IA.** Python puro, sin SDK de ningún proveedor.
  Cualquier agente de codificación que ejecute comandos de shell puede usarlo.
- **Una responsabilidad por agente.** `GitAgent` no toca datos, `DataAgent` no
  toca Docker. Si una tarea necesita dos agentes, se orquestan en secuencia.
- **Cero dependencias innecesarias.** Usa la stdlib donde puede; reutiliza las
  dependencias del proyecto (pandas, sklearn, etc.).
- **Conoce este template.** Los agentes saben que el código vive en
  `{{ project_slug }}/`, los datasets en `data/`, los modelos en `models/`.

## Agentes disponibles

| Agente | Hace |
|--------|------|
| `git` | Conventional Commits, changelog, release, PRs, tag_release completo |
| `data` | EDA, detección de fugas, correlaciones |
| `graph` | Audita figuras (vacías, aspect ratio) |
| `docker` | Lint Dockerfile, valida docker-compose |
| `ml` | Inspecciona modelos, importancia, overfitting |
| `review` | Funciones largas, except desnudos, duplicación |
| `documentation` | Sincroniza README/Makefile, CHANGELOG, bump versión |
| `notebook` | Extrae salidas de notebooks, inserta comentarios |
| `installer` | Instala agentes externos en `agents/external/` |
| `cicd` | Genera y valida workflows de CI/CD |
| `test` | Ejecuta pytest, resumen cobertura, módulos sin test |
| `dependency` | Detecta paquetes desactualizados y vulnerabilidades |
| `secrets` | Escanea secretos hardcodeados |
| `mlflow` | Lista runs, mejor run, comparativa rendimiento |
| `api` | Verifica endpoints documentados vs declarados |
| `env` | Gestiona el entorno: python version, uv sync, uv add |
| `make` | Valida Makefile, cadena del pipeline, sugiere targets |
| `refactor` | Refactoriza código: type hints, mutable defaults, bare excepts |
| `doctor` | Diagnóstico integral: entorno, git, datos, código, tests, dependencias |
| `schedule` | Valida, describe y calcula próximas ejecuciones de expresiones cron |
| `plan` | **Jefe de proyecto**: encargo → preguntas → delegación → qué verificar |
| `audit` | **Auditor del equipo**: mide uso, éxito y duración; propone mejoras |

## Roles y límites — `agents/contracts.py`

Cada agente tiene un contrato: qué puede (`can`), qué NO puede y a quién
derivarlo (`cannot`), qué información necesita (`needs`) y qué recursos posee
en exclusiva (`owns`). Un recurso, un dueño: dos agentes nunca escriben el
mismo archivo (validado por test). Si a una acción le falta información,
devuelve preguntas (`AgentResult.needs`) — nunca inventa valores.

Ver el contrato de cualquier agente: `uv run python -m agents describe git`.

## Flujo humano: describir → responder → verificar

```bash
uv run python -m agents plan "corre los tests; haz un tag release del proyecto git"
uv run python -m agents run plan answer --order <id> --step1-version 2.0.0
uv run python -m agents run plan execute --order <id>
uv run python -m agents audit suggest   # mejorar el equipo con datos, no impresiones
```

## GStack — Flujos autónomos (Git Stack)

GStack encadena operaciones de múltiples agentes en una pila que se ejecuta
secuencialmente, con commits automáticos entre cada paso. Es el nivel máximo
de automatización: los agentes trabajan y commitean por sí solos.

### Pipelines predefinidos

```bash
# Desarrollo autónomo: review → test → commit
uv run python -c "from agents.gstack import auto_develop; auto_develop()"

# Release autónomo: bump version → changelog → commit → tag
uv run python -c "from agents.gstack import auto_release; auto_release('1.2.0')"

# Corrección autónoma: test → review → fix → commit
uv run python -c "from agents.gstack import auto_fix; auto_fix()"

# Ciclo iterativo: review → test (x3) → commit
uv run python -c "from agents.gstack import auto_commit_cycle; auto_commit_cycle(phases=5)"
```

### Stack personalizada

```python
from agents.gstack import GStack

stack = GStack(auto_commit=True)
stack.push("data", "eda_report", filename="dataset.csv")
stack.push("ml", "check_overfitting")
stack.push("git", "commit_with_changelog", message="feat: EDA + validación overfitting")
result = stack.run()
print(result.summary)
```

## Uso rápido

```bash
# Listar agentes
uv run python -m agents list

# Ejecutar agente directamente
uv run python -m agents run git suggest_commit_message

# Pipelines autónomos (sin intervención humana)
uv run python -m agents pipeline analyze     # diagnóstico completo
uv run python -m agents pipeline develop     # review → test → commit
uv run python -m agents pipeline fix         # test → review → fix → commit
uv run python -m agents pipeline release --version 1.0.0
uv run python -m agents pipeline data --filename dataset.csv

# Doctor: diagnóstico + auto-fix
uv run python -m agents doctor
uv run python -m agents doctor --fix

# Ruteo por lenguaje natural
uv run python -m agents ask "revisa el Dockerfile"

# Release completo (versión + changelog + CI + commit + tag)
uv run python -m agents run git tag_release --version 1.9.0

# Entorno: sync, check, info
uv run python -m agents run env sync
uv run python -m agents run env check_lock_sync
uv run python -m agents run env info

# Makefile: validar pipeline y sugerir targets
uv run python -m agents run make validate
uv run python -m agents run make check_pipeline_chain
uv run python -m agents run make suggest_targets

# Refactor automatizado (dry-run primero)
uv run python -m agents run refactor fix_bare_excepts --dry-run true
uv run python -m agents run refactor add_type_hints
uv run python -m agents run refactor fix_mutable_defaults
```

## Instalación de agentes externos

```bash
uv run python -m agents run installer install_from_git --repo_url usuario/mi-agente
```

## Arquitectura

```
agents/
├── gstack/              # ← NUEVO: flujos autónomos con auto-commit
│   ├── __init__.py
│   ├── stack.py          # Cola de operaciones + ejecución secuencial
│   └── pipelines.py      # Pipelines predefinidos (auto_develop, auto_release...)
├── core/                 # BaseAgent, AgentResult, registro
├── agents/               # Implementaciones de agentes
├── tools/                # Herramientas (git, docker, pandas...)
├── prompts/              # Fichas markdown por agente
├── workspace/            # Archivos generados por agentes
└── external/             # Agentes de terceros
```

## Ver también

- `agents/README.md` — documentación completa del sistema de agentes
- `agents/prompts/` — fichas de cada agente
- `CHANGELOG.md` — historial de cambios del template
