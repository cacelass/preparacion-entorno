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

## Principios de comportamiento

Estos principios aplican a cualquier agente de codificación (Claude Code,
Codex, Cursor, Gemini, Cline, Copilot, opencode...) que trabaje en este
proyecto. No dependen de un proveedor ni de una herramienta concreta.

### Piensa antes de codear

No asumas. No escondas dudas. Superficia los trade-offs.

- Si una instrucción es ambigua, presenta múltiples interpretaciones en vez
  de elegir una en silencio
- Si algo no está claro, pregunta — no inventes
- Si existe un enfoque más simple, dilo
- Para cuando estés confuso: nombra qué no entiendes y pide aclaración

### Simplicidad primero

El mínimo código que resuelve el problema. Nada especulativo.

- Nada de funcionalidades que no se pidieron
- Nada de abstracciones para código que se usa una vez
- Nada de "flexibilidad" o "configurabilidad" no solicitada
- Nada de manejo de errores para escenarios imposibles
- Si 200 líneas pueden ser 50, reescríbelas

### Cambios quirúrgicos

Toca solo lo que debes. No mejoren código ajeno.

- No "mejores" código, comentarios o formato adyacente
- No refactorices cosas que no están rotas
- Respeta el estilo existente, aunque lo harías diferente
- Si ves código muerto no relacionado, menciónalo — no lo borres
- Al borrar código tuyo, elimina imports/variables/funciones que tus cambios
  dejaron sin usar. No toques código muerto preexistente

### Ejecución guiada por objetivos

Define criterios de éxito. Itera hasta verificarlos.

En vez de decir "añade validación", escribe "escribe tests para entradas
inválidas, luego haz que pasen". En vez de "arregla el bug", escribe
"escribe un test que lo reproduzca, luego haz que pase".

Para tareas multi-paso, usa un plan con verificación por paso:

```
1. [Paso] → verificar: [cómo]
2. [Paso] → verificar: [cómo]
```

### Concisión

Sé breve. Di lo mismo con la mitad de palabras.

- Elimina relleno ("I'd be happy to help", "Sure!", "Let me take a look")
- Preserva el contenido técnico: código, comandos, rutas, errores
- Usa frases cortas y directas. Un fragmento vale si es claro
- No repitas lo que el usuario ya sabe
- Una línea vale más que un párrafo

El test: si un ingeniero senior diría "esto es demasiado complicado",
simplifícalo. Si una respuesta puede perder la mitad de palabras sin perder
información, hazlo.

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
| `supervisor` | Coordina workers en competición y arbitra la mejor propuesta |
| `knowledge` | Construye y mantiene el grafo de conocimiento + bóveda Obsidian |
| `docsearch` | Busca/navega el grafo de conocimiento, poda nodos irrelevantes |
| `research` | Busca papers (arXiv/OpenAlex) relacionados con el proyecto |
| `memory` | **Memoria proactiva**: observa trayectorias de agentes, mantiene un banco estructurado (facts/state/traces) e inyecta contexto para combatir *behavioral state decay* en tareas largas |

## Workflows por dominio

Los workflow skills documentan pipelines completos de dominio (múltiples
agentes, rutas de archivos, pasos secuenciales). Se cargan bajo demanda
con `skill <name>` cuando la tarea abarca todo un dominio.

| Skill | Cuándo cargarlo | Agentes que orquesta |
|-------|-----------------|----------------------|
| `data_workflow` | Pipeline de datos: ingesta → features | `data`, `graph`, `knowledge` |
| `ml_workflow` | Ciclo de modelo: entrenar → evaluar | `ml`, `mlflow`, `graph`, `knowledge` |
| `dev_workflow` | Desarrollo: review → test → commit → release | `review`, `test`, `git` |
{% if use_api %}| `api_workflow` | API REST: diseño → código → test | `api`, `test`, `refactor`, `docker` |
{% endif %}{% if use_docker %}| `docker_workflow` | Docker: build → lint → compose | `docker`, `cicd` |
{% endif %}{% if use_monitoring %}| `monitoring_workflow` | Monitorización: dashboard → alerts | varios |
{% endif %}{% if use_optuna %}| `optuna_workflow` | Hyperparameter tuning: search → best | `ml`, `mlflow` |
{% endif %}{% if graphify_mode != "no" %}| `knowledge_workflow` | Grafo de conocimiento + vault | `knowledge`, `git` |
{% endif %}

Los prompts fuente viven en `agents/prompts/` y se instalan como skills con:

```bash
make skills
```

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

## Integración con opencode

Este proyecto tiene un **subagente gateway** (`orquestador`) configurado en `opencode.json`.
Presiona Tab en opencode para cambiar a él. El orquestador delega en los 27 agentes Python
vía `uv run python -m agents [ask|run|pipeline|doctor]`.

```
[opencode assistant]  ←  Tab  →  [orquestador subagent]
                                       │
                                  delegates via CLI (--json mode)
                                       │
                              [Python agent system]
                              ├── Orchestrator.dispatch() ← routing por keywords
                              ├── 27 agents (git, test, review, docker...)
                              ├── GStack pipelines (develop, fix, release...)
                              └── audit trail + contracts
```

### Setup

```bash
make skills          # copy agent prompts → .opencode/skills/
make opencode-init   # verifica que el agente orquestador está configurado
make agents-eval     # smoke + routing + contracts (verifica que todo funciona)
```

### Protocolo ninja (token-optimizado)

1. **Gateway loader** (`.opencode/agents/orquestador.md`, ~68 líneas) — siempre en contexto. Árbol de decisión, tabla de comandos, protocolo A2A, workflows por dominio y skills disponibles.

2. **Skills on demand** (`.opencode/skills/*.md`) — NO se cargan automáticamente. Usa `skill <name>` para cargar un workflow de dominio (pipeline completo) o un agente individual (~15 líneas).

3. **Ejecución** (`uv run python -m agents --json ...`) — los agentes Python hacen el trabajo real. Usa `--json` para salida estructurada que el LLM parsea sin ambigüedad.

```
[usuario] → orquestador.md (68 líneas, siempre en contexto)
                ↓ decide: ask/run/pipeline/doctor
                ↓ carga skill on demand (tool skill) si necesita detalle
                ↓ ejecuta: uv run python -m agents --json <comando>
                ↓ procesa resultado con protocolo A2A
          → [usuario]
```

### Evaluaciones

```bash
make agents-eval                          # smoke + routing + contracts
uv run python -m agents.evals.runner      # igual, más detallado
uv run python -m agents.evals.runner --json   # reporte JSON
uv run python -m agents.evals.runner --smoke  # solo smoke
```

### Mantenimiento

- `make skills` regenera `.opencode/skills/` desde `agents/prompts/`
- Si añades un agente nuevo: regístralo en `.opencode/agents/orquestador.md` y en `AGENTS.md`
- Si añades un workflow skill: regístralo en `.opencode/agents/orquestador.md` (con Jinja2 condicional), en `AGENTS.md` y en `agents/evals/runner.py`
- Ejecuta `make agents-eval` para verificar que todo funciona
- Los prompts originales en `agents/prompts/` son la fuente de verdad para skills

## Vault Obsidian — memoria compartida del equipo

El directorio `vault/` contiene una bóveda Obsidian que funciona como memoria
compartida del equipo de agentes. Cualquier agente puede leerla, pero solo
`knowledge` la escribe.

```
vault/
├── 00_META/IA_index.md         ← Punto de entrada: metadata + topología del equipo
├── 01_PROYECTO/                ← Documentación del proyecto (arquitectura, modelos, roadmap)
├── 02_DATOS/                   ← Documentación de datos (features, fuentes)
├── 04_VISUALIZACIONES/         ← Grafo de conocimiento (regenerado por graphify)
└── 05_AGENTES/                 ← Fichas individuales de cada agente (generadas desde contracts.py)
```

Los agentes usan el vault como fuente de contexto: `plan` consulta
`05_AGENTES/<Agent>.md` para decidir a quién delegar; `data` y `ml` delegan
la escritura de hallazgos en `knowledge` para mantener `02_DATOS/` y
`01_PROYECTO/modelos.md` actualizados.
