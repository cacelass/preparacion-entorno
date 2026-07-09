# agents/ — Sistema de agentes de este template

Sistema de agentes especializados, tipo plugin, integrado en el template
`dskit`. Cada proyecto generado con Copier a partir de este template incluye
esta carpeta completa y funcional desde el primer commit.

> 📄 **Novedad:** `AGENTS.md` en la raíz del proyecto — documentación rápida.
> **`agents/gstack/`** — flujos autónomos con auto-commit entre pasos, result passing,
> branching condicional y event logging.
> **`agents/helpers.py`** — `delegate_to()` para colaboración entre agentes.
> **Nuevo agente:** `doctor` — diagnóstico integral del proyecto.
>
> 🧭 **Trabajo en equipo:** `agents/contracts.py` — contratos de rol (qué puede,
> qué NO puede, qué necesita y qué recursos posee cada agente, con validación
> de que nadie pisa a nadie). **Agente `plan`** — describes el encargo, él
> pregunta lo que falte y delega. **Agente `audit`** + `agents/audit.py` —
> toda ejecución queda registrada y es medible para mejorar el equipo.
> Ver [Flujo de trabajo humano](#flujo-de-trabajo-humano-describir--responder--verificar).

## Filosofía

- **No es un chatbot.** Cada agente ejecuta tareas reales con herramientas
  (git, docker, pandas, sklearn...), no conversa. El punto de entrada en
  lenguaje natural (`Orchestrator.dispatch`) es una capa fina de ruteo por
  encima de acciones deterministas, no un envoltorio de un LLM.
- **Agnóstico de proveedor de IA.** `agents/` es Python puro, invocado por
  CLI (`python -m agents ...`). No importa el SDK de Anthropic, ni el de
  ningún otro proveedor, ni depende de ningún formato propio de una
  herramienta concreta (Claude Code, Cursor, opencode...). Cualquier agente
  de codificación que pueda ejecutar comandos de shell puede usar este
  sistema exactamente igual.
- **Una responsabilidad por agente.** `GitAgent` no toca datos, `DataAgent`
  no toca Docker. Si una tarea necesita dos agentes, se orquestan en
  secuencia (ver `DocumentationAgent.update_changelog`, que llama a
  `GitAgent` internamente) — no se crea un agente todopoderoso.
- **Cero dependencias innecesarias.** Todo lo que puede hacerse con la
  librería estándar de Python, se hace ahí (`subprocess`, `ast`, `sqlite3`,
  `urllib`, `importlib.metadata`...). Las dependencias reales del template
  (pandas, numpy, scikit-learn, matplotlib, joblib) se reutilizan; nada
  nuevo se añade a `pyproject.toml`.
- **Los agentes conocen este template**, no son genéricos: saben que el
  código vive en `{{ project_slug }}/`, que los datasets viven en
  `data/raw|interim|processed/`, que los modelos se guardan en
  `models/*.joblib`, etc. (ver `agents/context.py`).

## Arquitectura

```
agents/
├── __init__.py            # API pública: Orchestrator, BaseAgent, AgentResult...
├── __main__.py             # permite `python -m agents ...`
├── cli.py                  # CLI (list / describe / run / ask / plan / audit / tools / pipeline / doctor)
├── config.py                # lee .copier-answers.yml -> ProjectConfig
├── context.py                # SharedContext: rutas + config + workspace por agente
├── contracts.py              # contratos de rol: can/cannot/needs/owns por agente + validación
├── audit.py                  # log JSONL de toda ejecución (lo escribe BaseAgent.run)
├── helpers.py                # delegate_to(): colaboración entre agentes
├── orchestrator.py            # rutea lenguaje natural -> agente + acción
├── exceptions.py               # jerarquía de excepciones propia
├── gstack/                       # flujos autónomos con auto-commit, result passing, branching
│   ├── __init__.py               # GStack, pipelines predefinidos
│   ├── stack.py                  # Cola + ejecución secuencial + conditional run_if + event log
│   └── pipelines.py              # auto_develop, auto_release, auto_fix, auto_analyze...
├── core/
│   ├── base_agent.py            # BaseAgent, AgentResult, best_action()
│   └── registry.py               # @register_agent + auto-descubrimiento
├── tools/                          # herramientas reutilizables, una responsabilidad cada una
│   ├── git_tool.py · docker_tool.py · process_tool.py · filesystem_tool.py
│   ├── data_io_tool.py · dataframe_analysis_tool.py · sklearn_tool.py
│   ├── vision_tool.py · duckdb_tool.py · sqlite_tool.py · rest_tool.py
│   ├── code_analysis_tool.py · notebook_tool.py · agent_installer_tool.py
│   ├── stats_tool.py · validate_tool.py · cache_tool.py · parallel_tool.py · schedule_tool.py
├── agents/                          # los 20 agentes (+ plantilla de ejemplo)
│   ├── git_agent.py · data_agent.py · graph_agent.py · docker_agent.py
│   ├── ml_agent.py · review_agent.py · documentation_agent.py
│   ├── notebook_agent.py · installer_agent.py · cicd_agent.py
│   ├── test_agent.py · dependency_agent.py · secrets_agent.py · mlflow_agent.py
│   ├── api_agent.py · env_agent.py · make_agent.py · refactor_agent.py
│   ├── doctor_agent.py             # diagnóstico integral del proyecto
├── schedule_agent.py           # validación y descripción de cron
│   └── _template_agent.py            # plantilla — no se auto-registra (prefijo `_`)
├── external/                          # agentes de terceros / tuyos, fuera del núcleo
│   ├── README.md
│   └── __init__.py
├── workspace/                          # lo que generan los agentes — nunca en la raíz del proyecto
│   ├── README.md
│   └── <agente>/                        # creado bajo demanda por ctx.agent_workspace(nombre)
└── prompts/                            # una ficha markdown por agente (rol, cuándo usarlo)
```

### Flujo de una petición

```
Orchestrator.dispatch("genera el changelog desde el último tag")
  -> select_agent(): puntúa cada agente registrado con can_handle(query)
  -> agente ganador: GitAgent (keywords: "changelog", "git"...)
  -> agent.run("generate_changelog", since_tag=None)
  -> devuelve AgentResult(success, data, warnings)
```

Si prefieres evitar el ruteo por palabras clave, invoca el agente
directamente — es la vía recomendada cuando ya sabes qué necesitas:

```python
from agents.agents.git_agent import GitAgent
result = GitAgent().suggest_commit_message()
print(result.data["suggested_message"])
```

## Flujo de trabajo humano: describir → responder → verificar

El sistema está diseñado para que el humano invierta su tiempo en dirigir y
supervisar, no en ejecutar. El ciclo completo:

```bash
# 1. DESCRIBIR — un encargo en lenguaje natural (un paso por línea o con ';')
uv run python -m agents plan "corre los tests; actualiza el changelog; haz un tag release del proyecto git"

# 2. RESPONDER — el plan te devuelve TODAS las preguntas juntas (nunca inventa valores)
#    y guarda la orden en agents/workspace/plan/orden-<id>.json (editable a mano:
#    puedes cambiar agente/acción de un paso, borrar o reordenar antes de ejecutar)
uv run python -m agents run plan answer --order 20260709-120000 --step2-version 2.0.0

# 3. EJECUTAR — se niega si quedan huecos; delega cada paso al agente dueño
uv run python -m agents run plan execute --order 20260709-120000

# 4. VERIFICAR — resumen paso a paso + auditoría de lo ejecutado
uv run python -m agents audit            # uso, tasas de éxito, duraciones
uv run python -m agents audit failures   # fallos recientes con su mensaje
uv run python -m agents audit suggest    # mejoras propuestas con datos
```

Las tres reglas del equipo (definidas y validadas en `agents/contracts.py`,
test en `tests/test_contracts.py`):

1. **Nadie se pisa.** Cada recurso escribible (CHANGELOG.md, Makefile,
   `.github/workflows/`...) tiene UN dueño. Un agente que necesita tocar un
   recurso ajeno DELEGA en su dueño (`delegate_to`), no lo toca.
2. **Nadie improvisa fuera de su rol.** Cada contrato lista `cannot`: lo que
   ese agente no hace y a quién derivarlo. `python -m agents describe git`
   muestra el contrato completo de cualquier agente.
3. **Nadie inventa información.** Si falta un dato, la acción devuelve
   `AgentResult(success=False, needs=[preguntas])` — pregunta, no adivina.

### Auditoría: mejorar el equipo con datos

Toda ejecución que pasa por `run()` (CLI, Orchestrator, GStack,
`delegate_to`) se registra en `agents/workspace/audit/audit.jsonl` (JSONL,
sin dependencias — legible con `jq` o pandas). El agente `audit` lo agrega:
qué acción falla demasiado, cuál es lenta, qué agentes no se usan nunca.
Después de una temporada de uso, `audit suggest` te dice dónde invertir
(arreglar, documentar, retirar) sin depender de la memoria de nadie.
Punto ciego documentado: las llamadas directas a métodos
(`GitAgent().suggest_commit_message()`) no pasan por `run()` y no se auditan.

## Uso desde la CLI

```bash
uv run python -m agents list
uv run python -m agents describe git
uv run python -m agents run git suggest_commit_message
uv run python -m agents run git tag_release --version 1.9.0
uv run python -m agents run cicd generate_workflow
uv run python -m agents run test run_tests
uv run python -m agents run dependency check_outdated
uv run python -m agents run data eda_report --filename dataset.csv --target-col target
uv run python -m agents run installer install_from_git --repo_url usuario/repo
uv run python -m agents ask "revisa el Dockerfile"

# Jefe de proyecto y auditoría (ver "Flujo de trabajo humano" arriba)
uv run python -m agents plan "corre los tests; haz un tag release del proyecto git"
uv run python -m agents run plan status
uv run python -m agents audit suggest

# Pipelines autónomos
uv run python -m agents pipeline develop
uv run python -m agents pipeline fix
uv run python -m agents pipeline release --version 1.0.0
uv run python -m agents pipeline analyze
uv run python -m agents pipeline data --filename dataset.csv

# Doctor: diagnóstico + auto-fix
uv run python -m agents doctor
uv run python -m agents doctor --fix

# Herramientas
uv run python -m agents tools
```

## Instalar agentes — dos herramientas distintas, no redundantes

- **`installer` (agente, dentro de este sistema)**: instala un agente
  concreto (URL de git, atajo `usuario/repo`, o ruta local) dentro de un
  proyecto que **ya tiene** `agents/`. Es la vía normal para añadir un
  agente nuevo de un tercero o de otro proyecto tuyo.
- **`dskit-agents-installer` (Skill de Claude, fuera de este template)**:
  instala **toda la carpeta `agents/`** en un proyecto que todavía no la
  tiene. Solo hace falta la primera vez, en un proyecto sin este sistema —
  pregunta por ella si la necesitas.

## Los 22 agentes

Los ROLES (qué puede, qué no puede, qué necesita, qué posee) están en
`agents/contracts.py` — esta tabla resume la responsabilidad y las
herramientas de cada uno.

| Agente | Responsabilidad | Herramientas que usa |
|---|---|---|
| `plan` | **Jefe de proyecto**: encargo → orden de trabajo → preguntas → delegación → resumen de qué verificar. No ejecuta nada de dominio él mismo. | Orchestrator, GStack |
| `audit` | **Auditor del equipo**: uso, tasa de éxito y duración por agente/acción, fallos recientes, sugerencias de mejora con datos. | `agents/audit.py` (log JSONL) |
| `git` | Conventional Commits, changelog, release notes, breaking changes, resumen de PR, **commit+changelog en un paso** (`commit_with_changelog`), **release completo** (`tag_release`: versión + changelog + CI/CD + commit + tag) | `git_tool` |
| `data` | EDA: constantes, cardinalidad, missing, outliers, fuga de información, correlaciones | `data_io_tool`, `dataframe_analysis_tool` |
| `graph` | Audita `reports/figures/`: figuras vacías, aspect ratio inusual | `vision_tool` |
| `docker` | Lint de Dockerfile, validación de docker-compose | `docker_tool` |
| `ml` | Inspección de modelos `.joblib`, importancia de variables, overfitting | `sklearn_tool` |
| `review` | Funciones largas, demasiados argumentos, `except` desnudos, duplicación estructural | `code_analysis_tool` |
| `documentation` | README ↔ Makefile desincronizados, actualiza CHANGELOG.md, **sube versión en pyproject.toml+README** (`bump_version`), genera docs Sphinx | `filesystem_tool`, `process_tool`, agente `git` |
| `notebook` | Extrae salidas de un `.ipynb` (imágenes, texto) e inserta interpretaciones como celdas markdown — no interpreta nada él mismo, ver su docstring | `notebook_tool` |
| `installer` | Instala agentes externos (URL de git, atajo `usuario/repo`, o ruta local) en `agents/external/`, valida su estructura, confirma que quedan registrados | `agent_installer_tool` |
| `cicd` | Genera y valida `.github/workflows/*.yml` **del proyecto generado** (no del template), cruzando los `make <target>` invocados contra el Makefile real | `cicd_tool` |
| `test` | Ejecuta pytest, resume fallos/cobertura (JUnit XML + JSON de `pytest-cov`, ambos formatos reales verificados), detecta módulos sin test homónimo | `pytest_tool` |
| `dependency` | Detecta paquetes desactualizados y vulnerabilidades conocidas (OSV vía API de PyPI) contra `uv.lock`, valida sincronía con `uv lock --check`. Necesita internet | `dependency_tool` |
| `secrets` | Escanea el proyecto en busca de secretos hardcodeados. Usa `detect-secrets` si está instalado; si no, un heurístico propio mucho más limitado (avisado explícitamente) | `secrets_tool` |
| `mlflow` | Lista runs del experimento (`project_slug`, misma convención que `train_model.py`), encuentra el mejor por métrica, avisa si el run más reciente empeoró. Solo aplica con `use_mlflow=true` | `mlflow_tool` |
| `api` | Cruza endpoints `@app.get/post(...)` declarados en `api/main.py` contra los documentados en su docstring, y hace un smoke test real con `TestClient`. Solo aplica con `use_api=true` | `api_tool` |
| `env` | Gestiona el entorno de desarrollo: verifica versión de Python, sincroniza dependencias con `uv sync`, `uv lock --check`, añade dependencias con `uv add`. Sin dependencias de red externas (usa el binario `uv` del proyecto) | `process_tool` |
| `make` | Valida y gestiona el Makefile: verifica targets, chequea la cadena del pipeline (`pipeline → predict → train → features → data`), sugiere nuevos targets según la configuración del proyecto (api, monitoring, optuna, mlflow) | `process_tool` |
| `refactor` | Refactoriza código automáticamente: corrige mutables como argumento por defecto, `except:` → `except Exception:`, añade `-> None` a funciones públicas sin tipo de retorno, y detecta `weights_only=False` con sugerencia de corrección. Usa `dry_run=True` por defecto para revisión previa | `code_analysis_tool` |
| `doctor` | Diagnóstico integral del proyecto: python, git, estructura, tests, datos, dependencias, uso de disco. Ofrece `checkup` (todas las verificaciones), `disk_usage`, `summary`. | `process_tool` |
| `schedule` | Valida, describe en lenguaje natural y calcula próximas ejecuciones de expresiones cron. Alias: `@daily`, `@hourly`, `@weekly`, `@monthly`, `@yearly`. | `schedule_tool` |

Cada agente documenta en su propio docstring qué responsabilidades de la
lista original están implementadas y cuáles quedan como extensión (p. ej.
`GitAgent.detect_breaking_changes` solo mira mensajes de commit, no el diff
de la API pública — está señalado explícitamente en su `AgentResult.warnings`).

### Límites conocidos que quedaron documentados construyendo estos agentes

- **`secrets`**: sin `detect-secrets` instalado, no detecta tokens de Slack/Stripe/GitHub/JWT/etc. — solo claves AWS, cabeceras PEM y asignaciones de alta entropía.
- **`mlflow`**: la versión de mlflow que se instala sin fijar versión puede resolver su tracking URI por defecto a un `sqlite:///...mlflow.db` en vez del clásico `mlruns/` — este agente no fuerza ningún tracking_uri, usa el que resuelva mlflow (o `MLFLOW_TRACKING_URI` si lo fijas). Verifica tu versión instalada si esto te importa.
- **`api`**: `fastapi.testclient.TestClient` avisa (deprecation warning) de que prefiere `httpx2` sobre `httpx` en las versiones actuales de Starlette — `httpx` (el que ya declara `pyproject.toml`) sigue funcionando, pero puede que en algún momento quieras migrar la dependencia.

### Agentes que colaboran entre sí (no todo tiene que ser independiente)

`agents/agents/git_agent.py` es el ejemplo de referencia: `commit_with_changelog`
llama a `DocumentationAgent.update_changelog` antes de hacer el commit, y
`tag_release` encadena `DocumentationAgent.bump_version` +
`commit_with_changelog` + `git tag` en un único flujo. El patrón es siempre
el mismo: import perezoso (dentro del método, no a nivel de módulo, para
evitar ciclos de import) + instanciar el otro agente con `context=self.ctx`
+ llamar a `.run("accion", **kwargs)`. No hace falta un mecanismo especial
de orquestación entre agentes — son objetos Python normales.

## Cómo extender el sistema

### Añadir un agente nuevo

1. Copia `agents/agents/_template_agent.py` a `agents/agents/mi_agente.py`
   (quita el `_` inicial).
2. Define `name`, `description`, `capabilities` y los métodos de acción.
3. Nada más — el registro y la CLI lo descubren solos.

### Añadir una herramienta nueva

Créala en `agents/tools/`, decórala con `@register_tool("nombre")` si quieres
que aparezca en `python -m agents tools`, e impórtala desde el agente que la
necesite. Nunca dupliques una herramienta existente entre agentes.

### Agentes externos

`agents/external/` acepta dos vías (ver `agents/external/README.md`):
1. **Un archivo suelto**: mismo patrón `@register_agent`, se auto-descubre.
2. **Un paquete pip instalado** que expone un entry point del grupo
   `dskit.agents` en su propio `pyproject.toml` — útil si el agente externo
   tiene dependencias propias o lo compartes entre varios proyectos.
3. **El agente `installer`** automatiza la vía 1 (clona/copia, valida
   estructura con AST, confirma el registro) — pero instalar código de un
   origen que no controlas sigue siendo ejecución de código arbitrario al
   importarlo. La validación es estructural, no de seguridad — revisa el
   código tú mismo si el origen no es de confianza. Ver
   `agents/prompts/installer_agent.md`.

Ninguna de las tres vías requiere tocar `orchestrator.py`, `cli.py` ni
ningún otro agente.

## Extender el ruteo del Orchestrator

`Orchestrator.select_agent` elige el agente por palabras clave
(`BaseAgent.can_handle`), determinista a propósito (ver Filosofía).
`Orchestrator.dispatch` además adivina la **acción** con
`BaseAgent.best_action` (solapamiento de palabras con el nombre de la
acción) y solo la ejecuta sola si no necesita argumentos obligatorios
(`BaseAgent.can_auto_run`) — si hace falta un argumento que no se puede
adivinar de una frase (un `filename`, un `message`, una `version`...), se
informa de qué falta en vez de inventarlo.

Cuando dos acciones de un mismo agente comparten palabras con la consulta
(p. ej. "commit" aparece tanto en `commit_with_changelog` como en
`suggest_commit_message`), define `action_aliases()` en ese agente para
desambiguar — ver `GitAgent.action_aliases` para el caso real que motivó
este mecanismo. No hace falta definirlo para cada acción, solo donde hay
ambigüedad real.

Si más adelante quieres un ruteo basado en un LLM (de cualquier proveedor —
Anthropic, OpenAI, un modelo local...) decidiendo agente y acción, esos dos
métodos (`can_handle` y `best_action`) son el punto de extensión — no hace
falta tocar ningún agente ni el resto del `Orchestrator`.

---

## Agentes externos y lecturas recomendadas

Nada de lo que sigue viene integrado por defecto: son punteros a proyectos
de terceros que pueden servir de inspiración, o como base para un agente en
`agents/external/`. Las descripciones están verificadas contra la
documentación/README de cada proyecto en el momento de escribir esto — aun
así, estos repos cambian rápido, conviene revisar el estado actual antes de
depender de alguno.

### Frameworks de skills para agentes de codificación

- **[obra/superpowers](https://github.com/obra/superpowers)** — metodología
  de desarrollo compuesta por skills encadenadas (brainstorming → writing-plans
  → ejecución con revisión en dos fases). Se instala como plugin de Claude
  Code (`/plugin install superpowers@claude-plugins-official`) o vía
  `npx skills add`. La skill
  [`brainstorming`](https://www.skills.sh/obra/superpowers/brainstorming) en
  concreto fuerza una fase de diseño con aprobación explícita antes de
  escribir código — encaja bien como disciplina previa a pedirle a un agente
  de este sistema (p. ej. `review`) que toque código.
- **[mattpocock/skills](https://github.com/mattpocock/skills)** — colección
  de skills más pequeñas y componibles (no una metodología monolítica). La
  skill `teach` convierte el directorio actual en un espacio de aprendizaje
  con estado (`MISSION.md`, lecciones HTML numeradas, registro de progreso) —
  pensada para aprender un tema a lo largo de varias sesiones, no directamente
  relacionada con `dskit`, pero reutilizable en cualquier proyecto.
- **skill-creator** (Anthropic,
  [skills.sh/anthropics/skills/skill-creator](https://www.skills.sh/anthropics/skills/skill-creator)) —
  la skill oficial de Anthropic para crear otras skills (estructura de
  `SKILL.md`, cuándo separar en `scripts/`/`references/`/`assets/`, cómo
  redactar la descripción para que el auto-descubrimiento la dispare bien).
  Es la misma que usa este asistente internamente; te la he dejado como
  archivo aparte junto a este proyecto (ver el mensaje de chat) para que la
  tengas disponible sin depender de que seas usuario de Claude Code.

### Memoria a largo plazo para agentes

Ninguno de estos es "memoria para `agents/`" tal cual — son piezas de
infraestructura que podrías conectar si construyeras un agente conversacional
por encima de este sistema (algo que, recuerda, este template evita a
propósito, ver Filosofía).

- **[topoteretes/cognee](https://github.com/topoteretes/cognee)** — motor de
  memoria basado en un grafo de conocimiento auto-alojado (pipeline
  "Extract, Cognify, Load"); combina búsqueda vectorial y de grafo. Se usa en
  ~6 líneas de Python (`cognee.add` / `cognee.cognify` / `cognee.search`) o
  como plugin de Claude Code.
- **[redis/agent-memory-server](https://github.com/redis/agent-memory-server)** —
  servidor de memoria (API REST + servidor MCP) sobre Redis, con memoria de
  trabajo (por sesión) y memoria a largo plazo con extracción automática de
  temas/entidades vía LLM. Soporta OpenAI, Anthropic y otros proveedores vía
  LiteLLM.
- **[FalkorDB/FalkorDB](https://github.com/FalkorDB/FalkorDB)** — base de
  datos de grafos (OpenCypher) optimizada para GraphRAG, usa matrices dispersas
  (GraphBLAS) en vez del modelo de almacenamiento habitual de un grafo. Sirve
  como backend de grafo para sistemas de memoria tipo cognee/Mem0 (hay un
  plugin oficial `FalkorDB/mem0-falkordb`).
- **[supermemoryai/supermemory](https://github.com/supermemoryai/supermemory)** —
  API de memoria/contexto para agentes, con integraciones ya hechas para
  LangChain, LangGraph, OpenAI Agents SDK y la herramienta de memoria de
  Claude, entre otras.
- **[EverMind-AI/EverOS](https://github.com/EverMind-AI/EverOS)** — capa de
  memoria local-first y "Markdown-nativa" (los datos se guardan como
  `.md` + SQLite + LanceDB, sin servicios externos obligatorios), con
  recuperación híbrida (BM25 + vectorial). **Aviso explícito**: en el
  momento de escribir esto, la actividad del repositorio es de apenas unos
  días — es un proyecto extremadamente nuevo, sin trayectoria que evaluar
  todavía. No lo trates como una pieza probada en producción sin revisar tú
  mismo su estado actual.

### Deep research

- **[langchain-ai/local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher)** —
  asistente de investigación web completamente local (LangGraph + Ollama),
  itera búsqueda → resumen → reflexión hasta producir un informe con fuentes.
  LangChain mantiene también
  [`open_deep_research`](https://github.com/langchain-ai/open_deep_research),
  su versión no local con más proveedores de LLM/búsqueda — no estoy seguro
  de cuál de las dos está más activa en este momento, mejor revisar ambos
  repos antes de elegir.
- Los artículos que enlazaste (*Building agent memory with knowledge
  graphs* en theneuralmaze.substack.com, la comparativa de deep research en
  trilogyai.substack.com, y la lista
  [DavidZWZ/Awesome-Deep-Research](https://github.com/DavidZWZ/Awesome-Deep-Research))
  no los he podido leer en profundidad en esta sesión — te los enlazo tal
  cual, sin resumir contenido que no he verificado.

### Bucle de investigación autónoma

- **[karpathy/autoresearch](https://github.com/karpathy/autoresearch)** —
  de Andrej Karpathy: un agente modifica el código de entrenamiento
  (`train.py`), entrena 5 minutos en una única GPU, mide una métrica, hace
  `git commit` si mejora o revierte si empeora, y repite en bucle sin
  intervención humana. El patrón de fondo (proponer → medir → conservar si
  mejora, descartar si no, con git como historial verificable) es
  trasladable fuera de ML: es, básicamente, la misma idea que
  `MLAgent.check_overfitting` podría usar como bucle automático en vez de
  una comprobación puntual — no está implementado así en este template, es
  una dirección de extensión razonable si te interesa.

### Sobre `IBM/drop-agent`

No he podido encontrar un repositorio público con ese nombre exacto en
GitHub — es posible que el nombre haya cambiado, que sea un repo privado, o
que me esté fallando la búsqueda. Antes de construir nada sobre él, confirma
la URL exacta.
