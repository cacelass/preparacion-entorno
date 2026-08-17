# Referencia del sistema de agentes

Catálogo completo: qué agentes hay, qué workflows existen, cómo se orquestan
y dónde vive cada cosa. **Se carga bajo demanda a propósito**: son datos que
se consultan, no reglas que se obedecen, y tenerlos en contexto en cada
sesión degrada al modelo sin darle nada a cambio.

Las reglas del proyecto están en `AGENTS.md`.

## Agentes disponibles

Un solo sistema con dos capas. **No son mundos separados: el arnés es la capa
de decisión de tus agentes, no un sustituto de ellos.**

- **Razonan** — `lider`, `explorer`, `implementer`, `reviewer`: markdown en
  `.opencode/agents/`. Deciden *qué* se hace, *cómo* y *cuándo está hecho*.
- **Ejecutan** — los {{ 19 + (1 if use_rag else 0) + (1 if use_sdd else 0) + (1 if use_api else 0) + (1 if use_docker else 0) + (1 if use_mlflow else 0) + (1 if graphify_mode != 'no' else 0) + (4 if proyecto_perfil in ['completo', 'manual'] else 0) }} agentes Python de la tabla de abajo. Acciones
  deterministas, sin ambigüedad. Entre ellos, `harness` es el dueño mecánico
  del backlog y del progreso.
```
[usuario] → lider (primary)
              ├── explorer / implementer / reviewer   ← razonan, contexto limpio
              └── orquestador (subagent)
                      └── agentes Python: harness, plan, test, review, git...
```

La bisagra entre las dos capas es el agente `harness`: el líder **decide** que
una feature está lista, pero es `harness finish` quien la **cierra** — y se
niega si `init.sh` no pasa. Así la regla del arnés es código, no un prompt que
el modelo pueda ignorar.

| Agente | Hace |
|--------|------|
| `git` | Conventional Commits, changelog, release, PRs, tag_release y commit_feature (cierre de features) |
| `data` | EDA, detección de fugas, correlaciones |
| `graph` | Audita figuras (vacías, aspect ratio) |
{% if use_docker %}| `docker` | Lint Dockerfile, valida docker-compose |{% endif %}
| `ml` | Inspecciona modelos, importancia, overfitting |
| `review` | Funciones largas, except desnudos, duplicación |
| `documentation` | Sincroniza README/Makefile, CHANGELOG, bump versión |
| `notebook` | Extrae salidas de notebooks, inserta comentarios |
{% if proyecto_perfil in ['completo', 'manual'] %}| `installer` | Instala agentes externos en `agents/external/` |{% endif %}
| `cicd` | Genera y valida workflows de CI/CD |
| `test` | Ejecuta pytest, resumen cobertura, módulos sin test |
{% if use_sdd %}| `mutation` | **Mutation testing y CRAP**: ejecuta tools/mutate.py (¿muerden los tests?) y mide el riesgo de cambio por función |{% endif %}
| `dependency` | Detecta paquetes desactualizados y vulnerabilidades |
| `secrets` | Escanea secretos hardcodeados |
{% if use_mlflow %}| `mlflow` | Lista runs, mejor run, comparativa rendimiento |{% endif %}
{% if use_api %}| `api` | Verifica endpoints documentados vs declarados |{% endif %}
| `env` | Gestiona el entorno: python version, uv sync, uv add |
| `make` | Valida Makefile, cadena del pipeline, sugiere targets |
| `refactor` | Refactoriza código: type hints, mutable defaults, bare excepts |
| `doctor` | Diagnóstico integral: entorno, git, datos, código, tests, dependencias |
| `plan` | **Jefe de proyecto**: encargo → preguntas → delegación → qué verificar |
{% if proyecto_perfil in ['completo', 'manual'] %}| `audit` | **Auditor del equipo**: mide uso, éxito y duración; propone mejoras |
| `supervisor` | Coordina workers en competición y arbitra la mejor propuesta |{% endif %}
{% if graphify_mode != "no" %}| `knowledge` | Construye y mantiene el grafo de conocimiento + bóveda Obsidian |{% endif %}
{% if proyecto_perfil in ['completo', 'manual'] %}| `research` | Busca papers (arXiv/OpenAlex) relacionados con el proyecto |{% endif %}
| `memory` | **Memoria proactiva**: observa trayectorias de agentes, mantiene un banco estructurado (facts/state/traces) e inyecta contexto para combatir *behavioral state decay* en tareas largas |
| `doc` | **Documentación unificada y navegación del grafo**: busca en graphify (estructura), RAG (semántica) y vault Obsidian (notas) |
| `harness` | **Dueño del arnés**: backlog (`harness/featureslist.json`) y progreso (`harness/progress/`); ejecuta la puerta y **rehúsa cerrar** una feature sin `init.sh` en verde y evidencia real |
{% if use_rag %}| `rag` | **RAG semántico local**: indexa código, prompts, docs y vault en ChromaDB; busca en lenguaje natural y también indexa URLs externas |{% endif %}


## Workflows por dominio

Los workflow skills documentan pipelines completos de dominio (múltiples
agentes, rutas de archivos, pasos secuenciales). Se cargan bajo demanda
con `skill <name>` cuando la tarea abarca todo un dominio.

| Skill | Cuándo cargarlo | Agentes que orquesta |
|-------|-----------------|----------------------|
| `harness_workflow` | Ciclo del arnés: init.sh → backlog → implementar → revisar | `lider`, `explorer`, `implementer`, `reviewer`, `plan` |
| `data_workflow` | Pipeline de datos: ingesta → features | `data`, `graph`, `knowledge` |
| `ml_workflow` | Ciclo de modelo: entrenar → evaluar | `ml`, `mlflow`, `graph`, `knowledge` |
| `dev_workflow` | Desarrollo: review → test → commit → release | `review`, `test`, `git` |
{% if use_api %}| `api_workflow` | API REST: diseño → código → test | `api`, `test`, `refactor`, `docker` |
{% endif %}{% if use_docker %}| `docker_workflow` | Docker: build → lint → compose | `docker`, `cicd` |
{% endif %}{% if use_monitoring %}| `monitoring_workflow` | Monitorización: dashboard → alerts | varios |
{% endif %}{% if use_optuna %}| `optuna_workflow` | Hyperparameter tuning: search → best | `ml`, `mlflow` |
{% endif %}{% if graphify_mode != "no" %}| `knowledge_workflow` | Grafo de conocimiento + vault | `knowledge`, `git` |
{% endif %}{% if use_rag %}| `rag_workflow` | RAG semántico: index → search → URLs externas | `rag`, `plan`, `docsearch` |
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
# `tag_release` es crítica: autoriza con el nombre exacto de la versión.
uv run python -m agents run git tag_release --version 1.9.0 --yes --confirm-string 1.9.0

# Cierre de una feature del arnés (bump + CHANGELOG + commit, sin tag)
uv run python -m agents run git commit_feature --id DATA-001 --title "EDA del dataset" --dry-run true
uv run python -m agents run git commit_feature --id DATA-001 --title "EDA del dataset" --yes

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
# `installer.*` son críticas: autoriza con el nombre exacto del repo.
uv run python -m agents run installer install_from_git --repo_url usuario/mi-agente --yes --confirm-string usuario/mi-agente
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

El agente **primary** es el `lider` del arnés — es con quien hablas por defecto.
El `orquestador` pasó a **subagente**: es el gateway a los {{ 19 + (1 if use_rag else 0) + (1 if use_sdd else 0) + (1 if use_api else 0) + (1 if use_docker else 0) + (1 if use_mlflow else 0) + (1 if graphify_mode != 'no' else 0) + (4 if proyecto_perfil in ['completo', 'manual'] else 0) }} agentes Python,
al que el líder delega las acciones sueltas vía
`uv run python -m agents [ask|run|pipeline|doctor]`.

```
[usuario] → lider (primary)
              │
              ├── explorer / implementer / reviewer  (subagentes, contexto limpio)
              │        └── registran su informe con: run harness record
              │
              └── orquestador (subagent) ── routing por keywords
                       │
                       └── [Python agent system]
                           ├── {{ 19 + (1 if use_rag else 0) + (1 if use_sdd else 0) + (1 if use_api else 0) + (1 if use_docker else 0) + (1 if use_mlflow else 0) + (1 if graphify_mode != 'no' else 0) + (4 if proyecto_perfil in ['completo', 'manual'] else 0) }} agents (harness, git, test, review, docker{% if use_sdd %}, mutation{% endif %}{% if use_rag %}, rag{% endif %}, doc...)
                           ├── GStack pipelines (develop, fix, release...)
                           └── audit trail + contracts
```

Presiona Tab en opencode para cambiar entre agentes.

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
- **Los prompts se derivan del código.** Cada `agents/prompts/<agente>_agent.md`
  tiene la prosa escrita a mano (su criterio) y un bloque `AUTOGEN` con sus
  acciones y sus límites, sacados de `actions()` y de `contracts.py`. Tras
  tocar cualquiera de los dos: `make prompts-sync`. `make prompts-check` (y CI)
  falla si se desincronizan — no vuelve a haber dos fuentes de verdad.
- **Los agentes del arnés se escriben una sola vez.** La fuente es
  `.opencode/agents/*.md`; `make assistants-sync` los espeja a
  `.claude/agents/*.md` con el frontmatter que espera Claude Code. Nunca edites
  `.claude/agents/` — está gitignorado y se sobrescribe.
- Si añades un agente nuevo: regístralo en `.opencode/agents/orquestador.md` y en `AGENTS.md`
- Si añades un workflow skill: regístralo en `.opencode/agents/orquestador.md` (con Jinja2 condicional), en `AGENTS.md` y en `agents/evals/runner.py`
- Si añades un agente del **arnés**: crea su `.opencode/agents/<nombre>.md`, regístralo en `opencode.json`, en `HARNESS_AGENTS` de `agents/evals/runner.py` y en la lista de `init.sh`
- Si un recurso nuevo tiene dueño, decláralo en `agents/contracts.py` (los del arnés van en el docstring del módulo, no en `CONTRACTS`)
- Ejecuta `make agents-eval` para verificar que todo funciona
- Los prompts originales en `agents/prompts/` son la fuente de verdad para skills


## Vault Obsidian — memoria compartida del equipo

El directorio `docs/vault/` contiene una bóveda Obsidian que funciona como
memoria compartida del equipo de agentes. Cualquier agente puede leerla, pero
solo `knowledge` la escribe.

```
docs/vault/
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
