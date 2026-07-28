# Sistema de Agentes — dskit

Este proyecto incluye un sistema de agentes autónomos que automatizan todo el
ciclo de desarrollo: desde análisis de datos hasta release, pasando por
revisión de código, tests, dependencias y despliegue.

---

# Protocolo del arnés — LEE ESTO PRIMERO

Este fichero es el punto de entrada. Todo agente que trabaje en este proyecto
sigue estos pasos **en orden**, antes de escribir una sola línea de código.

```
1. ./init.sh                    ¿el entorno está sano?   si no → PARA
2. progress/current.md          ¿hay trabajo a medias?   si sí → retómalo
3. featureslist.json            primera feature pendiente con deps en done
4. marcar in_progress           en featureslist.json + rellenar current.md
5. delegar                      explorer → implementer → reviewer
6. verificar                    ./init.sh en verde + criterios uno a uno
7. done                         featureslist.json + resumen en history.md
```

## La regla que no se salta

**Ninguna feature se marca `done` sin que `./init.sh` pase en verde.**

Y no es una instrucción: es código. `harness finish` ejecuta la puerta antes de
tocar el backlog y devuelve `success=false` si está en rojo o si no le pasas
evidencia. No hay forma de rodearlo pidiéndoselo amablemente al modelo — la
única sería editar el JSON a mano, y eso está prohibido explícitamente.

`./init.sh` verifica el entorno, los ficheros del arnés, el formato del backlog
y ejecuta la suite de tests. Si sale `ENTORNO BLOQUEADO`, el agente para y lo
reporta — no implementa encima de un proyecto roto ni arregla el arnés por su
cuenta.

```bash
./init.sh            # verificación completa
./init.sh --quick    # solo estructura, sin tests (no vale para cerrar features)
./init.sh --json     # salida estructurada para consumo por agentes

uv run python -m agents --json run harness gate    # lo mismo, vía agente
```

## Piezas del arnés

| Fichero | Qué es |
|---------|--------|
| `AGENTS.md` | Este fichero. Punto de entrada y reglas del juego |
| `CLAUDE.md` | Puntero a este fichero para Claude Code. No duplica nada |
| `init.sh` | La puerta: decide si se puede trabajar. Exit != 0 → parar |
| `featureslist.json` | Backlog: qué hay que hacer, con criterios de aceptación |
| `progress/current.md` | Estado vivo de la feature en curso |
| `progress/history.md` | Append-only: lo cerrado y con qué evidencia |
| `progress/<agente>-<ID>.md` | Resultado de cada subagente |
| `.opencode/agents/*.md` | Definición de cada agente del arnés |

## Los agentes del arnés

| Agente | Capa | Hace |
|--------|------|------|
| `lider` | razona (primary) | Orquesta el ciclo. No escribe código de producto |
| `explorer` | razona (subagent) | Investiga en **solo lectura** y responde una pregunta |
| `implementer` | razona (subagent) | Implementa **una** feature con sus tests |
| `reviewer` | razona (subagent) | Aprueba o rechaza tras ejecutar la puerta |
| `harness` | ejecuta (Python) | **Único** que escribe `featureslist.json` y `progress/` |

Un recurso, un dueño: nadie edita el backlog ni el progreso a mano; todo pasa
por `harness`. El `implementer` es el único que toca el código de producto.

```bash
uv run python -m agents --json run harness next          # ¿qué toca?
uv run python -m agents --json run harness start --id DATA-001
uv run python -m agents --json run harness record --agent explorer --id DATA-001 --content "..."
uv run python -m agents --json run harness finish --id DATA-001 --evidence "$(make test 2>&1 | tail -5)"
uv run python -m agents --json run harness block --id DATA-001 --reason "falta el dataset"
uv run python -m agents --json run harness add --id API-002 --title "..." --criteria "a;b"
```

## Memoria externa: por qué existe `progress/`

La ventana de contexto se degrada mucho antes de llenarse. Por eso el estado
del trabajo vive en ficheros, no en la conversación:

- **Al lanzar un subagente, no le heredes contexto.** Dale el ID de la feature,
  sus criterios y las rutas que necesita. Nada más.
- **Todo subagente registra su resultado con `harness record` antes de devolver
  el control.** Si solo lo dice en su respuesta, se pierde.
- **El siguiente agente lee `progress/`, no el repositorio entero.**

Las tres memorias del proyecto no se pisan:

| Dónde | Dueño | Plazo |
|-------|-------|-------|
| `progress/` | `harness` | La feature en curso y el histórico de features |
| `agents/workspace/memory/` | `memory` | Trayectorias de ejecución de agentes |
| `vault/` | `knowledge` | Conocimiento estable del proyecto y sus datos |
{% if use_rag %}
Y las tres son buscables: `progress/` y `featureslist.json` entran en el índice
semántico, así que tras cerrar una feature basta con `make index-rag` para poder
preguntarle al histórico en lenguaje natural:

```bash
uv run python -m agents --json run rag search --query "¿por qué elegimos este modelo?"
uv run python -m agents --json run doc search --query "qué se decidió sobre las features"
```
{% endif %}
Detalles del formato en `progress/README.md`.

## Evidencia, no afirmaciones

Un agente no declara que algo funciona: lo demuestra. Cada criterio de
aceptación se cierra pegando la **salida real** del comando que lo prueba.
«Los tests pasan» sin la salida de `pytest` es motivo de rechazo automático.

## El arnés se automejora

Estos ficheros son parte del repositorio, así que se corrigen como cualquier
otro código. Si un fallo se cuela dos veces:

- ¿Es una comprobación automatizable? → a `init.sh`, y deja de depender de que
  alguien se acuerde.
- ¿Es una regla del proyecto? → a este fichero.
- ¿Es un criterio de revisión? → a `.opencode/agents/reviewer.md`.

Deja constancia del cambio en `progress/history.md`.

## Arranque

```bash
./init.sh                                    # verifica que se puede trabajar
make init                                    # lo mismo, vía Makefile
make harness-check                           # solo estructura del arnés
```

Y en el asistente, para arrancar el ciclo:

> Lee `AGENTS.md` y sigue el protocolo: ejecuta `./init.sh`, lee `progress/` y
> elige la primera feature pendiente.

---

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

Un solo sistema con dos capas. **No son mundos separados: el arnés es la capa
de decisión de tus agentes, no un sustituto de ellos.**

- **Razonan** — `lider`, `explorer`, `implementer`, `reviewer`: markdown en
  `.opencode/agents/`. Deciden *qué* se hace, *cómo* y *cuándo está hecho*.
- **Ejecutan** — los {% if use_rag %}30{% else %}29{% endif %} agentes Python de la tabla de abajo. Acciones
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
| `doc` | **Documentación unificada**: busca en graphify (estructura), RAG (semántica) y vault Obsidian (notas) |
| `harness` | **Dueño del arnés**: backlog (`featureslist.json`) y progreso (`progress/`); ejecuta la puerta y **rehúsa cerrar** una feature sin `init.sh` en verde y evidencia real |
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

El agente **primary** es el `lider` del arnés — es con quien hablas por defecto.
El `orquestador` pasó a **subagente**: es el gateway a los {% if use_rag %}30{% else %}29{% endif %} agentes Python,
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
                           ├── {% if use_rag %}30 agents (harness, git, test, review, docker, rag, doc...){% else %}29 agents (harness, git, test, review, docker, doc...){% endif %}
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
