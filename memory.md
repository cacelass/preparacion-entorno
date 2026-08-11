# Memory

Instrucciones, correcciones y preferencias guardadas por el usuario para recordar en futuras sesiones.

---

## Commits

- Usar mensajes convencionales (feat:, fix:, chore:, docs:)
- `ecommits` no está instalado en el entorno — hacer commits manuales
- Taggear tras commits grandes: `git tag -a vX.Y.Z -m "mensaje"`
- "comitea" significa commitear inmediatamente (no preguntar, no esperar cambio a code mode)

## Memory Agent (Jul 2026)

- Implementado `memory_agent.py` (proactive memory pattern, arXiv 2607.08716)
- Memory bank en `agents/workspace/memory/bank.json` con tres kinds: facts, state, traces
- `make skills` instala prompts como skills en `.opencode/skills/`
- `make agents-memory` muestra estado de la memoria de agentes
- Usar `npx autoskills -y` para skills del ecosistema (Node.js)

## Testing (Jul 2026)

- Hypothesis añadido a dev deps del template (`template/pyproject.toml`)
- `test_hypothesis.py` — property-based tests para invariantes de `AgentResult`, `can_handle`, `best_action`, `Contract`
- `test_audit_agent.py` — 12 tests (era 2). Cobertura: report vacío, agregación, tasa de fallo, failures con límite, suggest_improvements (fallo alto, lento, warnings, saludable), líneas corruptas, líneas vacías
- `test_doctor_agent.py` — 15 tests (era 2). Cobertura: todas las secciones, missing pyproject, missing slug, missing `__init__`, tests dir vacío, disk_usage inexistente/con archivos, summary, _human_size edge cases
- `test_docker_agent.py` — 14 tests (era 3). Linter: FROM sin tag, :latest, sin USER, ADD vs COPY, apt-get recommends, apt-get update separado, vacío, solo comentarios
- `test_test_agent.py` — 15 tests (era 4). Parsing JUnit XML (todos pasan, failures, errors, skipped, vacío, malformed, sin testsuite), coverage JSON (0%, 100%, sin totals), skeleton generation
- Los tests del template NO se pueden ejecutar directamente sobre el template sin renderizar (Jinja2 en source files rompe el parser de Python). Para probar: copiar `agents/` a tmp, quitar `{% raw %}` con sed, instalar deps.
- Bug encontrado: `docker_tool.py` comparaba `"apt-get update"` contra `line.upper()` (case mismatch). Arreglado.
- Bug encontrado: `memory` agent registrado sin contrato en `contracts.py`. Añadido.
- Abstracciones removidas: `base_tool.py` (ABC muerta), `prev()` reference resolution de `gstack/stack.py` (DSL no usado), `_discover_entry_points()` de `registry.py` (infraestructura de plugins sin paquetes externos)
- 219 tests pasan en agentes

## Code review checklist

Checklist de áreas clave a inspeccionar y herramientas a incorporar, según revisión exterior:
- **Código**: módulos bajo `{{ project_slug }}/` — data, features, models, utils, visualization. Verificar estructura de paquete Python (src/, `__init__.py`).
- **CI**: `.github/workflows/` — validar comandos existentes, eliminar redundancias.
- **Tests**: `tests/` — listar módulos sin cobertura. Meta ≥80%.
- **Docs**: README, LICENSE, CONTRIBUTING, SECURITY — imprescindibles.
- **Deps**: escaneo de vulnerabilidades (pip-audit), justificar dependencias.
- **Herramientas a integrar**: Black (formato), isort (imports), MyPy strict (tipado), Radon ≤10 (complejidad ciclomática), Bandit (seguridad), pre-commit hooks, pytest-cov --cov-fail-under=80.
- **Tests a añadir**: unitarios por módulo (data, features, modelos), integración (pipeline completo con dataset controlado), E2E (CLI/pipelines). Usar fixtures+parametrización en pytest.
- **Refactorizaciones**: eliminar sobreabstracción (funciones > clases encadenadas si solo hay una variante), eliminar código muerto, simplificar pipelines fragmentados, docstrings estilo Google en API pública.
- **Regla**: no abstracción anticipada. Solo extraer interfaz cuando haya ≥2 implementaciones reales.

## Self-maintenance (Jul 2026)

- dskit tiene su propio `Makefile` en la raíz para auto-mantenimiento
- `make setup` / `make update` / `make skills` desde la raíz de dskit
- `make recommended-tools` en proyectos generados muestra herramientas del ecosistema
- `make recommended-all` en proyectos generados como alias
- `uv pip install eticas-audit` para fairness/bias (ITACA)
- `npm install -g @synsci/openscience` para AI workbench científico

## Workflow Skills (Jul 2026)

- 8 workflow skills creados como Jinja2 en `template/agents/prompts/`: data, ml, dev, api, docker, monitoring, optuna, knowledge
- Cada workflow tiene condicionales `{% if %}` por copier options (`ml_type`, `use_api`, `use_docker`, `use_monitoring`, `use_optuna`, `graphify_mode`)
- Template gateway (`.opencode/agents/orquestador.md`) usa Jinja2 para incluir workflows condicionalmente
- Root gateway lista los 8 workflows con nota "si el proyecto tiene X"
- `AGENTS.md` template tiene tabla de workflows con Jinja2 condicional
- Eval runner en `template/agents/evals/runner.py` verifica smoke/routing/contracts de 27 agentes Python, NO de workflows (son documentales)
- Regenerar skills: `make skills` (o `cp template/agents/prompts/*.md .opencode/skills/`)
- Mantenimiento: añadir workflow = registrar en orquestador.md (Jinja2 cond), AGENTS.md, evals/runner.py

## RAG Agent (Jul 2026)

- Nuevo agente `rag` con ChromaDB + embeddings ONNX (all-MiniLM-L6-v2)
- Indexa: código fuente, prompts de agentes, docs/, vault/, README, AGENTS, CHANGELOG
- `rag index_urls` para indexar documentación externa (librerías, tutoriales)
- Copier option `use_rag` (bool, default false) con exclusión condicional en `_exclude`
- Dependencias: `chromadb>=1.10` (grupo opcional `rag`)
- Contrato en contracts.py, prompts, test, workflow skill
- `make index-rag` en Makefile, CI lo incluye en agent tests
- `.rag-index/` gitignored en root + template

## SDD: spec-driven development (Ago 2026)

Flujo de Robert C. Martin / BettaTech adaptado a dskit (sin tmux, sin agentes
LLM en paralelo — solo restricciones duras en código):

- Copier option `use_sdd` (bool, default false). Excluye/inclye condicionalmente:
  `tools/mutate.py`, `agents/agents/mutation_agent.py`, `agents/tools/mutation_tool.py`,
  `agents/prompts/mutation_agent.md`, `agents/tests/test_mutation_agent.py`, `features/`.
- Agente `mutation` (`mutation_agent.py`): `run_mutation_testing` (ejecuta
  `tools/mutate.py`, resumen killed/survived/score) y `crap_report` (CRAP =
  cc²·(1−cov/100)³+cc, radon + pytest-cov, umbral 30). `target` es OBLIGATORIO
  (si tiene default, el smoke test del runner lo ejecuta y falla).
- `tools/mutate.py`: mutador AST sin dependencias, muta operadores de
  comparación/booleanos/True/False, ejecuta pytest por mutante in-place con
  backup + restore en finally. `--tests` acepta directorio o archivo.
- Gherkin en `harness_agent.py`: estado `spec_ready`, acciones `write_feature`
  (genera `features/<id>.feature` con un escenario por criterio) y `approve`
  (puerta humana → `in_progress`). `validate_gherkin` valida estructura mínima.
- `harness` ahora es dueño de `features/` (contracts.py owns).
- Conteo de agentes en prompts/AGENTS.md/opencode.json: usar
  `{{ 27 + (1 if use_rag else 0) + (1 if use_sdd else 0) }}` (no `{% if %}` fijo).

### PRD vivo (`documentation update_prd`)

- `docs/prd.md` es un documento DERIVADO, no fuente de verdad: se regenera
  desde `references/00-objetivo.md` (SCOPE-001), `harness/featureslist.json`
  y `features/*.feature`. Nunca se edita a mano — se re-ejecuta la acción.
- El `lider` lo invoca al cerrar cada feature (paso 5b del protocolo).
- No es un "agente que escribe el PRD": eso duplicaría SCOPE-001 + la puerta
  Gherkin. El valor es que nace del mismo JSON que guía el arnés y no se desfasa.

## Perfiles de proyecto (Ago 2026)

copier.yml gana `proyecto_perfil` (minimo | estandar | completo | manual,
default estandar). En minimo/estandar NO se pregunta por cada extra — los
defaults se derivan del perfil; solo "manual" pregunta uno a uno.

| Perfil | Agentes | Extras |
|--------|---------|--------|
| minimo | 19 (núcleo) | todo apagado |
| estandar | 21 (núcleo + rag + mutation) | rag+sdd on, mcp/api/docker/mlflow off |
| completo | 29 (todos) | todo on + periféricos |
| manual | según respuestas | pregunta cada uno |

### Gating de agentes (`_exclude`)

- Por extra: `api_agent`, `docker_agent`, `mlflow_agent`, `knowledge_agent`
  (+ prompts + tests) se excluyen si su extra está apagado. `knowledge` solo
  existe si `graphify_mode != 'no'`.
- Poda de periféricos: `installer`, `supervisor`, `research`, `audit`
  (+ prompts + tests) solo en completo/manual.
- `audit.py` (módulo de logging del sistema) NO se excluye — solo `audit_agent.py`.
- Los CONTRATOS de los podados SE MANTIENEN en contracts.py: `validate_contracts`
  tolera "en CONTRACTS pero no registrado" (verificado), así el gating no rompe
  el test de contracts.
- `_routing` en evals/runner.py salta benchmarks cuyo agente no está registrado.
- Conteo de agentes: base 19 + extras; fórmula en prompts:
  `{{ 19 + (1 if use_rag) + (1 if use_sdd) + (1 if use_api) + (1 if use_docker)
  + (1 if use_mlflow) + (1 if graphify_mode != 'no')
  + (4 if proyecto_perfil in ['completo','manual']) }}`

### Sync opt-in

`_tasks` ya NO instala dependencias en minimo/estandar (solo `chmod +x init.sh`
+ `prompts_sync`). El proyecto nace sin venv; README/`make setup` documentan el
primer paso. En completo/manual el uv sync se mantiene.

### Mitigación de riesgos (verificada)

- Nadie importa knowledge/api/docker/mlflow/supervisor/research/audit/installer
  a nivel de módulo; `delegate_to` devuelve `success=false` si el agente no existe.
- `doc`/`plan` degradan sin knowledge/grafo (mensajes de "fuente no disponible").
- Documentado para el arnés en AGENTS.md ("Qué agentes hay según el perfil"),
  harness_workflow.md, orquestador.md, agents_reference.md.

### Lecciones de los videos (BettaTech / Uncle Bob)

1. El 70% del flujo Uncle Bob ya existía en dskit (lider/implementer/reviewer +
   gate en código). Lo que aporta valor: mutation testing, CRAP y contrato Gherkin.
2. **No portar tmux/swarm-forge**: quema tokens y el propio video 1 lo desaconseja.
   La conclusión correcta es "restricciones duras en CI/código, IA de supervisión".
3. La cobertura por líneas no prueba que los tests «muerdan» — un test puede
   cubrir una línea y no detectar su lógica rota. La mutación lo verifica.

### Lección Waymo (Y Combinator, "la demo es el 1% del trabajo")

Principio de diseño para dskit (no es spec de feature):
- La demo es el 1%: la cola de problemas difíciles apenas se mueve con cada ola de IA.
- "Cuenta tus nueves antes de las vistas de tu demo" → el gate de init.sh/evidencia.
- La evaluación y las métricas son el foso → `agents-eval`/`init.sh`/`evals/runner.py`.
- Los fallos en smoke de un render `--defaults` (slug vacío, sin extras) son
  preexistentes del entorno de prueba, no del agente nuevo.

### Lecciones de los videos (Pi / Oh-My-Pi / Oratomic-Caltech / Right-Harness)

Qué adoptar (encaja con la filosofía) y qué rechazar (choca con "simplicidad
primero" / "cero deps"). Los 4 videos validan el diseño; casi todo ya existía.

1. **Pi (Caleb Writes Code)**: núcleo mínimo + arnés extensible, open-closed,
   "build to delete". Es la filosofía exacta de dskit; los "hooks" de Pi = ya
   `policy_guard` (pre-tool) + contratos. Validación, no feature.
2. **Oh-My-Pi (Better Stack)**: model-agnostic, subagentes, review ya existen.
   **Rechazar LSP/DAP nativo y hashline edits** por ahora: dependencias pesadas,
   y dskit no edita diffs contra un LLM. El browser-tool tampoco (research ya
   fetchea).
3. **Hsin-Yuan Huang (Oratomic/Caltech)**: "filosofía → máquina de acumulación
   continua de conocimiento" es la visión ya construida (SCOPE-001, corpus
   knowledge, memory_agent/MemGPT, skills, graphify, RAG, verifiers). Lo único
   nuevo implementado: **el corpus sigue al objetivo** (`rag refresh --topics`
   derivado de `references/00-objetivo.md` + `--from-objective`). El daemon
   continuo autónomo NO: choca con las puertas humanas.
4. **Right Harness (sentdex)**: el arnés es el foso (mismo modelo, mejor arnés →
   mucho mejor). Valida `init.sh`/`agents-eval` como "harness mínimo" que dice
   la verdad frente a benchmarks de proveedor.

### Lecciones de omp.sh (Oh-My-Pi, la web)

Implementado (ROADMAP OMP-001..005):
- `commit_atomic`: split por área, lock files fuera, mensajes Conventional,
  rechazo de ciclos, dry-run primero.
- `memory_edit` + scoping (global/per-proyecto) en el banco de memoria.
- Patrón ttsr: reglas derivadas de un fallo, validadas (habría disparado o no).
- `review` con severidad P0-P3 + confidence + veredicto.
- Extractores site-aware en `rag index_urls` (GitHub/SO/arXiv → markdown).

Rechazado: motor nativo en Rust, snapcompact PNG, collab relay, ACP, github-fs
URLs, browser/imagen/tts, Redis/SQL session stores — son features de un TUI,
no de un template de agentes, o violan "cero deps".

### Nota copier + git (verificación de renders)

Al renderizar el template con copier desde el repo local, si `.git` tiene el
remoto `cacelass/dskit`, copier usa el **mirror cacheado** (`~/.cache/copier/git/`)
con la estructura PUSHEADA — no el árbol local con cambios sin pushear. Un
cambio de rutas local parece "no aplicar" hasta el push. Para verificar renders
con cambios locales: copiar template+copier.yml a /tmp SIN `.git`.

### Lecciones de trasgo (jesusvilela/trasgo, OMP-010)

Qué adoptar y qué descartar del codec de compresión de contexto de trasgo
("enseña a cualquier LLM un lenguaje JSON compacto en 3 ejemplos, 0 training"):

- **Adoptado (F1) — certeza como señal de primera clase (`μ.cert`).**
  `AgentResult.certainty` (0..1, default 1.0). `dispatch` propaga la confianza
  del ruteo heurístico; `harness finish` rechaza cerrar con certeza < 0.6
  (explícita o heredada del último informe del reviewer). Cierra el hueco de
  "enforcement theater" de la revisión dura: la evidencia ya no es solo un
  string, tiene un número que la avala. `audit` la guarda y `suggest` flagea
  "éxito con certeza baja".
- **Adoptado (F2) — packet §1 para el handoff de subagentes.** `harness record
  --packet` valida un JSON E/S/R/Δ/μ (ejes fijos, `μ.rol` obligatorio, `μ.cert`
  0..1, `§` como versión) y lo guarda como frontmatter del informe; la prosa
  sigue siendo `--content`. `next` resume el precedente con el packet (Δ +
  μ.cert) en vez del extracto crudo. Boot seed de 3 ejemplos en
  `harness_workflow.md`. Es un convenio de prompt + validador Python, no un
  esquema nuevo: no hay que entrenar nada.
- **Adoptado (F4) — `--json` sin prosa duplicada.** Si `data` codifica el
  resultado, `message` no viaja: el consumidor es una herramienta/agente y
  pagar dos veces por lo mismo es tirar contexto.
- **Rechazado:** el codec §1 como lenguaje de contexto general (exige modelos
  frontier — la propia tabla de trasgo muestra que 4B/7B fallan — y contradice
  "no es un chatbot" y "provider-agnostic"), la máquina T8/lambda (humo de
  README), y el CLI `trasgo pack/boot` (deps y superficie nueva).
- **Restricción del usuario: el RAG NO se toca.** Ya descarga papers e
  información a propósito para ahorrar tokens; no se le aplica compresión.
