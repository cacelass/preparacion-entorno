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

### Rúbrica del arnés (Aug 2026)

El cierre de features pasó de depender de dos señales sueltas a una rúbrica
binaria única en `template/agents/rubric.py`:

- **La rúbrica tiene dos capas.** `CRITERIOS_PUERTA` (GATE-1..4: init.sh verde,
  evidencia real, reviewer no ha rechazado, μ.cert ≥ umbral) los aplica
  `harness finish` en Python; `CRITERIOS_REVISION` (R-1..R-6) los aplica el
  reviewer como checklist binaria con evidencia. `UMBRAL_CERTEZA = 0.6`
  sustituyó a `FINISH_MIN_CERTAINTY` (borrada de harness_agent.py).
- **Lección clave que motivó el cambio (GATE-3).** En producción un `finish`
  solo miraba la certeza, no el veredicto del reviewer: una rúbrica
  desconectada del gate es "un sistema de alertas llamado gobernanza". Un
  `done` sobre un rechazo del reviewer se salta la revisión entera. La certeza
  explícita alta de quien cierra NO anula el rechazo (comparte el punto ciego
  de quien hizo la feature). Añadido `_VEREDICTO_RE` +
  `_ultimo_veredicto_reviewer(id)` en harness_agent.py.
- **Traza auditable.** `history.md` ahora registra `- **Revisión:** <veredicto>
  · μ.cert <n>` (o "sin informe de reviewer") y `finish` devuelve
  `criterios_puerta` en data. Las decisiones de criterio (librería, arquitectura,
  enfoque) no se bloquean — se declaran con `--decisions` y quedan en el histórico.
- **Reviewer con contexto mínimo.** `reviewer.md` ya NO lee la narrativa del
  implementer (`progress/implementer-<ID>.md`): la justificación transmite el
  punto ciego. Evalúa criterios + diff + evidencia reproducible.
- **Tests:** `test_rubric.py` (7 tests: ids únicos, preguntas binarias, capas
  disjuntas, umbral en rango) + 6 tests de GATE-3 en `test_harness_agent.py`.
  Suite completa del template: **660 pasan, 2 skipped** (sigue sin poder
  ejecutarse sobre el template sin renderizar; ver Testing arriba).
- **Nota técnica para regenerar los espejos .claude:** `uv run
  agents.prompts_sync --write` NO funciona sobre el template (pyproject y
  fuentes con Jinja2 rompen uv/imports). Trabajarlo: copiar `agents/` a tmp,
  quitar `{% raw %}` con sed, y desde ahí llamar
  `sync_assistants(write=True, context=SharedContext(root=<template>, config=ProjectConfig()))`.

### Escalera de fricción en la puerta (Aug 2026)

Ataca la fatiga de aprobaciones (Anthropic mide ~93% de permisos aprobados;
un gate que abre 93/100 es una caseta de peaje, no una puerta) con fricción
proporcional al daño, siguiendo el patrón type-to-confirm de GitHub:

- **Tres niveles**: reversible → no pregunta; `destructive` → `--yes`;
  `critical` (subset de destructive) → `--confirm-string "<nombre exacto>"`.
- **`critical` en contracts.py**: `git.tag_release`, `git.merge_branch`,
  `installer.*`. El token tiene que coincidir con el kwarg-objetivo
  (`version`, `source_branch`, `repo_url`, `local_path`...) declarado en
  `OBJETIVO_CONFIRMACION` de permissions.py — no es un "DELETE" memorizable,
  es la identidad de lo que se toca.
- **Fatiga** (`MAX_APROBACIONES_SIN_FALLO=5`): 5 destructivas aprobadas por
  humano seguidas sin fallo → la siguiente destructiva con objetivo nombrable
  exige también el nombre. Se lee de `audit.jsonl` con el nuevo campo
  `confirmed` (solo confirmación humana real, no DSKIT_ASSUME_YES). Cualquier
  `success=false` corta la racha. Es política fijada como constante, como el
  `UMBRAL_CERTEZA` de la rúbrica.
- **Copia**: `CONSECUENCIAS_CRITICAS` nombra qué se pierde y si es
  recuperable ("la copia es la seguridad, no el marco del diálogo").
- **Claves de implementación**: `confirm` y `confirm_string` se hacen `pop`
  antes de llamar a la acción (no deben filtrarse); `requiere_confirmacion`
  mantiene su firma; la fatiga necesita `ctx` opcional para no romper los
  tests de política pelada. El test end-to-end de installer usaba `--yes`
  para una crítica → ahora exige `--confirm-string`.
- **Fuera de alcance (señalizado)**: `DSKIT_ASSUME_YES=deny` (fail-closed en
  CI — dontAsk > bypassPermissions) y ventana de frescura tipo sudo.
- Suite del template: **677 pasan, 2 skipped**.

## Tickets del arnés: touched_files, claim/release y prioridad (Ago 2026)

Formalización de «si tocan los mismos ficheros, secuencial» y prioridad por
dependencias, siguiendo la filosofía de `template/.opencode/agents/lider.md`
(un recurso, un dueño):

- **`touched_files`** — campo OPCIONAL (lista de rutas) en `harness/featureslist.json`.
  Solo `harness` lo escribe. `harness add` lo crea a `[]`; `start`/`approve` lo
  reinician; `finish`/`block` lo liberan. No reescribimos los seeds del backlog.
- **`harness claim --id <ID> --files "<r1>;<r2>"`** — registra los ficheros que
  toca una feature in_progress; rechaza (success=false + needs) si otro feature
  activo (no done/blocked) ya los reclama. Es la semilla para paralelizar sin pisarse.
- **`harness release --id <ID>`** — libera (touched_files → []).
- **Prioridad** — `HarnessAgent._peso_dependientes(doc)` = cierre transitivo del
  grafo inverso (cuántas features dependen de cada una). `_eligible` ordena por
  peso descendente, sort estable (empates → orden de backlog). `next`/`status`
  exponen `prioridad` en `data`. Sin cambiar la filosofía secuencial (1 feature
  a la vez).
- **Robustez de `next`**: el aviso de `plan scope` ahora dispara si SCOPE-001
  está ENTRE los elegibles (no solo si es eligible[0]), porque el orden por peso
  puede desplazarlo (test: `test_next_propone_plan_scope_en_proyecto_sin_spec`).
- **Esquema validado en 3 sitios** (mismo contrato): `template/init.sh` (bloque
  Python), `template/agents/evals/runner.py::_problemas_backlog`,
  `.github/scripts/validate_template.py::validate_backlog`. WARN si dos features
  reclaman el mismo fichero.
- **Contrato** (contracts.py harness): can += claim/release + prioridad; cannot
  += "dejar que dos features reclamen el mismo fichero".
- **Doc**: prompts `harness_agent.md` (prosa + AUTOGEN a mano, no prompts_sync
  sobre template), `harness_workflow.md` (paso Reclamar + nota), `lider.md` +
  espejo `.claude/agents/lider.md` (paso 3b), `agents_reference.md`, `README.md`,
  `rag_tool.py::_backlog_a_markdown` (indexa touched_files).
- **Tests**: +19 en `test_harness_agent.py` (claim/release, solapamiento,
  finish/block/start liberan, prioridad, empates). Suite harness: **87 pasan**;
  suite completa agentes: 663 pasan / 16 skip (14 fail solo por deps opcionales
  ausentes: sklearn/graphify — no son regresión).
- **AUTOGEN manual**: sobre el template no corre `prompts_sync` (Jinja2 rompe
  imports); las filas claim/release del bloque AUTOGEN de harness_agent.md se
  añadieron a mano y deben coincidir con lo que generaría `make prompts-sync`.
