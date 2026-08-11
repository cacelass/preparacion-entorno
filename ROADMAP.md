# Roadmap — dskit (backlog del template)

Backlog propio de dskit: las mejoras al *template* y a su sistema de agentes.
Sigue el formato del arnés (ID, título, criterios de aceptación) y la regla
del `lider`: primero al backlog, después se implementa con TDD y evidencia.

## Pendientes

### OMP-007 — GStack: lock de pipeline

Dos `GStack.run()` concurrentes sobre el mismo árbol pueden pisarse los
cambios (uno commitea lo que el otro escribe). El pipeline toma un lock
exclusivo no bloqueante sobre `agents/workspace/gstack/.lock`.

**Criterios**
- `run()` toma `flock(LOCK_EX|LOCK_NB)`; si está tomado, devuelve
  `StackResult(success=False)` con el motivo, **sin ejecutar ningún paso**
- El lock se libera en `finally`; fallo al crear el lock (filesystem ro) deja
  pasar (fail-open, misma filosofía que `policy_guard`)
- `GStack(lock=False)` desactiva el bloqueo de forma explícita
- El evento queda anotado en `events.jsonl` (`pipeline_bloqueado`)
- Tests en `test_gstack_control.py`

### OMP-008 — Harness: la evidencia debe parecer salida de comando

`harness finish` aceptaba cualquier string no vacío como evidencia ("ok",
"hecho"). Ahora rechaza afirmaciones sueltas: la evidencia debe tener la
estructura de una salida de comando real (longitud y nº de palabras mínimos).
La verificación de que es *verdad* sigue siendo `init.sh` (gate); este check
solo obliga a documentar esa ejecución.

**Criterios**
- `_evidencia_plausible()`: rechaza texto < 24 chars o < 3 palabras
- `finish()` devuelve `success=false` + `needs` para "ok"/"hecho"/"los tests pasan"
- Tests existentes con `evidence="ok"` actualizados a salida real; nuevos
  tests de rechazo en `test_harness_agent.py`

### OMP-006 — Corpus: RL, metaheurísticas, modelos fundacionales y guardarraíles

Ampliar `docs/knowledge/` con los huecos de teoría profunda que el `lider` no
puede improvisar: aprendizaje por refuerzo, algoritmos genéticos / metaheurísticas
/ búsqueda, y modelos fundacionales (más allá de `llms-aplicados.md`). Añadir
nuevo fichero `ml/guardarraíles.md`.

**Criterios**
- `ml/reinforcement-learning.md`: MDP, value/policy iteration, DQN/PPO/SAC,
  off/on-policy, reward shaping, evaluación y su "cómo se rompe"
- `ml/metaheuristica.md`: algoritmos genéticos, recocido simulado, búsqueda
  local/global; cuándo usar frente a optimización basada en gradiente
- `ml/modelos-fundacionales.md`: pre-training, adapters, destilación,
  evaluación y coste de FMs — complementa, no duplica, a `llms-aplicados.md`
- `ml/guardarraíles.md`: capas de contención (frontera de entrada, filtros de
  salida, acciones limitadas, red teaming, monitoreo) — y referencia desde
  `fairness-y-seguridad.md`
- `docs/knowledge/index.md`, `sources.md` y `sources.json` registran los nuevos
  ficheros y sus topics/queries
- El RAG indexa los ficheros nuevos como `file_type: knowledge`

### OMP-001 — Git: split atómico de commits

`git` agent separa cambios no relacionados en commits atómicos ordenados
topológicamente; valida los tipos Conventional; excluye lock files; rechaza
ciclos. Inspirado en `omp commit`.

**Criterios**
- `git_tool` agrupa los cambios del árbol en áreas (fuente > tests > docs >
  config) y detecta conjuntos no relacionados
- Los lock files (`uv.lock`, `package-lock.json`…) quedan fuera del análisis
- El mensaje de cada commit pasa la validación de los 11 tipos Conventional
- El orden entre commits respeta dependencias (código antes que docs/tests)
- `--dry-run` propone el plan sin tocar nada; la ejecución real exige
  confirmación explícita (puerta de permisos)
- Los ciclos (un cambio que depende de otro incluido en un commit posterior)
  se rechazan antes de escribir nada

### OMP-002 — Memory: edición por id + scoping

`memory` agent gana `memory_edit` (update/forget/invalidate por id) y scope
(global/per-proyecto); los subagentes heredan la memoria del padre.
Inspirado en mnemopi.

**Criterios**
- `memory_edit --id <id> --action update|forget|invalidate` modifica el bank
  sin reescribir el resto
- El bank distingue scope `global` y `per-proyecto`; una búsqueda puede
  acotarse a uno
- Los subagentes lanzados por un agente heredan el scope del padre
- `memory status` refleja el scope y el número de entradas por kind

### OMP-003 — Reglas derivadas de fallos (patrón ttsr/omfg)

Documentar como patrón del arnés: una regla que solo cuesta cuando se viola,
derivada de un fallo real y validada contra el historial (habría disparado o
no). Inspirado en `ttsr`/`/omfg` de omp.

**Criterios**
- AGENTS.md documenta el patrón en "El arnés se automejora": regla → validar
  que habría disparado → registrar
- memory.md guarda las lecciones de omp.sh y los videos con qué adoptar y qué
  rechazar

### OMP-004 — Review: severidad y confianza en los hallazgos

`review` agent devuelve cada hallazgo con severidad P0-P3 y nivel de
confianza, y un veredicto correct/incorrect. Nada importante queda enterrado
en prosa. Inspirado en `/review` de omp.

**Criterios**
- Cada hallazgo de `review_package` lleva `severity` (P0-P3) y `confidence`
  (high/medium/low)
- El resultado incluye un veredicto y la lista ordenada por severidad
- Los tests cubren la asignación de severidad y el ordenado

### OMP-005 — RAG: extractores site-aware para URLs

`rag index_urls` detecta el dominio y extrae markdown estructurado con anclas
en vez de HTML genérico: GitHub (README/raw), Stack Overflow (pregunta +
respuestas) y arXiv (reutiliza `knowledge_tool`). Sin dependencias nuevas.

**Criterios**
- Una URL de GitHub/Stack Overflow/arXiv se indexa con estructura (título,
  secciones, código, enlaces) en lugar de texto plano
- El resto de URLs siguen usando el convertidor genérico actual
- Cada extractor es stdlib y está testado con HTML de ejemplo

### DOC-001 — Verificar el movimiento de la documentación a docs/

Cerrar la verificación del movimiento de `vault/` y `knowledge/` a `docs/`.

**Criterios**
- Render de prueba (estandar/completo) pasa `init.sh`, `make index-rag`,
  `rag search --file_type knowledge` y `doc vault_grep`
- `make docs` sigue compilando Sphinx
- Cero referencias de raíz a `vault/`/`knowledge/` fuera de `docs/` en el
  template renderizado

### DOC-002 — Lecciones de los videos y de omp.sh en memory.md

Registrar en `memory.md` el análisis de qué ideas encajan con la filosofía de
dskit (Pi, Oh-My-Pi, Oratomic-Caltech, Right-Harness, omp.sh) y cuáles se
descartan y por qué.

### OMP-010 — Trasgo: certeza como señal y codec §1 para ahorrar tokens

Incorporar de `jesusvilela/trasgo` (compresión de contexto en 3 ejemplos, sin
entrenar) las dos ideas que sobreviven la filosofía de dskit: la certeza
tipada (`μ.cert`) como puerta de cierre, y el packet compacto E/S/R/Δ/μ para
el handoff de subagentes. Solo se toca el flujo de agentes; **el RAG y el
corpus de papers no se tocan** (el usuario descarga papers a propósito).

**Criterios**
- `AgentResult.certainty` (0..1, default 1.0) retrocompatible; `dispatch`
  propaga la confianza del ruteo heurístico a la certeza del resultado
- `harness finish` rechaza cerrar si `certainty < 0.6` (explícito o heredado
  del último informe del reviewer), devolviendo `needs` — no cierra con dudas
- `harness record` acepta `--packet` (JSON §1 validado: ejes E/S/R/Δ/μ/§,
  `μ.rol` obligatorio, `μ.cert` 0..1) y lo escribe como frontmatter; la prosa
  sigue conviviendo como `--content`
- `harness next` resume el precedente con el packet (Δ + μ.cert) en vez del
  extracto crudo de 200 caracteres
- `audit` guarda `certainty` en cada entrada; `audit suggest` flagea "éxito
  con certeza baja" como señal de mejora
- `--json` omite `message` cuando `data` lo codifica (no pagar dos veces)
- Boot seed de 3 ejemplos en `harness_workflow.md`; espejo `.claude` sincronizado
- Tests: umbral en finish, validación de packet, propagación en dispatch,
  formato `--json`

### OMP-011 — Plan `scope`: entrevista → objetivo + tickets, sin PRD

Convertir la hoja en blanco del arranque (`references/00-objetivo.md` vacío,
backlog sin rellenar) en un formulario guiado. **No es un agente `product` ni
escribe el PRD** — el PRD vivo (`documentation update_prd`) ya nace del
objetivo + backlog y duplicarlo sería rehacer SCOPE-001 (ver lección en
memory.md). La entrevista adaptativa se implementa como acción nueva del
agente `plan`, que ya tiene toda la maquinaria (`needs`, `answer`, borrador).

**Criterios**
- `plan scope` devuelve las preguntas vía `needs` (pregunta, métrica de éxito
  con umbral numérico, datos de partida, criterio de parada — obligatorias;
  usuarios, alcance, riesgos — opcionales); nunca repite lo ya respondido y
  para en el mínimo
- `plan scope_answer` valida que la métrica sea numérica con umbral (el
  criterio #2 de SCOPE-001 — "que funcione bien" no pasa) y acepta
  `aceptar_riesgos`/`descartar_riesgos` para decidir los riesgos detectados
- **Detección de riesgos**: una heurística determinista identifica riesgos en
  las respuestas (login → SQLi, fuga de credenciales; pago → fraude; datos
  personales → GDPR...). `scope_commit` REHÚSA sembrar hasta que el usuario
  decide cada riesgo detectado; los aceptados se siembran como `RISK-NNN`
  "Mitigar: X", los descartados no
- `plan scope_commit` REHÚSA sin las respuestas obligatorias; escribe
  `references/00-objetivo.md` con el spec enriquecido y siembra el backlog en
  orden lógico (SCOPE-001 → RESEARCH-001 → EDA-001 → DATA-001 → FEAT-001 →
  MODEL-001, después las propuestas y los riesgos aceptados) delegando en
  `harness add` (idempotente)
- **Se propone solo**: la primera vez que `harness next` corre en un proyecto
  recién generado (sin `references/00-objetivo.md`), propone `run plan scope`
  — el ticket SCOPE-001 del backlog lo formaliza en su criterio
- El PRD no se entrevista: lo genera `documentation update_prd`, que ahora
  incluye la sección "Riesgos y mitigaciones" (vista de los tickets RISK-*)
- Tests en `test_plan_agent.py` y `test_documentation_agent.py`: detección de
  riesgos, gate de decisión obligatoria, sembrado solo de aceptados,
  no-duplicación, PRD con riesgos

## Cerradas

| ID | Título | Estado |
|----|--------|--------|
| DOC-001 | Verificar el movimiento de la documentación a docs/ | ✅ tests 606, renders OK, init.sh OK, RAG sin duplicados |
| OMP-001 | Git: split atómico de commits | ✅ 10 tests |
| OMP-002 | Memory: edición por id + scoping | ✅ 18 tests |
| OMP-003 | Reglas derivadas de fallos (patrón ttsr) | ✅ documentado en AGENTS.md + memory.md |
| OMP-004 | Review: severidad y confianza | ✅ 6 tests |
| OMP-005 | RAG: extractores site-aware | ✅ 14 tests |
| DOC-002 | Lecciones de los videos y de omp.sh en memory.md | ✅ memory.md |
| OMP-006 | Corpus: RL, metaheurísticas, modelos fundacionales y guardarraíles | ✅ 4 ficheros + refs corregidas + sources.json |
| OMP-010 | Trasgo: certeza como señal y codec §1 | ✅ 21 tests nuevos · 632 totales · CI verde |
| OMP-011 | Plan `scope`: entrevista → objetivo + tickets | ✅ 13 tests · detección de riesgos + gate de decisión · PRD con riesgos |
