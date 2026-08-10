# Roadmap — dskit (backlog del template)

Backlog propio de dskit: las mejoras al *template* y a su sistema de agentes.
Sigue el formato del arnés (ID, título, criterios de aceptación) y la regla
del `lider`: primero al backlog, después se implementa con TDD y evidencia.

## Pendientes

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
