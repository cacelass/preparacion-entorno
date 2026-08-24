# Changelog

Todos los cambios relevantes de esta plantilla se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).

---

## [No publicado]

### Demo: modelos a ONNX ejecutables en el navegador

Nueva opción `use_demo` (por defecto en el perfil `completo`): genera una demo
web estática (`demo/`) con 4 páginas (home, try model, docs y MCP) donde el
modelo entrenado corre en el navegador vía `onnxruntime-web` (WebAssembly),
sin servidor ni build tooling. Es un directorio **local, sin auto-deploy**.

- **`tools/export_onnx.py`** (`make demo-export`): convierte `.joblib`/`.pt` a
  ONNX embebiendo el preprocesado (scaler, PCA) en el grafo. `meta.json`
  describe features y modelos; `docs.html` se renderiza en **Python** desde el
  README (sin librerías JS de terceros).
- **`demo/`**: páginas estáticas con el nav inline (sin fetch de JS). El único
  JS propio es `app.js` (inferencia); `onnxruntime-web` se carga desde CDN.
- Los `.onnx` (KB–MB) viajan en el repo; sin auto-deploy ni GitHub Actions.
- Gating por extras: `demo/`, `export_onnx.py`, el test y el extra `onnx` se
  excluyen si `use_demo=false`; `mcp.html` si `use_mcp=false`.

### OMP-011: plan `scope` — la entrevista que construye el spec y siembra el backlog

Al empezar un proyecto, `plan scope` es la mega entrevista: pregunta lo
necesario (pregunta, métrica con umbral, datos, parada — obligatorias;
usuarios, alcance, riesgos — opcionales), valida la métrica numérica,
`scope_commit` escribe `references/00-objetivo.md` con el spec enriquecido y
siembra el backlog en orden lógico (SCOPE-001 → RESEARCH-001 → EDA-001 →
DATA-001 → FEAT-001 → MODEL-001, después las propuestas) delegando en
`harness add` (idempotente). El PRD no se entrevista: `documentation
update_prd` lo deriva del spec + backlog.

**Se propone solo**: la primera vez que se ejecuta `harness next` en un
proyecto recién generado (sin `references/00-objetivo.md`), el agente propone
`run plan scope` en vez de dejar rellenar el spec a mano. El ticket SCOPE-001
del backlog lo formaliza (criterio: spec construido con `plan scope`).

**Detección de riesgos (heurística del agente)**: al responder la entrevista,
el agente identifica riesgos del dominio (login → SQL injection, fuga de
credenciales; pago → fraude; datos personales → GDPR; upload → path
traversal; API → rate-limit...). `scope_commit` REHÚSA sembrar hasta que el
usuario decide cada riesgo con `aceptar_riesgos`/`descartar_riesgos`; los
aceptados se siembran como `RISK-NNN` "Mitigar: X" con `depends_on` en
SCOPE-001, los descartados no. `documentation update_prd` ahora incluye la
sección "Riesgos y mitigaciones" (vista del backlog, nunca fuente).
13 tests nuevos.

### OMP-010: certeza como señal (μ.cert) y codec §1 (trasgo) + corpus ampliado

**Agentes — certeza y ahorro de tokens (idea `μ.cert` de trasgo):**
- `AgentResult.certainty` (0..1, default 1.0); `dispatch` propaga la confianza
  del ruteo heurístico; `harness finish` rechaza cerrar con certeza < 0.6
  (explícita o heredada del último informe del reviewer).
- `harness record --packet` valida y guarda el informe §1 compacto
  (E/S/R/Δ/μ + `§`) como frontmatter; `next` resume el precedente con el
  packet (Δ + μ.cert) en vez del extracto crudo.
- `audit` guarda `certainty`; `audit suggest` flagea "éxito con certeza baja".
- `--json` omite `message` cuando `data` lo codifica (no pagar dos veces).
- Boot seed de 3 ejemplos en `harness_workflow.md`; espejo `.claude` sincronizado.

**Corpus — tres ficheros nuevos en `docs/knowledge/ml/`:**
- `evals-de-sistemas.md` — golden sets, evals-as-code, property-based para
  agentes, evaluar trayectorias, cuándo el eval miente (la eval es el foso).
- `contexto-y-memoria.md` — la ventana como recurso finito, memoria externa,
  handoff sin heredar contexto, compresión/eviction (fundamenta el codec §1).
- `neurodifuso.md` — ANFIS como candidato de nicho y su "cuándo NO" por la
  explosión combinatoria de reglas.

`index.md`, `sources.md` y `sources.json` registran los nuevos ficheros y
topics (44 topics). El RAG y el corpus de papers no se tocan.

### OMP-006: corpus de conocimiento ampliado

Nuevos ficheros de teoría profunda en `docs/knowledge/ml/`:
- `reinforcement-learning.md` — MDP, value/policy iteration, DQN/PPO/SAC,
  off/on-policy, reward design, sample efficiency y evaluación.
- `metaheuristica.md` — algoritmos genéticos, recocido simulado, búsqueda
  local/global; cuándo frente a gradiente y Bayesian optimization.
- `modelos-fundacionales.md` — pre-training, adaptación (prompt→RAG→LoRA→full
  FT), scaling laws, evaluación y coste de FMs.
- `guardarraíles.md` — capas de contención para modelos generativos expuestos
  (entrada, filtros, acciones limitadas, red teaming, monitoreo).

`fairness-y-seguridad.md` y `diseno-experimentos.md` quedan con la ortografía
corregida ("guardarraíles") y `fairness-y-seguridad.md` referencia al fichero
nuevo en lugar de duplicar la sección. `index.md`, `sources.md` y
`sources.json` registran los nuevos ficheros y topics (41 topics), y se
corrigieron referencias cruzadas rotas entre ficheros del corpus.

### Documentación unificada bajo `docs/`

`vault/` (bóveda Obsidian) y `knowledge/` (corpus de conocimiento profundo) se
mueven bajo `docs/`: `docs/vault/` y `docs/knowledge/`, junto a `docs/source/`
(Sphinx) y `docs/prd.md`. Todos los agentes, el RAG, los prompts, `copier.yml`
y la documentación apuntan a las rutas nuevas. El RAG indexa `docs/source`,
`docs/vault` y `docs/knowledge` sin duplicarlos.

- `knowledge_agent.setup_vault` crea ahora la bóveda en `docs/vault` por
  defecto (antes `knowledge/` — un solo vault, no dos).
- `rag refresh` gana `--from-objective`: si existe `references/00-objetivo.md`
  (SCOPE-001), incluye su pregunta como contexto del informe para que el
  `lider` derive topics desde el objetivo del proyecto. El patrón "el corpus
  sigue al objetivo" queda documentado y `KNOW-001` lo formaliza.

### OMP-001: commits atómicos en `git`

Nueva acción `git commit_atomic`: divide los cambios sin commitear en commits
atómicos por área (código antes que tests, tests antes que docs), excluye los
lock files, valida mensajes Conventional y rechaza ciclos de dependencias antes
de escribir. `--dry-run` propone el plan; sin él escribe en el historial y
pide confirmación (puerta de permisos). Inspirado en `omp commit`.

### OMP-002: edición y scoping de memoria

`memory` gana `memory_edit` (update/forget/invalidate por id) y cada entrada
lleva scope `global`/`per-proyecto` (por defecto per-proyecto; el banco es
compartido, así que los subagentes heredan la memoria del padre). `note`,
`search` y `status` soportan scope. Inspirado en mnemopi.

### OMP-003: reglas derivadas de un fallo

Patrón ttsr documentado en `AGENTS.md`: una regla derivada de un incidente solo
se registra si se valida que habría disparado contra el historial — una regla
que no habría saltado es ruido.

### OMP-004: severidad y veredicto en `review`

Cada hallazgo de `review_package`/`review_file` lleva `severity` (P0-P3) y
`confidence` (high/medium/low), ordenados por severidad, con veredicto
`correct`/`review`/`incorrect` (P0 bloquea). Inspirado en `/review` de omp.

### OMP-005: extractores site-aware en `rag index_urls`

GitHub (README raw), Stack Overflow (título + preguntas/respuestas con código
y enlaces) y arXiv (reutiliza `knowledge_tool`) se indexan con markdown
estructurado en vez de HTML plano. Sin dependencias nuevas (stdlib). El resto
de URLs siguen por el convertidor genérico.

### ROADMAP propio de dskit

Nuevo `ROADMAP.md` en la raíz: backlog del template con el formato del arnés
(IDs + criterios), donde se registran las mejoras y las lecciones de los videos
y de omp.sh (qué adoptar y qué rechazar y por qué).

### Corpus de conocimiento profundo dentro de `use_rag`

El RAG ya no indexa solo el proyecto: con `use_rag` activo el proyecto generado
incluye `knowledge/`, un corpus de teoría profunda que el `lider` consulta
antes de aconsejar (matemáticas, estadística, probabilidad, causalidad,
matrices, algoritmos y su aplicación, métricas, validación, interpretabilidad,
deuda técnica, fairness/seguridad, eficiencia y calidad de código, backend,
frontend y datos). Templado con Jinja: se adapta a `ml_type`, `task_type`,
`nn_model`, `use_api`, `use_docker`, `use_mlflow`, etc.

- **Fuentes con papers reales.** El corpus se autorizó a partir de papers
  canónicos (Shlens PCA, Halko randomized SVD, Adam, Transformer, ResNet,
  XGBoost, CatBoost, von Luxburg spectral clustering, Matrix Cookbook)
  descargados y convertidos a markdown con `markitdown`.
- **Nueva acción `rag refresh`.** Verifica que las fuentes de
  `knowledge/sources.json` siguen vigentes (versión más reciente en arXiv) y
  detecta papers nuevos por topic. `--dry-run` informa sin tocar nada; sin él
  descarga los nuevos a `knowledge/papers/` (HTML de arXiv o PDF→markitdown),
  actualiza `sources.json` y reindexa. `markitdown[pdf]` se añadió al extra
  `rag` como import opcional.
- **Ticket `KNOW-001` en el backlog por defecto.** El arnés formaliza el
  mantenimiento del corpus: descargar papers nuevos si los hay y verificar que
  los existentes siguen siendo útiles.
- **RAG**: `knowledge/` entra en el índice como `file_type: knowledge`; el
  `lider`, `orquestador`, `AGENTS.md` y `rag_workflow` documentan cómo
  consultarlo y mantenerlo.

### Spec-driven: el contrato antes del código

El flujo de Robert C. Martin / BettaTech, adaptado sin tmux ni agentes LLM en
paralelo — solo restricciones duras en código, como el resto del arnés. Nuevo
extra `use_sdd` (activo en el perfil `estandar`):

- **Contrato Gherkin con puerta humana.** `harness write_feature` escribe
  `features/<ID>.feature` (un escenario Given-When-Then por criterio de
  aceptación) y deja la feature en `spec_ready`. Solo `harness approve` —un
  paso explícito del humano— la mueve a `in_progress`. La ambigüedad se
  resuelve antes de codear.
- **Mutation testing sin dependencias.** `tools/mutate.py` muta operadores del
  código (agente `mutation`): si un mutante sobrevive a la suite, hay un hueco
  que la cobertura por líneas no ve. La métrica CRAP
  (`cc²·(1−cov/100)³+cc`, umbral 30) complementa la cobertura con la
  complejidad ciclomática (radon).

### PRD vivo

`docs/prd.md` es un documento **derivado**, no una fuente de verdad: se
regenera con `documentation update_prd` desde `references/00-objetivo.md`
(el SCOPE-001 del arnés), `harness/featureslist.json` y `features/*.feature`.
Nace del mismo JSON que guía el arnés, así que nunca se desfasa — si dice algo
que no coincide con el backlog, se regenera, no se edita a mano. El `lider` lo
invoca al cerrar cada feature.

### Perfiles de proyecto: menos preguntas, menos peso

Nueva opción `proyecto_perfil` (`minimo | estandar | completo | manual`,
default `estandar`). En los perfiles automáticos **no se pregunta** por cada
extra — los defaults se derivan del perfil; solo `manual` pregunta uno a uno.

| Perfil | Agentes | Qué incluye |
|--------|---------|-------------|
| `minimo` | 19 (núcleo) | Harness + agentes de calidad |
| `estandar` | 21 | Núcleo + RAG + spec-driven |
| `completo` | 29 | Todos, incluidos los periféricos |
| `manual` | según respuestas | Cada opción se pregunta |

Dos consecuencias de peso:

- **Gating de agentes por extra.** `api`, `docker`, `mlflow`, `knowledge`,
  `rag`, `mutation` y los periféricos (`installer`, `supervisor`, `research`,
  `audit`) solo se instalan si su extra/perfil lo pide. Un proyecto `minimo`
  baja de 29 a 19 agentes. `delegate_to` devuelve `success=false` si el agente
  no existe — documentado para el arnés en `AGENTS.md`.
- **Sync opt-in.** En `minimo`/`estandar` el proyecto ya no instala
  dependencias al generarse (`make setup` lo hace). Generar pasa de minutos+GB
  a segundos.

### El CI del proyecto generado nunca se había ejecutado — y estaba roto

`validate_template.py` probaba que la plantilla renderiza y `smoke` que el
proyecto arranca. Nadie probaba que el `.github/workflows/ci.yml` que se le
entrega al usuario **pase**. Al ejecutarlo por primera vez, fallaban 4 de sus 9
pasos.

- **`uv sync --frozen` con `uv.lock` en el `.gitignore`.** El lock se creaba al
  generar, pero no se commiteaba, así que el primer push de cualquier proyecto
  moría con `error: Unable to find lockfile at 'uv.lock'`. Ahora el lock se
  versiona: esto no es una librería, es un proyecto de datos, y el lock es lo
  único que hace reproducible el entorno entre máquinas.
- **El paso `Install dependencies` era un escalar YAML plano con `\` de
  continuación.** YAML pliega los saltos en espacios, así que al shell le
  llegaba `uv sync --extra dev --extra supervisado \ --extra api \ ...` con `\ `
  como espacio escapado, y uv recibía argumentos basura. Rompía el CI de
  cualquier proyecto con al menos un extra activo. Ahora es un bloque literal.
- **`mypy --strict` con 193 errores en 20 ficheros.** El gate de tipos jamás
  había pasado. Arreglados los 61 errores del código de producto —incluidos
  varios bugs de firma reales del tipo `def f(x: str = None)`— y configurado
  mypy en `pyproject.toml` con estricto global y relajado para `tests/`: anotar
  `-> None` en 155 funciones de test no atrapa ni un bug, y tener el gate en
  rojo permanente es peor que no tenerlo. El CI ya no pasa `--strict` por línea
  de comandos, de modo que tu máquina y CI comprueban exactamente lo mismo.
- **Dos tests en rojo desde hacía versiones** en `no_supervisado`, ambos por
  tests más estrictos que la implementación: uno exigía `labels_` a todos los
  modelos de clustering cuando `GaussianMixture` no lo tiene (y el código ya
  hacía `hasattr(...) else model.predict(X)`), y otro exigía un gráfico PCA por
  modelo ajustado ignorando que `evaluate_models` salta a propósito los que
  producen menos de dos clusters. Reescritos contra el contrato real.
- **`--cov-fail-under=20`.** Medida la cobertura real: 74% en supervisado y 67%
  en no_supervisado. El umbral no era una puerta, era un adorno — cabía borrar
  dos tercios de los tests sin que CI se inmutase. Subido a 60.

**Y el job que impide que vuelva a pasar.** `.github/scripts/run_generated_ci.py`
genera un proyecto, lo commitea, **lo clona** y ejecuta los pasos `run:` de su
workflow en el clon. El clon es exactamente lo que ve GitHub Actions al hacer
checkout, así que cualquier fichero que el CI necesite y el `.gitignore` se
trague se cae ahí y no en el repositorio de un usuario. Los pasos se leen del
workflow de verdad en lugar de replicarse en el script: una copia se
desincroniza y acabaríamos probando nuestra idea del CI en vez del CI.

### Salud del repositorio y ruta de actualización

- `CONTRIBUTING.md`, `SECURITY.md`, plantillas de issue y de PR, que no existían.
- Sección **«Actualizar un proyecto existente»** en el README: `copier update`
  no se mencionaba ni una vez en 401 líneas, pese a haber 41 versiones
  publicadas. Cubre el commit previo obligatorio, la resolución de conflictos,
  el cambio de opciones y la puerta del arnés después de actualizar.

### El RAG estaba roto de fábrica — reparado y medido

Auditoría del RAG ejecutándolo de verdad contra `chromadb` real, no leyéndolo.
El resumen es que no funcionaba: **`make index-rag` reventaba en cualquier
proyecto recién generado** y nadie se enteraba porque el CI no instalaba el
extra que lo habría destapado.

- **La colección no llegaba a crearse.** `_collection` hacía `get_collection` y
  capturaba `ValueError`, pero chroma moderno lanza `NotFoundError`, que no
  hereda de él. Como el `create_collection` vivía en el `except`, el índice no
  se creaba nunca: `NotFoundError: Collection [dskit-rag] does not exist` en el
  primer `index`, `search` y `status`. Ahora es `get_or_create_collection`.
- **El CI no instalaba `--extra rag`**, así que los tres tests que tocaban
  chromadb se saltaban *siempre* y el crash llevaba ahí sin que nadie lo viera.
  Añadido al workflow; los tests de integración ya no son decorativos.
- **El 22% del corpus era invisible.** El embedder trunca a 256 tokens (~1.000
  caracteres) y el troceador solo tenía suelo, nunca techo: se indexaban chunks
  de hasta 18.467 caracteres de los que solo entraban al vector los primeros
  mil. Medido sobre esta misma plantilla: 332 chunks por encima del límite,
  326.649 caracteres guardados pero irrecuperables. Ahora hay techo duro con
  solape real. Nueva medición: **0 chunks por encima del límite, 0% perdido**.
- **El índice nunca se limpiaba.** Era add-only: editar un fichero dejaba el
  chunk viejo dentro para siempre, y borrarlo no lo sacaba del índice. Peor,
  como el id hasheaba solo los primeros 80 caracteres, una edición más allá de
  ese punto producía el mismo id y `add` descartaba el cambio en silencio. Con
  `progress/` y `featureslist.json` indexados —que cambian en cada feature— eso
  contaminaba la memoria del arnés al ritmo al que se trabaja. Ahora el
  reindexado es por fichero y por huella de contenido: lo que no cambió no se
  re-embebe, lo que cambió se reemplaza y lo que se borró se purga.
- **`hybrid=True` no hacía nada.** Pasaba `includes=` (el kwarg de chroma es
  `include`), petaba, y el `except` caía al mismo query que `hybrid=False`.
  Ahora la búsqueda híbrida existe: BM25 Okapi en stdlib fundido con el ranking
  vectorial vía Reciprocal Rank Fusion. No es adorno — el embedder por defecto
  está entrenado en inglés y esta plantilla se documenta en español, así que
  buena parte de la señal útil es literal.
- **El índice no cubría el código que se despliega.** Solo entraba el paquete
  principal: `api/`, `chat/`, `monitoring/`, `tuning/` y `agents/` quedaban
  fuera, así que preguntar por el drift devolvía los prompts que lo describen y
  nunca `monitoring/monitor.py`. Añadidos a las fuentes.
- **El 70% de los chunks eran docstrings duplicados.** Se indexaba la función
  *y* su docstring por separado; como el docstring es prosa corta y limpia,
  ganaba el coseno a la implementación. Eliminada la duplicación: ahora los
  tipos de sección son `function`/`class`/`module`/`heading` y el `section_type`
  ya no etiqueta las funciones como `docstring`.

Y de paso: cada chunk de código lleva su ruta (y su clase) como cabecera, cada
sección de markdown arrastra las migas de sus ancestros, `_chunk_by_size` ya no
tira la última tanda ni emite números de línea inventados, el HTML de las URLs
se convierte a texto antes de indexar, reindexar una URL la reemplaza en vez de
duplicarla, y `search` acepta `min_score`.

**Embedder multilingüe opcional** (`--extra rag_multilingual` +
`DSKIT_RAG_EMBEDDER=multilingual`): `all-MiniLM-L6-v2` está entrenado en inglés.
Como ambos modelos dan vectores de 384 dimensiones, mezclar índices no daría
error de chroma sino resultados sin sentido — el embedder queda grabado en los
metadatos de la colección y el agente rechaza buscar si detecta el desajuste.

Los tests del RAG pasan de 15 a 50 y cubren lo que antes no se probaba nunca:
creación de la colección, reindexado incremental, purga de huérfanos, techo del
embedder, fusión híbrida y desajuste de embedder.

### El RAG ya se puede medir — y `min_score` se comía justo el híbrido

- **`min_score` borraba los aciertos léxicos.** El `score` que se devolvía era
  la similitud coseno, y un chunk que entraba solo por BM25 no estaba en la
  respuesta vectorial: se quedaba con el `0.0` por defecto. Es decir, el filtro
  de calidad eliminaba exactamente los resultados que el híbrido existe para
  rescatar, y el número que se imprimía al lado (`[0.0 lexico]`) no ordenaba
  nada. Ahora `score` es la fusión RRF —la que de verdad ordena— y `similarity`
  es el coseno, que se calcula también para los candidatos léxicos pidiendo sus
  vectores por id. El filtro se aplica antes de cortar por `top_k`, no después:
  filtrar devolvía menos resultados de los pedidos habiendo candidatos válidos
  esperando.
- **Suite de evaluación de recuperación** (`agents/evals/rag_eval.py` +
  `rag_golden.json`, `make eval-rag`, y suite `rag` en el runner). Mide
  `hit_rate`, `recall@k`, MRR y qué fracción de los aciertos aporta el léxico,
  en modo híbrido y en solo-vector. Primera medición sobre esta plantilla (2.086
  chunks, 212 fuentes, 12 consultas): **híbrido `hit_rate` 0.833 / MRR 0.444
  frente a solo-vector 0.583 / 0.329**. El híbrido se gana el sitio, y ahora eso
  es un dato y no una intuición. El veredicto va por umbral y no por pleno: los
  casos que fallan se ven uno a uno —son el mapa de dónde mejorar— pero lo que
  pone la suite en rojo es caer por debajo de la línea. Exigir 12/12 obligaría a
  escribir un juego de pruebas fácil.
- **`rag status` detecta el índice caducado.** Comparando las huellas de disco
  con las grabadas en los metadatos: buscar sobre un índice viejo no daba error,
  daba la respuesta de ayer, y `make index-rag` es manual.
- **Filtros y contexto en la búsqueda:** `--file_type` (code/doc/prompt/vault/
  harness/url, filtrado en chroma), `--source` por prefijo de ruta,
  `--max_per_source` para que un `top_k` no se lo coma un módulo largo, y
  `--expand N`, que recupera por chunk pequeño y devuelve el vecindario. El
  techo de 1.000 caracteres es el límite del embedder: manda sobre lo que se
  vectoriza, no sobre lo que se responde.
- **El BM25 se persiste** en `.rag-index/bm25.json` con índice invertido. La
  caché en memoria no servía de nada en una CLI de un solo disparo: cada
  búsqueda releía la colección entera y la retokenizaba. Medido: **961 ms con
  el volcado frente a 2.001 ms reconstruyendo**. De paso, puntuar deja de ser
  O(términos × chunks).
- **El RAG entra en el bucle.** Cuando ningún agente alcanza la confianza
  mínima, el orquestador ya no devuelve solo la lista de candidatos
  descartados: busca en el índice y dice dónde está escrita la respuesta. Y
  `harness next` adjunta los antecedentes de `progress/` relacionados con la
  feature que toca — rutas, no texto, que heredar contexto es justo lo que el
  arnés evita.

### La puerta de permisos: el contrato dejó de ser una frase

`contracts.py` decía desde hacía versiones que refactor «siempre con dry_run
primero, el humano aprueba». Nada lo comprobaba, y `RefactorAgent` tenía
`dry_run: bool = False` por defecto. Un contrato que no se ejecuta es
documentación.

- **`agents/permissions.py` + campo `destructive` en los contratos.**
  `BaseAgent.run()` se niega a ejecutar una acción marcada como irreversible
  sin `confirm=True` (`--yes` en la CLI) y devuelve la pregunta en `needs`. El
  intento queda auditado: saber qué quiso hacer un agente y no se le dejó es el
  dato que dice si la puerta estorba o está salvando el repositorio. Cubiertas:
  los commits, tags y ramas de `git`; las cuatro correcciones de `refactor`; y
  la instalación de agentes de terceros.
- **El auto-commit de `GStack` ya no ocurre solo.** Ese camino usa `GitTool`
  directamente y no pasaba por `run()`, así que la puerta no lo habría cubierto:
  ahora requiere `confirm=True` y, sin él, deja los cambios en el árbol y anota
  cada commit omitido en `events.jsonl`. Un pipeline que escribe en el historial
  de git sin que nadie lo pida es exactamente lo que esto viene a impedir.
- **La frontera es deliberada:** la puerta cubre `run()` —CLI, orquestador,
  GStack, `delegate_to`, o sea los automatismos—. Llamar al método directo desde
  Python no pregunta, porque ahí hay una persona escribiendo código a propósito.
- Los prompts generados marcan `⚠️ pide confirmación` en la tabla de acciones,
  que es la línea que el asistente tiene delante al elegir qué ejecutar.

### La frontera entre lo que el modelo pide y lo que se ejecuta

La puerta de permisos protege a los agentes Python, pero el asistente también
usa sus propias herramientas —`Bash`, `Read`, `Write`, MCP— y ahí no llegaba
ningún contrato de este repositorio. Si el modelo decidía `rm -rf`, leer `.env`
o hacer `git push`, nada de dskit lo veía pasar.

- **`agents/policy_guard.py`**, un hook `PreToolUse` que recibe por stdin lo
  que el modelo quiere hacer y decide antes de que ocurra. Bloquea el borrado
  recursivo fuera del proyecto, `sudo`, `git push`, `git reset --hard`,
  descargar-y-ejecutar en un paso, la lectura de `.env`/claves/`~/.ssh/`, las
  escrituras fuera de la raíz y las llamadas MCP que apunten a credenciales.
  Vive en `agents/` y no en `.claude/` para que la política sea una sola y la
  pueda invocar cualquier asistente. Cableado en `.claude/settings.json`, junto
  con una lista `deny` explícita.
  **No es un sandbox** y el módulo lo dice en su docstring: un comando
  suficientemente creativo se salta cualquier lista de patrones. Y ante un JSON
  que no entiende, deja pasar — un guardia roto que bloquea la sesión entera se
  desactiva a los diez minutos, y entonces no protege de nada.
  Los tests prueban las dos direcciones: que bloquea `rm -rf /` **y** que deja
  pasar `rm -rf build/`, porque un falso positivo es un fallo tan real como un
  falso negativo.
- **`agents/redaction.py`**: `secrets_tool` sabía encontrar secretos dentro de
  los ficheros, pero nadie miraba lo que los agentes **devuelven**, que va a dos
  sitios nada inocentes — la ventana del modelo y `audit.jsonl`, que se queda en
  el disco. Ahora `BaseAgent.run` redacta `message`, `warnings` y `needs`, y
  `audit.record` redacta lo que escribe. No toca `data`: ahí hay estructuras que
  otros agentes consumen por clave y reescribirlas a ciegas rompería el
  encadenado sin avisar.

### Contenido no confiable: el RAG ya no mezcla internet con tu repositorio

`rag index_urls` metía HTML descargado en el mismo índice que `AGENTS.md`, y
`search` los devolvía revueltos y con la misma pinta. Un párrafo de una web que
dijera «ignora las instrucciones anteriores» salía como un resultado más.

- Cada chunk guarda su procedencia (`trust`: `repo` o `externo`) y una marca
  `injection_flag` si el texto tiene forma de intento de dar órdenes.
- `rag search` presenta lo externo **en un bloque aparte y delimitado**, con un
  aviso en `warnings` de que son datos citados y no instrucciones.
- La regla queda escrita en `AGENTS.md`: **los datos que consume un agente no
  amplían lo que tiene permitido hacer**. Y no depende de que el modelo se dé
  cuenta: depende de que lo irreversible siga pidiendo confirmación. La lista de
  patrones esquiva lo evidente y nada más — está para que se note y para que un
  test pueda detectar un índice envenenado, no como defensa.

### Servidores MCP configurados desde copier

Nueva pregunta `use_mcp` + `mcp_servers` (filesystem acotado a `data/` y
`reports/`, git en lectura, fetch, sqlite, time). Genera `.mcp.json` para Claude
Code y el bloque `mcp` de `opencode.json` **desde la misma respuesta**, para que
no diverjan. No instala nada —los servidores se descargan solos con npx/uvx—,
pero la configuración generada avisa de las dos cosas que importan: que estás
ejecutando código de terceros con tus permisos, y que lo que devuelve un
servidor MCP es contenido no confiable como cualquier página web.

Deliberadamente **no** se añade un cliente MCP en Python: sería un subsistema
nuevo con dependencias de red en una plantilla cuyo argumento es funcionar
offline y sin dependencias innecesarias. dskit configura al anfitrión; el
anfitrión habla MCP.

### El arnés se muda a `harness/` y el backlog empieza por el rumbo

**Ruptura de layout.** `featureslist.json`, `progress/` y `memory.md` pasan a
vivir bajo `harness/`. Lo primero que veía alguien al abrir un proyecto
generado era el andamiaje de la IA, no su proyecto de datos. Es un directorio
**visible y no oculto** a propósito: el backlog es justo lo que quieres que un
humano abra. Se quedan en la raíz los ficheros que son convención y que las
herramientas buscan ahí (`AGENTS.md`, `CLAUDE.md`, `README.md`, `init.sh`,
`.claude/`, `.opencode/`).

`copier update` trae los ficheros nuevos pero **no borra los viejos**, así que
un proyecto actualizado se quedaría con las dos copias y el agente escribiendo
en una mientras alguien lee la otra. Eso no se avisa: `init.sh` lo detecta y
**para**, con los `git mv` exactos en pantalla. Trabajar sobre un backlog
duplicado es peor que no trabajar.

**Y el backlog ya no empieza por el pipeline.** Las tres primeras features
fijan la dirección antes de que nadie escriba una línea de código:

1. `SCOPE-001` — qué se quiere resolver: la pregunta, la métrica de éxito con
   umbral numérico y el criterio de parada, en `references/00-objetivo.md`.
   Sin un número, ninguna decisión posterior se toma sobre otra cosa que no
   sea intuición.
2. `RESEARCH-001` — qué se sabe ya del tema: papers con el agente `research`,
   resumidos en `references/01-estado-del-arte.md` diciendo qué se toma de
   cada fuente y qué se descarta, y con el rango de resultados que reporta la
   literatura anotado.
3. `EDA-001` — qué dicen los datos: los notebooks `0-0`, `0-1` y `0-2` sobre
   los datos reales, con una respuesta **por escrito** a si esos datos pueden
   contestar la pregunta de `SCOPE-001`. Si no pueden, es ahora cuando hay que
   enterarse.

`DATA-001` depende de `EDA-001`, y `MODEL-001` cierra el círculo: su baseline
se compara con el umbral de `SCOPE-001` y con el rango de `RESEARCH-001`, para
que «el modelo va bien» sea comparable con algo. El orden lo aplica el grafo de
`depends_on`, no una recomendación: el arnés no deja empezar por la cuarta.

### `GStack.to_mermaid()`

Vuelca la stack —y, si se le pasan los resultados, lo ejecutado, lo omitido y
lo que falló— como diagrama Mermaid pegable en `progress/`. Una stack con
`run_if` deja de leerse como una lista en cuanto pasa de tres pasos, y el
resumen de texto enseña el resultado pero no la forma.

### Agente `integration`: tests de integración con servicio real, sin mocks

Nueva opción `use_integration` (por defecto en el perfil `completo`): agente
Python `integration` que levanta servicios reales (p. ej. Postgres vía Docker)
declarados en `tests/compose.integration.yml` durante `pytest tests/integration/`
y los baja siempre al terminar (`finally`). Es la alternativa sin mocks que
propone la filosofía del arnés: un mock da "seguridad falsa".

- **`agents/agents/integration_agent.py`**: `run_integration_tests` (sube
  servicios con `docker compose up -d --wait`, corre la suite, baja en
  `finally`) y `status` (qué servicios hay y su estado).
- **`agents/tools/integration_tool.py`**: envoltorio fino sobre `docker compose`
  para el ciclo de vida de servicios de test (distinto de `docker_tool`, que es
  análisis estático).
- **`tests/compose.integration.yml` + `tests/integration/`**: ejemplo con
  Postgres (`postgres:16-alpine`) y un test que escribe/lee una fila real,
  marcado `@pytest.mark.integration`.
- **La suite normal no requiere Docker**: `pyproject.toml` excluye el marker
  `integration` del `addopts` (`-m 'not integration'`); solo `make integration`
  lo corre. El agente anula ese addopts con `-o addopts=` (nuevo parámetro
  `overrides` en `PytestTool.run`).
- **Gating por extra**: el agente, tool, prompt, test y los ficheros
  `tests/integration/` se excluyen si `use_integration=false`; el contrato en
  `contracts.py` se mantiene (tolerado por `validate_contracts`).
- **Extra `integration`** (`psycopg[binary]`) y target `make integration`.
- Contrato, prompt (+ AUTOGEN a mano), routing benchmarks y fórmulas de conteo
  de agentes actualizadas (AGENTS.md, orquestador.md, lider.md, agents_reference).

---

## [1.13.4] — 2026-07-29

### Consolidación de agentes: 30 → 28

Medido antes de tocar nada: **6 de los 30 agentes no se referenciaban desde
ningún workflow, gateway, gstack ni CLI** — solo desde el catálogo. Y había
clústeres de 6 agentes cubriendo el mismo dominio. Se fusionan los dos casos
donde la duplicación era literal, no parecida.

- **`schedule` → `cicd`.** No fue una fusión sino una eliminación:
  `cicd.validate_cron` ya devolvía todo lo que hacían sus tres acciones
  (validar, traducir a lenguaje natural y calcular próximas ejecuciones), y
  `cicd` ya importaba `ScheduleTool` y ya aliaseaba `cron`/`schedule`. Sus 5
  tests se portan a `test_cicd_agent.py`, que no tenía **ni uno** de cron. Uno
  de ellos hacía `len(result.data) == 3` y pasaba por casualidad: `data` es un
  dict de 3 claves, no 3 ejecuciones.
- **`docsearch` → `doc`**, con `prune` a `knowledge`. `doc.graph_query` y
  `docsearch.search` consultaban el mismo grafo; se conserva la
  implementación de `docsearch`, que **no cachea los fallos** — así un error
  transitorio de graphify deja de servirse desde caché para siempre.
- **`graphify-out/` tenía dos dueños**: `knowledge` lo construía y `docsearch`
  lo podaba. El test de contratos no lo detectaba porque cada uno lo describía
  con una cadena distinta, así que la regla «un recurso, un dueño» se estaba
  esquivando con la redacción. Ahora es de `knowledge`.

### Complejidad: 7 puntos calientes → 2, cero bloques E

- `evals/runner.py::_harness` era el bloque más complejo del sistema (E 37):
  93 líneas encadenando comprobaciones. Partido en cuatro funciones.
- `rag_tool.py::index_project` (E 35) tenía cinco bloques copiados con el mismo
  esqueleto; añadir una fuente significaba copiar el sexto. Ahora es una tabla
  `FUENTES` y un solo bucle.
- `doctor_agent.py::_satisfies_requires_python` (D 25) era una cadena de siete
  `if op == ...`: el mismo dato repetido en forma de código. Tabla de
  operadores.
- `doc.search` (D 28) y `doc.status` (D 21): una función por fuente.

Media global 3.819 → 3.746. Quedan `cli.main` (30) y
`review._deep_scan_file` (30), ambos preexistentes.

### Bug encontrado al refactorizar

`doc.search` escribía `"source": "graphify"` en los resultados del grafo, pero
el formateo comprueba `"source_type"`. **Los resultados de graphify se contaban
en el total pero no se imprimían nunca.**

---

## [1.13.3] — 2026-07-28

### La matriz de CI deja de elegirse a corazonadas

23 variables ramifican el template: 4·10^10 combinaciones teóricas, de las que
CI probaba 20 escogidas a mano. Y de los 14 flags opcionales, los 10 combos de
`smoke` solo ejercitaban **uno** — `use_rag`, y por venir activado por defecto,
no por decisión. `use_api`, `use_optuna`, `use_monitoring`, `use_docker`,
`use_mlflow`, `use_duckdb`, los boosters, SHAP, conformal, calibration y
graphify **no se instalaban ni se ejecutaban jamás**.

- **`.github/scripts/pairwise.py`** — cobertura all-pairs determinista: casi
  todos los fallos de interacción saltan con dos variables, no con siete, así
  que cubrir todos los pares posibles cuesta decenas de casos en vez de miles
  de millones. Sin `random`: misma `copier.yml`, misma matriz. Trae
  `--self-test`, porque una herramienta de test rota en silencio baja la
  cobertura sin poner nada en rojo.
- Las variables se **leen** de `copier.yml` en vez de copiarse. Ese era el
  agujero de fondo: `use_rag` llevaba versiones sin aparecer en ninguna
  combinación.
- **`.github/scripts/gen_smoke_matrix.py`** — reparte los flags opcionales
  entre los combos que ya existían: 0/14 → 14/14 **sin un job más**. Pondera
  el coste de instalación (redes neuronales ya arrastra torch).
- Render + AST: 20 → 194 combos, 6.940 → 67.512 checks, **2073/2073 pares
  (100%)**, en 2m29s.

### Cuatro bugs que la matriz encontró en su primera ejecución

Todos en configuraciones que llevaban desde siempre sin probarse.

- **`graphify` era 100% ininstalable.** El paquete en PyPI se llama
  `graphifyy` (doble y); `graphify` es el nombre del módulo que se importa.
  `uv sync` fallaba con "there are no versions of graphify[gemini]" en
  cualquier proyecto con `graphify_mode != "no"`. Segundo fallo en el mismo
  camino: la tarea post-generación comprobaba `graphify.__version__`, que el
  módulo no expone.
- **El Temperature Scaling nunca ha funcionado.** `torch.nn.functional as F`
  no se importaba y el bloque de calibración lo usa; al estar dentro de un
  `try/except`, fallaba en silencio imprimiendo "Calibración no disponible".
- **La API hacía dos inferencias por petición.** `pred` se asignaba sin usar y
  la respuesta volvía a llamar a `model.predict(X)`. Además `int()` truncaba
  el valor en regresión.
- `import io, base64, matplotlib` en una línea en el chat (E401).

### Correcciones de infraestructura

- `write_data_file.py` acepta JSON: el formato `"key=val key=val"` partía
  `graphify_mode=graphify + obsidian vault` por los espacios, así que esa
  opción era intestable. `use_rag` no estaba en `VALID_KEYS` y no había forma
  de controlarlo desde CI. `dskit_version` estaba clavado en 1.9.0; ahora sale
  de `copier.yml`.
- El JSON de la matriz se pasa por variable de entorno, no interpolado en la
  línea del shell.

---

## [1.13.2] — 2026-07-28

### Arnés (harness engineering)

Los proyectos generados nacen con un arnés: un entorno dentro del propio
repositorio que gobierna cómo trabaja un agente de IA sobre él.

- **`template/init.sh`** — la puerta. Verifica entorno (python/uv/venv), ficheros
  del arnés, esquema de `featureslist.json`, estructura del proyecto y ejecuta
  la suite de tests. Exit `!= 0` significa *no empieces a trabajar*. Modos
  `--quick` (sin tests) y `--json` (consumo por agentes).
- **`template/featureslist.json`** — backlog estructurado con `id`, `title`,
  `description`, `acceptance_criteria`, `status` y `depends_on`. Se siembra con
  features reales del proyecto y varía según `ml_type`, `use_api`, `use_docker`,
  `use_monitoring` y `use_optuna`.
- **`template/progress/`** — memoria fuera de la ventana de contexto:
  `current.md` (feature en curso), `history.md` (append-only de lo cerrado) y
  `<agente>-<FEATURE-ID>.md` (resultado de cada subagente). Evita el "teléfono
  descompuesto" entre agentes.
- **Cuatro agentes markdown** en `template/.opencode/agents/`: `lider`
  (orquesta, no escribe código de producto), `explorer` (investiga en solo
  lectura), `implementer` (una feature con sus tests), `reviewer` (aprueba o
  rechaza ejecutando `init.sh`, y puede endurecer sus propios criterios).
  Registrados en `template/opencode.json`.
- **Protocolo en `template/AGENTS.md`** — punto de entrada de cualquier agente.
  La regla que no se salta: ninguna feature se marca `done` sin `./init.sh` en
  verde, y cada criterio se cierra con la salida real del comando que lo prueba.

### Agente `harness` — la regla del arnés pasa a ser código

El arnés no es una capa aparte: su parte mecánica es un agente Python más.

- Nuevo agente **`harness`** (`agents/agents/harness_agent.py`), dueño único de
  `featureslist.json` y `progress/`. Acciones: `status`, `next`, `start`,
  `finish`, `block`, `record`, `add`, `gate`.
- **`finish` rehúsa cerrar una feature** si `./init.sh` no pasa en verde o si no
  se aporta evidencia (devuelve `needs`). Deja de ser una instrucción que el
  modelo puede ignorar: verificado con evidencia falsa y un test roto, el
  backlog no se tocó.
- Ningún agente edita el backlog ni el progreso a mano; los cuatro agentes
  markdown registran su informe con `harness record`.
- Contrato en `contracts.py`, prompt en `agents/prompts/harness_agent.md`,
  29 tests en `agents/tests/test_harness_agent.py`, benchmarks de routing.

### Jerarquía de agentes

- **`lider` pasa a ser el agente `primary`** de opencode: es el punto de entrada
  del proyecto. El `orquestador` pasa a `subagent` — sigue siendo el gateway a
  los agentes Python, pero ahora es el líder quien le delega.
- `explorer`, `implementer` y `reviewer` declarados como `subagent`.

### Contrato que faltaba y colisiones de routing (bugs preexistentes)

- **`doc` estaba registrado sin contrato** en `contracts.py`, con
  `test_contracts.py` en rojo desde entonces. Añadido.
- **`doc` declaraba palabras genéricas** (`buscar`, `rag`, `vault`, `grafo`,
  `documentacion`, `semantico`…) que le robaban el ruteo a `rag`, `knowledge`,
  `docsearch` y `documentation`, con `test_capability_collisions.py` en rojo.
  Repartidas por dueño: `doc` se queda con las que expresan «búscalo donde
  sea» (`todas las fuentes`, `dónde está documentado`, `qué hace`…), y cada
  fuente concreta recupera las suyas.

### Ruteo: 21/32 → 34/34

Dos defectos de fondo en `BaseAgent.can_handle`, que afectaban a todo el sistema:

- **Comparaba con acentos.** «documentación» no casaba con la palabra clave
  `documentacion`: misma palabra para el usuario, cero puntos para el ruteo. En
  un proyecto en español eso descartaba media consulta típica. Ahora normaliza
  (sin tocar la `ñ`, que no es un acento sino otra letra).
- **Contaba aciertos, no especificidad.** Dos palabras genéricas ganaban a una
  frase larga: en «busca en el grafo de conocimiento», `knowledge` sumaba 2
  (grafo + conocimiento) y `docsearch` solo 1 (busca en el grafo). Ahora cada
  acierto puntúa por las palabras que cubre.
- Vocabulario añadido a `refactor`, `audit`, `supervisor`, `graph`, `plan`,
  `rag`, `mlflow` y `docsearch` para las formas verbales que fallaban
  (`refactoriza`, `audita`, `planea`, `indexa`…).
- Nuevo `agents/tests/test_routing_scoring.py` que protege ambas propiedades.

### Otras integraciones

- `harness_workflow` como workflow skill, registrado en `orquestador.md` y en la
  tabla de workflows de `AGENTS.md`.
- **RAG indexa la memoria del arnés**: `progress/` y `featureslist.json` entran
  en el índice semántico con `file_type: harness`, así que el histórico se
  consulta en lenguaje natural en vez de releyéndolo.
- El agente `doctor` incluye una verificación `harness` en su `checkup`.
- `make check` incluye ahora `harness-check`; el workflow de CI del proyecto
  generado ejecuta la puerta (`./init.sh --quick`) como paso bloqueante.
- `contracts.py` delimita las tres memorias que no se pisan: `progress/`
  (`harness`), `agents/workspace/memory/` (`memory`) y `vault/` (`knowledge`).

### El arnés deja de depender de opencode

Hasta ahora el arnés solo funcionaba si usabas opencode: un proyecto generado
no traía nada para Claude Code —que lee `CLAUDE.md`, no `AGENTS.md`— ni sus
subagentes. Con el asistente más extendido quedándose fuera, medio template no
servía.

- **`template/CLAUDE.md`** — puntero a `AGENTS.md` vía `@AGENTS.md`, más las
  tres reglas que no se saltan. Deliberadamente no duplica nada: dos copias de
  las reglas divergen, y ese es justo el problema que arregla el resto de este
  release.
- **`.claude/agents/*.md` generados**, no escritos a mano. La fuente sigue
  siendo `.opencode/agents/`; `sync_assistants()` los espeja añadiendo el
  frontmatter YAML de Claude Code. Al `explorer` se le declara `tools:` de solo
  lectura — dárselo por escrito al asistente es más fiable que pedírselo en el
  prompt. Gitignorados, y `copier.yml` los genera al crear el proyecto.
- **`.claude/settings.json`** con un hook `SessionEnd` que ejecuta
  `./init.sh --quick`: cerrar la sesión ya no puede dejar el proyecto roto sin
  que nadie se entere. Es la pieza de *hooks* que faltaba del patrón. Incluye
  una allowlist de permisos para los comandos del arnés.
- `make assistants-sync`; `make prompts-check` y CI detectan también el espejo
  desincronizado. 5 tests más en `test_prompts_sync.py`.

### Prompts autosuficientes y a prueba de deriva

Solo 9 de 30 prompts listaban comandos ejecutables: cargar `skill ml_agent`
daba el criterio del agente pero no sus acciones, así que el asistente gastaba
un `describe <agente>` extra antes de poder hacer nada. Y las reglas de
`contracts.py` estaban además reescritas a mano en la prosa de cada prompt —
dos fuentes de verdad que ya habían divergido.

- **`agents/prompts_sync.py`** — cada prompt conserva su prosa escrita a mano
  (el criterio, que es lo valioso) y gana un bloque `AUTOGEN` con su tabla de
  acciones —marcando qué argumentos son obligatorios— y sus límites derivados
  del contrato (`role`, `cannot`, `needs`, `owns`, `collaborates`).
- `make prompts-sync` regenera; `make prompts-check` solo comprueba y sale con
  código 1 si algo se desincronizó. CI lo ejecuta como paso bloqueante, así que
  añadir una acción sin regenerar se caza en el PR.
- `copier.yml` lo ejecuta al generar el proyecto: nace con los prompts al día.
- Idempotente y no destructivo: verificado sobre un proyecto real (30 prompts
  regenerados, segunda pasada sin cambios, prosa intacta). 10 tests en
  `agents/tests/test_prompts_sync.py`, incluido el ciclo de deriva completo.

### El agente `refactor` ya no puede salirse del proyecto (bug grave)

`RefactorAgent._py_files()` aceptaba cualquier `within` y hacía `rglob("*.py")`
sin excluir nada salvo `__pycache__`. Con `--within .` entraba en `.venv/` y
reescribía los ficheros de los paquetes instalados.

El daño no se queda en ese proyecto: **uv instala por hardlink desde su caché
global**, así que reescribir un fichero dentro de `.venv/` corrompe la copia
cacheada en `~/.cache/uv/archive-v0/`, y a partir de ahí *todos* los proyectos
que instalen esa versión reciben el paquete roto. Encontrado en esta máquina:
21 paquetes envenenados (numpy, pandas, matplotlib, statsmodels, joblib,
plotly, chromadb…), con la firma inconfundible de `fix_mutable_defaults`
(`def f(arraysNoneparent_index=[])`). Purgados.

- `FORBIDDEN_DIRS` — `.venv`, `venv`, `site-packages`, `node_modules`, `build`,
  `dist`, `.git`, `.tox`, cachés… se ignoran venga como venga `within`.
- La ruta se resuelve y se comprueba con `is_relative_to(root)`: ni `--within ..`
  ni un symlink que apunte fuera pueden sacar al agente del proyecto.
- `agents/tests/test_refactor_scope.py` — 11 tests, incluido uno que ejecuta el
  fix de verdad y verifica que el fichero plantado en `.venv/` sigue intacto.

### Bugfix del template

- `tests/test_paths.py` llamaba a `paths.make_dirs()`, que no existe (es
  `ensure_dirs()`), y aun renombrando fallaba porque la guarda `_dirs_created`
  la deja en no-op tras el import. **Todo proyecto generado nacía con un test
  en rojo** — y por tanto con la puerta del arnés bloqueada. El CI no lo veía
  porque solo valida sintaxis del render, no ejecuta la suite generada.

### Verificación

- Nueva suite `harness` en `agents/evals/runner.py` (`--harness`): comprueba que
  `init.sh` existe y es ejecutable, que están los ficheros y agentes del arnés,
  que `AGENTS.md` documenta el protocolo y que `featureslist.json` cumple el
  mismo esquema que valida `init.sh`.
- `make init`, `make harness-check` y `make backlog` en el Makefile del template;
  `make opencode-check` verifica también las definiciones del arnés.
- `validate_template.py` valida ahora los `.json` renderizados en las 20
  combinaciones (las condicionales Jinja2 en JSON rompen fácil por comas
  colgantes) y aplica el esquema del backlog. `.vscode/` queda exento por ser
  JSONC.
- `copier.yml` añade la tarea `chmod +x init.sh`.
- Tests nuevos del arnés y del ruteo en `agents/tests/`
  (`test_harness_agent.py`, `test_routing_scoring.py`) y cobertura de la
  indexación de `progress/` en `test_rag_agent.py`.

---

## [1.9.1] — 2026-07-10

### Obsidian Vault + Graphify

- `use_graphify` y `use_obsidian` fusionadas en una sola variable `graphify_mode` con opciones: `no` · `solo graphify` · `graphify + obsidian vault`.
- Cuando se elige `graphify + obsidian vault`, se genera `vault/` con estructura por dominios (00_META..07_REFERENCIAS), plantillas Obsidian en `00_META/templates/`, índice con wikilinks, y configuración `.obsidian/`.
- Las plantillas usan `{% raw %}...{% endraw %}` para evitar colisión entre Jinja2 y `{{title}}`/`{{date}}` de Obsidian.
- `_tasks` valida que graphify se importa correctamente cuando `graphify_mode != "no"`.

### Bugfixes

- `template/.copier-answers.yml` eliminado — pisaba el archivo que Copier genera automáticamente, dejándolo vacío.
- `test_agent.py`: `{{exc}}` escapado con `{% raw %}` para Jinja2.
- `check_copier.py`, `write_data_file.py`, `validate_template.py`: sincronizados con `use_obsidian` y `use_graphify`.
- Dockerfile, docker-compose.yml, .dockerignore movidos de raíz a `template/` (Copier nunca los procesaba por `_subdirectory: template`).

---

## [1.9.0] — 2026-07-06

### Sistema de agentes y release automático

- Se añadió y documentó la carpeta `agents/` como capa de agentes especializados para git, documentación, CI/CD, pruebas, dependencias, API, secretos, notebooks e instalación de agentes externos.
- `GitAgent` ahora puede coordinar `update_changelog`, `bump_version`, `commit_with_changelog` y `tag_release` en un flujo único de release.
- `BaseAgent` y `Orchestrator` mejoraron el ruteo determinista: además de escoger agente, ahora también resuelven la acción con alias y validación de argumentos.
- Se documentó el nuevo workspace por agente y la colaboración entre agentes para evitar ciclos de import y duplicación de lógica.

---

## [1.8.2] — 2026-05-22

### Corrección de bugs — auditoría completa

Auditoría automática con Jinja2 StrictUndefined sobre **17 combinaciones × 59 archivos**
(1 003 checks de renderizado + AST) y suite semántica con 80+ aserciones de contenido.
Resultado final: **0 bugs** en renderizado, AST y semántica.

---

#### `main.py` — 3 bugs críticos (`redes_neuronales + regresion`)

- **`output_dim` incorrecto**: `len(y_train.unique())` devuelve el número de clases únicos,
  que en regresión es el número de valores distintos del target continuo, no `1`.
  Corregido con bloque `{% if task_type == 'regresion' %}` que asigna `output_dim = 1`.
- **`evaluate_models` llamada con `num_classes=output_dim`** en regresión: la firma de
  `evaluate_models` para regresión no acepta ese parámetro y lanzaba `TypeError`.
  Corregido condicionando la llamada con `{% if task_type == 'regresion' %}`.
- **`best["Accuracy"]` y `best["F1"]` en el print final** de regresión: esas columnas no
  existen en el DataFrame de regresión (que tiene `RMSE`, `MAE`, `R2`).
  Corregido mostrando `RMSE`/`MAE`/`R²` para regresión y `Accuracy`/`F1` para clasificación.

---

#### `tuning/tune_model.py` — 5 bugs críticos (NN)

- **`_OBJECTIVES = {"{{ nn_model }}": None}`**: el sentinel `None` hacía que `tune_models()`
  saltara el objetivo real con `if objective_fn is None: continue`. Nunca se ejecutaba
  ningún trial para redes neuronales. Corregido a `_objective_nn`.
- **Optimizador `AdamW` hardcodeado**: ignoraba `optimizer_type`. Para SGD, RMSProp y Adagrad
  se generaba código con `AdamW` en lugar del optimizador elegido. Corregido con el mismo
  bloque `{% if optimizer_type %}` que usa `train_model.py`.
- **Loss `CrossEntropyLoss` hardcodeada**: ignoraba `nn_loss_fn`. Para MSELoss, L1Loss y
  BCEWithLogitsLoss se generaba `CrossEntropyLoss`. Corregido con bloque `{% if nn_loss_fn %}`.
- **`dtype=torch.long` para targets en regresión**: `MSELoss` y `L1Loss` requieren
  `torch.float32`. Los `TensorDataset` y `val_y` creaban tensores `long` en ambos casos,
  causando `RuntimeError` en el primer backward. Corregido con bloque `{% if task_type %}`.
- **`criterion(model(Xb), yb)` sin `.squeeze()`** en regresión: la salida del modelo tiene
  shape `(batch, 1)` y el target `(batch,)`. Sin `.squeeze()`, MSELoss/L1Loss lanzan
  `ValueError: shape mismatch`. Corregido a `criterion(model(Xb).squeeze(), yb)`.

---

#### `{{ project_slug }}/features/build_features.py` — 2 bugs críticos

- **`process_input()` ausente en bloque `no_supervisado`**: la API, el chat y `try_model()`
  llaman a `process_input()` para cualquier `ml_type`. Al no existir, el import fallaba con
  `ImportError`. Añadida implementación completa con `scaler.joblib` y `encoders.joblib`.
- **`process_input()` ausente en bloque `hibrido`**: mismo problema. Añadida implementación
  que además detecta y aplica automáticamente la transformación dimensional guardada en
  `artifacts/` (PCA, UMAP, KMeans-features o IsolationForest).

---

#### `{{ project_slug }}/models/train_model.py` — 2 bugs

- **`joblib` e `ARTIFACTS_DIR` no importados** en el bloque NN: el bloque `supervisado`
  los importaba pero el bloque `redes_neuronales` no. Cualquier llamada a `joblib.dump`
  lanzaba `NameError`. Añadidos al bloque de imports NN.
- **`output_dim.joblib` no se guardaba tras entrenar**: la API infería `output_dim=2` por
  defecto al no encontrar el artefacto. Roto para regresión (`output_dim` debería ser `1`)
  y para clasificación multiclase (3+ clases). Añadida llamada `joblib.dump(output_dim, ...)`
  al final de `train_models()`.

---

#### `tests/conftest.py` — 3 bugs

- **`tuning.tune_model` no parcheado** con `monkeypatch` cuando `use_optuna=True`:
  los tests de tuning usaban rutas reales del sistema de ficheros.
- **`monitoring.monitor` no parcheado** cuando `use_monitoring=True`: ídem.
- **`api.main` no parcheado** cuando `use_api=True`: ídem.
  Los tres módulos añadidos a `candidate_modules` condicionados con `{% if %}`.

---

#### `tests/test_predict_model.py` — 5 bugs (NN)

- **`"MLP"` hardcodeado** en todos los asserts: fallaba para CNN1D, LSTM, GRU, Transformer.
  Corregido a `MODEL_NAME` (importado de `train_model`).
- **Sin tests de regresión**: solo había tests de clasificación. Para `task_type=regresion`
  el bloque NN quedaba con 0 tests de evaluación. Añadidos 5 tests de regresión
  (`RMSE`, `MAE`, `R2`, scatter PNG, predicciones float).
- **`train_models()` sin `val_split=`**: la función requiere el parámetro; sin él usaba
  el default de `0.1` que podía reducir los datos de entrenamiento por debajo del
  `batch_size`, causando un crash en el `DataLoader`.
- **Columna `"Accuracy"` en tests de regresión**: `evaluate_models` regresión devuelve
  `RMSE`/`MAE`/`R2`. El assert `"Accuracy" in df_res.columns` siempre fallaba.
- **`num_classes=` en `evaluate_models()` de regresión**: la firma de regresión no acepta
  ese argumento. Eliminado del bloque `{% else %}` (regresión).

---

#### `tests/test_train_model.py` — 3 correcciones (NN)

- **`train_models()` sin `val_split=0.2`** en los 3 calls NN: añadido en todos.

---

#### `tests/test_tuning.py` — 2 bugs (NN regresión)

- **`output_dim=int(y_train.nunique())`** en los tests NN de regresión: devuelve el número
  de valores únicos del target continuo, no `1`. Corregido a `output_dim=1` condicionado
  con `{% if task_type == 'regresion' %}`.

---

#### `pyproject.toml` — 1 bug

- **Extra `monitoring` sin `evidently`**: al seleccionar `use_monitoring=True`, el entorno
  se generaba sin la dependencia principal. Añadido `evidently` y `scipy` al extra.

---

#### `Makefile` — 1 bug

- **Target `lock:` ausente**: no había forma de regenerar `uv.lock` tras cambiar
  dependencias en `pyproject.toml`. Añadido `lock: uv lock` y declarado en `.PHONY`.

---

#### `README.md` — reescrito

- Badge de versión actualizado a `1.8.2`
- Tabla de variables completa con `optimizer_type`, `nn_loss_fn`, `cluster_model`,
  `use_catboost`, `use_docker`, `use_shap`
- Tabla de módulos opcionales con flag → descripción → make target
- Sección "Notas por tipo de ML" con detalles de `output_dim.joblib`, `process_input()`,
  y comportamiento del optimizador/pérdida en NN
- Makefile — referencia completa con todos los targets incluyendo `make lock`
- Estructura de directorios actualizada con `models/artifacts/`

---

## [1.8.1] — 2026-05-20

### Añadido

#### Notebooks educativos adaptativos (condicionados por `ml_type == 'redes_neuronales'`)

- **`0-0-DescargaDatos.ipynb`**: análisis de rangos de features pre-normalización y
  checklist automático de preparación NN (muestras, nulos, balance, varianza).
- **`0-1-ProcesamientoDatos.ipynb`**: forma exacta del tensor de entrada por arquitectura
  y verificación de dtypes/balance de clases.
- **`0-2-Ejecucion.ipynb`**: demo interactiva de autograd, curva de convergencia del
  optimizador elegido, comparativa de funciones de pérdida y evaluación con TorchMetrics.

#### Nuevas opciones de configuración NN

- **`optimizer_type`**: `AdamW` (default) · `Adam` · `SGD` · `RMSProp` · `Adagrad`
- **`nn_loss_fn`**: `Auto` · `CrossEntropyLoss` · `MSELoss` · `L1Loss` · `BCEWithLogitsLoss`

#### TorchMetrics integrado en `train_model.py`

- Clasificación: `MulticlassAccuracy`, `F1Score`, `Precision`, `Recall` (macro)
- Regresión: `MAE`, `RMSE`, `R²`
- Métricas train/val logueadas en TensorBoard (`Train/*`, `Val/*`)
- Degradación silenciosa si `torchmetrics` no está instalado

#### `predict_model.py` NN regresión

- `evaluate_models` con rama completa para regresión: RMSE, MAE, MAPE, R²
- `_plot_regression_scatter` — scatter predicho vs real con línea y=x
- `_plot_residuals` — histograma de residuos + Q-Q plot
- TorchMetrics como fuente primaria, sklearn como fallback (`_HAS_TM`)

#### Extra `monitoring` en `pyproject.toml`

- Añadidos `evidently` y `scipy` cuando `use_monitoring=True`
- Verificación en `_tasks` con `import evidently`

#### Otros

- `make lock` — target `uv lock` en Makefile
- `api/__init__.py` — el directorio `api/` ya es paquete Python importable
- `docs/source/conf.py` — `sys.path` corregido de `../../src/` a `../..`
- `make test` — añadido `--cov={{ project_slug }}` con report `term-missing` y HTML
