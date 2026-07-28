# Changelog

Todos los cambios relevantes de esta plantilla se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).

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
