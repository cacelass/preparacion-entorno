# DSKIT

![version](https://img.shields.io/badge/dskit-1.15.0-blue)
![CI](https://github.com/cacelass/dskit/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)
![uv](https://img.shields.io/badge/gestor-uv-green)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

**Template profesional para Data Science y AI Engineering**

Plantilla [copier](https://copier.readthedocs.io) para iniciar proyectos de ML de forma organizada, reproducible y lista para producción. Construida sobre `uv`, con una arquitectura modular que cubre el flujo completo desde la ingesta de datos hasta el modelo evaluado, exportado y servido como API o interfaz de chat.

Y con un **arnés de IA** dentro del propio repositorio: un entorno que gobierna cómo trabajan los agentes sobre el proyecto, con puerta de entrada, backlog verificable y una definición de «hecho» que se aplica en código, no pidiéndosela al modelo.

---

## Índice

- [DSKIT](#dskit)
  - [Índice](#índice)
  - [Arnés de IA](#arnés-de-ia)
    - [Los agentes](#los-agentes)
    - [Funciona con cualquier asistente](#funciona-con-cualquier-asistente)
  - [Características](#características)
    - [Tipos de ML y arquitecturas](#tipos-de-ml-y-arquitecturas)
    - [Módulos opcionales](#módulos-opcionales)
    - [Pipeline de calidad integrado](#pipeline-de-calidad-integrado)
  - [Requisitos previos](#requisitos-previos)
  - [Instalación rápida](#instalación-rápida)
  - [Actualizar un proyecto existente](#actualizar-un-proyecto-existente)
  - [Variables de configuración](#variables-de-configuración)
    - [Validaciones automáticas](#validaciones-automáticas)
  - [Uso](#uso)
  - [Estructura generada](#estructura-generada)
  - [Makefile — referencia completa](#makefile--referencia-completa)
  - [Notas por tipo de ML](#notas-por-tipo-de-ml)
    - [`supervisado`](#supervisado)
    - [`redes_neuronales`](#redes_neuronales)
    - [`no_supervisado`](#no_supervisado)
    - [`hibrido`](#hibrido)
  - [Changelog](#changelog)
  - [License](#license)

---

## Arnés de IA

Un modelo de IA genera código mucho más rápido de lo que un humano lo revisa. El arnés (*harness*) es el entorno que pone las riendas: vive dentro del repositorio generado, así que viaja con el proyecto y lo comparte todo el equipo.

```
./init.sh → harness/progress/ → harness/featureslist.json → implementar → revisar → done
    │
    └── si falla: el agente PARA. No se trabaja sobre un proyecto roto.
```

| Pieza | Qué hace |
|---|---|
| `AGENTS.md` | Punto de entrada. Lo primero que lee cualquier agente |
| `init.sh` | La puerta: entorno, ficheros del arnés, backlog y suite de tests |
| `harness/featureslist.json` | Backlog con criterios de aceptación **verificables** |
| `harness/progress/` | Memoria fuera de la ventana de contexto: tarea actual e histórico |
| `.opencode/agents/` | Los cuatro agentes del arnés |

**La regla que no se salta:** ninguna feature se marca `done` sin que `./init.sh` pase en verde y sin evidencia real del comando que lo demuestra. No es una instrucción en un prompt — la aplica `harness finish` en Python, así que no se puede rodear pidiéndoselo amablemente al modelo.

```bash
make init            # ¿se puede trabajar?
make backlog         # estado de las features
make harness-check   # solo estructura, sin tests
```

### Los agentes

Dos capas que no hacen lo mismo, y esa separación es el diseño:

- **Razonan** — `lider` (dirige el ciclo), `explorer` (investiga en solo lectura), `implementer` (escribe código y tests), `reviewer` (aprueba o rechaza). Markdown en `.opencode/agents/`.
- **Ejecutan** — 30 agentes Python en `agents/agents/` (`git`, `test`, `review`, `docker`, `data`, `ml`, `plan`, `doctor`, `harness`…). Acciones deterministas, sin ambigüedad.

El arnés **no sustituye** a los agentes Python: delega en ellos todo lo repetible. Cada uno tiene un contrato en `agents/contracts.py` que declara qué puede, qué no y qué recursos posee en exclusiva — un recurso, un dueño, validado por test.

```bash
uv run python -m agents --json ask "revisa el Dockerfile"   # ruteo automático
uv run python -m agents --json run harness next             # ¿qué toca?
uv run python -m agents.evals.runner                        # harness + smoke + routing + contracts
```

Los prompts de los agentes se **derivan del código**: cada uno conserva su criterio escrito a mano y lleva un bloque generado con sus acciones y sus límites, sacados de `actions()` y `contracts.py`. `make prompts-check` (y CI) falla si se desincronizan, para que no haya dos fuentes de verdad.

### Funciona con cualquier asistente

`AGENTS.md` es la fuente única de reglas; `CLAUDE.md` solo apunta a él sin duplicar nada. Los cuatro agentes del arnés se escriben una vez en `.opencode/agents/` y `make assistants-sync` los espeja a `.claude/agents/` con el frontmatter que espera Claude Code. El proyecto trae además un hook `SessionEnd` que ejecuta la puerta al cerrar la sesión, para no dejar el repositorio roto sin que nadie se entere.

---

## Características

### Perfil de proyecto
`proyecto_perfil` (default `estandar`) decide qué extras y qué agentes se
instalan de una vez. En `minimo`/`estandar`/`completo` **no se pregunta** por
cada extra — los defaults se derivan del perfil; solo `manual` pregunta uno a
uno (el flujo detallado clásico).

| Perfil | Agentes | Qué incluye |
|--------|---------|-------------|
| `minimo` | ~19 (núcleo) | Harness + agentes de calidad (git, test, review, data, ml...) |
| `estandar` | núcleo + rag + mutation | Arnés de calidad: RAG + spec-driven |
| `completo` | todos | Todos los extras + periféricos (supervisor, research, audit, installer) |
| `manual` | según lo elegido | Cada opción se pregunta una a una |

En `minimo`/`estandar` el proyecto **no instala dependencias al generarse** —
ejecuta `make setup` cuando quieras tener el entorno listo. En `completo`/`manual`
el `uv sync` automático se mantiene.

### Tipos de ML y arquitecturas
- **4 tipos de ML**: `supervisado`, `no_supervisado`, `redes_neuronales`, `hibrido`
- **2 tipos de tarea** (`task_type`): `clasificacion` o `regresion`
- **6 arquitecturas NN** (`nn_model`): MLP · CNN1D · LSTM · GRU · Transformer · ResNet
- **Optimizador configurable** (`optimizer_type`): AdamW · Adam · SGD · RMSProp · Adagrad
- **Función de pérdida configurable** (`nn_loss_fn`): Auto · CrossEntropyLoss · MSELoss · L1Loss · BCEWithLogitsLoss
- **Selector de modelo** (`model_type`): todos · RandomForest · ExtraTrees · GradientBoosting · AdaBoost · XGBoost · LightGBM · CatBoost · LogisticRegression · KNN · DecisionTree · SVM

### Módulos opcionales
| Flag | Descripción | Make target |
|---|---|---|
| `use_api` | API REST FastAPI — `/health`, `/info`, `/predict` | `make serve` |
| `use_optuna` | HPO automática por modelo con Optuna | `make tune` |
| `use_monitoring` | Drift KS/chi² + performance vs baseline (Evidently) | `make monitor` |
| `use_mlflow` | Tracking de experimentos, artifacts y Model Registry | `make mlflow` |
| `use_duckdb` | Carga CSV/Parquet/JSON con SQL directo | `make query` |
| `use_docker` | Docker + interfaz de chat Gradio | `make docker-run` |
| `use_rag` | RAG local (ChromaDB + BM25): indexa el código, los prompts, los docs y la memoria del arnés, y busca fundiendo vector y léxico. Sin API key, offline | `make index-rag` |
| `use_sdd` | Spec-driven (Robert C. Martin): contrato Gherkin con puerta humana antes de codear, agente de mutation testing y métrica CRAP | `make mutation` · `make crap` |
| `use_conformal` | Conformal Prediction — sets/intervalos con garantía de cobertura, *distribution-free* | automático |
| `use_calibration` | Temperature Scaling — calibra la confianza del modelo *(redes_neuronales)* | automático |
| `graphify_mode` | `no` · `solo graphify` · `graphify + obsidian vault` | automático |
| papers + guía modelos | vault/07_REFERENCIAS/ (notas por modelo) + vault/01_PROYECTO/guiia_modelos.md | automático |
| `use_shap` | SHAP values — importancia de features | automático |
| `use_xgboost` | XGBoost (supervisado/híbrido) | automático |
| `use_lightgbm` | LightGBM (supervisado/híbrido) | automático |
| `use_catboost` | CatBoost (supervisado/híbrido) | automático |

### Pipeline de calidad integrado
- **TorchMetrics** en bucle de entrenamiento y evaluación NN (Accuracy/F1/Precision/Recall para clasificación, MAE/RMSE/R² para regresión)
- **Early stopping** y **validation split** configurables en redes neuronales
- **TensorBoard** integrado en redes neuronales (`make tb`)
- **19–29 agentes especializados** en `agents/` (según el perfil) para changelog, releases, CI/CD, tests, dependencias, API, datos, modelos y documentación — con contratos que impiden que dos agentes escriban el mismo recurso y gating por perfil (los ligados a un extra solo se instalan si el extra está activo)
- **`make check`** — lint + typecheck + test + arnés, la batería completa
- **`make smoke`** — tests de humo que verifican que el pipeline arranca sin errores
- **`make profile`** — profiling con cProfile + snakeviz
- **`make lock`** — regenera `uv.lock` tras cambios en dependencias
- **CI que ejecuta la puerta del arnés** y comprueba que los prompts no se han desincronizado del código
- **PRD vivo**: `docs/prd.md` se regenera desde el backlog al cerrar cada feature (`documentation update_prd`)
- `uv sync` automático tras generar el proyecto *(solo en perfiles `completo`/`manual`; en `minimo`/`estandar` ejecuta `make setup`)*

---

## Requisitos previos

```bash
pip install copier uv
```

Python >= 3.10 requerido.

---

## Instalación rápida

```bash
copier copy --trust gh:cacelass/dskit nombre_proyecto
```

O desde copia local:

```bash
copier copy --trust ./dskit nombre_proyecto
```

Copier ejecuta `uv sync` automáticamente. Si falla, hazlo manualmente:

```bash
cd nombre_proyecto
uv sync --extra dev --extra <ml_type>
```

---

## Actualizar un proyecto existente

Un proyecto generado no es una copia muerta: guarda las respuestas en
`.copier-answers.yml` y puede traerse las mejoras de versiones posteriores de
la plantilla sin volver a empezar.

```bash
cd tu_proyecto
git add -A && git commit -m "chore: antes de actualizar dskit"   # imprescindible
copier update --trust
```

`copier update` calcula qué cambió en la plantilla entre tu versión y la última,
y aplica **solo esa diferencia** sobre tu proyecto, respetando tus
modificaciones. Donde tus cambios choquen con los de la plantilla te dejará
marcadores de conflicto (`<<<<<<<`), igual que un merge de git; búscalos antes
de dar la actualización por buena:

```bash
grep -rn '<<<<<<<' --exclude-dir=.git .
```

Por eso el commit previo no es opcional: es lo único que te deja ver el
resultado con `git diff` y descartarlo con `git checkout .` si no te convence.

**Cambiar de opciones.** `copier update` reutiliza tus respuestas anteriores.
Para revisarlas —o activar un módulo que dejaste fuera— usa
`copier update --trust --defaults=false`, o edita `.copier-answers.yml` antes de
actualizar. Nota que activar un extra añade sus ficheros, pero desactivarlo no
siempre borra los que ya existen: revisa el diff.

**Después de actualizar**, pasa la puerta del arnés antes de seguir trabajando:

```bash
uv sync --extra dev --extra <ml_type>
./init.sh
```

**Saltos de varias versiones.** No hay migraciones automáticas todavía, así que
si vienes de una versión bastante anterior, lee el [CHANGELOG](CHANGELOG.md)
entre tu versión y la actual: los cambios que requieren acción manual se anotan
ahí. Actualizar versión a versión da conflictos más pequeños y legibles que un
salto largo de golpe.

---

## Variables de configuración

Copier muestra solo las preguntas relevantes según las respuestas anteriores.

| Variable | Valores | Condición | Descripción |
|---|---|---|---|
| `project_name` | texto | siempre | Nombre del proyecto |
| `project_slug` | `[a-z0-9_]` | siempre | Nombre del paquete Python (auto desde project_name) |
| `project_author_name` | texto | siempre | Nombre del autor |
| `project_author_email` | email | siempre | Email (validado) |
| `project_description` | texto | siempre | Descripción breve |
| `proyecto_perfil` | `minimo` · `estandar` · `completo` · `manual` | siempre | Perfil: fija los defaults de los extras y no pregunta por ellos (default `estandar`) |
| `ml_type` | `supervisado` · `no_supervisado` · `redes_neuronales` · `hibrido` | siempre | Determina el código generado |
| `task_type` | `clasificacion` · `regresion` | supervisado, redes_neuronales, hibrido | Tipo de tarea |
| `model_type` | `todos` · `RandomForest` · `ExtraTrees` · `GradientBoosting` · `AdaBoost` · `XGBoost` · `LightGBM` · `CatBoost` · `LogisticRegression` · `KNN` · `DecisionTree` · `SVM` | supervisado, hibrido | Modelo a entrenar |
| `cluster_model` | `todos` · `KMeans` · `AgglomerativeClustering` · `DBSCAN` · `GaussianMixture` · `SpectralClustering` · `Birch` | no_supervisado | Algoritmo de clustering |
| `nn_model` | `MLP` · `CNN1D` · `LSTM` · `GRU` · `Transformer` · `ResNet` | redes_neuronales | Arquitectura |
| `optimizer_type` | `AdamW` · `Adam` · `SGD` · `RMSProp` · `Adagrad` | redes_neuronales | Optimizador PyTorch |
| `nn_loss_fn` | `Auto` · `CrossEntropyLoss` · `MSELoss` · `L1Loss` · `BCEWithLogitsLoss` | redes_neuronales | Función de pérdida |
| `use_xgboost` | true/false | manual + supervisado, hibrido | Añade XGBoost |
| `use_lightgbm` | true/false | manual + supervisado, hibrido | Añade LightGBM |
| `use_catboost` | true/false | manual + supervisado, hibrido | Añade CatBoost |
| `use_shap` | true/false | manual + supervisado, hibrido | SHAP values |
| `graphify_mode` | `no` · `solo graphify` · `graphify + obsidian vault` | manual | Grafo de conocimiento + vault Obsidian opcional |
| `use_mlflow` | true/false | manual | MLflow tracking |
| `use_monitoring` | true/false | manual | Drift + performance monitoring |
| `use_optuna` | true/false | manual | HPO con Optuna |
| `use_duckdb` | true/false | manual | DuckDB SQL sobre ficheros |
| `use_api` | true/false | manual | API REST FastAPI |
| `use_docker` | true/false | manual | Docker + chat Gradio |
| `use_rag` | true/false | manual | RAG local híbrido (ChromaDB + BM25) — activo en `estandar`/`completo` |
| `use_sdd` | true/false | manual | Spec-driven: contrato Gherkin + mutation testing + CRAP — activo en `estandar`/`completo` |
| `use_mcp` | true/false | manual | Servidores MCP para el asistente (filesystem, git, fetch...) |
| `use_conformal` | true/false | manual + supervisado, hibrido, redes_neuronales | Conformal Prediction |
| `use_calibration` | true/false | manual + redes_neuronales | Temperature Scaling |
| `project_open_source_license` | `No license file` · `MIT` · `BSD-3-Clause` · `Apache-2.0` | siempre | Licencia del proyecto generado |
| `python_version` | `3.10`–`3.13` | siempre | Versión de Python |
| `project_version` | texto | siempre | Versión inicial del proyecto |

### Validaciones automáticas
- `project_slug`: solo `[a-z0-9_]` empezando por letra — los guiones se transforman a `_`
- `project_author_email`: formato válido requerido

---

## Uso

```bash
make help        # ver todos los comandos disponibles
make init        # la puerta del arnés: ¿se puede trabajar?
make backlog     # estado de harness/featureslist.json
make pipeline    # pipeline completo: data → features → train → predict
make run         # ejecuta main.py
make data        # solo ingesta de datos
make features    # solo preprocesado
make train       # solo entrenamiento
make predict     # solo evaluación
make check       # lint + typecheck + test + arnés
make smoke       # tests de humo rápidos
make test        # suite completa de tests con cobertura
make lint        # ruff check
make format      # ruff format
make profile     # cProfile → reports/profile.prof
make lock        # regenera uv.lock
make tb          # TensorBoard localhost:6006   (redes_neuronales)
make mlflow      # MLflow UI localhost:5000     (use_mlflow=true)
make monitor     # drift + performance report  (use_monitoring=true)
make tune        # HPO con Optuna              (use_optuna=true)
make serve       # API REST localhost:8000     (use_api=true)
make query       # Shell DuckDB interactivo   (use_duckdb=true)
make index-rag   # indexa el proyecto en ChromaDB (use_rag=true)
make docker-run  # construye la imagen y lanza el chat (use_docker=true)
```

---

## Estructura generada

```
nombre_proyecto/
├── <project_slug>/
│   ├── data/
│   │   └── make_dataset.py       ← carga pandas / DuckDB (si use_duckdb)
│   ├── features/
│   │   └── build_features.py     ← preprocesado + process_input() para inferencia
│   ├── models/
│   │   ├── train_model.py        ← adaptado a ml_type, task_type, nn_model,
│   │   │                            optimizer_type, nn_loss_fn
│   │   └── predict_model.py      ← evaluate_models() + TorchMetrics + figuras
│   ├── utils/paths.py
│   └── visualization/visualize.py
├── api/                          ← (use_api=true)
│   ├── main.py                   ← FastAPI: /health /info /predict + lifespan
│   └── schemas.py                ← Pydantic V2: PredictRequest, PredictResponse
├── tuning/                       ← (use_optuna=true)
│   └── tune_model.py             ← _objective_* por modelo + tune_models()
├── monitoring/                   ← (use_monitoring=true)
│   └── monitor.py                ← check_drift, check_performance, run_monitoring
├── chat/                         ← (use_docker=true)
│   └── app.py                    ← interfaz Gradio conectada al modelo
├── vault/                        ← (graphify_mode = "graphify + obsidian vault")
│   ├── .obsidian/                ← configuración del vault
│   ├── 00_META/                  ← templates + índice
│   ├── 01_PROYECTO/
│   ├── 02_DATOS/
│   ├── 03_MODELOS/
│   ├── 04_VISUALIZACIONES/
│   ├── 05_AGENTES/
│   ├── 06_OBSERVACIONES/
│   └── 07_REFERENCIAS/
├── AGENTS.md                     ← protocolo del arnés: fuente única de reglas
├── CLAUDE.md                     ← puntero a AGENTS.md para Claude Code
├── init.sh                       ← la puerta: ¿se puede trabajar?
├── harness/                      ← TODO el estado del arnés, fuera de la raíz
│   ├── featureslist.json         ← backlog con criterios de aceptación
│   ├── memory.md                 ← preferencias que persisten entre sesiones
│   └── progress/                 ← memoria fuera de la ventana de contexto
│       ├── current.md            ← feature en curso
│       ├── history.md            ← append-only de lo cerrado
│       └── <agente>-<ID>.md      ← informe de cada subagente
├── .opencode/agents/             ← lider · explorer · implementer · reviewer
├── .claude/                      ← settings.json (hook SessionEnd) + agents/ espejados
├── agents/                       ← agentes especializados, docs y utilidades de release
│   ├── README.md                 ← guía completa del sistema de agentes
│   ├── agents/                   ← los 30 agentes: git, test, harness, data, ml...
│   ├── contracts.py              ← qué puede y qué no cada agente (un recurso, un dueño)
│   ├── tools/                    ← utilidades reutilizables por agente
│   ├── prompts/                  ← fichas por agente (bloque autogenerado + criterio)
│   ├── prompts_sync.py           ← mantiene prompts y subagentes al día
│   ├── evals/runner.py           ← harness + smoke + routing + contracts
│   └── workspace/                ← espacio de trabajo por agente
├── data/{raw,interim,processed,external}/
├── models/                       ← pesos .pt / .joblib + artifacts/
│   └── artifacts/                ← scaler.joblib, encoders.joblib, output_dim.joblib…
├── notebooks/
├── reports/
│   ├── figures/
│   ├── monitoring/
│   └── resultados_*.csv
├── tests/
│   ├── conftest.py               ← patch_paths fixture (parchea todas las rutas)
│   ├── test_train_model.py
│   ├── test_predict_model.py
│   ├── test_build_features.py
│   ├── test_make_dataset.py
│   ├── test_api.py               ← (use_api=true)
│   ├── test_tuning.py            ← (use_optuna=true)
│   └── test_monitoring.py        ← (use_monitoring=true)
├── .copier-answers.yml
├── Makefile
├── pyproject.toml
└── main.py
```

---

## Makefile — referencia completa

| Target | Descripción |
|---|---|
| `make pipeline` | Pipeline completo: data → features → train → predict |
| `make run` | Ejecuta `main.py` |
| `make data` | Ingesta de datos |
| `make features` | Preprocesado |
| `make train` | Entrenamiento |
| `make predict` | Evaluación + figuras + CSV |
| `make check` | lint + typecheck + test + arnés |
| `make test` | pytest con cobertura (`--cov=<slug>`) |
| `make smoke` | Solo `@pytest.mark.smoke` |
| `make lint` | `ruff check` |
| `make format` | `ruff format` |
| `make typecheck` | `mypy --strict` |
| `make security` | bandit + pip-audit |
| `make audit` | radon cc + auditoría del equipo de agentes |
| `make profile` | cProfile → `reports/profile.prof` |
| `make lock` | `uv lock` — regenera lockfile |
| `make docs` | Sphinx autodoc |
| **Arnés** | |
| `make init` | Ejecuta `./init.sh` — la puerta |
| `make harness-check` | Solo estructura del arnés, sin tests |
| `make backlog` | Estado de `harness/featureslist.json` |
| **Agentes** | |
| `make agents-list` | Lista los 30 agentes |
| `make agents-doctor` | Diagnóstico integral |
| `make agents-eval` | harness + smoke + routing + contracts |
| `make prompts-sync` | Regenera prompts desde el código y los contratos |
| `make prompts-check` | Falla si un prompt se desincronizó |
| `make assistants-sync` | Espeja los subagentes a `.claude/agents/` |
| `make skills` | Instala los prompts como skills en `.opencode/skills/` |
| **Opcionales** | |
| `make tb` | TensorBoard :6006 *(redes_neuronales)* |
| `make mlflow` | MLflow UI :5000 *(use_mlflow)* |
| `make monitor` | Drift + performance report *(use_monitoring)* |
| `make tune` | HPO Optuna *(use_optuna)* |
| `make serve` | API REST :8000 *(use_api)* |
| `make query` | Shell DuckDB *(use_duckdb)* |
| `make index-rag` | Indexa el proyecto en ChromaDB, incremental *(use_rag)* |
| `make index-rag-rebuild` | Reconstruye el índice desde cero *(use_rag)* |
| `make docker-run` | Construye la imagen y lanza el chat *(use_docker)* |
| `make clean-all` | Limpia cachés, modelos y figuras |
| `make info` | Muestra versiones del entorno |

---

## Notas por tipo de ML

### `supervisado`
- `model_type=todos` entrena y evalúa todos los modelos disponibles en paralelo y devuelve un ranking por F1/RMSE
- SHAP values (`use_shap=True`) se calculan automáticamente sobre el mejor modelo tras la evaluación
- Con `use_optuna=True`, cada modelo tiene su propia función objetivo (`_objective_rf`, `_objective_xgb`, etc.) y los mejores parámetros se guardan en `artifacts/`
- El `DECISION_THRESHOLD` (umbral de clasificación binaria) se puede ajustar manualmente o calcular automáticamente con `find_optimal_threshold()` en `predict_model.py`
- Cuando `use_mlflow=True`, cada `train_models()` abre un run automáticamente y loguea parámetros, métricas y el modelo como artifact

### `redes_neuronales`
- `output_dim` se calcula automáticamente: `n_clases` para clasificación, `1` para regresión
- `output_dim.joblib` se guarda en `models/artifacts/` para que la API lo cargue correctamente
- TorchMetrics proporciona métricas consistentes entre entrenamiento y evaluación
- El optimizador elegido (`optimizer_type`) se aplica en `train_model.py`, `tune_model.py` y `load_checkpoint()`
- La función de pérdida (`nn_loss_fn=Auto`) se selecciona automáticamente según `task_type`

### `no_supervisado`
- `process_input()` disponible para inferencia desde API y chat
- Monitorización de drift disponible aunque no haya variable objetivo

### `hibrido`
- `process_input()` aplica automáticamente la transformación dimensional guardada (PCA, UMAP, KMeans-features o IsolationForest)
- Compatible con todos los flags opcionales simultáneamente

---

## Changelog

Ver [CHANGELOG.md](CHANGELOG.md).

---

## License

Apache-2.0
