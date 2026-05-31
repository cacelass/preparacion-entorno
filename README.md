# DSKIT

![version](https://img.shields.io/badge/dskit-1.8.5-blue)
![python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)
![uv](https://img.shields.io/badge/gestor-uv-green)
![license](https://img.shields.io/badge/license-GPL--3.0-lightgrey)

**Template profesional para Data Science y AI Engineering**

Plantilla [copier](https://copier.readthedocs.io) para iniciar proyectos de ML de forma organizada, reproducible y lista para producción. Construida sobre `uv`, con una arquitectura modular que cubre el flujo completo desde la ingesta de datos hasta el modelo evaluado, exportado y servido como API o interfaz de chat.

---

## Índice

- [DSKIT](#dskit)
  - [Índice](#índice)
  - [Características](#características)
    - [Tipos de ML y arquitecturas](#tipos-de-ml-y-arquitecturas)
    - [Módulos opcionales](#módulos-opcionales)
    - [Pipeline de calidad integrado](#pipeline-de-calidad-integrado)
  - [Requisitos previos](#requisitos-previos)
  - [Instalación rápida](#instalación-rápida)
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

## Características

### Tipos de ML y arquitecturas
- **4 tipos de ML**: `supervisado`, `no_supervisado`, `redes_neuronales`, `hibrido`
- **2 tipos de tarea** (`task_type`): `clasificacion` o `regresion`
- **5 arquitecturas NN** (`nn_model`): MLP · CNN1D · LSTM · GRU · Transformer
- **Optimizador configurable** (`optimizer_type`): AdamW · Adam · SGD · RMSProp · Adagrad
- **Función de pérdida configurable** (`nn_loss_fn`): Auto · CrossEntropyLoss · MSELoss · L1Loss · BCEWithLogitsLoss
- **Selector de modelo** (`model_type`): todos · RandomForest · XGBoost · LightGBM · LogisticRegression · KNN · DecisionTree · SVM · CatBoost

### Módulos opcionales
| Flag | Descripción | Make target |
|---|---|---|
| `use_api` | API REST FastAPI — `/health`, `/info`, `/predict` | `make serve` |
| `use_optuna` | HPO automática por modelo con Optuna | `make tune` |
| `use_monitoring` | Drift KS/chi² + performance vs baseline (Evidently) | `make monitor` |
| `use_mlflow` | Tracking de experimentos, artifacts y Model Registry | `make mlflow` |
| `use_duckdb` | Carga CSV/Parquet/JSON con SQL directo | `make query` |
| `use_docker` | Docker + interfaz de chat Gradio | `make chat` |
| `use_shap` | SHAP values — importancia de features | automático |
| `use_xgboost` | XGBoost (supervisado/híbrido) | automático |
| `use_lightgbm` | LightGBM (supervisado/híbrido) | automático |
| `use_catboost` | CatBoost (supervisado/híbrido) | automático |

### Pipeline de calidad integrado
- **TorchMetrics** en bucle de entrenamiento y evaluación NN (Accuracy/F1/Precision/Recall para clasificación, MAE/RMSE/R² para regresión)
- **Early stopping** y **validation split** configurables en redes neuronales
- **TensorBoard** integrado en redes neuronales (`make tb`)
- **`make smoke`** — tests de humo que verifican que el pipeline arranca sin errores
- **`make profile`** — profiling con cProfile + snakeviz
- **`make lock`** — regenera `uv.lock` tras cambios en dependencias
- `uv sync` automático tras generar el proyecto

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

## Variables de configuración

Copier muestra solo las preguntas relevantes según las respuestas anteriores.

| Variable | Valores | Condición | Descripción |
|---|---|---|---|
| `project_name` | texto | siempre | Nombre del proyecto |
| `project_slug` | `[a-z0-9_]` | siempre | Nombre del paquete Python (auto desde project_name) |
| `project_author_name` | texto | siempre | Nombre del autor |
| `project_author_email` | email | siempre | Email (validado) |
| `project_description` | texto | siempre | Descripción breve |
| `ml_type` | `supervisado` · `no_supervisado` · `redes_neuronales` · `hibrido` | siempre | Determina el código generado |
| `task_type` | `clasificacion` · `regresion` | supervisado, redes_neuronales, hibrido | Tipo de tarea |
| `model_type` | `todos` · `RandomForest` · `XGBoost` · `LightGBM` · `LogisticRegression` · `KNN` · `DecisionTree` · `SVM` · `CatBoost` | supervisado, hibrido | Modelo a entrenar |
| `cluster_model` | `todos` · `KMeans` · `DBSCAN` · `AgglomerativeClustering` · `GaussianMixture` · `HDBSCAN` | no_supervisado | Algoritmo de clustering |
| `nn_model` | `MLP` · `CNN1D` · `LSTM` · `GRU` · `Transformer` | redes_neuronales | Arquitectura |
| `optimizer_type` | `AdamW` · `Adam` · `SGD` · `RMSProp` · `Adagrad` | redes_neuronales | Optimizador PyTorch |
| `nn_loss_fn` | `Auto` · `CrossEntropyLoss` · `MSELoss` · `L1Loss` · `BCEWithLogitsLoss` | redes_neuronales | Función de pérdida |
| `use_xgboost` | true/false | supervisado, hibrido | Añade XGBoost |
| `use_lightgbm` | true/false | supervisado, hibrido | Añade LightGBM |
| `use_catboost` | true/false | supervisado, hibrido | Añade CatBoost |
| `use_shap` | true/false | supervisado, hibrido | SHAP values |
| `use_mlflow` | true/false | siempre | MLflow tracking |
| `use_monitoring` | true/false | siempre | Drift + performance monitoring |
| `use_optuna` | true/false | siempre | HPO con Optuna |
| `use_duckdb` | true/false | siempre | DuckDB SQL sobre ficheros |
| `use_api` | true/false | siempre | API REST FastAPI |
| `use_docker` | true/false | siempre | Docker + chat Gradio |
| `python_version` | `3.10`–`3.13` | siempre | Versión de Python |
| `project_version` | texto | siempre | Versión inicial del proyecto |

### Validaciones automáticas
- `project_slug`: solo `[a-z0-9_]` empezando por letra — los guiones se transforman a `_`
- `project_author_email`: formato válido requerido

---

## Uso

```bash
make help        # ver todos los comandos disponibles
make run         # pipeline completo: data → features → train → predict
make data        # solo ingesta de datos
make features    # solo preprocesado
make train       # solo entrenamiento
make predict     # solo evaluación
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
make chat        # Interfaz Gradio            (use_docker=true)
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
| `make run` | Pipeline completo |
| `make data` | Ingesta de datos |
| `make features` | Preprocesado |
| `make train` | Entrenamiento |
| `make predict` | Evaluación + figuras + CSV |
| `make test` | pytest con cobertura (`--cov=<slug>`) |
| `make smoke` | Solo `@pytest.mark.smoke` |
| `make lint` | `ruff check` |
| `make format` | `ruff format` |
| `make profile` | cProfile → `reports/profile.prof` |
| `make lock` | `uv lock` — regenera lockfile |
| `make docs` | Sphinx autodoc |
| `make tb` | TensorBoard :6006 *(redes_neuronales)* |
| `make mlflow` | MLflow UI :5000 *(use_mlflow)* |
| `make monitor` | Drift + performance report *(use_monitoring)* |
| `make tune` | HPO Optuna *(use_optuna)* |
| `make serve` | API REST :8000 *(use_api)* |
| `make query` | Shell DuckDB *(use_duckdb)* |
| `make chat` | Gradio :7860 *(use_docker)* |
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

GPL-3.0