# DSKIT

![version](https://img.shields.io/badge/dskit-1.7.0-blue)
![python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)
![uv](https://img.shields.io/badge/gestor-uv-green)
![license](https://img.shields.io/badge/license-GPL--3.0-lightgrey)

**Template profesional para Data Science y AI Engineering**

Plantilla basada en [copier](https://copier.readthedocs.io), diseñada para iniciar proyectos de ML de forma organizada, reproducible y lista para producción. Construida sobre `uv`, Sphinx y una arquitectura modular que cubre todo el flujo de trabajo desde la ingesta de datos hasta el modelo evaluado, exportado y servido como API.

---

## Índice

- [DSKIT](#dskit)
  - [Índice](#índice)
  - [Características](#características)
  - [Requisitos previos](#requisitos-previos)
  - [Instalación rápida](#instalación-rápida)
  - [Variables](#variables)
    - [Validaciones](#validaciones)
  - [Uso](#uso)
  - [Estructura generada](#estructura-generada)
  - [Makefile](#makefile)
  - [Changelog](#changelog)
  - [License](#license)

---

## Características

- **4 tipos de ML** con código y tests listos desde el primer `make run`:
  `supervisado`, `no_supervisado`, `redes_neuronales`, `hibrido`
- **2 tipos de tarea** (`task_type`): `clasificacion` o `regresion`
- **5 arquitecturas de red neuronal**: MLP, CNN1D, LSTM, GRU, Transformer
- **XGBoost y LightGBM** opcionales en supervisado e híbrido
- **Selector de modelo** (`model_type`): `todos`, RandomForest, XGBoost, LightGBM, LogisticRegression, KNN, DecisionTree, **SVM**
- **API REST** opcional (`use_api`): FastAPI con `/health`, `/info` y `/predict` — `make serve`
- **DuckDB** opcional (`use_duckdb`): carga CSV/Parquet/JSON con SQL directo — `make query`
- **Optuna** opcional (`use_optuna`): HPO automática por modelo — `make tune` + `make train`
- **Monitoring** opcional (`use_monitoring`): drift KS/chi², performance vs baseline — `make monitor`
- **MLflow** opcional: tracking de experimentos, artifacts y Model Registry
- **Early stopping** y **validation split** configurables en redes neuronales
- **`uv sync` automático** tras generar el proyecto
- **`make smoke`**: tests de humo para verificar que el pipeline arranca
- **`make profile`**: profiling con cProfile + snakeviz
- **TensorBoard** integrado en redes neuronales (`make tb`)
- Gestión de entornos con `uv` y grupos de dependencias por tipo de ML
- Documentación con Sphinx, tests con pytest, linting con ruff

---

## Requisitos previos

```bash
sudo apt install pipx
pipx ensurepath
pipx install copier
pip install copier uv
```

Python >= 3.10 requerido.

---

## Instalación rápida

```bash
copier copy --trust gh:cacelass/dskit nombre_proyecto
```

O desde una copia local:

```bash
copier copy --trust ./dskit nombre_proyecto
```

Copier ejecuta `uv sync` automáticamente tras generar. Si falla, hazlo manualmente:

```bash
cd nombre_proyecto
uv sync --extra dev --extra <ml_type>
source .venv/bin/activate
```

> Los iconos de micrófono en los prompts son parte de la UI de copier y no son configurables.

---

## Variables

Copier muestra solo las preguntas relevantes según las respuestas anteriores — las variables condicionales no aparecen si no aplican.

| Variable | Valores | Condición | Descripción |
|---|---|---|---|
| `project_name` | texto | siempre | Nombre del proyecto |
| `project_author_name` | texto | siempre | Nombre del autor |
| `project_author_email` | email | siempre | Email (validado) |
| `project_description` | texto | siempre | Descripción breve |
| `ml_type` | `supervisado` · `no_supervisado` · `redes_neuronales` · `hibrido` | siempre | Determina qué código se genera |
| `task_type` | `clasificacion` · `regresion` | supervisado, redes_neuronales, hibrido | Tipo de tarea |
| `nn_model` | `MLP` · `CNN1D` · `LSTM` · `GRU` · `Transformer` | solo redes_neuronales | Arquitectura de red |
| `model_type` | `todos` · `RandomForest` · `XGBoost` · `LightGBM` · `LogisticRegression` · `KNN` · `DecisionTree` · `SVM` | solo supervisado e hibrido | Modelo a entrenar |
| `use_xgboost` | true · false | solo supervisado e hibrido | Añade XGBoost |
| `use_lightgbm` | true · false | solo supervisado e hibrido | Añade LightGBM |
| `use_mlflow` | true · false | siempre | Integra MLflow |
| `use_monitoring` | true · false | siempre | Drift detection y performance tracking |
| `use_optuna` | true · false | siempre | HPO automática con Optuna |
| `use_duckdb` | true · false | siempre | Carga con DuckDB (CSV/Parquet/JSON + SQL) |
| `use_api` | true · false | siempre | Genera API REST con FastAPI |
| `python_version` | `3.10` – `3.13` | siempre | Versión de Python |
| `project_version` | texto | siempre | Versión inicial |

### Validaciones

Copier valida automáticamente antes de generar:
- Slug solo con `[a-z0-9_]` empezando por letra
- Email con formato válido

---

## Uso

```bash
make help       # ver todos los comandos disponibles
make run        # pipeline completo
make smoke      # tests de humo rápidos
make profile    # cProfile → reports/profile.prof
make tb         # TensorBoard localhost:6006 (solo redes_neuronales)
make mlflow     # MLflow UI localhost:5000 (solo si use_mlflow=true)
make monitor    # drift + performance report (solo si use_monitoring=true)
make tune       # HPO con Optuna (solo si use_optuna=true)
make serve      # API REST localhost:8000   (solo si use_api=true)
make query      # Shell DuckDB interactivo  (solo si use_duckdb=true)
```

---

## Estructura generada

```
nombre_proyecto/
├── <project_slug>/           ← paquete Python
│   ├── data/make_dataset.py  ← carga pandas + DuckDB (si use_duckdb=true)
│   ├── features/build_features.py
│   ├── models/
│   │   ├── train_model.py    ← adaptado al ml_type, task_type y nn_model
│   │   └── predict_model.py  ← métricas, figuras y CSVs en reports/
│   ├── utils/paths.py
│   └── visualization/visualize.py
├── api/                      ← solo si use_api=true
│   ├── main.py               ← FastAPI: /health /info /predict
│   └── schemas.py            ← Pydantic V2
├── tuning/                   ← solo si use_optuna=true
│   └── tune_model.py         ← objetivos Optuna por modelo + tune_models()
├── monitoring/               ← solo si use_monitoring=true
│   └── monitor.py            ← check_drift, check_performance, run_monitoring
├── data/{raw,interim,processed,external}/
├── models/                   ← pesos .pt / .joblib + best_params_*.joblib
├── notebooks/
├── reports/
│   ├── figures/              ← matrices de confusión, real vs predicho, SHAP
│   ├── monitoring/           ← drift_report.csv + drift_report.html
│   └── resultados_*.csv      ← métricas ordenadas
├── tests/
│   ├── conftest.py
│   ├── test_train_model.py
│   ├── test_api.py           ← solo si use_api=true
│   ├── test_tuning.py        ← solo si use_optuna=true
│   └── test_monitoring.py    ← solo si use_monitoring=true
├── .copier-answers.yml       ← generado por copier, habilita copier update
├── Makefile
├── pyproject.toml
└── main.py
```

---

## Makefile

| Target | Descripción |
|---|---|
| `make run` | Pipeline completo (`main.py`) |
| `make data / train / predict` | Pasos individuales |
| `make test` | pytest completo |
| `make smoke` | Solo `@pytest.mark.smoke` |
| `make lint / format` | ruff check / ruff format |
| `make profile` | cProfile → `reports/profile.prof` |
| `make tb` | TensorBoard localhost:6006 *(redes_neuronales)* |
| `make mlflow` | MLflow UI localhost:5000 *(use_mlflow=true)* |
| `make monitor` | Drift + performance report *(use_monitoring=true)* |
| `make tune` | HPO con Optuna *(use_optuna=true)* |
| `make serve` | API REST localhost:8000 *(use_api=true)* |
| `make query` | Shell DuckDB interactivo *(use_duckdb=true)* |
| `make clean-all` | Cachés + modelos + figuras |

---

## Changelog

Ver [CHANGELOG.md](CHANGELOG.md).

---

## License

GPL-3.0