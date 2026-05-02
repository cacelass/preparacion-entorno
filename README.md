# DSKIT

![version](https://img.shields.io/badge/dskit-1.1.2-blue)
![python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)
![uv](https://img.shields.io/badge/gestor-uv-green)
![license](https://img.shields.io/badge/license-GPL--3.0-lightgrey)

**Template profesional para Data Science y AI Engineering**

Plantilla basada en [copier](https://copier.readthedocs.io), diseñada para iniciar proyectos de ML de forma organizada, reproducible y lista para producción. Construida sobre `uv`, Sphinx y una arquitectura modular que cubre todo el flujo de trabajo desde la ingesta de datos hasta el modelo evaluado y exportado.

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
- **Selector de modelo** (`model_type`): entrena uno o todos
- **MLflow** opcional: tracking de experimentos, artifacts y Model Registry
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
| `model_type` | `todos` · `RandomForest` · `XGBoost` · `LightGBM` · `LogisticRegression` · `KNN` · `DecisionTree` | solo supervisado e hibrido | Modelo a entrenar |
| `use_xgboost` | true · false | solo supervisado e hibrido | Añade XGBoost |
| `use_lightgbm` | true · false | solo supervisado e hibrido | Añade LightGBM |
| `use_mlflow` | true · false | siempre | Integra MLflow |
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
```

---

## Estructura generada

```
nombre_proyecto/
├── <project_slug>/           ← paquete Python
│   ├── data/make_dataset.py
│   ├── features/build_features.py
│   ├── models/
│   │   ├── train_model.py    ← adaptado al ml_type, task_type y nn_model
│   │   └── predict_model.py  ← métricas, figuras y CSVs en reports/
│   ├── utils/paths.py
│   └── visualization/visualize.py
├── data/{raw,interim,processed,external}/
├── models/                   ← pesos .pt / .joblib
├── notebooks/
│   ├── 0-0-DescargaDatos.ipynb
│   ├── 0-1-ProcesamientoDatos.ipynb
│   └── 0-2-Ejecucion.ipynb   ← celdas adaptadas al ml_type generado
├── reports/
│   ├── figures/              ← matrices de confusión, real vs predicho, residuos
│   └── resultados_*.csv      ← métricas ordenadas por métrica principal
├── tests/
│   ├── conftest.py           ← fixtures adaptadas al task_type
│   └── test_train_model.py   ← tests por arquitectura + @pytest.mark.smoke
├── ayuda/                    ← referencia rápida de comandos y modelos
├── .env.example
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
| `make clean-all` | Cachés + modelos + figuras |

---

## Changelog

Ver [CHANGELOG.md](CHANGELOG.md).

---

## Documentación completa

Ver la [Wiki del proyecto](https://github.com/cacelass/dskit/wiki) para guías detalladas, referencia de variables, troubleshooting y changelog.

---

## License

GPL-3.0
