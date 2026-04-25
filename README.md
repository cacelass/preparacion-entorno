# DSKIT

![version](https://img.shields.io/badge/dskit-0.3.0-blue)
![python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)
![uv](https://img.shields.io/badge/gestor-uv-green)
![license](https://img.shields.io/badge/license-GPL--3.0-lightgrey)

**Template profesional para Data Science y AI Engineering**

Plantilla de proyectos basada en Cookiecutter, diseñada para iniciar cualquier
proyecto de ML de forma **organizada, reproducible y lista para producción**.
Construida sobre `uv`, Sphinx y una arquitectura modular que cubre todo el flujo
de trabajo, desde la ingesta de datos hasta el modelo evaluado y exportado.

---

## Índice

- [DSKIT](#dskit)
  - [Índice](#índice)
  - [Características](#características)
  - [Requisitos previos](#requisitos-previos)
  - [Instalación rápida](#instalación-rápida)
  - [Variables de cookiecutter](#variables-de-cookiecutter)
    - [Validaciones automáticas (`pre_gen_project.py`)](#validaciones-automáticas-pre_gen_projectpy)
  - [Uso](#uso)
  - [Estructura generada](#estructura-generada)
  - [Makefile](#makefile)
  - [Changelog](#changelog)
  - [License](#license)

---

## Características

- **4 tipos de ML** con código y tests listos desde el primer `make run`:
  `supervisado`, `no_supervisado`, `redes_neuronales`, `hibrido`
- **5 arquitecturas de red neuronal** seleccionables: MLP, CNN1D, LSTM, GRU, Transformer
- **XGBoost y LightGBM** opcionales en supervisado e híbrido
- **Selector de modelo** (`model_type`): entrena uno o todos
- **`hooks/`**: validación antes de generar + setup automático del entorno tras generar
- **`make smoke`**: tests de humo para verificar que el pipeline arranca
- **`make profile`**: profiling con cProfile + snakeviz integrado
- **TensorBoard** integrado en redes neuronales (`make tb`)
- **Exportación de predicciones a CSV** con probabilidades y columna `correcto`
- Gestión de entornos con `uv` y grupos de dependencias por tipo de ML
- Documentación con Sphinx, tests con pytest, linting con ruff

---

## Requisitos previos

```bash
pip install cookiecutter uv
```

Python >= 3.10 requerido.

---

## Instalación rápida

```bash
cookiecutter https://github.com/cacelass/dskit.git
```

El hook `post_gen_project.py` ejecuta automáticamente `uv sync` al terminar.
Si falla, hazlo manualmente:

```bash
cd <nombre_proyecto>
uv sync --extra dev --extra <ml_type>
source .venv/bin/activate
```

---

## Variables de cookiecutter

| Variable | Valores | Descripción |
|---|---|---|
| `project_name` | texto | Nombre del proyecto |
| `project_author_name` | texto | Nombre del autor |
| `project_author_email` | email | Email (validado) |
| `project_description` | texto | Descripción breve |
| `ml_type` | `supervisado` · `no_supervisado` · `redes_neuronales` · `hibrido` | Tipo de ML — determina qué código se genera |
| `nn_model` | `MLP` · `CNN1D` · `LSTM` · `GRU` · `Transformer` | Arquitectura de red (solo en `redes_neuronales`) |
| `model_type` | `todos` · `RandomForest` · `XGBoost` · `LightGBM` · `LogisticRegression` · `KNN` · `DecisionTree` | Modelo a entrenar (solo en `supervisado` e `hibrido`) |
| `use_xgboost` | `no` · `si` | Añade XGBoost |
| `use_lightgbm` | `no` · `si` | Añade LightGBM |
| `python_version` | `3.10` – `3.13` | Versión de Python |
| `project_version` | `0.1.0` | Versión inicial |

### Validaciones automáticas (`pre_gen_project.py`)

Aborta con mensaje claro si:
- `ml_type` / `nn_model` / `model_type` no válido
- `model_type=XGBoost` sin `use_xgboost=si` (y viceversa con LightGBM)
- `python_version` no soportada, email sin formato válido, slug con caracteres inválidos

Advierte sin abortar si variables sin efecto están activadas (ej. `nn_model=LSTM` con `ml_type=supervisado`).

---

## Uso

```bash
make help       # ver todos los comandos
make run        # pipeline completo
make smoke      # tests de humo rápidos
make profile    # cProfile → reports/profile.prof
make tb         # TensorBoard (solo redes_neuronales)
```

---

## Estructura generada

```
<project_slug>/
├── <project_slug>/
│   ├── data/make_dataset.py
│   ├── features/build_features.py
│   ├── models/
│   │   ├── train_model.py    ← adaptado al ml_type y nn_model elegidos
│   │   └── predict_model.py  ← exporta predicciones y métricas a reports/
│   ├── utils/paths.py
│   └── visualization/visualize.py
├── data/{raw,interim,processed,external}/
├── models/                   ← pesos .pt / .joblib
├── notebooks/
│   └── 0-2-Ejecucion.ipynb  ← celdas adaptadas al ml_type generado
├── reports/
│   ├── figures/              ← matrices de confusión, distribuciones
│   ├── predicciones_*.csv    ← y_true, y_pred, proba_*, correcto
│   └── resultados_*.csv      ← métricas ordenadas por F1
├── tests/
│   ├── conftest.py           ← fixtures y patch_paths automático
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
| `make run` | `main.py` completo |
| `make data / train / predict` | pasos individuales del pipeline |
| `make test` | pytest completo |
| `make smoke` | solo `@pytest.mark.smoke` |
| `make lint / format` | ruff |
| `make profile` | cProfile → `reports/profile.prof` |
| `make tb` | TensorBoard localhost:6006 *(redes_neuronales)* |
| `make clean-all` | cachés + modelos + figuras |

---

## Changelog

Ver [CHANGELOG.md](CHANGELOG.md).

---

## License

GPL-3.0