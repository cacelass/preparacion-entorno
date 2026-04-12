# DSKIT

**Template profesional para Data Analyst y Data Science**

Plantilla de proyectos de Ciencia de Datos basada en Cookiecutter, diseñada para iniciar cualquier proyecto de forma **organizada, reproducible y profesional**.  
Está construida sobre `uv`, Sphinx y una arquitectura modular que cubre todo el flujo de trabajo de Machine Learning, desde la ingesta de datos hasta el modelo desplegado.

Diseñada para eliminar la fricción del setup inicial y garantizar consistencia entre proyectos desde el primer commit.

---

## Índice

- [DSKIT](#dskit)
  - [Índice](#índice)
  - [Características principales](#características-principales)
  - [Requisitos previos](#requisitos-previos)
  - [Instalación](#instalación)
  - [Uso](#uso)
  - [Estructura del proyecto generado](#estructura-del-proyecto-generado)
  - [License](#license)

---

## Características principales

- Estructura modular lista para pipelines completos de Data Science:
  `data`, `features`, `models`, `utils`, `visualization`
- Carpeta `notebooks` preconfigurada para análisis exploratorio y prototipado
- Scripts desacoplados para:
  - preprocesamiento
  - ingeniería de características
  - entrenamiento
  - inferencia/predicción
- Gestión de entornos con `uv`, incluyendo grupos de dependencias opcionales según el tipo de proyecto
- Notebooks preparados para descarga, procesamiento y ejecución de datos
- Documentación profesional integrada con Sphinx
- Estructura de tests unitarios con `pytest`
- `Makefile` con automatización de tareas comunes

---

## Requisitos previos

Antes de usar el template, instala estas herramientas:

- [git](https://git-scm.com/) >= 2.x
- [cookiecutter](http://cookiecutter.readthedocs.org/en/latest/installation.html) >= 1.4.0
- [uv](https://github.com/astral-sh/uv) para gestión de entornos
- Python >= 3.10

---

## Instalación

```bash
# 1. Instalar dependencias del sistema
apt install git python3-pip python3-venv

# 2. Instalar cookiecutter
apt install cookiecutter

# 3. Instalar uv
wget -qO- https://astral.sh/uv/install.sh | sh
```

---

## Uso

En el directorio donde quieras crear tu proyecto:

```bash
cookiecutter https://github.com/cacelass/dskit.git
```

Una vez generado el proyecto, configura el entorno:

```bash
cd <nombre_del_proyecto>

# Dependencias de desarrollo (linting, tests, docs)
uv sync --extra dev

# Dependencias de ML según el tipo de proyecto:
uv sync --extra supervisado
uv sync --extra no_supervisado
uv sync --extra redes_neuronales
uv sync --extra hibrido

source .venv/bin/activate
```

---

## Estructura del proyecto generado

```
.
├── {{ cookiecutter.project_slug }}
│   ├── {{ cookiecutter.project_slug }}
│   │   ├── data
│   │   │   ├── __init__.py
│   │   │   └── make_dataset.py
│   │   ├── features
│   │   │   ├── build_features.py
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── predict_model.py
│   │   │   └── train_model.py
│   │   ├── utils
│   │   │   ├── __init__.py
│   │   │   └── paths.py
│   │   └── visualization
│   │       ├── __init__.py
│   │       └── visualize.py
│   ├── ayuda               ← recursos de referencia (papers, cheatsheets, notas)
│   ├── data
│   │   ├── external
│   │   ├── interim
│   │   ├── processed
│   │   └── raw
│   ├── docs
│   │   ├── make.bat
│   │   ├── Makefile
│   │   └── source
│   │       ├── conf.py
│   │       ├── index.rst
│   │       └── _static
│   ├── LICENSE
│   ├── Makefile
│   ├── models
│   ├── notebooks
│   │   ├── 0-0-{{ cookiecutter.project_author_name }}-DescargaDatos.ipynb
│   │   ├── 0-1-{{ cookiecutter.project_author_name }}-ProcesamientoDatos.ipynb
│   │   └── 0-2-{{ cookiecutter.project_author_name }}-Ejecucion.ipynb
│   ├── pyproject.toml
│   ├── README.md
│   ├── references
│   ├── reports
│   │   └── figures
│   ├── tasks.py
│   └── tests
│       ├── __init__.py
│       └── test_proba.py
├── cookiecutter.json
├── pyproject.toml
└── README.md
```
---

## License

GPL-3.0
