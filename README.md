# Creacion de entornos para trabajos de datos

**Template Profesional para Data Analyst & Data Science**

Este repositorio es un **template de proyecto de Data Science**, diseñado para que analistas y científicos de datos puedan iniciar cualquier proyecto de manera organizada y profesional.  
Incluye una estructura modular para trabajar con **datos, notebooks, features, modelos, visualización y documentación**, permitiendo reproducibilidad y buenas prácticas desde el inicio.

---

## Índice

- [Características principales](#características-principales)
- [Requisitos previos](#requisitos-previos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del proyecto generado](#estructura-del-proyecto-generado)

---

## Características principales

- Estructura modular de proyecto **lista para pipelines de Data Science**.
- Carpeta `notebooks` para desarrollo exploratorio.
- Scripts separados para **preprocesamiento, features, entrenamiento y predicción**.
- Documentación con **Sphinx** integrada.
- Entorno reproducible con **uv** y gestión de dependencias opcionales por tipo de ML.
- Carpeta `tests` para pruebas unitarias de tus scripts.

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
cookiecutter https://github.com/cacelass/preparacion-entorno.git
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