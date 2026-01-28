# Creacion de entornos para trabajos de datos
**Template Profesional para Data Analyst & Data Science**

Este repositorio es un **template de proyecto de Data Science**, diseñado para que analistas y científicos de datos puedan iniciar cualquier proyecto de manera organizada y profesional.  
Incluye una estructura modular para trabajar con **datos, notebooks, features, modelos, visualización y documentación**, permitiendo reproducibilidad y buenas prácticas desde el inicio.

---

## Características principales

- Estructura modular de proyecto **lista para pipelines de Data Science**.  
- Carpeta `notebooks` para desarrollo exploratorio.  
- Scripts separados para **preprocesamiento, features, entrenamiento y predicción**.  
- Documentación con **Sphinx** integrada.  
- Entorno reproducible con **uv** y gestión de dependencias opcionales (`dev` y `ml`).  
- Carpeta `tests` para pruebas unitarias de tus scripts.

---

## Requerimientos

- [git](https://git-scm.com/) >= 2.x  
- [Cookiecutter Python package](http://cookiecutter.readthedocs.org/en/latest/installation.html) >= 1.4.0  
- [uv](https://github.com/astral-sh/uv) para gestión de entornos y ejecución de comandos  
- Python >= 3.10 recomendado  


Instalación rápida en Linux:

```bash
apt install git python3-pip python3-venv
apt install cookiecutter
wget -qO- https://astral.sh/uv/install.sh | sh
cookiecutter https://github.com/cacelass/preparacion-entorno.git
```

## Estructura de directorios y archivos resultantes

    .
    ├── {{ cookiecutter.project_slug }}
    │   ├── {{ cookiecutter.project_module_name }}
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
    │   ├── ayuda
    │   ├── data
    │   │   ├── external
    │   │   ├── interim
    │   │   ├── processed
    │   │   └── raw
    │   │       └── y
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
    │   │   ├── 0-0-{{ cookiecutter.project_author_name }}-Descargadatos.ipynb
    │   │   ├── 0-1-{{ cookiecutter.project_author_name }}-ProcesamientoDatos.ipynb
    │   │   └── 0-2-{{ cookiecutter.project_author_name }}-Ejecucion.ipynb
    │   ├── pyproject.toml
    │   ├── README.md
    │   ├── references
    │   ├── reports
    │   │   └── figures
    │   ├── setup.py
    │   ├── tasks.py
    │   └── tests
    │       ├── __init__.py
    │       └── test_proba.py
    ├── cookiecutter.json
    ├── install.md
    ├── pyproject.toml
    └── README.md
---
