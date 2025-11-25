# Cookiecutter Personal

## Requerimientos

- [git](https://git-scm.com/) >= 2.x
- [Cookiecutter Python package](http://cookiecutter.readthedocs.org/en/latest/installation.html) >= 1.4.0
- [uv](https://github.com/astral-sh/uv)


### Instalación rápida de uv

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```
### Instalación rápida de cookiecutter

```bash
apt install cookiecutter
```

### Configura tu entorno de proyecto

1. Instala todas las dependencias del proyecto:
   ```bash
   cd <nombre_directorio_creado>
   uv sync
   ```

Esto instalará todas las dependencias que se encuentran en el archivo `pyproject.toml`.

## Estructura de directorios y archivos resultantes

    {{ cookiecutter.project_slug }}
        ├── data
        │   ├── processed      <- Conjuntos de datos finales y limpios, listos para el modelado.
        │   └── raw            <- Datos originales sin modificar, tal como fueron obtenidos.
        │
        ├── notebooks          <- Notebooks de Jupyter. La convención de nombres usa un número
        │                         (para ordenar), las iniciales del autor y una breve descripción
        │                         separada por guiones. Ejemplo:
        │                         `1.0-jvelezmagic-exploracion-inicial-datos`.
        │
        ├── .gitignore         <- Lista de archivos y carpetas que `git` debe ignorar.
        │
        ├── pyproject.toml
        │                      
        └── README.md

---

> Si necesitas instalar más paquetes en tu proyecto, agrégalos a `pyproject.toml` y vuelve a ejecutar `uv sync` dentro del entorno.
