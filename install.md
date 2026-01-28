
### Instalación rápida de uv

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```
### Instalación rápida de cookiecutter

```bash
apt install cookiecutter
```

## Crear un nuevo proyecto

En el directorio en el que quieras guardar tu proyecto:

```bash
cookiecutter https://github.com/cacelass/preparacion-entorno.git
```
### Configura tu entorno de proyecto

1. Instala todas las dependencias del proyecto:
   ```bash
   cd <nombre_directorio_creado>
   uv sync --extra dev
   uv sync --extra ml
   source .venv/bin/activate
   ```

Esto instalará todas las dependencias que se encuentran en el archivo `pyproject.toml`.
