### Instalación rápida de uv

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```


### Inicializacion
```bash
cd <nombre_directorio_creado>
uv sync
uv run pytest
uvx ty check <documento>.py
sphinx-apidoc -o docs/source/ src/.
cd doc/ --> make html
```


### Configura tu entorno de proyecto

1. Instala todas las dependencias del proyecto:
   ```bash
   cd <nombre_directorio_creado>
   uv sync
   source .venv/bin/activate
   ```

> Si agregas nuevos paquetes a `pyproject.toml`, recuerda volver a ejecutar `uv sync` para actualizar tu entorno.

---
