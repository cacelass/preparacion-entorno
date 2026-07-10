---
tags:
  - agente
  - entorno
---
# Env Agent

> Dueño del entorno: versión de Python, uv sync/lock, dependencias declaradas.

## Contrato

- **Rol:** Environment manager
- **Capacidades:** verificar python; uv sync; uv lock --check; añadir dependencias con uv add
- **Límites:** no juzga obsolescencia (→ dependency); no toca versión del proyecto (→ documentation)
- **Necesita:** el nombre del paquete para añadir dependencia
- **Recursos:** uv.lock, .venv/, pyproject.toml (dependencias)
- **Colabora con:** dependency

## Responsabilidades

1. Mantener el entorno Python sincronizado
2. Verificar la versión de Python
3. Añadir/actualizar dependencias
4. Validar uv.lock

## Dependencias

- Python {{ python_version if python_version is defined else '3.11+' }}
- uv package manager
