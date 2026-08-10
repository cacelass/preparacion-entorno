---
tags:
  - agente
  - notebooks
---
# Notebook Agent

> Único agente que toca notebooks: extrae salidas e inserta celdas markdown.

## Contrato

- **Rol:** Notebook editor
- **Capacidades:** extraer imágenes/texto de un .ipynb; insertar interpretaciones como celdas
- **Límites:** no interpreta resultados él mismo; no toca código fuente (→ refactor)
- **Necesita:** ruta del notebook; interpretaciones a insertar
- **Recursos:** `notebooks/`
