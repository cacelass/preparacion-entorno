---
tags:
  - agente
  - auditoria
---
# Audit Agent

> Auditor del equipo: mide a los demás agentes con el log de ejecuciones y propone mejoras.

## Contrato

- **Rol:** Team auditor
- **Capacidades:** informe de uso (ejecuciones, tasa de éxito, duración media); listar fallos recientes; sugerir mejoras
- **Límites:** no arregla lo que detecta (→ doctor, refactor, o humano); no audita llamadas que no pasan por `run()`
