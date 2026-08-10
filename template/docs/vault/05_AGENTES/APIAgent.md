---
tags:
  - agente
  - api
---
# API Agent

> Revisor de la API FastAPI (solo con use_api=true): endpoints vs docs + smoke test.

## Contrato

- **Rol:** API reviewer
- **Capacidades:** cruzar endpoints declarados contra documentados; smoke test con TestClient
- **Límites:** no modifica api/main.py; no despliega la API
- **Colabora con:** test

## Dependencias

- API habilitada: {{ 'sí' if use_api else 'no' }}
