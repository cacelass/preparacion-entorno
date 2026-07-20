# API Workflow — API REST

## Pipeline
```
make serve  →  docs en /docs  →  tests
```

## Pasos

| Paso | Comando | Agente |
|------|---------|--------|
| Servir | `make serve` | `api` (verifica endpoints) |
| Documentación | `http://localhost:8000/docs` | `api` (cross-check docs vs código) |
| Tests | `pytest tests/test_api.py` | `test` |

## Paths
- `api/main.py` — FastAPI app con endpoints
- `api/schemas.py` — modelos Pydantic
- `tests/test_api.py` — tests de integración

## Agente `api`
- `check_endpoints` — verifica que endpoints documentados existen en el código
- `validate_schemas` — valida esquemas de entrada/salida
- Se integra con `test` para verificar cobertura de endpoints

## Problemas comunes
- Servidor no arranca → dependencias faltantes, puerto ocupado
- Endpoint 404 → ruta no registrada en `api/main.py`
- Schema mismatch → `api` + `validate_schemas` para diagnosticar
