# Monitoring Workflow — Monitorización

## Pipeline
```
make monitor  →  reports/monitoring/
```

## Pasos

| Paso | Comando | Qué hace | Agente |
|------|---------|----------|--------|
| Ejecutar | `make monitor` | Drift detection + performance report | `ml` (overfitting, comparativa) |
| Revisar | `reports/monitoring/*.json` | Reportes generados | — |

## Paths
- `monitoring/monitor.py` — script de monitorización
- `reports/monitoring/` — reportes generados (JSON)
- `reports/figures/` — gráficos de drift y rendimiento

## Agentes involucrados
- `ml` — `check_overfitting`, `model_comparison` (diagnóstico post-monitor)
- `graph` — `audit_figures` (verifica gráficos de monitor)
- Si se detecta drift, el flujo recomendado es: `pipeline fix` → reentrenar → re-monitorizar

## Problemas comunes
- Monitor no disponible → `make setup` o `uv sync --extra monitoring`
- Sin datos históricos → ejecutar varias veces para establecer baseline
- Falsos positivos de drift → ajustar umbral de detección en `monitor.py`
