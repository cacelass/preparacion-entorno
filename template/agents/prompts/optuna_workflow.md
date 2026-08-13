# Optuna Workflow — Hiperparámetros

## Pipeline
```
make tune  →  mejores params  →  make train (con mejores params)
```

## Pasos

| Paso | Comando | Qué hace | Agente |
|------|---------|----------|--------|
| Tuning | `make tune` | Optimización con Optuna, n_trials configurable | `ml` (analyze_study) |
| Aplicar | Editar `main.py` o `train_model.py` | Usar mejores params encontrados | `refactor` |
| Reentrenar | `make train` | Entrenar modelo final con mejores params | `ml` (inspect_model) |

## Paths
- `tools/tune_model.py` — script de optimización
- Los estudios se almacenan en la base de datos de Optuna (local por defecto)

## Agente `ml` — acciones relevantes
- `analyze_study` — analiza estudios de Optuna: mejores trials, importancia de hiperparámetros, curva de convergencia
- `list_models` — lista modelos disponibles tras reentrenar

## Problemas comunes
- Tuning lento → reducir n_trials, usar pruning (TPESampler con early_stopping)
- Sin estudio previo → ejecutar `make tune` primero
- Mejores params no mejoran → el modelo base puede no ser adecuado, probar otro algoritmo
