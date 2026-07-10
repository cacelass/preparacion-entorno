# Arquitectura — {{ project_name }}

> {{ project_description }}

## Stack tecnológico

- **Lenguaje:** Python ({{ python_version if python_version is defined else '3.11+' }})
- **ML:** {{ ml_type }}
- **API:** FastAPI ({{ 'sí' if use_api else 'no' }})
- **Tracking:** MLflow ({{ 'sí' if use_mlflow else 'no' }})
- **Dataset público:** {{ use_public_dataset if use_public_dataset is defined else 'N/A' }}
- **Grafo de conocimiento:** {{ graphify_mode }}

## Estructura del proyecto

```
{{ project_slug }}/
├── data/              — Datos (raw/processed/interim/external)
├── features/          — Ingeniería de features
├── models/            — Modelos entrenados (.joblib, .pkl)
├── visualization/     — Código de visualización
├── api/               — FastAPI (si use_api)
├── pipelines/         — Pipelines de entrenamiento/evaluación
│
tests/
├── test_*.py          — Tests del proyecto
│
agents/                — Sistema multi-agente (dskit)
├── contracts.py       — Contratos de rol
├── agents/            — Implementación de agentes
├── prompts/           — Prompts de agentes
├── workspace/         — Workspace de agentes
│
vault/                 — Bóveda Obsidian del proyecto
├── 00_META/           — Metadatos e índice para IA
├── 01_PROYECTO/       — Documentación del proyecto
├── 02_DATOS/          — Documentación de datos
├── 04_VISUALIZACIONES/— Visualizaciones y grafos
├── 05_AGENTES/        — Fichas de agentes
```

## Pipeline de ML

```
data → features → train → evaluate → predict
  └── EDA (data agent)     └── análisis (ml agent)
```
