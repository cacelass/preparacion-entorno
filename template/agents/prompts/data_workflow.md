# Data Workflow — Pipeline de datos

## Pipeline
```
make data  →  make features
data/raw/     data/processed/ → data/interim/
```

## Pasos y agentes

| Paso | Comando | Qué hace | Agente |
|------|---------|----------|--------|
| Ingesta | `make data` | `make_dataset.py`: carga raw, limpia, escribe processed | `data` (EDA, calidad) |
| Features | `make features` | `build_features.py`: split, escala, codifica, LOGCOLS | `data` (skewness, correlaciones) |
{% if use_duckdb %} | Exploración SQL | `make query` | DuckDB shell sobre `data/raw/` | `data` (list_datasets) |{% endif %}

## Paths
- `data/raw/` — datos crudos (inmutables)
- `data/processed/` — datos limpios (salida de make data)
- `data/interim/` — features (salida de make features)
- `data/external/` — datos de referencia externos

## Agente `data` — acciones clave
- `eda_report` — perfil completo: nulos, cardinalidad, constantes, outliers, correlaciones
- `detect_skewness` — sugiere columnas para LOGCOLS
- `detect_leakage` — fuga de información (necesita target_col)
- `generate_plots` — pairplot, heatmap, histogramas, boxplots
- `list_datasets` — descubre datasets en raw/interim/processed

## Problemas comunes
- `make data` falla → columnas faltantes, tipos incorrectos, encoding
- `make features` falla → nulos inesperados en split, cardinalidad alta en nuevas variables
- Leakage → `data detect_leakage` antes de modelar
- Skewness no tratada → modelos lineales rinden mal
