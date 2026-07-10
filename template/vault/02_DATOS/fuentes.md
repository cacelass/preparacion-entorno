# Fuentes de Datos — {{ project_name }}

> Origen y descripción de los datasets utilizados.

## Fuentes

| Fuente | Tipo | Tamaño | Split | Procedencia |
|--------|------|--------|-------|-------------|
{% if use_public_dataset is defined and use_public_dataset %} | {{ use_public_dataset }} | Público | | train/{{ test_size if test_size is defined else '0.2' }} | {{ use_public_dataset }} |{% endif %}

## Estructura de datos

- **Raw:** `data/raw/`
- **Processed:** `data/processed/`
- **Interim:** `data/interim/`
- **External:** `data/external/`

## Calidad de datos

Ver análisis de [[05_AGENTES/DataAgent|data agent]].
