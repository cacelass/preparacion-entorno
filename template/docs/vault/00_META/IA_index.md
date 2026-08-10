# Índice para IA — {{ project_name }}

> Generado por dskit — template para agentes de IA.
> Este archivo está diseñado para ser leído por agentes de IA (graphify, docsearch, knowledge)
> como punto de entrada al vault del proyecto.

## Metadata del proyecto

- **Nombre:** {{ project_name }}
- **Slug:** {{ project_slug }}
- **Versión:** {{ project_version }}
- **Descripción:** {{ project_description }}
- **Autor:** {{ project_author_name }}
- **ML Type:** {{ ml_type }}
- **Task Type:** {{ task_type if task_type is defined else 'N/A' }}

## Estructura del vault

| Archivo | Propósito |
|---------|-----------|
| `00_META/IA_index.md` | Éste archivo — punto de entrada para IA |
| `00_META/templates/` | Plantillas Obsidian para nuevas notas |
| `01_PROYECTO/agentes.md` | Sistema de agentes del proyecto |
| `01_PROYECTO/arquitectura.md` | Arquitectura del proyecto |
| `01_PROYECTO/modelos.md` | Modelos de ML |
| `01_PROYECTO/roadmap.md` | Roadmap del proyecto |
| `02_DATOS/features.md` | Features del dataset |
| `02_DATOS/fuentes.md` | Fuentes de datos |
| `04_VISUALIZACIONES/grafo_conocimiento.md` | Visualización del grafo de conocimiento |
| `05_AGENTES/_index.md` | Directorio de agentes |
| `05_AGENTES/*.md` | Fichas individuales de cada agente |

## Modelo de datos del proyecto

```
{{ project_name }} ({{ ml_type }})
├── Datos → {{ project_slug }}/data/
├── Features → {{ project_slug }}/features/
├── Modelos → {{ project_slug }}/models/
└── Visualizaciones → {{ project_slug }}/visualization/
```
