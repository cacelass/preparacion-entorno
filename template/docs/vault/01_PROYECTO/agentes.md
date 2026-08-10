# Sistema de Agentes — {{ project_name }}

> Este proyecto utiliza dskit, un sistema multi-agente con contratos explícitos.
> Cada agente tiene un rol, capacidades, límites, recursos propios y colaboraciones definidas.

## Arquitectura del sistema

```
Humano
  └── plan (Jefe de proyecto: descompone encargos y delega)
        ├── knowledge (Grafo de conocimiento + Vault Obsidian)
        │     ├── docsearch (Búsqueda en el grafo)
        │     └── research (Búsqueda externa — arXiv, OpenAlex)
        ├── review (Revisor de código)
        │     └── refactor (Modifica código fuente)
        ├── test (Ejecuta tests)
        ├── data (EDA y calidad de datos)
        ├── ml (Análisis de modelos)
        │     └── mlflow (Tracking de experimentos)
        ├── git (Historial git)
        │     └── documentation (Documentación y versión)
        ├── cicd (Workflows CI/CD)
        │     └── make (Makefile)
        ├── env (Entorno Python)
        │     └── dependency (Vulnerabilidades)
        ├── docker (Configuración Docker)
        ├── api (API FastAPI)
        ├── graph (Auditor de figuras)
        ├── notebook (Notebooks Jupyter)
        ├── secrets (Escáner de secretos)
        ├── installer (Agentes externos)
        ├── doctor (Diagnóstico integral)
        ├── schedule (Experto en cron)
        ├── supervisor (Coordinación en competencia)
        └── audit (Auditor del equipo)
```

## Reglas del equipo

1. **Nadie se pisa** — cada recurso escribible tiene UN único dueño
2. **Nadie improvisa fuera de su rol** — `cannot` define lo que NO hace y a quién derivarlo
3. **Nadie inventa información** — `needs` lista lo que necesita; si falta, pregunta

Ver `template/agents/contracts.py` para los contratos completos.
