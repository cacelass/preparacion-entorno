# {{ project_name }}

![Python](https://img.shields.io/badge/Python-{{ python_version }}+-blue?logo=python&logoColor=white)
{% if ml_type == 'supervisado' %}![ML Type](https://img.shields.io/badge/ML-Supervised%20{{ task_type | capitalize }}-orange)
{% elif ml_type == 'no_supervisado' %}![ML Type](https://img.shields.io/badge/ML-Unsupervised%20Clustering-orange)
{% elif ml_type == 'redes_neuronales' %}![ML Type](https://img.shields.io/badge/ML-Neural%20Networks%20{{ nn_model }}-orange)
{% elif ml_type == 'hibrido' %}![ML Type](https://img.shields.io/badge/ML-Hybrid-orange)
{% endif %}{% if use_mlflow %}![Tracking](https://img.shields.io/badge/Experiment%20Tracking-MLflow-blue?logo=mlflow)
{% endif %}![Version](https://img.shields.io/badge/Version-{{ project_version }}-green)
![Author](https://img.shields.io/badge/Author-{{ project_author_name | replace(" ", "%20") | replace("-", "--") }}-blueviolet)
![Template](https://img.shields.io/badge/Generado%20con-dskit-58a6ff?logo=github)

> {{ project_description }}

**Tipo de ML:** `{{ ml_type }}`{% if ml_type == "redes_neuronales" %} — arquitectura: `{{ nn_model }}`{% endif %}
**Autor:** {{ project_author_name }}
**Versión:** {{ project_version }}{% if use_xgboost %} · XGBoost ✓{% endif %}{% if use_lightgbm %} · LightGBM ✓{% endif %}{% if use_catboost or model_type == 'CatBoost' %} · CatBoost ✓{% endif %}

---

## Tabla de contenidos

- [Estructura del proyecto](#estructura-del-proyecto)
- [Inicio rápido](#inicio-rápido)
- [Arnés de IA](#arnés-de-ia)
- [Sistema de agentes](#sistema-de-agentes){% if use_docker %}
- [Docker — Interfaz de chat](#docker--interfaz-de-chat){% endif %}{% if graphify_mode == "graphify + obsidian vault" %}
- [Obsidian Vault](#obsidian-vault){% endif %}

---

## Estructura del proyecto

```
{{ project_slug }}/
├── data/
│   ├── raw/            ← datos originales (nunca modificar)
│   ├── interim/        ← datos en proceso
│   └── processed/      ← datos listos para modelar
├── models/             ← modelos entrenados (.joblib / .pt)
│   └── artifacts/      ← encoders, scalers, etc.
├── notebooks/
│   ├── 0-0-...-Descargadatos.ipynb
│   ├── 0-1-...-ProcesamientoDatos.ipynb
│   └── 0-2-...-Ejecucion.ipynb
├── reports/figures/    ← gráficos generados
├── {{ project_slug }}/
│   ├── data/           make_dataset.py
│   ├── features/       build_features.py
│   ├── models/         train_model.py · predict_model.py
│   ├── visualization/  visualize.py
│   └── utils/          paths.py
├── tests/
├── progress/           ← memoria del arnés (tarea actual + histórico)
├── AGENTS.md           ← protocolo del arnés: punto de entrada de la IA
├── init.sh             ← la puerta: ¿se puede trabajar?
├── featureslist.json   ← backlog con criterios de aceptación
├── main.py             ← pipeline completo
├── Makefile
└── pyproject.toml
```

---

## Arnés de IA

El proyecto trae un **arnés** (*harness*): un entorno dentro del propio
repositorio que gobierna cómo trabaja un agente de IA sobre él. No es un
chatbot — es un ciclo con puerta de entrada, backlog y verificación.

```
./init.sh → progress/ → featureslist.json → implementar → revisar → done
    │
    └── si falla: el agente PARA. No se trabaja sobre un proyecto roto.
```

Funciona con **cualquier asistente**: `AGENTS.md` es la fuente única de reglas y
`CLAUDE.md` solo apunta a él, sin duplicar nada. Los cuatro subagentes se
escriben una vez en `.opencode/agents/` y `make assistants-sync` los espeja a
`.claude/agents/` con el formato que espera Claude Code.

| Pieza | Qué hace |
|-------|----------|
| `AGENTS.md` | Punto de entrada. Lo primero que lee cualquier agente |
| `init.sh` | Verifica entorno, ficheros del arnés y que los tests pasan |
| `featureslist.json` | Qué hay que hacer, con criterios de aceptación explícitos |
| `progress/` | Memoria fuera de la ventana de contexto: tarea actual e histórico |
| `.opencode/agents/` | `lider`, `explorer`, `implementer`, `reviewer` |

```bash
make init            # ¿se puede trabajar?
make harness-check   # solo estructura, sin tests
make backlog         # estado de las features
```

Y para arrancar el ciclo en tu asistente:

> Lee `AGENTS.md` y sigue el protocolo: ejecuta `./init.sh`, lee `progress/` y
> elige la primera feature pendiente.

**La regla que no se salta:** ninguna feature se marca `done` sin que
`./init.sh` pase en verde. El agente demuestra su trabajo, no lo declara.

El protocolo completo está en [`AGENTS.md`](AGENTS.md).


{% if use_docker %}
## Docker — Interfaz de chat

Generado con la plantilla **[dskit](https://github.com/cacelass/dskit)**.

La imagen Docker incluye una interfaz web de chat para interactuar con los
modelos entrenados directamente desde el navegador.

### Arrancar

```bash
# Construir y lanzar (entrena automaticamente si no hay modelos)
make docker-run

# O directamente con Docker Compose
docker compose up -d
```

La interfaz estara disponible en **http://localhost:8080**

> Si no existe ningun modelo entrenado, el contenedor intentara entrenar
> automaticamente al arrancar (requiere `dataset.csv` en la raiz del proyecto).

### Comandos Docker

```bash
make docker-run     # construir imagen + arrancar contenedor
make docker-update  # reconstruir con los ultimos cambios
make docker-down    # parar y eliminar contenedores
```

### Comandos disponibles en el chat

| Comando | Descripcion |
|---|---|
| `status` | Estado del sistema y modelos cargados |
| `predict` | Prediccion interactiva paso a paso |
| `info` | Detalle de features y clases |
| `train` | Lanzar entrenamiento desde el chat |
| `reload` | Recargar modelos del disco |
| `help` | Mostrar ayuda |

---
{% endif %}

## Inicio rápido

```bash
# 1. Instalar dependencias
make setup

# 2. Activar entorno
source .venv/bin/activate

# 3. Colocar datos en data/raw/ y editar DATA_FILE / TARGET_COL en main.py

# 4. Explorar con notebooks
invoke lab

# 5. Pipeline completo
python main.py
```

Consulta el archivo `ayuda` para más detalles.

## Sistema de agentes

Este proyecto incluye un sistema de **agentes autónomos** que automatizan
tareas de desarrollo: git, tests, documentación, CI/CD, dependencias, revisión
de código, secretos y más. Todos los agentes son agnósticos de proveedor de IA
y se ejecutan vía comandos de shell.

```bash
# Listar agentes disponibles
uv run python -m agents list

# Ejecutar un agente (ej: análisis de dependencias)
uv run python -m agents run dependency check_outdated
```

Ver `AGENTS.md` para la documentación completa del sistema.

{% if graphify_mode == "graphify + obsidian vault" %}

---

## Obsidian Vault

El proyecto incluye un vault de Obsidian en `vault/` con estructura por dominios:

| Carpeta | Propósito |
|---------|-----------|
| `00_META/` | Índice general + plantillas para nuevas notas |
| `01_PROYECTO/` | Visión general, objetivos, decisiones |
| `02_DATOS/` | Documentación de datasets |
| `03_MODELOS/` | Experimentos, hiperparámetros, resultados |
| `04_VISUALIZACIONES/` | Análisis visual y gráficos |
| `05_AGENTES/` | Grafo de conocimiento generado por graphify |
| `06_OBSERVACIONES/` | Hallazgos y notas diarias |
| `07_REFERENCIAS/` | Papers y documentación externa |

Abre `vault/` como carpeta en Obsidian para explorar y documentar el proyecto.

{% endif %}

---

Template generado con https://github.com/cacelass/dskit
