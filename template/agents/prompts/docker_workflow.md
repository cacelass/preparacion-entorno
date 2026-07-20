# Docker Workflow — Contenedores

## Pipeline
```
make docker-run  →  make docker-update  →  make docker-down
```

## Pasos

| Paso | Comando | Qué hace | Agente |
|------|---------|----------|--------|
| Build + run | `make docker-run` | Construye imagen, levanta contenedor en :8080 | `docker` (lint Dockerfile) |
| Actualizar | `make docker-update` | Reconstruye con cambios, recrea contenedor | `docker` (lint) |
| Bajar | `make docker-down` | Para y elimina contenedores | — |

## Paths
- `Dockerfile` — definición de imagen
- `docker-compose.yml` — orquestación de servicios
- `chat/app.py` — interfaz web del chat
- `chat/entrypoint.sh` — script de entrada

## Agente `docker`
- `lint` — valida Dockerfile: FROM con tag, sin :latest, USER, COPY vs ADD, apt-get
- `validate_compose` — valida docker-compose.yml
- Se ejecuta automáticamente en pipelines de release

## Problemas comunes
- Docker no instalado → instalar Docker Engine
- Puerto ocupado → cambiar mapping en docker-compose.yml
- Imagen grande → optimizar capas, usar .dockerignore
- `docker-run` lento → cache de capas, --no-cache para rebuild forzado
