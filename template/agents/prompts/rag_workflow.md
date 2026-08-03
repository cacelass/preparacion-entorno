# RAG Workflow — Índice local del proyecto

## Pipeline
```
rag index  →  rag search  →  (uso en agentes)
```

## Pasos

| Paso | Comando | Qué hace | Agente |
|------|---------|----------|--------|
| Index | `run rag index` | Escanea código, prompts, docs, vault y la memoria del arnés; trocea, embebe y guarda en ChromaDB. Incremental | `rag` |
| Rebuild | `run rag index --rebuild` | Tira el índice y lo reconstruye. Obligatorio al cambiar de embedder | `rag` |
| Buscar | `run rag search --query "..."` | Búsqueda híbrida: vector + BM25 léxico fundidos con RRF | `rag` |
| URL ext | `run rag index_urls --urls '["..."]'` | Indexa docs de librerías externas (HTML → texto) | `rag` |
| Estado | `run rag status` | Fragmentos, fuentes y embedder activo | `rag` |

## Paths
- `.rag-index/` — base de datos vectorial ChromaDB (gitignored)

## Qué entra en el índice
- El paquete del proyecto, `api/`, `chat/`, `monitoring/`, `tuning/` y `agents/`
- Prompts de agentes, `docs/`, `vault/`
- La memoria del arnés: `harness/progress/` y `harness/featureslist.json` (aplanado a markdown)
- README, AGENTS.md, CHANGELOG.md, CONTRIBUTING.md

Quedan fuera los tests y los directorios de caché. Cada chunk de código lleva su
ruta —y su clase, si es un método— como cabecera, para que `def fit()` no sea un
fragmento anónimo; cada sección de markdown arrastra los títulos de sus
ancestros.

## Reindexado
Incremental **por fichero y por huella de contenido**: lo que no cambió no se
vuelve a embeber, lo que cambió se reemplaza y lo que se borró se purga. Por eso
`make index-rag` tras cerrar una feature es barato y el histórico del arnés no
acumula versiones obsoletas.

## Por qué la búsqueda es híbrida
El embedder por defecto está entrenado en inglés y este proyecto se documenta en
español, así que buena parte de la señal útil es literal (`train_model`, `drift`,
`GradientBoosting`). El ranking vectorial y el BM25 se funden con Reciprocal Rank
Fusion; cada resultado indica en `match` qué rama lo encontró.

Para embeddings multilingües de verdad: `uv sync --extra rag_multilingual`,
`export DSKIT_RAG_EMBEDDER=multilingual` y `make index-rag-rebuild`.

## Integración con otros agentes
- `plan` puede consultar `rag search` antes de planificar
- `doc search` funde este índice con el grafo graphify y el vault
- `knowledge` construye el grafo; `rag` es el índice vectorial. No se pisan

## Dependencias
- `chromadb>=0.5.0,<2.0` (instalar con `uv sync --extra rag`)
- Modelo ONNX `all-MiniLM-L6-v2` (descarga automática en la primera ejecución)
- El BM25 es stdlib: sin dependencias añadidas
