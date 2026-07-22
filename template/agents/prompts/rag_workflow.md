# RAG Workflow — Índice semántico del proyecto

## Pipeline
```
rag index  →  rag search  →  (uso en agentes)
```

## Pasos

| Paso | Comando | Qué hace | Agente |
|------|---------|----------|--------|
| Index | `run rag index` | Escanea código + prompts + docs + vault, chunking, embedding, almacena en ChromaDB | `rag` |
| Buscar | `run rag search --query "..."` | Búsqueda semántica en lenguaje natural | `rag` |
| URL ext | `run rag index_urls --urls '["..."]'` | Indexa docs de librerías externas | `rag` |
| Estado | `run rag status` | Muestra stats del índice | `rag` |

## Paths
- `.rag-index/` — base de datos vectorial ChromaDB (gitignored)

## Integración con otros agentes
- `plan` puede consultar `rag search` antes de planificar para encontrar documentación relevante
- `docsearch` busca en el grafo graphify; `rag` es el complemento semántico local
- Los chunks incluyen código (docstrings + firmas), prompts de agentes, docs y vault

## Dependencias
- `chromadb>=1.10` (instalar con `uv sync --extra rag`)
- Modelo ONNX `all-MiniLM-L6-v2` (descarga automática en primera ejecución, ~23MB)
