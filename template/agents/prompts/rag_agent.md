# RAG Agent — Búsqueda semántica local

Indexa la documentación del proyecto (código, prompts, docs, vault) con
ChromaDB + embeddings ONNX y busca en lenguaje natural. Sin API key.

## Acciones

### `index` — Construye/actualiza el índice semántico
Escanea código, prompts, docs, vault, README, AGENTS.md y CHANGELOG.md.
Incremental: solo indexa fragmentos nuevos.
```bash
uv run python -m agents run rag index
```

### `search` — Busca en lenguaje natural
```bash
uv run python -m agents run rag search --query "cómo se entrena el modelo?"
uv run python -m agents run rag search --query "parámetros del GradientBoosting" --top_k 5
```

### `index_urls` — Indexa documentación externa
Útil para docs de librerías, tutoriales, etc.
```bash
uv run python -m agents run rag index_urls --urls '["https://docs.pola.rs/api/python/stable/"]'
```

### `status` — Estado del índice
```bash
uv run python -m agents run rag status
```

## Notas
- Requiere: `uv sync --extra rag` (chromadb)
- Los índices viven en `.rag-index/` (gitignored)
- Los embeddings son locales (all-MiniLM-L6-v2 via ONNX), sin datos externos
