# RAG Agent — Búsqueda local híbrida

Indexa la documentación y el código del proyecto con ChromaDB y busca en
lenguaje natural. Sin API key, offline.

La búsqueda es **híbrida**: funde el ranking vectorial con un BM25 léxico
(stdlib) mediante Reciprocal Rank Fusion. El embedder por defecto está entrenado
en inglés y este proyecto se documenta en español, así que buena parte de la
señal útil es literal (`train_model`, `drift`, `GradientBoosting`) y el vector
solo no la ve.

## Acciones

### `index` — Construye/actualiza el índice
Escanea el paquete principal, `api/`, `chat/`, `monitoring/`, `tuning/`,
`agents/`, los prompts, `docs/`, `vault/`, `progress/`, `featureslist.json`,
README, AGENTS.md y CHANGELOG.md.

Incremental **por fichero y por huella de contenido**: lo que no ha cambiado no
se vuelve a embeber, lo que cambió se reemplaza y lo que se borró desaparece del
índice. Sin eso, el índice acumulaba versiones obsoletas de `progress/` a cada
feature cerrada.

```bash
uv run python -m agents run rag index
uv run python -m agents run rag index --rebuild   # tira el índice y empieza de cero
```

### `search` — Busca en lenguaje natural
```bash
uv run python -m agents run rag search --query "cómo se entrena el modelo?"
uv run python -m agents run rag search --query "parámetros del GradientBoosting" --top_k 5
uv run python -m agents run rag search --query "drift" --min_score 0.35   # filtra ruido
uv run python -m agents run rag search --query "drift" --hybrid false     # solo vector
```

Cada resultado trae `score` (similitud coseno) y `match` (`vector` o `lexico`,
según qué rama lo encontró).

### `index_urls` — Indexa documentación externa
El HTML se convierte a texto antes de indexar, y reindexar una URL **reemplaza**
su contenido anterior en vez de duplicarlo.
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
- Embeddings locales, sin datos externos
- **Los chunks se cortan a 1.000 caracteres**: el embedder por defecto trunca a
  256 tokens y todo lo que pase de ahí se guardaría en la base sin entrar en el
  vector — indexado e irrecuperable.

## Embedder multilingüe (opcional)

Por defecto se usa `all-MiniLM-L6-v2` (ONNX, sin dependencias extra) que está
entrenado **en inglés**. Para embeddings que entiendan español de verdad:

```bash
uv sync --extra rag_multilingual        # arrastra sentence-transformers + torch
export DSKIT_RAG_EMBEDDER=multilingual
uv run python -m agents run rag index --rebuild
```

El `--rebuild` no es opcional: los dos modelos dan vectores de 384 dimensiones,
así que mezclarlos no daría error de ChromaDB — daría resultados sin sentido. El
embedder queda grabado en los metadatos de la colección y el agente rechaza
buscar si detecta el desajuste.
