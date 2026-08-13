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
Escanea el paquete principal, `api/`, `chat/`, `tools/`,
`agents/`, los prompts, `docs/` (fichas raíz, `docs/source/` de Sphinx, el
vault `docs/vault/` y el corpus `docs/knowledge/`), `harness/progress/`,
`harness/featureslist.json`, README, AGENTS.md y CHANGELOG.md.

Incremental **por fichero y por huella de contenido**: lo que no ha cambiado no
se vuelve a embeber, lo que cambió se reemplaza y lo que se borró desaparece del
índice. Sin eso, el índice acumulaba versiones obsoletas de `harness/progress/` a cada
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
su contenido anterior en vez de duplicarlo. Para GitHub, Stack Overflow y arXiv
se usa un extractor específico que devuelve markdown estructurado (título,
secciones, bloques de código, enlaces) en vez de HTML plano — lo que el RAG
puede citar sin perder la fuente.
```bash
uv run python -m agents run rag index_urls --urls '["https://docs.pola.rs/api/python/stable/"]'
uv run python -m agents run rag index_urls --urls '["https://github.com/cacelass/dskit"]'
uv run python -m agents run rag index_urls --urls '["https://stackoverflow.com/questions/..."]'
uv run python -m agents run rag index_urls --urls '["https://arxiv.org/abs/1412.6980"]'
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

<!-- BEGIN AUTOGEN — lo regenera `make prompts-sync`; no lo edites a mano -->

## Acciones

| Acción | Argumentos |
|--------|------------|
| `run rag index` | `--rebuild` |
| `run rag index_urls` | `--urls` (obligatorio) |
| `run rag search` | `--query` (obligatorio) · `--top_k`, `--hybrid`, `--min_score`, `--file_type`, `--source`, `--max_per_source`, `--expand` |
| `run rag status` | — |
| `run rag evaluate` | `--top_k` |

## Límites

**Rol.** RAG local: indexa código, prompts, docs/ (incl. vault y corpus de conocimiento), la memoria del arnés y URLs externas; busca en lenguaje natural fundiendo similitud vectorial (ChromaDB) con BM25 léxico.

**No hace:**
- construir o modificar el grafo graphify → knowledge
- buscar papers académicos nuevos → research
- ejecutar código ni modificar archivos del proyecto

**Necesita que le den:** que exista un índice (ejecutar 'rag index' primero)

**Escribe en (nadie más toca esto):** .rag-index/ (índice vectorial ChromaDB, gitignored)

**Se apoya en:** knowledge, doc, plan

<!-- END AUTOGEN -->
