# Doc Agent — Documentación unificada del proyecto

Combina 3 fuentes de conocimiento: **graphify** (grafo estructural), **RAG** (búsqueda semántica vectorial), **vault** (notas Obsidian).

## Acciones

| Acción | Parámetros | Descripción |
|--------|-----------|-------------|
| `search` | `--query`, `--sources` (all/graph/rag/vault) | Busca en las fuentes indicadas |
| `graph_query` | `--question` | Consulta estructural al grafo graphify |
| `rag_search` | `--query`, `--top-k` | Búsqueda semántica vía ChromaDB |
| `vault_grep` | `--pattern` | Búsqueda textual directa en vault markdown |
| `index` | — | Construye grafo graphify + índice RAG en un paso |
| `status` | — | Estado de cada fuente |

## Fuentes

- **Graphify**: relaciones estructurales entre módulos, dependencias, nodos del proyecto
- **RAG**: embeddings semánticos de código, prompts, docs y vault (requiere chromadb)
- **Vault**: notas markdown de Obsidian generadas por `knowledge build`

## Integración

- `doc search` es el punto de entrada único para preguntas sobre el proyecto
- Cuando un agente necesita contexto, puede delegar a `doc` en vez de consultar cada fuente por separado
- El subagente `doc` en opencode permite chatear directamente con la documentación del proyecto
