# RAG Workflow — Índice local del proyecto

## Pipeline
```
rag index  →  rag search  →  (uso en agentes)
```

## Pasos

| Paso | Comando | Qué hace | Agente |
|------|---------|----------|--------|
| Index | `run rag index` | Escanea código, prompts, `docs/` (incluye `docs/vault/` y el corpus `docs/knowledge/`), la memoria del arnés; trocea, embebe y guarda en ChromaDB. Incremental | `rag` |
| Rebuild | `run rag index --rebuild` | Tira el índice y lo reconstruye. Obligatorio al cambiar de embedder | `rag` |
| Buscar | `run rag search --query "..."` | Búsqueda híbrida: vector + BM25 léxico fundidos con RRF | `rag` |
| Buscar corpus | `run rag search --query "..." --file_type knowledge` | Solo dentro del corpus de conocimiento profundo | `rag` |
| URL ext | `run rag index_urls --urls '["..."]'` | Indexa docs externas; GitHub/SO/arXiv se extraen con estructura (título, código, enlaces), el resto HTML→texto | `rag` |
| Mantener corpus | `run rag refresh --dry-run` | Informe: papers nuevos por topic + fuentes con versión más nueva. No escribe nada | `rag` |
| Actualizar corpus | `run rag refresh` | Descarga los papers nuevos a `docs/knowledge/papers/` (HTML o PDF→markitdown), actualiza `sources.json` y reindexa | `rag` |
| Estado | `run rag status` | Fragmentos, fuentes y embedder activo | `rag` |

## Paths
- `.rag-index/` — base de datos vectorial ChromaDB (gitignored)

## Qué entra en el índice
- El paquete del proyecto, `api/`, `chat/`, `tools/` y `agents/`
- Prompts de agentes y `docs/` (fichas raíz, `docs/source/` de Sphinx)
- El corpus de conocimiento profundo `docs/knowledge/` (incluidos los papers
  descargados por `rag refresh` en `docs/knowledge/papers/`), etiquetado como
  `file_type: knowledge`
- El vault Obsidian `docs/vault/`, etiquetado como `file_type: vault`
- La memoria del arnés: `harness/progress/` y `harness/featureslist.json` (aplanado a markdown)
- README, AGENTS.md, CHANGELOG.md, CONTRIBUTING.md

Quedan fuera los tests y los directorios de caché. Cada chunk de código lleva su
ruta —y su clase, si es un método— como cabecera, para que `def fit()` no sea un
fragmento anónimo; cada sección de markdown arrastra los títulos de sus
ancestros.

## El corpus de conocimiento (`docs/knowledge/`)

Teoría profunda (matemáticas, estadística, probabilidad, matrices, algoritmos
y su aplicación, e ingeniería) que el `lider` consulta antes de aconsejar. Se
consulta con `--file_type knowledge` y se mantiene con `rag refresh`, que lee
`docs/knowledge/sources.json`, verifica cada fuente contra arXiv y descarga los
papers nuevos a `docs/knowledge/papers/`. `index.md` es el mapa; `sources.md`
el registro humano; `sources.json` el registro máquina que `refresh` actualiza.

## El corpus sigue al objetivo

`rag refresh` itera los topics de `sources.json`; por sí solos son genéricos
(pca-svd, transformers...). Para que el corpus crezca hacia la pregunta del
proyecto, tras cerrar `SCOPE-001` el `lider` deriva topics desde
`references/00-objetivo.md` y los pasa a `rag refresh --topics "..."` (primero
en `--dry-run`):

```bash
uv run python -m agents --json run rag refresh --dry-run --topics "causality, uplift"
uv run python -m agents --json run rag refresh --from-objective --dry-run
```

`--from-objective` lee el objetivo si existe e incluye su pregunta como contexto
en el informe; si falta el fichero (SCOPE-001 sin cerrar), avisa y sigue con los
topics del registro. Los topics nuevos que aporten valor se añaden a
`sources.json` (decisión del `lider`, ver `KNOW-001`).

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
