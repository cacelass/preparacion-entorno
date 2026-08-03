# Supervisor Agent — Workers que compiten

Hace que varios workers trabajen **por separado** sobre la misma tarea, prueba
cada propuesta con una métrica, elige la mejor y la pule.

## Acciones

### `research` — Competición de búsqueda de papers
Lanza el `research` agent con cada backend (arXiv, OpenAlex) en paralelo, puntúa
cada propuesta (relevancia + cobertura + volumen), elige la ganadora y pule
fusionando lo mejor de ambas.
```bash
uv run python -m agents run supervisor research --max_results 10
uv run python -m agents run supervisor research --backends '["arxiv","openalex"]'
```

### `compete` — Competición genérica
Enfrenta candidatos arbitrarios; cada uno es `{agent, action, kwargs, label}`.
```bash
uv run python -m agents run supervisor compete --candidates \
  '[{"agent":"research","action":"find_papers","kwargs":{"backend":"arxiv"},"label":"arxiv"},
    {"agent":"research","action":"find_papers","kwargs":{"backend":"openalex"},"label":"openalex"}]'
```

## Cómo puntúa (determinista, no un juez LLM)
- **research**: `0.5·relevancia_media + 0.4·cobertura_keywords + 0.1·volumen`.
- **compete**: premia éxito + riqueza de `data`, penaliza warnings.

La métrica es el punto de extensión si quieres un arbitraje más sofisticado —
el resto del sistema no cambia.

<!-- BEGIN AUTOGEN — lo regenera `make prompts-sync`; no lo edites a mano -->

## Acciones

| Acción | Argumentos |
|--------|------------|
| `run supervisor research` | `--max_results`, `--top_keywords`, `--backends` |
| `run supervisor compete` | `--candidates` (obligatorio) · `--parallel` |
| `run supervisor synthesize` | `--perspectives` (obligatorio) · `--parallel`, `--question` |

## Límites

**Rol.** Coordina workers en paralelo: los pone a COMPETIR y arbitra, o los abre en abanico y SINTETIZA.

**No hace:**
- orquestar un encargo secuencial paso a paso → plan (él delega a dueños, no arbitra)
- hacer el trabajo de los workers él mismo — solo coordina y evalúa
- interpretar lo que sintetiza — agrupa hechos; razonar sobre ellos es de la capa de arriba

**Necesita que le den:** la tarea a poner en competición y el criterio de evaluación

**Se apoya en:** research

<!-- END AUTOGEN -->
