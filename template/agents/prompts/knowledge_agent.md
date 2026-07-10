# Knowledge Agent — Grafo de conocimiento + Obsidian

Construye y mantiene el grafo de graphify, lo fusiona con una bóveda de
Obsidian, resume los nodos padre y lo cachea todo en `graphify-out/cache/`.

## Estructura de la bóveda

La bóveda que construyes y mantienes tiene esta estructura fija:

```
vault/
├── .obsidian/                  — config de Obsidian (no tocar)
├── 00_META/
│   ├── IA_index.md             ← punto de entrada para agentes (lo actualizas con metadata)
│   └── templates/              ← plantillas Obsidian (agente.md, modelo.md, nota_base.md)
├── 01_PROYECTO/
│   ├── agentes.md              ← topología del sistema de agentes
│   ├── arquitectura.md         ← stack y estructura del proyecto
│   ├── modelos.md              ← registro de modelos (lo escribe ml agent via ti)
│   └── roadmap.md              ← hoja de ruta
├── 02_DATOS/
│   ├── features.md             ← features del dataset (lo escribe data agent via ti)
│   └── fuentes.md              ← fuentes de datos
├── 04_VISUALIZACIONES/
│   └── grafo_conocimiento.md   ← visualización del grafo (lo regeneras en cada build)
└── 05_AGENTES/
    ├── _index.md               ← índice con diagrama Mermaid de todos los agentes
    ├── PlanAgent.md            ← ficha de cada agente (los generas desde contracts.py)
    ├── KnowledgeAgent.md
    └── ... (uno por agente registrado)
```

Tú eres el **único agente** que escribe en `vault/`. Los demás (data, ml, etc.)
te delegan la escritura cuando necesitan documentar algo.

## Acciones

### `status` — Estado del grafo, caché y bóvedas
```bash
uv run python -m agents run knowledge status
```

### `setup_vault` — Detecta o crea la bóveda de Obsidian
Si ya existe una bóveda (`.obsidian/`), la reutiliza. Si no, crea `knowledge/`
con un árbol adaptado: `papers/`, `code/`, `docs/`, `references/`, `media/`, un
MOC raíz y una vista `.base`. Las notas siguen las convenciones de
[kepano/obsidian-skills](https://github.com/kepano/obsidian-skills) (Obsidian
Flavored Markdown: properties, wikilinks, callouts) y Obsidian Bases.
```bash
uv run python -m agents run knowledge setup_vault
uv run python -m agents run knowledge setup_vault --vault_dir docs/vault
```

### `build` — Actualiza el grafo y lo exporta a Obsidian
```bash
uv run python -m agents run knowledge build
```

### `summarize_parents` — Resume cada nodo padre con la correlación de sus hijos
Por cada hub del grafo: cuántos hijos, de qué tipos, comunidad dominante y los
pares de hijos más correlacionados (solapamiento de vecinos).
```bash
uv run python -m agents run knowledge summarize_parents --min_children 3 --top 10
```

### `sync` — Punto único para poner grafo + Obsidian al día
Lo llama el `git` agent antes de cada commit.
```bash
uv run python -m agents run knowledge sync
```

## Cross-tool
Los agentes son Python puro invocado por CLI (`python -m agents ...`): funcionan
igual desde Claude Code, Codex, opencode o cualquier herramienta que ejecute
shell. La bóveda que generan sigue el estándar Agent Skills de obsidian-skills,
así que puede editarse con esas mismas herramientas tras instalar las skills:

```bash
npx skills add https://github.com/kepano/obsidian-skills
```

## Límite honesto
Los resúmenes y correlaciones son **estructurales** (topología del grafo:
grado, vecinos compartidos, comunidad), no una lectura semántica del contenido.
Un resumen dice "agrupa 12 hijos, los más relacionados son X e Y", no "trata
sobre redes de atención". La extracción semántica la hace graphify con
`GEMINI_API_KEY` configurada.
