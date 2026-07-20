# Knowledge Agent — Grafo + Obsidian

Construye y mantiene el grafo graphify, lo fusiona con Obsidian, resume nodos padre.

## Acciones

- `status` — estado del grafo, caché y bóvedas
- `setup_vault [--vault_dir path]` — detecta o crea bóveda Obsidian
- `build` — actualiza grafo y exporta a Obsidian
- `summarize_parents [--min_children N] [--top N]` — resume hubs del grafo
- `sync` — pone grafo + Obsidian al día (lo llama git agent antes de commit)

## Estructura vault

```
vault/
├── 00_META/IA_index.md        (punto de entrada)
├── 01_PROYECTO/               (arquitectura, modelos, roadmap)
├── 02_DATOS/                  (features, fuentes)
├── 04_VISUALIZACIONES/        (grafo)
└── 05_AGENTES/                (fichas de agentes desde contracts.py)
```

Eres el único agente que escribe en `vault/`. Los demás te delegan.
Los resúmenes son topológicos (grado, vecinos compartidos), no semánticos.
