---
tags:
  - agente
  - conocimiento
---
# Knowledge Agent

> Dueño del grafo de conocimiento y la bóveda Obsidian: los construye y mantiene al día.

## Contrato

- **Rol:** Constructor y mantenedor del grafo y vault
- **Capacidades:** construir/reconstruir el grafo (graphify); crear el vault Obsidian; resumir nodos padre; sync
- **Límites:** no busca papers nuevos (→ research); no navega el grafo (→ docsearch)
- **Recursos:** `graphify-out/` (construcción y sync); vault Obsidian del proyecto
- **Colabora con:** docsearch, research

## Responsabilidades

1. Ejecutar graphify para construir el grafo de conocimiento
2. Generar y mantener el vault Obsidian del proyecto
3. Resumir nodos padre del grafo
4. Sincronizar cambios

## Archivos

- `agents/agents/knowledge_agent.py`
- `agents/contracts.py:` `CONTRACTS["knowledge"]`
