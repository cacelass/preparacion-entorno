# Memory

Instrucciones, correcciones y preferencias guardadas por el usuario para recordar en futuras sesiones.

---

## Commits

- Usar mensajes convencionales (feat:, fix:, chore:, docs:)
- `ecommits` no está instalado en el entorno — hacer commits manuales
- Taggear tras commits grandes: `git tag -a vX.Y.Z -m "mensaje"`

## Memory Agent (Jul 2026)

- Implementado `memory_agent.py` (proactive memory pattern, arXiv 2607.08716)
- Memory bank en `agents/workspace/memory/bank.json` con tres kinds: facts, state, traces
- `make skills` instala prompts como skills en `.opencode/skills/`
- `make agents-memory` muestra estado de la memoria de agentes
- Usar `npx autoskills -y` para skills del ecosistema (Node.js)

## Self-maintenance (Jul 2026)

- dskit tiene su propio `Makefile` en la raíz para auto-mantenimiento
- `make setup` / `make update` / `make skills` desde la raíz de dskit
- `make recommended-tools` en proyectos generados muestra herramientas del ecosistema
- `make recommended-all` en proyectos generados como alias
- `uv pip install eticas-audit` para fairness/bias (ITACA)
- `npm install -g @synsci/openscience` para AI workbench científico

