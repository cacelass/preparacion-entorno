# {{ project_name }}

Las reglas de este proyecto viven en `AGENTS.md` — el estándar que comparten
todos los asistentes. Este fichero solo existe porque Claude Code lee
`CLAUDE.md`; **no dupliques nada aquí**, o las dos copias divergirán.

@AGENTS.md

## Lo mínimo que no puedes saltarte

1. **Ejecuta `./init.sh` antes de tocar nada.** Si sale `ENTORNO BLOQUEADO`,
   para y repórtalo. No se implementa encima de un proyecto roto.
2. **Ninguna feature se cierra sin la puerta en verde y evidencia real.**
   Lo aplica `harness finish` en código: rechaza si `init.sh` falla o si no le
   pasas la salida del comando que lo demuestra.
3. **No edites `featureslist.json` ni `progress/` a mano.** Su dueño es el
   agente `harness`.

```bash
uv run python -m agents --json run harness next     # ¿qué toca?
uv run python -m agents --json run harness gate     # ¿se puede trabajar?
uv run python -m agents --json ask "<lo que sea>"   # ruteo automático
```

## Subagentes

En `.claude/agents/`: `lider` (dirige el ciclo), `explorer` (investiga en solo
lectura), `implementer` (escribe código y tests), `reviewer` (aprueba o
rechaza). Se generan desde `.opencode/agents/` con `make assistants-sync`, así
que **no los edites en `.claude/`** — se sobrescriben. Edita el original.

## Commits

Solo cuando se pidan explícitamente. Conventional Commits (`feat:`, `fix:`,
`chore:`, `docs:`, `refactor:`).
