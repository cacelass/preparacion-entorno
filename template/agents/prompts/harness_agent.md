# harness — dueño del backlog y del progreso

Ejecuta la parte mecánica del arnés. No decide nada: los agentes markdown
(`lider`, `explorer`, `implementer`, `reviewer`) razonan, este escribe.

## Acciones

| Acción | Qué hace |
|--------|----------|
| `status` | Recuento del backlog + qué está in_progress + qué es elegible |
| `next` | La feature que toca (in_progress, o la primera con deps en done) |
| `start --id <ID>` | Abre la feature y vuelca sus criterios en `progress/current.md` |
| `gate [--quick true]` | Ejecuta `./init.sh` y devuelve el veredicto estructurado |
| `finish --id <ID> --evidence "<salida real>"` | Cierra la feature y escribe el histórico |
| `block --id <ID> --reason "<motivo>"` | Marca bloqueada |
| `record --agent <a> --id <ID> --content "<informe>"` | Guarda `progress/<a>-<ID>.md` |
| `add --id <ID> --title "<t>" --criteria "a;b;c"` | Añade feature al backlog |

```bash
uv run python -m agents --json run harness next
uv run python -m agents --json run harness start --id DATA-001
uv run python -m agents --json run harness gate
uv run python -m agents --json run harness finish --id DATA-001 --evidence "$(make test 2>&1 | tail -5)"
```

## Lo que rechaza

- **`finish` sin `./init.sh` en verde** → `success=false`. La regla del arnés es
  código, no un consejo.
- **`finish` sin `evidence`** → devuelve `needs`. Una afirmación no es evidencia.
- **`start` con otra feature abierta** → una cosa a la vez.
- **`start` con `depends_on` sin cerrar** → primero las dependencias.
- **`add` con `depends_on` inexistente** → el backlog no se corrompe.

## Por qué existe

Editar JSON a mano desde un prompt se rompe: comas, ids duplicados, estados
inventados, un `done` que nadie verificó. Este agente hace esas operaciones
de forma determinista y es el único dueño de `featureslist.json` y `progress/`.

Ver el ciclo completo: `skill harness_workflow`.
