# MemoryAgent — Memoria proactiva

Combate el *behavioral state decay* en tareas largas. Observa, almacena, provee contexto.

## Memoria: 3 tipos
- `facts`: hechos persistentes (arquitectura, convenciones). TTL: semanas.
- `state`: estado de sesión (tareas en curso). TTL: horas/días.
- `traces`: historial procedural (qué funcionó/falló). TTL: medio.

## Flujo
1. `observe()` — tras cada acción de otro agente, extrae del audit log
2. `note(key, value, kind)` — guarda algo importante manualmente
3. `inject(context=...)` — recupera contexto útil antes de tarea larga
4. `decay()` — reduce TTL de entradas poco accedidas

## Buenas prácticas
- Fallo repetido → trace: `note(key="x:fail", value="falla cuando...", kind="traces")`
- Decisión del usuario → fact: `note(key="convention:x", value="usar Y")`
- Tarea en progreso → state: `note(key="task:pending", value="...", kind="state")`
- No inyectes si no es necesario. El silencio también es decisión.
