# Orchestrator — Ruteo automático de agentes Python

Tu trabajo es decidir qué agente de `agents/agents/` (o `agents/external/`) debe
atender una petición, no resolverla tú mismo con conocimiento general. El sistema
usa scoring por keywords: `BaseAgent.can_handle(query)` puntúa cada agente según
coincidencia de palabras clave.

## Cómo funciona el ruteo

1. `Orchestrator.select_agent(query)` → puntúa los {% if use_rag %}29{% else %}28{% endif %} agentes, elige el de mayor score
2. Si el score máximo es < `MIN_CONFIDENCE (0.15)`, no selecciona ninguno
3. `Orchestrator.dispatch(query)` → selecciona agente + adivina acción + ejecuta
4. Si la acción necesita argumentos obligatorios, devuelve `needs=[...]` en vez de inventar

## Reglas

- Si ningún agente alcanza 0.15, dilo explícitamente. No fuerces un falso positivo.
- Las `capabilities` de cada agente son keywords. Coincidencia por palabra completa (`\b`).
- Si un agente devuelve `needs`, es porque falta info. Pregunta al usuario.
- El gateway opencode (subagente `orquestador`) es quien invoca este sistema.
  Tú (skill) eres documentación de contexto — no tomes decisiones de routing.

## Agentes ({% if use_rag %}29{% else %}28{% endif %})

Ver `agents/agents/` o ejecutar `uv run python -m agents list`.
