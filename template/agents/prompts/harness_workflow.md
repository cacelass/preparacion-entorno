# Harness Workflow — Ciclo del arnés

Capa que gobierna a las demás: decide **qué** se hace y **cuándo está hecho**.
Los workflows de dominio (`data`, `ml`, `dev`...) dicen **cómo**.

## Pipeline
```
init.sh → harness/progress/ → harness/featureslist.json → explorer → implementer → reviewer → done
   │                                                                      │
   └────────────────── si init.sh falla: PARAR ───────────────────────────┘
```

## Paso a paso

| Paso | Comando | Quién | Verificación |
|------|---------|-------|-------------|
| Puerta | `run harness gate` (o `make init`) | `harness` | `success=true` = se puede trabajar |
| Elegir | `run harness next` | `harness` | retoma lo abierto o la primera con deps en `done` |
| Abrir | `run harness start --id <ID>` | `harness` | criterios volcados en `current.md` |
| Investigar | subagente `explorer` (solo lectura) | `explorer` | `run harness record --agent explorer` |
| Descomponer | `run plan brief --text "<feature>"` | `plan` | orden de trabajo con pasos y agentes |
| Implementar | subagente `implementer` | `implementer` | código + tests + `harness record` |
| Revisar | subagente `reviewer` | `reviewer` | `harness gate` en verde + criterios uno a uno |
| Cerrar | `run harness finish --id <ID> --evidence "..."` | `harness` | **rechaza** si la puerta falla o no hay evidencia |
| Commit del cierre | `run git commit_feature --id <ID> --title "..." [--dry-run true]` | `git` | bump + CHANGELOG + commit, con confirmación del usuario |

El `lider` decide en qué orden pasa todo esto; `harness` lo ejecuta. Ningún
agente edita `harness/featureslist.json` ni `harness/progress/` a mano.

## Las tres reglas

1. **Nada se cierra sin `./init.sh` en verde.** Es la única definición de «hecho»
   — y la aplica `harness finish` en código, no un prompt.
2. **Evidencia, no afirmaciones.** Cada criterio se cierra con la salida real del
   comando que lo prueba. `finish` sin `--evidence` devuelve `needs`.
3. **Todo subagente registra con `harness record` antes de devolver el control.**
   Lo que solo vive en la ventana de contexto se pierde.

## El cierre (README + versión + commit)

`harness finish` marca la feature `done`; `git commit_feature` cierra el ciclo:

- `--dry-run true` devuelve la propuesta (siguiente versión patch, mensaje
  `feat(<id>): <título>`, ficheros que entrarían) **sin escribir nada**.
- Solo tras la confirmación del usuario se ejecuta sin `--dry-run`: bump de
  versión, entrada en CHANGELOG y commit. Sin tag, sin push.

## Contexto: qué NO hacer

- No heredes el contexto del líder a los subagentes. Pásales el ID de la
  feature, sus criterios y las rutas. Nada más.
- No releas el repositorio en cada subagente. `harness/progress/` existe justo para eso.
- No pases el contenido de un informe de subagente a otro: pasa la **ruta**.

## Protocolo §1 — informes compactos de subagente (ahorro de tokens)

Los subagentes reportan en prosa larga que el siguiente agente relee entera.
El protocolo §1 factoriza el informe a sus dimensiones mínimas — entidades,
estado, relaciones, cambios y certeza — en una línea JSON que `harness record`
guarda como frontmatter. El siguiente agente lee el packet; la prosa queda
debajo por si hace falta. Es un convenio de prompt, no un esquema nuevo: se
induce con ejemplos, sin entrenar nada.

Tres ejemplos, uno por rol del arnés:

```
EXPLORER (investigación, solo lectura)
{"§":1,"E":{"X":["data/raw/clientes.csv","dataset"]},
 "S":{"X.filas":"2.4M","X.nulos":{"ingresos":0.31}},
 "R":["X→features:fuente"],
 "Δ":["X.estado:nuevo→descargado@EDA-001"],
 "μ":{"rol":"explorer","cert":0.9}}
= "clientes.csv (2.4M filas) es la fuente de las features; ingresos tiene un
   31% de nulos. Descargado para EDA-001. Alta confianza."

IMPLEMENTER (implementación con evidencia)
{"§":1,"E":{"M":["src/model.py","feature"]},
 "S":{"M.firma":"predict(df)->df","tests":14},
 "R":["M→data/raw/clientes.csv:consume"],
 "Δ":["M.nuevo→implementado@FEAT-007","tests:0→14@FEAT-007"],
 "μ":{"rol":"implementer","cert":0.95,"evidencia":"pytest: 14 passed"}}
= "src/model.py implementado con 14 tests en verde (pytest 14 passed).
   Consume clientes.csv."

REVIEWER (veredicto)
{"§":1,"E":{"F":["FEAT-007","feature"]},
 "S":{"F.criterios":3,"F.cumplidos":2},
 "R":[],
 "Δ":["F.estado:in_progress→reviewed@FEAT-007"],
 "μ":{"rol":"reviewer","cert":0.55,"veredicto":"rechazado",
      "por_que":"falta cubrir el criterio 2 (caso vacío)"}}
= "Revisión de FEAT-007: 2/3 criterios cumplidos. Rechazado — falta el caso
   vacío. Certeza 0.55: el implementer acertó el grueso, pero no el todo."
```

Reglas:
- `E` (qué entidades tocó), `S` (estado resultante), `R` (relaciones nuevas),
  `Δ` (qué cambió y cuándo), `μ` (rol, `cert` 0..1, y `veredicto`/`evidencia`
  si aplica).
- `harness record --packet '<json>'` valida el JSON y lo guarda como frontmatter;
  la prosa sigue siendo el `--content`. Ambos conviven.
- El reviewer SIEMPRE declara `cert` en su packet: `harness finish` aplica la
  rúbrica de la puerta (`agents/rubric.py`, GATE-1..4) y rechaza un `done` si
  la certeza del reviewer quedó por debajo del umbral (`UMBRAL_CERTEZA`), si
  la evidencia no parece real o si el reviewer rechazó la feature (GATE-3).
- Un packet sin `cert` no es un fallo (la prosa sigue existiendo), pero el
  arnés lo trata como confianza plena — como siempre fue.

## Cómo engancha con los agentes Python

El arnés decide y verifica; el trabajo determinista lo hacen los agentes:

| Necesidad del arnés | Agente | Acción |
|---------------------|--------|--------|
| Backlog y progreso (todo cambio de estado) | `harness` | `status`, `next`, `start`, `finish`, `block`, `record`, `add`, `gate` |
| Descomponer una feature en pasos | `plan` | `brief`, `answer`, `execute` |
| Comprobar que la suite pasa | `test` | `run_tests`, `coverage_summary` |
| Revisar el código de una feature | `review` | `review_package` |
| Diagnóstico antes de abrir trabajo | `doctor` | `python -m agents doctor` |
| Saber si el equipo va bien | `audit` | `suggest` |
| Contexto de sesiones anteriores | `memory` | `status`, `search` |
| Buscar dónde está algo | `doc` | `search` |
| Cerrar la feature (versión + changelog + commit) | `git` | `commit_feature` |

**Perfiles reducidos.** En `minimo`/`estandar` algunos agentes no existen
(`audit`, `supervisor`, `research`, `installer`; y `api`/`docker`/`mlflow`/
`knowledge`/`rag`/`mutation` si su extra está apagado). Antes de delegar,
confirma con `uv run python -m agents list`. Un agente ausente devuelve
`success=false` — no es un fallo, es un proyecto sin ese extra.

```bash
uv run python -m agents --json run plan brief --text "<descripción de la feature>"
uv run python -m agents --json run test run_tests
uv run python -m agents --json run review review_package
```

## Ficheros

| Fichero | Dueño | Qué es |
|---------|-------|--------|
| `AGENTS.md` | humano | Protocolo y reglas del juego |
| `init.sh` | humano + `reviewer` | La puerta. El reviewer puede endurecerla |
| `harness/featureslist.json` | `harness` | Backlog con criterios de aceptación |
| `harness/progress/current.md` | `harness` | Feature en curso |
| `harness/progress/history.md` | `harness` | Append-only de lo cerrado |
| `harness/progress/<agente>-<ID>.md` | `harness` (vía `record`) | Resultado de una ejecución |
| `.opencode/agents/*.md` | cada agente | Su propia definición (automejorable) |

## Automejora

Si el mismo fallo se cuela dos veces: si es automatizable va a `init.sh`; si es
regla del proyecto va a `AGENTS.md`; si es criterio de revisión va a
`.opencode/agents/reviewer.md`. Deja constancia en `harness/progress/history.md`.

## Comandos

```bash
make init            # ./init.sh — verificación completa
make harness-check   # ./init.sh --quick — sin tests
make backlog         # estado de harness/featureslist.json
./init.sh --json     # salida estructurada para agentes
uv run python -m agents.evals.runner --harness   # valida las piezas del arnés
{% if use_rag %}make index-rag       # mete harness/progress/ e histórico en el índice semántico
{% endif %}```
