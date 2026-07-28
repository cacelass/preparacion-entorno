"""
agents.contracts — Contratos de rol: qué puede, qué NO puede y qué necesita cada agente.

Este archivo es la fuente única de verdad sobre los ROLES del equipo de
agentes. El código de cada agente define CÓMO hace las cosas; este archivo
define QUÉ le corresponde hacer y dónde están sus límites — en un solo lugar
legible, en vez de repartido por 20 docstrings.

Las tres reglas del equipo
--------------------------
1. **Nadie se pisa.** Cada recurso escribible del proyecto (CHANGELOG.md,
   Makefile, .github/workflows/...) tiene UN único dueño (`owns`).
   `validate_contracts()` falla si dos agentes declaran el mismo recurso —
   y hay un test que lo ejecuta en cada suite (tests/test_contracts.py).
2. **Nadie improvisa fuera de su rol.** `cannot` lista lo que el agente NO
   hace y a quién derivarlo. Si te encuentras añadiendo a un agente una
   acción que aparece en su `cannot`, estás construyendo el agente
   equivocado.
3. **Nadie inventa información.** `needs` lista lo que el agente necesita
   que le den. Si falta, la acción devuelve `AgentResult(success=False,
   needs=[...])` — pregunta, no adivina (ver `AgentResult.needs`).

Colaboración sin solaparse
--------------------------
`collaborates` documenta las relaciones reales de delegación (via
`delegate_to()` o instanciación directa con `context=self.ctx`). La regla:
un agente puede PEDIR trabajo a otro, nunca hacerlo él mismo. Ejemplo real:
`git.commit_with_changelog` delega la escritura del CHANGELOG en
`documentation` (dueño del archivo) en vez de escribirlo él.

Todo recurso no listado en ningún `owns` es de solo lectura para todos.
Además, cada agente posee implícitamente `agents/workspace/<su_nombre>/`.

El vault Obsidian como contexto compartido
------------------------------------------
El vault (`vault/`) es la memoria compartida del equipo. Cualquier agente
puede LEERLO, pero solo `knowledge` lo ESCRIBE. El vault contiene:

- `00_META/IA_index.md` — punto de entrada: metadata del proyecto, estructura
  del vault, modelo de datos. TODO agente debería leerlo al activarse.
- `01_PROYECTO/` — documentación del proyecto (agentes, arquitectura,
  modelos, roadmap). `ml` y `data` lo consultan frecuentemente.
- `02_DATOS/` — documentación de datos (features, fuentes). Dueño: `data`.
- `04_VISUALIZACIONES/grafo_conocimiento.md` — visualización del grafo.
  Dueño: `knowledge`.
- `05_AGENTES/` — fichas de cada agente con su contrato expandido. TODO
  agente tiene su ficha aquí, actualizada por `knowledge`.

Regla: si un agente necesita contexto sobre el proyecto, primero lee
`vault/00_META/IA_index.md`. Si necesita contexto sobre otro agente, lee
`vault/05_AGENTES/<Agent>.md`.

El arnés: quién razona y quién ejecuta
--------------------------------------
El arnés (ver `AGENTS.md` y `agents/prompts/harness_workflow.md`) tiene dos
capas y la división es estricta:

- **Razonan** — agentes markdown en `.opencode/agents/` (`lider`, `explorer`,
  `implementer`, `reviewer`). Deciden QUÉ feature toca, cómo implementarla y
  si el trabajo vale. No aparecen en `CONTRACTS` porque no son código: no
  tienen módulo en `agents/agents/` y `validate_contracts()` audita el registro.
- **Ejecuta** — el agente Python `harness` (abajo en este mismo diccionario).
  Es el dueño real de `featureslist.json` y `progress/`: cambiar un estado,
  escribir el histórico o ejecutar la puerta son operaciones deterministas, y
  un LLM editando JSON a mano se equivoca.

Por eso la regla del arnés no es una instrucción sino código:
`harness.finish()` rechaza cerrar una feature si `./init.sh` no pasa o si no
se aporta evidencia. No hay forma de saltársela pidiéndoselo al modelo.

Recursos del arnés que no tienen dueño en `CONTRACTS`:

- `init.sh` — dueño: el humano; `reviewer` puede endurecerlo (añadir checks).
- `.opencode/agents/*.md` — cada agente es dueño de su propia definición.

Las tres memorias del proyecto no se solapan, cada una tiene su plazo:

| Soporte | Dueño | Alcance |
|---------|-------|---------|
| `progress/` | `harness` | La feature en curso y el histórico de features |
| `agents/workspace/memory/` | `memory` | Trayectorias de ejecución de agentes |
| `vault/` | `knowledge` | Conocimiento estable del proyecto y sus datos |

Si un agente Python necesita saber en qué se está trabajando, LEE
`progress/current.md` — no lo escribe.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Contract:
    """Contrato de rol de un agente. Solo documentación estructurada + validable."""

    role: str                                  # una línea: su misión
    can: tuple[str, ...] = ()                  # qué hace (resumen honesto, no marketing)
    cannot: tuple[str, ...] = ()               # qué NO hace y a quién derivar ("X → agente y")
    needs: tuple[str, ...] = ()                # información que debe recibir (si falta: preguntar)
    owns: tuple[str, ...] = ()                 # recursos que SOLO este agente puede modificar
    collaborates: tuple[str, ...] = ()         # agentes a los que delega o que le delegan

    def as_dict(self) -> dict:
        return {
            "role": self.role,
            "can": list(self.can),
            "cannot": list(self.cannot),
            "needs": list(self.needs),
            "owns": list(self.owns),
            "collaborates": list(self.collaborates),
        }


CONTRACTS: dict[str, Contract] = {
    # ── Coordinación ─────────────────────────────────────────────────────
    "plan": Contract(
        role="Jefe de proyecto: convierte un encargo humano en una orden de trabajo, pregunta lo que falte y delega.",
        can=(
            "descomponer un encargo en pasos y asignar cada paso al agente responsable",
            "detectar qué información falta y devolver las preguntas ANTES de ejecutar nada",
            "ejecutar la orden de trabajo aprobada (via GStack) y resumir qué debe verificar el humano",
            "leer vault/00_META/IA_index.md para obtener contexto del proyecto y la topología de agentes",
            "consultar vault/05_AGENTES/<Agent>.md para decidir a quién delegar cada paso",
        ),
        cannot=(
            "ejecutar ninguna acción de dominio él mismo → siempre delega en el agente dueño",
            "inventar argumentos que no le han dado → los convierte en preguntas",
            "ejecutar una orden con preguntas sin responder",
        ),
        needs=("el encargo (brief) en lenguaje natural", "las respuestas a las preguntas que genere"),
        collaborates=("todos — es el punto de entrada que delega en el resto",),
    ),
    "audit": Contract(
        role="Auditor del equipo: mide a los demás agentes con el log de ejecuciones y propone mejoras.",
        can=(
            "informe de uso: ejecuciones, tasa de éxito y duración media por agente/acción",
            "listar los fallos recientes con su mensaje",
            "sugerir mejoras con heurísticas (acciones que fallan mucho, agentes sin uso, acciones lentas)",
        ),
        cannot=(
            "arreglar él mismo lo que detecta → doctor (diagnóstico), refactor (código), o el humano",
            "auditar llamadas directas a métodos que no pasaron por run() — quedan fuera del log",
        ),
        owns=(),
        collaborates=(),
    ),

    "supervisor": Contract(
        role="Coordina workers en COMPETICIÓN: lanza N variantes de una tarea y arbitra cuál gana.",
        can=(
            "lanzar propuestas que compiten (p. ej. búsquedas de papers en paralelo) y elegir la mejor",
        ),
        cannot=(
            "orquestar un encargo secuencial paso a paso → plan (él delega a dueños, no arbitra)",
            "hacer el trabajo de los workers él mismo — solo coordina y evalúa",
        ),
        needs=("la tarea a poner en competición y el criterio de evaluación",),
        collaborates=("research",),
    ),

    # ── Conocimiento e investigación ─────────────────────────────────────
    "knowledge": Contract(
        role="Dueño del grafo de conocimiento y la bóveda Obsidian: los construye y mantiene al día.",
        can=(
            "construir/reconstruir el grafo (graphify), crear la bóveda, resumir nodos padre, sync",
            "poblar vault/05_AGENTES/ con fichas individuales de cada agente desde contracts.py",
            "actualizar vault/04_VISUALIZACIONES/grafo_conocimiento.md tras cada build del grafo",
        ),
        cannot=(
            "buscar o navegar por el grafo → docsearch",
            "buscar papers nuevos → research (knowledge los indexa cuando ya existen)",
        ),
        owns=(
            "graphify-out/ (construcción y sync del grafo)",
            "vault/ (bóveda Obsidian del proyecto — todo vault/00_META/, 01_PROYECTO/, 04_VISUALIZACIONES/, 05_AGENTES/)",
        ),
        collaborates=("docsearch", "research"),
    ),
    "docsearch": Contract(
        role="Buscador del grafo de conocimiento: consulta, navega vecinos y poda nodos irrelevantes.",
        can=(
            "buscar en el grafo, listar vecinos/referencias, podar nodos innecesarios (con backup)",
            "buscar en vault/00_META/ y vault/05_AGENTES/ como fuente secundaria de contexto",
        ),
        cannot=(
            "construir o reconstruir el grafo → knowledge",
            "buscar fuera del grafo (papers nuevos, web) → research",
        ),
        needs=("la consulta o el nodo del que partir",),
        owns=("graphify-out/ (poda de nodos, con .bak)",),
        collaborates=("knowledge",),
    ),
    "rag": Contract(
        role="RAG semántico local: indexa código, prompts, docs, vault y URLs externas; busca en lenguaje natural con ChromaDB + embeddings ONNX.",
        can=(
            "indexar el proyecto (código, prompts, docs, vault, README, CHANGELOG) en ChromaDB",
            "indexar URLs externas (documentación de librerías, tutoriales)",
            "buscar en el índice con consultas en lenguaje natural",
            "devolver fragmentos relevantes con puntuación de similitud",
        ),
        cannot=(
            "construir o modificar el grafo graphify → knowledge",
            "buscar papers académicos nuevos → research",
            "ejecutar código ni modificar archivos del proyecto",
        ),
        needs=("que exista un índice (ejecutar 'rag index' primero)",),
        owns=(".rag-index/ (índice vectorial ChromaDB, gitignored)",),
        collaborates=("knowledge", "docsearch", "plan"),
    ),
    "research": Contract(
        role="Investigador externo: busca papers (arXiv/OpenAlex) relacionados con el proyecto. Solo lee.",
        can=("extraer keywords del proyecto, buscar papers y rankearlos (necesita internet)",),
        cannot=(
            "indexar papers en el grafo o la bóveda → knowledge",
            "decidir qué paper adoptar — presenta candidatos, el humano (o supervisor) elige",
        ),
        collaborates=("knowledge", "supervisor"),
    ),

    # ── Código y calidad ─────────────────────────────────────────────────
    "review": Contract(
        role="Revisor de código: encuentra problemas y los reporta. Solo lee, nunca modifica.",
        can=(
            "detectar funciones largas, exceso de argumentos, except desnudos, duplicación, TODO/FIXME",
        ),
        cannot=(
            "modificar código → refactor",
            "ejecutar tests → test",
            "juzgar el diseño del modelo de ML → ml",
        ),
        collaborates=("refactor",),
    ),
    "refactor": Contract(
        role="Único agente autorizado a modificar código fuente del paquete, siempre con dry_run primero.",
        can=(
            "corregir mutables por defecto, except: desnudos, añadir -> None, señalar weights_only=False",
        ),
        cannot=(
            "refactorizar sin revisión previa: dry_run=True es el modo por defecto, el humano aprueba",
            "tocar notebooks → notebook",
            "tocar el Makefile → make",
        ),
        needs=("qué archivo/paquete tocar, o confirmación para aplicar (dry_run=False)",),
        owns=("codigo fuente del paquete ({project_slug}/)",),
        collaborates=("review",),
    ),
    "test": Contract(
        role="Ejecuta la suite de tests y explica los resultados.",
        can=("correr pytest, resumir fallos y cobertura, detectar módulos sin test homónimo",),
        cannot=(
            "arreglar los tests que fallan → refactor (código) o el humano",
            "escribir tests nuevos completos — solo detecta huecos",
        ),
        collaborates=(),
    ),

    # ── Datos y ML ───────────────────────────────────────────────────────
    "data": Contract(
        role="Analista de datos: EDA y calidad de datasets. Lee data/, escribe solo en su workspace.",
        can=(
            "EDA: constantes, cardinalidad, missing, outliers, correlaciones, fuga de información",
            "documentar hallazgos en vault/02_DATOS/ (features.md, fuentes.md) via knowledge",
        ),
        cannot=(
            "modificar los datasets de data/ — los informes van a su workspace o al vault via knowledge",
            "entrenar o evaluar modelos → ml",
            "auditar figuras → graph",
        ),
        needs=("filename del dataset", "target_col para análisis de fuga/correlación con el target"),
        collaborates=("knowledge",),  # delega escritura de hallazgos al vault
    ),
    "ml": Contract(
        role="Analista de modelos entrenados: inspecciona .joblib, importancias, overfitting.",
        can=(
            "inspeccionar modelos guardados, comparar modelos, analizar estudios de Optuna",
            "documentar resultados en vault/01_PROYECTO/modelos.md via knowledge",
        ),
        cannot=(
            "entrenar modelos — eso es del pipeline (make train), no de un agente",
            "analizar datasets crudos → data",
            "consultar experimentos MLflow → mlflow",
        ),
        needs=("métricas de train/test para juzgar overfitting — no las inventa",),
        collaborates=("mlflow", "knowledge"),
    ),
    "mlflow": Contract(
        role="Consulta el tracking de experimentos MLflow (solo con use_mlflow=true).",
        can=("listar runs, encontrar el mejor por métrica, avisar si el último run empeoró",),
        cannot=(
            "borrar o modificar runs",
            "juzgar el modelo en sí → ml",
        ),
        collaborates=("ml",),
    ),
    "graph": Contract(
        role="Auditor de figuras: revisa reports/figures/ (vacías, corruptas, aspect ratio raro).",
        can=("detectar figuras vacías o sospechosas en reports/figures/",),
        cannot=("regenerar figuras — eso es del pipeline de visualización",),
        collaborates=(),
    ),
    "notebook": Contract(
        role="Único agente que toca notebooks: extrae salidas e inserta celdas markdown.",
        can=("extraer imágenes/texto de un .ipynb, insertar interpretaciones como celdas",),
        cannot=(
            "interpretar los resultados él mismo — inserta las interpretaciones que le den",
            "tocar código fuente del paquete → refactor",
        ),
        needs=("ruta del notebook", "las interpretaciones a insertar (las redacta el humano o un LLM)"),
        owns=("notebooks/",),
        collaborates=(),
    ),

    # ── Entrega y entorno ────────────────────────────────────────────────
    "git": Contract(
        role="Único agente que escribe en el historial git: commits, tags, releases.",
        can=(
            "mensajes Conventional Commits desde el diff real, changelog, resumen de PR",
            "commit_with_changelog y tag_release (flujos que encadenan documentation + git)",
        ),
        cannot=(
            "escribir CHANGELOG.md/README.md él mismo → delega en documentation (su dueño)",
            "hacer push a remotos — decisión del humano",
        ),
        needs=("la versión, para tag_release", "el mensaje, para commit si no quiere el sugerido"),
        owns=("historial git (commits, tags, ramas)",),
        collaborates=("documentation", "cicd"),
    ),
    "documentation": Contract(
        role="Dueño de la documentación: CHANGELOG.md, README.md, docs/ y la versión del proyecto.",
        can=(
            "actualizar CHANGELOG.md, detectar README ↔ Makefile desincronizados",
            "bump_version en pyproject.toml + README, generar docs Sphinx",
        ),
        cannot=(
            "hacer commit de lo que escribe → git",
            "tocar la sección de dependencias de pyproject.toml → env",
        ),
        needs=("la nueva versión, para bump_version",),
        owns=("CHANGELOG.md", "README.md", "docs/", "pyproject.toml (campo version)"),
        collaborates=("git",),
    ),
    "cicd": Contract(
        role="Dueño de los workflows de GitHub Actions del proyecto generado.",
        can=("generar y validar .github/workflows/*.yml cruzando los targets contra el Makefile real",),
        cannot=(
            "modificar el Makefile → make",
            "hacer commit del workflow → git",
        ),
        owns=(".github/workflows/",),
        collaborates=("make", "git"),
    ),
    "make": Contract(
        role="Dueño del Makefile: valida targets y la cadena del pipeline, sugiere targets nuevos.",
        can=("verificar targets, chequear pipeline → predict → train → features → data",),
        cannot=(
            "generar workflows de CI → cicd",
            "ejecutar el pipeline completo — sugiere, el humano ejecuta",
        ),
        owns=("Makefile",),
        collaborates=("cicd",),
    ),
    "env": Contract(
        role="Dueño del entorno: versión de Python, uv sync/lock, dependencias declaradas.",
        can=("verificar python, uv sync, uv lock --check, añadir dependencias con uv add",),
        cannot=(
            "juzgar si una dependencia está obsoleta o es vulnerable → dependency",
            "tocar la versión del proyecto en pyproject.toml → documentation",
        ),
        needs=("el nombre del paquete, para añadir una dependencia",),
        owns=("uv.lock", ".venv/", "pyproject.toml (dependencias)"),
        collaborates=("dependency",),
    ),
    "dependency": Contract(
        role="Vigilante de dependencias: obsolescencia y vulnerabilidades contra PyPI/OSV. Solo lee.",
        can=("detectar paquetes desactualizados y vulnerabilidades conocidas (necesita internet)",),
        cannot=(
            "actualizar o instalar nada → env (dueño de uv.lock)",
        ),
        collaborates=("env",),
    ),
    "docker": Contract(
        role="Revisor de la configuración Docker: lint de Dockerfile y docker-compose.",
        can=("lint de Dockerfile, validación de docker-compose.yml",),
        cannot=(
            "construir o ejecutar imágenes — solo análisis estático",
            "editar el Dockerfile — reporta, el humano decide",
        ),
        collaborates=(),
    ),
    "api": Contract(
        role="Revisor de la API FastAPI (solo con use_api=true): endpoints vs docs + smoke test.",
        can=("cruzar endpoints declarados contra documentados, smoke test con TestClient",),
        cannot=(
            "modificar api/main.py — reporta discrepancias",
            "desplegar la API",
        ),
        collaborates=("test",),
    ),
    "secrets": Contract(
        role="Escáner de secretos hardcodeados. Solo lee y reporta.",
        can=("escanear el proyecto con detect-secrets o un heurístico propio (más limitado, avisado)",),
        cannot=(
            "borrar o rotar secretos encontrados — decisión del humano",
        ),
        collaborates=(),
    ),
    "installer": Contract(
        role="Dueño de agents/external/: instala y valida agentes de terceros.",
        can=("instalar un agente desde git/ruta local, validar su estructura, confirmar registro",),
        cannot=(
            "garantizar que el código externo es seguro — la validación es estructural, no de seguridad",
            "instalar dependencias del agente externo → env",
        ),
        needs=("repo_url o ruta local del agente a instalar",),
        owns=("agents/external/",),
        collaborates=("env",),
    ),
    "doctor": Contract(
        role="Diagnóstico integral del proyecto: agrega las verificaciones de los demás.",
        can=("checkup completo (python, git, estructura, tests, datos, dependencias, disco)",),
        cannot=(
            "arreglar lo que encuentra por su cuenta → cada dueño (pipeline fix lo orquesta)",
        ),
        collaborates=("env", "test", "data", "dependency", "git"),
    ),
    "schedule": Contract(
        role="Experto en cron: valida, describe y calcula próximas ejecuciones. No programa nada.",
        can=("validar expresiones cron, describirlas en lenguaje natural, calcular próximas ejecuciones",),
        cannot=(
            "instalar crontabs o programar tareas reales en el sistema — solo analiza expresiones",
        ),
        needs=("la expresión cron a analizar",),
        collaborates=(),
    ),
    "memory": Contract(
        role="Memoria proactiva: observa trayectorias de agentes y mantiene un banco de memoria estructurado contra el decaimiento del estado en tareas largas.",
        can=(
            "observar el log de auditoría y extraer hechos, estado y trazas",
            "buscar recuerdos por texto o tipo (facts/state/traces)",
            "inyectar recordatorios relevantes en el contexto del agente activo",
            "tomar una instantánea del estado actual del proyecto",
            "almacenar una nota arbitraria en la memoria",
            "olvidar entradas específicas del banco de memoria",
        ),
        cannot=(
            "modificar el workspace de otros agentes — solo escribe en agents/workspace/memory/",
            "ejecutar acciones de dominio — solo observa e inyecta contexto",
        ),
        needs=("el log de auditoría para observar",),
        owns=("agents/workspace/memory/",),
        collaborates=("audit",),
    ),
    "doc": Contract(
        role="Documentación unificada: responde dónde está algo consultando grafo, índice semántico y vault.",
        can=(
            "buscar en las tres fuentes a la vez (graphify, RAG, vault) y fusionar resultados",
            "consultar el grafo estructural de graphify",
            "buscar en el vault Obsidian por texto",
            "informar de qué fuentes están disponibles y cuáles no",
        ),
        cannot=(
            "escribir documentación → eso es de 'documentation' (README, CHANGELOG) o 'knowledge' (vault)",
            "construir el grafo ni el índice → delega en 'knowledge' y 'rag'",
            "responder si ninguna fuente está disponible → lo dice, no inventa",
        ),
        needs=("la pregunta en lenguaje natural",),
        collaborates=("rag", "knowledge", "docsearch"),
    ),
    "harness": Contract(
        role="Dueño mecánico del arnés: mantiene el backlog y el progreso, y ejecuta la puerta init.sh.",
        can=(
            "leer y actualizar featureslist.json (abrir, cerrar, bloquear y añadir features)",
            "escribir progress/current.md y añadir entradas a progress/history.md",
            "guardar los informes de los subagentes en progress/<agente>-<FEATURE-ID>.md",
            "ejecutar ./init.sh y devolver el veredicto estructurado",
            "rechazar el cierre de una feature si la puerta no pasa o si no hay evidencia",
        ),
        cannot=(
            "decidir QUÉ feature toca ni cómo implementarla → eso lo razonan los agentes "
            "markdown del arnés (lider, explorer, implementer, reviewer)",
            "escribir código del producto → 'refactor' y el implementer",
            "ejecutar los tests por su cuenta → los ejecuta init.sh, o el agente 'test'",
            "cerrar una feature sin evidencia → devuelve needs, nunca la da por buena",
        ),
        needs=("el id de la feature", "la evidencia real de verificación para cerrarla"),
        owns=("featureslist.json", "progress/"),
        collaborates=("plan", "test", "review", "memory"),
    ),
}


def contract_for(agent_name: str) -> Contract | None:
    """Contrato de un agente, o None si no lo tiene (los externos pueden no tenerlo)."""
    return CONTRACTS.get(agent_name)


def validate_contracts(registered_names: set[str] | None = None) -> list[str]:
    """
    Valida la coherencia del equipo. Devuelve la lista de problemas (vacía = OK).

    - Ningún recurso puede tener dos dueños (regla 1: nadie se pisa).
    - Todo agente del núcleo registrado debe tener contrato (los de
      `agents/external/` quedan exentos: no podemos exigir contrato a código
      de terceros, pero sí avisamos si chocan en `owns` cuando lo declaran).
    - Los contratos no deben referirse a agentes que no existen.
    """
    problems: list[str] = []

    owners: dict[str, str] = {}
    for name, contract in CONTRACTS.items():
        for resource in contract.owns:
            if resource in owners:
                problems.append(
                    f"Recurso '{resource}' tiene dos dueños: '{owners[resource]}' y '{name}'. "
                    f"Un recurso, un dueño — decide cuál y actualiza el otro contrato."
                )
            owners[resource] = name

    if registered_names is not None:
        core_without_contract = registered_names - set(CONTRACTS)
        for name in sorted(core_without_contract):
            problems.append(
                f"El agente '{name}' está registrado pero no tiene contrato en agents/contracts.py. "
                f"Define su rol, límites y recursos antes de usarlo en equipo."
            )

        for name, contract in CONTRACTS.items():
            for collaborator in contract.collaborates:
                if collaborator.startswith("todos"):
                    continue
                if collaborator not in registered_names and collaborator not in CONTRACTS:
                    problems.append(
                        f"El contrato de '{name}' dice colaborar con '{collaborator}', que no existe."
                    )

    return problems
