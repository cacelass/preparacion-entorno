"""
agents.agents.plan_agent — Jefe de proyecto: encargo → preguntas → delegación.

El flujo de trabajo que implementa (pensado para que el humano solo tenga
que describir, responder, y verificar):

    1. `intake(brief)`   — Descompone el encargo en pasos, asigna cada paso
                           al agente responsable (ruteo del Orchestrator) y
                           detecta TODOS los argumentos que faltan. Devuelve
                           las preguntas y guarda la orden de trabajo en
                           `agents/workspace/plan/orden-<id>.json`.
    2. (humano)          — Revisa el plan (el JSON es editable a mano: puedes
                           cambiar agente/acción de un paso, borrar pasos,
                           reordenar) y responde las preguntas con `answer`.
    3. `execute(order)`  — Se NIEGA a ejecutar si quedan preguntas sin
                           responder. Si está completa, delega cada paso vía
                           GStack (cada ejecución queda auditada) y devuelve
                           el resumen + qué verificar.
    4. `status()`        — Estado de todas las órdenes de trabajo.

Límites (ver su contrato en agents/contracts.py):
- No ejecuta ninguna acción de dominio él mismo: siempre delega en el
  agente dueño del recurso.
- No inventa argumentos: lo que falta se pregunta (AgentResult.needs).
- La descomposición del brief es heurística (frases separadas por saltos de
  línea, ';' o conectores tipo "y luego"). Para encargos complejos, escribe
  el brief con un paso por línea — o edita el JSON de la orden antes de
  ejecutar. Un agente de codificación (Claude, etc.) también puede escribir
  la orden JSON directamente y saltarse el intake.
"""

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime
from pathlib import Path

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent

# Conectores que separan pasos dentro de una misma línea del brief.
_STEP_SPLIT = re.compile(
    r"[\n;]+|(?:,\s+)?\by\s+(?:luego|después|despues|entonces)\b|\bdespués de eso\b",
    re.IGNORECASE,
)

# ── plan scope: la entrevista que construye el spec ──────────────────────────
# La entrevista (`scope start` → `scope answer` → `scope commit`) capta lo
# necesario para el spec (`references/00-objetivo.md`, feature SCOPE-001) y
# siembra el backlog en el orden lógico del arnés. El PRD no se entrevista:
# `documentation update_prd` lo deriva del spec + backlog (ver lider.md).

#: Campos del spec. (clave, pregunta, obligatorio). El orden es el del fichero.
_SPEC_FIELDS: tuple[tuple[str, str, bool], ...] = (
    ("pregunta", "¿Qué pregunta responde este proyecto?", True),
    ("metrica", "¿Cuál es la métrica de éxito, con un umbral numérico? (p. ej. 'F1 macro >= 0.80 en validación')", True),
    ("datos", "¿Con qué datos de partida se cuenta?", True),
    ("parada", "¿Qué resultado haría replantear o abandonar el proyecto? (criterio de parada)", True),
    ("usuarios", "¿Para quién es el resultado? (opcional)", False),
    ("alcance", "¿Qué entra y qué NO entra en el alcance? (opcional)", False),
    ("riesgos", "¿Qué riesgos o restricciones se conocen? (opcional)", False),
)
_SPEC_REQUIRED = tuple(k for k, _, req in _SPEC_FIELDS if req)

#: El rumbo del arnés: las features de dirección en su orden lógico, con sus
#: depends_on. `scope commit` garantiza que existan (sembrado idempotente).
_BACKLOG_DIRECCION: tuple[dict, ...] = (
    {"id": "SCOPE-001", "title": "Definir qué se quiere resolver",
     "description": "Escribir en references/00-objetivo.md qué pregunta responde este proyecto, "
                    "cómo se mide que la respuesta es buena (umbral numérico), con qué datos se "
                    "cuenta y cuándo se da por terminado.",
     "criteria": ["references/00-objetivo.md con los apartados del spec",
                  "La métrica de éxito es un número con umbral, no 'que funcione bien'",
                  "El criterio de parada dice qué resultado haría abandonar el proyecto"],
     "depends_on": []},
    {"id": "RESEARCH-001", "title": "Qué se sabe ya sobre este problema",
     "description": "Reunir el estado del arte antes de improvisar una arquitectura: qué enfoques se usan, qué métricas reporta la literatura y qué se descarta.",
     "criteria": ["references/01-estado-del-arte.md resume cada fuente en 2-3 líneas",
                  "Queda anotado el rango de resultados que reporta la literatura"],
     "depends_on": ["SCOPE-001"]},
    {"id": "EDA-001", "title": "Exploración inicial en los notebooks",
     "description": "Recorrer los notebooks de la plantilla sobre los datos reales y decidir si pueden responder la pregunta de SCOPE-001.",
     "criteria": ["Los notebooks 0-0, 0-1 y 0-2 se ejecutan sin errores",
                  "references/02-eda.md responde si los datos pueden contestar la pregunta"],
     "depends_on": ["RESEARCH-001"]},
    {"id": "DATA-001", "title": "EDA del dataset principal",
     "description": "Análisis exploratorio del dataset real: distribuciones, nulos, calidad y los hallazgos que condicionan el pipeline.",
     "criteria": ["Un EDA del dataset principal con hallazgos accionables"],
     "depends_on": ["EDA-001"]},
    {"id": "FEAT-001", "title": "Pipeline de features reproducible",
     "description": "Convertir en código reproducible las features que sobrevivieron al EDA.",
     "criteria": ["Pipeline de features reproducible y testeado"],
     "depends_on": ["DATA-001"]},
    {"id": "MODEL-001", "title": "Baseline entrenado y evaluado",
     "description": "Entrenar el baseline y compararlo con el umbral de SCOPE-001 y el rango de RESEARCH-001.",
     "criteria": ["Baseline evaluado contra el umbral del spec"],
     "depends_on": ["FEAT-001"]},
)

#: Comparadores que hacen que una métrica sea "numérica con umbral".
_METRICA_COMPARADOR = re.compile(r"(>=|<=|>|<|=)\s*\d+(?:[.,]\d+)?")

#: Detección de riesgos a partir de las respuestas de la entrevista. Cada fila
#: es (patrón, [riesgos]). Es una heurística determinista — misma filosofía que
#: `can_handle` — anclada en la taxonomía de `docs/knowledge/ml/gestion-riesgo.md`.
#: Los riesgos detectados NO se siembran solos: `scope_commit` obliga al usuario
#: a decidir cada uno (aceptar/descartar) antes de sembrar.
_RIESGOS_HEURISTICA: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"\b(login|auth|autentic|usuario|contraseña|password|sesión|session|token)\b",
     ("sql injection", "fuga de credenciales", "enumeración de usuarios")),
    (r"\b(pago|tarjeta|card|cobro|factura|invoice|transacción|dinero)\b",
     ("fraude", "doble cobro", "cumplimiento PCI")),
    (r"\b(datos personales|email|correo|dni|teléfono|telefono|dirección|pii)\b",
     ("privacidad (GDPR)", "fuga de datos")),
    (r"\b(upload|subida|adjunto|fichero|archivo|file)\b",
     ("path traversal", "subida de malware")),
    (r"\b(api pública|endpoint|api publica|rest)\b",
     ("abuso de rate-limit", "exposición de datos")),
    (r"\b(dataset|conjunto de datos|features|entrenamiento|train)\b",
     ("sesgo de datos", "fuga de datos de entrenamiento")),
    (r"\b(modelo|model|serving|inferencia|despliegue|deploy)\b",
     ("drift y degradación silenciosa", "fallo de monitoreo")),
)


def _detectar_riesgos(texto: str) -> list[str]:
    """
    Riesgos que la heurística identifica en `texto` (respuestas de la entrevista).

    Devuelve los riesgos detectados, en orden de la tabla, sin duplicados.
    Es una *propuesta*: el humano decide si aplican (ver `scope_commit`).
    """
    if not texto:
        return []
    low = texto.lower()
    detectados: list[str] = []
    for patron, riesgos in _RIESGOS_HEURISTICA:
        if re.search(patron, low) and riesgos:
            for riesgo in riesgos:
                if riesgo not in detectados:
                    detectados.append(riesgo)
    return detectados


@register_agent
class PlanAgent(BaseAgent):
    name = "plan"
    description = (
        "Jefe de proyecto: convierte un encargo en una orden de trabajo, "
        "pregunta lo que falte, delega cada paso al agente dueño y resume qué verificar. "
        "También dirige la entrevista de arranque (`plan scope`) que construye el spec "
        "y siembra el backlog."
    )
    capabilities = [
        "planificar", "plan de trabajo", "orden de trabajo", "workorder",
        "encargo", "delegar", "brief", "organiza el trabajo",
        "planea", "planifica",
        "scope", "objetivo", "entrevista", "arrancar el proyecto",
        "empezar el proyecto", "spec del proyecto",
    ]

    def actions(self) -> dict:
        return {
            "intake": self.intake,
            "answer": self.answer,
            "execute": self.execute,
            "status": self.status,
            "scope": self.scope,
            "scope_answer": self.scope_answer,
            "scope_commit": self.scope_commit,
        }

    # ── helpers internos ──────────────────────────────────────────────────

    def _orders_dir(self) -> Path:
        return self.ctx.agent_workspace("plan")

    def _order_path(self, order_id: str) -> Path:
        return self._orders_dir() / f"orden-{order_id}.json"

    def _load_order(self, order_id: str) -> dict | None:
        path = self._order_path(order_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_order(self, order: dict) -> None:
        self._order_path(order["id"]).write_text(
            json.dumps(order, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _required_params(method) -> list[str]:
        """Parámetros sin valor por defecto de una acción (los que hay que preguntar)."""
        required = []
        for pname, param in inspect.signature(method).parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        return required

    @staticmethod
    def _questions(order: dict) -> list[str]:
        """Preguntas pendientes de una orden: una por argumento faltante + pasos sin asignar."""
        questions = []
        for i, step in enumerate(order["steps"]):
            for param in step["missing"]:
                questions.append(
                    f"Paso {i} ({step['agent']}.{step['action']}): falta '{param}'. "
                    f"Responde con: run plan answer --order {order['id']} --step{i}-{param.replace('_', '-')} VALOR"
                )
        for fragment in order.get("unmatched", []):
            questions.append(
                f"No sé qué agente debe hacer: '{fragment}'. Añade el paso a mano en "
                f"{order.get('_file', 'la orden JSON')} (agente + acción), o quítalo del encargo."
            )
        return questions

    # ── plan scope: helpers ──────────────────────────────────────────────────

    def _scope_path(self) -> Path:
        return self._orders_dir() / "scope.json"

    def _load_scope(self) -> dict:
        path = self._scope_path()
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"answers": {}, "features": []}

    def _save_scope(self, scope: dict) -> None:
        self._orders_dir().mkdir(parents=True, exist_ok=True)
        self._scope_path().write_text(
            json.dumps(scope, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _scope_questions(self, scope: dict) -> list[str]:
        """Las preguntas pendientes de la entrevista, en orden de _SPEC_FIELDS."""
        answered = scope.get("answers", {})
        return [
            f"{clave}: {pregunta}"
            for clave, pregunta, _req in _SPEC_FIELDS
            if not answered.get(clave)
        ]

    @staticmethod
    def _metrica_valida(metrica: str) -> bool:
        """El criterio #2 de SCOPE-001: un número con umbral, no 'que funcione bien'."""
        return bool(_METRICA_COMPARADOR.search(metrica))

    def _escribir_objetivo(self, scope: dict) -> Path:
        """Escribe references/00-objetivo.md con el spec enriquecido."""
        a = scope.get("answers", {})
        secciones = []
        for clave, _pregunta, _req in _SPEC_FIELDS:
            valor = a.get(clave)
            if not valor:
                continue
            titulo = {
                "pregunta": "Pregunta a responder",
                "metrica": "Métrica de éxito",
                "datos": "Datos de partida",
                "parada": "Criterio de parada",
                "usuarios": "Usuarios",
                "alcance": "Alcance y no-alcance",
                "riesgos": "Restricciones y riesgos",
            }[clave]
            secciones.append(f"## {titulo}\n\n{valor.strip()}\n")
        texto = "# Objetivo del proyecto\n\n" + "\n".join(secciones)
        ruta = self.ctx.root / "references" / "00-objetivo.md"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(texto, encoding="utf-8")
        return ruta

    def _sembrar_backlog(self, scope: dict) -> list[str]:
        """
        Garantiza que el backlog tiene las features de dirección en orden lógico.

        Idempotente: `harness add` rechaza ids duplicados, así que se añaden
        solo las que faltan, en su orden de `_BACKLOG_DIRECCION`, y después las
        features de producto que el usuario propuso durante la entrevista y los
        riesgos (cada uno → un ticket `RISK-NNN` "Mitigar <riesgo>").
        """
        from agents.agents.harness_agent import HarnessAgent

        harness = HarnessAgent(context=self.ctx)
        sembrados: list[str] = []
        for feat in _BACKLOG_DIRECCION:
            result = harness.add(
                id=feat["id"], title=feat["title"], description=feat["description"],
                criteria=";".join(feat["criteria"]),
                depends_on=";".join(feat["depends_on"]),
            )
            if result.success:
                sembrados.append(feat["id"])

        # Features propuestas. Cada ítem puede ser un id explícito ("API-001")
        # o una descripción ("login que distinga por usuario") — en ese caso se
        # le auto-asigna FEAT-NNN. Así la entrevista acepta lenguaje natural.
        propuestas = [t.strip() for t in scope.get("features", []) if t.strip()]
        if propuestas:
            doc, _error = harness._load()
            existentes = (
                {f["id"] for f in doc.get("features", [])}
                if doc is not None else set()
            )
            siguiente_feat = 1
            while f"FEAT-{siguiente_feat:03d}" in existentes:
                siguiente_feat += 1
            for prop in propuestas:
                if re.match(r"^[A-Z]+-\d+$", prop):
                    fid, titulo = prop, prop
                else:
                    fid, titulo = f"FEAT-{siguiente_feat:03d}", prop
                    siguiente_feat += 1
                result = harness.add(
                    id=fid, title=titulo,
                    description=f"Propuesto en la entrevista de arranque: {titulo}.",
                    criteria="Criterios de aceptación por definir en la entrevista",
                )
                if result.success:
                    sembrados.append(fid)

        # Riesgos → tickets RISK-NNN "Mitigar <riesgo>". Se numeran tras los
        # existentes (RISK-001, RISK-002...) para no pisar nada. Vienen de dos
        # sitios: los declarados por el usuario (`riesgos="..."`) y los
        # detectados por la heurística que el usuario aceptó explícitamente.
        declarados = [r.strip() for r in scope.get("answers", {}).get("riesgos", "").split(";") if r.strip()]
        aceptados = [
            r for r, decision in scope.get("decisiones_riesgo", {}).items()
            if decision == "aceptar"
        ]
        riesgos = list(dict.fromkeys(declarados + aceptados))  # sin duplicados, orden estable
        if riesgos:
            doc, error = harness._load()
            existentes = (
                {f["id"] for f in doc.get("features", [])}
                if doc is not None else set()
            )
            siguiente = 1
            while f"RISK-{siguiente:03d}" in existentes:
                siguiente += 1
            for riesgo in riesgos:
                rid = f"RISK-{siguiente:03d}"
                result = harness.add(
                    id=rid, title=f"Mitigar: {riesgo}",
                    description=f"Riesgo detectado en la entrevista de arranque: {riesgo}.",
                    criteria="Riesgo mitigado, aceptado y documentado, o con un plan de mitigación aprobado",
                    depends_on="SCOPE-001",
                )
                if result.success:
                    sembrados.append(rid)
                    siguiente += 1
        return sembrados

    # ── plan scope: la entrevista ────────────────────────────────────────────

    def scope(self, reset: bool = False) -> AgentResult:
        """
        Inicia (o continúa) la entrevista de arranque: pregunta lo necesario
        para el spec. Adaptativa: nunca repite lo ya respondido; pregunta solo
        lo que falta. El borrador vive en agents/workspace/plan/scope.json.
        """
        if reset:
            scope = {"answers": {}, "features": []}
            self._save_scope(scope)
        else:
            scope = self._load_scope()

        questions = self._scope_questions(scope)
        if not questions:
            return AgentResult(
                True, self.name, "scope",
                "Entrevista completa. Revisa el borrador y ejecuta "
                "`run plan scope_commit` para escribir el spec y sembrar el backlog.",
                data=scope, needs=[],
            )
        mensaje = (
            f"Entrevista de arranque: faltan {len(questions)} respuesta(s). "
            "Responde con `run plan scope_answer` (una o varias a la vez)."
        )
        return AgentResult(True, self.name, "scope", mensaje, data=scope, needs=questions)

    def scope_answer(self, **answers) -> AgentResult:
        """
        Responde a la entrevista. Claves: pregunta, metrica, datos, parada,
        usuarios, alcance, riesgos, features (separadas por ';'). La métrica se
        valida: debe ser un número con umbral — 'que funcione bien' no pasa.

        También decide los riesgos detectados por el agente:
          aceptar_riesgos="a;b"   → se sembrarán como tickets RISK-NNN
          descartar_riesgos="a;b" → no se sembrarán
        """
        scope = self._load_scope()
        a = scope.setdefault("answers", {})
        aplicado: list[str] = []

        for clave, valor in answers.items():
            valor = str(valor).strip()
            if not valor:
                continue
            if clave == "features":
                nuevas = [f.strip() for f in valor.split(";") if f.strip()]
                existentes = scope.setdefault("features", [])
                for f in nuevas:
                    if f not in existentes:
                        existentes.append(f)
                aplicado.append("features")
                continue
            if clave in ("aceptar_riesgos", "descartar_riesgos"):
                decisiones = scope.setdefault("decisiones_riesgo", {})
                para = [r.strip() for r in valor.split(";") if r.strip()]
                for r in para:
                    decisiones[r] = "aceptar" if clave == "aceptar_riesgos" else "descartar"
                aplicado.append(clave)
                continue
            if clave not in {k for k, _, _ in _SPEC_FIELDS}:
                return AgentResult(
                    False, self.name, "scope_answer",
                    f"Clave desconocida: '{clave}'. Válidas: "
                    f"{[k for k, _, _ in _SPEC_FIELDS]} + 'features' + "
                    f"'aceptar_riesgos' + 'descartar_riesgos'.",
                    data=scope, needs=self._scope_questions(scope),
                )
            if clave == "metrica" and not self._metrica_valida(valor):
                return AgentResult(
                    False, self.name, "scope_answer",
                    "La métrica de éxito debe ser un número con umbral "
                    "(p. ej. 'F1 macro >= 0.80 en validación'). 'Que funcione bien' no pasa.",
                    data=scope, needs=self._scope_questions(scope),
                )
            a[clave] = valor
            aplicado.append(clave)

        self._save_scope(scope)
        pending = self._scope_questions(scope)
        mensaje = (
            (f"Aplicado: {', '.join(aplicado)}. " if aplicado else "Nada que aplicar. ")
            + (
                f"Quedan {len(pending)} pregunta(s)."
                if pending
                else "Entrevista completa — ejecuta `run plan scope_commit`."
            )
        )
        return AgentResult(bool(aplicado), self.name, "scope_answer", mensaje,
                           data=scope, needs=pending)

    def _pendientes_riesgo(self, scope: dict) -> list[str]:
        """
        Riesgos detectados por la heurística que el usuario aún no ha decidido.

        Se escanea el texto de las respuestas (pregunta, features, alcance,
        datos, usuarios) y se restan: los declarados por el usuario en
        `riesgos` y los ya decididos (aceptar/descartar). El resto son los que
        `scope_commit` obliga a decidir antes de sembrar.
        """
        respuestas = scope.get("answers", {})
        texto = " ".join(
            str(respuestas.get(c, "")) for c in ("pregunta", "alcance", "datos", "usuarios")
        ) + " " + " ".join(scope.get("features", []))
        detectados = _detectar_riesgos(texto)

        declarados = {
            r.strip().lower()
            for r in str(respuestas.get("riesgos", "")).split(";") if r.strip()
        }
        decisiones = scope.get("decisiones_riesgo", {})
        pendientes = []
        for r in detectados:
            if r.lower() in declarados or r in decisiones:
                continue
            pendientes.append(r)
        return pendientes

    def scope_commit(self) -> AgentResult:
        """
        Cierra la entrevista: escribe el spec y siembra el backlog.

        REHUSA si faltan respuestas obligatorias (pregunta, metrica, datos,
        parada): dirigir es del humano. REHUSA también si la heurística
        detectó riesgos sin decidir: cada uno debe aceptarse o descartarse con
        `scope_answer aceptar_riesgos/descartar_riesgos` antes de sembrar.
        El PRD no se escribe aquí — `documentation update_prd` lo deriva del
        spec + backlog.
        """
        scope = self._load_scope()
        a = scope.get("answers", {})
        faltan = [k for k in _SPEC_REQUIRED if not a.get(k)]
        if faltan:
            return AgentResult(
                False, self.name, "scope_commit",
                f"Faltan respuestas obligatorias: {faltan}. Responde con `run plan scope_answer`.",
                data=scope, needs=self._scope_questions(scope),
            )

        pendientes_riesgo = self._pendientes_riesgo(scope)
        if pendientes_riesgo:
            return AgentResult(
                False, self.name, "scope_commit",
                "La heurística detectó riesgos sin decidir: "
                + ", ".join(pendientes_riesgo)
                + ". Decide cada uno antes de sembrar: "
                "`scope_answer aceptar_riesgos=\"...\"` o `descartar_riesgos=\"...\"`.",
                data={"pendientes_riesgo": pendientes_riesgo}, needs=[
                    f"Riesgo detectado: {r}. Responde `scope_answer aceptar_riesgos=\"{r}\"` "
                    f"para sembrarlo como RISK ticket, o `descartar_riesgos=\"{r}\"`."
                    for r in pendientes_riesgo
                ],
            )

        objetivo = self._escribir_objetivo(scope)
        sembrados = self._sembrar_backlog(scope)

        mensaje = (
            f"Spec escrito en {objetivo.relative_to(self.ctx.root)} y backlog "
            f"con {len(sembrados)} feature(s) sembrada(s): {', '.join(sembrados) or 'ninguna nueva'}."
            "\nRegenera el PRD: `run documentation update_prd`. "
            "\nCierra SCOPE-001 cuando la puerta pase: `run harness finish --id SCOPE-001 --evidence ...`."
        )
        return AgentResult(True, self.name, "scope_commit", mensaje,
                           data={"objetivo": str(objetivo.relative_to(self.ctx.root)),
                                 "sembradas": sembrados})

    # ── acciones ──────────────────────────────────────────────────────────

    def intake(self, brief: str) -> AgentResult:
        """
        Descompone `brief` en pasos, asigna agentes y devuelve las preguntas
        necesarias para poder ejecutar sin inventar nada.
        """
        from agents.orchestrator import Orchestrator

        fragments = [f.strip(" .,") for f in _STEP_SPLIT.split(brief) if f and f.strip(" .,")]
        if not fragments:
            return AgentResult(
                False, self.name, "intake",
                "El encargo está vacío. Describe qué quieres hacer (un paso por línea funciona mejor).",
                needs=["el encargo (brief) con al menos un paso"],
            )

        orch = Orchestrator(context=self.ctx)
        steps: list[dict] = []
        unmatched: list[str] = []

        for fragment in fragments:
            decision = orch.select_agent(fragment)
            if decision.agent_name is None or decision.agent_name == self.name:
                unmatched.append(fragment)
                continue
            agent = orch._get_instance(decision.agent_name)  # noqa: SLF001 — colaboración interna del sistema
            action = agent.best_action(fragment)
            if action is None:
                unmatched.append(fragment)
                continue
            missing = self._required_params(agent.actions()[action])
            steps.append({
                "fragment": fragment,
                "agent": decision.agent_name,
                "action": action,
                "confidence": round(decision.confidence, 2),
                "kwargs": {},
                "missing": missing,
            })

        order_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        order = {
            "id": order_id,
            "created": datetime.now().isoformat(timespec="seconds"),
            "brief": brief,
            "status": "borrador",
            "steps": steps,
            "unmatched": unmatched,
        }
        self._save_order(order)
        order["_file"] = str(self._order_path(order_id))

        questions = self._questions(order)
        plan_lines = [
            f"  [{i}] {s['agent']}.{s['action']}  (confianza {s['confidence']}) ← '{s['fragment']}'"
            for i, s in enumerate(steps)
        ]
        summary = (
            f"Orden de trabajo {order_id} creada con {len(steps)} paso(s):\n"
            + "\n".join(plan_lines)
            + (f"\n  Sin asignar: {unmatched}" if unmatched else "")
            + (
                f"\n\nAntes de ejecutar necesito {len(questions)} respuesta(s) — no invento valores."
                if questions
                else f"\n\nLista para ejecutar: run plan execute --order {order_id}"
            )
            + f"\nRevisa/edita el plan en {order['_file']}"
        )
        return AgentResult(
            True, self.name, "intake", summary,
            data=order, needs=questions,
        )

    def answer(self, order: str, **answers) -> AgentResult:
        """
        Responde preguntas de una orden. Claves aceptadas:
          step0_filename=...  → argumento 'filename' del paso 0
          filename=...        → mismo argumento en TODOS los pasos que lo pidan
        """
        wo = self._load_order(order)
        if wo is None:
            return AgentResult(False, self.name, "answer", f"No existe la orden '{order}'.")
        if wo["status"] in ("completado", "fallido"):
            return AgentResult(False, self.name, "answer", f"La orden {order} ya se ejecutó ({wo['status']}).")

        applied = []
        for key, value in answers.items():
            match = re.match(r"step(\d+)_(.+)", key)
            targets = []
            if match:
                idx, param = int(match.group(1)), match.group(2)
                if idx < len(wo["steps"]):
                    targets = [(idx, param)]
            else:
                targets = [(i, key) for i, s in enumerate(wo["steps"]) if key in s["missing"]]
            for idx, param in targets:
                step = wo["steps"][idx]
                step["kwargs"][param] = value
                if param in step["missing"]:
                    step["missing"].remove(param)
                applied.append(f"paso {idx}: {param}={value}")

        wo["status"] = "borrador"
        self._save_order(wo)
        pending = self._questions(wo)
        message = (
            (f"Aplicado: {', '.join(applied)}. " if applied else "Ninguna respuesta coincidió con lo pedido. ")
            + (
                f"Quedan {len(pending)} pregunta(s)."
                if pending
                else f"Todo respondido — ejecuta con: run plan execute --order {order}"
            )
        )
        return AgentResult(bool(applied), self.name, "answer", message, data=wo, needs=pending)

    def execute(self, order: str, auto_commit: bool = False) -> AgentResult:
        """
        Ejecuta una orden completa delegando cada paso vía GStack.
        Se niega si quedan preguntas: dirigir es del humano, adivinar no es de nadie.
        """
        wo = self._load_order(order)
        if wo is None:
            return AgentResult(False, self.name, "execute", f"No existe la orden '{order}'.")

        pending = self._questions(wo)
        if pending:
            return AgentResult(
                False, self.name, "execute",
                f"La orden {order} tiene {len(pending)} pregunta(s) sin responder — no ejecuto con huecos.",
                needs=pending,
            )
        if not wo["steps"]:
            return AgentResult(False, self.name, "execute", f"La orden {order} no tiene pasos.")

        from agents.gstack.stack import GStack

        stack = GStack(auto_commit=auto_commit, context=self.ctx)
        for step in wo["steps"]:
            stack.push(step["agent"], step["action"], **step["kwargs"])
        result = stack.run()

        wo["status"] = "completado" if result.success else "fallido"
        wo["executed"] = datetime.now().isoformat(timespec="seconds")
        wo["results"] = [
            {"agent": r.agent, "action": r.action, "success": r.success, "message": r.message}
            for r in result.results
        ]
        self._save_order(wo)

        to_verify = [
            f"  [{i}] {r.agent}.{r.action}: {r.message}" for i, r in enumerate(result.results)
        ]
        message = (
            f"Orden {order} {wo['status']}.\n{result.summary}\n\n"
            "Para el humano — verifica:\n" + "\n".join(to_verify)
            + "\n(auditoría completa: run audit report)"
        )
        return AgentResult(result.success, self.name, "execute", message, data=wo)

    def status(self, order: str | None = None) -> AgentResult:
        """Estado de una orden concreta, o listado de todas."""
        if order is not None:
            wo = self._load_order(order)
            if wo is None:
                return AgentResult(False, self.name, "status", f"No existe la orden '{order}'.")
            return AgentResult(
                True, self.name, "status",
                f"Orden {order}: {wo['status']} ({len(wo['steps'])} pasos, "
                f"{len(self._questions(wo))} preguntas pendientes)",
                data=wo, needs=self._questions(wo),
            )

        orders = sorted(self._orders_dir().glob("orden-*.json"))
        listing = []
        for path in orders:
            try:
                wo = json.loads(path.read_text(encoding="utf-8"))
                listing.append({
                    "id": wo["id"], "status": wo["status"],
                    "steps": len(wo["steps"]), "brief": wo["brief"][:80],
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return AgentResult(
            True, self.name, "status",
            f"{len(listing)} orden(es) de trabajo en {self._orders_dir()}",
            data=listing,
        )
