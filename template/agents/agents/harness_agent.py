"""
agents.agents.harness_agent — Dueño mecánico del arnés.

El arnés (ver AGENTS.md) tiene dos capas: los agentes markdown de
`.opencode/agents/` razonan (qué feature toca, cómo implementarla) y este
agente ejecuta. Todo lo que es determinista —leer el backlog, cambiar un
estado, escribir el histórico, ejecutar la puerta— vive aquí, en Python, y no
en un prompt: un LLM editando JSON a mano se equivoca, `json.dump` no.

La regla del arnés deja de ser una instrucción y pasa a ser código:
`finish()` aplica los criterios de la puerta de `agents/rubric.py` (GATE-1..4)
y REHÚSA cerrar una feature si `./init.sh` no pasa en verde, si la evidencia
no parece real, si el reviewer la rechazó o si la certeza quedó baja. No hay
forma de saltársela pidiéndoselo amablemente al modelo.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.rubric import CRITERIOS_PUERTA, UMBRAL_CERTEZA
from agents.tools.process_tool import run_command

VALID_STATUS = ("pending", "spec_ready", "in_progress", "done", "blocked")
REQUIRED_FIELDS = ("id", "title", "description", "acceptance_criteria", "status")

#: Rechazos seguidos del reviewer antes de bloquear la feature y escalar.
#: Tres es suficiente para corregir un despiste; a partir de ahí el problema
#: casi nunca es el código, sino el criterio o cómo está planteada la feature.
MAX_REVIEW_ROUNDS = 3

#: Acepta "1.0", "0.8", ".5"… el formato de `certainty` que escribe `record`,
#: con o sin el negrita markdown del header (`- **Certeza:** 0.5`).
_CERT_RE = re.compile(r"Certeza:\*{0,2}\s*(\d+(?:\.\d+)?)")

#: El veredicto que `record` escribe en la cabecera (`- **Veredicto:** aprobado`).
_VEREDICTO_RE = re.compile(r"Veredicto:\*{0,2}\s*([^\n]+)")

_RECHAZOS = ("rechazado", "rechaza", "rejected", "fail", "ko")

#: Longitud mínima que tiene una salida de comando real. "ok", "hecho" o "pasa"
#: son afirmaciones, no evidencia — y `finish()` las rechaza (ver
#: `_evidencia_plausible`).
_EVIDENCIA_MIN_LEN = 24


def _evidencia_plausible(evidence: str) -> bool:
    """
    ¿Esto parece la salida literal de un comando, o una afirmación?

    Un pytest/make/init.sh real siempre produce varias palabras y algo de
    estructura. Una evidencia inventada suele ser corta y llana ("ok",
    "los tests pasan"). La puerta no puede saber si la salida es verdad, pero
    sí puede exigir que no sea una afirmación suelta: la verificación de que
    es verdad ya la hace `gate()` ejecutando `init.sh` — este check solo
    obliga a que la evidencia documente esa ejecución, no a que se la inventen.
    """
    texto = evidence.strip()
    if len(texto) < _EVIDENCIA_MIN_LEN:
        return False
    return len(texto.split()) >= 3


def _es_rechazo(verdict: str) -> bool:
    return verdict.strip().lower() in _RECHAZOS


#: Ejes del protocolo §1 (ver prompts/harness_workflow.md). `μ` es obligatorio
#: porque es donde vive el rol y la certeza (`cert`) que `finish` lee. `§` es
#: la versión del codec (1), igual que en el seed de trasgo — se acepta y se
#: ignora salvo para futuras migraciones.
_PACKET_AXES = ("E", "S", "R", "Δ", "μ", "§")


def _validar_packet(packet: str) -> tuple[dict | None, str]:
    """
    Valida un packet §1 y devuelve (dict, error). Error vacío = válido.

    El packet es la forma compacta de un informe de subagente (ver el boot
    seed de prompts/harness_workflow.md): un JSON con los ejes E/S/R/Δ/μ.
    No se exige que `E` y `S` estén llenos — un informe puede no tocar
    entidades nuevas — pero sí que el JSON sea parseable, que no meta ejes
    desconocidos (un typo en 'Entidades' silenciaría el ahorro de tokens) y
    que declare `μ` con `rol`. El `cert` es opcional aquí: la prosa del
    `--content` sigue existiendo igualmente.
    """
    try:
        doc = json.loads(packet)
    except json.JSONDecodeError as exc:
        return None, f"packet no es JSON válido: {exc}"
    if not isinstance(doc, dict):
        return None, "el packet debe ser un objeto JSON"

    claves = set(doc)
    ejes = set(_PACKET_AXES)
    extra = claves - ejes
    if extra:
        return None, f"ejes desconocidos en el packet: {sorted(extra)} (válidos: {sorted(ejes)})"
    if "μ" not in doc:
        return None, "el packet debe declarar el eje μ (rol, cert)"
    mu = doc["μ"]
    if not isinstance(mu, dict) or not isinstance(mu.get("rol"), str) or not mu["rol"]:
        return None, "μ.rol es obligatorio (qué agente reporta este packet)"
    cert = mu.get("cert")
    if cert is not None:
        try:
            cert = float(cert)
        except (TypeError, ValueError):
            return None, f"μ.cert debe ser un número 0..1, no '{cert}'"
        if not 0.0 <= cert <= 1.0:
            return None, f"μ.cert debe estar entre 0 y 1, no '{cert}'"
        mu["cert"] = round(cert, 3)
    return doc, ""


def _certeza_de_informe(path: Path) -> float | None:
    """Lee la certeza (`μ.cert`) que `record` guardó en la cabecera del informe."""
    try:
        texto = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = _CERT_RE.search(texto)
    if not match:
        return None
    try:
        valor = float(match.group(1))
    except ValueError:
        return None
    return min(max(valor, 0.0), 1.0)


#: Frontmatter §1: `<!-- §1: {...} -->` al principio del informe (ver `record`).
_PACKET_RE = re.compile(r"<!--\s*§1:\s*(\{.*?\})\s*-->", re.DOTALL)


def _leer_packet(path: Path) -> dict | None:
    """Extrae el packet §1 del frontmatter de un informe, o None si no lo tiene."""
    try:
        texto = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = _PACKET_RE.search(texto)
    if not match:
        return None
    try:
        doc = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return doc if isinstance(doc, dict) else None


def _packet_resumen(packet: dict) -> str:
    """
    Comprime un packet §1 a una línea legible para el siguiente agente: qué
    cambió (`Δ`) y con qué certeza (`μ.cert`). El resto del packet vive en el
    fichero; el handoff no necesita más que el resumen.
    """
    deltas = packet.get("Δ", [])
    mu = packet.get("μ", {})
    cert = mu.get("cert")
    base = "; ".join(str(d) for d in deltas) if isinstance(deltas, list) and deltas else "(sin cambios)"
    if isinstance(cert, (int, float)):
        return f"Δ: {base} · μ.cert {float(cert):.2f}"
    return f"Δ: {base}"


def validate_gherkin(text: str) -> list[str]:
    """
    Valida la estructura mínima de un contrato Gherkin sin dependencias.

    No es un parser completo de Gherkin: comprueba lo que el arnés necesita
    (una Feature, al menos un Scenario, y pasos Given/When/Then en cada uno)
    para que un `.feature` escrito a mano no pase la puerta con un formato
    roto. La semántica —si los escenarios capturan bien el comportamiento—
    es de la revisión humana, no de un validador de sintaxis.
    """
    problems: list[str] = []
    if "Feature:" not in text:
        problems.append("falta 'Feature:'")

    scenarios = re.findall(r"(?m)^\s*Scenario:.*$", text)
    if not scenarios:
        problems.append("no hay ningún 'Scenario:'")

    steps = re.findall(r"(?m)^\s+(Given|When|Then|And|But)\b", text)
    if scenarios and not steps:
        problems.append("ningún escenario tiene pasos Given/When/Then")
    return problems

CURRENT_TEMPLATE = """# Tarea actual

**Feature:** {fid}
**Estado:** {status}
**Iniciada:** {started}
**Responsable:** {owner}

## Objetivo

{description}

## Criterios de aceptación

{criteria}

## Bitácora

{log}

## Bloqueos

{blockers}
"""

IDLE_CURRENT = """# Tarea actual

> Estado vivo de la ejecución en curso. Es la memoria **fuera** de la ventana de
> contexto: cualquier agente que arranque de cero lee este fichero y sabe dónde
> está el trabajo sin releer el proyecto entero.

**Feature:** _(ninguna)_
**Estado:** idle
**Iniciada:** —
**Responsable:** —

## Objetivo

_(sin trabajo en curso)_

## Criterios de aceptación

_(copiar aquí los `acceptance_criteria` de la feature al empezar)_

## Bitácora

_(una línea por paso: qué se hizo, qué fichero se tocó, qué verificó)_

## Bloqueos

_(qué impide avanzar y qué se necesita para desbloquearlo — vacío si nada)_
"""


@register_agent
class HarnessAgent(BaseAgent):
    name = "harness"
    description = (
        "Dueño del arnés: lee y actualiza harness/featureslist.json y harness/progress/, y "
        "ejecuta la puerta init.sh. No cierra una feature si la puerta no pasa."
    )
    # Ojo: "feature"/"features" NO van aquí — son del agente `data`
    # (feature engineering). Un keyword, un dueño.
    capabilities = [
        "arnes", "arnés", "harness", "backlog",
        "tarea pendiente", "siguiente tarea", "progreso", "progress",
        "criterios de aceptacion", "criterios de aceptación", "puerta", "gate",
    ]

    def actions(self) -> dict:
        return {
            "status": self.status,
            "next": self.next,
            "start": self.start,
            "claim": self.claim,
            "release": self.release,
            "write_feature": self.write_feature,
            "approve": self.approve,
            "finish": self.finish,
            "block": self.block,
            "record": self.record,
            "gate": self.gate,
            "add": self.add,
        }

    # -- rutas ---------------------------------------------------------------
    #: Todo el estado del arnés vive bajo `harness/`. Antes estaba repartido
    #: por la raíz (`featureslist.json`, `progress/`, `memory.md`) y lo primero
    #: que veía alguien al abrir el proyecto era el andamiaje de la IA, no su
    #: proyecto de datos. Es un directorio visible y no oculto a propósito: el
    #: backlog es justo lo que quieres que un humano abra.
    HARNESS_DIR = "harness"

    @property
    def _harness_dir(self) -> Path:
        return self.ctx.root / self.HARNESS_DIR

    @property
    def _backlog_file(self) -> Path:
        return self._harness_dir / "featureslist.json"

    @property
    def _progress_dir(self) -> Path:
        return self._harness_dir / "progress"

    @property
    def _current_file(self) -> Path:
        return self._progress_dir / "current.md"

    @property
    def _history_file(self) -> Path:
        return self._progress_dir / "history.md"

    @property
    def _features_dir(self) -> Path:
        return self.ctx.root / "features"

    # -- backlog -------------------------------------------------------------
    def _load(self) -> tuple[dict | None, str]:
        """Devuelve (documento, error). Si error != "", el documento es None."""
        if not self._backlog_file.exists():
            return None, f"No existe {self._backlog_file.name}. El arnés está incompleto."
        try:
            doc = json.loads(self._backlog_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return None, f"harness/featureslist.json no es JSON válido: {exc}"
        if not isinstance(doc, dict) or not isinstance(doc.get("features"), list):
            return None, "harness/featureslist.json debe ser un objeto con la clave 'features' (lista)."
        return doc, ""

    def _save(self, doc: dict) -> None:
        self._backlog_file.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _find(doc: dict, feature_id: str) -> dict | None:
        for feat in doc["features"]:
            if isinstance(feat, dict) and feat.get("id") == feature_id:
                return feat
        return None

    @staticmethod
    def _peso_dependientes(doc: dict) -> dict[str, int]:
        """
        Cuántas features dependen (transitivamente) de cada una.

        Es la prioridad de desbloqueo: implementar la feature que más desbloquea
        antes despeja el camino del resto. El cierre transitivo se hace con un
        recorrido iterativo sobre el grafo inverso (dep -> dependiente) para no
        heredar el contexto de los LLM: `depends_on` es un campo de cada
        feature, y aquí se lee una vez.
        """
        dependientes: dict[str, set[str]] = {}
        for feat in doc["features"]:
            if not isinstance(feat, dict):
                continue
            fid = feat.get("id")
            if not fid:
                continue
            dependientes.setdefault(fid, set())
            for dep in feat.get("depends_on", []):
                dependientes.setdefault(dep, set()).add(fid)
        peso: dict[str, int] = {}
        for fid in dependientes:
            vistos: set[str] = set()
            pila = list(dependientes[fid])
            while pila:
                actual = pila.pop()
                if actual in vistos:
                    continue
                vistos.add(actual)
                pila.extend(dependientes.get(actual, ()))
            peso[fid] = len(vistos)
        return peso

    @staticmethod
    def _eligible(doc: dict) -> list[dict]:
        """
        Pendientes cuyas dependencias están todas en done, por prioridad de
        desbloqueo (más dependientes primero). El sort es estable: ante el
        mismo peso, manda el orden del backlog.
        """
        done = {f["id"] for f in doc["features"] if f.get("status") == "done"}
        candidatos = [
            f
            for f in doc["features"]
            if f.get("status") == "pending"
            and all(dep in done for dep in f.get("depends_on", []))
        ]
        peso = HarnessAgent._peso_dependientes(doc)
        return sorted(candidatos, key=lambda f: -peso.get(f.get("id", ""), 0))

    def _fail(self, action: str, message: str, **kw: Any) -> AgentResult:
        return AgentResult(success=False, agent=self.name, action=action, message=message, **kw)

    # -- acciones ------------------------------------------------------------
    def status(self) -> AgentResult:
        """Estado del backlog y de la tarea en curso."""
        doc, error = self._load()
        if doc is None:
            return self._fail("status", error)

        features = doc["features"]
        counts = {status: 0 for status in VALID_STATUS}
        for feat in features:
            counts[feat.get("status", "pending")] = counts.get(feat.get("status", "pending"), 0) + 1

        running = [f["id"] for f in features if f.get("status") == "in_progress"]
        warnings = []
        if len(running) > 1:
            warnings.append(
                f"{len(running)} features in_progress a la vez ({', '.join(running)}). "
                f"El arnés espera una: cierra o bloquea las demás."
            )

        eligible = self._eligible(doc)
        return AgentResult(
            success=True,
            agent=self.name,
            action="status",
            message=(
                f"{len(features)} features · {counts['pending']} pending · "
                f"{counts['in_progress']} in_progress · {counts['done']} done · "
                f"{counts['blocked']} blocked"
            ),
            data={
                "counts": counts,
                "in_progress": running,
                "eligible": [f["id"] for f in eligible],
                "prioridad": self._peso_dependientes(doc),
                "features": [
                    {"id": f.get("id"), "title": f.get("title"), "status": f.get("status")}
                    for f in features
                ],
            },
            warnings=warnings,
        )

    def next(self) -> AgentResult:
        """La feature que toca: la in_progress si la hay, si no la primera elegible."""
        doc, error = self._load()
        if doc is None:
            return self._fail("next", error)

        running = [f for f in doc["features"] if f.get("status") == "in_progress"]
        if running:
            feat = running[0]
            return AgentResult(
                success=True, agent=self.name, action="next",
                message=f"Retoma {feat['id']} — {feat['title']} (ya estaba in_progress).",
                data=feat,
            )

        eligible = self._eligible(doc)
        peso = self._peso_dependientes(doc)
        if not eligible:
            blocked = [f["id"] for f in doc["features"] if f.get("status") == "blocked"]
            pending = [f["id"] for f in doc["features"] if f.get("status") == "pending"]
            if pending:
                return self._fail(
                    "next",
                    "Hay features pendientes pero ninguna tiene sus dependencias en done. "
                    "Revisa depends_on o desbloquea lo que falte.",
                    data={"pending": pending, "blocked": blocked},
                )
            return AgentResult(
                success=True, agent=self.name, action="next",
                message="Sin trabajo pendiente: el backlog está cerrado.",
                data={"blocked": blocked},
            )

        # Primera vez en un proyecto recién generado: SCOPE-001 manda aunque otra
        # feature pendiente desbloquee a más (el orden por peso lo desplazaría).
        # No rellenes el spec a mano: la entrevista `plan scope` lo construye y
        # siembra el backlog en orden lógico — el agente lo propone, no espera.
        scope = next((f for f in eligible if f.get("id") == "SCOPE-001"), None)
        if scope is not None and not (self.ctx.root / "references" / "00-objetivo.md").exists():
            return AgentResult(
                success=True, agent=self.name, action="next",
                message=(
                    f"Siguiente: {scope['id']} — {scope['title']}. Este proyecto no tiene spec todavía: "
                    "ejecuta `run plan scope` para la entrevista de arranque que escribe "
                    "references/00-objetivo.md y siembra el backlog en orden lógico."
                ),
                data={**scope, "sugerencia": "plan scope",
                      "motivo": "sin references/00-objetivo.md (proyecto recién generado)"},
            )

        feat = eligible[0]
        return AgentResult(
            success=True, agent=self.name, action="next",
            message=f"Siguiente: {feat['id']} — {feat['title']}",
            data={**feat, "antecedentes": self._antecedentes(feat),
                  "prioridad": peso.get(feat.get("id", ""), 0)},
        )

    def _antecedentes(self, feat: dict) -> list[dict]:
        """
        Qué se hizo antes que se parezca a esta feature, según `harness/progress/`.

        `harness/progress/history.md` crece con cada feature cerrada y nadie lo relee
        entero. Buscar en él por la descripción de lo que toca ahora es la
        forma barata de que el líder pueda pasarle al subagente la ruta del
        precedente en vez de nada. Se devuelven rutas y líneas, no el texto
        completo: heredar contexto es justo lo que el arnés evita.

        Si el proyecto no tiene RAG, no hay antecedentes y ya está: es
        información de más, nunca un requisito.
        """
        try:
            from agents.tools.rag_tool import RagTool

            if not RagTool.available():
                return []
            consulta = f"{feat.get('title', '')} {feat.get('description', '')}".strip()
            if not consulta:
                return []
            hits = RagTool.search(
                self.ctx.root, consulta, top_k=3, source="harness/progress/", max_per_source=1
            )
        except Exception:  # noqa: BLE001 — una pista de más no puede tumbar `next`
            return []

        antecedentes = []
        for h in hits:
            if "error" in h:
                continue
            item: dict = {"source": h["source"], "line": h["line"]}
            packet = _leer_packet(self.ctx.root / h["source"])
            if packet is not None:
                # El protocolo §1: el precedente se resume en su packet (Δ + μ),
                # unos pocos tokens, en vez del extracto de 200 caracteres.
                item["packet"] = packet
                item["extracto"] = _packet_resumen(packet)
            else:
                item["extracto"] = h["text"][:200]
            antecedentes.append(item)
        return antecedentes

    def _ultima_certeza_reviewer(self, feature_id: str) -> float | None:
        """
        La certeza del último informe del reviewer sobre `feature_id`, o None.

        `finish` la usa como señal `μ.cert` cuando quien cierra no pasa una
        certeza explícita: si el reviewer dudó al aprobar, el 'done' hereda
        esa duda. Si no hay informe de reviewer (o no tiene certeza), se
        devuelve None — y `finish` asume confianza plena, como siempre fue.
        """
        path = self._progress_dir / f"reviewer-{feature_id}.md"
        if not path.exists():
            return None
        return _certeza_de_informe(path)

    def _ultimo_veredicto_reviewer(self, feature_id: str) -> str | None:
        """
        El veredicto del último informe del reviewer sobre `feature_id`, o None.

        Es el criterio GATE-3 de la rúbrica: `finish` no cierra una feature
        que el reviewer ha rechazado, aunque quien cierra diga que confía. Un
        veredicto no es una señal suave como la certeza — es un NO explícito,
        y saltárselo es la «rúbrica desconectada del gate» que convierte la
        revisión en un sistema de alertas llamado gobernanza.
        """
        path = self._progress_dir / f"reviewer-{feature_id}.md"
        if not path.exists():
            return None
        try:
            texto = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        match = _VEREDICTO_RE.search(texto)
        if not match:
            return None
        return match.group(1).strip()

    def start(self, *, id: str = "", owner: str = "implementer") -> AgentResult:
        """Abre una feature: status in_progress y harness/progress/current.md rellenado."""
        if not id:
            return self._fail("start", "Falta el id de la feature.",
                              needs=["¿Qué feature abro? Usa el id de featureslist.json (ej. DATA-001)."])

        doc, error = self._load()
        if doc is None:
            return self._fail("start", error)

        feat = self._find(doc, id)
        if feat is None:
            return self._fail("start", f"No existe la feature '{id}' en el backlog.")

        running = [f["id"] for f in doc["features"] if f.get("status") == "in_progress"]
        if running and running != [id]:
            return self._fail(
                "start",
                f"Ya hay trabajo abierto: {', '.join(running)}. Ciérralo o bloquéalo antes de abrir '{id}'.",
                data={"in_progress": running},
            )

        done = {f["id"] for f in doc["features"] if f.get("status") == "done"}
        missing_deps = [dep for dep in feat.get("depends_on", []) if dep not in done]
        if missing_deps:
            return self._fail(
                "start",
                f"'{id}' depende de {', '.join(missing_deps)}, que no están en done.",
                data={"missing_deps": missing_deps},
            )

        feat["status"] = "in_progress"
        feat["started"] = date.today().isoformat()
        # Abrir una feature reinicia el contador de rondas: si venía bloqueada
        # por agotar el bucle, se reabre con las tres rondas enteras — el
        # humano ya intervino, no tiene sentido heredar el castigo anterior.
        feat["review_rounds"] = 0
        feat["touched_files"] = []
        feat.pop("blocked_reason", None)
        self._save(doc)

        self._progress_dir.mkdir(parents=True, exist_ok=True)
        criteria = "\n".join(f"- [ ] {c}" for c in feat.get("acceptance_criteria", []))
        self._current_file.write_text(
            CURRENT_TEMPLATE.format(
                fid=feat["id"],
                status="in_progress",
                started=feat["started"],
                owner=owner,
                description=feat.get("description", ""),
                criteria=criteria or "_(sin criterios definidos)_",
                log="_(pendiente)_",
                blockers="_(ninguno)_",
            ),
            encoding="utf-8",
        )

        return AgentResult(
            success=True, agent=self.name, action="start",
            message=f"{id} abierta (in_progress) y volcada en harness/progress/current.md.",
            data={"id": id, "criteria": feat.get("acceptance_criteria", [])},
        )

    # -- contratar ficheros: claim / release --------------------------------
    def claim(self, *, id: str = "", files: str = "") -> AgentResult:
        """
        Reclama los ficheros que una feature tocará (touched_files).

        «Un recurso, un dueño»: si otro feature activo ya reclama alguno de
        esos ficheros, se rechaza — dos implementaciones no pueden pisarse a la
        vez. Es la formalización de «si tocan los mismos ficheros, secuencial».
        `files` es una lista separada por `;`.
        """
        if not id or not files:
            missing = []
            if not id:
                missing.append("¿Qué feature reclama ficheros? (id de featureslist.json)")
            if not files:
                missing.append("¿Qué ficheros tocará? Sepáralos con ';'.")
            return self._fail("claim", "Faltan datos para reclamar.", needs=missing)

        doc, error = self._load()
        if doc is None:
            return self._fail("claim", error)

        feat = self._find(doc, id)
        if feat is None:
            return self._fail("claim", f"No existe la feature '{id}' en el backlog.")
        if feat.get("status") != "in_progress":
            return self._fail(
                "claim",
                f"'{id}' no está in_progress (está en '{feat.get('status')}'). "
                f"Abre la feature con `harness start` (o `approve`) antes de reclamar ficheros.",
            )

        reclamados = [f.strip() for f in files.split(";") if f.strip()]
        if not reclamados:
            return self._fail("claim", "La lista de ficheros va vacía.",
                              needs=["Pasa al menos un fichero en --files, separados por ';'."])

        conflictos = []
        for otra in doc["features"]:
            if not isinstance(otra, dict) or otra.get("id") == id:
                continue
            if otra.get("status") in ("done", "blocked"):
                continue
            choque = [f for f in reclamados if f in set(otra.get("touched_files", []))]
            if choque:
                conflictos.append(f"{otra.get('id')} → {', '.join(choque)}")

        if conflictos:
            return self._fail(
                "claim",
                f"No se reclama para '{id}': {', '.join(conflictos)}",
                data={"conflictos": conflictos},
                needs=[
                    f"Elige otra feature o coordina: {', '.join(c.split(' → ')[0] for c in conflictos)} "
                    f"ya toca esos ficheros."
                ],
            )

        reclamados_previos = feat.setdefault("touched_files", [])
        for f in reclamados:
            if f not in reclamados_previos:
                reclamados_previos.append(f)
        self._save(doc)

        return AgentResult(
            success=True, agent=self.name, action="claim",
            message=f"{id} reclama {len(reclamados_previos)} fichero(s): {', '.join(reclamados_previos)}.",
            data={"id": id, "touched_files": reclamados_previos},
        )

    def release(self, *, id: str = "") -> AgentResult:
        """Libera los ficheros reclamados por una feature (touched_files → [])."""
        if not id:
            return self._fail("release", "Falta el id de la feature.",
                              needs=["¿Qué feature libera sus ficheros? Usa su id de featureslist.json."])

        doc, error = self._load()
        if doc is None:
            return self._fail("release", error)

        feat = self._find(doc, id)
        if feat is None:
            return self._fail("release", f"No existe la feature '{id}' en el backlog.")

        previos = list(feat.get("touched_files", []))
        if not previos:
            return AgentResult(
                success=True, agent=self.name, action="release",
                message=f"{id} no tenía ficheros reclamados.",
                data={"id": id, "touched_files": []},
            )

        feat["touched_files"] = []
        self._save(doc)
        return AgentResult(
            success=True, agent=self.name, action="release",
            message=f"{id} liberó {len(previos)} fichero(s): {', '.join(previos)}.",
            data={"id": id, "released": previos, "touched_files": []},
        )

    # -- contrato Gherkin (flujo SDD) ----------------------------------------
    def write_feature(self, *, id: str = "", content: str = "") -> AgentResult:
        """
        Escribe el contrato Gherkin de una feature en `features/<id>.feature`.

        Flujo spec-driven: antes de codear, la feature pasa por `spec_ready`
        y un humano aprueba los escenarios (`approve`). `content` es el texto
        Gherkin; si no se pasa, se genera un borrador con un escenario por
        criterio de aceptación. El fichero es el estado de la spec, fuera del
        JSON — igual que `harness/progress/` lo es del progreso.
        """
        if not id:
            return self._fail("write_feature", "Falta el id de la feature.",
                              needs=["¿Qué feature documento? Usa el id de featureslist.json."])

        doc, error = self._load()
        if doc is None:
            return self._fail("write_feature", error)

        feat = self._find(doc, id)
        if feat is None:
            return self._fail("write_feature", f"No existe la feature '{id}' en el backlog.")
        if feat.get("status") == "done":
            return self._fail("write_feature", f"'{id}' ya está cerrada.")

        gherkin = content.strip() if content.strip() else self._draft_feature(feat)
        problems = validate_gherkin(gherkin)
        if problems:
            return self._fail("write_feature", f"El Gherkin no es válido: {'; '.join(problems)}.")

        self._features_dir.mkdir(parents=True, exist_ok=True)
        path = self._features_dir / f"{id}.feature"
        path.write_text(gherkin.rstrip() + "\n", encoding="utf-8")
        feat["status"] = "spec_ready"
        self._save(doc)

        return AgentResult(
            success=True, agent=self.name, action="write_feature",
            message=f"Contrato Gherkin escrito en features/{id}.feature "
                    f"({feat.get('status')} → spec_ready).",
            data={"path": str(path.relative_to(self.ctx.root)),
                  "scenarios": gherkin.count("Scenario:"),
                  "draft": not bool(content.strip())},
            warnings=(
                ["Borrador generado desde acceptance_criteria: revisa que los "
                 "escenarios capturen los casos límite antes de aprobar."]
                if not content.strip() else []
            ),
        )

    def _draft_feature(self, feat: dict) -> str:
        """Un escenario Given-When-Then por criterio de aceptación."""
        lines = [f"Feature: {feat.get('title', feat.get('id', ''))}", ""]
        for i, criterion in enumerate(feat.get("acceptance_criteria", []), start=1):
            lines += [
                f"  Scenario: S{i} — {criterion}",
                "    Given el sistema en su estado inicial",
                "    When se ejecuta el comportamiento de esta feature",
                f"    Then {criterion}",
                "",
            ]
        return "\n".join(lines)

    def approve(self, *, id: str = "", owner: str = "implementer") -> AgentResult:
        """
        Puerta humana del flujo SDD: aprueba la spec de una feature en
        `spec_ready` y la abre (`in_progress`). Solo un humano aprueba —
        esto es un paso explícito, no algo que el líder decide solo.
        """
        if not id:
            return self._fail("approve", "Falta el id de la feature.",
                              needs=["¿Qué feature apruebas? Usa el id de featureslist.json."])

        doc, error = self._load()
        if doc is None:
            return self._fail("approve", error)

        feat = self._find(doc, id)
        if feat is None:
            return self._fail("approve", f"No existe la feature '{id}' en el backlog.")
        if feat.get("status") != "spec_ready":
            return self._fail(
                "approve",
                f"'{id}' no está en spec_ready (está en '{feat.get('status')}'). "
                f"Escribe primero el contrato con `harness write_feature`.",
            )

        feature_file = self._features_dir / f"{id}.feature"
        if not feature_file.exists():
            return self._fail(
                "approve",
                f"No existe features/{id}.feature — ejecuta `harness write_feature` primero.",
            )

        feat["status"] = "in_progress"
        feat["started"] = date.today().isoformat()
        feat["review_rounds"] = 0
        feat["touched_files"] = []
        feat.pop("blocked_reason", None)
        self._save(doc)

        self._progress_dir.mkdir(parents=True, exist_ok=True)
        criteria = "\n".join(f"- [ ] {c}" for c in feat.get("acceptance_criteria", []))
        self._current_file.write_text(
            CURRENT_TEMPLATE.format(
                fid=feat["id"],
                status="in_progress",
                started=feat["started"],
                owner=owner,
                description=feat.get("description", ""),
                criteria=criteria or "_(sin criterios definidos)_",
                log="Spec aprobada por el humano; contrato en "
                    f"features/{id}.feature.",
                blockers="_(ninguno)_",
            ),
            encoding="utf-8",
        )

        return AgentResult(
            success=True, agent=self.name, action="approve",
            message=f"Spec de {id} aprobada e in_progress. Implementa contra features/{id}.feature.",
            data={"id": id, "path": str(feature_file.relative_to(self.ctx.root))},
        )

    def gate(self, *, quick: bool = False) -> AgentResult:
        """Ejecuta ./init.sh y devuelve el veredicto estructurado."""
        script = self.ctx.root / "init.sh"
        if not script.exists():
            return self._fail("gate", "No existe init.sh: el arnés no tiene puerta.")

        args = ["bash", str(script), "--json"]
        if quick:
            args.append("--quick")
        proc = run_command(args, cwd=self.ctx.root, timeout=900)

        try:
            report = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return self._fail(
                "gate",
                f"init.sh no devolvió JSON (exit {proc.returncode}).",
                data={"stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]},
            )

        failed = [c for c in report.get("checks", []) if c.get("status") == "fail"]
        return AgentResult(
            success=bool(report.get("ready")),
            agent=self.name,
            action="gate",
            message=(
                "ENTORNO LISTO — se puede trabajar."
                if report.get("ready")
                else f"ENTORNO BLOQUEADO — {len(failed)} check(s) fallando."
            ),
            data=report,
            warnings=[f"{c['check']}: {c['detail']}" for c in failed],
        )

    def finish(self, *, id: str = "", evidence: str = "", changes: str = "",
               decisions: str = "", pending: str = "", certainty: float | None = None) -> AgentResult:
        """
        Cierra una feature aplicando la rúbrica de la puerta (`agents/rubric.py`,
        GATE-1..4) en código. REHÚSA si init.sh no pasa en verde, si la evidencia
        no parece real, si el reviewer rechazó la feature o si la certeza quedó
        por debajo del umbral: es la regla del arnés, y aquí es código, no una
        instrucción que se pueda ignorar.

        `certainty` (0..1, idea `μ.cert`) es cuánta confianza tiene quien cierra
        de que la feature está bien. Si no se pasa, se lee del último informe
        del `reviewer` (si lo hay); si ninguno existe, se asume 1.0. Por debajo
        de `UMBRAL_CERTEZA` se rechaza: una feature que nadie avala con
        seguridad no se cierra por la vía fácil.
        """
        if not id:
            return self._fail("finish", "Falta el id de la feature.",
                              needs=["¿Qué feature cierro? Usa su id de featureslist.json."])

        doc, error = self._load()
        if doc is None:
            return self._fail("finish", error)

        feat = self._find(doc, id)
        if feat is None:
            return self._fail("finish", f"No existe la feature '{id}' en el backlog.")
        if feat.get("status") == "done":
            return self._fail("finish", f"'{id}' ya está en done.")

        gate = self.gate()
        if not gate.success:
            return self._fail(
                "finish",
                f"NO se cierra '{id}': la puerta no pasa. {gate.message}",
                data=gate.data,
                warnings=gate.warnings,
            )

        if not evidence:
            return self._fail(
                "finish",
                f"'{id}' no se cierra sin evidencia.",
                needs=[
                    "Pega la salida real del comando que demuestra cada criterio "
                    "(pytest, make check, ./init.sh). Una afirmación no es evidencia."
                ],
            )

        if not _evidencia_plausible(evidence):
            return self._fail(
                "finish",
                f"'{id}' no se cierra: la evidencia no parece la salida de un comando.",
                needs=[
                    "Pega la salida LITERAL del comando que lo demuestra (pytest, "
                    "make check, ./init.sh). 'los tests pasan' es una afirmación, "
                    "no evidencia: si no puedes pegar la salida, no lo has ejecutado."
                ],
            )

        # GATE-3: el veredicto del reviewer es parte de la puerta. Un 'done'
        # sobre un rechazo se salta la revisión entera — la certeza no puede
        # anularlo porque quien cierra comparte el punto ciego de quien hizo
        # la feature (ver agents/rubric.py).
        ultimo_veredicto = self._ultimo_veredicto_reviewer(id)
        if _es_rechazo(ultimo_veredicto or ""):
            return self._fail(
                "finish",
                f"'{id}' no se cierra: el último veredicto del reviewer es rechazo "
                f"(rúbrica GATE-3).",
                needs=[
                    f"Reabre el bucle implementer ↔ reviewer: lee "
                    f"harness/progress/reviewer-{id}.md, arregla lo que bloquea y "
                    f"haz que el reviewer registre un veredicto 'aprobado'. Cerrar "
                    f"sobre un rechazo es saltarse la puerta."
                ],
            )

        if certainty is None:
            certainty = self._ultima_certeza_reviewer(id)
        if certainty is not None and certainty < UMBRAL_CERTEZA:
            return self._fail(
                "finish",
                f"'{id}' no se cierra: certeza {certainty:.2f} por debajo del "
                f"umbral ({UMBRAL_CERTEZA}). El reviewer dudó, y un 'done' "
                f"con dudas es una ronda que iba a fallar.",
                needs=[
                    f"Revisa harness/progress/reviewer-{id}.md: ¿qué le falta a la "
                    f"feature para que el reviewer la avale? No se cierra con certeza "
                    f"baja — se reabre el bucle implementer ↔ reviewer."
                ],
            )

        feat["status"] = "done"
        feat["closed"] = date.today().isoformat()
        # Al cerrar se liberan los ficheros reclamados: la feature ya no se
        # trabaja, y lo que ella reclamó vuelve a estar disponible.
        feat["touched_files"] = []
        self._save(doc)

        gate_line = gate.data.get("checks", []) if isinstance(gate.data, dict) else []
        pytest_line = next(
            (c["detail"] for c in gate_line if c.get("check") == "pytest"), "init.sh en verde"
        )

        # Traza de la revisión en el histórico: veredicto + certeza usadas al
        # cerrar, para que un humano pueda auditar el cierre a posteriori sin
        # tener que volver a leer el informe del reviewer.
        informe_reviewer = self._progress_dir / f"reviewer-{id}.md"
        if informe_reviewer.exists():
            revision = ultimo_veredicto or "sin veredicto"
            if certainty is not None:
                revision += f" · μ.cert {certainty:.2f}"
        else:
            revision = "sin informe de reviewer"

        entry = (
            f"\n## {id} — {feat.get('title', '')}\n\n"
            f"- **Cerrada:** {feat['closed']}\n"
            f"- **Verificación:** ./init.sh en verde · {pytest_line}\n"
            f"- **Revisión:** {revision}\n"
            f"- **Cambios:** {changes or '_(no indicados)_'}\n"
            f"- **Decisiones:** {decisions or '_(ninguna reseñable)_'}\n"
            f"- **Pendiente:** {pending or '_(nada)_'}\n\n"
            f"<details><summary>Evidencia</summary>\n\n```\n{evidence.strip()}\n```\n\n</details>\n"
        )
        self._progress_dir.mkdir(parents=True, exist_ok=True)
        with self._history_file.open("a", encoding="utf-8") as fh:
            fh.write(entry)

        self._current_file.write_text(IDLE_CURRENT, encoding="utf-8")

        return AgentResult(
            success=True, agent=self.name, action="finish",
            message=f"{id} cerrada. Histórico actualizado y current.md en idle.",
            data={"id": id, "closed": feat["closed"],
                  "criterios_puerta": [cid for cid, _ in CRITERIOS_PUERTA],
                  "revision": revision},
        )

    def block(self, *, id: str = "", reason: str = "") -> AgentResult:
        """Marca una feature como bloqueada, con el motivo."""
        if not id or not reason:
            missing = []
            if not id:
                missing.append("¿Qué feature bloqueo? (id de featureslist.json)")
            if not reason:
                missing.append("¿Por qué se bloquea? Sin motivo no sirve de nada.")
            return self._fail("block", "Faltan datos para bloquear.", needs=missing)

        doc, error = self._load()
        if doc is None:
            return self._fail("block", error)

        feat = self._find(doc, id)
        if feat is None:
            return self._fail("block", f"No existe la feature '{id}' en el backlog.")

        feat["status"] = "blocked"
        feat["blocked_reason"] = reason
        # Al bloquear se liberan los ficheros reclamados: otra feature puede
        # tomar el relevo sin esperar a que se desbloquee esta.
        feat["touched_files"] = []
        self._save(doc)
        return AgentResult(
            success=True, agent=self.name, action="block",
            message=f"{id} bloqueada: {reason}",
            data={"id": id, "reason": reason},
        )

    def record(self, *, agent: str = "", id: str = "", content: str = "",
               verdict: str = "ok", certainty: float | None = None,
               packet: str = "") -> AgentResult:
        """Guarda el informe de un subagente en harness/progress/<agente>-<ID>.md."""
        if not agent or not id or not content:
            missing = []
            if not agent:
                missing.append("¿Qué subagente escribe? (explorer, implementer, reviewer)")
            if not id:
                missing.append("¿Sobre qué feature? (id de featureslist.json)")
            if not content:
                missing.append("¿Qué contenido? El informe no puede ir vacío.")
            return self._fail("record", "Faltan datos para guardar el informe.", needs=missing)

        # Protocolo §1: el packet compacto (JSON) es la cabecera del informe.
        # Se valida aquí — un JSON roto o con ejes inventados no entra al disco.
        if packet:
            packet_doc, error = _validar_packet(packet)
            if packet_doc is None:
                return self._fail("record", f"packet inválido: {error}",
                                  needs=["Envía el packet como JSON §1 (E/S/R/Δ/μ), o usa solo --content."])
            if certainty is None and "cert" in packet_doc["μ"]:
                certainty = float(packet_doc["μ"]["cert"])
        elif content.strip():
            # Sin packet: se intenta inducir la certeza desde la prosa de la
            # cabecera si alguien ya la escribió a mano — no se exige nada.
            pass

        self._progress_dir.mkdir(parents=True, exist_ok=True)
        path = self._progress_dir / f"{agent}-{id}.md"
        header = (
            f"# {agent} · {id}\n\n"
            f"- **Fecha:** {date.today().isoformat()}\n"
            f"- **Veredicto:** {verdict}\n"
        )
        if certainty is not None:
            header += f"- **Certeza:** {min(max(certainty, 0.0), 1.0):.2f}\n"
        header += "\n"
        if packet and packet_doc is not None:
            header += f"<!-- §1: {json.dumps(packet_doc, ensure_ascii=False)} -->\n\n"
        path.write_text(header + content.strip() + "\n", encoding="utf-8")

        # El bucle implementer <-> reviewer es un patrón evaluador-optimizador,
        # y esos bucles necesitan tope: sin él, un reviewer exigente y un
        # implementer que no acierta queman contexto para siempre y nadie se
        # entera de cuántas vueltas llevan. Al agotarse, la feature se bloquea
        # sola y se escala al humano — en código, no confiando en que el líder
        # lleve la cuenta.
        rounds = None
        if agent == "reviewer" and _es_rechazo(verdict):
            doc, error = self._load()
            if doc is None:
                return self._fail("record", error)
            feat = self._find(doc, id)
            if feat is not None:
                rounds = int(feat.get("review_rounds", 0)) + 1
                feat["review_rounds"] = rounds
                if rounds >= MAX_REVIEW_ROUNDS:
                    feat["status"] = "blocked"
                    feat["blocked_reason"] = (
                        f"El reviewer rechazó {rounds} veces seguidas: el bucle se agotó."
                    )
                self._save(doc)

                if rounds >= MAX_REVIEW_ROUNDS:
                    return self._fail(
                        "record",
                        f"Informe guardado, pero '{id}' se bloquea: {rounds} rechazos seguidos.",
                        data={"path": str(path.relative_to(self.ctx.root)),
                              "verdict": verdict, "review_rounds": rounds},
                        needs=[
                            f"El reviewer ha rechazado '{id}' {rounds} veces. Repetir la misma "
                            f"iteración no lo va a arreglar: lee harness/progress/reviewer-{id}.md y "
                            f"decide si el criterio es correcto, si la feature está mal "
                            f"planteada o si hace falta partirla en varias."
                        ],
                    )

        return AgentResult(
            success=True, agent=self.name, action="record",
            message=(
                f"Informe guardado en harness/progress/{path.name}"
                + (f" · ronda de revisión {rounds}/{MAX_REVIEW_ROUNDS}" if rounds else "")
            ),
            warnings=(
                [f"Van {rounds} rechazos de {MAX_REVIEW_ROUNDS}: a la siguiente se bloquea."]
                if rounds and rounds == MAX_REVIEW_ROUNDS - 1 else []
            ),
            data={"path": str(path.relative_to(self.ctx.root)), "verdict": verdict,
                  "review_rounds": rounds},
        )

    def add(self, *, id: str = "", title: str = "", description: str = "",
            criteria: str = "", depends_on: str = "") -> AgentResult:
        """Añade una feature al backlog. `criteria` y `depends_on` van separados por `;`."""
        missing = []
        if not id:
            missing.append("¿Qué id le pongo? (ej. API-002)")
        if not title:
            missing.append("¿Cuál es el título de la feature?")
        if not criteria:
            missing.append("¿Cuáles son los criterios de aceptación? Sepáralos con ';'.")
        if missing:
            return self._fail("add", "Faltan datos para añadir la feature.", needs=missing)

        doc, error = self._load()
        if doc is None:
            return self._fail("add", error)
        if self._find(doc, id) is not None:
            return self._fail("add", f"Ya existe una feature con id '{id}'.")

        feature = {
            "id": id,
            "title": title,
            "description": description or title,
            "acceptance_criteria": [c.strip() for c in criteria.split(";") if c.strip()],
            "status": "pending",
            "depends_on": [d.strip() for d in depends_on.split(";") if d.strip()],
            "touched_files": [],
        }
        unknown = [d for d in feature["depends_on"] if self._find(doc, d) is None]
        if unknown:
            return self._fail("add", f"depends_on apunta a features que no existen: {', '.join(unknown)}.")

        doc["features"].append(feature)
        self._save(doc)
        return AgentResult(
            success=True, agent=self.name, action="add",
            message=f"{id} añadida al backlog ({len(feature['acceptance_criteria'])} criterios).",
            data=feature,
        )
