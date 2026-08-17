"""
test_harness_agent.py — Tests del agente `harness`.

Lo que se prueba de verdad es la regla del arnés: `finish` no cierra una
feature si la puerta falla o si no hay evidencia. Es la razón de existir del
agente, así que es lo que más tests tiene.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from agents.agents.harness_agent import HarnessAgent, validate_gherkin

BACKLOG = {
    "version": 1,
    "project": "Test",
    "features": [
        {
            "id": "A-001",
            "title": "Primera",
            "description": "d",
            "acceptance_criteria": ["c1", "c2"],
            "status": "pending",
            "depends_on": [],
        },
        {
            "id": "B-001",
            "title": "Segunda",
            "description": "d",
            "acceptance_criteria": ["c1"],
            "status": "pending",
            "depends_on": ["A-001"],
        },
    ],
}

GATE_OK = """#!/usr/bin/env bash
echo '{"ready": true, "errors": 0, "warnings": 0, "checks": [{"status":"ok","check":"pytest","detail":"3 passed"}]}'
exit 0
"""

GATE_FAIL = """#!/usr/bin/env bash
echo '{"ready": false, "errors": 1, "warnings": 0, "checks": [{"status":"fail","check":"pytest","detail":"1 failed"}]}'
exit 1
"""


def _write_gate(root: Path, body: str) -> None:
    script = root / "init.sh"
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def harness(context) -> HarnessAgent:
    (context.root / "harness").mkdir(exist_ok=True)
    (context.root / "harness" / "featureslist.json").write_text(
        json.dumps(BACKLOG, indent=2), encoding="utf-8"
    )
    (context.root / "harness" / "progress").mkdir(exist_ok=True)
    (context.root / "harness" / "progress" / "history.md").write_text("# Historial\n", encoding="utf-8")
    return HarnessAgent(context=context)


# -- lectura del backlog ------------------------------------------------------
def test_status_cuenta_por_estado(harness):
    result = harness.status()
    assert result.success
    assert result.data["counts"]["pending"] == 2
    assert result.data["eligible"] == ["A-001"]  # B-001 depende de A-001


def test_status_sin_backlog_falla(context):
    result = HarnessAgent(context=context).status()
    assert not result.success
    assert "featureslist.json" in result.message


def test_status_con_json_corrupto_falla(context):
    (context.root / "harness").mkdir(exist_ok=True)
    (context.root / "harness" / "featureslist.json").write_text("{roto", encoding="utf-8")
    result = HarnessAgent(context=context).status()
    assert not result.success
    assert "JSON válido" in result.message


def test_status_avisa_de_dos_in_progress(harness):
    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    doc["features"][0]["status"] = "in_progress"
    doc["features"][1]["status"] = "in_progress"
    (harness.ctx.root / "harness" / "featureslist.json").write_text(json.dumps(doc))
    result = harness.status()
    assert result.success
    assert any("in_progress a la vez" in w for w in result.warnings)


def test_next_respeta_dependencias(harness):
    result = harness.next()
    assert result.success
    assert result.data["id"] == "A-001"


def test_next_propone_plan_scope_en_proyecto_sin_spec(harness):
    """Primera vez en un proyecto recién generado: `next` propone `plan scope`
    (la entrevista que construye el spec) en vez de dejar rellenar a mano."""
    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    doc["features"].insert(0, {
        "id": "SCOPE-001", "title": "Definir qué se quiere resolver",
        "description": "d", "acceptance_criteria": ["a"], "status": "pending",
        "depends_on": [],
    })
    (harness.ctx.root / "harness" / "featureslist.json").write_text(json.dumps(doc))
    assert not (harness.ctx.root / "references" / "00-objetivo.md").exists()

    result = harness.next()
    assert result.success
    assert result.data["id"] == "SCOPE-001"
    assert result.data.get("sugerencia") == "plan scope"
    assert "plan scope" in result.message


def test_next_con_spec_no_sugiere_scope(harness):
    """Si el spec ya existe, `next` se comporta normal (no propone la entrevista)."""
    (harness.ctx.root / "references").mkdir(exist_ok=True)
    (harness.ctx.root / "references" / "00-objetivo.md").write_text(
        "# Objetivo\n\n## Pregunta\n¿Q?\n", encoding="utf-8"
    )
    result = harness.next()
    assert result.success
    assert result.data.get("sugerencia") != "plan scope"


def test_next_retoma_lo_abierto(harness):
    harness.start(id="A-001")
    result = harness.next()
    assert result.success
    assert result.data["id"] == "A-001"
    assert "Retoma" in result.message


def test_next_sin_elegibles_por_dependencias(harness):
    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    doc["features"][0]["status"] = "blocked"
    (harness.ctx.root / "harness" / "featureslist.json").write_text(json.dumps(doc))
    result = harness.next()
    assert not result.success
    assert "depends_on" in result.message or "dependencias" in result.message


# -- abrir --------------------------------------------------------------------
def test_start_marca_in_progress_y_escribe_current(harness):
    result = harness.start(id="A-001")
    assert result.success
    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "in_progress"
    current = (harness.ctx.root / "harness" / "progress" / "current.md").read_text()
    assert "A-001" in current
    assert "c1" in current and "c2" in current


def test_start_sin_id_pide_el_id(harness):
    result = harness.start()
    assert not result.success
    assert result.needs


def test_start_de_feature_inexistente_falla(harness):
    assert not harness.start(id="NO-EXISTE").success


def test_start_bloquea_si_ya_hay_otra_abierta(harness):
    harness.start(id="A-001")
    result = harness.start(id="B-001")
    assert not result.success
    assert "A-001" in result.message


def test_start_exige_dependencias_cerradas(harness):
    result = harness.start(id="B-001")
    assert not result.success
    assert "A-001" in result.message


# -- la puerta ----------------------------------------------------------------
def test_gate_en_verde(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    result = harness.gate()
    assert result.success
    assert result.data["ready"] is True


def test_gate_en_rojo(harness):
    _write_gate(harness.ctx.root, GATE_FAIL)
    result = harness.gate()
    assert not result.success
    assert any("pytest" in w for w in result.warnings)


def test_gate_sin_script_falla(harness):
    result = harness.gate()
    assert not result.success
    assert "init.sh" in result.message


# -- cerrar: la regla que no se salta -----------------------------------------
def test_finish_rechaza_si_la_puerta_falla(harness):
    _write_gate(harness.ctx.root, GATE_FAIL)
    harness.start(id="A-001")
    result = harness.finish(id="A-001", evidence="3 passed")
    assert not result.success
    assert "no se cierra" in result.message.lower()
    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "in_progress"  # NO se tocó


def test_finish_rechaza_sin_evidencia(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    result = harness.finish(id="A-001")
    assert not result.success
    assert result.needs
    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "in_progress"


def test_finish_cierra_y_escribe_historial(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    result = harness.finish(id="A-001", evidence="pytest: 3 passed, 0 failed en 0.4s", changes="src/x.py")
    assert result.success
    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "done"
    history = (harness.ctx.root / "harness" / "progress" / "history.md").read_text()
    assert "A-001" in history
    assert "3 passed" in history
    assert "src/x.py" in history


def test_finish_deja_current_en_idle(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    harness.finish(id="A-001", evidence="pytest: 3 passed, 0 failed en 0.4s")
    current = (harness.ctx.root / "harness" / "progress" / "current.md").read_text()
    assert "idle" in current
    assert "A-001" not in current


def test_finish_desbloquea_la_dependiente(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    harness.finish(id="A-001", evidence="pytest: 3 passed, 0 failed en 0.4s")
    result = harness.next()
    assert result.success
    assert result.data["id"] == "B-001"


def test_finish_rechaza_evidencia_afirmacion(harness):
    """'ok' / 'hecho' son afirmaciones, no salida de comando: se rechazan."""
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    result = harness.finish(id="A-001", evidence="ok")
    assert not result.success
    assert result.needs
    assert "comando" in result.message.lower()
    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "in_progress"


def test_finish_rechaza_afirmacion_corta_aunque_tenga_palabras(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    result = harness.finish(id="A-001", evidence="los tests pasan")
    assert not result.success
    assert result.needs


def test_finish_de_algo_ya_cerrado_falla(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    harness.finish(id="A-001", evidence="ok")
    assert not harness.finish(id="A-001", evidence="ok").success


# -- bloquear y registrar -----------------------------------------------------
def test_block_exige_motivo(harness):
    result = harness.block(id="A-001")
    assert not result.success
    assert result.needs


def test_block_guarda_el_motivo(harness):
    result = harness.block(id="A-001", reason="falta el dataset")
    assert result.success
    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "blocked"
    assert doc["features"][0]["blocked_reason"] == "falta el dataset"


def test_record_escribe_el_informe_del_subagente(harness):
    result = harness.record(agent="explorer", id="A-001", content="## Respuesta\nEstá en x.py")
    assert result.success
    path = harness.ctx.root / "harness" / "progress" / "explorer-A-001.md"
    assert path.exists()
    text = path.read_text()
    assert "explorer · A-001" in text
    assert "x.py" in text


def test_record_sin_contenido_pide_datos(harness):
    result = harness.record(agent="explorer", id="A-001")
    assert not result.success
    assert result.needs


# -- añadir -------------------------------------------------------------------
def test_add_crea_feature_pendiente(harness):
    result = harness.add(id="C-001", title="Tercera", criteria="uno;dos")
    assert result.success
    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    nueva = doc["features"][-1]
    assert nueva["id"] == "C-001"
    assert nueva["status"] == "pending"
    assert nueva["acceptance_criteria"] == ["uno", "dos"]


def test_add_rechaza_id_duplicado(harness):
    assert not harness.add(id="A-001", title="Otra", criteria="uno").success


def test_add_rechaza_dependencia_inexistente(harness):
    result = harness.add(id="C-001", title="Tercera", criteria="uno", depends_on="NO-EXISTE")
    assert not result.success
    assert "no existen" in result.message


def test_add_sin_criterios_pide_criterios(harness):
    result = harness.add(id="C-001", title="Tercera")
    assert not result.success
    assert result.needs


# -- el bucle evaluador-optimizador está acotado -------------------------------
# implementer <-> reviewer es un patrón adversario, y esos bucles necesitan
# tope: sin él, un reviewer exigente y un implementer que no acierta queman
# contexto indefinidamente sin que nadie lleve la cuenta.

def _rechazar(harness, veces: int) -> list:
    return [
        harness.record(agent="reviewer", id="A-001", verdict="rechazado",
                       content=f"## Bloqueantes\nfalta el test {i}")
        for i in range(veces)
    ]


def test_cada_rechazo_incrementa_la_ronda(harness):
    harness.start(id="A-001")
    r1, r2 = _rechazar(harness, 2)
    assert r1.data["review_rounds"] == 1
    assert r2.data["review_rounds"] == 2
    assert r1.success and r2.success


def test_avisa_en_la_ronda_previa_al_limite(harness):
    harness.start(id="A-001")
    _, r2 = _rechazar(harness, 2)
    assert any("se bloquea" in w for w in r2.warnings)


def test_al_agotar_el_bucle_bloquea_y_escala(harness):
    harness.start(id="A-001")
    resultados = _rechazar(harness, 3)
    ultimo = resultados[-1]

    assert not ultimo.success
    assert ultimo.needs, "debe escalar al humano, no limitarse a fallar"
    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "blocked"
    assert "3" in doc["features"][0]["blocked_reason"]


def test_el_informe_se_guarda_aunque_se_agote_el_bucle(harness):
    harness.start(id="A-001")
    _rechazar(harness, 3)
    informe = (harness.ctx.root / "harness" / "progress" / "reviewer-A-001.md").read_text()
    assert "falta el test 2" in informe, "el ultimo informe no debe perderse al bloquear"


def test_una_aprobacion_no_cuenta_como_ronda(harness):
    harness.start(id="A-001")
    r = harness.record(agent="reviewer", id="A-001", verdict="aprobado", content="ok")
    assert r.success
    assert r.data["review_rounds"] is None


def test_el_informe_de_otro_subagente_no_cuenta_como_ronda(harness):
    harness.start(id="A-001")
    r = harness.record(agent="implementer", id="A-001", verdict="fail", content="no pude")
    assert r.success
    assert r.data["review_rounds"] is None


def test_reabrir_la_feature_reinicia_el_contador(harness):
    harness.start(id="A-001")
    _rechazar(harness, 3)                      # queda blocked
    harness.start(id="A-001")                  # el humano la reabre

    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "in_progress"
    assert doc["features"][0]["review_rounds"] == 0
    assert "blocked_reason" not in doc["features"][0]

    r = harness.record(agent="reviewer", id="A-001", verdict="rechazado", content="otra vez")
    assert r.success and r.data["review_rounds"] == 1


# -- contrato Gherkin (flujo SDD) ---------------------------------------------
def test_validate_gherkin_acepta_contrato_valido():
    texto = """Feature: Filtrado por fecha

  Scenario: S1 — fecha inclusiva
    Given una nota creada el 2024-01-01
    When se filtra por esa fecha
    Then aparece en el resultado
"""
    assert validate_gherkin(texto) == []


def test_validate_gherkin_rechaza_sin_feature():
    texto = """Scenario: S1 — algo
    Given un estado
    When pasa algo
    Then cambia
"""
    assert "Feature:" in " ".join(validate_gherkin(texto))


def test_validate_gherkin_rechaza_sin_scenario():
    texto = "Feature: algo\n\n  Given x\n"
    assert any("Scenario" in p for p in validate_gherkin(texto))


def test_validate_gherkin_rechaza_sin_pasos():
    texto = """Feature: algo

  Scenario: S1 — título
"""
    assert validate_gherkin(texto)  # sin Given/When/Then → no vacío


def test_write_feature_genera_borrador_y_deja_spec_ready(harness):
    result = harness.write_feature(id="A-001")
    assert result.success
    assert result.data["draft"] is True
    assert result.data["scenarios"] == 2  # A-001 tiene c1 y c2

    path = harness.ctx.root / "features" / "A-001.feature"
    assert path.exists()
    texto = path.read_text()
    assert "Feature:" in texto
    assert "Scenario:" in texto and "c1" in texto and "c2" in texto

    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "spec_ready"


def test_write_feature_con_content_propio_no_es_borrador(harness):
    gherkin = """Feature: Propia

  Scenario: S1 — límite
    Given x
    When y
    Then z
"""
    result = harness.write_feature(id="A-001", content=gherkin)
    assert result.success
    assert result.data["draft"] is False
    assert "Given x" in (harness.ctx.root / "features" / "A-001.feature").read_text()


def test_write_feature_con_gherkin_invalido_falla(harness):
    result = harness.write_feature(id="A-001", content="no es gherkin")
    assert not result.success
    assert "Gherkin" in result.message


def test_write_feature_sin_id_pide_id(harness):
    result = harness.write_feature()
    assert not result.success
    assert result.needs


def test_write_feature_de_feature_inexistente_falla(harness):
    assert not harness.write_feature(id="NO-EXISTE").success


def test_approve_abre_feature_solo_desde_spec_ready(harness):
    harness.write_feature(id="A-001")
    result = harness.approve(id="A-001")
    assert result.success

    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "in_progress"
    current = (harness.ctx.root / "harness" / "progress" / "current.md").read_text()
    assert "features/A-001.feature" in current


def test_approve_sin_spec_ready_rechaza(harness):
    result = harness.approve(id="A-001")  # está pending, nunca tuvo spec
    assert not result.success
    assert "spec_ready" in result.message


def test_approve_sin_contrato_en_disco_rechaza(harness):
    harness.write_feature(id="A-001")
    (harness.ctx.root / "features" / "A-001.feature").unlink()
    result = harness.approve(id="A-001")
    assert not result.success
    assert "A-001.feature" in result.message


def test_approve_sin_id_pide_id(harness):
    result = harness.approve()
    assert not result.success
    assert result.needs


# -- F1: certeza (μ.cert) como puerta de cierre ---------------------------------
def test_finish_rechaza_certeza_baja_explicita(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    result = harness.finish(
        id="A-001", evidence="pytest: 3 passed, 0 failed en 0.4s", certainty=0.4
    )
    assert not result.success
    assert result.needs, "una feature con dudas no se cierra: escala, no falla en seco"
    assert "certeza" in result.message.lower()
    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "in_progress"  # NO se tocó


def test_finish_acepta_certeza_suficiente(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    result = harness.finish(
        id="A-001", evidence="pytest: 3 passed, 0 failed en 0.4s", certainty=0.9
    )
    assert result.success


def test_finish_hereda_la_certeza_baja_del_reviewer(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    harness.record(agent="reviewer", id="A-001", verdict="aprobado",
                   content="## Criterios\ncumplidos", certainty=0.5)
    result = harness.finish(id="A-001", evidence="pytest: 3 passed, 0 failed en 0.4s")
    assert not result.success, "si el reviewer dudó, el done hereda la duda"
    assert "certeza" in result.message.lower()


def test_finish_sin_reviewer_ni_certeza_cierra(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    result = harness.finish(id="A-001", evidence="pytest: 3 passed, 0 failed en 0.4s")
    assert result.success, "sin señal de duda, confianza plena — como siempre fue"


# -- GATE-3: el veredicto del reviewer es parte de la puerta ---------------------
# La rúbrica (agents/rubric.py) conecta la revisión con el gate: un 'done'
# sobre un rechazo del reviewer se salta la revisión entera. La certeza alta
# de quien cierra no lo anula — comparte el punto ciego de quien hizo la
# feature, que es justo lo que la revisión independiente corrige.

def test_finish_rechaza_sobre_un_rechazo_del_reviewer(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    harness.record(agent="reviewer", id="A-001", verdict="rechazado",
                   content="## Bloqueantes\nfalta el test", certainty=0.9)
    result = harness.finish(id="A-001", evidence="pytest: 3 passed, 0 failed en 0.4s")
    assert not result.success
    assert result.needs, "cerrar sobre un rechazo escala al bucle, no falla en seco"
    assert "rechazo" in result.message.lower() or "rechazado" in result.message.lower()
    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "in_progress"  # NO se tocó


def test_finish_rechaza_aunque_la_certeza_explicita_sea_alta(harness):
    """Una certeza alta de quien cierra no anula el NO del reviewer (GATE-3)."""
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    harness.record(agent="reviewer", id="A-001", verdict="rechazado",
                   content="## Bloqueantes\nfalta el test")
    result = harness.finish(id="A-001", evidence="pytest: 3 passed, 0 failed en 0.4s",
                            certainty=0.99)
    assert not result.success, "el rechazo del reviewer es un NO explícito, no una duda suave"


def test_finish_acepta_veredicto_aprobado_del_reviewer(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    harness.record(agent="reviewer", id="A-001", verdict="aprobado",
                   content="## Criterios\nR-1: cumplido", certainty=0.95)
    result = harness.finish(id="A-001", evidence="pytest: 3 passed, 0 failed en 0.4s")
    assert result.success


def test_finish_registra_veredicto_y_certeza_en_el_historial(harness):
    """Traza del cierre en history.md: qué revisión avaló el done y con qué certeza."""
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    harness.record(agent="reviewer", id="A-001", verdict="aprobado",
                   content="## Criterios\nR-1: cumplido", certainty=0.9)
    result = harness.finish(id="A-001", evidence="pytest: 3 passed, 0 failed en 0.4s")
    assert result.success
    history = (harness.ctx.root / "harness" / "progress" / "history.md").read_text()
    assert "**Revisión:**" in history
    assert "aprobado" in history
    assert "0.90" in history


def test_finish_sin_informe_de_reviewer_lo_dice_en_el_historial(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    harness.finish(id="A-001", evidence="pytest: 3 passed, 0 failed en 0.4s")
    history = (harness.ctx.root / "harness" / "progress" / "history.md").read_text()
    assert "sin informe de reviewer" in history

def test_finish_devuelve_los_criterios_de_puerta_aplicados(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    result = harness.finish(id="A-001", evidence="pytest: 3 passed, 0 failed en 0.4s")
    assert result.data["criterios_puerta"] == ["GATE-1", "GATE-2", "GATE-3", "GATE-4"]


def test_record_guarda_la_certeza_en_la_cabecera(harness):
    harness.record(agent="reviewer", id="A-001", verdict="aprobado",
                   content="## Criterios\ncumplidos", certainty=0.42)
    path = harness.ctx.root / "harness" / "progress" / "reviewer-A-001.md"
    assert "**Certeza:** 0.42" in path.read_text()


def test_record_clampa_la_certeza_a_01(harness):
    harness.record(agent="reviewer", id="A-001", verdict="aprobado",
                   content="ok", certainty=7.0)
    path = harness.ctx.root / "harness" / "progress" / "reviewer-A-001.md"
    assert "**Certeza:** 1.00" in path.read_text()


# -- F2: protocolo §1 (packet compacto de subagente) -----------------------------
PACKET_OK = json.dumps({
    "§": 1,
    "E": {"X": ["src/model.py", "feature"]},
    "S": {"X.tests": 14},
    "R": [],
    "Δ": ["X.nuevo→implementado@FEAT-007"],
    "μ": {"rol": "implementer", "cert": 0.95, "evidencia": "pytest: 14 passed"},
}, ensure_ascii=False)


def test_record_con_packet_escribe_frontmatter(harness):
    result = harness.record(agent="implementer", id="A-001",
                            content="## Qué hice\nsrc/model.py", packet=PACKET_OK)
    assert result.success
    path = harness.ctx.root / "harness" / "progress" / "implementer-A-001.md"
    text = path.read_text()
    assert "<!-- §1:" in text
    assert "implementado@FEAT-007" in text
    assert "Qué hice" in text, "la prosa convive con el packet"


def test_record_con_packet_hereda_la_certeza_del_packet(harness):
    harness.record(agent="implementer", id="A-001",
                   content="## Qué hice\nx", packet=PACKET_OK)
    path = harness.ctx.root / "harness" / "progress" / "implementer-A-001.md"
    assert "**Certeza:** 0.95" in path.read_text()


def test_record_rechaza_packet_no_json(harness):
    result = harness.record(agent="explorer", id="A-001", content="hola", packet="{roto")
    assert not result.success
    assert "packet" in result.message.lower()


def test_record_rechaza_packet_con_ejes_desconocidos(harness):
    result = harness.record(agent="explorer", id="A-001", content="hola",
                            packet='{"E":{},"μ":{"rol":"explorer"},"Zeta":1}')
    assert not result.success
    assert "desconocidos" in result.message


def test_record_rechaza_packet_sin_mu_rol(harness):
    result = harness.record(agent="explorer", id="A-001", content="hola",
                            packet='{"E":{},"S":{}}')
    assert not result.success
    assert "μ" in result.message


def test_record_rechaza_cert_fuera_de_rango(harness):
    result = harness.record(agent="explorer", id="A-001", content="hola",
                            packet='{"E":{},"μ":{"rol":"explorer","cert":1.7}}')
    assert not result.success
    assert "0 y 1" in result.message


def test_packet_sin_content_sigue_exigiendo_contenido(harness):
    result = harness.record(agent="explorer", id="A-001", packet=PACKET_OK)
    assert not result.success
    assert result.needs, "el packet resume, pero la prosa/evidencia sigue siendo obligatoria"


def test_antecedentes_devuelve_el_resumen_del_packet(harness):
    """`next` resume el precedente con su packet Δ/μ.cert, no con el extracto crudo."""
    from agents.agents.harness_agent import _leer_packet, _packet_resumen

    (harness.ctx.root / "harness" / "progress" / "implementer-B-001.md").write_text(
        "# implementer · B-001\n\n"
        "- **Fecha:** 2026-01-01\n- **Veredicto:** ok\n- **Certeza:** 0.95\n\n"
        f"<!-- §1: {PACKET_OK} -->\n\n## Qué hice\nx\n",
        encoding="utf-8",
    )
    packet = _leer_packet(harness.ctx.root / "harness" / "progress" / "implementer-B-001.md")
    assert packet is not None
    resumen = _packet_resumen(packet)
    assert "implementado@FEAT-007" in resumen
    assert "0.95" in resumen
