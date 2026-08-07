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
    result = harness.finish(id="A-001", evidence="3 passed en 0.4s", changes="src/x.py")
    assert result.success
    doc = json.loads((harness.ctx.root / "harness" / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "done"
    history = (harness.ctx.root / "harness" / "progress" / "history.md").read_text()
    assert "A-001" in history
    assert "3 passed en 0.4s" in history
    assert "src/x.py" in history


def test_finish_deja_current_en_idle(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    harness.finish(id="A-001", evidence="ok")
    current = (harness.ctx.root / "harness" / "progress" / "current.md").read_text()
    assert "idle" in current
    assert "A-001" not in current


def test_finish_desbloquea_la_dependiente(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    harness.finish(id="A-001", evidence="ok")
    result = harness.next()
    assert result.success
    assert result.data["id"] == "B-001"


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
