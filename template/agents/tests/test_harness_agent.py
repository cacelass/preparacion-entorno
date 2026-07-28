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

from agents.agents.harness_agent import HarnessAgent

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
    (context.root / "featureslist.json").write_text(
        json.dumps(BACKLOG, indent=2), encoding="utf-8"
    )
    (context.root / "progress").mkdir(exist_ok=True)
    (context.root / "progress" / "history.md").write_text("# Historial\n", encoding="utf-8")
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
    (context.root / "featureslist.json").write_text("{roto", encoding="utf-8")
    result = HarnessAgent(context=context).status()
    assert not result.success
    assert "JSON válido" in result.message


def test_status_avisa_de_dos_in_progress(harness):
    doc = json.loads((harness.ctx.root / "featureslist.json").read_text())
    doc["features"][0]["status"] = "in_progress"
    doc["features"][1]["status"] = "in_progress"
    (harness.ctx.root / "featureslist.json").write_text(json.dumps(doc))
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
    doc = json.loads((harness.ctx.root / "featureslist.json").read_text())
    doc["features"][0]["status"] = "blocked"
    (harness.ctx.root / "featureslist.json").write_text(json.dumps(doc))
    result = harness.next()
    assert not result.success
    assert "depends_on" in result.message or "dependencias" in result.message


# -- abrir --------------------------------------------------------------------
def test_start_marca_in_progress_y_escribe_current(harness):
    result = harness.start(id="A-001")
    assert result.success
    doc = json.loads((harness.ctx.root / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "in_progress"
    current = (harness.ctx.root / "progress" / "current.md").read_text()
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
    doc = json.loads((harness.ctx.root / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "in_progress"  # NO se tocó


def test_finish_rechaza_sin_evidencia(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    result = harness.finish(id="A-001")
    assert not result.success
    assert result.needs
    doc = json.loads((harness.ctx.root / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "in_progress"


def test_finish_cierra_y_escribe_historial(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    result = harness.finish(id="A-001", evidence="3 passed en 0.4s", changes="src/x.py")
    assert result.success
    doc = json.loads((harness.ctx.root / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "done"
    history = (harness.ctx.root / "progress" / "history.md").read_text()
    assert "A-001" in history
    assert "3 passed en 0.4s" in history
    assert "src/x.py" in history


def test_finish_deja_current_en_idle(harness):
    _write_gate(harness.ctx.root, GATE_OK)
    harness.start(id="A-001")
    harness.finish(id="A-001", evidence="ok")
    current = (harness.ctx.root / "progress" / "current.md").read_text()
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
    doc = json.loads((harness.ctx.root / "featureslist.json").read_text())
    assert doc["features"][0]["status"] == "blocked"
    assert doc["features"][0]["blocked_reason"] == "falta el dataset"


def test_record_escribe_el_informe_del_subagente(harness):
    result = harness.record(agent="explorer", id="A-001", content="## Respuesta\nEstá en x.py")
    assert result.success
    path = harness.ctx.root / "progress" / "explorer-A-001.md"
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
    doc = json.loads((harness.ctx.root / "featureslist.json").read_text())
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
