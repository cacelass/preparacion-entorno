"""
test_cli.py — Tests de la CLI: el formato `--json` no duplica la información.

Cuando el consumidor es una herramienta/agente, `data` codifica el resultado y
`message` (prosa) lo repetiría: pagar tokens dos veces por lo mismo es tirar
contexto. La regla: si hay `data`, viaja `data`; si no, viaja `message`.
"""

from __future__ import annotations

import io
import json
import sys

from agents.cli import _print_result
from agents.core.base_agent import AgentResult


def _run_json(result: AgentResult) -> dict:
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        _print_result(result, json_mode=True)
    finally:
        sys.stdout = old
    return json.loads(buf.getvalue())


def test_json_con_data_omite_message_duplicado():
    result = AgentResult(True, "git", "status", "mensaje prosa", data={"a": 1})
    out = _run_json(result)
    assert out["data"] == {"a": 1}
    assert "message" not in out, "el packet (data) ya codifica el resultado"
    assert out["certainty"] == 1.0
    assert out["agent"] == "git" and out["action"] == "status"


def test_json_sin_data_mantiene_message():
    result = AgentResult(False, "harness", "finish", "no se cierra", needs=["evidencia"])
    out = _run_json(result)
    assert out["message"] == "no se cierra"
    assert out["needs"] == ["evidencia"]


def test_json_siempre_expone_la_certeza():
    result = AgentResult(True, "env", "check", "ok", data={})
    result.certainty = 0.4
    out = _run_json(result)
    assert out["certainty"] == 0.4
