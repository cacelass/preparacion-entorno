"""Tests de la auditoría: toda ejecución vía run() queda registrada y es analizable."""

from __future__ import annotations

from agents import audit
from agents.agents.audit_agent import AuditAgent
from agents.orchestrator import Orchestrator


def test_run_records_audit_entry(context, project_root):
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "mi_paquete"\nversion = "0.1.0"\nrequires-python = ">=3.10"\n'
    )
    orchestrator = Orchestrator(context=context)
    result = orchestrator.run("env", "check_python_version")
    assert result.success

    entries = audit.read_entries(context)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["agent"] == "env"
    assert entry["action"] == "check_python_version"
    assert entry["success"] is True
    assert entry["duration_ms"] >= 0
    # No se guardan los valores de los kwargs (pueden contener secretos), solo nombres.
    assert entry["kwarg_names"] == []


def test_failed_actions_are_recorded_too(context):
    orchestrator = Orchestrator(context=context)
    result = orchestrator.run("env", "check_python_version")  # sin pyproject.toml → falla
    assert not result.success

    entries = audit.read_entries(context)
    assert entries and entries[-1]["success"] is False


def test_audit_agent_report_and_suggestions(context, project_root):
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "mi_paquete"\nversion = "0.1.0"\nrequires-python = ">=3.10"\n'
    )
    orchestrator = Orchestrator(context=context)
    orchestrator.run("env", "check_python_version")
    orchestrator.run("env", "check_python_version")

    report = AuditAgent(context=context).report()
    assert report.success
    assert report.data[0]["accion"] == "env.check_python_version"
    assert report.data[0]["runs"] == 2

    suggestions = AuditAgent(context=context).suggest_improvements()
    assert suggestions.success
    # Con solo 2 runs y todo OK, no debe acusar a env de nada; como los demás
    # agentes no se han usado, sí debe señalarlos como "sin uso".
    assert not any("env.check_python_version' falla" in s for s in suggestions.data)
    assert any("sin ninguna ejecución auditada" in s for s in suggestions.data)


def test_audit_report_on_empty_log(context):
    result = AuditAgent(context=context).report()
    assert result.success
    assert result.data == []


def test_run_records_certainty_from_result(context, project_root):
    """`BaseAgent.run` audita la certeza (μ.cert) que el resultado declara."""
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "mi_paquete"\nversion = "0.1.0"\nrequires-python = ">=3.10"\n'
    )
    orchestrator = Orchestrator(context=context)
    result = orchestrator.run("env", "check_python_version")
    assert result.success

    entry = audit.read_entries(context)[-1]
    assert entry["certainty"] == 1.0, "un agente determinista que ejecutó bien tiene certeza plena"


def test_audit_agent_flagea_exito_con_certeza_baja(context):
    """Un 'éxito' que nadie avala con seguridad es una ronda que pudo fallar."""
    import json as _json
    from datetime import datetime, timezone

    from agents import audit

    path = audit.audit_log_path(context)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as f:
        for _ in range(3):  # MIN_RUNS_TO_JUDGE = 3
            f.write(_json.dumps({
                "timestamp": now, "agent": "env", "action": "check_python_version",
                "success": True, "duration_ms": 1.0, "message": "ok",
                "warnings": 0, "kwarg_names": [], "certainty": 0.4,
            }) + "\n")

    suggestions = AuditAgent(context=context).suggest_improvements()
    assert suggestions.success
    assert any("certeza baja" in s for s in suggestions.data)
    assert any("env.check_python_version" in s for s in suggestions.data)


def test_exito_con_certeza_plena_no_genera_sugerencia(context):
    """Certeza 1.0 = señal de duda ausente → la heurística calla."""
    import json as _json
    from datetime import datetime, timezone

    from agents import audit

    path = audit.audit_log_path(context)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as f:
        for _ in range(3):
            f.write(_json.dumps({
                "timestamp": now, "agent": "env", "action": "check_python_version",
                "success": True, "duration_ms": 1.0, "message": "ok",
                "warnings": 0, "kwarg_names": [], "certainty": 1.0,
            }) + "\n")

    suggestions = AuditAgent(context=context).suggest_improvements()
    assert not any("certeza baja" in s for s in suggestions.data)
