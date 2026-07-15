from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.agents.audit_agent import AuditAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def audit_context(tmp_path):
    ctx = SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))
    ctx.agent_workspace("audit")
    return ctx


def _log_path(ctx: SharedContext) -> Path:
    return ctx.agent_workspace("audit") / "audit.jsonl"


def _write(ctx: SharedContext, entries: list[dict]) -> Path:
    path = _log_path(ctx)
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return path


def test_report_empty_when_no_logs(audit_context):
    agent = AuditAgent(context=audit_context)
    result = agent.report(last=10)
    assert result.success
    assert result.data == []


def test_failures_empty_when_no_logs(audit_context):
    agent = AuditAgent(context=audit_context)
    result = agent.failures(last=10)
    assert result.success
    assert result.data == []


def test_report_aggregates_successful_runs(audit_context):
    _write(audit_context, [
        {"agent": "git", "action": "status", "success": True, "duration_ms": 10.0, "warnings": 0, "message": "ok"},
        {"agent": "git", "action": "status", "success": True, "duration_ms": 15.0, "warnings": 0, "message": "ok"},
        {"agent": "test", "action": "run_tests", "success": True, "duration_ms": 500.0, "warnings": 1, "message": "ok"},
    ])
    agent = AuditAgent(context=audit_context)
    result = agent.report(last=50)
    assert result.success
    assert len(result.data) == 2
    rows = {r["accion"]: r for r in result.data}
    assert rows["git.status"]["runs"] == 2
    assert rows["test.run_tests"]["runs"] == 1


def test_report_shows_failure_rate(audit_context):
    _write(audit_context, [
        {"agent": "git", "action": "commit", "success": True, "duration_ms": 100.0, "warnings": 0, "message": "ok"},
        {"agent": "git", "action": "commit", "success": False, "duration_ms": 50.0, "warnings": 0, "message": "fail"},
    ])
    agent = AuditAgent(context=audit_context)
    result = agent.report(last=50)
    row = result.data[0]
    assert row["exito"] == "50%"


def test_failures_lists_only_failed(audit_context):
    _write(audit_context, [
        {"agent": "git", "action": "commit", "success": True, "duration_ms": 10.0, "message": "ok"},
        {"agent": "git", "action": "commit", "success": False, "duration_ms": 10.0, "message": "error", "error": "git error"},
    ])
    agent = AuditAgent(context=audit_context)
    result = agent.failures(last=50)
    assert len(result.data) == 1
    assert result.data[0]["accion"] == "git.commit"


def test_failures_respects_last_limit(audit_context):
    _write(audit_context, [
        {"agent": "g", "action": str(i), "success": False, "duration_ms": 1.0, "message": "fail"}
        for i in range(10)
    ])
    agent = AuditAgent(context=audit_context)
    result = agent.failures(last=3)
    assert len(result.data) == 3


def test_suggest_improvements_high_failure_rate(audit_context):
    _write(audit_context, [
        {"agent": "git", "action": "commit", "success": False, "duration_ms": 10.0, "warnings": 0, "message": "fail"}
        for _ in range(5)
    ] + [
        {"agent": "git", "action": "commit", "success": True, "duration_ms": 10.0, "warnings": 0, "message": "ok"}
        for _ in range(1)
    ])
    agent = AuditAgent(context=audit_context)
    result = agent.suggest_improvements(last=50)
    assert any("falla" in s for s in result.data)


def test_suggest_improvements_slow_action(audit_context):
    _write(audit_context, [
        {"agent": "test", "action": "run_tests", "success": True, "duration_ms": 35_000.0, "warnings": 0, "message": "slow"}
        for _ in range(4)
    ])
    agent = AuditAgent(context=audit_context)
    result = agent.suggest_improvements(last=50)
    assert any("tarda" in s for s in result.data)


def test_suggest_improvements_noisy_warnings(audit_context):
    _write(audit_context, [
        {"agent": "docker", "action": "lint", "success": True, "duration_ms": 10.0, "warnings": 3, "message": "ok"}
        for _ in range(4)
    ])
    agent = AuditAgent(context=audit_context)
    result = agent.suggest_improvements(last=50)
    assert any("warnings" in s for s in result.data)


def test_suggest_improvements_healthy_team(audit_context):
    """No sugerencias si todas las acciones tienen buen rendimiento."""
    _write(audit_context, [
        {"agent": "git", "action": "status", "success": True, "duration_ms": 5.0, "warnings": 0, "message": "ok"}
        for _ in range(10)
    ])
    agent = AuditAgent(context=audit_context)
    result = agent.suggest_improvements(last=50)
    assert result.success
    none_of = ["falla", "tarda", "warnings"]
    assert not any(pat in str(result.data) for pat in none_of)


def test_corrupted_log_line_skipped(audit_context):
    _log_path(audit_context).write_text(
        '{"agent": "git", "action": "status", "success": true, "duration_ms": 1.0, "warnings": 0, "message": "ok"}\n'
        'not json at all\n'
        '{"agent": "git", "action": "commit", "success": false, "duration_ms": 1.0, "warnings": 0, "message": "fail"}\n'
    )
    agent = AuditAgent(context=audit_context)
    result = agent.report(last=50)
    assert len(result.data) == 2


def test_empty_lines_skipped(audit_context):
    _log_path(audit_context).write_text("\n\n\n")
    agent = AuditAgent(context=audit_context)
    result = agent.report(last=50)
    assert result.data == []
