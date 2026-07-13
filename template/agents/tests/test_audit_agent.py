from __future__ import annotations

import pytest

from agents.agents.audit_agent import AuditAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def audit_context(tmp_path):
    (tmp_path / ".agents" / "audit").mkdir(parents=True)
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_report_returns_empty_when_no_logs(audit_context):
    agent = AuditAgent(context=audit_context)
    result = agent.report(last=10)
    assert result.success
    assert result.data == []


def test_failures_returns_empty_when_no_logs(audit_context):
    agent = AuditAgent(context=audit_context)
    result = agent.failures(last=10)
    assert result.success
    assert result.data == []
