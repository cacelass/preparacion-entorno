"""
tests/test_integration_agent.py — Unitarios del agente `integration`.

No levantan Docker de verdad: se mockea `IntegrationTool` y `PytestTool.run`.
La clave de diseño que se verifica aquí es que los servicios se bajan SIEMPRE
(al final, incluso cuando los tests fallan) — el mismo patrón de
`tools/mutate.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.agents.integration_agent import IntegrationAgent
from agents.config import ProjectConfig
from agents.context import SharedContext
from agents.tools.integration_tool import IntegrationTool


@pytest.fixture
def integration_context(tmp_path):
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def _write_compose(root: Path) -> None:
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "compose.integration.yml").write_text(
        "services:\n  postgres:\n    image: postgres:16-alpine\n", encoding="utf-8"
    )


def _write_junit(root: Path, failures: int = 0) -> None:
    junit = root / "agents" / "workspace" / "integration" / "junit.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    failure = ""
    if failures:
        failure = '<failure message="boom">trace</failure>'
    junit.write_text(
        f'<testsuites><testsuite errors="0" failures="{failures}" skipped="0" '
        f'tests="2" time="0.1"><testcase classname="t" name="a"/><testcase '
        f'classname="t" name="b">{failure}</testcase></testsuite></testsuites>',
        encoding="utf-8",
    )


class _FakeResult:
    def __init__(self, ok: bool, stdout: str = "", stderr: str = ""):
        self.ok = ok
        self.stdout = stdout
        self.stderr = stderr


def test_run_integration_sin_compose_falla(integration_context, monkeypatch):
    agent = IntegrationAgent(context=integration_context)
    result = agent.run_integration_tests()
    assert not result.success
    assert "compose.integration.yml" in result.message


def test_run_integration_up_ok_baja_servicios(integration_context, monkeypatch):
    _write_compose(integration_context.root)
    fake = IntegrationTool(project_root=integration_context.root)
    down_called = []

    def fake_up(*, compose_file=None, timeout=300):
        return _FakeResult(ok=True)

    def fake_down(*, compose_file=None, timeout=120):
        down_called.append(True)
        return _FakeResult(ok=True)

    monkeypatch.setattr(fake, "up", fake_up)
    monkeypatch.setattr(fake, "down", fake_down)
    monkeypatch.setattr(agent := IntegrationAgent(context=integration_context), "integration", fake)
    # Evitamos pytest de verdad: escribimos el JUnit y saltamos PytestTool.run.
    _write_junit(integration_context.root)
    monkeypatch.setattr(
        "agents.agents.integration_agent.PytestTool.run",
        lambda *a, **k: _FakeResult(ok=True),
    )

    result = agent.run_integration_tests()

    assert result.success
    assert "2/2" in result.message
    assert down_called, "los servicios deben bajarse siempre al terminar"


def test_run_integration_falla_baja_servicios(integration_context, monkeypatch):
    """Aunque los tests fallen, los servicios se bajan (finally)."""
    _write_compose(integration_context.root)
    fake = IntegrationTool(project_root=integration_context.root)
    down_called = []

    def fake_up(*, compose_file=None, timeout=300):
        return _FakeResult(ok=True)

    def fake_down(*, compose_file=None, timeout=120):
        down_called.append(True)
        return _FakeResult(ok=True)

    monkeypatch.setattr(fake, "up", fake_up)
    monkeypatch.setattr(fake, "down", fake_down)
    monkeypatch.setattr(agent := IntegrationAgent(context=integration_context), "integration", fake)
    _write_junit(integration_context.root, failures=1)
    monkeypatch.setattr(
        "agents.agents.integration_agent.PytestTool.run",
        lambda *a, **k: _FakeResult(ok=True),
    )

    result = agent.run_integration_tests()

    assert not result.success
    assert "1 fallo" in result.message
    assert down_called, "los servicios deben bajarse aunque los tests fallen"


def test_run_integration_up_falla_no_corre_tests(integration_context, monkeypatch):
    _write_compose(integration_context.root)
    fake = IntegrationTool(project_root=integration_context.root)
    monkeypatch.setattr(fake, "up", lambda *a, **k: _FakeResult(ok=False, stderr="boom"))
    monkeypatch.setattr(agent := IntegrationAgent(context=integration_context), "integration", fake)

    result = agent.run_integration_tests()

    assert not result.success
    assert "No se pudieron levantar" in result.message


def test_status_sin_compose_falla(integration_context):
    agent = IntegrationAgent(context=integration_context)
    result = agent.status()
    assert not result.success
    assert "compose.integration.yml" in result.message


def test_status_con_compose(integration_context, monkeypatch):
    _write_compose(integration_context.root)
    fake = IntegrationTool(project_root=integration_context.root)
    monkeypatch.setattr(fake, "ps", lambda *a, **k: _FakeResult(ok=True, stdout="postgres up"))
    monkeypatch.setattr(agent := IntegrationAgent(context=integration_context), "integration", fake)

    result = agent.status()

    assert result.success
    assert "postgres up" in result.data["ps"]
