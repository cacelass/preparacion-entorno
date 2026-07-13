from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agents.agents.env_agent import EnvAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def env_context(tmp_path):
    (tmp_path / "mi_paquete").mkdir()
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "mi_paquete"\nrequires-python = ">=3.10"\n')
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_info_reports_python_version(env_context):
    agent = EnvAgent(context=env_context)
    result = agent.info()
    assert result.success
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in result.data["python_version"]


def test_info_warns_no_env_file(env_context):
    agent = EnvAgent(context=env_context)
    result = agent.info()
    assert any("No hay .env" in w for w in result.warnings)


def test_info_no_warning_when_env_exists(env_context):
    (env_context.root / ".env").write_text("GEMINI_API_KEY=test\n")
    agent = EnvAgent(context=env_context)
    result = agent.info()
    assert not any("No hay .env" in w for w in result.warnings)


def test_check_python_version_parses_pyproject(env_context):
    agent = EnvAgent(context=env_context)
    result = agent.check_python_version()
    assert result.success
    assert "3.10" in result.data["required"]


def test_check_python_version_missing_pyproject(tmp_path):
    ctx = SharedContext(root=tmp_path, config=ProjectConfig(project_slug="x"))
    agent = EnvAgent(context=ctx)
    result = agent.check_python_version()
    assert not result.success


def test_add_dependency_rejects_empty_package(env_context):
    agent = EnvAgent(context=env_context)
    result = agent.add_dependency(package="")
    assert not result.success
