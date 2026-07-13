from __future__ import annotations

import pytest

from agents.agents.docker_agent import DockerAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def docker_context(tmp_path):
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_lint_dockerfile_missing(docker_context):
    agent = DockerAgent(context=docker_context)
    result = agent.lint_dockerfile()
    assert not result.success


def test_validate_compose_missing(docker_context):
    agent = DockerAgent(context=docker_context)
    result = agent.validate_compose()
    assert not result.success


def test_lint_dockerfile_valid(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\nWORKDIR /app\nCOPY . .\n")
    ctx = SharedContext(root=tmp_path, config=ProjectConfig(project_slug="x"))
    agent = DockerAgent(context=ctx)
    result = agent.lint_dockerfile()
    assert result.success
