from __future__ import annotations

from pathlib import Path

import pytest

from agents.agents.docker_agent import DockerAgent
from agents.config import ProjectConfig
from agents.context import SharedContext
from agents.tools.docker_tool import DockerTool


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


# ─── DockerTool.lint_dockerfile edge cases ─────────────────────────────────


def test_lint_from_without_tag_warns(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python\nWORKDIR /app\n")
    findings = DockerTool.lint_dockerfile(dockerfile)
    assert any("versión fijada" in f.message for f in findings)


def test_lint_latest_tag_warns(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:latest\n")
    findings = DockerTool.lint_dockerfile(dockerfile)
    assert any("versión fijada" in f.message for f in findings)


def test_lint_no_user_warns(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\nRUN pip install pandas\n")
    findings = DockerTool.lint_dockerfile(dockerfile)
    assert any("USER" in f.message for f in findings)


def test_lint_user_present_no_warning(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\nRUN pip install pandas\nUSER appuser\n")
    findings = DockerTool.lint_dockerfile(dockerfile)
    assert not any("USER" in f.message for f in findings)


def test_lint_add_local_file_warns(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\nADD local.tar.gz /app/\n")
    findings = DockerTool.lint_dockerfile(dockerfile)
    assert any("ADD" in f.message for f in findings)


def test_lint_add_url_no_warning(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\nADD https://example.com/file.tar.gz\n")
    findings = DockerTool.lint_dockerfile(dockerfile)
    assert not any("ADD" in f.message for f in findings)


def test_lint_apt_without_recommends_warns(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM ubuntu:22.04\nRUN apt-get update && apt-get install -y curl\n"
    )
    findings = DockerTool.lint_dockerfile(dockerfile)
    assert any("--no-install-recommends" in f.message for f in findings)


def test_lint_apt_update_separate_warns(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM ubuntu:22.04\nRUN apt-get update\nRUN apt-get install -y curl\n")
    findings = DockerTool.lint_dockerfile(dockerfile)
    assert any("apt-get update" in f.message for f in findings)


def test_lint_empty_dockerfile_no_crash(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("")
    findings = DockerTool.lint_dockerfile(dockerfile)
    assert len(findings) >= 0


def test_lint_multiline_dockerfile_no_crash(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM python:3.12-slim\n# comentario\n\nWORKDIR /app\n")
    findings = DockerTool.lint_dockerfile(dockerfile)
    assert isinstance(findings, list)


def test_lint_only_comments_no_crash(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("# This is a comment\n# Another comment\n")
    findings = DockerTool.lint_dockerfile(dockerfile)
    assert isinstance(findings, list)
