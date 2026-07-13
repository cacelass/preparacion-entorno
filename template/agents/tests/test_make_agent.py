from __future__ import annotations

import pytest

from agents.agents.make_agent import MakeAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def make_context(tmp_path):
    makefile = tmp_path / "Makefile"
    makefile.write_text("""
.PHONY: all test clean
all: test
test:
	python -m pytest
clean:
	rm -rf __pycache__
""")
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_list_targets_finds_targets(make_context):
    agent = MakeAgent(context=make_context)
    result = agent.list_targets()
    assert result.success
    assert "test" in result.data
    assert "clean" in result.data


def test_validate_valid_makefile(make_context):
    agent = MakeAgent(context=make_context)
    result = agent.validate()
    assert result.success


def test_check_pipeline_chain_missing(make_context):
    agent = MakeAgent(context=make_context)
    result = agent.check_pipeline_chain()
    assert not result.success


def test_suggest_targets(make_context):
    agent = MakeAgent(context=make_context)
    result = agent.suggest_targets()
    assert result.success


def test_validate_missing_makefile(tmp_path):
    ctx = SharedContext(root=tmp_path, config=ProjectConfig(project_slug="x"))
    agent = MakeAgent(context=ctx)
    result = agent.validate()
    assert not result.success
