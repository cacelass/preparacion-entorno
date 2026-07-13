from __future__ import annotations

import pytest

from agents.agents.graph_agent import GraphAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def graph_context(tmp_path):
    (tmp_path / "reports" / "figures").mkdir(parents=True)
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_list_figures_empty_when_no_figures(graph_context):
    agent = GraphAgent(context=graph_context)
    result = agent.list_figures()
    assert result.success
    assert result.data == []


def test_list_figures_finds_png(graph_context):
    (graph_context.root / "reports" / "figures" / "plot.png").write_text("dummy")
    agent = GraphAgent(context=graph_context)
    result = agent.list_figures()
    assert len(result.data) == 1


def test_audit_figures_empty_when_no_figures(graph_context):
    agent = GraphAgent(context=graph_context)
    result = agent.audit_figures()
    assert result.success
