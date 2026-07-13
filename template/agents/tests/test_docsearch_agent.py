from __future__ import annotations

import pytest

from agents.agents.docsearch_agent import DocSearchAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def docsearch_context(tmp_path):
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_search_fails_without_graph(docsearch_context):
    agent = DocSearchAgent(context=docsearch_context)
    result = agent.search(question="test")
    assert not result.success
    assert "graph" in result.message.lower()
