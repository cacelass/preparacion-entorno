from __future__ import annotations

import pytest

from agents.agents.research_agent import ResearchAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def research_context(tmp_path):
    (tmp_path / "README.md").write_text("# Project\nThis is a data science project about classification.\n")
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_project_keywords_extracts_from_readme(research_context):
    agent = ResearchAgent(context=research_context)
    result = agent.project_keywords(top=5)
    assert result.success
    assert len(result.data) > 0
