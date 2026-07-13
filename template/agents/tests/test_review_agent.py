from __future__ import annotations

import pytest

from agents.agents.review_agent import ReviewAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def review_context(tmp_path):
    (tmp_path / "mi_paquete").mkdir()
    (tmp_path / "mi_paquete" / "__init__.py").write_text("")
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_review_file_nonexistent(review_context):
    agent = ReviewAgent(context=review_context)
    result = agent.review_file(relative_path="no_existe.py")
    assert not result.success


def test_review_file_short(review_context):
    f = review_context.root / "mi_paquete" / "example.py"
    f.write_text("def foo():\n    return 42\n")
    agent = ReviewAgent(context=review_context)
    result = agent.review_file(relative_path="mi_paquete/example.py")
    assert result.success
