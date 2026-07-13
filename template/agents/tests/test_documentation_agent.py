from __future__ import annotations

import pytest

from agents.agents.documentation_agent import DocumentationAgent
from agents.config import ProjectConfig
from agents.context import SharedContext


@pytest.fixture
def doc_context(tmp_path):
    (tmp_path / "mi_paquete").mkdir()
    (tmp_path / "README.md").write_text("# Test\n## Install\n```bash\nmake setup\n```\n## Targets\nmake test\n")
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_check_readme_makefile_sync_no_makefile(doc_context):
    agent = DocumentationAgent(context=doc_context)
    result = agent.check_readme_makefile_sync()
    assert not result.success


def test_build_docs_no_sphinx_conf(doc_context):
    agent = DocumentationAgent(context=doc_context)
    result = agent.build_docs()
    assert not result.success
