from __future__ import annotations

import json

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


# -- PRD vivo -----------------------------------------------------------------
def _proyecto_con_prd(tmp_path):
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness" / "featureslist.json").write_text(
        json.dumps(
            {
                "version": 1,
                "project": "Test",
                "features": [
                    {
                        "id": "SCOPE-001",
                        "title": "Definir objetivo",
                        "status": "done",
                        "description": "d",
                        "acceptance_criteria": ["c1"],
                    },
                    {
                        "id": "MODEL-001",
                        "title": "Baseline",
                        "status": "pending",
                        "description": "d",
                        "acceptance_criteria": ["c1"],
                        "depends_on": ["SCOPE-001"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "00-objetivo.md").write_text(
        "## Pregunta\n¿qué predecir?\n\n## Métrica\nF1 >= 0.8\n", encoding="utf-8"
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "features").mkdir()
    (tmp_path / "features" / "MODEL-001.feature").write_text(
        "Feature: Baseline\n\n  Scenario: S1\n    Given x\n    When y\n    Then z\n",
        encoding="utf-8",
    )


def test_update_prd_dry_run_no_escribe(doc_context):
    agent = DocumentationAgent(context=doc_context)
    result = agent.update_prd(dry_run=True)
    assert result.success
    assert not (doc_context.root / "docs" / "prd.md").exists()


def test_update_prd_genera_docs_prd(doc_context):
    _proyecto_con_prd(doc_context.root)
    agent = DocumentationAgent(context=doc_context)
    result = agent.update_prd()
    assert result.success
    prd = (doc_context.root / "docs" / "prd.md").read_text()
    assert "Product Requirements Document" in prd
    assert "generado" in prd


def test_update_prd_incluye_objetivo(doc_context):
    _proyecto_con_prd(doc_context.root)
    agent = DocumentationAgent(context=doc_context)
    result = agent.update_prd(dry_run=True)
    assert "F1 >= 0.8" in result.data["markdown"]


def test_update_prd_incluye_tabla_del_backlog(doc_context):
    _proyecto_con_prd(doc_context.root)
    agent = DocumentationAgent(context=doc_context)
    result = agent.update_prd(dry_run=True)
    md = result.data["markdown"]
    assert "| SCOPE-001 | done |" in md
    assert "| MODEL-001 | pending |" in md


def test_update_prd_sin_objetivo_avisa(doc_context):
    agent = DocumentationAgent(context=doc_context)
    result = agent.update_prd(dry_run=True)
    assert "sin definir" in result.data["markdown"]


def test_update_prd_incluye_contratos_gherkin(doc_context):
    _proyecto_con_prd(doc_context.root)
    agent = DocumentationAgent(context=doc_context)
    result = agent.update_prd(dry_run=True)
    assert "MODEL-001.feature" in result.data["markdown"]
    assert "Baseline" in result.data["markdown"]


def test_update_prd_con_backlog_ilegible_avisa(doc_context):
    (doc_context.root / "harness").mkdir()
    (doc_context.root / "harness" / "featureslist.json").write_text(
        "{roto", encoding="utf-8"
    )
    agent = DocumentationAgent(context=doc_context)
    result = agent.update_prd(dry_run=True)
    assert result.warnings
    assert "ilegible" in result.warnings[0]
