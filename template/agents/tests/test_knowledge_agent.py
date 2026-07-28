from __future__ import annotations

import pytest

from agents.agents.knowledge_agent import KnowledgeAgent
from agents.config import ProjectConfig
from agents.context import SharedContext
from agents.tools.graphify_tool import GraphifyTool


@pytest.fixture
def know_context(tmp_path):
    (tmp_path / "mi_paquete").mkdir()
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def test_status_reports_graphify_availability(know_context):
    agent = KnowledgeAgent(context=know_context)
    result = agent.status()
    assert result.success
    assert isinstance(result.data["graphify_available"], bool)
    assert isinstance(result.data["graph_exists"], bool)


def test_status_warns_when_no_vault(know_context):
    agent = KnowledgeAgent(context=know_context)
    result = agent.status()
    assert any("bóveda de Obsidian" in w for w in result.warnings)


def test_setup_vault_creates_tree(know_context):
    agent = KnowledgeAgent(context=know_context)
    result = agent.setup_vault()
    assert result.success
    assert result.data["created"] is True
    vault_path = know_context.root / "knowledge"
    assert vault_path.exists()
    assert (vault_path / ".obsidian").exists()
    assert (vault_path / "00-index" / "MOC.md").exists()


def test_setup_vault_creates_expected_dirs(know_context):
    agent = KnowledgeAgent(context=know_context)
    result = agent.setup_vault(vault_dir="test_vault")
    assert result.success
    vault = know_context.root / "test_vault"
    for folder in ("00-index", "papers", "code", "docs", "references", "media"):
        assert (vault / folder).exists(), f"Falta carpeta {folder}"


def test_setup_vault_reuses_existing(know_context):
    agent = KnowledgeAgent(context=know_context)
    first = agent.setup_vault()
    assert first.success

    second = agent.setup_vault()
    assert second.success
    assert second.data["created"] is False


def test_setup_vault_custom_dir(know_context):
    agent = KnowledgeAgent(context=know_context)
    result = agent.setup_vault(vault_dir="mi_boveda")
    assert result.success
    assert (know_context.root / "mi_boveda").exists()


def test_setup_vault_create_if_missing_false(know_context):
    agent = KnowledgeAgent(context=know_context)
    result = agent.setup_vault(vault_dir="no_existe", create_if_missing=False)
    assert not result.success


def test_build_fails_if_no_graph(know_context):
    agent = KnowledgeAgent(context=know_context)
    result = agent.build()
    if GraphifyTool.is_available(know_context.root):
        assert result.success
    else:
        assert not result.success


def test_summarize_parents_fails_without_graph(know_context):
    agent = KnowledgeAgent(context=know_context)
    result = agent.summarize_parents()
    assert not result.success
    assert "No hay grafo" in result.message


def test_sync_without_graph(know_context):
    agent = KnowledgeAgent(context=know_context)
    result = agent.sync()
    if GraphifyTool.is_available(know_context.root):
        assert result.success
    else:
        assert result.data.get("skipped") is True


# -- poda del grafo: absorbida de `docsearch` ----------------------------------
# `graphify-out/` tenia DOS duenos declarados: knowledge lo construia y
# docsearch lo podaba. El test de contratos no lo detectaba porque cada uno lo
# describia con una cadena distinta. La poda vive ahora donde el grafo.

def test_prune_falla_sin_grafo(context):
    from agents.agents.knowledge_agent import KnowledgeAgent

    result = KnowledgeAgent(context=context).prune(node_types=["reference"])
    assert not result.success


def test_prune_exige_saber_que_podar(context, monkeypatch):
    from agents.agents.knowledge_agent import KnowledgeAgent

    agent = KnowledgeAgent(context=context)
    monkeypatch.setattr(agent, "_require_graph", lambda accion: None)
    result = agent.prune()
    assert not result.success
    assert "Indica qué podar" in result.message
