"""
Tests de la acción `memory_edit` y del scoping del agente `memory`.

Cierran el contrato de OMP-002 en la capa del agente: update/forget/invalidate
por id, scope en note/search/status, y la herencia del banco por los subagentes
(banco compartido en agents/workspace/memory/).
"""

from __future__ import annotations

import pytest

from agents.agents.memory_agent import MemoryAgent
from agents.tools.memory_tool import MemoryTool


@pytest.fixture
def agent(context):
    return MemoryAgent(context=context)


def test_note_con_scope(agent):
    result = agent.note(key="k", value="v", scope="global")
    assert result.success
    assert result.data["scope"] == "global"


def test_note_scope_invalido(agent):
    result = agent.note(key="k", value="v", scope="otro")
    assert not result.success


def test_memory_edit_update_por_id(agent):
    agent.note(key="k", value="v")
    result = agent.memory_edit(id="k", value="v2", scope="global")
    assert result.success
    entry = MemoryTool.recall(agent._ws, "k")
    assert entry["value"] == "v2"
    assert entry["scope"] == "global"


def test_memory_edit_forget(agent):
    agent.note(key="k", value="v")
    result = agent.memory_edit(id="k", action="forget")
    assert result.success
    assert MemoryTool.recall(agent._ws, "k") is None


def test_memory_edit_invalidate(agent):
    agent.note(key="k", value="v")
    result = agent.memory_edit(id="k", action="invalidate")
    assert result.success
    assert MemoryTool.recall(agent._ws, "k") is None


def test_memory_edit_accion_invalida_pide_datos(agent):
    result = agent.memory_edit(id="k", action="borrar")
    assert not result.success
    assert result.needs


def test_memory_edit_inexistente(agent):
    result = agent.memory_edit(id="no_existe")
    assert not result.success


def test_search_filtra_por_scope(agent):
    agent.note(key="a", value="alpha", scope="global")
    agent.note(key="b", value="beta")
    assert len(agent.search(query="", scope="global").data) == 1
    assert len(agent.search(query="").data) == 2


def test_status_incluye_conteo_por_scope(agent):
    agent.note(key="a", value="x", scope="global")
    agent.note(key="b", value="y")
    result = agent.status()
    assert result.data["entries_by_scope"]["global"] == 1
    assert result.data["entries_by_scope"]["per-proyecto"] == 1


def test_banco_compartido_lo_heredan_los_subagentes(context):
    """El banco vive en agents/workspace/memory/ (shared): quien lo lea desde
    otro agente hereda lo escrito. Esto es la "herencia" de scope de OMP-002."""
    MemoryAgent(context=context).note(key="decision", value="usar XGBoost")
    from agents.tools.memory_tool import MemoryTool

    bank = MemoryTool.bank_dir(context.agent_workspace("memory"))
    assert (bank / "bank.json").exists()
    assert MemoryTool.recall(context.agent_workspace("memory"), "decision") is not None
