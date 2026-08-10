"""
Tests del banco de memoria (`MemoryTool`) con scoping y edición por id.

Cierran el contrato de OMP-002: cada entrada lleva un scope (global |
per-proyecto, por defecto per-proyecto) y se puede editar por id
(update/forget/invalidate) sin reescribir el resto del banco.
"""

from __future__ import annotations

import pytest

from agents.tools.memory_tool import MemoryTool


def test_write_con_scope(tmp_path):
    MemoryTool.write(tmp_path, "facts", "k", "v", scope="global")
    entry = MemoryTool.read(tmp_path, "facts", "k")
    assert entry["scope"] == "global"


def test_default_scope_es_per_proyecto(tmp_path):
    MemoryTool.write(tmp_path, "facts", "k", "v")
    assert MemoryTool.recall(tmp_path, "k")["scope"] == "per-proyecto"


def test_write_scope_invalido_rechazado(tmp_path):
    with pytest.raises(ValueError):
        MemoryTool.write(tmp_path, "facts", "k", "v", scope="otro")


def test_edit_actualiza_value_y_scope(tmp_path):
    MemoryTool.write(tmp_path, "facts", "k", "v")
    edited = MemoryTool.edit(tmp_path, "k", value="v2", scope="global")
    assert edited is not None
    assert edited["value"] == "v2"
    assert edited["scope"] == "global"
    assert MemoryTool.recall(tmp_path, "k")["value"] == "v2"


def test_edit_invalida_con_ttl_cero(tmp_path):
    MemoryTool.write(tmp_path, "facts", "k", "v")
    assert MemoryTool.edit(tmp_path, "k", ttl=0) is None
    assert MemoryTool.recall(tmp_path, "k") is None, "invalidada no puede resucitar"


def test_edit_inexistente_devuelve_none(tmp_path):
    assert MemoryTool.edit(tmp_path, "no_existe", value="x") is None


def test_edit_scope_invalido_rechazado(tmp_path):
    MemoryTool.write(tmp_path, "facts", "k", "v")
    with pytest.raises(ValueError):
        MemoryTool.edit(tmp_path, "k", scope="otro")


def test_search_filtra_por_scope(tmp_path):
    MemoryTool.write(tmp_path, "facts", "a", "alpha", scope="global")
    MemoryTool.write(tmp_path, "facts", "b", "beta")
    assert len(MemoryTool.search(tmp_path, scope="global")) == 1
    assert len(MemoryTool.search(tmp_path, scope="per-proyecto")) == 1
    assert len(MemoryTool.search(tmp_path)) == 2, "sin filtro devuelve todos"
