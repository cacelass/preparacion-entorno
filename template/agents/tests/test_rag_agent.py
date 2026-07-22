"""
Tests para el RAG agent: chunking, index, search, status.

Se ejecutan sin chromadb real — RagTool.chunk_* y RagTool.status son
deterministas y no requieren chroma. Las acciones index/search usan
chromadb y se skipean si no está disponible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.tools.rag_tool import RagTool


# -- chunking determinista (no requiere chromadb) -------------------------------

SAMPLE_PY = """
def hello(name: str) -> str:
    \"\"\"Saluda a alguien.\"\"\"
    return f"Hola {name}"

class Calculator:
    \"\"\"Calculadora simple.\"\"\"

    def add(self, a: int, b: int) -> int:
        return a + b
"""

SAMPLE_PY_LONG = """
def hello(name: str) -> str:
    \"\"\"Saluda a alguien con un saludo personalizado y amigable.\"\"\"
    greeting = f"Hola {name}"
    return greeting

def goodbye(name: str) -> str:
    \"\"\"Despide a alguien cordialmente.\"\"\"
    farewell = f"Adios {name}"
    return farewell
"""

SAMPLE_MD = """
# Proyecto

Descripción del proyecto.

## Instalación

Cómo instalar.

## Uso

Cómo usar.
"""


class TestChunking:
    def test_chunk_py_functions(self):
        chunks = RagTool.chunk_py(SAMPLE_PY, "test.py")
        assert len(chunks) >= 2
        assert any("def hello" in c["text"] for c in chunks)
        assert any("class Calculator" in c["text"] for c in chunks)
        for c in chunks:
            assert "id" in c
            assert "text" in c
            assert "metadata" in c

    def test_chunk_py_docstring_extracted(self):
        chunks = RagTool.chunk_py(SAMPLE_PY_LONG, "test.py")
        docstring_chunks = [c for c in chunks if c["metadata"].get("section_type") == "docstring"]
        assert len(docstring_chunks) >= 1

    def test_chunk_py_empty(self):
        chunks = RagTool.chunk_py("", "empty.py")
        assert chunks == []

    def test_chunk_md_sections(self):
        chunks = RagTool.chunk_md(SAMPLE_MD, "test.md")
        assert len(chunks) >= 3
        assert any("# Proyecto" in c["text"] for c in chunks)
        assert any("## Instalación" in c["text"] for c in chunks)
        assert any("## Uso" in c["text"] for c in chunks)

    def test_chunk_md_no_headings(self):
        chunks = RagTool.chunk_md("solo texto plano sin headers", "plain.md")
        assert len(chunks) >= 1

    def test_make_chunk_id_stable(self):
        a = RagTool._make_chunk("hello world", "f.py", 1, "function")
        b = RagTool._make_chunk("hello world", "f.py", 1, "function")
        assert a["id"] == b["id"]

    def test_make_chunk_different_source(self):
        a = RagTool._make_chunk("hello world", "f1.py", 1, "function")
        b = RagTool._make_chunk("hello world", "f2.py", 1, "function")
        assert a["id"] != b["id"]

    def test_metadata_file_type(self):
        chunks = RagTool.chunk_py(SAMPLE_PY, "src/mod.py")
        code_chunks = [c for c in chunks if c["metadata"].get("file_type") == "code"]
        assert len(code_chunks) >= 1

    def test_metadata_section_type(self):
        chunks = RagTool.chunk_md("# Title\n\nContent", "doc.md")
        assert any(c["metadata"].get("section_type") in ("heading", "paragraph") for c in chunks)


class TestStatus:
    def test_status_no_chroma(self, monkeypatch):
        monkeypatch.setattr("agents.tools.rag_tool.HAS_CHROMA", False)
        info = RagTool.status(Path("/tmp"))
        assert info.get("available") is False

    def test_status_available(self, monkeypatch):
        monkeypatch.setattr("agents.tools.rag_tool.HAS_CHROMA", True)
        monkeypatch.setattr(
            "agents.tools.rag_tool.RagTool._client",
            lambda root: type("FakeClient", (), {"get_collection": lambda self, name: type("FakeCol", (), {"count": lambda self: 42})()}),
        )
        info = RagTool.status(Path("/tmp"))
        assert info.get("available") is True


# -- index / search requieren chromadb real -- marcamos como integración --------

pytestmark_integration = pytest.mark.skipif(
    not RagTool.available(), reason="chromadb no instalado"
)


class TestSearchIntegration:
    def test_search_empty_index(self, tmp_path):
        if not RagTool.available():
            pytest.skip("chromadb no instalado")
        results = RagTool.search(tmp_path, "test query")
        assert results == []

    def test_index_then_search(self, tmp_path):
        if not RagTool.available():
            pytest.skip("chromadb no instalado")
        (tmp_path / "README.md").write_text("# Test Project\n\nThis is a test project for RAG.")
        result = RagTool.index_project(tmp_path)
        assert result["total_chunks"] > 0
        results = RagTool.search(tmp_path, "test project")
        assert len(results) > 0
        assert any("README.md" in r["source"] for r in results)
