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
        # Ojo: el doble tiene que ser una INSTANCIA. Devolver la clase deja
        # `get_collection` sin enlazar, y el `name` que le pasa `_collection`
        # cae en el `self` → TypeError que no tiene nada que ver con el codigo.
        fake_collection = type("FakeCol", (), {"count": lambda self: 42})()
        fake_client = type(
            "FakeClient", (), {"get_collection": lambda self, name: fake_collection}
        )()
        monkeypatch.setattr("agents.tools.rag_tool.RagTool._client", lambda root: fake_client)
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


# -- memoria del arnés como fuente indexable ------------------------------------

class TestHarnessMemoryIsIndexed:
    """
    El histórico del arnés (progress/) y el backlog son justo lo que un agente
    nuevo necesita poder preguntar en lenguaje natural ("¿por qué elegimos
    esto?") sin releerlo entero. Si dejan de indexarse, la memoria del arnés
    deja de ser buscable y nadie se entera.
    """

    def test_progress_se_clasifica_como_harness(self):
        from agents.tools.rag_tool import _file_type

        assert _file_type("progress/history.md") == "harness"
        assert _file_type("progress/explorer-DATA-001.md") == "harness"
        assert _file_type("featureslist.json") == "harness"

    def test_otras_fuentes_no_se_clasifican_como_harness(self):
        from agents.tools.rag_tool import _file_type

        assert _file_type("README.md") == "doc"
        assert _file_type("vault/00_META/IA_index.md") == "vault"
        assert _file_type("agents/prompts/git_agent.md") == "prompt"
        assert _file_type("mi_paquete/data/loader.py") == "code"

    def test_index_recoge_progress_y_backlog(self, tmp_path):
        if not RagTool.available():
            pytest.skip("chromadb no instalado")
        (tmp_path / "progress").mkdir()
        (tmp_path / "progress" / "history.md").write_text(
            "# Historial\n\n## DATA-001 — EDA\n\n"
            "Elegimos K=4 porque el silhouette caía a partir de ahí. " * 5
        )
        (tmp_path / "featureslist.json").write_text(
            '{"features": [{"id": "X-1", "title": "Una feature de prueba", '
            '"description": "' + "descripción larga " * 20 + '", '
            '"acceptance_criteria": ["make test pasa"], "status": "pending"}]}'
        )
        RagTool.index_project(tmp_path)
        sources = {r["source"] for r in RagTool.search(tmp_path, "por qué elegimos K=4")}
        assert any("progress/" in s or s == "featureslist.json" for s in sources)
