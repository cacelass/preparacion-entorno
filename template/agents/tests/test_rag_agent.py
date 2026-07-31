"""
Tests para el RAG agent: chunking, index, search, status.

El troceado es determinista y no necesita chroma. Lo que sí lo necesita
(index/search) está marcado con `requiere_chroma`: en CI el extra `rag` se
instala a propósito, porque la versión anterior de estos tests se saltaba en
silencio todo lo que tocaba chromadb y así se coló un `index` que ni siquiera
llegaba a crear la colección.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.tools.rag_tool import (
    _MAX_CHUNK_CHARS,
    RagTool,
    _Bm25,
    _embedder_id,
    _embedding_function,
    _html_a_texto,
    _tokenizar,
)

requiere_chroma = pytest.mark.skipif(
    not RagTool.available(), reason="chromadb no instalado (uv sync --extra rag)"
)


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

    def test_chunk_py_docstring_va_dentro_de_su_funcion(self):
        """
        El docstring NO se indexa aparte. Duplicarlo inflaba el índice hasta que
        el 70% de los chunks eran docstrings, y como son prosa corta y limpia
        ganaban el coseno a la implementación: se preguntaba "cómo se entrena el
        modelo" y salía todo menos `train_model.py`.
        """
        chunks = RagTool.chunk_py(SAMPLE_PY_LONG, "test.py")
        assert not [c for c in chunks if c["metadata"]["section_type"] == "docstring"]
        hello = [c for c in chunks if "def hello" in c["text"]]
        assert hello and "saludo personalizado" in hello[0]["text"]

    def test_chunk_py_tipos_de_seccion(self):
        chunks = RagTool.chunk_py(SAMPLE_PY, "test.py")
        tipos = {c["metadata"]["section_type"] for c in chunks}
        assert "function" in tipos and "class" in tipos

    def test_chunk_py_lleva_la_ruta_del_fichero(self):
        """Sin la ruta dentro del texto, un `def fit()` es un chunk anónimo."""
        chunks = RagTool.chunk_py(SAMPLE_PY, "modelos/train_model.py")
        assert all("modelos/train_model.py" in c["text"] for c in chunks)

    def test_chunk_py_metodos_llevan_su_clase(self):
        chunks = RagTool.chunk_py(SAMPLE_PY, "test.py")
        add = [c for c in chunks if "def add" in c["text"]]
        assert add and "class Calculator" in add[0]["text"]

    def test_chunk_py_empty(self):
        chunks = RagTool.chunk_py("", "empty.py")
        assert chunks == []

    def test_chunk_md_sections(self):
        chunks = RagTool.chunk_md(SAMPLE_MD, "test.md")
        assert len(chunks) >= 3
        assert any("# Proyecto" in c["text"] for c in chunks)
        assert any("## Instalación" in c["text"] for c in chunks)
        assert any("## Uso" in c["text"] for c in chunks)

    def test_chunk_md_migas_de_pan(self):
        """Una subsección arrastra el título de su padre o no dice de qué va."""
        chunks = RagTool.chunk_md(SAMPLE_MD, "test.md")
        uso = [c for c in chunks if "## Uso" in c["text"]]
        assert uso and "Proyecto" in uso[0]["text"]

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

    def test_id_distingue_textos_con_prefijo_comun(self):
        """
        El id hasheaba solo los primeros 80 caracteres: dos chunks con el mismo
        arranque colisionaban y `add` se quedaba en silencio con el primero, así
        que la edición se perdía sin error ninguno.
        """
        a = RagTool._make_chunk("A" * 200 + "PRIMERO", "f.py", 3, "heading")
        b = RagTool._make_chunk("A" * 200 + "SEGUNDO", "f.py", 3, "heading")
        assert a["id"] != b["id"]

    def test_metadata_file_type(self):
        chunks = RagTool.chunk_py(SAMPLE_PY, "src/mod.py")
        code_chunks = [c for c in chunks if c["metadata"].get("file_type") == "code"]
        assert len(code_chunks) >= 1

    def test_metadata_section_type(self):
        chunks = RagTool.chunk_md("# Title\n\nContent", "doc.md")
        assert any(c["metadata"].get("section_type") in ("heading", "paragraph") for c in chunks)


class TestTechoDelEmbedder:
    """
    El embedder por defecto trunca a 256 tokens (~1.000 chars). Un chunk más
    largo se guarda entero pero solo se vectoriza su principio: el resto queda
    en la base e irrecuperable. Sobre este mismo repo eran el 22% del corpus.
    """

    def test_ningun_chunk_supera_el_techo(self):
        md = "# Guía\n\n" + ("Este es un párrafo largo de documentación técnica. " * 400)
        for c in RagTool.chunk_md(md, "g.md"):
            assert c["metadata"]["char_len"] <= _MAX_CHUNK_CHARS

    def test_techo_tambien_en_codigo(self):
        py = "def enorme():\n" + "    x = 1  # relleno de una función larguísima\n" * 400
        for c in RagTool.chunk_py(py, "g.py"):
            assert c["metadata"]["char_len"] <= _MAX_CHUNK_CHARS

    def test_partir_no_pierde_contenido(self):
        texto = "\n".join(f"línea número {i} con algo de relleno" for i in range(300))
        piezas = RagTool._partir(texto)
        assert len(piezas) > 1
        recompuesto = " ".join(p for p, _ in piezas)
        assert "línea número 0 " in recompuesto
        assert "línea número 299 " in recompuesto

    def test_partir_solapa(self):
        texto = "\n".join(f"línea número {i} con algo de relleno" for i in range(300))
        piezas = [p for p, _ in RagTool._partir(texto)]
        cola = piezas[0].split("\n")[-1]
        assert cola in piezas[1]

    def test_linea_unica_gigantesca_no_cuelga(self):
        """Un JSON minificado en una sola línea no puede bloquear el troceado."""
        piezas = RagTool._partir("x" * 9000)
        assert piezas and all(len(p) <= _MAX_CHUNK_CHARS for p, _ in piezas)

    def test_chunk_by_size_no_tira_la_cola(self):
        """La última tanda se descartaba si no llegaba al suelo de 100 chars."""
        texto = "Frase de prueba número uno. " * 70
        chunks = RagTool._chunk_by_size(texto, "t.txt")
        indexado = sum(c["metadata"]["char_len"] for c in chunks)
        assert indexado >= len(texto) * 0.95

    def test_chunk_by_size_line_es_una_linea_real(self):
        contenido = "Una frase. " * 200  # todo en una sola línea física
        for c in RagTool._chunk_by_size(contenido, "t.txt"):
            assert c["metadata"]["line"] <= contenido.count("\n")


class TestLexico:
    def test_tokenizar_ignora_acentos(self):
        assert _tokenizar("¿Cómo se entrena?") == _tokenizar("como se entrena")

    def test_bm25_ordena_por_relevancia(self):
        docs = [
            _tokenizar("el modelo se entrena con gradient boosting"),
            _tokenizar("la api expone un endpoint de salud"),
        ]
        marcadores = _Bm25(docs).puntua(_tokenizar("entrenar el modelo"))
        assert marcadores[0] > marcadores[1]

    def test_bm25_corpus_vacio(self):
        assert _Bm25([]).puntua(["algo"]) == []

    def test_rrf_premia_lo_que_sale_en_los_dos_rankings(self):
        """Salir en los dos rankings pesa más que ir primero en uno solo."""
        fusion = RagTool._rrf([["solo_vector", "en_ambos"], ["en_ambos", "solo_lexico"]])
        assert fusion["en_ambos"] > fusion["solo_vector"]
        assert fusion["en_ambos"] > fusion["solo_lexico"]

    def test_html_se_convierte_a_texto(self):
        html = "<html><head><style>p{color:red}</style></head><body><p>Hola mundo</p></body></html>"
        texto = _html_a_texto(html)
        assert "Hola mundo" in texto
        assert "color:red" not in texto and "<p>" not in texto

    def test_markdown_pasa_intacto(self):
        md = "# Título\n\nUn párrafo."
        assert _html_a_texto(md) == md


class TestEmbedder:
    """
    Los dos embedders dan vectores de 384 dimensiones, así que mezclar índices
    NO produce error de chroma: produce resultados sin sentido. La detección es
    la única barrera, y por eso se prueba aunque no haya sentence-transformers.
    """

    def test_id_por_defecto(self, monkeypatch):
        monkeypatch.delenv("DSKIT_RAG_EMBEDDER", raising=False)
        assert _embedder_id() == "onnx"

    def test_id_desconocido_no_rompe(self, monkeypatch):
        monkeypatch.setenv("DSKIT_RAG_EMBEDDER", "inventado")
        assert _embedder_id() == "onnx"

    def test_id_multilingue(self, monkeypatch):
        monkeypatch.setenv("DSKIT_RAG_EMBEDDER", "multilingual")
        assert _embedder_id() == "multilingual"

    def test_onnx_no_pide_funcion_de_embedding(self, monkeypatch):
        """`None` = no se pasa el kwarg y chroma usa su ONNX. Sin importar torch."""
        monkeypatch.delenv("DSKIT_RAG_EMBEDDER", raising=False)
        assert _embedding_function(_embedder_id()) is None

    def test_desajuste_detectado(self, monkeypatch):
        coleccion = type(
            "Col", (), {"metadata": {"embedder": "multilingual"}, "count": lambda self: 10}
        )()
        monkeypatch.setenv("DSKIT_RAG_EMBEDDER", "onnx")
        aviso = RagTool._desajuste_embedder(coleccion)
        assert aviso and "rebuild" in aviso

    def test_indice_vacio_no_es_desajuste(self, monkeypatch):
        coleccion = type(
            "Col", (), {"metadata": {"embedder": "multilingual"}, "count": lambda self: 0}
        )()
        monkeypatch.setenv("DSKIT_RAG_EMBEDDER", "onnx")
        assert RagTool._desajuste_embedder(coleccion) is None

    @requiere_chroma
    def test_embedder_no_instalado_da_mensaje_accionable(self, tmp_path, monkeypatch):
        """
        Pedir el embedder multilingüe sin tenerlo instalado sacaba una traza de
        chroma («The sentence_transformers python package is not installed»)
        desde dentro de `get_or_create_collection`. Tiene que salir como error
        del agente, con el comando que lo arregla.
        """
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("sentence-transformers instalado: este camino no aplica")

        monkeypatch.setenv("DSKIT_RAG_EMBEDDER", "multilingual")
        resultado = RagTool.search(tmp_path, "contenido")
        assert "error" in resultado[0]
        assert "rag_multilingual" in resultado[0]["error"]


class TestStatus:
    def test_status_no_chroma(self, monkeypatch):
        monkeypatch.setattr("agents.tools.rag_tool.HAS_CHROMA", False)
        info = RagTool.status(Path("/tmp"))
        assert info.get("available") is False

    @requiere_chroma
    def test_status_available(self, tmp_path):
        info = RagTool.status(tmp_path)
        assert info.get("available") is True
        assert info.get("total_chunks") == 0


# -- index / search: requieren chromadb real (CI instala --extra rag) -----------


@requiere_chroma
class TestIndexIntegration:
    def test_colección_se_crea_a_la_primera(self, tmp_path):
        """
        `get_collection` + `except ValueError` no capturaba el `NotFoundError`
        de chroma moderno, así que la colección no llegaba a crearse nunca y el
        primer `index` de cualquier proyecto nuevo reventaba.
        """
        (tmp_path / "README.md").write_text("# Test\n\n" + "Contenido del proyecto. " * 20)
        resultado = RagTool.index_project(tmp_path)
        assert "error" not in resultado
        assert resultado["total_chunks"] > 0

    def test_search_empty_index(self, tmp_path):
        assert RagTool.search(tmp_path, "test query") == []

    def test_index_then_search(self, tmp_path):
        (tmp_path / "README.md").write_text("# Test Project\n\nThis is a test project for RAG.")
        result = RagTool.index_project(tmp_path)
        assert result["total_chunks"] > 0
        results = RagTool.search(tmp_path, "test project")
        assert len(results) > 0
        assert any("README.md" in r["source"] for r in results)

    def test_reindexar_sin_cambios_no_reembebe(self, tmp_path):
        (tmp_path / "README.md").write_text("# Test\n\n" + "Contenido estable. " * 20)
        RagTool.index_project(tmp_path)
        segunda = RagTool.index_project(tmp_path)
        assert segunda["new_chunks"] == 0
        assert segunda["unchanged_files"] >= 1

    def test_editar_un_fichero_sustituye_sus_chunks(self, tmp_path):
        """
        El add-only dejaba el chunk viejo dentro para siempre: el índice
        acumulaba versiones obsoletas de `progress/` a cada feature cerrada.
        """
        doc = tmp_path / "README.md"
        doc.write_text("# Test\n\n" + "Usamos regresión logística. " * 20)
        RagTool.index_project(tmp_path)
        doc.write_text("# Test\n\n" + "Usamos gradient boosting. " * 20)
        RagTool.index_project(tmp_path)

        textos = " ".join(r["text"] for r in RagTool.search(tmp_path, "qué modelo usamos"))
        assert "gradient boosting" in textos
        assert "regresión logística" not in textos

    def test_borrar_un_fichero_lo_saca_del_indice(self, tmp_path):
        (tmp_path / "progress").mkdir()
        historia = tmp_path / "progress" / "history.md"
        historia.write_text("# Historial\n\n" + "Decisión antigua sobre el pipeline. " * 20)
        (tmp_path / "README.md").write_text("# Test\n\n" + "Documento que se queda. " * 20)
        RagTool.index_project(tmp_path)

        historia.unlink()
        resultado = RagTool.index_project(tmp_path)
        assert resultado["deleted_chunks"] > 0
        fuentes = {r["source"] for r in RagTool.search(tmp_path, "decisión sobre el pipeline")}
        assert not any(s.startswith("progress/") for s in fuentes)

    def test_rebuild_parte_de_cero(self, tmp_path):
        (tmp_path / "README.md").write_text("# Test\n\n" + "Contenido. " * 20)
        primera = RagTool.index_project(tmp_path)
        rehecho = RagTool.index_project(tmp_path, rebuild=True)
        assert rehecho["total_chunks"] == primera["total_chunks"]
        assert rehecho["new_chunks"] == primera["new_chunks"]

    def test_index_url_reemplaza_en_vez_de_duplicar(self, tmp_path):
        RagTool.index_url(tmp_path, "https://ej.com", "# Doc\n\n" + "Versión uno. " * 20)
        segunda = RagTool.index_url(tmp_path, "https://ej.com", "# Doc\n\n" + "Versión dos. " * 20)
        assert segunda["replaced"] > 0
        textos = " ".join(r["text"] for r in RagTool.search(tmp_path, "versión"))
        assert "Versión uno" not in textos


@requiere_chroma
class TestBusquedaHibrida:
    CORPUS = (
        "# Entrenamiento\n\n"
        + "El script train_model.py ajusta un GradientBoostingClassifier. " * 8
        + "\n\n# Monitorización\n\n"
        + "Aquí se calcula el drift de las variables en producción. " * 8
    )

    def test_hibrido_encuentra_por_termino_literal(self, tmp_path):
        """
        El embedder por defecto está entrenado en inglés y la documentación va
        en español: media señal útil es literal. Sin la rama léxica, buscar un
        identificador exacto podía no devolverlo.
        """
        (tmp_path / "README.md").write_text(self.CORPUS)
        RagTool.index_project(tmp_path)
        resultados = RagTool.search(tmp_path, "GradientBoostingClassifier", top_k=3)
        assert any("GradientBoosting" in r["text"] for r in resultados)

    def test_min_score_filtra(self, tmp_path):
        (tmp_path / "README.md").write_text(self.CORPUS)
        RagTool.index_project(tmp_path)
        assert RagTool.search(tmp_path, "entrenamiento", min_score=1.1) == []

    def test_hybrid_false_sigue_funcionando(self, tmp_path):
        (tmp_path / "README.md").write_text(self.CORPUS)
        RagTool.index_project(tmp_path)
        assert RagTool.search(tmp_path, "drift", hybrid=False)


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

    @requiere_chroma
    def test_index_recoge_progress_y_backlog(self, tmp_path):
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
