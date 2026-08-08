"""
Tests del mantenimiento del corpus de conocimiento (`knowledge_tool` + `rag refresh`).

Cierran el contrato de `rag refresh`: leer `knowledge/sources.json`, verificar
cada fuente contra arXiv (¿versión más reciente?), detectar papers nuevos y —
solo sin `--dry-run` — descargarlos a `knowledge/papers/`, actualizar el
registro y reindexar. La red y el descargador se simulan: lo que se testea es
la lógica, no arXiv.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.tools.knowledge_tool import KnowledgeTool
from agents.tools.rag_tool import _file_type

_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1412.6980v9</id>
    <title>Adam: A Method for Stochastic Optimization</title>
    <summary>We introduce Adam, an algorithm for first-order gradient-based optimization.</summary>
    <published>2014-12-22T00:00:00Z</published>
    <updated>2024-01-18T00:00:00Z</updated>
    <author><name>Diederik P. Kingma</name></author>
  </entry>
</feed>
"""


def _fuentes(tmp_path: Path, version: int = 3) -> dict:
    data = {
        "version": 1,
        "updated": "2026-01-01",
        "topics": [
            {
                "topic": "optimizadores",
                "queries": ["adam optimizer"],
                "sources": [
                    {"arxiv_id": "1412.6980", "title": "Adam", "version": version,
                     "status": "activa", "last_checked": "2026-01-01"},
                ],
            }
        ],
    }
    (tmp_path / "knowledge").mkdir(parents=True, exist_ok=True)
    (tmp_path / "knowledge" / "sources.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return data


def _paper(arxiv_id: str, version: int = 1, publicado: str = "2026-08-01") -> dict:
    return {
        "arxiv_id": arxiv_id,
        "version": version,
        "title": "Paper nuevo",
        "abstract": "resumen",
        "published": publicado + "T00:00:00Z",
        "updated": publicado + "T00:00:00Z",
        "url": f"http://arxiv.org/abs/{arxiv_id}v{version}",
        "authors": [],
    }


# -- parsing Atom ---------------------------------------------------------------
def test_parsea_atom_y_extrae_id_y_version():
    papers = KnowledgeTool._parse_atom(_ATOM)
    assert len(papers) == 1
    assert papers[0]["arxiv_id"] == "1412.6980"
    assert papers[0]["version"] == 9
    assert papers[0]["title"].startswith("Adam")
    assert papers[0]["published"].startswith("2014")


def test_atom_roto_no_revienta():
    assert KnowledgeTool._parse_atom("<no es xml") == []


# -- registro de fuentes --------------------------------------------------------
def test_carga_y_guarda_el_registro(tmp_path):
    _fuentes(tmp_path)
    data = KnowledgeTool.load_sources(tmp_path)
    assert data is not None and data["topics"][0]["topic"] == "optimizadores"


def test_registro_ausente_o_corrupto_devuelve_none(tmp_path):
    assert KnowledgeTool.load_sources(tmp_path) is None
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "sources.json").write_text("{no json", encoding="utf-8")
    assert KnowledgeTool.load_sources(tmp_path) is None


def test_sin_registro_refresh_no_hace_nada(tmp_path):
    informe = KnowledgeTool.refresh(tmp_path, dry_run=True)
    assert "error" in informe


def test_el_fichero_sanea_ids_con_barra():
    assert KnowledgeTool._nombre_fichero("cs/0211133") == "cs-0211133.md"


# -- descarga a markdown -------------------------------------------------------
def test_prefiere_el_html_de_arxiv_cuando_existe(tmp_path, monkeypatch):
    monkeypatch.setattr(KnowledgeTool, "_html_disponible", lambda arxiv_id: True)
    monkeypatch.setattr(
        KnowledgeTool, "_fetch",
        lambda arxiv_id, base: b"<html><body><h1>Titulo</h1><p>Cuerpo del paper.</p></body></html>",
    )
    md = KnowledgeTool.fetch_paper_markdown("9999.0001")
    assert "Cuerpo del paper" in md
    assert "<html" not in md, "el HTML se convierte a texto, no se guarda crudo"


def test_cae_al_pdf_con_markitdown_si_no_hay_html(tmp_path, monkeypatch):
    monkeypatch.setattr(KnowledgeTool, "_html_disponible", lambda arxiv_id: False)
    monkeypatch.setattr(KnowledgeTool, "_fetch", lambda arxiv_id, base: b"%PDF-1.4 fake")
    monkeypatch.setattr(KnowledgeTool, "_pdf_a_markdown", lambda pdf: "# Paper\ncontenido")
    md = KnowledgeTool.fetch_paper_markdown("9999.0001")
    assert md == "# Paper\ncontenido"


def test_sin_html_y_sin_markitdown_falla_con_instruccion(tmp_path, monkeypatch):
    monkeypatch.setattr(KnowledgeTool, "_html_disponible", lambda arxiv_id: False)
    monkeypatch.setattr(KnowledgeTool, "_fetch", lambda arxiv_id, base: b"%PDF-1.4 fake")
    monkeypatch.setattr(KnowledgeTool, "_pdf_a_markdown",
                        lambda pdf: (_ for _ in ()).throw(KnowledgeError("markitdown no instalado")))
    from agents.tools.knowledge_tool import KnowledgeError
    with pytest.raises(KnowledgeError):
        KnowledgeTool.fetch_paper_markdown("9999.0001")


# -- refresh en seco (no escribe nada) -----------------------------------------
def test_dry_run_informa_sin_tocar_nada(tmp_path, monkeypatch):
    _fuentes(tmp_path, version=3)
    monkeypatch.setattr(
        KnowledgeTool, "paper_status",
        lambda arxiv_id, **kw: _paper("1412.6980", version=9, publicado="2022-01-01"),
    )
    monkeypatch.setattr(
        KnowledgeTool, "arxiv_search",
        lambda query, **kw: [_paper("9999.0001", version=1, publicado="2026-08-01")],
    )

    informe = KnowledgeTool.refresh(tmp_path, dry_run=True)

    assert informe["dry_run"] is True
    assert len(informe["updated_sources"]) == 1, "la fuente 1412.6980 tiene v9 > v3"
    assert informe["updated_sources"][0]["desde"] == 3
    assert informe["updated_sources"][0]["hasta"] == 9
    assert len(informe["new_papers"]) == 1
    assert informe["new_papers"][0]["arxiv_id"] == "9999.0001"
    assert informe["errors"] == []
    # no se ha escrito nada
    assert not (tmp_path / "knowledge" / "papers").exists()
    guardado = json.loads((tmp_path / "knowledge" / "sources.json").read_text(encoding="utf-8"))
    assert guardado["topics"][0]["sources"][0]["version"] == 3


def test_dry_run_marca_superada_cuando_hay_version_nueva(tmp_path, monkeypatch):
    _fuentes(tmp_path, version=3)
    monkeypatch.setattr(KnowledgeTool, "paper_status", lambda *a, **kw: _paper("1412.6980", version=9))
    monkeypatch.setattr(KnowledgeTool, "arxiv_search", lambda *a, **kw: [])
    informe = KnowledgeTool.refresh(tmp_path, dry_run=True)
    assert informe["topics"][0]["sources"][0]["status"] == "superada"
    assert informe["topics"][0]["sources"][0]["version_nueva"] == 9


def test_dry_run_respeta_el_filtro_de_topics(tmp_path, monkeypatch):
    _fuentes(tmp_path)
    monkeypatch.setattr(KnowledgeTool, "paper_status", lambda *a, **kw: _paper("1412.6980", version=9))
    monkeypatch.setattr(KnowledgeTool, "arxiv_search", lambda *a, **kw: [_paper("9999.0001")])
    informe = KnowledgeTool.refresh(tmp_path, dry_run=True, topics=["otro"])
    assert informe["topics"] == []
    assert informe["new_papers"] == []


# -- refresh real (escribe y reindexa) -----------------------------------------
def test_refresh_descarga_actualiza_y_reindexa(tmp_path, monkeypatch):
    _fuentes(tmp_path)
    monkeypatch.setattr(KnowledgeTool, "paper_status", lambda *a, **kw: _paper("1412.6980", version=3))
    monkeypatch.setattr(
        KnowledgeTool, "arxiv_search",
        lambda query, **kw: [_paper("9999.0001", publicado="2026-08-01")],
    )
    monkeypatch.setattr(KnowledgeTool, "fetch_paper_markdown", lambda arxiv_id: f"# {arxiv_id}\ncontenido")
    monkeypatch.setattr(
        "agents.tools.knowledge_tool.RagTool.index_project",
        lambda root: {"total_chunks": 5, "new_chunks": 1},
    )

    informe = KnowledgeTool.refresh(tmp_path, dry_run=False)

    ruta = tmp_path / "knowledge" / "papers" / "optimizadores" / "9999.0001.md"
    assert ruta.exists(), "el paper nuevo se descarga a knowledge/papers/"
    assert "contenido" in ruta.read_text(encoding="utf-8")
    assert len(informe["downloads"]) == 1
    assert informe["reindex"]["total_chunks"] == 5

    guardado = json.loads((tmp_path / "knowledge" / "sources.json").read_text(encoding="utf-8"))
    ids = [s["arxiv_id"] for s in guardado["topics"][0]["sources"]]
    assert "9999.0001" in ids, "el nuevo paper entra en el registro"
    assert guardado["topics"][0]["sources"][0]["last_checked"].startswith("2026")


def test_refresh_sin_red_falla_de_forma_controlada(tmp_path, monkeypatch):
    _fuentes(tmp_path)
    monkeypatch.setattr(KnowledgeTool, "paper_status", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("sin red")))
    monkeypatch.setattr(KnowledgeTool, "arxiv_search", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("sin red")))
    informe = KnowledgeTool.refresh(tmp_path, dry_run=True)
    assert informe["errors"], "los fallos de red se reportan, no se tragan"
    assert informe["topics"][0]["sources"][0]["error"]


# -- el corpus se etiqueta como knowledge en el RAG ----------------------------
def test_el_corpus_se_clasifica_como_knowledge():
    assert _file_type("knowledge/matematicas/probabilidad.md") == "knowledge"
    assert _file_type("knowledge/papers/optimizadores/1412.6980.md") == "knowledge"
    assert _file_type("README.md") == "doc"


def test_refresh_con_topics_como_string_se_normaliza(context, monkeypatch):
    """Desde la CLI los topics llegan como texto 'a,b'; la acción los separa."""
    _fuentes(context.root)
    monkeypatch.setattr(KnowledgeTool, "paper_status", lambda *a, **kw: _paper("1412.6980", version=3))
    monkeypatch.setattr(KnowledgeTool, "arxiv_search", lambda *a, **kw: [])
    informe = KnowledgeTool.refresh(context.root, dry_run=True, topics="optimizadores")
    assert [t["topic"] for t in informe["topics"]] == ["optimizadores"]


def test_la_accion_refresh_del_agente_expone_dry_run(context):
    from agents.agents.rag_agent import RagAgent

    agente = RagAgent(context=context)
    assert "refresh" in agente.actions()
    assert any("papers" in a for a in agente.action_aliases()["refresh"])
