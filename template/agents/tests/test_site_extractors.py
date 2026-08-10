"""
Tests de los extractores site-aware (`rag index_urls`).

Cierran el contrato de OMP-005: GitHub, Stack Overflow y arXiv se indexan con
estructura (título, secciones, código, enlaces) en lugar de HTML plano, y el
resto de URLs siguen por el convertidor genérico. Sin red: todo se simula.
"""

from __future__ import annotations

import pytest

from agents.tools.site_extractors import (
    extract,
    github_raw_url,
    site_kind,
    _stackoverflow_a_markdown,
)


# -- detección ----------------------------------------------------------------

def test_site_kind_detecta_github():
    assert site_kind("https://github.com/owner/repo/blob/main/README.md") == "github"


def test_site_kind_detecta_stackoverflow():
    assert site_kind("https://stackoverflow.com/questions/12345/cosa") == "stackoverflow"


def test_site_kind_detecta_arxiv():
    assert site_kind("https://arxiv.org/abs/1412.6980") == "arxiv"


def test_site_kind_devuelve_none_para_otras_urls():
    assert site_kind("https://docs.pola.rs/api/") is None


# -- GitHub -------------------------------------------------------------------

def test_github_raw_url_readme_por_defecto():
    assert github_raw_url("https://github.com/cacelass/dskit") == (
        "https://raw.githubusercontent.com/cacelass/dskit/HEAD/README.md"
    )


def test_github_raw_url_resuelve_blob():
    assert github_raw_url("https://github.com/o/r/blob/main/src/foo.py") == (
        "https://raw.githubusercontent.com/o/r/HEAD/src/foo.py"
    )


def test_github_no_match_devuelve_none():
    assert github_raw_url("https://github.com") is None


def test_github_extrae_readme(monkeypatch):
    class _Resp:
        status = 200
        text = "# Mi repo\n\nCodigo: `x = 1`\n"

    monkeypatch.setattr(
        "agents.tools.site_extractors.RestTool.get", lambda *a, **kw: _Resp()
    )
    md = extract("https://github.com/o/r", "<html/>")
    assert md is not None
    assert md.startswith("# o/r — README.md")
    assert "> Fuente: https://github.com/o/r" in md
    assert "`x = 1`" in md


def test_github_raw_no_200_devuelve_none(monkeypatch):
    class _Resp:
        status = 404
        text = "Not found"

    monkeypatch.setattr(
        "agents.tools.site_extractors.RestTool.get", lambda *a, **kw: _Resp()
    )
    assert extract("https://github.com/o/r", "<html/>") is None


# -- Stack Overflow -----------------------------------------------------------

_SO_HTML = """
<html><body>
<a class="question-hyperlink">¿Cómo ordenar una lista en Python?</a>
<div class="post-text">
<p>Quiero ordenar <code>x = [3,1,2]</code>. <a href="/a/1">ver ejemplo</a></p>
<pre><code>lista.sort()</code></pre>
</div>
<div class="post-text">
<p>Usa <code>sorted()</code>.</p>
</div>
</body></html>
"""


def test_stackoverflow_extrae_titulo_codigo_y_enlaces():
    md = _stackoverflow_a_markdown(_SO_HTML)
    assert "¿Cómo ordenar una lista en Python?" in md, "debe conservar el título"
    assert "lista.sort()" in md, "el código no puede perderse"
    assert "```" in md, "los bloques de código van en fences"
    assert "[ver ejemplo]" in md and "href" not in md, "los enlaces se conservan como markdown"


def test_stackoverflow_html_roto_no_revienta():
    assert _stackoverflow_a_markdown("<no es html") == ""


# -- arXiv --------------------------------------------------------------------

def test_arxiv_reutiliza_knowledge_tool(monkeypatch):
    monkeypatch.setattr(
        "agents.tools.site_extractors.KnowledgeTool",
        type("KT", (), {"fetch_paper_markdown": staticmethod(lambda aid: f"# Paper {aid}")}),
    )
    md = extract("https://arxiv.org/abs/1412.6980", "<html/>")
    assert md == "# Paper 1412.6980"


def test_arxiv_sin_id_devuelve_none():
    assert extract("https://arxiv.org/", "<html/>") is None


# -- fallback: el resto sigue por el convertidor genérico -----------------------

def test_extract_no_siteaware_devuelve_none():
    assert extract("https://docs.pola.rs/api/", "<html>doc</html>") is None
