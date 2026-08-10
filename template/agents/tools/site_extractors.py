"""
agents.tools.site_extractors — Markdown estructurado para sitios conocidos.

`rag index_urls` convierte HTML genérico a texto plano (`_html_a_texto`), que
pierde estructura. Para GitHub, Stack Overflow y arXiv un extractor específico
devuelve markdown con título, secciones, código y enlaces — lo que el RAG
puede indexar y el `lider` puede citar sin perder de dónde salió.

Cada extractor es stdlib (`html.parser`, `urllib`), sin dependencias nuevas.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from agents.tools.knowledge_tool import KnowledgeTool
from agents.tools.rest_tool import RestTool

_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(?P<id>[^v?#/]+(?:/[^v?#/]+)?)")
_GITHUB_RE = re.compile(r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/#]+)(?P<path>/[^#?]*)?")
_SO_RE = re.compile(r"stackoverflow\.com/questions/(?P<id>\d+)")

_IGNORE_TAGS = {"script", "style", "nav", "footer", "form", "head", "header", "aside"}


def site_kind(url: str) -> str | None:
    """'arxiv' | 'github' | 'stackoverflow' | None según el dominio."""
    if _ARXIV_RE.search(url):
        return "arxiv"
    if _GITHUB_RE.search(url):
        return "github"
    if _SO_RE.search(url):
        return "stackoverflow"
    return None


def extract(url: str, html: str) -> str | None:
    """Markdown estructurado del sitio, o None si no aplica ningún extractor.

    `html` es el contenido ya descargado; los extractores que necesitan otra
    fuente (GitHub raw, arXiv) la descargan ellos mismos.
    """
    kind = site_kind(url)
    if kind == "stackoverflow":
        return _stackoverflow_a_markdown(html)
    if kind == "github":
        return _github_a_markdown(url)
    if kind == "arxiv":
        return _arxiv_a_markdown(url)
    return None


# -- Stack Overflow -----------------------------------------------------------

class _Markdownify(HTMLParser):
    """Convierte un fragmento HTML a markdown conservando enlaces, listas y
    bloques de código. Suficiente para preguntas/respuestas de SO."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._ignore_depth = 0
        self._pre = 0
        self._list = 0
        self._link_href: str | None = None
        self._link_text: list[str] = []

    def _emit(self, text: str) -> None:
        if self._ignore_depth == 0:
            self.out.append(text)

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs = dict(attrs)
        if tag in _IGNORE_TAGS:
            self._ignore_depth += 1
            return
        if tag == "pre":
            self._pre += 1
            self._emit("\n```\n")
        elif tag == "code" and not self._pre:
            self._emit("`")
        elif tag == "a":
            self._link_href = attrs.get("href")
            self._link_text = []
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("*")
        elif tag in ("h1", "h2", "h3", "h4"):
            self._emit("\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self._emit("\n- ")
        elif tag in ("ul", "ol"):
            self._emit("\n")
        elif tag == "br":
            self._emit("\n")
        elif tag in ("p", "div"):
            self._emit("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _IGNORE_TAGS:
            if self._ignore_depth > 0:
                self._ignore_depth -= 1
            return
        if tag == "pre":
            self._pre = max(0, self._pre - 1)
            self._emit("\n```\n")
        elif tag == "code" and not self._pre:
            self._emit("`")
        elif tag == "a":
            if self._link_href and self._link_text:
                texto = "".join(self._link_text)
                self._emit(f"[{texto}]({self._link_href})")
            self._link_href = None
            self._link_text = []
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("*")
        elif tag in ("h1", "h2", "h3", "h4", "p", "div"):
            self._emit("\n\n")

    def handle_data(self, data: str) -> None:
        if self._ignore_depth:
            return
        if self._pre:
            self.out.append(data)
            return
        if self._link_href is not None:
            self._link_text.append(data)
        self._emit(data)


class _StackOverflow(HTMLParser):
    """Extrae el título de la pregunta y el cuerpo de las respuestas."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.posts: list[str] = []
        self._in_title = False
        self._in_post = 0
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        cls = dict(attrs).get("class", "")
        if "question-hyperlink" in cls:
            self._in_title = True
            return
        if "post-text" in cls:
            self._in_post += 1
            self._buf = []
            return
        if self._in_post:
            self._buf.append(f"<{tag}" + "".join(
                f' {k}="{v}"' for k, v in attrs if v
            ) + ">")

    def handle_endtag(self, tag: str) -> None:
        if self._in_title and tag == "a":
            self._in_title = False
            return
        if self._in_post:
            if tag == "div":
                self._in_post -= 1
                if self._in_post == 0:
                    md = _Markdownify()
                    md.feed("".join(self._buf))
                    self.posts.append(md.out and "".join(md.out).strip() or "")
            else:
                self._buf.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif self._in_post:
            self._buf.append(data)


def _stackoverflow_a_markdown(html: str) -> str:
    parser = _StackOverflow()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 — HTML roto no debe tumbar el indexado
        return ""
    parts: list[str] = []
    if parser.title.strip():
        parts.append(f"# {parser.title.strip()}\n")
    for i, post in enumerate(p for p in parser.posts if p):
        parts.append(f"## {'Pregunta' if i == 0 else f'Respuesta {i}'}\n{post}")
    return "\n\n".join(parts).strip()


# -- GitHub -------------------------------------------------------------------

def github_raw_url(url: str) -> str | None:
    """URL raw del README (o del fichero) de un repo de GitHub, o None."""
    match = _GITHUB_RE.search(url)
    if not match:
        return None
    owner, repo = match.group("owner"), match.group("repo")
    ruta = (match.group("path") or "").strip("/")
    if ruta.endswith(".git"):
        ruta = ""
    if ruta.startswith(("blob/", "tree/", "raw/")):
        # la ruta tras blob/tree/raw empieza por la rama; el raw no la necesita
        partes = ruta.split("/")
        ruta = "/".join(partes[2:]) or ""
    raw_path = ruta or "README.md"
    return f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{raw_path}"


def _github_a_markdown(url: str) -> str | None:
    raw_url = github_raw_url(url)
    if not raw_url:
        return None
    owner = _GITHUB_RE.search(url).group("owner")
    repo = _GITHUB_RE.search(url).group("repo")
    try:
        resp = RestTool.get(raw_url, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    if resp.status != 200:
        return None
    cuerpo = resp.text
    if "<html" in cuerpo[:2000].lower():
        return None  # raw devolvió una página, no el fichero
    cabecera = f"# {owner}/{repo} — {raw_url.rsplit('/', 1)[-1]}\n\n> Fuente: {url}\n"
    return cabecera + cuerpo.strip()


# -- arXiv --------------------------------------------------------------------

def _arxiv_a_markdown(url: str) -> str | None:
    match = _ARXIV_RE.search(url)
    if not match:
        return None
    arxiv_id = match.group("id")
    try:
        return KnowledgeTool.fetch_paper_markdown(arxiv_id)
    except Exception:  # noqa: BLE001
        return None
