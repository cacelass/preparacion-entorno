"""
agents.tools.knowledge_tool — Mantenimiento del corpus de conocimiento
(`docs/knowledge/`): verifica que las fuentes siguen vigentes y detecta papers
nuevos relevantes.

`rag refresh` es quien lo usa: lee `docs/knowledge/sources.json` (el registro
máquina), consulta la API de arXiv por cada topic del corpus y:

- **Verifica** cada fuente activa: si arXiv tiene una versión más reciente,
  la marca como superada en el informe.
- **Detecta** papers nuevos: busca cada topic con sus queries y filtra por
  publicación reciente.
- **Descarga** los nuevos a `docs/knowledge/papers/<tema>/<id>.md` — el HTML de
  arXiv cuando existe (sin dependencias); si no, el PDF convertido con
  `markitdown` (opcional). Actualiza `sources.json` y reindexa el corpus.

Como `research_tool`, la red va por `RestTool` (urllib, stdlib). La
"relevancia" es la propia búsqueda de arXiv, no una lectura semántica:
honestidad de diseño, igual que el resto del arnés. Sin conexión nada se
rompe: cada llamada falla de forma controlada y el informe lo refleja.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import tempfile
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from agents.tools.registry import register_tool
from agents.tools.rest_tool import RestTool
from agents.tools.rag_tool import RagTool, _html_a_texto

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_ABS = "https://arxiv.org/abs"
ARXIV_HTML = "https://arxiv.org/html"
ARXIV_PDF = "https://arxiv.org/pdf"

_ATOM = {"a": "http://www.w3.org/2005/Atom"}
_ABS_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(?P<id>[^v/]+(?:/[^v/]+)?)v?(?P<v>\d+)?")


class KnowledgeError(RuntimeError):
    """Fallo de mantenimiento del corpus (red, markitdown, fuentes corruptas)."""


@register_tool("knowledge")
class KnowledgeTool:
    # -- registro de fuentes -------------------------------------------------

    @staticmethod
    def _sources_path(root: Path) -> Path:
        return root / "docs" / "knowledge" / "sources.json"

    @staticmethod
    def load_sources(root: Path) -> dict[str, Any] | None:
        """El registro máquina del corpus. `None` si no existe o está corrupto."""
        ruta = KnowledgeTool._sources_path(root)
        if not ruta.exists():
            return None
        try:
            data = json.loads(ruta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or not isinstance(data.get("topics"), list):
            return None
        return data

    @staticmethod
    def save_sources(root: Path, data: dict[str, Any]) -> None:
        ruta = KnowledgeTool._sources_path(root)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # -- arXiv ---------------------------------------------------------------

    @staticmethod
    def _parse_atom(atom_xml: str) -> list[dict[str, Any]]:
        """Entradas de arXiv con lo que `refresh` necesita: id, versión, fechas."""
        papers: list[dict[str, Any]] = []
        try:
            root = ET.fromstring(atom_xml)
        except ET.ParseError:
            return papers
        for entry in root.findall("a:entry", _ATOM):
            url = (entry.findtext("a:id", default="", namespaces=_ATOM) or "").strip()
            m = _ABS_RE.search(url)
            if m is None:
                continue
            papers.append({
                "arxiv_id": m.group("id"),
                "version": int(m.group("v")) if m.group("v") else 1,
                "title": (entry.findtext("a:title", default="", namespaces=_ATOM) or "").strip().replace("\n", " "),
                "abstract": (entry.findtext("a:summary", default="", namespaces=_ATOM) or "").strip(),
                "published": entry.findtext("a:published", default="", namespaces=_ATOM) or "",
                "updated": entry.findtext("a:updated", default="", namespaces=_ATOM) or "",
                "url": url,
                "authors": [
                    (a.findtext("a:name", default="", namespaces=_ATOM) or "").strip()
                    for a in entry.findall("a:author", _ATOM)
                ],
            })
        return papers

    @staticmethod
    def arxiv_search(query: str, *, max_results: int = 5, timeout: int = 20) -> list[dict[str, Any]]:
        """Papers de arXiv para una query, por relevancia. Lanza si la red falla."""
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        })
        resp = RestTool.get(f"{ARXIV_API}?{params}", timeout=timeout)
        return KnowledgeTool._parse_atom(resp.text)

    @staticmethod
    def paper_status(arxiv_id: str, *, timeout: int = 20) -> dict[str, Any] | None:
        """Estado de una fuente en arXiv: versión más reciente y fechas."""
        params = urllib.parse.urlencode({"id_list": arxiv_id, "max_results": 5})
        resp = RestTool.get(f"{ARXIV_API}?{params}", timeout=timeout)
        for paper in KnowledgeTool._parse_atom(resp.text):
            if paper["arxiv_id"] == arxiv_id:
                return paper
        return None

    # -- descarga a markdown -------------------------------------------------

    @staticmethod
    def _html_disponible(arxiv_id: str) -> bool:
        try:
            resp = RestTool.get(f"{ARXIV_HTML}/{arxiv_id}", timeout=20)
        except Exception:  # noqa: BLE001 — sin red el fallback a PDF lo cubre
            return False
        return resp.status == 200 and "<html" in resp.text[:2000].lower()

    @staticmethod
    def _fetch(arxiv_id: str, base: str) -> bytes:
        resp = RestTool.get(f"{base}/{arxiv_id}", timeout=40)
        if resp.status != 200:
            raise KnowledgeError(f"HTTP {resp.status} al descargar {base}/{arxiv_id}")
        return resp.text.encode("utf-8")

    @staticmethod
    def _pdf_a_markdown(pdf_bytes: bytes) -> str:
        """PDF → markdown con markitdown (opcional). Falla con instrucción."""
        try:
            from markitdown import MarkItDown
        except ImportError as exc:
            raise KnowledgeError(
                "markitdown no está instalado y el paper no tiene HTML de arXiv. "
                "Ejecuta: uv sync --extra rag"
            ) from exc
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            md = MarkItDown().convert(tmp.name)
        return str(md.text_content or "").strip()

    @staticmethod
    def fetch_paper_markdown(arxiv_id: str) -> str:
        """
        Texto markdown de un paper: HTML de arXiv si existe (sin dependencias),
        PDF→markitdown si no. Lanza `KnowledgeError` con instrucción si falla.
        """
        if KnowledgeTool._html_disponible(arxiv_id):
            cuerpo = KnowledgeTool._fetch(arxiv_id, ARXIV_HTML).decode("utf-8", errors="replace")
            return _html_a_texto(cuerpo) or cuerpo
        pdf = KnowledgeTool._fetch(arxiv_id, ARXIV_PDF)
        return KnowledgeTool._pdf_a_markdown(pdf)

    @staticmethod
    def _nombre_fichero(arxiv_id: str) -> str:
        return arxiv_id.replace("/", "-") + ".md"

    # -- refresh -------------------------------------------------------------

    @staticmethod
    def refresh(
        root: Path,
        *,
        dry_run: bool = False,
        months: int = 6,
        max_new: int = 3,
        topics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Verifica las fuentes del corpus y detecta/descarga papers nuevos.

        `dry_run=True` no escribe nada: devuelve el informe (papers nuevos,
        fuentes superadas, errores). Sin dry-run descarga a `docs/knowledge/papers/`,
        actualiza `sources.json` y reindexa el RAG si chromadb está disponible.
        """
        data = KnowledgeTool.load_sources(root)
        if data is None:
            return {
                "error": "No existe docs/knowledge/sources.json (¿se generó el proyecto con use_rag?).",
            }

        hoy = _dt.date.today()
        desde = hoy - _dt.timedelta(days=months * 30)
        if isinstance(topics, str):
            topics = [t.strip() for t in topics.split(",") if t.strip()]
        topics_pedidos = set(topics or [])

        informe: dict[str, Any] = {
            "dry_run": dry_run,
            "meses": months,
            "new_papers": [],
            "updated_sources": [],
            "downloads": [],
            "errors": [],
            "topics": [],
        }

        for tema in data.get("topics", []):
            if topics_pedidos and tema.get("topic") not in topics_pedidos:
                continue

            bloque: dict[str, Any] = {
                "topic": tema.get("topic"),
                "searched": False,
                "new_candidates": [],
                "sources": [],
            }
            vistos = {s.get("arxiv_id") for s in tema.get("sources", [])}

            # 1. vigencia de las fuentes existentes (¿hay versión más nueva?)
            for src in tema.get("sources", []):
                estado = {"arxiv_id": src.get("arxiv_id")}
                try:
                    actual = KnowledgeTool.paper_status(src["arxiv_id"])
                except Exception as exc:  # noqa: BLE001 — sin red, controlado
                    informe["errors"].append(f"{tema.get('topic')}/{src.get('arxiv_id')}: {exc}")
                    estado["error"] = "no se pudo consultar arXiv (¿sin red?)"
                    bloque["sources"].append(estado)
                    continue
                if actual is None:
                    estado["status"] = "no_encontrado"
                    estado["nota"] = "el ID ya no devuelve resultados en arXiv"
                    bloque["sources"].append(estado)
                    continue
                if actual["version"] > int(src.get("version", 0)):
                    estado["status"] = "superada"
                    estado["version_actual"] = src.get("version")
                    estado["version_nueva"] = actual["version"]
                    estado["titulo"] = actual["title"]
                    informe["updated_sources"].append({
                        "arxiv_id": src["arxiv_id"],
                        "desde": src.get("version"), "hasta": actual["version"],
                    })
                else:
                    estado["status"] = "al_dia"
                bloque["sources"].append(estado)

            # 2. papers nuevos relevantes (primera query del topic)
            nuevas: list[dict[str, Any]] = []
            for query in tema.get("queries", [])[:1]:
                try:
                    resultados = KnowledgeTool.arxiv_search(query, max_results=max_new + 3)
                except Exception as exc:  # noqa: BLE001 — sin red, controlado
                    informe["errors"].append(f"{tema.get('topic')}: {exc}")
                    break
                bloque["searched"] = True
                for paper in resultados:
                    if paper["arxiv_id"] in vistos:
                        continue
                    try:
                        publicado = _dt.date.fromisoformat(paper["published"][:10])
                    except ValueError:
                        continue
                    if publicado >= desde:
                        nuevas.append(paper)
                        vistos.add(paper["arxiv_id"])
                break  # una query por topic basta para detectar novedad

            bloque["new_candidates"] = nuevas[:max_new]
            informe["new_papers"].extend(
                {"topic": tema.get("topic"), **p} for p in nuevas[:max_new]
            )
            informe["topics"].append(bloque)

        if not dry_run:
            destino = root / "docs" / "knowledge" / "papers"
            for tema in informe["topics"]:
                entradas = next(
                    (t for t in data.get("topics", []) if t.get("topic") == tema["topic"]), None
                )
                if entradas is None:
                    continue
                for paper in tema["new_candidates"]:
                    try:
                        md = KnowledgeTool.fetch_paper_markdown(paper["arxiv_id"])
                    except KnowledgeError as exc:
                        informe["errors"].append(f"descarga {paper['arxiv_id']}: {exc}")
                        continue
                    ruta = destino / tema["topic"] / KnowledgeTool._nombre_fichero(paper["arxiv_id"])
                    ruta.parent.mkdir(parents=True, exist_ok=True)
                    ruta.write_text(md, encoding="utf-8")
                    informe["downloads"].append({
                        "arxiv_id": paper["arxiv_id"],
                        "titulo": paper["title"],
                        "ruta": str(ruta.relative_to(root)),
                    })
                    entradas.setdefault("sources", []).append({
                        "arxiv_id": paper["arxiv_id"],
                        "title": paper["title"],
                        "version": paper["version"],
                        "status": "nueva",
                        "last_checked": hoy.isoformat(),
                    })

            # 3. marcar última verificación y persistir
            for tema in data.get("topics", []):
                for src in tema.get("sources", []):
                    src["last_checked"] = hoy.isoformat()
            data["updated"] = hoy.isoformat()
            KnowledgeTool.save_sources(root, data)

            # 4. reindexar el corpus (opcional: chroma puede no estar instalado)
            try:
                resultado = RagTool.index_project(root)
                informe["reindex"] = {
                    "total_chunks": resultado.get("total_chunks"),
                    "new_chunks": resultado.get("new_chunks"),
                }
            except Exception as exc:  # noqa: BLE001 — la búsqueda funciona sin reindexar
                informe["errors"].append(f"reindexar: {exc}")

        informe["fuentes"] = sum(len(t.get("sources", [])) for t in informe["topics"])
        return informe
