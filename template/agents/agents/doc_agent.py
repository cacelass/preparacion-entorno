"""
agents.agents.doc_agent — Documentación unificada del proyecto.

Combina tres fuentes de conocimiento:
1. Graphify → grafo estructural (dependencias, nodos, relaciones)
2. RAG → búsqueda semántica vectorial (ChromaDB)
3. Vault → búsqueda textual directa en markdown de Obsidian
"""

from __future__ import annotations

import re
from pathlib import Path

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent

try:
    from agents.tools.graphify_tool import GraphifyTool
    HAS_GRAPHIFY = True
except ImportError:
    HAS_GRAPHIFY = False

try:
    from agents.tools.rag_tool import RagTool
    HAS_RAG = True
except ImportError:
    HAS_RAG = False


@register_agent
class DocAgent(BaseAgent):
    name = "doc"
    description = (
        "Documentación unificada del proyecto: busca en el grafo graphify "
        "(estructura), el índice RAG (semántica) y el vault Obsidian (notas)."
    )
    # Es el punto de entrada UNIFICADO: sus palabras son las que expresan
    # "búscalo donde sea", no los nombres de las fuentes concretas — esos
    # pertenecen a rag, knowledge y docsearch (un keyword, un dueño).
    capabilities = [
        "doc", "todas las fuentes", "busqueda unificada", "búsqueda unificada",
        "documentacion unificada", "donde esta documentado",
        "dónde está documentado", "que hace", "qué hace",
        "como funciona", "cómo funciona", "explica", "informacion", "información",
    ]

    def action_aliases(self) -> dict:
        return {
            "search": ["busca", "consulta", "encuentra", "informacion", "documentacion"],
            "graph_query": ["grafo", "graph", "estructura", "dependencias", "nodos"],
            "rag_search": ["semantico", "semantic", "vectorial", "embedding"],
            "vault_grep": ["vault", "obsidian", "grep", "texto en vault"],
            "index": ["indexar", "construye indice", "reindexa todo"],
            "status": ["estado", "fuentes", "disponible"],
        }

    def actions(self) -> dict:
        return {
            "search": self.search,
            "graph_query": self.graph_query,
            "rag_search": self.rag_search,
            "vault_grep": self.vault_grep,
            "index": self.index,
            "status": self.status,
        }

    def search(self, *, query: str, sources: str = "all") -> AgentResult:
        """Busca en todas las fuentes disponibles."""
        if not query.strip():
            return AgentResult(False, self.name, "search", "Proporciona una consulta.")

        combined = []
        warnings = []

        if sources in ("all", "graph") and HAS_GRAPHIFY and GraphifyTool.graph_exists(self.ctx.root):
            try:
                graph = GraphifyTool.load_graph(self.ctx.root)
                matches = [
                    n for n in graph.get("nodes", [])
                    if query.lower() in n.get("label", "").lower()
                       or query.lower() in n.get("id", "").lower()
                ]
                for n in matches[:10]:
                    combined.append({
                        "source": "graphify",
                        "label": n.get("label", n.get("id", "?")),
                        "type": n.get("type", "node"),
                    })
            except Exception:
                warnings.append("error al leer el grafo graphify")

        if sources in ("all", "rag") and HAS_RAG and RagTool.available():
            try:
                rag_results = RagTool.search(self.ctx.root, query, top_k=5)
                for r in rag_results:
                    r["source_type"] = "rag"
                    combined.append(r)
            except Exception:
                warnings.append("error al consultar RAG")

        if sources in ("all", "vault"):
            vault_path = self.ctx.root / "vault"
            if vault_path.exists():
                try:
                    for md_file in vault_path.rglob("*.md"):
                        text = md_file.read_text(encoding="utf-8", errors="replace")
                        if query.lower() in text.lower():
                            for lineno, line in enumerate(text.split("\n"), 1):
                                if query.lower() in line.lower():
                                    combined.append({
                                        "source": "vault",
                                        "source_type": "vault",
                                        "file": str(md_file.relative_to(self.ctx.root)),
                                        "line": lineno,
                                        "text": line.strip()[:200],
                                    })
                except Exception:
                    warnings.append("error al leer vault")

        if not combined:
            return AgentResult(
                True, self.name, "search",
                "No se encontraron resultados en ninguna fuente.",
                data=[], warnings=warnings,
            )

        lines = []
        for r in combined[:10]:
            if r.get("source_type") == "graphify":
                lines.append(f"  [graphify] {r['label']} ({r['type']})")
            elif r.get("source_type") == "rag":
                lines.append(f"  [rag] {r.get('source','?')}:{r.get('line','?')} — {r.get('text','')[:100]}")
            elif r.get("source_type") == "vault":
                lines.append(f"  [vault] {r['file']}:{r['line']} — {r['text']}")

        return AgentResult(
            True, self.name, "search",
            f"{len(combined)} resultado(s) de {len(set(r.get('source_type','') for r in combined))} fuente(s).\n"
            + "\n".join(lines[:10]),
            data=combined[:10], warnings=warnings,
        )

    def graph_query(self, *, question: str) -> AgentResult:
        """Consulta el grafo graphify (estructural)."""
        if not HAS_GRAPHIFY:
            return AgentResult(
                False, self.name, "graph_query",
                "graphify no está disponible.",
            )
        if not GraphifyTool.graph_exists(self.ctx.root):
            return AgentResult(
                False, self.name, "graph_query",
                "No hay grafo. Ejecuta 'knowledge build' primero.",
            )
        try:
            from agents.tools.cache_tool import CacheTool
            mtime = int(GraphifyTool.graph_json(self.ctx.root).stat().st_mtime)
            cache_key = f"graph_query_{mtime}_{question[:50]}"
            result = CacheTool.disk_cache(name=cache_key)(
                lambda: GraphifyTool.query(self.ctx.root, question)
            )()
            return AgentResult(
                True, self.name, "graph_query", str(result)[:500],
                data={"answer": str(result)},
            )
        except Exception as exc:
            return AgentResult(
                False, self.name, "graph_query", f"Error al consultar grafo: {exc}",
            )

    def rag_search(self, *, query: str, top_k: int = 10) -> AgentResult:
        """Búsqueda semántica pura vía RAG."""
        if not HAS_RAG:
            return AgentResult(
                False, self.name, "rag_search",
                "RAG no disponible (chromadb no instalado).",
            )
        if not RagTool.available():
            return AgentResult(
                False, self.name, "rag_search",
                "chromadb no instalado. Ejecuta: uv sync --extra rag",
            )
        results = RagTool.search(self.ctx.root, query, top_k=top_k)
        if not results:
            return AgentResult(
                True, self.name, "rag_search",
                "No hay resultados. Ejecuta 'rag index' primero.",
                data=[],
            )
        if "error" in results[0]:
            return AgentResult(False, self.name, "rag_search", results[0]["error"])
        lines = []
        for r in results[:5]:
            lines.append(f"  [{r['score']}] {r['source']}:{r['line']} — {r['text'][:120]}")
        return AgentResult(
            True, self.name, "rag_search",
            f"{len(results)} resultado(s). Top:\n" + "\n".join(lines),
            data=results,
        )

    def vault_grep(self, *, pattern: str) -> AgentResult:
        """Busca texto directamente en el vault Obsidian."""
        vault_path = self.ctx.root / "vault"
        if not vault_path.exists():
            return AgentResult(
                False, self.name, "vault_grep",
                "No hay directorio vault/ en el proyecto.",
            )
        matches = []
        for md_file in vault_path.rglob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
                for lineno, line in enumerate(text.split("\n"), 1):
                    if pattern.lower() in line.lower():
                        matches.append({
                            "file": str(md_file.relative_to(self.ctx.root)),
                            "line": lineno,
                            "text": line.strip()[:200],
                        })
            except Exception:
                pass
        if not matches:
            return AgentResult(
                True, self.name, "vault_grep",
                f"No se encontró '{pattern}' en vault/.",
                data=[],
            )
        lines = [f"  {m['file']}:{m['line']} — {m['text']}" for m in matches[:20]]
        return AgentResult(
            True, self.name, "vault_grep",
            f"{len(matches)} coincidencia(s) en vault/:\n" + "\n".join(lines),
            data=matches[:20],
        )

    def index(self) -> AgentResult:
        """Construye el grafo graphify + indexa RAG en un solo paso."""
        results = {}
        warnings = []

        if HAS_GRAPHIFY and GraphifyTool.is_available(self.ctx.root):
            try:
                built = GraphifyTool.build(self.ctx.root)
                results["graphify"] = {"success": built.returncode == 0}
                if built.returncode == 0:
                    g = GraphifyTool.load_graph(self.ctx.root)
                    results["graphify"]["nodes"] = len(g.get("nodes", []))
                    results["graphify"]["edges"] = len(g.get("edges", []))
            except Exception as exc:
                warnings.append(f"graphify build falló: {exc}")
                results["graphify"] = {"success": False}
        else:
            results["graphify"] = {"skipped": True}

        if HAS_RAG and RagTool.available():
            try:
                rag_result = RagTool.index_project(self.ctx.root)
                results["rag"] = rag_result
            except Exception as exc:
                warnings.append(f"rag index falló: {exc}")
                results["rag"] = {"error": str(exc)}
        else:
            results["rag"] = {"skipped": True}

        lines = []
        for src, res in results.items():
            if res.get("skipped"):
                lines.append(f"  {src}: omitido (no disponible)")
            elif res.get("success") or "total_chunks" in res:
                lines.append(f"  {src}: OK")
            else:
                lines.append(f"  {src}: error")

        return AgentResult(
            True, self.name, "index",
            "Indexación completa:\n" + "\n".join(lines),
            data=results, warnings=warnings,
        )

    def status(self) -> AgentResult:
        """Estado de cada fuente de documentación."""
        sources = {}

        graph_avail = HAS_GRAPHIFY and GraphifyTool.is_available(self.ctx.root)
        graph_exists = HAS_GRAPHIFY and GraphifyTool.graph_exists(self.ctx.root)
        sources["graphify"] = {
            "available": graph_avail,
            "graph_exists": graph_exists,
        }
        if graph_exists:
            try:
                g = GraphifyTool.load_graph(self.ctx.root)
                sources["graphify"]["nodes"] = len(g.get("nodes", []))
                sources["graphify"]["edges"] = len(g.get("edges", []))
            except Exception:
                pass

        sources["rag"] = {"available": HAS_RAG and RagTool.available()}
        if HAS_RAG and RagTool.available():
            try:
                info = RagTool.status(self.ctx.root)
                sources["rag"]["chunks"] = info.get("total_chunks", 0)
            except Exception:
                pass

        vault_path = self.ctx.root / "vault"
        sources["vault"] = {
            "exists": vault_path.exists(),
            "files": len(list(vault_path.rglob("*.md"))) if vault_path.exists() else 0,
        }

        lines = []
        for name, info in sources.items():
            if name == "graphify":
                avail = info.get("available", False) and info.get("graph_exists", False)
                status = "✓" if avail else "✗"
                details = f"{info.get('nodes', 0)} nodos / {info.get('edges', 0)} aristas" if avail else "no disponible"
                lines.append(f"  {status} graphify — {details}")
            elif name == "rag":
                status = "✓" if info.get("available") else "✗"
                details = f"{info.get('chunks', 0)} fragmentos" if info.get("available") else "chromadb no instalado"
                lines.append(f"  {status} RAG — {details}")
            elif name == "vault":
                status = "✓" if info.get("exists") else "✗"
                details = f"{info.get('files', 0)} archivos" if info.get("exists") else "no existe"
                lines.append(f"  {status} vault — {details}")

        return AgentResult(
            True, self.name, "status",
            "Fuentes de documentación:\n" + "\n".join(lines),
            data=sources,
        )
