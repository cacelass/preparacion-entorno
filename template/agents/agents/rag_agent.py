"""
agents.agents.rag_agent — RAG local con ChromaDB: búsqueda híbrida.

Indexa la documentación del proyecto (código, prompts, docs, vault) y busca
fundiendo similitud vectorial con BM25 léxico. También indexa URLs externas.
"""

from __future__ import annotations

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.rag_tool import RagTool
from agents.tools.rest_tool import RestTool


@register_agent
class RagAgent(BaseAgent):
    name = "rag"
    description = (
        "RAG local: indexa código, prompts, docs y vault del proyecto con "
        "ChromaDB. Busca en lenguaje natural fundiendo vector y BM25 léxico, y "
        "también puede indexar URLs externas."
    )
    capabilities = [
        "rag", "semantico", "semantic", "indexar", "index",
        "consulta semantic", "embedding", "chroma", "vector",
        "indice semantico", "busca en la documentacion",
        "encuentra en la documentacion",
        "indexa",
    ]

    def action_aliases(self) -> dict:
        return {
            "index": ["indexar", "construye el indice", "reindexa", "build rag"],
            "index_urls": ["indexa urls", "url", "pagina web", "documentacion externa"],
            "search": ["busca", "consulta", "encuentra", "pregunta"],
            "status": ["estado del indice", "info rag"],
        }

    def actions(self) -> dict:
        return {
            "index": self.index,
            "index_urls": self.index_urls,
            "search": self.search,
            "status": self.status,
        }

    def index(self, *, rebuild: bool = False) -> AgentResult:
        if not RagTool.available():
            return AgentResult(
                False, self.name, "index",
                "chromadb no está instalado. Ejecuta: uv sync --extra rag",
            )
        result = RagTool.index_project(self.ctx.root, rebuild=rebuild)
        if "error" in result:
            return AgentResult(False, self.name, "index", result["error"])
        return AgentResult(
            True, self.name, "index",
            f"Índice actualizado: {result['total_chunks']} fragmentos de "
            f"{result['sources']} fuente(s) [{result['embedder']}]. "
            f"+{result['new_chunks']} nuevos, {result['updated_files']} fichero(s) "
            f"reindexado(s), {result['unchanged_files']} sin cambios, "
            f"-{result['deleted_chunks']} huérfanos.",
            data=result,
        )

    def index_urls(self, *, urls: list[str]) -> AgentResult:
        if not RagTool.available():
            return AgentResult(
                False, self.name, "index_urls",
                "chromadb no está instalado. Ejecuta: uv sync --extra rag",
            )
        if not urls:
            return AgentResult(False, self.name, "index_urls", "Proporciona al menos una URL.")

        indexed = []
        errors = []
        for url in urls:
            try:
                resp = RestTool.get(url, timeout=30)
                result = RagTool.index_url(self.ctx.root, url, resp.text)
                if "error" in result:
                    errors.append({"url": url, "error": result["error"]})
                else:
                    indexed.append(result)
            except Exception as exc:
                errors.append({"url": url, "error": str(exc)[:200]})

        msg = f"Indexadas {len(indexed)} URL(s)."
        if errors:
            msg += f" {len(errors)} error(es)."
        return AgentResult(
            not errors or bool(indexed), self.name, "index_urls", msg,
            data={"indexed": indexed, "errors": errors},
            warnings=[f"{e['url']}: {e['error']}" for e in errors] if errors else None,
        )

    def search(self, *, query: str, top_k: int = 10, hybrid: bool = True,
               min_score: float = 0.0) -> AgentResult:
        if not RagTool.available():
            return AgentResult(
                False, self.name, "search",
                "chromadb no está instalado. Ejecuta: uv sync --extra rag",
            )
        results = RagTool.search(
            self.ctx.root, query, top_k=top_k, hybrid=hybrid, min_score=min_score
        )
        if not results:
            return AgentResult(
                True, self.name, "search",
                "No hay resultados. Ejecuta 'rag index' primero para construir el índice.",
                data=[],
            )
        if "error" in results[0]:
            return AgentResult(False, self.name, "search", results[0]["error"])

        lines = []
        for r in results[:5]:
            lines.append(
                f"  [{r['score']} {r['match']}] {r['source']}:{r['line']} — {r['text'][:120]}"
            )
        return AgentResult(
            True, self.name, "search",
            f"{len(results)} resultado(s). Top:\n" + "\n".join(lines),
            data=results,
        )

    def status(self) -> AgentResult:
        info = RagTool.status(self.ctx.root)
        if not info.get("available"):
            return AgentResult(
                False, self.name, "status",
                "chromadb no instalado. Ejecuta: uv sync --extra rag",
                data=info,
            )
        if info.get("mismatch"):
            return AgentResult(False, self.name, "status", info["mismatch"], data=info)
        return AgentResult(
            True, self.name, "status",
            f"Índice RAG: {info['total_chunks']} fragmentos de {info['sources']} "
            f"fuente(s) en '{info['collection']}' — embedder {info['embedder_desc']}.",
            data=info,
        )
