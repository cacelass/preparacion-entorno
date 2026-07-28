"""
agents.tools.rag_tool — RAG semántico local con ChromaDB + ONNX embedding.

Indexa documentación del proyecto (código, prompts, docs, vault, README, y la
memoria del arnés: progress/ y featureslist.json) y
permite búsqueda semántica. Sin GPU, sin API key, funciona offline.

Dependencias: chromadb (con embedding all-MiniLM-L6-v2 vía ONNX).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from agents.tools.registry import register_tool

try:
    import chromadb
    from chromadb.config import Settings

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
COLLECTION_NAME = "dskit-rag"

_MIN_CHUNK_CHARS = 100
_MAX_CHUNK_CHARS = 1500


def _file_type(source: str) -> str:
    if source.startswith("url:"):
        return "url"
    if source.endswith(".py"):
        return "code"
    if "/prompts/" in source:
        return "prompt"
    if "/vault/" in source or source.startswith("vault/"):
        return "vault"
    if source.startswith("progress/") or source == "featureslist.json":
        return "harness"
    return "doc"


@register_tool("rag")
class RagTool:
    @staticmethod
    def available() -> bool:
        return HAS_CHROMA

    @staticmethod
    def _client(root: Path) -> chromadb.ClientAPI | None:
        if not HAS_CHROMA:
            return None
        db_dir = root / ".rag-index"
        db_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(db_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

    @staticmethod
    def _collection(client: chromadb.ClientAPI) -> Any:
        try:
            return client.get_collection(COLLECTION_NAME)
        except ValueError:
            return client.create_collection(
                COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

    # -- ID helpers ----------------------------------------------------------

    @staticmethod
    def _chunk_id(source: str, line: int, text: str, section_type: str) -> str:
        return hashlib.md5(f"{source}:{line}:{section_type}:{text[:80]}".encode()).hexdigest()[:16]

    @staticmethod
    def _add_chunk(chunks: list, text: str, source: str, line: int,
                   section_type: str = "fallback") -> None:
        """
        Apila el chunk. Aquí vive la POLÍTICA de tamaño, no en `_make_chunk`.

        **La estructura gana al tamaño.** Un corte estructural —un encabezado,
        una función, una clase— es una unidad semántica y se indexa aunque sea
        corta: una sección de 40 caracteres es buscable, y perderla es peor que
        guardar un chunk pequeño. El suelo solo aplica a los cortes `fallback`,
        que parten por tamaño a ciegas: ahí un fragmento diminuto no significa
        nada y solo mete ruido en el índice.
        """
        chunk = RagTool._make_chunk(text, source, line, section_type)
        if chunk is None:
            return
        if section_type == "fallback" and len(chunk["text"]) < _MIN_CHUNK_CHARS:
            return
        chunks.append(chunk)

    @staticmethod
    def _ensure_not_empty(chunks: list, content: str, source: str,
                          section_type: str = "fallback") -> list:
        """
        Un documento nunca desaparece del índice por ser corto.

        Si ningún fragmento llegó al suelo, se emite uno con el texto entero:
        una nota de 90 caracteres sigue siendo buscable, y perderla sin aviso
        es peor que indexar un chunk pequeno.
        """
        if chunks or not content.strip():
            return chunks
        chunk = RagTool._make_chunk(content, source, 0, section_type)
        return [chunk] if chunk is not None else []

    @staticmethod
    def _make_chunk(text: str, source: str, line: int, section_type: str = "fallback") -> dict[str, Any] | None:
        """Construye el chunk. Sin política: solo rechaza el texto vacío."""
        text = text.strip()
        if not text:
            return None
        return {
            "id": RagTool._chunk_id(source, line, text, section_type),
            "text": text,
            "metadata": {
                "source": source,
                "line": line,
                "char_len": len(text),
                "file_type": _file_type(source),
                "section_type": section_type,
            },
        }

    # -- Recursive chunking --------------------------------------------------

    @staticmethod
    def chunk_py(content: str, source: str) -> list[dict[str, Any]]:
        chunks = []
        lines = content.split("\n")
        current_func: list[str] = []
        current_start = 0
        current_type = "fallback"

        for i, line in enumerate(lines):
            match_def = re.match(r"^(\s*)(async\s+)?def\s+(\w+)", line)
            match_class = re.match(r"^(\s*)class\s+(\w+)", line)

            if match_def or match_class:
                if current_func:
                    text = "\n".join(current_func)
                else:
                    text = ""

                RagTool._add_chunk(chunks, text, source, current_start, current_type)

                current_func = [line]
                current_start = i
                current_type = "docstring" if "def " in line else "class"

            elif current_func is not None:
                current_func.append(line)

        if current_func:
            text = "\n".join(current_func)
        else:
            text = ""

        RagTool._add_chunk(chunks, text, source, current_start, current_type)

        if not chunks:
            chunks = RagTool._chunk_by_size(content, source)
        chunks = RagTool._ensure_not_empty(chunks, content, source, "fallback")

        # Extract docstrings as independent chunks
        doc_chunks = []
        for c in chunks:
            docstring = RagTool._extract_docstring(c["text"])
            if docstring:
                ds_chunk = RagTool._make_chunk(
                    docstring, source, c["metadata"]["line"], "docstring"
                )
                if ds_chunk:
                    doc_chunks.append(ds_chunk)
        chunks.extend(doc_chunks)

        return chunks

    @staticmethod
    def _extract_docstring(text: str) -> str | None:
        m = re.search(r'"""(.*?)"""', text, re.DOTALL)
        if m:
            ds = m.group(1).strip()
            return ds if len(ds) >= 20 else None
        m = re.search(r"'''(.*?)'''", text, re.DOTALL)
        if m:
            ds = m.group(1).strip()
            return ds if len(ds) >= 20 else None
        return None

    @staticmethod
    def chunk_md(content: str, source: str) -> list[dict[str, Any]]:
        chunks = []
        lines = content.split("\n")
        current_section: list[str] = []
        current_start = 0
        current_level = 0

        for i, line in enumerate(lines):
            m = re.match(r"^(#+)\s", line)
            if m:
                level = len(m.group(1))
                if current_section:
                    text = "\n".join(current_section)
                    RagTool._add_chunk(chunks, text, source, current_start, "heading")
                current_section = [line]
                current_start = i
                current_level = level
            else:
                is_blank = line.strip() == ""
                if is_blank and current_section and len(current_section) > 1:
                    prev = current_section[-1].strip()
                    if prev == "":
                        para_text = "\n".join(current_section)
                        if len(para_text) > _MAX_CHUNK_CHARS:
                            for j in range(0, len(current_section), 20):
                                sub_para = current_section[j:j+20]
                                sub_text = "\n".join(sub_para)
                                if sub_text.strip() and len(sub_text) >= _MIN_CHUNK_CHARS:
                                    RagTool._add_chunk(chunks, sub_text, source, current_start, "paragraph")
                            current_section = [""]
                            continue
                current_section.append(line)

        if current_section:
            text = "\n".join(current_section)
            RagTool._add_chunk(
                chunks, text, source, current_start,
                "heading" if current_level > 0 else "paragraph",
            )

        if not chunks:
            chunks = RagTool._chunk_by_size(content, source)

        return RagTool._ensure_not_empty(
            chunks, content, source, "heading" if current_level > 0 else "paragraph"
        )

    @staticmethod
    def _chunk_by_size(content: str, source: str) -> list[dict[str, Any]]:
        chunks = []
        sentences = re.split(r"(?<=[.!?])\s+", content)
        sentence_chunks = []
        current_len = 0
        current_sentences = []
        chunk_start = 0

        for s in sentences:
            sentence_chunks.append(s)

        for i in range(0, len(sentence_chunks), 1):
            current_sentences.append(sentence_chunks[i])
            current_len += len(sentence_chunks[i])

            if current_len >= _MAX_CHUNK_CHARS:
                text = " ".join(current_sentences)
                if len(text) >= _MIN_CHUNK_CHARS:
                    RagTool._add_chunk(chunks, text, source, chunk_start, "fallback")
                current_sentences = []
                current_len = 0
                chunk_start = i

        if current_sentences:
            text = " ".join(current_sentences)
            if len(text) >= _MIN_CHUNK_CHARS:
                RagTool._add_chunk(chunks, text, source, chunk_start, "fallback")

        if not chunks:
            words = content.split()
            for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
                text = " ".join(words[i : i + CHUNK_SIZE])
                if len(text) >= _MIN_CHUNK_CHARS:
                    RagTool._add_chunk(chunks, text, source, i, "fallback")

        return chunks

    # -- Indexing ------------------------------------------------------------

    @staticmethod
    def index_project(root: Path) -> dict[str, Any]:
        client = RagTool._client(root)
        if client is None:
            return {"error": "chromadb no disponible. Instala: uv sync --extra rag"}
        collection = RagTool._collection(client)

        all_chunks: list[dict[str, Any]] = []

        src_dir = root / "{{ project_slug }}"
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                try:
                    text = py_file.read_text(encoding="utf-8")
                    all_chunks.extend(RagTool.chunk_py(text, str(py_file.relative_to(root))))
                except Exception:
                    pass

        prompts_dir = root / "agents" / "prompts"
        if prompts_dir.exists():
            for md_file in prompts_dir.glob("*.md"):
                try:
                    text = md_file.read_text(encoding="utf-8")
                    all_chunks.extend(RagTool.chunk_md(text, str(md_file.relative_to(root))))
                except Exception:
                    pass

        docs_dir = root / "docs"
        if docs_dir.exists():
            for doc_file in docs_dir.rglob("*.*"):
                if doc_file.suffix in (".md", ".rst"):
                    try:
                        text = doc_file.read_text(encoding="utf-8")
                        all_chunks.extend(RagTool.chunk_md(text, str(doc_file.relative_to(root))))
                    except Exception:
                        pass

        vault_dir = root / "vault"
        if vault_dir.exists():
            for md_file in vault_dir.rglob("*.md"):
                try:
                    text = md_file.read_text(encoding="utf-8")
                    all_chunks.extend(RagTool.chunk_md(text, str(md_file.relative_to(root))))
                except Exception:
                    pass

        # Memoria del arnés: el histórico de features cerradas y sus decisiones
        # es justo lo que un agente nuevo necesita buscar en lenguaje natural
        # ("¿por qué elegimos K=4?") sin releer todo progress/.
        progress_dir = root / "progress"
        if progress_dir.exists():
            for md_file in sorted(progress_dir.glob("*.md")):
                try:
                    text = md_file.read_text(encoding="utf-8")
                    all_chunks.extend(RagTool.chunk_md(text, str(md_file.relative_to(root))))
                except Exception:
                    pass

        backlog = root / "featureslist.json"
        if backlog.exists():
            try:
                doc = json.loads(backlog.read_text(encoding="utf-8"))
                lines = []
                for feat in doc.get("features", []):
                    criteria = "\n".join(f"- {c}" for c in feat.get("acceptance_criteria", []))
                    lines.append(
                        f"## {feat.get('id')} — {feat.get('title')} [{feat.get('status')}]\n\n"
                        f"{feat.get('description', '')}\n\nCriterios de aceptación:\n{criteria}\n"
                    )
                if lines:
                    all_chunks.extend(RagTool.chunk_md("\n".join(lines), "featureslist.json"))
            except Exception:
                pass

        for doc_file in ("README.md", "AGENTS.md", "CHANGELOG.md", "CONTRIBUTING.md"):
            fp = root / doc_file
            if fp.exists():
                try:
                    text = fp.read_text(encoding="utf-8")
                    all_chunks.extend(RagTool.chunk_md(text, doc_file))
                except Exception:
                    pass

        all_chunks = [c for c in all_chunks if c is not None]

        existing_ids = set(collection.get(include=[])["ids"])
        new_chunks = [c for c in all_chunks if c["id"] not in existing_ids]
        if new_chunks:
            collection.add(
                ids=[c["id"] for c in new_chunks],
                documents=[c["text"] for c in new_chunks],
                metadatas=[c["metadata"] for c in new_chunks],
            )

        count = collection.count()
        return {
            "total_chunks": count,
            "new_chunks": len(new_chunks),
            "sources": len(set(c["metadata"]["source"] for c in all_chunks)),
        }

    @staticmethod
    def index_url(root: Path, url: str, text: str) -> dict[str, Any]:
        client = RagTool._client(root)
        if client is None:
            return {"error": "chromadb no disponible"}
        collection = RagTool._collection(client)
        chunks = RagTool.chunk_md(text, f"url:{url}")
        chunks = [c for c in chunks if c is not None]
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{**c["metadata"], "type": "url"} for c in chunks],
        )
        return {"chunks_indexed": len(chunks), "url": url}

    # -- Search --------------------------------------------------------------

    @staticmethod
    def search(root: Path, query: str, top_k: int = 10, hybrid: bool = False) -> list[dict[str, Any]]:
        client = RagTool._client(root)
        if client is None:
            return [{"error": "chromadb no disponible"}]
        collection = RagTool._collection(client)
        if collection.count() == 0:
            return []

        if hybrid:
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    includes=["documents", "distances", "metadatas"],
                )
            except Exception:
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k,
                )
        else:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
            )

        output = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i][:500],
                "source": meta.get("source", "?"),
                "line": meta.get("line", 0),
                "file_type": meta.get("file_type", "?"),
                "section_type": meta.get("section_type", "?"),
                "score": round(1 - results["distances"][0][i], 4),
            })
        return output

    @staticmethod
    def status(root: Path) -> dict[str, Any]:
        client = RagTool._client(root)
        if client is None:
            return {"available": False}
        collection = RagTool._collection(client)
        count = collection.count()
        return {"available": True, "total_chunks": count, "collection": COLLECTION_NAME}
