"""
agents.tools.graphify_tool — Puente entre el template y graphify.

graphify (github.com/anomalyco/graphify) convierte cualquier carpeta de código,
docs, papers, imágenes o vídeo en un grafo de conocimiento navegable
(`graphify-out/graph.json`). Esta herramienta centraliza TODA la interacción
con él para que los agentes (`knowledge`, `docsearch`, `git`) no reimplementen
cada uno la misma lógica de "¿dónde está el intérprete?", "¿existe el grafo?",
"¿cómo lanzo un --update?".

Límite honesto (igual que `vision_tool`): esta herramienta NO entiende
semánticamente el contenido de los nodos. Los "resúmenes de nodo padre" y las
"correlaciones" que calcula son **estructurales** — se derivan de la topología
del grafo (grado, vecinos compartidos, comunidad dominante), no de leer el
texto. Un resumen aquí dice "este nodo agrupa 12 hijos, los más conectados
entre sí son X e Y", no "este nodo trata sobre redes de atención".

No añade dependencias: graphify vive en su propio intérprete (`.graphify_python`)
y aquí solo se lee `graph.json` con la stdlib (json) y se lanzan subprocesos.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from agents.tools.registry import register_tool

# Subcarpeta de caché dentro de graphify-out/. Va al .gitignore del template:
# es contenido derivado y voluminoso, no debe versionarse.
CACHE_SUBDIR = "cache"


@register_tool("graphify")
class GraphifyTool:
    # -- localización ---------------------------------------------------------
    @staticmethod
    def out_dir(root: Path) -> Path:
        return root / "graphify-out"

    @staticmethod
    def graph_json(root: Path) -> Path:
        return GraphifyTool.out_dir(root) / "graph.json"

    @staticmethod
    def cache_dir(root: Path) -> Path:
        return GraphifyTool.out_dir(root) / CACHE_SUBDIR

    @staticmethod
    def graph_exists(root: Path) -> bool:
        return GraphifyTool.graph_json(root).exists()

    @staticmethod
    def resolve_python(root: Path) -> str | None:
        """
        Devuelve el intérprete de Python que graphify dejó anotado en
        ``graphify-out/.graphify_python`` (escrito por el skill /graphify), o
        None si el marcador no existe o apunta a algo inexistente. NO cae al
        binario ``graphify``: eso no es un intérprete y no sirve para
        ``python -m graphify`` (ver ``command_prefix``).
        """
        marker = GraphifyTool.out_dir(root) / ".graphify_python"
        if marker.exists():
            candidate = marker.read_text(encoding="utf-8").strip()
            if candidate and Path(candidate).exists():
                return candidate
        return None

    @staticmethod
    def command_prefix(root: Path) -> list[str] | None:
        """
        Prefijo correcto para invocar graphify:
          - si hay intérprete anotado → ``[python, "-m", "graphify"]``
          - si no, pero hay binario ``graphify`` en el PATH → ``[graphify]``
          - si no hay ninguno → None.

        Distinguir ambos importa: ``graphify -m graphify ...`` (binario con
        ``-m``) es un comando inválido — ese era el bug de mezclar los dos.
        """
        python_bin = GraphifyTool.resolve_python(root)
        if python_bin is not None:
            return [python_bin, "-m", "graphify"]
        binary = shutil.which("graphify")
        if binary:
            return [binary]
        return None

    @staticmethod
    def is_available(root: Path) -> bool:
        return GraphifyTool.command_prefix(root) is not None

    # -- Obsidian -------------------------------------------------------------
    @staticmethod
    def detect_obsidian_vaults(root: Path, *, max_depth: int = 4) -> list[Path]:
        """
        Busca bóvedas de Obsidian bajo ``root``: cualquier carpeta que
        contenga un subdirectorio ``.obsidian/`` es la raíz de una bóveda.
        Devuelve las raíces de bóveda encontradas (no los ``.obsidian``).

        No desciende dentro de ``.git``, ``.venv``, ``node_modules`` ni
        ``graphify-out``: se podan del recorrido (con ``os.walk``, no
        ``rglob``, que sí entraría en ellos), porque esto corre en cada commit
        y esos árboles pueden ser enormes.
        """
        import os

        skip = {".git", ".venv", "venv", "node_modules", "graphify-out",
                "__pycache__", ".mypy_cache", ".ruff_cache"}
        vaults: list[Path] = []
        root = root.resolve()
        for dirpath, dirnames, _ in os.walk(root):
            if ".obsidian" in dirnames:
                vaults.append(Path(dirpath))
            rel = Path(dirpath).relative_to(root)
            depth = 0 if rel == Path(".") else len(rel.parts)
            # Poda in situ: no descender a skip, a .obsidian, ni más allá de max_depth.
            dirnames[:] = [
                d for d in dirnames
                if d not in skip and d != ".obsidian" and (depth + 1) <= max_depth
            ]
        return sorted(set(vaults))

    # -- lectura del grafo ----------------------------------------------------
    @staticmethod
    def load_graph(root: Path) -> dict[str, Any]:
        """
        Lee ``graphify-out/graph.json``. Devuelve el dict crudo con al menos
        ``nodes`` y ``edges`` (listas). Lanza FileNotFoundError si no existe —
        deja que el agente lo convierta en un AgentResult con mensaje claro.
        """
        path = GraphifyTool.graph_json(root)
        if not path.exists():
            raise FileNotFoundError(f"No existe {path}. Ejecuta graphify primero.")
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("nodes", [])
        data.setdefault("edges", [])
        return data

    @staticmethod
    def _adjacency(graph: dict[str, Any]) -> dict[str, set[str]]:
        """Construye adyacencia no dirigida {id_nodo: set(vecinos)} desde edges."""
        adj: dict[str, set[str]] = defaultdict(set)
        for edge in graph.get("edges", []):
            src = str(edge.get("source", edge.get("from", "")))
            tgt = str(edge.get("target", edge.get("to", "")))
            if not src or not tgt or src == tgt:
                continue
            adj[src].add(tgt)
            adj[tgt].add(src)
        return adj

    @staticmethod
    def _node_index(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {str(n.get("id")): n for n in graph.get("nodes", []) if n.get("id") is not None}

    # -- resúmenes de nodo padre (estructurales) ------------------------------
    @staticmethod
    def parent_summaries(
        graph: dict[str, Any],
        *,
        min_children: int = 3,
        top: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Para los nodos "padre" del grafo (los de mayor grado — hubs/god nodes),
        produce un resumen ESTRUCTURAL de sus hijos (vecinos directos):

          - cuántos hijos tiene y de qué tipos
          - la comunidad dominante entre los hijos (si el grafo tiene comunidades)
          - los pares de hijos más correlacionados entre sí, medido por
            solapamiento de vecinos (índice de Jaccard sobre sus vecindarios) —
            una señal de "estos dos hijos hablan de lo mismo".

        Devuelve como mucho ``top`` resúmenes, ordenados por número de hijos.
        Solo incluye nodos con al menos ``min_children`` hijos.
        """
        adj = GraphifyTool._adjacency(graph)
        nodes = GraphifyTool._node_index(graph)

        parents = sorted(adj.items(), key=lambda kv: len(kv[1]), reverse=True)
        summaries: list[dict[str, Any]] = []
        for parent_id, children in parents:
            if len(children) < min_children:
                continue
            parent_node = nodes.get(parent_id, {})

            # Tipos de los hijos
            type_counts: dict[str, int] = defaultdict(int)
            comm_counts: dict[str, int] = defaultdict(int)
            for child in children:
                cnode = nodes.get(child, {})
                type_counts[str(cnode.get("type", "desconocido"))] += 1
                comm = cnode.get("community")
                if comm is not None:
                    comm_counts[str(comm)] += 1

            dominant_community = (
                max(comm_counts.items(), key=lambda kv: kv[1])[0] if comm_counts else None
            )

            # Correlación entre hijos: pares con mayor solapamiento de vecinos.
            correlated = GraphifyTool._correlated_pairs(children, adj, nodes, top=5)

            summaries.append({
                "id": parent_id,
                "label": parent_node.get("label", parent_id),
                "type": parent_node.get("type", "desconocido"),
                "n_children": len(children),
                "child_types": dict(type_counts),
                "dominant_community": dominant_community,
                "correlated_children": correlated,
                "summary": GraphifyTool._render_summary(
                    parent_node.get("label", parent_id),
                    len(children), dict(type_counts), dominant_community, correlated,
                ),
            })
            if len(summaries) >= top:
                break
        return summaries

    @staticmethod
    def _correlated_pairs(
        children: Iterable[str],
        adj: dict[str, set[str]],
        nodes: dict[str, dict[str, Any]],
        *,
        top: int = 5,
        max_children: int = 200,
    ) -> list[dict[str, Any]]:
        """
        Pares de hijos más "correlacionados", medido por Jaccard de sus
        vecindarios (excluyendo al padre común). Determinista: mide relación
        estructural, no semántica.

        El coste es O(k²) en el nº de hijos ``k``; en un hub gigante (miles de
        vecinos) eso explota. Se limita a los ``max_children`` hijos de mayor
        grado — los más informativos para la correlación — de forma
        determinista, para acotar el trabajo sin depender del orden de entrada.
        """
        children = sorted(children, key=lambda c: (-len(adj.get(c, set())), c))[:max_children]
        pairs: list[tuple[float, str, str]] = []
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                a, b = children[i], children[j]
                na, nb = adj.get(a, set()), adj.get(b, set())
                union = na | nb
                if not union:
                    continue
                jaccard = len(na & nb) / len(union)
                # Una arista directa entre los dos hijos también cuenta como
                # correlación fuerte, aunque no compartan otros vecinos.
                direct = b in na
                score = jaccard + (0.5 if direct else 0.0)
                if score > 0:
                    pairs.append((score, a, b))
        pairs.sort(reverse=True)
        result = []
        for score, a, b in pairs[:top]:
            result.append({
                "a": nodes.get(a, {}).get("label", a),
                "b": nodes.get(b, {}).get("label", b),
                "score": round(score, 3),
                "shared_neighbors": len(adj.get(a, set()) & adj.get(b, set())),
            })
        return result

    @staticmethod
    def _render_summary(
        label: str,
        n_children: int,
        type_counts: dict[str, int],
        dominant_community: str | None,
        correlated: list[dict[str, Any]],
    ) -> str:
        types_str = ", ".join(f"{n} {t}" for t, n in
                              sorted(type_counts.items(), key=lambda kv: -kv[1]))
        parts = [f"'{label}' agrupa {n_children} nodo(s) hijo ({types_str})"]
        if dominant_community is not None:
            parts.append(f"comunidad dominante {dominant_community}")
        if correlated:
            top_pair = correlated[0]
            parts.append(
                f"los más relacionados entre sí: '{top_pair['a']}' ↔ '{top_pair['b']}' "
                f"({top_pair['shared_neighbors']} vecino(s) en común)"
            )
        return "; ".join(parts) + "."

    # -- poda de nodos --------------------------------------------------------
    @staticmethod
    def prune(
        graph: dict[str, Any],
        *,
        node_types: Iterable[str] | None = None,
        node_ids: Iterable[str] | None = None,
        drop_isolated: bool = False,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """
        Devuelve una COPIA del grafo sin los nodos indicados (y sin sus aristas).
        No escribe a disco — el agente decide si persistir con ``save_graph``.

          - node_types : elimina todo nodo cuyo ``type`` esté en el conjunto
                         (p. ej. quitar 'reference' para limpiar referencias).
          - node_ids   : elimina nodos concretos por id.
          - drop_isolated : tras podar, elimina también los nodos que se
                            quedaron sin ninguna arista.

        Devuelve (grafo_podado, stats) donde stats cuenta nodos/aristas quitados.
        """
        drop_types = {str(t) for t in (node_types or [])}
        drop_ids = {str(i) for i in (node_ids or [])}

        kept_nodes = []
        removed_ids: set[str] = set()
        for n in graph.get("nodes", []):
            nid = str(n.get("id"))
            if nid in drop_ids or str(n.get("type", "")) in drop_types:
                removed_ids.add(nid)
            else:
                kept_nodes.append(n)

        kept_edges = []
        removed_edges = 0
        for e in graph.get("edges", []):
            src = str(e.get("source", e.get("from", "")))
            tgt = str(e.get("target", e.get("to", "")))
            if src in removed_ids or tgt in removed_ids:
                removed_edges += 1
            else:
                kept_edges.append(e)

        if drop_isolated:
            connected: set[str] = set()
            for e in kept_edges:
                connected.add(str(e.get("source", e.get("from", ""))))
                connected.add(str(e.get("target", e.get("to", ""))))
            # Aislado = superviviente de la poda por tipo/id que no aparece en
            # ninguna arista restante. Se calcula sobre kept_nodes (no sobre
            # todos los nodos) con un set de ids: O(n), sin comparar dicts.
            isolated_ids = {str(n.get("id")) for n in kept_nodes
                            if str(n.get("id")) not in connected}
            kept_nodes = [n for n in kept_nodes if str(n.get("id")) not in isolated_ids]
            removed_ids |= isolated_ids
            isolated_removed = len(isolated_ids)
        else:
            isolated_removed = 0

        pruned = dict(graph)
        pruned["nodes"] = kept_nodes
        pruned["edges"] = kept_edges
        stats = {
            "nodes_removed": len(removed_ids),
            "edges_removed": removed_edges,
            "isolated_removed": isolated_removed,
            "nodes_remaining": len(kept_nodes),
            "edges_remaining": len(kept_edges),
        }
        return pruned, stats

    @staticmethod
    def save_graph(root: Path, graph: dict[str, Any], *, backup: bool = True) -> Path:
        """Escribe el grafo a ``graph.json``, dejando antes un ``graph.json.bak``."""
        path = GraphifyTool.graph_json(root)
        if backup and path.exists():
            shutil.copy2(path, path.with_suffix(".json.bak"))
        path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    # -- Obsidian Flavored Markdown (convenciones kepano/obsidian-skills) ------
    # Estas notas siguen la spec de github.com/kepano/obsidian-skills para que
    # la bóveda sea óptima y editable por cualquier agente que tenga instaladas
    # esas skills (Claude Code, Codex, opencode). Frontmatter con properties,
    # wikilinks [[...]], callouts > [!type] y tags anidados.
    @staticmethod
    def obsidian_frontmatter(
        title: str,
        tags: list[str],
        *,
        aliases: list[str] | None = None,
        cssclasses: list[str] | None = None,
    ) -> str:
        """Bloque de properties (YAML frontmatter) de una nota de Obsidian."""
        lines = ["---", f"title: {title}"]
        if tags:
            lines.append("tags:")
            lines += [f"  - {t}" for t in tags]
        if aliases:
            lines.append("aliases:")
            lines += [f"  - {a}" for a in aliases]
        if cssclasses:
            lines.append("cssclasses:")
            lines += [f"  - {c}" for c in cssclasses]
        lines.append("---")
        return "\n".join(lines)

    @staticmethod
    def obsidian_note(
        title: str,
        tags: list[str],
        body: str,
        *,
        aliases: list[str] | None = None,
        cssclasses: list[str] | None = None,
    ) -> str:
        """Nota completa: frontmatter + cuerpo en Obsidian Flavored Markdown."""
        front = GraphifyTool.obsidian_frontmatter(
            title, tags, aliases=aliases, cssclasses=cssclasses
        )
        return f"{front}\n\n{body.rstrip()}\n"

    @staticmethod
    def knowledge_base(*, name: str = "Nodos del grafo", tag: str = "knowledge") -> str:
        """
        Devuelve un archivo Obsidian Bases (`.base`, YAML) que muestra las notas
        de la bóveda etiquetadas con ``tag`` como tabla y como tarjetas. Sigue
        la spec obsidian-bases de kepano/obsidian-skills.
        """
        return (
            "filters:\n"
            "  and:\n"
            f"    - 'file.hasTag(\"{tag}\")'\n"
            "properties:\n"
            "  file.name:\n"
            "    displayName: Nota\n"
            "  tags:\n"
            "    displayName: Etiquetas\n"
            "views:\n"
            "  - type: table\n"
            f"    name: \"{name}\"\n"
            "    order:\n"
            "      - file.name\n"
            "      - tags\n"
            "  - type: cards\n"
            "    name: \"Tarjetas\"\n"
            "    order:\n"
            "      - file.name\n"
        )

    # -- ejecución de graphify (subprocesos) ----------------------------------
    @staticmethod
    def run_cli(root: Path, args: list[str], *, timeout: int = 180) -> subprocess.CompletedProcess:
        """
        Lanza graphify con el prefijo correcto (intérprete + ``-m graphify`` o
        el binario del PATH) en la raíz del proyecto. El que llama inspecciona
        ``returncode``, ``stdout`` y ``stderr``.
        """
        prefix = GraphifyTool.command_prefix(root)
        if prefix is None:
            raise FileNotFoundError(
                "graphify no está disponible (ni .graphify_python ni binario en PATH). "
                "Ejecuta el skill /graphify una vez para instalarlo."
            )
        cmd = [*prefix, *args]
        return subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=timeout)

    @staticmethod
    def build(root: Path, *, timeout: int = 300) -> subprocess.CompletedProcess:
        """
        Construye/actualiza el grafo. Si ya existe ``graph.json`` usa
        ``--update`` (solo archivos nuevos/cambiados); si no, hace un build
        completo — ``--update`` sin grafo previo no tiene manifest del que
        partir.
        """
        args = [str(root), "--update"] if GraphifyTool.graph_exists(root) else [str(root)]
        return GraphifyTool.run_cli(root, args, timeout=timeout)

    @staticmethod
    def update(root: Path, *, timeout: int = 180) -> subprocess.CompletedProcess:
        """Re-extrae solo los archivos nuevos o cambiados (``graphify . --update``)."""
        return GraphifyTool.run_cli(root, [str(root), "--update"], timeout=timeout)

    @staticmethod
    def export_obsidian(root: Path, vault_dir: Path, *, timeout: int = 180) -> subprocess.CompletedProcess:
        """Exporta el grafo como bóveda de Obsidian (``graphify export obsidian --dir``)."""
        return GraphifyTool.run_cli(
            root, ["export", "obsidian", "--dir", str(vault_dir)], timeout=timeout
        )

    @staticmethod
    def query(root: Path, question: str, *, budget: int | None = None,
              timeout: int = 120) -> subprocess.CompletedProcess:
        """Consulta el grafo en lenguaje natural (``graphify query``)."""
        args = ["query", question]
        if budget:
            args += ["--budget", str(budget)]
        return GraphifyTool.run_cli(root, args, timeout=timeout)
