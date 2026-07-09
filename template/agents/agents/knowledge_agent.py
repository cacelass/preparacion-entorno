"""
agents.agents.knowledge_agent — Grafo de conocimiento + Obsidian, cacheado.

Este es el agente que "cachea lo máximo posible": envuelve graphify
(`graphify-out/graph.json`) y una bóveda de Obsidian, y mantiene ambos en
sincronía. Su razón de ser:

  1. Detecta si el proyecto ya tiene una bóveda de Obsidian; si no, ofrece
     crear una con una estructura de árbol adaptada al grafo (carpetas por
     tipo de nodo: papers, code, docs, references, media).
  2. Construye/actualiza el grafo con graphify y lo exporta a esa bóveda,
     fusionando ambos mundos.
  3. Resume cada NODO PADRE (hub) con un resumen de sus nodos hijo, incluyendo
     la correlación estructural entre ellos — para que, por ejemplo, un nodo
     "información" con muchos papers hijos tenga un resumen de qué agrupan y
     cómo se relacionan.
  4. Cachea en `graphify-out/cache/` (ignorado por git) todo lo caro:
     los resúmenes de nodo padre y las consultas.

Toda la interacción con graphify vive en `agents.tools.graphify_tool`; aquí
solo está la orquestación y el formateo a `AgentResult`.

Límite honesto: los resúmenes y correlaciones son ESTRUCTURALES (topología del
grafo), no una lectura semántica del contenido de los nodos. Ver el docstring
de `graphify_tool` para el detalle.
"""

from __future__ import annotations

from pathlib import Path

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.cache_tool import CacheTool
from agents.tools.graphify_tool import GraphifyTool

# Árbol adaptado que se crea dentro de una bóveda nueva. Cada carpeta agrupa un
# tipo de nodo del grafo; graphify luego rellena una nota por nodo.
_VAULT_TREE = {
    "00-index": "Mapas de contenido (MOC) y punto de entrada de la bóveda.",
    "papers": "Artículos, PDFs y sus extracciones.",
    "code": "Módulos y símbolos extraídos por AST.",
    "docs": "Documentación, notas y markdown del proyecto.",
    "references": "Referencias externas, enlaces y citas.",
    "media": "Imágenes y transcripciones de audio/vídeo.",
}


@register_agent
class KnowledgeAgent(BaseAgent):
    name = "knowledge"
    description = (
        "Grafo de conocimiento (graphify) + Obsidian, cacheado al máximo. "
        "Crea/sincroniza una bóveda con estructura de árbol, resume cada nodo "
        "padre con la correlación entre sus hijos, y mantiene el grafo al día."
    )
    capabilities = [
        "conocimiento", "knowledge", "grafo", "graph", "graphify", "obsidian",
        "boveda", "vault", "indexar", "papers", "nodo padre", "resumen del grafo",
    ]

    def action_aliases(self) -> dict:
        return {
            "setup_vault": ["obsidian", "boveda", "vault", "crea la boveda", "estructura"],
            "build": ["construye", "genera el grafo", "indexa", "graphify"],
            "summarize_parents": ["resume", "nodo padre", "padres", "resumen", "correlacion"],
            "sync": ["sincroniza", "actualiza", "update"],
            "status": ["estado", "situacion"],
        }

    def actions(self) -> dict:
        return {
            "status": self.status,
            "setup_vault": self.setup_vault,
            "build": self.build,
            "summarize_parents": self.summarize_parents,
            "sync": self.sync,
        }

    # -------------------------------------------------------------------------
    def _cache_dir(self) -> Path:
        """Fija (y crea) el directorio de caché en graphify-out/cache/."""
        cache_dir = GraphifyTool.cache_dir(self.ctx.root)
        CacheTool.set_cache_dir(cache_dir)
        return cache_dir

    def status(self) -> AgentResult:
        """Resume el estado del grafo, la caché y las bóvedas de Obsidian."""
        root = self.ctx.root
        available = GraphifyTool.is_available(root)
        graph_exists = GraphifyTool.graph_exists(root)
        vaults = GraphifyTool.detect_obsidian_vaults(root)
        cache_dir = GraphifyTool.cache_dir(root)
        n_cache = len(list(cache_dir.glob("*.joblib"))) if cache_dir.exists() else 0

        n_nodes = n_edges = 0
        if graph_exists:
            try:
                graph = GraphifyTool.load_graph(root)
                n_nodes, n_edges = len(graph["nodes"]), len(graph["edges"])
            except Exception:  # noqa: BLE001 — grafo corrupto, no debe tumbar el status
                pass

        warnings = []
        if not available:
            warnings.append(
                "graphify no está instalado. Ejecuta el skill /graphify una vez "
                "para instalarlo y construir el grafo inicial."
            )
        if not vaults:
            warnings.append(
                "No se detectó ninguna bóveda de Obsidian. Ejecuta "
                "'knowledge setup_vault' para crear una con estructura de árbol."
            )

        return AgentResult(
            available or graph_exists, self.name, "status",
            f"graphify {'disponible' if available else 'no instalado'}; "
            f"grafo con {n_nodes} nodo(s)/{n_edges} arista(s); "
            f"{len(vaults)} bóveda(s) Obsidian; {n_cache} entrada(s) en caché.",
            data={
                "graphify_available": available,
                "graph_exists": graph_exists,
                "nodes": n_nodes,
                "edges": n_edges,
                "obsidian_vaults": [str(v.relative_to(root)) if v.is_relative_to(root) else str(v) for v in vaults],
                "cache_entries": n_cache,
                "cache_dir": str(cache_dir.relative_to(root)) if cache_dir.is_relative_to(root) else str(cache_dir),
            },
            warnings=warnings,
        )

    def setup_vault(self, *, vault_dir: str | None = None, create_if_missing: bool = True) -> AgentResult:
        """
        Detecta la bóveda de Obsidian del proyecto. Si no hay ninguna y
        ``create_if_missing`` es True, crea una en ``vault_dir`` (por defecto
        ``knowledge/``) con la estructura de árbol adaptada al grafo.

        Es la respuesta determinista a "¿existe alguna carpeta de obsidian?":
        si existe, la reutiliza; si no, la construye.
        """
        root = self.ctx.root
        existing = GraphifyTool.detect_obsidian_vaults(root)
        if existing and vault_dir is None:
            return AgentResult(
                True, self.name, "setup_vault",
                f"Ya existe {len(existing)} bóveda(s) de Obsidian: "
                f"{', '.join(str(v.relative_to(root)) for v in existing)}. "
                f"Se reutilizará(n) — no hace falta crear otra.",
                data={"vaults": [str(v) for v in existing], "created": False},
            )

        target = (root / (vault_dir or "knowledge")).resolve()
        if not create_if_missing and not (target / ".obsidian").exists():
            return AgentResult(
                False, self.name, "setup_vault",
                f"No hay bóveda en {target} y create_if_missing=False.",
            )

        # Estructura de árbol: .obsidian (marca de bóveda) + carpetas por tipo.
        (target / ".obsidian").mkdir(parents=True, exist_ok=True)
        created_dirs = []
        for folder, purpose in _VAULT_TREE.items():
            path = target / folder
            path.mkdir(parents=True, exist_ok=True)
            keep = path / "README.md"
            if not keep.exists():
                keep.write_text(f"# {folder}\n\n{purpose}\n", encoding="utf-8")
            created_dirs.append(folder)

        # MOC raíz: el índice de la bóveda desde el que se navega el árbol.
        moc = target / "00-index" / "MOC.md"
        if not moc.exists():
            moc.write_text(
                "# Mapa de contenido\n\n"
                "Bóveda generada por el `knowledge` agent de dskit. El grafo de "
                "graphify se exporta aquí y se organiza en este árbol:\n\n"
                + "\n".join(f"- [[{f}/README|{f}]] — {p}" for f, p in _VAULT_TREE.items())
                + "\n\n> Ejecuta `knowledge build` para poblar la bóveda con el grafo actual.\n",
                encoding="utf-8",
            )

        return AgentResult(
            True, self.name, "setup_vault",
            f"Bóveda creada en {target.relative_to(root)} con {len(created_dirs)} "
            f"carpeta(s) de árbol: {', '.join(created_dirs)}.",
            data={"vault": str(target), "tree": created_dirs, "created": True},
        )

    def build(self, *, vault_dir: str | None = None, export_obsidian: bool = True) -> AgentResult:
        """
        Actualiza el grafo con graphify y, si hay bóveda, lo exporta a Obsidian.
        Fija la caché en ``graphify-out/cache/`` antes de empezar para que todo
        lo caro (resúmenes, consultas posteriores) se reutilice.
        """
        root = self.ctx.root
        if not GraphifyTool.is_available(root):
            return AgentResult(
                False, self.name, "build",
                "graphify no está instalado. Ejecuta el skill /graphify una vez primero.",
            )

        self._cache_dir()
        warnings: list[str] = []

        try:
            update = GraphifyTool.update(root)
        except FileNotFoundError as exc:
            return AgentResult(False, self.name, "build", str(exc))
        if update.returncode != 0:
            warnings.append(f"graphify --update devolvió error: {update.stderr.strip()[:200]}")

        exported_to = None
        if export_obsidian:
            vaults = GraphifyTool.detect_obsidian_vaults(root)
            if vault_dir:
                vault = (root / vault_dir).resolve()
            elif vaults:
                vault = vaults[0]
            else:
                vault = None
            if vault is not None:
                try:
                    exp = GraphifyTool.export_obsidian(root, vault)
                    if exp.returncode == 0:
                        exported_to = str(vault.relative_to(root)) if vault.is_relative_to(root) else str(vault)
                    else:
                        warnings.append(f"export obsidian falló: {exp.stderr.strip()[:200]}")
                except FileNotFoundError as exc:
                    warnings.append(str(exc))
            else:
                warnings.append(
                    "No hay bóveda de Obsidian — se omitió la exportación. "
                    "Ejecuta 'knowledge setup_vault' para crear una."
                )

        graph_stats = {}
        if GraphifyTool.graph_exists(root):
            try:
                g = GraphifyTool.load_graph(root)
                graph_stats = {"nodes": len(g["nodes"]), "edges": len(g["edges"])}
            except Exception:  # noqa: BLE001
                pass

        msg = f"Grafo actualizado ({graph_stats.get('nodes', '?')} nodos)."
        if exported_to:
            msg += f" Exportado a Obsidian en {exported_to}."
        return AgentResult(
            True, self.name, "build", msg,
            data={"graph": graph_stats, "obsidian": exported_to}, warnings=warnings,
        )

    def summarize_parents(self, *, min_children: int = 3, top: int = 10, no_cache: bool = False) -> AgentResult:
        """
        Genera un resumen por cada nodo padre (hub) con la correlación entre
        sus hijos. Responde a "el nodo padre debe tener un resumen de qué
        tratan los nodos hijos" — cacheado en graphify-out/cache/.
        """
        root = self.ctx.root
        if not GraphifyTool.graph_exists(root):
            return AgentResult(
                False, self.name, "summarize_parents",
                "No hay grafo (graphify-out/graph.json). Ejecuta 'knowledge build' primero.",
            )
        self._cache_dir()

        try:
            graph = GraphifyTool.load_graph(root)
        except Exception as exc:  # noqa: BLE001
            return AgentResult(False, self.name, "summarize_parents", f"No se pudo leer el grafo: {exc}")

        def _compute() -> list:
            return GraphifyTool.parent_summaries(graph, min_children=min_children, top=top)

        if no_cache:
            summaries = _compute()
        else:
            # Clave de caché ligada al tamaño del grafo: si cambia, recomputa.
            cache_key = f"parents_{len(graph['nodes'])}_{len(graph['edges'])}_{min_children}_{top}"
            summaries = CacheTool.disk_cache(name=cache_key)(_compute)()

        if not summaries:
            return AgentResult(
                True, self.name, "summarize_parents",
                f"Ningún nodo tiene ≥{min_children} hijos — el grafo es plano o pequeño.",
                data=[],
            )

        lines = [f"{s['label']}: {s['summary']}" for s in summaries[:top]]
        return AgentResult(
            True, self.name, "summarize_parents",
            f"{len(summaries)} nodo(s) padre resumido(s):\n" + "\n".join(lines),
            data=summaries,
        )

    def sync(self, *, vault_dir: str | None = None) -> AgentResult:
        """
        Punto de entrada único para "pon el grafo y Obsidian al día". Lo usa el
        `git` agent antes de un commit. Equivale a ``build`` pero con un mensaje
        pensado para el flujo de commit y sin fallar si no hay grafo todavía.
        """
        root = self.ctx.root
        if not GraphifyTool.graph_exists(root) and not GraphifyTool.is_available(root):
            return AgentResult(
                True, self.name, "sync",
                "No hay grafo ni graphify instalado — nada que sincronizar.",
                data={"skipped": True},
            )
        return self.build(vault_dir=vault_dir, export_obsidian=True)
