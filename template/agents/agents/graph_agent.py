"""
agents.agents.graph_agent — Inspección de gráficos en `reports/figures/`.

Ver `agents/tools/vision_tool.py` para el límite honesto de lo que esto
puede hacer: métricas estructurales (dimensiones, varianza de píxeles,
aspect ratio), no comprensión semántica del contenido del gráfico.
"""

from __future__ import annotations

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent


@register_agent
class GraphAgent(BaseAgent):
    name = "graph"
    description = (
        "Inspecciona los gráficos generados en reports/figures/: detecta figuras "
        "vacías o mal renderizadas, aspect ratios inusuales, baja resolución, "
        "bordes blancos excesivos y archivos demasiado grandes."
    )
    capabilities = [
        "grafico", "grafico", "figura", "plot", "reports/figures",
        "visualizacion", "chart", "resolucion", "calidad imagen",
        "figuras", "figuras del reporte",
    ]

    def actions(self) -> dict:
        return {
            "list_figures": self.list_figures,
            "audit_figures": self.audit_figures,
        }

    def list_figures(self) -> AgentResult:
        from agents.tools.vision_tool import VisionTool

        figures = VisionTool.list_figures(self.ctx.figures_dir)
        return AgentResult(
            True, self.name, "list_figures",
            f"{len(figures)} figura(s) en {self.ctx.figures_dir.relative_to(self.ctx.root)}.",
            data=[str(p.name) for p in figures],
        )

    def audit_figures(self) -> AgentResult:
        from agents.tools.vision_tool import VisionTool

        figures = VisionTool.list_figures(self.ctx.figures_dir)
        if not figures:
            return AgentResult(
                True, self.name, "audit_figures",
                "No hay figuras en reports/figures/ todavía (ejecuta 'make train' o 'make pipeline' primero).",
                data=[],
            )

        results = []
        all_warnings: list[str] = []
        for fig_path in figures:
            metrics = VisionTool.inspect(fig_path)
            entry: dict = {
                "file": fig_path.name,
                "width": metrics.width,
                "height": metrics.height,
                "aspect_ratio": metrics.aspect_ratio,
                "pixel_std": metrics.pixel_std,
                "mostly_blank": metrics.mostly_blank,
                "warnings": list(metrics.warnings),
                "size_kb": round(fig_path.stat().st_size / 1024, 1),
            }

            # Baja resolución (< 500px en la dimensión menor)
            min_dim = min(metrics.width, metrics.height)
            if 0 < min_dim < 500:
                entry["warnings"].append(f"Baja resolución ({min_dim}px en la menor dimensión)")

            # Archivo muy grande (> 5 MB)
            if fig_path.stat().st_size > 5 * 1024 * 1024:
                entry["warnings"].append(f"Archivo grande ({entry['size_kb'] / 1024:.1f} MB)")

            # Aspect ratio extremo (muy panorámico o muy vertical)
            if metrics.aspect_ratio > 0 and abs(metrics.aspect_ratio - 4/3) > 2.0:
                entry["warnings"].append(f"Aspect ratio inusual ({metrics.aspect_ratio:.2f}:1) — posible borde blanco")

            results.append(entry)
            all_warnings.extend(f"{fig_path.name}: {w}" for w in entry["warnings"])

        n_flagged = sum(1 for r in results if r["warnings"])
        return AgentResult(
            True, self.name, "audit_figures",
            f"{len(results)} figura(s) analizada(s), {n_flagged} con avisos.",
            data=results, warnings=all_warnings,
        )
