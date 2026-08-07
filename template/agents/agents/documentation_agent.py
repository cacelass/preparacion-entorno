"""
agents.agents.documentation_agent — Mantiene README, CHANGELOG y docs/ al día.

Conoce el formato exacto que ya usa este template:
  - CHANGELOG.md sigue Keep a Changelog (ver la cabecera real del archivo).
  - README.md documenta los targets de `make` en una sección "Comandos" /
    similar — este agente compara esa lista contra los targets reales del
    Makefile para detectar documentación desincronizada, en vez de asumir
    que README y Makefile nunca se desalinean.
  - `make docs` ya sabe generar la documentación Sphinx; este agente solo
    la invoca, no reimplementa esa lógica.
"""

from __future__ import annotations

import json
import re
from datetime import date

from agents.agents.git_agent import GitAgent
from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.exceptions import MissingDependencyError, ToolExecutionError
from agents.tools.filesystem_tool import FilesystemTool
from agents.tools.process_tool import run_command

_MAKE_TARGET_RE = re.compile(r"^([a-zA-Z_-]+):", re.MULTILINE)
# Targets internos que no tiene sentido exigir en el README (ayudan al propio Makefile).
_INTERNAL_TARGETS = {"help", ".PHONY", ".DEFAULT_GOAL"}


@register_agent
class DocumentationAgent(BaseAgent):
    name = "documentation"
    description = "Sincroniza README con el Makefile real, actualiza CHANGELOG.md, genera docs Sphinx."
    capabilities = ["readme", "changelog", "documentacion", "docs", "sphinx"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fs = FilesystemTool(root=self.ctx.root)

    def actions(self) -> dict:
        return {
            "check_readme_makefile_sync": self.check_readme_makefile_sync,
            "update_changelog": self.update_changelog,
            "update_prd": self.update_prd,
            "build_docs": self.build_docs,
            "bump_version": self.bump_version,
        }

    def check_readme_makefile_sync(self) -> AgentResult:
        """
        Compara los targets reales de `Makefile` con las menciones de `make
        <target>` en `README.md`. Señala targets del Makefile que el README
        no menciona — un indicio de documentación desactualizada.
        """
        makefile_path = self.ctx.root / "Makefile"
        readme_path = self.ctx.readme_file
        if not makefile_path.exists() or not readme_path.exists():
            return AgentResult(False, self.name, "check_readme_makefile_sync", "Falta Makefile o README.md en la raíz.")

        makefile_text = makefile_path.read_text(encoding="utf-8")
        readme_text = readme_path.read_text(encoding="utf-8")

        targets = {
            t for t in _MAKE_TARGET_RE.findall(makefile_text)
            if t not in _INTERNAL_TARGETS and not t.startswith(".")
        }
        undocumented = sorted(t for t in targets if f"make {t}" not in readme_text)

        warnings = [f"'make {t}' no aparece mencionado en README.md" for t in undocumented]
        return AgentResult(
            True, self.name, "check_readme_makefile_sync",
            f"{len(targets)} targets en Makefile, {len(undocumented)} sin mencionar en README.md.",
            data={"all_targets": sorted(targets), "undocumented": undocumented}, warnings=warnings,
        )

    def update_changelog(self, *, since_tag: str | None = None, dry_run: bool = True,
                         feature_id: str = "", feature_title: str = "") -> AgentResult:
        """
        Genera una entrada de changelog (vía GitAgent) e, si `dry_run=False`,
        la inserta en `CHANGELOG.md` justo después de la cabecera del
        archivo (antes de la primera entrada de versión existente).

        Si se pasan `feature_id` y `feature_title`, la entrada se construye a
        partir de la feature del arnés (`### Añadido · <title> (<id>)`) en vez
        de derivarse del git log. Es el modo que usa `GitAgent.commit_feature`
        al cerrar una feature: entre features no hay tags, y derivar del log
        duplicaría entradas de un cierre al siguiente.
        """
        if feature_id and feature_title:
            entry = (
                f"## [Unreleased] — {date.today().isoformat()}\n\n"
                f"### Añadido\n\n"
                f"- {feature_title} ({feature_id})\n"
            )
            if dry_run:
                return AgentResult(
                    True, self.name, "update_changelog",
                    "Entrada de feature generada en modo dry_run (no se escribió en disco).",
                    data=entry,
                )
            return self._insert_changelog(entry)

        git_agent = GitAgent(context=self.ctx)
        changelog_result = git_agent.run("generate_changelog", since_tag=since_tag)
        if not changelog_result.success or not changelog_result.data:
            return AgentResult(
                changelog_result.success, self.name, "update_changelog",
                changelog_result.message or "No hay cambios nuevos que añadir al changelog.",
            )

        entry = changelog_result.data
        if dry_run:
            return AgentResult(
                True, self.name, "update_changelog",
                "Entrada generada en modo dry_run (no se escribió en disco). "
                "Vuelve a llamar con dry_run=False para aplicarla.",
                data=entry,
            )

        return self._insert_changelog(entry)

    def _insert_changelog(self, entry: str) -> AgentResult:
        """Inserta `entry` en CHANGELOG.md tras la cabecera del archivo."""
        if not self.ctx.changelog_file.exists():
            new_content = f"# Changelog\n\n{entry}\n"
        else:
            current = self.ctx.changelog_file.read_text(encoding="utf-8")
            # Inserta tras la primera línea en blanco que sigue al título (cabecera del archivo),
            # que es donde ya viven las notas introductorias de CHANGELOG.md en este template.
            marker = "\n---\n"
            if marker in current:
                head, _, tail = current.partition(marker)
                new_content = f"{head}{marker}\n{entry}\n{tail.lstrip(chr(10))}"
            else:
                new_content = f"{current.rstrip()}\n\n{entry}\n"

        self.ctx.changelog_file.write_text(new_content, encoding="utf-8")
        return AgentResult(
            True, self.name, "update_changelog", "CHANGELOG.md actualizado.", data=entry,
        )

    # -- PRD vivo -------------------------------------------------------------
    def update_prd(self, *, dry_run: bool = False) -> AgentResult:
        """
        Genera `docs/prd.md` a partir del estado REAL del proyecto: el objetivo
        (`references/00-objetivo.md`), el backlog (`harness/featureslist.json`)
        y los contratos Gherkin (`features/*.feature`).

        Es un documento DERIVADO, no una fuente de verdad: si el backlog
        cambia, `docs/prd.md` se queda atrás hasta que se vuelve a ejecutar
        esta acción. Por eso el `lider` la llama al cerrar una feature. Un PRD
        a mano siempre acaba desfasado; este nace del mismo JSON que guía el
        arnés.
        """
        sections: list[str] = []
        warnings: list[str] = []

        objetivo = self._objetivo_section()
        sections.append(objetivo["markdown"])

        backlog = self._backlog_section()
        sections.append(backlog["markdown"])
        warnings.extend(backlog["warnings"])

        contratos = self._gherkin_section()
        sections.append(contratos["markdown"])
        warnings.extend(contratos["warnings"])

        prd = self._render_prd(sections)

        if dry_run:
            return AgentResult(
                True, self.name, "update_prd",
                "PRD generado en modo dry_run (no se escribió en disco).",
                data={"markdown": prd, "path": "docs/prd.md"}, warnings=warnings,
            )

        target = self.ctx.docs_dir / "prd.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(prd, encoding="utf-8")
        return AgentResult(
            True, self.name, "update_prd",
            "docs/prd.md regenerado desde el estado actual del proyecto.",
            data={"path": str(target.relative_to(self.ctx.root))}, warnings=warnings,
        )

    def _objetivo_section(self) -> dict:
        """El objetivo del proyecto desde references/00-objetivo.md (SCOPE-001)."""
        ref = self.ctx.root / "references" / "00-objetivo.md"
        if not ref.exists():
            return {
                "markdown": (
                    "## Objetivo\n\n"
                    "_(sin definir — ejecuta la feature SCOPE-001 del backlog, que escribe "
                    "`references/00-objetivo.md` con la pregunta, la métrica de éxito y el "
                    "criterio de parada)_\n"
                )
            }
        return {"markdown": f"## Objetivo\n\n{ref.read_text(encoding='utf-8').strip()}\n"}

    def _backlog_section(self) -> dict:
        """Resumen del backlog: recuento por estado + features listadas."""
        backlog_file = self.ctx.root / "harness" / "featureslist.json"
        if not backlog_file.exists():
            return {
                "markdown": "## Alcance\n\n_(no hay backlog: falta harness/featureslist.json)_\n",
                "warnings": [],
            }
        try:
            doc = json.loads(backlog_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return {
                "markdown": "## Alcance\n\n_(backlog ilegible: " + str(exc) + ")_\n",
                "warnings": [f"harness/featureslist.json ilegible: {exc}"],
            }

        features = doc.get("features", []) if isinstance(doc, dict) else []
        if not features:
            return {"markdown": "## Alcance\n\n_(backlog vacío)_\n", "warnings": []}

        counts: dict[str, int] = {}
        for f in features:
            status = f.get("status", "pending")
            counts[status] = counts.get(status, 0) + 1

        lines = ["## Alcance", ""]
        resumen = " · ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        lines.append(f"**Backlog:** {resumen}")
        lines.append("")
        lines.append("| Feature | Estado | Título |")
        lines.append("|---------|--------|--------|")
        for f in features:
            fid = f.get("id", "?")
            status = f.get("status", "pending")
            title = (f.get("title") or "").replace("|", "\\|")
            lines.append(f"| {fid} | {status} | {title} |")
        lines.append("")

        # El PRD también es un roadmap: las features con depends_on ordenan el trabajo.
        por_hacer = [f for f in features if f.get("status") not in ("done", "blocked")]
        if por_hacer:
            lines.append("**Pendiente de cerrar:** " + ", ".join(
                f.get("id", "?") for f in por_hacer
            ))
            lines.append("")

        return {"markdown": "\n".join(lines), "warnings": []}

    def _gherkin_section(self) -> dict:
        """Los contratos Gherkin existentes (features/*.feature), si el proyecto los tiene."""
        features_dir = self.ctx.root / "features"
        if not features_dir.is_dir():
            return {"markdown": "", "warnings": []}
        contratos = sorted(features_dir.glob("*.feature"))
        if not contratos:
            return {"markdown": "", "warnings": []}

        lines = ["## Contratos de aceptación (Gherkin)", ""]
        for c in contratos:
            first = c.read_text(encoding="utf-8").strip().splitlines()
            titulo = next((ln for ln in first if ln.startswith("Feature:")), c.name)
            lines.append(f"- `{c.name}` — {titulo.removeprefix('Feature:').strip()}")
        lines.append("")
        return {"markdown": "\n".join(lines), "warnings": []}

    def _render_prd(self, sections: list[str]) -> str:
        header = (
            f"# Product Requirements Document\n\n"
            f"> Documento **generado** (`documentation update_prd`) desde el estado del "
            f"proyecto: `references/00-objetivo.md`, `harness/featureslist.json` y "
            f"`features/*.feature`. No lo edites a mano — se sobrescribe. "
            f"Actualizado: {date.today().isoformat()}\n"
        )
        return header + "\n" + "\n".join(s for s in sections if s) + "\n"

    def build_docs(self) -> AgentResult:
        """
        Ejecuta `sphinx-apidoc` + build HTML, igual que `make docs`. Requiere
        que `sphinx` esté instalado (extra `dev`: `uv sync --extra dev`).
        """
        module = self.ctx.config.project_slug
        if not module:
            return AgentResult(
                False, self.name, "build_docs",
                "project_slug está vacío — revisa .copier-answers.yml antes de generar documentación.",
            )
        try:
            apidoc = run_command(
                ["uv", "run", "sphinx-apidoc", "-o", "docs/source/", module],
                cwd=self.ctx.root, timeout=120,
            )
        except MissingDependencyError as exc:
            return AgentResult(False, self.name, "build_docs", str(exc))

        if not apidoc.ok:
            return AgentResult(False, self.name, "build_docs", "sphinx-apidoc falló.", data=apidoc.stderr)

        try:
            html = run_command(["make", "html"], cwd=self.ctx.docs_dir, timeout=180)
        except (MissingDependencyError, ToolExecutionError) as exc:
            return AgentResult(False, self.name, "build_docs", str(exc))

        if not html.ok:
            return AgentResult(False, self.name, "build_docs", "El build HTML de Sphinx falló.", data=html.stderr)

        return AgentResult(True, self.name, "build_docs", "Documentación generada en docs/build/html/.")

    def bump_version(self, *, new_version: str) -> AgentResult:
        """
        Actualiza el número de versión en `pyproject.toml` (`version = "..."`)
        y en `README.md` (el badge `Version-X-green` y la línea
        `**Versión:** X`) — comprobé estos tres sitios exactos leyendo el
        `README.md`/`pyproject.toml` reales de este template antes de escribir
        este método, no son una suposición.

        Si algún patrón no aparece en el archivo (p. ej. porque el usuario
        reescribió el README a mano y quitó el badge), se avisa explícitamente
        en vez de fallar en silencio o fingir que se actualizó algo que no
        estaba ahí.
        """
        changed_files = []
        warnings = []

        pyproject_path = self.ctx.pyproject_file
        if pyproject_path.exists():
            text = pyproject_path.read_text(encoding="utf-8")
            new_text, n = re.subn(
                r'^version = "[^"]*"', f'version = "{new_version}"', text, count=1, flags=re.MULTILINE
            )
            if n:
                pyproject_path.write_text(new_text, encoding="utf-8")
                changed_files.append(str(pyproject_path.relative_to(self.ctx.root)))
            else:
                warnings.append("No se encontró 'version = \"...\"' en pyproject.toml — no se tocó.")
        else:
            warnings.append("No existe pyproject.toml en la raíz del proyecto.")

        if self.ctx.readme_file.exists():
            text = self.ctx.readme_file.read_text(encoding="utf-8")
            text, n_badge = re.subn(r"Version-[^-]+-green", f"Version-{new_version}-green", text, count=1)
            text, n_line = re.subn(r"(\*\*Versión:\*\*\s*)[^\s{·]+", rf"\g<1>{new_version}", text, count=1)
            if n_badge or n_line:
                self.ctx.readme_file.write_text(text, encoding="utf-8")
                changed_files.append(str(self.ctx.readme_file.relative_to(self.ctx.root)))
            if not n_badge:
                warnings.append("No se encontró el badge 'Version-X-green' en README.md.")
            if not n_line:
                warnings.append("No se encontró la línea '**Versión:** X' en README.md.")
        else:
            warnings.append("No existe README.md en la raíz del proyecto.")

        if not changed_files:
            return AgentResult(False, self.name, "bump_version", "No se actualizó ningún archivo.", warnings=warnings)

        return AgentResult(
            True, self.name, "bump_version",
            f"Versión actualizada a '{new_version}' en: {', '.join(changed_files)}.",
            data={"new_version": new_version, "changed_files": changed_files}, warnings=warnings,
        )
