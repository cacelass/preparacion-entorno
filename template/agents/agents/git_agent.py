"""
agents.agents.git_agent — Automatización de Git para este template.

Conoce la estructura del proyecto: sabe que el código vive en
`{{ project_slug }}`, que los tests viven en `tests/` (y por tanto puede
avisar si un commit toca código sin tocar tests), y que el CHANGELOG del
proyecto sigue el formato Keep a Changelog (mismo que usa `CHANGELOG.md` en
la raíz de este template).

Al hacer commit, también detecta si el proyecto tiene `graphify-out/` (grafo
de conocimiento, ver github.com/anomalyco/graphify) o `.obsidian/` (bóveda de
Obsidian) y los actualiza automáticamente.
"""

from __future__ import annotations

import os
import re
from datetime import date

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.exceptions import MissingDependencyError
from agents.tools.git_tool import GitTool
from agents.tools.process_tool import ProcessResult, run_command


@register_agent
class GitAgent(BaseAgent):
    name = "git"
    description = (
        "Conventional Commits, análisis de diffs, changelog, release notes, "
        "detección de breaking changes y preparación de Pull Requests."
    )
    capabilities = [
        "git", "commit", "diff", "release", "pull request", "pr", "genera", "generar",
        "genera el changelog", "breaking change", "rama", "branch",
        "cierra la feature", "cerrar feature", "commit de la feature",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.git = GitTool(repo_root=self.ctx.root)

    def action_aliases(self) -> dict:
        return {
            # "haz un commit" no comparte ninguna palabra con "suggest_commit_message"
            # (inglés) — sin esto, best_action no tendría forma de distinguir cuál
            # de las dos acciones de commit es la más natural para una petición genérica.
            "commit_with_changelog": ["commit", "confirma", "guarda los cambios", "haz commit"],
            "suggest_commit_message": ["sugiere", "sugerir", "mensaje", "propon un mensaje"],
            "commit_feature": ["cierra la feature", "cerrar feature", "commit de la feature", "feature lista"],
            "commit_atomic": ["separa", "dividir", "commits atomicos", "split", "separar en commits"],
            "tag_release": ["tag", "etiqueta", "version", "versión", "release", "publica", "lanzamiento"],
        }

    def actions(self) -> dict:
        return {
            "status": self.status,
            "analyze_diff": self.analyze_diff,
            "suggest_commit_message": self.suggest_commit_message,
            "generate_changelog": self.generate_changelog,
            "generate_release_notes": self.generate_release_notes,
            "detect_breaking_changes": self.detect_breaking_changes,
            "prepare_pr_summary": self.prepare_pr_summary,
            "commit_with_changelog": self.commit_with_changelog,
            "commit_atomic": self.commit_atomic,
            "commit_feature": self.commit_feature,
            "tag_release": self.tag_release,
            "create_branch": self.create_branch,
            "merge_branch": self.merge_branch,
        }

    def _guard_repo(self, action: str) -> AgentResult | None:
        try:
            if not self.git.is_repo():
                return AgentResult(False, self.name, action, "Este directorio no es un repositorio git.")
        except MissingDependencyError as exc:
            return AgentResult(False, self.name, action, str(exc))
        return None

    # -------------------------------------------------------------------------
    def status(self) -> AgentResult:
        guard = self._guard_repo("status")
        if guard:
            return guard
        entries = self.git.status_porcelain()
        branch = self.git.current_branch()
        return AgentResult(
            True, self.name, "status",
            f"Rama '{branch}', {len(entries)} archivo(s) modificado(s).",
            data={"branch": branch, "changes": entries},
        )

    def analyze_diff(self, *, staged: bool = False) -> AgentResult:
        guard = self._guard_repo("analyze_diff")
        if guard:
            return guard

        changed = self.git.changed_files(staged=staged)
        if not changed:
            return AgentResult(True, self.name, "analyze_diff", "No hay cambios que analizar.", data=[])

        stat = self.git.diff_stat(staged=staged)
        touches_tests = any(f.startswith("tests/") for f in changed)
        touches_source = any(f.startswith(f"{self.ctx.config.project_slug}/") for f in changed)

        warnings = []
        if touches_source and not touches_tests:
            warnings.append(
                "El diff toca código en "
                f"{self.ctx.config.project_slug}/ pero no toca tests/ — "
                f"considera si necesitas añadir o actualizar un test."
            )

        return AgentResult(
            True, self.name, "analyze_diff",
            f"{len(changed)} archivo(s) modificado(s).",
            data={"changed_files": changed, "stat": stat},
            warnings=warnings,
        )

    def suggest_commit_message(self, *, staged: bool = True) -> AgentResult:
        guard = self._guard_repo("suggest_commit_message")
        if guard:
            return guard

        changed = self.git.changed_files(staged=staged)
        if not changed:
            return AgentResult(
                False, self.name, "suggest_commit_message",
                "No hay cambios en staging (ni en el working tree) que resumir."
            )

        commit_type = self.git.guess_commit_type(changed)
        scope = self.ctx.config.project_slug if any(
            f.startswith(f"{self.ctx.config.project_slug}/") for f in changed
        ) else ""
        scope_part = f"({scope})" if scope else ""
        files_preview = ", ".join(changed[:3]) + (f" y {len(changed) - 3} más" if len(changed) > 3 else "")

        suggestion = f"{commit_type}{scope_part}: actualiza {files_preview}"
        return AgentResult(
            True, self.name, "suggest_commit_message",
            "Sugerencia generada — revísala antes de usarla, es un punto de partida.",
            data={"suggested_message": suggestion, "detected_type": commit_type, "changed_files": changed},
            warnings=["El 'subject' es un placeholder genérico: sustitúyelo por una descripción real del cambio."],
        )

    def generate_changelog(self, *, since_tag: str | None = None, max_count: int = 100) -> AgentResult:
        guard = self._guard_repo("generate_changelog")
        if guard:
            return guard

        tag = since_tag or self.git.last_tag()
        commits = self.git.log(max_count=max_count, since_tag=tag)
        if not commits:
            return AgentResult(True, self.name, "generate_changelog", "No hay commits nuevos desde el último tag.", data="")

        grouped: dict[str, list[str]] = {}
        unclassified: list[str] = []
        for commit in commits:
            parsed = self.git.parse_conventional_commit(commit["subject"])
            if parsed:
                grouped.setdefault(parsed["type"], []).append(parsed["subject"])
            else:
                unclassified.append(commit["subject"])

        # Mismas etiquetas de sección que ya usa CHANGELOG.md de este template.
        section_titles = {
            "feat": "### Añadido", "fix": "### Corrección de bugs", "docs": "### Documentación",
            "refactor": "### Refactorización", "perf": "### Rendimiento", "test": "### Tests",
            "build": "### Build / dependencias", "ci": "### CI", "chore": "### Mantenimiento",
            "revert": "### Reversiones",
        }

        lines = [f"## [Unreleased] — {date.today().isoformat()}", ""]
        for commit_type, title in section_titles.items():
            if commit_type in grouped:
                lines.append(title)
                lines.append("")
                lines.extend(f"- {msg}" for msg in grouped[commit_type])
                lines.append("")
        if unclassified:
            lines.append("### Otros")
            lines.append("")
            lines.extend(f"- {msg}" for msg in unclassified)
            lines.append("")

        markdown = "\n".join(lines).rstrip() + "\n"
        return AgentResult(
            True, self.name, "generate_changelog",
            f"Changelog generado a partir de {len(commits)} commit(s) desde {tag or 'el inicio del repo'}.",
            data=markdown,
        )

    def generate_release_notes(self, *, since_tag: str | None = None) -> AgentResult:
        changelog_result = self.generate_changelog(since_tag=since_tag)
        if not changelog_result.success:
            return changelog_result
        breaking = self.detect_breaking_changes(since_tag=since_tag)
        header = f"# Release notes — {date.today().isoformat()}\n\n"
        if breaking.data:
            header += "**Contiene breaking changes — revisa la sección correspondiente antes de actualizar.**\n\n"
        body = header + changelog_result.data
        return AgentResult(
            True, self.name, "generate_release_notes", "Release notes generadas.", data=body,
        )

    def detect_breaking_changes(self, *, since_tag: str | None = None, max_count: int = 100) -> AgentResult:
        guard = self._guard_repo("detect_breaking_changes")
        if guard:
            return guard

        tag = since_tag or self.git.last_tag()
        commits = self.git.log(max_count=max_count, since_tag=tag)
        breaking = []
        for commit in commits:
            parsed = self.git.parse_conventional_commit(commit["subject"])
            if parsed and parsed["breaking"]:
                breaking.append(commit["subject"])
            elif "BREAKING CHANGE" in commit["subject"].upper():
                breaking.append(commit["subject"])

        note = (
            "Esta detección solo mira mensajes de commit (marca '!' de Conventional "
            "Commits o texto 'BREAKING CHANGE'). No analiza el diff en busca de "
            "funciones públicas eliminadas o firmas cambiadas — ese análisis más "
            "profundo es una extensión natural de este agente, no está implementado "
            "todavía."
        )
        return AgentResult(
            True, self.name, "detect_breaking_changes",
            f"{len(breaking)} breaking change(s) detectado(s) por mensaje de commit.",
            data=breaking, warnings=[note],
        )

    def prepare_pr_summary(self, *, since_tag: str | None = None) -> AgentResult:
        guard = self._guard_repo("prepare_pr_summary")
        if guard:
            return guard

        diff_result = self.analyze_diff(staged=False)
        changelog_result = self.generate_changelog(since_tag=since_tag)
        branch = self.git.current_branch()

        if not diff_result.data and not changelog_result.data:
            return AgentResult(True, self.name, "prepare_pr_summary", "No hay cambios que resumir para un PR.")

        title = f"{branch}: cambios pendientes de revisión"
        body_parts = [f"## Resumen\n\nRama: `{branch}`\n"]
        if isinstance(diff_result.data, dict) and diff_result.data.get("changed_files"):
            body_parts.append("## Archivos modificados\n")
            body_parts.extend(f"- `{f}`" for f in diff_result.data["changed_files"])
            body_parts.append("")
        if changelog_result.data:
            body_parts.append("## Changelog\n")
            body_parts.append(changelog_result.data)

        return AgentResult(
            True, self.name, "prepare_pr_summary", "Resumen de PR generado.",
            data={"title": title, "body": "\n".join(body_parts)},
            warnings=diff_result.warnings,
        )

    def commit_with_changelog(self, *, message: str, since_tag: str | None = None) -> AgentResult:
        """
        Actualiza CHANGELOG.md (delegando en `DocumentationAgent`, no
        reimplementando esa lógica aquí — ver Filosofía en `agents/README.md`:
        "no tienen que ser independientes, pueden ayudarse entre sí") y hace
        `git add` + `git commit` incluyendo ese cambio en el mismo commit.

        Import perezoso de `DocumentationAgent` dentro del método (no a nivel
        de módulo): `documentation_agent.py` ya importa `GitAgent` a nivel de
        módulo para su propia acción `update_changelog` — un import circular
        a nivel de módulo en los dos sentidos rompería la carga del paquete.
        Haciendo este import dentro del método, para cuando se ejecuta ambos
        módulos ya están cargados y no hay ciclo real.
        """
        guard = self._guard_repo("commit_with_changelog")
        if guard:
            return guard

        from agents.agents.documentation_agent import DocumentationAgent

        doc_agent = DocumentationAgent(context=self.ctx)
        changelog_result = doc_agent.run("update_changelog", since_tag=since_tag, dry_run=False)

        warnings = list(changelog_result.warnings)
        if changelog_result.success and changelog_result.data:
            changelog_relative = str(self.ctx.changelog_file.relative_to(self.ctx.root))
            stage = self.git.add(changelog_relative)
            if not stage.ok:
                warnings.append(f"No se pudo hacer 'git add {changelog_relative}': {stage.stderr.strip()}")
        else:
            warnings.append(
                "CHANGELOG.md no se actualizó (sin commits nuevos que resumir, o el agente de "
                "documentación falló) — se hace el commit solo con el mensaje dado, sin changelog."
            )

        commit_result = self.git.commit(message)
        if not commit_result.ok:
            return AgentResult(
                False, self.name, "commit_with_changelog",
                f"'git commit' falló: {commit_result.stderr.strip()}", warnings=warnings,
            )

        graphify_warnings = self._update_graphify()
        warnings.extend(graphify_warnings)

        obsidian_warnings = self._sync_obsidian()
        warnings.extend(obsidian_warnings)

        extras = []
        if changelog_result.success and changelog_result.data:
            extras.append("CHANGELOG.md actualizado")
        if not graphify_warnings:
            extras.append("graphify actualizado")
        if not obsidian_warnings:
            extras.append("Obsidian indexado")
        extras_str = f" ({', '.join(extras)})" if extras else ""

        return AgentResult(
            True, self.name, "commit_with_changelog",
            f"Commit creado: '{message}'{extras_str}",
            data={"stdout": commit_result.stdout, "changelog_updated": bool(changelog_result.success and changelog_result.data)},
            warnings=warnings,
        )

    def commit_atomic(self, *, dry_run: bool = False, subjects: str | None = None) -> AgentResult:
        """
        Divide los cambios sin commitear en commits atómicos por área.

        Los lock files (`uv.lock`, `package-lock.json`...) no entran en ningún
        commit — se reportan aparte. El plan se ordena por dependencias (código
        antes que tests, tests antes que docs). Si un área temprana depende de
        una tardía (ciclo), el plan se RECHAZA antes de escribir nada y se
        sugiere commitear a mano.

        Con `dry_run=True` solo propone el plan (y nunca pide permiso, como
        toda propuesta). Sin dry-run commitea cada grupo en orden — la puerta
        de permisos pide confirmación porque escribe en el historial git.

        `subjects` es opcional: mensajes Conventional separados por `;`, uno
        por grupo, en orden. Sin él se generan mensajes placeholder (`feat:
        actualiza ...`) que hay que revisar.
        """
        guard = self._guard_repo("commit_atomic")
        if guard:
            return guard

        changed = [path for _, path in self.git.status_porcelain()]
        if not changed:
            return AgentResult(True, self.name, "commit_atomic", "No hay cambios que commitear.", data={"groups": []})

        plan = GitTool.plan_atomic(changed, self.ctx.root)
        if plan["cycle"]:
            return AgentResult(
                False, self.name, "commit_atomic",
                f"Plan rechazado antes de escribir nada: {plan['cycle']}",
                data={"plan": plan},
            )

        groups = plan["groups"]
        if not groups:
            return AgentResult(
                True, self.name, "commit_atomic",
                "Solo hay lock files — no se genera ningún commit atómico.",
                data={"groups": [], "excluded": plan["excluded"]},
            )

        subs = [s.strip() for s in subjects.split(";")] if subjects else []
        for i, group in enumerate(groups):
            provided = subs[i] if i < len(subs) and subs[i] else None
            if provided:
                if not GitTool.parse_conventional_commit(provided):
                    return AgentResult(
                        False, self.name, "commit_atomic",
                        f"El mensaje {i + 1} '{provided}' no es Conventional Commits.",
                        needs=[f"Proporciona un mensaje Conventional para el grupo '{group['area']}' (--subjects)."],
                    )
                group["message"] = provided
            else:
                preview = ", ".join(group["files"][:2]) + (f" y {len(group['files']) - 2} más" if len(group["files"]) > 2 else "")
                group["message"] = f"{group['type']}: actualiza {preview}"

        if dry_run:
            return AgentResult(
                True, self.name, "commit_atomic",
                f"{len(groups)} commit(s) atómico(s) propuesto(s) — nada escrito.",
                data={"groups": groups, "excluded": plan["excluded"]},
                warnings=[
                    "Los subjects por defecto son placeholders — pasa --subjects "
                    "'feat: ...; test: ...; docs: ...' (uno por grupo, en orden) para controlarlos.",
                ],
            )

        warnings: list[str] = []
        if plan["excluded"]:
            warnings.append(f"Lock files fuera del plan: {', '.join(plan['excluded'])}")

        created: list[dict] = []
        for group in groups:
            add = self.git.add(*group["files"])
            if not add.ok:
                warnings.append(f"'git add' falló para {group['files']}: {add.stderr.strip()}")
                continue
            commit_result = self.git.commit_paths(group["message"], group["files"])
            if not commit_result.ok:
                warnings.append(f"'git commit' falló: {commit_result.stderr.strip()}")
                continue
            created.append({"area": group["area"], "message": group["message"]})

        return AgentResult(
            True, self.name, "commit_atomic",
            f"{len(created)}/{len(groups)} commit(s) atómico(s) creado(s).",
            data={"created": created, "excluded": plan["excluded"]},
            warnings=warnings,
        )

    def commit_feature(self, *, id: str = "", title: str = "", message: str = "",
                       dry_run: bool = False) -> AgentResult:
        """
        Cierre de una feature del arnés: README y versión al día + commit.

        Es el paso que el arnés espera tras `harness finish`: sube el patch de
        la versión (0.1.0 → 0.1.1) en pyproject.toml y README, añade la
        feature al CHANGELOG y commitea todo junto con un mensaje Conventional
        (`feat(<id>): <title>`). No crea tag: el tag es de `tag_release`.

        Con `dry_run=True` solo PROPONE: devuelve la versión que tocaría, el
        mensaje y los ficheros que entrarían, sin escribir nada. El líder la
        usa para mostrar la propuesta al usuario antes de confirmar.
        """
        guard = self._guard_repo("commit_feature")
        if guard:
            return guard

        if not id or not title:
            return AgentResult(
                False, self.name, "commit_feature",
                "Faltan datos para cerrar la feature.",
                needs=["¿Qué feature cierro? (--id)", "¿Cuál es su título? (--title)"],
            )

        from agents.agents.documentation_agent import DocumentationAgent

        doc_agent = DocumentationAgent(context=self.ctx)
        next_version = self._next_patch_version()
        changed = self._proposed_files()
        suggested = message or f"feat({id}): {title}"

        if dry_run:
            bump_part = f"bump a '{next_version}'" if next_version else "sin bump (no hay versión)"
            return AgentResult(
                True, self.name, "commit_feature",
                f"Propuesta de cierre de {id}: {bump_part}, commit "
                f"'{suggested}'. Nada escrito todavía.",
                data={
                    "id": id, "title": title, "next_version": next_version,
                    "suggested_message": suggested, "changed_files": changed,
                },
                warnings=(
                    ["No se pudo leer la versión en pyproject.toml — se omite el bump."]
                    if not next_version else []
                ),
            )

        warnings: list[str] = []
        bump_success = False
        if next_version:
            bump = doc_agent.run("bump_version", new_version=next_version)
            warnings.extend(bump.warnings)
            bump_success = bump.success
        else:
            warnings.append("No se pudo leer la versión en pyproject.toml — se omite el bump.")

        changelog = doc_agent.run(
            "update_changelog", feature_id=id, feature_title=title, dry_run=False
        )
        if not changelog.success:
            warnings.append(changelog.message)

        stage = self.git.add("-A")
        if not stage.ok:
            warnings.append(f"No se pudo hacer 'git add -A': {stage.stderr.strip()}")

        commit_result = self.git.commit(suggested)
        if not commit_result.ok:
            return AgentResult(
                False, self.name, "commit_feature",
                f"'git commit' falló: {commit_result.stderr.strip()}", warnings=warnings,
            )

        extras = []
        if next_version and bump_success:
            extras.append(f"versión {next_version}")
        if changelog.success:
            extras.append("CHANGELOG.md actualizado")
        extras_str = f" ({', '.join(extras)})" if extras else ""
        return AgentResult(
            True, self.name, "commit_feature",
            f"{id} cerrada con commit '{suggested}'{extras_str}.",
            data={
                "id": id, "title": title, "message": suggested, "next_version": next_version,
            },
            warnings=warnings,
        )

    def _next_patch_version(self) -> str | None:
        """Lee `version = "X.Y.Z"` de pyproject.toml y devuelve el siguiente patch."""
        pyproject = self.ctx.pyproject_file
        if not pyproject.exists():
            return None
        match = re.search(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"',
                          pyproject.read_text(encoding="utf-8"), re.MULTILINE)
        if not match:
            return None
        major, minor, patch = (int(g) for g in match.groups())
        return f"{major}.{minor}.{patch + 1}"

    def _proposed_files(self) -> list[str]:
        """Ficheros que entrarían en el commit: cambios rastreados + sin trackear."""
        return [path for _, path in self.git.status_porcelain()]

    def tag_release(self, *, version: str, message: str | None = None, since_tag: str | None = None) -> AgentResult:
        """
        Hace TODO lo que conlleva un release en un único paso, en vez de
        obligar a encadenar acciones a mano: actualiza la versión en
        pyproject.toml y README.md (delegando en
        `DocumentationAgent.bump_version`), actualiza CHANGELOG.md
        (delegando en `commit_with_changelog`, que a su vez delega en
        `DocumentationAgent.update_changelog`), genera un workflow de CI si
        el proyecto todavía no tiene ninguno (delegando en `CICDAgent`) o
        avisa si el que ya existe tiene problemas sin resolver, hace un
        único commit con todo eso junto, y crea el tag de git apuntando a
        ese commit.

        Si `version` ya existe como tag, falla explícitamente en vez de
        sobreescribirlo — un tag movido es un problema real para cualquiera
        que ya lo haya usado.
        """
        guard = self._guard_repo("tag_release")
        if guard:
            return guard

        if self.git.tag_exists(version):
            return AgentResult(False, self.name, "tag_release", f"El tag '{version}' ya existe — no se sobreescribe.")

        from agents.agents.documentation_agent import DocumentationAgent

        doc_agent = DocumentationAgent(context=self.ctx)
        bump_result = doc_agent.run("bump_version", new_version=version)
        warnings = list(bump_result.warnings)
        if bump_result.success:
            stage = self.git.add(*bump_result.data["changed_files"])
            if not stage.ok:
                warnings.append(f"No se pudo hacer 'git add' de los archivos de versión: {stage.stderr.strip()}")
        else:
            warnings.append(
                "No se pudo actualizar la versión en pyproject.toml/README.md "
                "(ver warnings de bump_version) — se continúa igualmente con el commit y el tag."
            )

        # CI/CD: si el proyecto generado todavía no tiene ningún workflow, se
        # crea uno ahora — "tagear sin CI" es exactamente el caso que puede
        # dar problemas más adelante sin que nadie se entere hasta que ya es
        # tarde. Si ya existe uno, se valida y se avisa (no se bloquea el
        # release por esto: un falso positivo de la validación no debería
        # impedir taguear).
        from agents.agents.cicd_agent import CICDAgent

        cicd_agent = CICDAgent(context=self.ctx)
        workflows_result = cicd_agent.run("list_workflows")
        if workflows_result.success and not workflows_result.data:
            generate_result = cicd_agent.run("generate_workflow")
            if generate_result.success:
                stage_ci = self.git.add(".github/workflows")
                if stage_ci.ok:
                    warnings.append(
                        "No había ningún workflow de CI — se generó '.github/workflows/ci.yml' "
                        "y se incluyó en este mismo commit de release."
                    )
                else:
                    warnings.append(f"Se generó el workflow de CI pero 'git add' falló: {stage_ci.stderr.strip()}")
            else:
                warnings.append(f"No había CI y no se pudo generar automáticamente: {generate_result.message}")
        elif workflows_result.success and workflows_result.data:
            for filename in workflows_result.data:
                validation = cicd_agent.run("validate_workflow", filename=filename)
                if not validation.success:
                    warnings.append(f"'{filename}' tiene problemas de CI sin resolver: {validation.data['problems']}")

        commit_message = message or f"chore(release): {version}"
        commit_result = self.commit_with_changelog(message=commit_message, since_tag=since_tag)
        warnings.extend(commit_result.warnings)
        if not commit_result.success:
            return AgentResult(
                False, self.name, "tag_release",
                f"No se pudo crear el commit de release: {commit_result.message}", warnings=warnings,
            )

        tag_result = self.git.create_tag(version, message=commit_message)
        if not tag_result.ok:
            return AgentResult(
                False, self.name, "tag_release",
                f"El commit se creó pero 'git tag' falló: {tag_result.stderr.strip()}", warnings=warnings,
            )

        return AgentResult(
            True, self.name, "tag_release",
            f"Tag '{version}' creado — versión, CHANGELOG.md y el commit de release quedaron todos juntos.",
            data={"version": version, "changelog_updated": commit_result.data.get("changelog_updated", False)},
            warnings=warnings,
        )

    def create_branch(self, *, branch_name: str, base_branch: str | None = None) -> AgentResult:
        """Crea una rama nueva a partir de main (o base_branch si se especifica)."""
        guard = self._guard_repo("create_branch")
        if guard:
            return guard

        if self._git("rev-parse", "--verify", f"refs/heads/{branch_name}", check=False).ok:
            return AgentResult(False, self.name, "create_branch", f"La rama '{branch_name}' ya existe.")

        base = base_branch or "main"
        checkout = self._git("checkout", base)
        if not checkout.ok:
            return AgentResult(False, self.name, "create_branch", f"No se pudo cambiar a '{base}': {checkout.stderr.strip()}")

        result = self._git("checkout", "-b", branch_name)
        if not result.ok:
            return AgentResult(False, self.name, "create_branch", f"No se pudo crear la rama: {result.stderr.strip()}")

        return AgentResult(
            True, self.name, "create_branch",
            f"Rama '{branch_name}' creada desde '{base}' y activada.",
            data={"branch": branch_name, "base": base},
        )

    def merge_branch(self, *, source_branch: str, target_branch: str | None = None) -> AgentResult:
        """Fusiona source_branch en target_branch (o en la rama actual si no se especifica)."""
        guard = self._guard_repo("merge_branch")
        if guard:
            return guard

        target = target_branch or self.git.current_branch()
        checkout = self._git("checkout", target)
        if not checkout.ok:
            return AgentResult(False, self.name, "merge_branch", f"No se pudo cambiar a '{target}': {checkout.stderr.strip()}")

        result = self._git("merge", source_branch, "--no-ff")
        if not result.ok:
            return AgentResult(
                False, self.name, "merge_branch",
                f"Conflicto al fusionar '{source_branch}' en '{target}': {result.stderr.strip()}",
            )

        return AgentResult(
            True, self.name, "merge_branch",
            f"'{source_branch}' fusionado en '{target}' (--no-ff).",
            data={"source": source_branch, "target": target},
        )

    def _load_dotenv(self) -> None:
        """Carga .env si existe para que graphify vea GEMINI_API_KEY."""
        env_path = self.ctx.root / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
            except ImportError:
                pass

    def _update_graphify(self) -> list[str]:
        """
        Si el proyecto tiene ``graphify-out/``, lanza ``graphify --update``
        para re-extraer solo los archivos que cambiaron en el commit.

        También verifica que GEMINI_API_KEY esté configurada para extracción
        semántica (no bloqueante), y sugiere instalar el hook post-commit si
        no está instalado.
        """
        self._load_dotenv()
        graphify_out = self.ctx.root / "graphify-out"
        if not graphify_out.exists():
            return []

        warnings: list[str] = []
        py_cmd = graphify_out / ".graphify_python"
        if not py_cmd.exists():
            warnings.append("graphify-out/ existe pero .graphify_python no — no se puede actualizar el grafo.")
            return warnings

        python_bin = py_cmd.read_text(encoding="utf-8").strip()
        if not python_bin:
            warnings.append(".graphify_python está vacío — no se puede actualizar el grafo.")
            return warnings

        _gemini = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not _gemini:
            warnings.append(
                "GEMINI_API_KEY no está configurada — graphify hará solo extracción "
                "estructural (AST). Las relaciones semánticas requieren API key gratuita en ai.google.dev."
            )

        hook_result = run_command(
            [python_bin, "-m", "graphify", "hook", "status"],
            cwd=self.ctx.root, timeout=30,
        )
        if hook_result.ok and "not installed" in hook_result.stdout.lower():
            warnings.append(
                "El hook post-commit de graphify no está instalado. "
                "Ejecuta 'graphify hook install' para que el grafo se actualice "
                "automáticamente en cada commit incluso sin este agente."
            )

        result = run_command(
            [python_bin, "-m", "graphify", str(self.ctx.root), "--update"],
            cwd=self.ctx.root, timeout=120,
        )
        if result.ok:
            print(f"    graphify: grafo actualizado ({result.stdout.strip()[:100]})")
        else:
            warnings.append(f"graphify --update falló: {result.stderr.strip()[:200]}")
        return warnings

    def _sync_obsidian(self) -> list[str]:
        """
        Si el proyecto tiene ``.obsidian/``, intenta añadir el estado actual
        del proyecto a la bóveda de Obsidian. Usa obsidian-cli si está
        disponible.
        """
        obsidian_dir = self.ctx.root / ".obsidian"
        if not obsidian_dir.exists():
            return []

        warnings: list[str] = []
        try:
            import subprocess
            result = subprocess.run(
                ["obsidian-cli", "index", str(self.ctx.root)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                print(f"    obsidian: bóveda indexada ({result.stdout.strip()[:100]})")
            else:
                warnings.append(f"obsidian-cli index falló: {result.stderr.strip()[:200]}")
        except FileNotFoundError:
            warnings.append(
                ".obsidian/ detectado pero obsidian-cli no está instalado. "
                "Instálalo con: npm install -g @obsidian/cli"
            )
        except Exception as exc:
            warnings.append(f"obsidian sync falló: {exc}")
        return warnings

    def _git(self, *args: str) -> ProcessResult:
        """Atajo interno para comandos git directos."""
        return run_command(["git", *args], cwd=self.ctx.root)
