"""
agents.tools.git_tool — Envoltorio sobre el binario `git`.

Se usa `git` por subprocess a propósito, no GitPython: es una dependencia
menos y cubre el 100% de lo que necesita `GitAgent` (diff, log, status).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from agents.tools.process_tool import ProcessResult, run_command
from agents.tools.registry import register_tool

_CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?P<scope>\([^)]+\))?(?P<breaking>!)?: (?P<subject>.+)$"
)

# Heurística de clasificación por ruta de archivo tocado -> tipo de Conventional Commit.
# No es infalible (un cambio en tests/ podría ser un "fix" real), pero da una
# sugerencia razonable de partida — el commit final lo revisa una persona.
_PATH_TYPE_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^tests?/"), "test"),
    (re.compile(r"^docs?/"), "docs"),
    (re.compile(r"README|CHANGELOG"), "docs"),
    (re.compile(r"^\.github/"), "ci"),
    (re.compile(r"Dockerfile|docker-compose"), "build"),
    (re.compile(r"pyproject\.toml|uv\.lock|requirements"), "build"),
]

#: Lock files: nunca entran en un commit atómico de área (cambian en bloque al
#: instalar deps y ahogan cualquier agrupación). Se reportan aparte.
LOCK_FILES: tuple[str, ...] = (
    "uv.lock", "package-lock.json", "poetry.lock", "Cargo.lock", "yarn.lock",
    "pnpm-lock.yaml", "composer.lock", "go.sum", "Pipfile.lock",
)

#: (prefijo en la ruta, área, tipo conventional, prioridad). Menor prioridad se
#: commitea antes: el código antes que sus tests, los tests antes que su doc.
_AREA_RULES: list[tuple[str, str, str, int]] = [
    (".github/", "ci", "ci", 4),
    ("Dockerfile", "build", "build", 3),
    ("docker-compose", "build", "build", 3),
    ("pyproject.toml", "build", "build", 3),
    ("requirements", "build", "build", 3),
    ("tests/", "test", "test", 1),
    ("docs/", "docs", "docs", 2),
    ("README", "docs", "docs", 2),
    ("CHANGELOG", "docs", "docs", 2),
]

#: Área por defecto (código de producto): primero y tipo `feat`.
_DEFAULT_AREA: tuple[str, str, int] = ("code", "feat", 0)


@register_tool("git")
@dataclass
class GitTool:
    repo_root: Path

    def _git(self, *args: str, check: bool = False) -> ProcessResult:
        return run_command(["git", *args], cwd=self.repo_root, check=check)

    def is_repo(self) -> bool:
        return self._git("rev-parse", "--is-inside-work-tree").ok

    def status_porcelain(self) -> list[tuple[str, str]]:
        """Devuelve [(código_estado, ruta), ...] tal y como `git status --porcelain`."""
        result = self._git("status", "--porcelain")
        entries = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            code, path = line[:2].strip(), line[3:].strip()
            entries.append((code, path))
        return entries

    def diff(self, *, staged: bool = False, name_only: bool = False) -> str:
        args = ["diff"]
        if staged:
            args.append("--staged")
        if name_only:
            args.append("--name-only")
        return self._git(*args).stdout

    def diff_stat(self, *, staged: bool = False) -> str:
        args = ["diff", "--stat"]
        if staged:
            args.insert(1, "--staged")
        return self._git(*args).stdout

    def changed_files(self, *, staged: bool = False) -> list[str]:
        raw = self.diff(staged=staged, name_only=True)
        files = [f for f in raw.splitlines() if f.strip()]
        if not files and staged:
            # nada en staging: mirar también working tree, es lo que suele querer el usuario
            files = [f for f in self.diff(staged=False, name_only=True).splitlines() if f.strip()]
        return files

    def log(self, *, max_count: int = 20, since_tag: str | None = None) -> list[dict[str, str]]:
        """Devuelve una lista de commits como dicts {hash, subject, author, date}."""
        rev_range = f"{since_tag}..HEAD" if since_tag else "HEAD"
        fmt = "%H%x1f%s%x1f%an%x1f%ad"
        result = self._git(
            "log", rev_range, f"--max-count={max_count}", f"--pretty=format:{fmt}", "--date=short"
        )
        commits = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split("\x1f")
            if len(parts) == 4:
                commits.append(
                    {"hash": parts[0], "subject": parts[1], "author": parts[2], "date": parts[3]}
                )
        return commits

    def last_tag(self) -> str | None:
        result = self._git("describe", "--tags", "--abbrev=0")
        return result.stdout.strip() or None if result.ok else None

    def current_branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    def commit(self, message: str) -> ProcessResult:
        return self._git("commit", "-m", message)

    def commit_paths(self, message: str, paths: list[str]) -> ProcessResult:
        """Commitea SOLO las rutas dadas, ignorando el resto del staging.

        Imprescindible para el commit atómico: si el usuario hizo `git add .`
        antes, un `git commit -m` plano se llevaría todo lo staged. Con
        `-- <paths>` el commit queda acotado a esos ficheros.
        """
        return self._git("commit", "-m", message, "--", *paths)

    def add(self, *paths: str) -> ProcessResult:
        return self._git("add", *paths)

    def create_tag(self, tag: str, *, message: str | None = None) -> ProcessResult:
        if message:
            return self._git("tag", "-a", tag, "-m", message)
        return self._git("tag", tag)

    def tag_exists(self, tag: str) -> bool:
        return self._git("rev-parse", tag).ok

    @staticmethod
    def parse_conventional_commit(message: str) -> dict[str, str] | None:
        """Extrae {type, scope, breaking, subject} de un mensaje Conventional Commits, o None."""
        match = _CONVENTIONAL_COMMIT_RE.match(message.strip().splitlines()[0])
        if not match:
            return None
        data = match.groupdict()
        data["scope"] = (data["scope"] or "").strip("()")
        data["breaking"] = bool(data["breaking"])
        return data

    @staticmethod
    def guess_commit_type(changed_files: list[str]) -> str:
        """Heurística: infiere el tipo Conventional Commit dominante a partir de las rutas tocadas."""
        if not changed_files:
            return "chore"
        counts: dict[str, int] = {}
        for path in changed_files:
            matched = False
            for pattern, commit_type in _PATH_TYPE_HINTS:
                if pattern.search(path):
                    counts[commit_type] = counts.get(commit_type, 0) + 1
                    matched = True
                    break
            if not matched:
                counts["feat"] = counts.get("feat", 0) + 1
        return max(counts, key=counts.get)

    # -- commit atómico -------------------------------------------------------

    @staticmethod
    def _classify_area(path: str) -> tuple[str, str, int]:
        """(área, tipo conventional, prioridad) de una ruta tocada."""
        for prefix, area, commit_type, priority in _AREA_RULES:
            if prefix in path:
                return area, commit_type, priority
        return _DEFAULT_AREA

    @staticmethod
    def _import_edges(files: list[str], repo_root: Path) -> list[tuple[str, str]]:
        """(importador, importado) entre ficheros .py del mismo cambio.

        Se resuelven los `import`/`from` a ficheros que también cambian: solo
        interesan las dependencias DENTRO del conjunto, para ordenar los
        commits. Un fichero sin `.py` no aporta aristas.
        """
        py = [f for f in files if f.endswith(".py")]
        if not py:
            return []
        modulos: dict[str, str] = {}
        for f in py:
            mod = f[:-3].replace("/", ".")
            modulos[mod] = f
            if f.endswith("/__init__.py"):
                modulos[mod[: -len(".__init__")]] = f

        def _resolver(mod: str) -> str | None:
            partes = mod.split(".")
            for i in range(len(partes), 0, -1):
                if ".".join(partes[:i]) in modulos:
                    return modulos[".".join(partes[:i])]
            return None

        edges: list[tuple[str, str]] = []
        for f in py:
            try:
                texto = (repo_root / f).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in re.finditer(r"^\s*(?:import|from)\s+([\w.]+)", texto, re.MULTILINE):
                destino = _resolver(match.group(1))
                if destino and destino != f:
                    edges.append((f, destino))
        return edges

    @staticmethod
    def _detect_cycle(groups: list[dict], edges: list[tuple[str, str]]) -> str | None:
        """Ciclo = un grupo que iría ANTES depende de otro que iría DESPUÉS.

        Con la partición por área y el orden por prioridad (código antes que
        tests, tests antes que docs...), un `import` del área temprana hacia la
        tardía rompería el orden de dependencias: se rechaza antes de escribir.
        """
        idx = {g["area"]: i for i, g in enumerate(groups)}
        for importer, imported in edges:
            area_imp = GitTool._classify_area(importer)[0]
            area_dep = GitTool._classify_area(imported)[0]
            if area_imp in idx and area_dep in idx and idx[area_imp] < idx[area_dep]:
                return (
                    f"el grupo '{area_imp}' depende de '{area_dep}', que iría "
                    f"después ({importer} importa {imported}). "
                    f"Commitea a mano o mueve la dependencia al área correcta."
                )
        return None

    @staticmethod
    def plan_atomic(changed_files: list[str], repo_root: Path) -> dict:
        """Divide los cambios en commits atómicos por área, ordenados por prioridad.

        Devuelve ``{"groups": [...], "excluded": [...], "cycle": None|str}``.
        Cada grupo es ``{"area", "type", "priority", "files"}``. Los lock files
        no entran en ningún grupo (``excluded``). Si ``cycle`` no es ``None``,
        el plan no es ejecutable y hay que commitear a mano.
        """
        excluded = [f for f in changed_files if Path(f).name in LOCK_FILES]
        files = [f for f in changed_files if Path(f).name not in LOCK_FILES]

        by_area: dict[str, dict] = {}
        for f in files:
            area, commit_type, priority = GitTool._classify_area(f)
            group = by_area.setdefault(
                area, {"area": area, "type": commit_type, "priority": priority, "files": []}
            )
            group["files"].append(f)

        groups = sorted(by_area.values(), key=lambda g: g["priority"])
        return {
            "groups": groups,
            "excluded": excluded,
            "cycle": GitTool._detect_cycle(groups, GitTool._import_edges(files, repo_root)),
        }
