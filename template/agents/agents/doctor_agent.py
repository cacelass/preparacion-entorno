"""
agents.agents.doctor_agent — Diagnóstico integral del proyecto.

Revisa el estado del proyecto en todas las dimensiones: entorno, git,
estructura de datos, código, tests, dependencias y configuración.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.process_tool import run_command


def _satisfies_requires_python(current: tuple[int, ...], requires: str) -> bool:
    """
    ¿Cumple `current` el `requires-python` del pyproject?

    Antes se comprobaba con `current in requires`, es decir, subcadena: con
    `requires-python = ">=3.12"`, un Python 3.13 daba FALLO (la cadena "3.13"
    no aparece en ">=3.12") y un Python 3.1 daba OK (sí aparece). El
    diagnóstico decía justo lo contrario de la realidad.

    Se comparan tuplas de enteros y se soportan las cláusulas que aparecen en
    la práctica, separadas por comas. Una cláusula que no se entienda se da
    por buena: es un diagnóstico, no un resolutor de dependencias — mejor no
    avisar que avisar en falso.
    """
    ops = (">=", "<=", "==", "!=", "~=", ">", "<")
    for raw in requires.split(","):
        clause = raw.strip()
        if not clause:
            continue
        op = next((o for o in ops if clause.startswith(o)), None)
        if op is None:
            continue
        try:
            target = tuple(int(part) for part in clause[len(op):].strip().split(".") if part.isdigit())
        except ValueError:
            continue
        if not target:
            continue
        head = current[: len(target)]
        if op == ">=" and not head >= target:
            return False
        if op == ">" and not head > target:
            return False
        if op == "<=" and not head <= target:
            return False
        if op == "<" and not head < target:
            return False
        if op == "==" and head != target:
            return False
        if op == "!=" and head == target:
            return False
        if op == "~=" and (head < target or current[: len(target) - 1] != target[:-1]):
            return False
    return True


@register_agent
class DoctorAgent(BaseAgent):
    name = "doctor"
    description = "Diagnóstico integral: entorno, git, datos, código, tests, dependencias, config."
    capabilities = ["diagnóstico", "health", "healthcheck", "check", "doctor", "estado", "status"]

    def actions(self) -> dict:
        return {
            "checkup": self.checkup,
            "disk_usage": self.disk_usage,
            "summary": self.summary,
        }

    def checkup(self) -> AgentResult:
        """Ejecuta todas las verificaciones y devuelve un dict con el estado."""
        checks = {
            "python": self._check_python(),
            "git": self._check_git(),
            "project_config": self._check_project_config(),
            "structure": self._check_structure(),
            "tests": self._check_tests(),
            "lock": self._check_lock_sync(),
            "data": self._check_data(),
            "harness": self._check_harness(),
        }
        ok = sum(1 for c in checks.values() if c.get("ok"))
        total = len(checks)
        all_ok = ok == total
        return AgentResult(
            all_ok, self.name, "checkup",
            f"{ok}/{total} verificaciones superadas",
            data=checks,
            warnings=[
                f"{k}: {v['message']}"
                for k, v in checks.items()
                if not v.get("ok")
            ],
        )

    def disk_usage(self) -> AgentResult:
        """Muestra el tamaño de los directorios principales del proyecto."""
        dirs = {
            "data": self.ctx.data_dir,
            "models": self.ctx.models_dir,
            "reports": self.ctx.reports_dir,
            "notebooks": self.ctx.notebooks_dir,
            "agents/workspace": self.ctx.workspace_dir,
        }
        sizes = {}
        for label, path in dirs.items():
            if path.exists():
                total_bytes = sum(
                    f.stat().st_size for f in path.rglob("*") if f.is_file()
                )
                sizes[label] = self._human_size(total_bytes)
            else:
                sizes[label] = "no existe"
        return AgentResult(True, self.name, "disk_usage", "Uso de disco por directorio.", data=sizes)

    def summary(self) -> AgentResult:
        """Resumen ejecutivo del proyecto."""
        pyproject = self._load_pyproject()
        git_result = run_command(
            ["git", "log", "--oneline", "-5"], cwd=self.ctx.root
        )
        git_log = git_result.stdout.strip() if git_result.ok else "no disponible"
        project_name = (
            pyproject.get("project", {}).get("name", "desconocido")
            if pyproject else "desconocido"
        )
        python_v = f"{sys.version_info.major}.{sys.version_info.minor}"
        test_dirs = list(self.ctx.tests_dir.glob("test_*.py")) if self.ctx.tests_dir.exists() else []
        data_files = list(self.ctx.raw_data_dir.glob("*")) if self.ctx.raw_data_dir.exists() else []

        data = {
            "project": project_name,
            "ml_type": self.ctx.config.ml_type,
            "python": python_v,
            "data_files": len(data_files),
            "test_files": len(test_dirs),
            "git_log": git_log,
        }
        return AgentResult(True, self.name, "summary", f"{project_name} ({self.ctx.config.ml_type})", data=data)

    # ---- helpers internos ----

    def _load_pyproject(self) -> dict | None:
        import tomllib
        try:
            with open(self.ctx.pyproject_file, "rb") as f:
                return tomllib.load(f)
        except Exception:
            return None

    def _check_python(self) -> dict:
        pyproject = self._load_pyproject()
        if pyproject is None:
            return {"ok": False, "message": "pyproject.toml no encontrado"}
        requires = pyproject.get("project", {}).get("requires-python", "")
        current = sys.version_info[:2]
        current_str = f"{current[0]}.{current[1]}"
        if not requires:
            return {"ok": True, "message": f"Python {current_str} (sin restricción)"}
        ok = _satisfies_requires_python(current, requires)
        estado = "cumple" if ok else "NO cumple"
        return {"ok": ok, "message": f"Python {current_str} {estado} '{requires}'"}

    def _check_git(self) -> dict:
        try:
            result = run_command(["git", "status", "--porcelain"], cwd=self.ctx.root)
        except Exception as e:
            return {"ok": False, "message": f"git no disponible: {e}"}
        if not result.ok:
            return {"ok": False, "message": "no es un repositorio git"}
        changes = result.stdout.strip()
        n_changes = len([line for line in changes.split("\n") if line.strip()]) if changes else 0
        branch_result = run_command(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=self.ctx.root
        )
        branch = branch_result.stdout.strip() if branch_result.ok else "?"
        if n_changes == 0:
            return {"ok": True, "message": f"Working directory clean ({branch})"}
        return {"ok": True, "message": f"{n_changes} archivo(s) sin commit ({branch})"}

    def _check_project_config(self) -> dict:
        config = self.ctx.config
        if not config.project_slug:
            return {"ok": False, "message": "configuración del proyecto no encontrada"}
        return {
            "ok": True,
            "message": f"{config.project_slug} (ml_type={config.ml_type}, mlflow={config.use_mlflow})",
        }

    def _check_structure(self) -> dict:
        required = [
            self.ctx.package_dir / "__init__.py",
            self.ctx.pyproject_file,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            return {"ok": False, "message": f"faltan: {', '.join(missing)}"}
        return {"ok": True, "message": "estructura del proyecto correcta"}

    def _check_tests(self) -> dict:
        tests_dir = self.ctx.tests_dir
        if not tests_dir.exists():
            return {"ok": True, "message": "directorio tests/ no existe (puede no ser necesario)"}
        test_files = list(tests_dir.glob("test_*.py"))
        n = len(test_files)
        if n == 0:
            return {"ok": False, "message": "tests/ existe pero no contiene tests"}
        return {"ok": True, "message": f"{n} archivo(s) de test"}

    def _check_harness(self) -> dict:
        """
        ¿Están las piezas del arnés? No ejecuta la puerta (eso es
        `harness gate`): aquí solo se comprueba que el andamiaje existe, para
        que un diagnóstico no tarde lo que tarda la suite de tests.
        """
        missing = [
            rel
            for rel in ("init.sh", "featureslist.json", "progress/current.md", "AGENTS.md")
            if not (self.ctx.root / rel).exists()
        ]
        if missing:
            return {"ok": False, "message": f"faltan piezas del arnés: {', '.join(missing)}"}

        try:
            import json

            doc = json.loads((self.ctx.root / "featureslist.json").read_text(encoding="utf-8"))
            features = doc.get("features", [])
        except (OSError, json.JSONDecodeError) as exc:
            return {"ok": False, "message": f"featureslist.json ilegible: {exc}"}

        pending = sum(1 for f in features if f.get("status") == "pending")
        running = [f.get("id") for f in features if f.get("status") == "in_progress"]
        if len(running) > 1:
            return {"ok": False, "message": f"{len(running)} features in_progress a la vez: {', '.join(running)}"}
        activa = running[0] if running else "ninguna"
        return {"ok": True, "message": f"arnés completo · en curso: {activa} · {pending} pendiente(s)"}

    def _check_lock_sync(self) -> dict:
        result = run_command(["uv", "lock", "--check"], cwd=self.ctx.root)
        if result.ok:
            return {"ok": True, "message": "uv.lock sincronizado"}
        return {"ok": False, "message": "uv.lock desincronizado (corre 'uv lock')"}

    def _check_data(self) -> dict:
        raw = self.ctx.raw_data_dir
        if not raw.exists():
            return {"ok": True, "message": "data/raw/ no existe (proyecto nuevo)"}
        files = list(raw.glob("*"))
        return {"ok": True, "message": f"{len(files)} archivo(s) en data/raw/"}

    @staticmethod
    def _human_size(bytes_: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if bytes_ < 1024:
                return f"{bytes_:.1f} {unit}"
            bytes_ /= 1024
        return f"{bytes_:.1f} TB"
