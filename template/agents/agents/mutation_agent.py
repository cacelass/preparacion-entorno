"""
agents.agents.mutation_agent — Mutation testing y métrica CRAP.

Implementa la capa de validación del flujo spec-driven (Robert C. Martin /
BettaTech): la cobertura por líneas no prueba que los tests «muerdan» — un
test puede cubrir una línea y no detectar que su lógica está mal. La mutación
lo comprueba de forma determinista.

Dos herramientas, dos responsabilidades:

1. `run_mutation_testing` — ejecuta `tools/mutate.py` (mutador propio, sin
   dependencias, en la raíz del proyecto) que altera operadores de un módulo
   y ejecuta la suite por cada mutante. Si un mutante «sobrevive» a los
   tests, hay un hueco en la suite.
2. `crap_report` — calcula la métrica CRAP (`complexity^2 * (1 - coverage/100)^3
   + complexity`) por función usando radon (ya es dev-dep del proyecto) y la
   cobertura que ya genera `pytest-cov`. CRAP > 30 es la señal clásica de
   «código complejo y mal probado».

Este agente NO decide nada: ejecuta las herramientas, resume los números y
deja el juicio (qué sobrevivientes son aceptables, si el umbral de CRAP es el
correcto) al reviewer/humano.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.exceptions import ToolExecutionError
from agents.tools.process_tool import run_command
from agents.tools.pytest_tool import PytestTool

CRAP_THRESHOLD = 30.0


def crap_value(cc: float, coverage: float) -> float:
    """CRAP = cc^2 * (1 - coverage/100)^3 + cc (fórmula de cambio de riesgo)."""
    return cc**2 * (1 - coverage / 100.0) ** 3 + cc


@register_agent
class MutationAgent(BaseAgent):
    name = "mutation"
    description = (
        "Mutation testing (¿muerden los tests?) y métrica CRAP: ejecuta tools/mutate.py "
        "y calcula la complejidad-cubierta por función. No arregla código."
    )
    capabilities = [
        "mutation",
        "mutación",
        "mutante",
        "mutantes",
        "crap",
        "muerden",
        "survivor",
        "survivors",
        "superviviente",
        "supervivientes",
        "mutation testing",
        "mutation score",
        "score de mutación",
    ]

    def actions(self) -> dict:
        return {
            "run_mutation_testing": self.run_mutation_testing,
            "crap_report": self.crap_report,
        }

    def action_aliases(self) -> dict:
        return {
            "run_mutation_testing": [
                "mutation",
                "muta",
                "muerden",
                "survivor",
                "mutante",
            ],
            "crap_report": ["crap", "complejidad", "riesgo de cambio"],
        }

    # -- helpers -------------------------------------------------------------
    @property
    def _mutate_script(self) -> Path:
        return self.ctx.root / "tools" / "mutate.py"

    def _resolve_target(self, target: str, action: str) -> AgentResult | Path:
        """El módulo objetivo, validado dentro de la raíz del proyecto."""
        if not target:
            return AgentResult(
                False,
                self.name,
                action,
                "Falta el módulo objetivo.",
                needs=[
                    "¿Qué módulo quieres analizar? (--target, ruta relativa al proyecto)"
                ],
            )
        candidate = (self.ctx.root / target).resolve()
        root = self.ctx.root.resolve()
        if not candidate.exists() or not candidate.is_file():
            return AgentResult(
                False, self.name, action, f"No existe el archivo '{target}'."
            )
        if not candidate.is_relative_to(root):
            return AgentResult(
                False, self.name, action, f"'{target}' está fuera del proyecto."
            )
        return candidate

    # -- mutation testing -----------------------------------------------------
    def run_mutation_testing(
        self, *, target: str, tests: str = "tests/", timeout: int = 60
    ) -> AgentResult:
        """
        Ejecuta tools/mutate.py sobre un módulo y resume el resultado.

        `target` es una ruta relativa al módulo de producción (p. ej.
        `{{ project_slug }}/features/build_features.py`). Cada mutante se
        ejecuta con la suite en `tests` y un timeout por mutante.
        """
        resolved = self._resolve_target(target, "run_mutation_testing")
        if isinstance(resolved, AgentResult):
            return resolved

        if not self._mutate_script.exists():
            return AgentResult(
                False,
                self.name,
                "run_mutation_testing",
                "No existe tools/mutate.py — el proyecto se creó sin el extra SDD "
                "(use_sdd). Regenera el proyecto con esa opción o ejecuta el mutador "
                "por tu cuenta.",
                data={"expected": "tools/mutate.py"},
            )

        try:
            proc = run_command(
                [
                    "uv",
                    "run",
                    "python",
                    str(self._mutate_script),
                    str(resolved.relative_to(self.ctx.root)),
                    "--tests",
                    tests,
                    "--timeout",
                    str(timeout),
                ],
                cwd=self.ctx.root,
                timeout=timeout * 60,
            )
        except ToolExecutionError as exc:
            return AgentResult(False, self.name, "run_mutation_testing", str(exc))

        stdout = proc.stdout
        report = self._parse_report(stdout)
        if report is None:
            return AgentResult(
                False,
                self.name,
                "run_mutation_testing",
                f"El mutador no devolvió un informe parseable (exit {proc.returncode}).",
                data={"stdout": stdout[-2000:], "stderr": proc.stderr[-2000:]},
            )

        survivors = report["survived"]
        warnings = []
        if survivors:
            warnings.append(
                f"{survivors} mutante(s) sobrevivieron: hay código de producción que los "
                f"tests no protegen. Revisa los sitios marcados y decide si merecen test "
                f"(o si están fuera del alcance de la feature)."
            )
        return AgentResult(
            survivors == 0,
            self.name,
            "run_mutation_testing",
            f"Score de mutación {report['score']}% sobre {target} "
            f"({report['killed']} killed, {report['survived']} survived, "
            f"{report['timeout']} timeout).",
            data=report,
            warnings=warnings,
        )

    @staticmethod
    def _parse_report(stdout: str) -> dict | None:
        """Extrae el informe del mutador (killed/survived/timeout/score/detail)."""
        match = re.search(
            r"(\d+) sitio\(s\) · killed (\d+) · survived (\d+) · timeout (\d+)",
            stdout,
        )
        if not match:
            return None
        score_match = re.search(r"Score de mutación: ([\d.]+)%", stdout)
        detail = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line or "✔" not in line and "✘" not in line:
                continue
            status = "killed" if "✔" in line else "survived"
            site = line.split()[-1]
            detail.append({"site": site, "status": status})
        return {
            "total": int(match.group(1)),
            "killed": int(match.group(2)),
            "survived": int(match.group(3)),
            "timeout": int(match.group(4)),
            "score": float(score_match.group(1)) if score_match else 0.0,
            "detail": detail,
        }

    # -- CRAP -----------------------------------------------------------------
    def crap_report(self, *, target: str) -> AgentResult:
        """
        Métrica CRAP por función: complejidad ciclomática (radon) y cobertura
        (pytest-cov). CRAP = cc^2 * (1 - coverage/100)^3 + cc. CRAP > 30 es
        la señal clásica de riesgo de cambio.
        """
        resolved = self._resolve_target(target, "crap_report")
        if isinstance(resolved, AgentResult):
            return resolved

        complexity = self._radon_complexity(resolved)
        if complexity is None:
            return AgentResult(
                False,
                self.name,
                "crap_report",
                "radon no está disponible o no pudo analizar el módulo.",
                data={"expected": "radon (dev-dep del proyecto)"},
            )
        if not complexity:
            return AgentResult(
                True,
                self.name,
                "crap_report",
                f"'{target}' no tiene funciones que analizar.",
                data=[],
            )

        coverage = self._module_coverage(resolved)
        if coverage is None:
            warnings = [
                "No se pudo medir la cobertura — CRAP se calcula con 0%, "
                "lo que infla el riesgo. Ejecuta primero la suite."
            ]
            coverage = 0.0
        else:
            warnings = []

        rows = []
        worst = []
        for fn in complexity:
            cc = fn["complexity"]
            crap = round(crap_value(cc, coverage), 2)
            rows.append({**fn, "coverage": coverage, "crap": crap})
            if crap > CRAP_THRESHOLD:
                worst.append(
                    {"name": fn["name"], "line": fn["lineno"], "crap": crap, "cc": cc}
                )

        if worst:
            warnings.append(
                f"{len(worst)} función(es) con CRAP > {CRAP_THRESHOLD:.0f}: "
                + ", ".join(f"{w['name']} ({w['crap']})" for w in worst)
                + " — código complejo y poco probado. Testear más o reducir complejidad."
            )

        return AgentResult(
            not worst,
            self.name,
            "crap_report",
            f"CRAP sobre '{target}': {len(rows)} función(es), "
            f"{len(worst)} por encima del umbral {CRAP_THRESHOLD:.0f}, "
            f"cobertura {coverage:.1f}%.",
            data={
                "coverage": coverage,
                "threshold": CRAP_THRESHOLD,
                "functions": rows,
                "worst": worst,
            },
            warnings=warnings,
        )

    def _radon_complexity(self, path: Path) -> list[dict] | None:
        """Complejidad ciclomática por función vía radon (JSON)."""
        try:
            proc = run_command(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "radon",
                    "cc",
                    str(path.relative_to(self.ctx.root)),
                    "-j",
                    "-s",
                ],
                cwd=self.ctx.root,
                timeout=120,
            )
        except ToolExecutionError:
            return None
        if not proc.ok:
            return None
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return None
        for file_functions in data.values():
            return [f for f in file_functions if f.get("type") == "function"]
        return []

    def _module_coverage(self, path: Path) -> float | None:
        """Cobertura del módulo que contiene `path` (proxy del fichero objetivo)."""
        module = self.ctx.config.project_slug
        if not module:
            return None
        if not (self.ctx.root / module).exists():
            return None
        cov_path = self.ctx.agent_workspace("mutation") / "coverage.json"
        cov_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            PytestTool.run_with_coverage(
                self.ctx.root, module=module, coverage_json_path=cov_path, timeout=600
            )
        except ToolExecutionError:
            return None
        if not cov_path.exists():
            return None
        try:
            report = PytestTool.parse_coverage_json(cov_path)
        except Exception:  # noqa: BLE001 — un JSON raro no debe tumbar el informe
            return None
        return report.get("total_percent_covered")
