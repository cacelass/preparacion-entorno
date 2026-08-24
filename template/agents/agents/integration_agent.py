"""
agents.agents.integration_agent — Ejecuta tests de integración contra servicios reales.

Levanta los servicios declarados en `tests/compose.integration.yml` (p. ej.
Postgres vía Docker) durante los tests y los baja SIEMPRE al terminar. Es la
alternativa sin mocks que propone la filosofía del arnés: un mock da
"seguridad falsa" — aquí el test habla con la infraestructura real.

Dos responsabilidades claras:

1. `run_integration_tests` — sube los servicios con `docker compose up -d
   --wait`, corre `pytest tests/integration/` y baja los servicios en
   `finally` (aunque los tests fallen). Resumen desde el reporte JUnit.
2. `status` — qué servicios de integración hay declarados y si están
   levantados (solo lectura, para diagnosticar antes de correr).

Este agente NO decide qué probar ni escribe tests nuevos: ejecuta y resume.
"""

from __future__ import annotations

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.exceptions import MissingDependencyError, ToolExecutionError
from agents.tools.integration_tool import COMPOSE_FILE_DEFAULT, IntegrationTool
from agents.tools.pytest_tool import PytestTool


@register_agent
class IntegrationAgent(BaseAgent):
    name = "integration"
    description = (
        "Ejecuta tests de integración contra servicios reales (Postgres vía Docker), "
        "sin mocks: levanta tests/compose.integration.yml, corre tests/integration/ y "
        "baja los servicios al terminar. No escribe tests nuevos."
    )
    capabilities = [
        "integration",
        "integración",
        "tests de integración",
        "servicio real",
        "sin mocks",
        "sin mock",
        "postgres",
        "base de datos real",
        "docker compose test",
        "levanta servicios",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.integration = IntegrationTool(project_root=self.ctx.root)

    def actions(self) -> dict:
        return {
            "run_integration_tests": self.run_integration_tests,
            "status": self.status,
        }

    def action_aliases(self) -> dict:
        return {
            "run_integration_tests": [
                "integración",
                "tests de integración",
                "servicio real",
                "sin mocks",
                "levantar servicios",
                "postgres",
            ],
            "status": ["servicios", "estado de integración"],
        }

    # -- helpers -------------------------------------------------------------
    def _compose_file_exists(self) -> bool:
        return (self.ctx.root / COMPOSE_FILE_DEFAULT).exists()

    def _compose_missing(self, action: str) -> AgentResult | None:
        """Devuelve un AgentResult de error si falta la config o Docker."""
        if not self._compose_file_exists():
            return AgentResult(
                False,
                self.name,
                action,
                f"No existe {COMPOSE_FILE_DEFAULT} — el proyecto se creó sin el extra "
                "use_integration. Regenera el proyecto con esa opción o declara "
                "los servicios de test por tu cuenta.",
                data={"expected": COMPOSE_FILE_DEFAULT},
            )
        return None

    # -- integración -----------------------------------------------------------
    def run_integration_tests(self) -> AgentResult:
        """
        Levanta los servicios declarados, corre `pytest tests/integration/` y
        los baja SIEMPRE al terminar (finally). Resumen desde el reporte JUnit.
        """
        # Nota: se compara con None (no `if missing:`) porque `AgentResult`
        # define __bool__ == success — un error de configuración es falsy y
        # seguiríamos adelante intentando levantar servicios.
        missing = self._compose_missing("run_integration_tests")
        if missing is not None:
            return missing

        workdir = self.ctx.agent_workspace("integration")
        junit_path = workdir / "junit.xml"

        try:
            up = self.integration.up()
        except MissingDependencyError as exc:
            return AgentResult(False, self.name, "run_integration_tests", str(exc))
        except ToolExecutionError as exc:
            return AgentResult(False, self.name, "run_integration_tests", str(exc))

        if not up.ok:
            return AgentResult(
                False, self.name, "run_integration_tests",
                "No se pudieron levantar los servicios de integración.",
                data={"stdout": up.stdout[-2000:], "stderr": up.stderr[-2000:]},
            )

        try:
            try:
                process = PytestTool.run(
                    self.ctx.root,
                    path="tests/integration/",
                    junit_xml_path=junit_path,
                    # La suite normal excluye el marker `integration` en el
                    # addopts de pyproject; aquí lo anulamos para correrlos.
                    overrides=["-o", "addopts="],
                )
            except (MissingDependencyError, ToolExecutionError) as exc:
                return AgentResult(False, self.name, "run_integration_tests", str(exc))

            if not junit_path.exists():
                return AgentResult(
                    False, self.name, "run_integration_tests",
                    "pytest no llegó a generar el reporte JUnit — probablemente un "
                    "error de colección.",
                    data={"stdout": process.stdout[-2000:], "stderr": process.stderr[-2000:]},
                )

            summary = PytestTool.parse_junit_xml(junit_path)
            warnings = [f"{f.classname}::{f.name}: {f.message}" for f in summary.failed_tests]
            return AgentResult(
                summary.failures == 0 and summary.errors == 0, self.name,
                "run_integration_tests",
                f"{summary.passed}/{summary.total} pasaron, {summary.failures} fallo(s), "
                f"{summary.errors} error(es), {summary.skipped} omitido(s) en "
                f"{summary.duration_seconds:.1f}s contra los servicios reales.",
                data=summary.__dict__, warnings=warnings,
            )
        finally:
            # Los servicios se bajan SIEMPRE, aunque los tests fallen — el patrón
            # de tools/mutate.py: lo que se levanta se limpia en finally.
            try:
                self.integration.down()
            except (MissingDependencyError, ToolExecutionError):
                pass

    def status(self) -> AgentResult:
        """Qué servicios de integración hay declarados y si están levantados."""
        missing = self._compose_missing("status")
        if missing is not None:
            return missing

        try:
            ps = self.integration.ps()
        except MissingDependencyError as exc:
            return AgentResult(False, self.name, "status", str(exc))
        except ToolExecutionError as exc:
            return AgentResult(False, self.name, "status", str(exc))

        return AgentResult(
            True, self.name, "status",
            "Servicios de integración declarados en tests/compose.integration.yml.",
            data={"ps": ps.stdout},
            warnings=[] if ps.ok else [ps.stderr or "No se pudo leer el estado de los servicios."],
        )
