"""
agents.tools.integration_tool — Envoltorio sobre `docker compose` para tests de integración.

Levanta y baja servicios reales (p. ej. Postgres) declarados en
`tests/compose.integration.yml` durante la ejecución de `tests/integration/`.
Es distinto de `DockerTool` (análisis estático de Dockerfiles/compose): aquí
se trata de gestionar el CICLO DE VIDA de servicios de test, sin mocks.

Sigue el principio de `tools/mutate.py`: lo que se levanta se baja SIEMPRE
(en `finally`), para no dejar contenedores huérfanos cuando un test falla.
"""

from __future__ import annotations

from pathlib import Path

from agents.tools.process_tool import ProcessResult, run_command
from agents.tools.registry import register_tool

COMPOSE_FILE_DEFAULT = "tests/compose.integration.yml"


@register_tool("integration")
class IntegrationTool:
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def _compose(self, compose_file: Path | None, *args: str, timeout: int) -> ProcessResult:
        """`docker compose -f <file> <args>` desde la raíz del proyecto."""
        file = compose_file or (self.project_root / COMPOSE_FILE_DEFAULT)
        cmd = ["docker", "compose", "-f", str(file), *args]
        return run_command(cmd, cwd=self.project_root, timeout=timeout)

    def up(self, compose_file: Path | None = None, timeout: int = 300) -> ProcessResult:
        """Levanta los servicios declarados y espera a que estén sanos (`--wait`)."""
        return self._compose(compose_file, "up", "-d", "--wait", timeout=timeout)

    def down(self, compose_file: Path | None = None, timeout: int = 120) -> ProcessResult:
        """Baja los servicios (con volúmenes anónimos; los datos de test no persisten)."""
        return self._compose(compose_file, "down", "--remove-orphans", timeout=timeout)

    def ps(self, compose_file: Path | None = None, timeout: int = 60) -> ProcessResult:
        """Estado de los servicios de integración (`docker compose ps`)."""
        return self._compose(compose_file, "ps", timeout=timeout)
