"""
conftest.py — Fixtures de los tests de integración.

Suben los servicios de `tests/compose.integration.yml` una vez por sesión y
los bajan al terminar. El ejemplo conecta a Postgres vía psycopg; la URL se
expone en la fixture `database_url` para que el test actúe contra el servicio
real, sin mocks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

COMPOSE_FILE = Path(__file__).resolve().parents[1] / "compose.integration.yml"

DATABASE_URL = "postgresql://test:test@localhost:54329/test"


@pytest.fixture(scope="session", autouse=True)
def _integration_services():
    """Sube los servicios reales al empezar y los baja siempre al terminar."""
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--wait"],
        check=True,
        capture_output=True,
        text=True,
    )
    yield
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "down", "--remove-orphans"],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def database_url() -> str:
    """URL de conexión al servicio real de test."""
    return DATABASE_URL
