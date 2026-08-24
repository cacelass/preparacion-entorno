"""
tests/integration/test_db.py — Ejemplo de test de integración contra un servicio real.

Escribe y lee una fila en el Postgres real levantado por
`tests/compose.integration.yml` (sin mocks). Es la prueba de que el contrato
funciona: si se revierte el cambio o el servicio no está, el test falla.

Marcado `@pytest.mark.integration`: la suite normal (`make test`/init.sh) lo
excluye vía `-m 'not integration'` en pyproject; solo `make integration` lo corre.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_postgres_escribe_y_lee(database_url: str) -> None:
    # Import local: la suite normal (`-m 'not integration'`) importa este módulo
    # en la recolección aunque no lo ejecute, así que psycopg no debe estar en
    # el import de módulo (solo se necesita al correr `make integration`).
    import psycopg

    with psycopg.connect(database_url) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS ping (v int)")
        conn.execute("DELETE FROM ping")
        conn.execute("INSERT INTO ping (v) VALUES (%s)", (42,))
        row = conn.execute("SELECT v FROM ping").fetchone()

    assert row == (42,), "el valor escrito no se leyó del servicio real"
