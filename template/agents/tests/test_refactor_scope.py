"""
test_refactor_scope.py — El agente `refactor` no puede salirse del proyecto.

Esto no es un test defensivo de manual: `uv` instala los paquetes en `.venv/`
enlazados por hardlink a su caché global. Un `refactor --within .` que entre en
`.venv/` no rompe solo ese proyecto — reescribe el fichero cacheado y deja rota
esa versión del paquete para TODOS los proyectos futuros de la máquina, sin
dejar rastro de la causa. Ya pasó con un numpy.
"""

from __future__ import annotations

import pytest

from agents.agents.refactor_agent import RefactorAgent

VULNERABLE = "def f(x=[]):\n    return x\n"


@pytest.fixture
def agent(context) -> RefactorAgent:
    return RefactorAgent(context=context)


def _plant(root, relpath: str) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(VULNERABLE, encoding="utf-8")


def test_no_entra_en_venv_aunque_se_le_pida_la_raiz(agent):
    _plant(agent.ctx.root, ".venv/lib/python3.13/site-packages/numpy/core.py")
    _plant(agent.ctx.root, "mi_paquete/propio.py")

    files = agent._py_files(".")
    nombres = {str(p.relative_to(agent.ctx.root)) for p in files}

    assert "mi_paquete/propio.py" in nombres
    assert not any(".venv" in n for n in nombres)


def test_apuntar_directamente_al_venv_no_devuelve_nada(agent):
    _plant(agent.ctx.root, ".venv/lib/python3.13/site-packages/numpy/core.py")
    assert agent._py_files(".venv") == []


@pytest.mark.parametrize(
    "forbidden",
    ["node_modules", "build", "dist", ".git", ".tox", ".mypy_cache", "site-packages"],
)
def test_directorios_prohibidos_se_ignoran(agent, forbidden):
    _plant(agent.ctx.root, f"{forbidden}/algo.py")
    assert agent._py_files(".") == [] or not any(
        forbidden in p.parts for p in agent._py_files(".")
    )


def test_no_sale_del_proyecto_con_rutas_relativas(agent):
    assert agent._py_files("..") == []
    assert agent._py_files("../..") == []


def test_el_fix_real_no_toca_el_venv(agent):
    """La prueba de verdad: ejecutar la acción y comprobar el fichero en disco."""
    _plant(agent.ctx.root, ".venv/lib/python3.13/site-packages/numpy/core.py")
    _plant(agent.ctx.root, "mi_paquete/propio.py")

    agent.fix_mutable_defaults(within=".")

    intacto = (agent.ctx.root / ".venv/lib/python3.13/site-packages/numpy/core.py").read_text()
    assert intacto == VULNERABLE, "el agente reescribió un fichero dentro de .venv"

    propio = (agent.ctx.root / "mi_paquete/propio.py").read_text()
    assert "is None" in propio, "el agente debería haber corregido el código del proyecto"
