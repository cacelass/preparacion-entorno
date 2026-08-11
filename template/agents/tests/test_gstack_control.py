"""
Tests del control de GStack: auto-commit autorizado y volcado a Mermaid.

El auto-commit de GStack no pasa por `BaseAgent.run` —usa `GitTool`
directamente—, así que la puerta de permisos no lo cubriría sola. Estos tests
existen porque ese camino ya commiteó una vez en un repositorio real.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from agents import permissions
from agents.core.base_agent import AgentResult
from agents.gstack.stack import GStack, StackResult


@pytest.fixture(autouse=True)
def _sin_variables_de_entorno(monkeypatch):
    monkeypatch.delenv(permissions.VAR_ASSUME_YES, raising=False)
    monkeypatch.delenv(permissions.VAR_CONFIRM, raising=False)


def _commits(root) -> list[str]:
    salida = subprocess.run(
        ["git", "log", "--oneline"], cwd=root, capture_output=True, text=True, check=False,
    ).stdout
    return [linea for linea in salida.splitlines() if linea.strip()]


def _ensucia(root, texto: str) -> None:
    """
    Modifica un fichero YA RASTREADO por git.

    Crear uno nuevo no vale: el auto-commit de GStack usa `changed_files()`,
    que no ve los ficheros sin rastrear — así que un `nuevo.txt` no habría
    disparado ningún commit y el test habría pasado en verde sin probar nada.
    """
    (root / "README.md").write_text(f"# Proyecto de prueba\n\n{texto}\n")


def _eventos(ctx) -> list[dict]:
    ruta = ctx.agent_workspace("gstack") / "events.jsonl"
    if not ruta.exists():
        return []
    return [json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines() if linea]


# -- autorización del auto-commit ---------------------------------------------
def test_sin_confirmar_no_commitea_pero_lo_anota(context):
    _ensucia(context.root, "cambio sin comitear")
    antes = _commits(context.root)

    stack = GStack(auto_commit=True, context=context)
    stack.push("env", "info")
    resultado = stack.run()

    assert resultado.success, "el paso se ejecuta igual: lo que se omite es el commit"
    assert _commits(context.root) == antes, "no puede haber tocado el historial"
    assert any(e.get("event") == "auto_commit_bloqueado" for e in _eventos(context))


def test_con_confirm_si_commitea(context):
    _ensucia(context.root, "cambio a comitear")
    antes = _commits(context.root)

    stack = GStack(auto_commit=True, context=context, confirm=True)
    stack.push("env", "info")
    stack.run()

    assert len(_commits(context.root)) == len(antes) + 1


def test_la_variable_de_entorno_tambien_autoriza(context, monkeypatch):
    monkeypatch.setenv(permissions.VAR_ASSUME_YES, "1")
    _ensucia(context.root, "cambio a comitear")
    antes = _commits(context.root)

    stack = GStack(auto_commit=True, context=context)
    stack.push("env", "info")
    stack.run()

    assert len(_commits(context.root)) == len(antes) + 1


# -- mermaid -------------------------------------------------------------------
def test_to_mermaid_dibuja_los_pasos_en_orden(context):
    stack = GStack(context=context)
    stack.push("env", "info")
    stack.push("test", "run_tests", run_if=lambda r, m: True)

    diagrama = stack.to_mermaid()
    assert diagrama.startswith("flowchart TD")
    assert 'p0["env.info"]' in diagrama
    assert "condicional" in diagrama, "un paso con run_if debe verse como tal"
    assert "inicio --> p0" in diagrama
    assert "p1 --> fin" in diagrama


def test_to_mermaid_colorea_los_resultados(context):
    stack = GStack(context=context)
    stack.push("env", "info")
    stack.push("env", "info")

    resultados = [
        AgentResult(True, "env", "info", "ok"),
        AgentResult(True, "env", "__skipped__", "omitido"),
    ]
    diagrama = stack.to_mermaid(resultados)
    assert "class p0 ok" in diagrama
    assert "class p1 omitido" in diagrama
    assert "classDef omitido" in diagrama


def test_to_mermaid_de_una_stack_vacia_no_revienta(context):
    diagrama = GStack(context=context).to_mermaid()
    assert "inicio --> fin" in diagrama


# -- lock de pipeline ----------------------------------------------------------
def test_lock_tomado_bloquea_el_segundo_pipeline_sin_ejecutar(context):
    """
    Dos GStack a la vez pueden pisarse el árbol de trabajo. El segundo debe
    devolver fallo ANTES de ejecutar ningún paso — el lock se comprueba antes
    de tocar el orquestador.
    """
    a = GStack(context=context)
    a.push("env", "info")
    holder = a._try_lock()  # noqa: SLF001 — el test sostiene el lock a propósito
    assert holder is not None
    try:
        b = GStack(context=context)
        b.push("env", "info")
        b.push("env", "info")
        resultado = b.run()
        assert not resultado.success
        assert "lock" in resultado.message.lower()
        assert resultado.results == [], "no debe haber ejecutado ningún paso"
    finally:
        a._release_lock(holder)


def test_el_lock_se_libera_y_se_puede_volver_a_tomar(context):
    a = GStack(context=context)
    a.push("env", "info")
    holder = a._try_lock()  # noqa: SLF001
    assert holder is not None
    a._release_lock(holder)
    holder2 = a._try_lock()  # noqa: SLF001 — ya libre, debe poder tomarlo otro
    assert holder2 is not None
    a._release_lock(holder2)


def test_lock_false_omite_el_bloqueo(context, monkeypatch):
    """`lock=False` es la salida explícita: el pipeline confía en que no hay concurrencia."""
    stack = GStack(context=context, lock=False)
    stack.push("env", "info")

    def _no_bloquear(self):  # si se llama, el flag no funcionó
        raise AssertionError("con lock=False no se debe intentar tomar el lock")

    monkeypatch.setattr(GStack, "_try_lock", _no_bloquear)
    monkeypatch.setattr(
        GStack, "_ejecutar",
        lambda self: StackResult(success=True, steps=self._steps, results=[]),
    )
    resultado = stack.run()
    assert resultado.success
