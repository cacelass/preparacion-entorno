"""
Tests de la puerta de permisos: `agents/permissions.py` + `BaseAgent.run`.

Lo que se comprueba aquí no es que la puerta "esté documentada" sino que
BLOQUEA de verdad: el contrato ya decía antes que refactor debía pedir
permiso y el código no lo pedía. Un test que solo lea el contrato repetiría
el mismo error.
"""

from __future__ import annotations

import pytest

from agents import permissions
from agents.agents.git_agent import GitAgent
from agents.agents.refactor_agent import RefactorAgent


@pytest.fixture(autouse=True)
def _sin_variables_de_entorno(monkeypatch):
    """La puerta se prueba activa: el entorno del que ejecuta no debe abrirla."""
    monkeypatch.delenv(permissions.VAR_ASSUME_YES, raising=False)
    monkeypatch.delenv(permissions.VAR_CONFIRM, raising=False)


# -- la política --------------------------------------------------------------
def test_una_accion_destructiva_pide_confirmacion():
    assert permissions.requiere_confirmacion("git", "commit_feature", {})


def test_una_accion_normal_no_pide_nada():
    assert not permissions.requiere_confirmacion("git", "analyze_diff", {})


def test_un_dry_run_nunca_pide_confirmacion():
    """Enseñar una propuesta no cambia nada: preguntar ahí es ruido."""
    assert not permissions.requiere_confirmacion("git", "commit_feature", {"dry_run": True})


def test_la_variable_de_entorno_desactiva_la_puerta(monkeypatch):
    monkeypatch.setenv(permissions.VAR_ASSUME_YES, "1")
    assert permissions.puerta_desactivada()
    assert not permissions.requiere_confirmacion("git", "commit_feature", {})


def test_dskit_confirm_a_cero_tambien_la_desactiva(monkeypatch):
    monkeypatch.setenv(permissions.VAR_CONFIRM, "0")
    assert permissions.puerta_desactivada()


def test_un_agente_sin_contrato_no_bloquea_nada():
    assert permissions.acciones_destructivas("agente-que-no-existe") == ()


# -- la puerta en BaseAgent.run -----------------------------------------------
def test_run_no_ejecuta_una_accion_destructiva_sin_confirmar(context):
    """El commit NO debe llegar a git: se corta antes de invocar el método."""
    resultado = GitAgent(context=context).run(
        "commit_feature", id="X-001", title="algo",
    )
    assert not resultado.success
    assert resultado.needs, "debe devolver la pregunta, no un error mudo"
    assert "NO se ha ejecutado" in resultado.message

    log = subprocess_log(context)
    assert "X-001" not in log, "no puede haberse creado ningún commit"


def test_run_deja_pasar_lo_no_destructivo(context):
    resultado = GitAgent(context=context).run("status")
    assert resultado.success


def test_run_deja_pasar_con_confirm(context, monkeypatch):
    """
    Con `confirm=True` la puerta se aparta y el argumento NO llega al método
    (que no lo acepta): si se filtrara, esto reventaría con un TypeError.
    """
    llamadas = {}

    def falso_commit_feature(**kwargs):
        llamadas.update(kwargs)
        from agents.core.base_agent import AgentResult
        return AgentResult(True, "git", "commit_feature", "simulado")

    agente = GitAgent(context=context)
    monkeypatch.setattr(agente, "actions", lambda: {"commit_feature": falso_commit_feature})

    resultado = agente.run("commit_feature", id="X-001", title="algo", confirm=True)
    assert resultado.success
    assert "confirm" not in llamadas


def test_refactor_tambien_esta_detras_de_la_puerta(context):
    resultado = RefactorAgent(context=context).run("fix_bare_excepts")
    assert not resultado.success
    assert "NO se ha ejecutado" in resultado.message


def test_el_bloqueo_queda_auditado(context):
    GitAgent(context=context).run("commit_feature", id="X-001", title="algo")
    registro = context.agent_workspace("audit") / "audit.jsonl"
    assert registro.exists()
    assert "falta confirmación" in registro.read_text(encoding="utf-8")


def subprocess_log(context) -> str:
    import subprocess

    return subprocess.run(
        ["git", "log", "--oneline"], cwd=context.root,
        capture_output=True, text=True, check=False,
    ).stdout
