"""
Tests de la puerta de permisos: `agents/permissions.py` + `BaseAgent.run`.

Lo que se comprueba aquí no es que la puerta "esté documentada" sino que
BLOQUEA de verdad: el contrato ya decía antes que refactor debía pedir
permiso y el código no lo pedía. Un test que solo lea el contrato repetiría
el mismo error.
"""

from __future__ import annotations

import pytest

from agents import audit, permissions
from agents.agents.git_agent import GitAgent
from agents.agents.refactor_agent import RefactorAgent
from agents.contracts import CONTRACTS
from agents.core.base_agent import AgentResult


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


# -- la escalera: críticas exigen el nombre exacto -----------------------------
def test_una_accion_critica_exige_nombre():
    assert permissions.exige_nombre("git", "tag_release", {})


def test_una_destructiva_normal_no_exige_nombre_sin_fatiga():
    """Sin racha de fatiga, una destructiva basta con --yes."""
    assert not permissions.exige_nombre("git", "commit_feature", {}, ctx=None)


def test_el_nombre_debe_coincidir_con_el_objetivo():
    assert permissions.nombre_valido("git", "tag_release", "1.0.0", {"version": "1.0.0"})
    assert not permissions.nombre_valido("git", "tag_release", "2.0.0", {"version": "1.0.0"})
    assert not permissions.nombre_valido("git", "tag_release", "", {"version": "1.0.0"})


def test_toda_critica_tiene_objetivo_declarado():
    """Una crítica sin objetivo nombrable se cerraría en rojo a propósito; no
    debe existir ninguna así en los contratos."""
    for agente, contrato in CONTRACTS.items():
        for accion in contrato.critical:
            assert permissions.objetivo_confirmacion(agente, accion) is not None, (
                f"({agente}, {accion}) es crítica pero no tiene objetivo de confirmación"
            )


def test_critica_es_subconjunto_de_destructiva():
    """Lo crítico es siempre destructivo: el bloque 'no se deshacen' y el gate
    base no pueden perderlas."""
    for contrato in CONTRACTS.values():
        assert set(contrato.critical) <= set(contrato.destructive)


def test_un_dry_run_critico_nunca_pide_nombre():
    assert not permissions.exige_nombre("git", "tag_release", {"dry_run": True})


def test_la_variable_de_entorno_desactiva_tambien_la_escalera(monkeypatch):
    monkeypatch.setenv(permissions.VAR_ASSUME_YES, "1")
    assert not permissions.exige_nombre("git", "tag_release", {})


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


# -- la escalera en BaseAgent.run ----------------------------------------------
def _agente_fake(context, accion="tag_release"):
    """GitAgent con la acción sustituida por una que registra kwargs."""
    llamadas = {}

    def falso(**kwargs):
        llamadas.update(kwargs)
        return AgentResult(True, "git", accion, "simulado")

    agente = GitAgent(context=context)
    agente.actions = lambda: {accion: falso}
    return agente, llamadas


def test_run_critica_sin_confirm_no_ejecuta(context):
    agente, llamadas = _agente_fake(context)
    resultado = agente.run("tag_release", version="1.0.0")
    assert not resultado.success
    assert "crítica" in resultado.message.lower()
    assert '--confirm-string "1.0.0"' in resultado.needs[0]
    assert "tag_release" not in llamadas


def test_run_critica_con_yes_sin_nombre_no_ejecuta(context):
    """El --yes de reflejo no basta para una crítica: falta el nombre exacto."""
    agente, llamadas = _agente_fake(context)
    resultado = agente.run("tag_release", version="1.0.0", confirm=True)
    assert not resultado.success
    assert '--confirm-string "1.0.0"' in resultado.needs[0]
    assert "tag_release" not in llamadas


def test_run_critica_con_nombre_correcto_ejecuta(context):
    agente, llamadas = _agente_fake(context)
    resultado = agente.run("tag_release", version="1.0.0", confirm=True,
                           confirm_string="1.0.0")
    assert resultado.success
    assert llamadas.get("version") == "1.0.0"
    assert "confirm" not in llamadas and "confirm_string" not in llamadas


def test_run_critica_con_nombre_equivocado_no_ejecuta(context):
    agente, llamadas = _agente_fake(context)
    resultado = agente.run("tag_release", version="1.0.0", confirm=True,
                           confirm_string="2.0.0")
    assert not resultado.success
    assert "tag_release" not in llamadas


def test_run_destructiva_normal_con_yes_sigue_pasando(context):
    """Sin fatiga, las destructivas de nivel normal no escalan a nombre."""
    agente, llamadas = _agente_fake(context, "commit_feature")
    resultado = agente.run("commit_feature", id="X-001", title="algo", confirm=True)
    assert resultado.success
    assert llamadas.get("id") == "X-001"


# -- fatiga de aprobaciones -----------------------------------------------------
def _aprobar_n_veces(context, n):
    """`n` aprobaciones destructivas seguidas (confirmación humana) en el log."""
    for _ in range(n):
        audit.record(
            context, agent="git", action="commit_feature", success=True,
            duration_ms=1.0, message="simulado", confirmed=True,
        )


def test_fatiga_no_se_activa_con_cuatro_aprobaciones(context):
    _aprobar_n_veces(context, permissions.MAX_APROBACIONES_SIN_FALLO - 1)
    assert not permissions.fatiga_activa(context)


def test_fatiga_se_activa_con_cinco_aprobaciones_seguidas(context):
    _aprobar_n_veces(context, permissions.MAX_APROBACIONES_SIN_FALLO)
    assert permissions.fatiga_activa(context)


def test_un_fallo_corta_la_racha_y_rearma_la_vigilancia(context):
    _aprobar_n_veces(context, permissions.MAX_APROBACIONES_SIN_FALLO)
    audit.record(
        context, agent="git", action="commit_feature", success=False,
        duration_ms=1.0, message="falló", confirmed=False,
    )
    assert not permissions.fatiga_activa(context)


def test_bajo_fatiga_una_destructiva_normal_escala_a_nombre(context):
    """
    La prueba del conjunto: tras la racha, `commit_feature` con `--yes` solo
    se bloquea, y se desbloquea tecleando el id de la feature.
    """
    _aprobar_n_veces(context, permissions.MAX_APROBACIONES_SIN_FALLO)
    agente, llamadas = _agente_fake(context, "commit_feature")

    bloqueado = agente.run("commit_feature", id="X-001", title="algo", confirm=True)
    assert not bloqueado.success
    assert "--confirm-string" in bloqueado.needs[0]
    assert "commit_feature" not in llamadas

    autorizado = agente.run("commit_feature", id="X-001", title="algo",
                            confirm=True, confirm_string="X-001")
    assert autorizado.success
    assert llamadas.get("id") == "X-001"


def test_aprobaciones_bajo_asume_yes_no_marcan_confirmacion_humana(context, monkeypatch):
    """Bajo DSKIT_ASSUME_YES no hay confirmación humana: no alimenta la fatiga."""
    import json

    monkeypatch.setenv(permissions.VAR_ASSUME_YES, "1")
    agente, llamadas = _agente_fake(context, "commit_feature")
    agente.run("commit_feature", id="X-001", title="algo")  # sin confirm: la puerta está abierta
    assert llamadas.get("id") == "X-001"

    registro = context.agent_workspace("audit") / "audit.jsonl"
    ultima = json.loads(registro.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert ultima["confirmed"] is False
    assert not permissions.fatiga_activa(context)


def subprocess_log(context) -> str:
    import subprocess

    return subprocess.run(
        ["git", "log", "--oneline"], cwd=context.root,
        capture_output=True, text=True, check=False,
    ).stdout
