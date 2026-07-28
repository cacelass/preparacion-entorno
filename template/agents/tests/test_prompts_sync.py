"""
test_prompts_sync.py — Los prompts no pueden desincronizarse del código.

El bloque autogenerado de cada prompt sale de `actions()` y de
`contracts.py`. Antes esa información estaba reescrita a mano en la prosa, y
dos fuentes de verdad divergen siempre — el agente `doc` llegó a estar
registrado sin contrato sin que nadie lo notara.
"""

from __future__ import annotations

from agents.core.registry import agent_registry
from agents.prompts_sync import BEGIN, END, _splice, render_block, sync, sync_assistants


def _agent(context, name: str):
    agent_registry.discover()
    return agent_registry.get(name)(context=context)


# -- el bloque ----------------------------------------------------------------
def test_el_bloque_lista_todas_las_acciones(context):
    agente = _agent(context, "git")
    bloque = render_block("git", agente)
    for accion in agente.actions():
        assert f"run git {accion}" in bloque


def test_el_bloque_marca_los_argumentos_obligatorios(context):
    """`commit_with_changelog` necesita `message`: debe verse que es obligatorio."""
    bloque = render_block("git", _agent(context, "git"))
    linea = next(l for l in bloque.splitlines() if "commit_with_changelog" in l)
    assert "--message" in linea and "obligatorio" in linea


def test_el_bloque_trae_los_limites_del_contrato(context):
    from agents.contracts import contract_for

    bloque = render_block("test", _agent(context, "test"))
    contrato = contract_for("test")
    assert contrato.role in bloque
    for prohibido in contrato.cannot:
        assert prohibido in bloque


def test_el_bloque_va_delimitado(context):
    bloque = render_block("git", _agent(context, "git"))
    assert bloque.startswith(BEGIN)
    assert bloque.rstrip().endswith(END)


# -- el empalme ---------------------------------------------------------------
def test_splice_conserva_la_prosa_escrita_a_mano():
    prosa = "# Prompt — X\n\nEl criterio de este agente, escrito a mano.\n"
    resultado = _splice(prosa, f"{BEGIN}\nnuevo\n{END}\n")
    assert "El criterio de este agente, escrito a mano." in resultado
    assert "nuevo" in resultado


def test_splice_reemplaza_el_bloque_viejo_sin_duplicarlo():
    texto = f"# Prompt\n\nprosa\n\n{BEGIN}\nviejo\n{END}\n"
    resultado = _splice(texto, f"{BEGIN}\nnuevo\n{END}\n")
    assert "viejo" not in resultado
    assert resultado.count(BEGIN) == 1
    assert "prosa" in resultado


def test_splice_es_idempotente():
    prosa = "# Prompt\n\nprosa\n"
    bloque = f"{BEGIN}\ncontenido\n{END}\n"
    una = _splice(prosa, bloque)
    dos = _splice(una, bloque)
    assert una == dos


# -- el chequeo ---------------------------------------------------------------
def test_check_no_escribe_nada(context):
    prompts = context.root / "agents" / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "git_agent.md").write_text("# Prompt — GitAgent\n\nprosa\n", encoding="utf-8")

    sync(write=False, context=context)

    assert (prompts / "git_agent.md").read_text() == "# Prompt — GitAgent\n\nprosa\n"


def test_write_si_escribe_el_bloque(context):
    prompts = context.root / "agents" / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "git_agent.md").write_text("# Prompt — GitAgent\n\nprosa\n", encoding="utf-8")

    sync(write=True, context=context)

    texto = (prompts / "git_agent.md").read_text()
    assert "prosa" in texto
    assert BEGIN in texto and "run git " in texto


def test_reporta_los_agentes_registrados_sin_prompt(context):
    # El proyecto temporal no tiene agents/prompts/, así que TODOS los agentes
    # registrados deben salir como "sin prompt".
    (context.root / "agents" / "prompts").mkdir(parents=True, exist_ok=True)
    _, sin_prompt = sync(write=False, context=context)
    assert set(sin_prompt) == set(agent_registry.all())


# -- espejo a otros asistentes -------------------------------------------------

def _sembrar_opencode(context, nombre: str = "lider") -> None:
    d = context.root / ".opencode" / "agents"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{nombre}.md").write_text(
        f"# {nombre.capitalize()} — orquestador del arnés\n\n"
        "Diriges el ciclo de trabajo del proyecto.\n\n## Protocolo\n\npaso 1\n",
        encoding="utf-8",
    )


def test_espeja_los_subagentes_a_claude(context):
    _sembrar_opencode(context)
    pendientes = sync_assistants(write=True, context=context)
    assert "lider" in pendientes

    generado = (context.root / ".claude" / "agents" / "lider.md").read_text()
    assert generado.startswith("---")
    assert "name: lider" in generado
    assert "description: Diriges el ciclo de trabajo del proyecto" in generado
    assert "## Protocolo" in generado, "el cuerpo del prompt debe viajar entero"


def test_el_explorer_se_espeja_con_herramientas_de_solo_lectura(context):
    _sembrar_opencode(context, "explorer")
    sync_assistants(write=True, context=context)
    generado = (context.root / ".claude" / "agents" / "explorer.md").read_text()
    assert "tools:" in generado
    assert "Write" not in generado.split("---")[1], "el explorer no debe poder escribir"


def test_el_espejo_es_idempotente(context):
    _sembrar_opencode(context)
    sync_assistants(write=True, context=context)
    assert sync_assistants(write=False, context=context) == []


def test_detecta_que_el_espejo_se_quedo_viejo(context):
    _sembrar_opencode(context)
    sync_assistants(write=True, context=context)

    fuente = context.root / ".opencode" / "agents" / "lider.md"
    fuente.write_text(fuente.read_text() + "\n## Regla nueva\n\nno te la saltes\n", encoding="utf-8")

    assert sync_assistants(write=False, context=context) == ["lider"]
    sync_assistants(write=True, context=context)
    assert "Regla nueva" in (context.root / ".claude" / "agents" / "lider.md").read_text()


def test_check_no_escribe_el_espejo(context):
    _sembrar_opencode(context)
    sync_assistants(write=False, context=context)
    assert not (context.root / ".claude" / "agents" / "lider.md").exists()
