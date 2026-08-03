"""
Tests de la redacción de credenciales (`agents/redaction.py`).

Lo importante aquí no es que tape tokens —eso lo hace cualquier regex— sino
que **no destroce los mensajes normales**. Una redacción que convierte
`token de sesión` en `token de [REDACTADO]` hace ilegible la salida de los 30
agentes, y lo primero que se hace con algo así es quitarlo de en medio.
"""

from __future__ import annotations

from agents.core.base_agent import AgentResult
from agents.redaction import MARCA, contiene_secreto, redactar, redactar_resultado


# -- tapa lo que hay que tapar -------------------------------------------------
def test_tapa_un_token_de_github():
    texto = "falló con gh" + "p_" + "a" * 36
    assert MARCA in redactar(texto)
    assert "a" * 36 not in redactar(texto)


def test_tapa_una_clave_de_aws():
    assert redactar("clave AKIAIOSFODNN7EXAMPLE encontrada") == f"clave {MARCA} encontrada"


def test_tapa_un_jwt():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.abcdefghijklmnop"
    assert jwt not in redactar(f"Authorization: Bearer {jwt}")


def test_tapa_la_contrasena_de_una_url_pero_deja_el_host():
    salida = redactar("conectando a postgres://usuario:s3cr3t0@db.local/prod")
    assert "s3cr3t0" not in salida
    assert "usuario" in salida, "el usuario ayuda a entender el mensaje"


def test_tapa_una_asignacion():
    salida = redactar("exportado API_KEY=abcdef123456 al entorno")
    assert "abcdef123456" not in salida
    assert "API_KEY" in salida, "el nombre de la variable no es el secreto"


def test_tapa_una_clave_privada():
    assert MARCA in redactar("-----BEGIN RSA PRIVATE KEY-----")


# -- y no toca lo demás ---------------------------------------------------------
def test_no_toca_un_mensaje_normal():
    mensajes = [
        "12 figura(s) analizada(s), 3 con avisos.",
        "Siguiente: DATA-001 — EDA del dataset",
        "El token de sesión caducó",
        "Revisa el password del formulario en la documentación",
        "commit_feature necesita el id y el título",
        "",
    ]
    for mensaje in mensajes:
        assert redactar(mensaje) == mensaje, f"no debería tocar: {mensaje!r}"


def test_contiene_secreto_solo_avisa_cuando_toca():
    assert contiene_secreto("AKIAIOSFODNN7EXAMPLE")
    assert not contiene_secreto("todo en orden, 0 hallazgos")


# -- integración con AgentResult -------------------------------------------------
def test_redacta_message_warnings_y_needs():
    resultado = AgentResult(
        True, "env", "info", "cargado API_KEY=supersecreto123 del entorno",
        warnings=["ojo: AKIAIOSFODNN7EXAMPLE en el .env"],
        needs=["¿confirmas con token=ghp_" + "b" * 36 + "?"],
    )
    redactar_resultado(resultado)
    assert "supersecreto123" not in resultado.message
    assert "AKIAIOSFODNN7EXAMPLE" not in resultado.warnings[0]
    assert "b" * 36 not in resultado.needs[0]


def test_no_toca_el_data():
    """
    `data` lo consumen otros agentes por clave. Reescribirlo a ciegas rompería
    el encadenado entre agentes sin avisar; el canal que hay que proteger es
    el texto que acaba leyendo un modelo.
    """
    payload = {"ruta": "/tmp/x", "clave": "AKIAIOSFODNN7EXAMPLE"}
    resultado = AgentResult(True, "x", "y", "ok", data=payload)
    redactar_resultado(resultado)
    assert resultado.data == payload


def test_run_redacta_de_punta_a_punta(context, monkeypatch):
    """La redacción tiene que ocurrir en `run`, no en cada agente."""
    from agents.agents.env_agent import EnvAgent

    agente = EnvAgent(context=context)
    monkeypatch.setattr(
        agente, "actions",
        lambda: {"info": lambda: AgentResult(
            True, "env", "info", "GITHUB_TOKEN=ghp_" + "c" * 36,
        )},
    )
    resultado = agente.run("info")
    assert "c" * 36 not in resultado.message
    assert MARCA in resultado.message


def test_el_log_de_auditoria_no_guarda_secretos(context, monkeypatch):
    from agents.agents.env_agent import EnvAgent

    agente = EnvAgent(context=context)
    monkeypatch.setattr(
        agente, "actions",
        lambda: {"info": lambda: AgentResult(True, "env", "info", "AKIAIOSFODNN7EXAMPLE")},
    )
    agente.run("info")
    registro = context.agent_workspace("audit") / "audit.jsonl"
    assert "AKIAIOSFODNN7EXAMPLE" not in registro.read_text(encoding="utf-8")
