"""
agents.redaction — Tapa credenciales antes de que salgan del proceso.

Por qué existe
--------------
`secrets_tool` sabe *encontrar* secretos desde que existe el proyecto, pero
solo los buscaba dentro de los ficheros del repositorio. Nadie miraba la otra
dirección: lo que un agente **devuelve**. Y ahí sale todo — un `env info` con
un `.env` cargado, un `analyze_diff` sobre un commit que trae una clave, la
salida de un comando que imprime su configuración. Ese texto va a dos sitios
que no son inocentes:

1. **A la ventana del modelo**, donde deja de estar bajo tu control.
2. **A `audit.jsonl`**, que se queda en el disco y, si el proyecto no lo
   ignora, viaja en un commit.

Esto no impide que un secreto se filtre a un fichero — para eso está el
escáner. Impide que se filtre por el canal que el propio arnés abre.

Cómo se aplica
--------------
`BaseAgent.run()` redacta el `message` y los `warnings` de todo resultado, y
`audit.record` redacta lo que escribe. Las expresiones son las mismas que usa
`secrets_tool`: una sola definición de qué es un secreto, no dos que se
desincronizan.

Lo que NO hace, a propósito
---------------------------
No toca `AgentResult.data`. Ahí viven estructuras que otros agentes consumen
por clave (rutas, métricas, listas de ficheros) y reescribirlas a ciegas
rompería el encadenado entre agentes sin avisar. El `data` es para máquinas;
el `message` es lo que acaba leyendo un modelo o una persona.
"""

from __future__ import annotations

import re
from typing import Any

from agents.tools.secrets_tool import (
    _AWS_KEY_RE,
    _GITHUB_TOKEN_RE,
    _GITLAB_TOKEN_RE,
    _JWT_RE,
    _OPENAI_TOKEN_RE,
    _PRIVATE_KEY_RE,
    _SLACK_TOKEN_RE,
    _URL_PASSWORD_RE,
)

MARCA = "[REDACTADO]"

#: Patrones de token completo: se sustituye toda la coincidencia.
_COMPLETOS = (
    _AWS_KEY_RE, _GITHUB_TOKEN_RE, _OPENAI_TOKEN_RE, _GITLAB_TOKEN_RE,
    _SLACK_TOKEN_RE, _JWT_RE, _PRIVATE_KEY_RE,
)

#: `esquema://usuario:contraseña@host` — se conserva el esquema y el usuario,
#: que son los que hacen falta para entender el mensaje, y se tapa solo la
#: clave. El esquema es genérico y no solo `https?` a propósito: la cadena de
#: conexión que más veces acaba en un log es la de la base de datos
#: (`postgres://`, `mysql://`, `redis://`, `mongodb+srv://`).
_URL_CON_CLAVE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://[^:/\s]+):([^@/\s]+)@")

#: `API_KEY=valor` y equivalentes. Aquí el valor sí puede ser cualquier cosa,
#: así que la coincidencia se ancla al nombre de la variable y se tapa lo que
#: venga después — con o sin comillas, que en un mensaje suelen faltar.
_ASIGNACION = re.compile(
    r"(?i)\b(api[_-]?key|secret|password|passwd|token|access[_-]?key)"
    r"(\s*[:=]\s*)(['\"]?)([^\s'\"]{6,})(['\"]?)"
)


def redactar(texto: str) -> str:
    """Devuelve `texto` con las credenciales sustituidas por `[REDACTADO]`."""
    if not texto:
        return texto
    for patron in _COMPLETOS:
        texto = patron.sub(MARCA, texto)
    texto = _URL_CON_CLAVE.sub(rf"\1:{MARCA}@", texto)
    texto = _URL_PASSWORD_RE.sub(f"https://{MARCA}@", texto)
    return _ASIGNACION.sub(rf"\1\2\3{MARCA}\5", texto)


def contiene_secreto(texto: str) -> bool:
    """`True` si `redactar` cambiaría algo. Útil para avisar, no solo tapar."""
    return bool(texto) and redactar(texto) != texto


def redactar_lista(textos: list[str] | None) -> list[str]:
    return [redactar(t) for t in (textos or [])]


def redactar_resultado(resultado: Any) -> Any:
    """
    Redacta `message` y `warnings` de un `AgentResult`, en el sitio.

    Se muta en vez de reconstruir porque `AgentResult` lo crean 30 agentes con
    firmas distintas y copiarlo aquí obligaría a mantener esta función al día
    con cada campo nuevo. Lo que se toca son dos strings.
    """
    mensaje = getattr(resultado, "message", None)
    if isinstance(mensaje, str):
        resultado.message = redactar(mensaje)
    avisos = getattr(resultado, "warnings", None)
    if isinstance(avisos, list):
        resultado.warnings = redactar_lista(avisos)
    necesita = getattr(resultado, "needs", None)
    if isinstance(necesita, list):
        resultado.needs = redactar_lista(necesita)
    return resultado
