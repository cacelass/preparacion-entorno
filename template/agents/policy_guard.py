"""
agents.policy_guard — La frontera entre lo que el modelo pide y lo que se ejecuta.

El problema que resuelve
------------------------
`agents/permissions.py` protege las acciones **de los agentes Python**. Pero
el asistente no solo llama a agentes: también usa sus propias herramientas
(`Bash`, `Read`, `Write`, `Edit`), y ahí dskit no pinta nada — esa frontera es
del arnés anfitrión (Claude Code, opencode…). Si el modelo decide `rm -rf`,
leer `.env` o hacer `git push`, ningún contrato de este repositorio lo ve
pasar.

Lo que sí puede hacer dskit es **poner la política**, en código, y que el
anfitrión la invoque antes de cada llamada a herramienta. Eso es este módulo:
un guardia que se ejecuta como hook `PreToolUse`, recibe por stdin lo que el
modelo quiere hacer y decide.

    stdin: {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}
    exit 0 → adelante
    exit 2 → bloqueado (stderr se le devuelve al modelo como motivo)

Está en `agents/` y no en `.claude/` a propósito: la política es una sola y
la puede llamar cualquier asistente que sepa ejecutar un comando, igual que
el resto del sistema de agentes.

Qué NO es
---------
No es un sandbox. Un comando suficientemente creativo se salta cualquier lista
de patrones, y quien te diga lo contrario te está vendiendo algo. Es la capa
que convierte los accidentes y las inyecciones evidentes en un error legible,
mientras que el aislamiento de verdad (contenedor, usuario sin privilegios,
red cerrada) sigue siendo cosa de dónde ejecutas el asistente.

Ante la duda, deja pasar
------------------------
Si el JSON no se entiende o el hook falla, sale con 0. Un guardia roto que
bloquea toda la sesión se desactiva a los diez minutos, y entonces no protege
de nada. Lo que se juega aquí es defensa en profundidad, no la única puerta.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PERMITIR = 0
BLOQUEAR = 2

#: Ficheros que un agente no tiene por qué leer nunca. No es que sean
#: secretos «probablemente»: es que si acaban en la ventana del modelo, ya
#: están fuera de tu control y no hay vuelta atrás.
_RUTAS_PROHIBIDAS = (
    ".env", ".env.local", ".env.production",
    "id_rsa", "id_ed25519", ".netrc", ".pgpass", "credentials",
    ".aws/", ".ssh/", ".gnupg/", ".docker/config.json",
    "kubeconfig", ".kube/config",
)

#: Sufijos de material criptográfico.
_SUFIJOS_PROHIBIDOS = (".pem", ".key", ".p12", ".pfx", ".keystore")

#: `.env.example` existe para leerse: es la plantilla sin valores. Excluirlo
#: evita que el guardia bloquee justo el fichero que documenta el resto.
_EXCEPCIONES = (".env.example", ".env.template", ".env.sample")

#: Comandos que no se deshacen o que sacan datos de la máquina. Cada patrón
#: está aquí por un motivo concreto, no por sonar peligroso.
_COMANDOS_BLOQUEADOS: tuple[tuple[str, str], ...] = (
    (r"\brm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*[rf][a-zA-Z]*\s+(/|~|\$HOME)(\s|$)",
     "borrado recursivo fuera del proyecto"),
    (r"\bgit\s+push\b",
     "el push es siempre una decisión del humano (ver AGENTS.md)"),
    (r"\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f)",
     "descarta trabajo sin posibilidad de recuperarlo"),
    (r"\bgit\s+(push\s+.*--force|.*\s--force-with-lease)",
     "reescribe historial publicado"),
    (r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh",
     "descargar y ejecutar en un solo paso: no hay forma de revisar qué se ejecuta"),
    (r"\bsudo\b",
     "elevar privilegios queda fuera de lo que un agente decide"),
    (r"\bmkfs(\.|\s)|\bdd\s+if=.*of=/dev/",
     "escritura directa sobre dispositivos"),
    (r"\bchmod\s+(-R\s+)?777\b",
     "permisos abiertos a todo el mundo"),
    (r":\(\)\s*\{.*\};\s*:",
     "fork bomb"),
    (r"\bhistory\s+-c\b|>\s*~/\.bash_history",
     "borrar el rastro de lo ejecutado"),
)

_HERRAMIENTAS_DE_FICHERO = {"Read", "Write", "Edit", "NotebookEdit"}
_HERRAMIENTAS_DE_SHELL = {"Bash", "BashOutput"}


def _es_ruta_prohibida(ruta: str) -> str | None:
    """Devuelve el motivo si `ruta` no se puede tocar, o `None`."""
    if not ruta:
        return None
    normalizada = ruta.replace("\\", "/")
    nombre = normalizada.rsplit("/", 1)[-1]

    if nombre in _EXCEPCIONES:
        return None
    if nombre.endswith(_SUFIJOS_PROHIBIDOS):
        return f"'{nombre}' parece material criptográfico"
    for prohibida in _RUTAS_PROHIBIDAS:
        if prohibida.endswith("/"):
            if f"/{prohibida}" in f"/{normalizada}":
                return f"'{prohibida}' guarda credenciales"
        elif nombre == prohibida:
            return f"'{prohibida}' guarda credenciales"
    return None


def _fuera_del_proyecto(ruta: str, raiz: Path) -> bool:
    """`True` si una ESCRITURA cae fuera de la raíz del proyecto."""
    if not ruta:
        return False
    try:
        destino = Path(ruta)
        destino = destino if destino.is_absolute() else raiz / destino
        destino.resolve().relative_to(raiz.resolve())
    except (ValueError, OSError):
        return True
    return False


def _valores_de_texto(dato: object, profundidad: int = 0) -> list[str]:
    """Todas las cadenas de una estructura anidada, para poder inspeccionarlas."""
    if profundidad > 5:
        return []
    if isinstance(dato, str):
        return [dato]
    if isinstance(dato, dict):
        return [v for x in dato.values() for v in _valores_de_texto(x, profundidad + 1)]
    if isinstance(dato, (list, tuple)):
        return [v for x in dato for v in _valores_de_texto(x, profundidad + 1)]
    return []


def _revisar_comando(comando: str) -> str | None:
    for patron, motivo in _COMANDOS_BLOQUEADOS:
        if re.search(patron, comando):
            return motivo
    return None


def evaluar(evento: dict, raiz: Path | None = None) -> tuple[int, str]:
    """
    (código de salida, motivo). Separado de `main` para poder testearlo.
    """
    raiz = raiz or Path.cwd()
    herramienta = evento.get("tool_name", "")
    entrada = evento.get("tool_input") or {}
    if not isinstance(entrada, dict):
        return PERMITIR, ""

    if herramienta in _HERRAMIENTAS_DE_FICHERO:
        ruta = str(entrada.get("file_path") or entrada.get("notebook_path") or "")
        motivo = _es_ruta_prohibida(ruta)
        if motivo:
            return BLOQUEAR, (
                f"Bloqueado por la política del proyecto: {motivo}. "
                f"Si necesitas su contenido, pide al humano el valor concreto — "
                f"no leas el fichero."
            )
        if herramienta != "Read" and _fuera_del_proyecto(ruta, raiz):
            return BLOQUEAR, (
                f"Bloqueado: '{ruta}' queda fuera de la raíz del proyecto "
                f"({raiz}). Los agentes escriben dentro del proyecto."
            )

    # Un servidor MCP es código de terceros al que el modelo le pasa
    # argumentos. No se sabe qué herramientas expone ni cómo se llaman sus
    # parámetros, así que la única comprobación honesta es genérica: que
    # ninguno de los valores que recibe apunte a un fichero de credenciales.
    if herramienta.startswith("mcp__"):
        for valor in _valores_de_texto(entrada):
            motivo = _es_ruta_prohibida(valor)
            if motivo:
                return BLOQUEAR, (
                    f"Bloqueado por la política del proyecto: {motivo}. "
                    f"La llamada a '{herramienta}' referencia '{valor}'."
                )

    if herramienta in _HERRAMIENTAS_DE_SHELL:
        comando = str(entrada.get("command") or "")
        motivo = _revisar_comando(comando)
        if motivo:
            return BLOQUEAR, (
                f"Bloqueado por la política del proyecto: {motivo}. "
                f"Comando: {comando[:200]}"
            )
        # Un comando de shell también puede leer un fichero prohibido.
        for token in re.findall(r"[\w./~-]+", comando):
            prohibido = _es_ruta_prohibida(token)
            if prohibido:
                return BLOQUEAR, (
                    f"Bloqueado por la política del proyecto: {prohibido}. "
                    f"El comando referencia '{token}'."
                )

    return PERMITIR, ""


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        evento = json.load(sys.stdin)
    except Exception:  # noqa: BLE001 — ver «Ante la duda, deja pasar»
        return PERMITIR
    if not isinstance(evento, dict):
        return PERMITIR

    try:
        codigo, motivo = evaluar(evento)
    except Exception as exc:  # noqa: BLE001
        print(f"policy_guard falló y deja pasar: {exc}", file=sys.stderr)
        return PERMITIR

    if codigo != PERMITIR:
        print(motivo, file=sys.stderr)
    return codigo


if __name__ == "__main__":
    sys.exit(main())
