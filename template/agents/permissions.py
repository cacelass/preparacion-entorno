"""
agents.permissions — La puerta que se pregunta antes de lo que no se deshace.

Por qué existe
--------------
`agents/contracts.py` ya decía cosas como «refactor siempre con dry_run
primero, el humano aprueba». Pero eso era una frase en una tupla: nada en el
código la comprobaba, y el propio `RefactorAgent` tiene `dry_run: bool = False`
por defecto. Un contrato que no se ejecuta es documentación, no un límite.

Esto lo convierte en código. Las acciones marcadas como destructivas en el
contrato de su agente no se ejecutan a través de `BaseAgent.run()` sin
autorización explícita — y `run()` es justo el camino de los automatismos: la
CLI, el `Orchestrator`, `GStack` y `delegate_to()`. Es la diferencia entre un
pipeline que commitea solo y uno que te dice qué iba a commitear.

Qué NO cubre, a propósito
-------------------------
Llamar al método directamente (`GitAgent().tag_release(version="1.0.0")`) no
pasa por la puerta. Ahí hay una persona escribiendo Python con un objetivo
concreto; el riesgo que esto ataja es el del bucle autónomo que encadena
pasos sin que nadie mire.

Cómo se autoriza
----------------
- `confirm=True` como argumento de `run()` — autoriza esa llamada y solo esa.
- `--yes` en la CLI — lo mismo, desde el terminal.
- `DSKIT_ASSUME_YES=1` (o `DSKIT_CONFIRM=0`) — desactiva la puerta entera.
  Para CI y automatismos donde ya hay una decisión humana detrás. Ponerla en
  tu shell de trabajo diario es desactivar la única red que hay.

Un `dry_run=True` nunca pregunta: no cambia nada, y pedir permiso para
enseñar una propuesta convertiría la puerta en ruido que se aprende a
ignorar.
"""

from __future__ import annotations

import os

#: Valores que cuentan como sí/no en las variables de entorno. Se aceptan en
#: español y en inglés porque el proyecto se escribe en los dos.
_SI = {"1", "true", "yes", "si", "sí", "on"}
_NO = {"0", "false", "no", "off"}

VAR_ASSUME_YES = "DSKIT_ASSUME_YES"
VAR_CONFIRM = "DSKIT_CONFIRM"


def puerta_desactivada() -> bool:
    """`True` si el entorno ya autorizó todo (CI, automatismos)."""
    if os.environ.get(VAR_ASSUME_YES, "").strip().lower() in _SI:
        return True
    return os.environ.get(VAR_CONFIRM, "").strip().lower() in _NO


def acciones_destructivas(agente: str) -> tuple[str, ...]:
    """Lo que el contrato del agente declara como irreversible."""
    from agents.contracts import contract_for

    contrato = contract_for(agente)
    return contrato.destructive if contrato is not None else ()


def requiere_confirmacion(agente: str, accion: str, kwargs: dict) -> bool:
    if puerta_desactivada():
        return False
    if accion not in acciones_destructivas(agente):
        return False
    return not kwargs.get("dry_run")


def peticion(agente: str, accion: str, kwargs: dict) -> tuple[str, list[str]]:
    """(mensaje, needs) para el `AgentResult` que se devuelve sin ejecutar."""
    argumentos = ", ".join(f"{k}={v!r}" for k, v in sorted(kwargs.items()) if k != "confirm")
    llamada = f"{agente}.{accion}({argumentos})"
    mensaje = (
        f"'{llamada}' es una acción irreversible y no se ha autorizado, así que "
        f"NO se ha ejecutado. Autoriza esta llamada con confirm=True (o --yes en "
        f"la CLI), o desactiva la puerta con {VAR_ASSUME_YES}=1 si sabes lo que haces."
    )
    return mensaje, [
        f"¿Autorizas {llamada}? Repite la llamada con confirm=True para ejecutarla.",
    ]
