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

La escalera de fricción
-----------------------
Un `--yes` para todo es una caseta de peaje, no una puerta: cuantas más
confirmaciones se aprueban por reflejo, menos se leen (Anthropic mide ~93% de
aprobados). Por eso la fricción es proporcional al daño:

- Reversible o `--dry_run`: no pregunta. Enseñar una propuesta no cambia nada.
- Destructiva (`destructive`): autoriza `--yes` (o `confirm=True`).
- Crítica (`critical`, subconjunto de `destructive`): exige **el nombre exacto
  del objetivo** vía `--confirm-string`, como el type-to-confirm de GitHub.
  El token cambia con cada operación (`version`, `branch`, `repo_url`...), así
  que no se puede aprobar por reflejo.
- **Fatiga**: N aprobaciones destructivas seguidas sin ningún fallo
  (`MAX_APROBACIONES_SIN_FALLO`) — la puerta deja de fiarse del reflejo y exige
  también el nombre en la siguiente. Un fallo rearma la vigilancia.

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
- `--confirm-string "<nombre>"` — la autorización de las críticas (y de
  cualquier destructiva bajo fatiga): el nombre tiene que coincidir con el
  objetivo de la llamada (ver `OBJETIVO_CONFIRMACION`).
- `DSKIT_ASSUME_YES=1` (o `DSKIT_CONFIRM=0`) — desactiva la puerta entera,
  incluida la escalera crítica. Para CI y automatismos donde ya hay una
  decisión humana detrás. Ponerla en tu shell de trabajo diario es desactivar
  la única red que hay.

Un `dry_run=True` nunca pregunta: no cambia nada, y pedir permiso para
enseñar una propuesta convertiría la puerta en ruido que se aprende a ignorar.
"""

from __future__ import annotations

import os

#: Valores que cuentan como sí/no en las variables de entorno. Se aceptan en
#: español y en inglés porque el proyecto se escribe en los dos.
_SI = {"1", "true", "yes", "si", "sí", "on"}
_NO = {"0", "false", "no", "off"}

VAR_ASSUME_YES = "DSKIT_ASSUME_YES"
VAR_CONFIRM = "DSKIT_CONFIRM"

#: Aprobaciones destructivas seguidas sin ningún fallo a partir de las cuales
#: la puerta deja de fiarse del `--yes` de reflejo (ver `fatiga_activa`). Es
#: política fijada aquí — como `UMBRAL_CERTEZA` en la rúbrica — no algo que el
#: sistema se autoconceda.
MAX_APROBACIONES_SIN_FALLO = 5

#: El kwarg que es "el nombre de la cosa" para cada acción que se escala. El
#: `--confirm-string` tiene que coincidir EXACTO con ese kwarg: así el token no
#: es un "DELETE" memorizable, es la identidad de lo que se va a tocar.
OBJETIVO_CONFIRMACION: dict[tuple[str, str], str] = {
    ("git", "commit_feature"): "id",
    ("git", "create_branch"): "branch_name",
    ("git", "tag_release"): "version",
    ("git", "merge_branch"): "source_branch",
    ("installer", "install_from_git"): "repo_url",
    ("installer", "install_from_path"): "local_path",
}

#: Consecuencia concreta de cada acción crítica: qué se toca y si se puede
#: deshacer. La copia es la seguridad — un "es irreversible" genérico no
#: protege nada porque no dice qué se va a perder.
CONSECUENCIAS_CRITICAS: dict[tuple[str, str], str] = {
    ("git", "tag_release"): (
        "crea la etiqueta '{version}' en el historial compartido: se queda ahí "
        "para siempre, y moverla después de un push exige fuerza y confunde a "
        "quien ya la tiene."
    ),
    ("git", "merge_branch"): (
        "funde '{source_branch}' en '{target_branch}': el historial queda "
        "reescrito, y deshacerlo tras un push exige reset --hard y fuerza."
    ),
    ("installer", "install_from_git"): (
        "clona código de terceros en agents/external/ y lo registra: se "
        "ejecutará en este entorno como un agente más, y desinstalarlo deja "
        "rastros."
    ),
    ("installer", "install_from_path"): (
        "copia código a agents/external/ y lo registra: se ejecutará en este "
        "entorno como un agente más, y desinstalarlo deja rastros."
    ),
}


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


def acciones_criticas(agente: str) -> tuple[str, ...]:
    """El subconjunto de destructivas con radio de explosión alto."""
    from agents.contracts import contract_for

    contrato = contract_for(agente)
    return contrato.critical if contrato is not None else ()


def es_critica(agente: str, accion: str) -> bool:
    return accion in acciones_criticas(agente)


def objetivo_confirmacion(agente: str, accion: str) -> str | None:
    """El kwarg cuyo valor es "el nombre de la cosa" para esta acción."""
    return OBJETIVO_CONFIRMACION.get((agente, accion))


def requiere_confirmacion(agente: str, accion: str, kwargs: dict) -> bool:
    """`True` si esta llamada destructiva necesita alguna autorización."""
    if puerta_desactivada():
        return False
    if accion not in acciones_destructivas(agente):
        return False
    return not kwargs.get("dry_run")


def exige_nombre(agente: str, accion: str, kwargs: dict, ctx=None) -> bool:
    """
    `True` si `--yes` no basta y hay que escribir el nombre exacto del objetivo.

    Las acciones críticas siempre. Las demás, cuando la fatiga ha ganado la
    racha (ver `fatiga_activa`) y la acción tiene un objetivo nombrable. La
    puerta desactivada y el dry-run eximen: no hay estado que proteger.
    """
    if puerta_desactivada() or kwargs.get("dry_run"):
        return False
    if es_critica(agente, accion):
        return True
    if objetivo_confirmacion(agente, accion) is None:
        return False
    return fatiga_activa(ctx)


def nombre_valido(agente: str, accion: str, confirm_string, kwargs: dict) -> bool:
    """`True` si el `confirm_string` dado coincide con el objetivo de la llamada."""
    objetivo = objetivo_confirmacion(agente, accion)
    if objetivo is None:
        # Sin objetivo nombrable no hay nada que verificar; pero una crítica
        # sin objetivo declarado es un fallo de configuración → cerrar en rojo
        # (no se fía del --yes en una acción que debía pedir nombre).
        return not es_critica(agente, accion)
    if not confirm_string:
        return False
    esperado = kwargs.get(objetivo)
    return esperado is not None and str(confirm_string) == str(esperado)


def fatiga_activa(ctx=None) -> bool:
    """
    `True` si hubo `MAX_APROBACIONES_SIN_FALLO` aprobaciones destructivas
    seguidas sin ningún fallo intercalado.

    Se lee del log de auditoría (el que ya registra cada `run()`): una
    aprobación cuenta cuando es confirmación humana real (`confirmed=true`) de
    una acción destructiva. Cualquier `success=false` corta la racha — fallar
    rearma la vigilancia. Un `ctx` ausente (llamadas de test a la política
    pelada) significa puerta no fatigada.
    """
    if ctx is None:
        return False
    from agents.audit import read_entries

    racha = 0
    # read_entries devuelve de viejo a nuevo: la racha se cuenta desde el final.
    for entrada in reversed(read_entries(ctx, last=200)):
        if entrada.get("success") is False:
            break
        if (
            entrada.get("confirmed") is True
            and entrada.get("action") in acciones_destructivas(entrada.get("agent", ""))
        ):
            racha += 1
            if racha >= MAX_APROBACIONES_SIN_FALLO:
                return True
    return False


def _llamada(agente: str, accion: str, kwargs: dict) -> str:
    argumentos = ", ".join(f"{k}={v!r}" for k, v in sorted(kwargs.items()))
    return f"{agente}.{accion}({argumentos})"


def peticion(agente: str, accion: str, kwargs: dict) -> tuple[str, list[str]]:
    """(mensaje, needs) para un bloqueo que basta resolver con `--yes`."""
    llamada = _llamada(agente, accion, kwargs)
    mensaje = (
        f"'{llamada}' es una acción irreversible y no se ha autorizado, así que "
        f"NO se ha ejecutado. Autoriza esta llamada con confirm=True (o --yes en "
        f"la CLI), o desactiva la puerta con {VAR_ASSUME_YES}=1 si sabes lo que haces."
    )
    return mensaje, [
        f"¿Autorizas {llamada}? Repite la llamada con confirm=True para ejecutarla.",
    ]


def peticion_nombre(agente: str, accion: str, kwargs: dict) -> tuple[str, list[str]]:
    """
    (mensaje, needs) para un bloqueo que exige el nombre exacto del objetivo.

    El `needs` muestra el string que hay que escribir (es el objetivo de la
    propia llamada, no un secreto): quien ejecuta tiene que teclearlo a
    propósito, y un reflejo no lo produce.
    """
    llamada = _llamada(agente, accion, kwargs)
    objetivo = objetivo_confirmacion(agente, accion)
    esperado = kwargs.get(objetivo) if objetivo else None

    consecuencia = CONSECUENCIAS_CRITICAS.get((agente, accion))
    if consecuencia:
        valores = {k: v for k, v in kwargs.items() if v is not None}
        try:
            consecuencia = consecuencia.format(**valores)
        except (KeyError, IndexError):
            pass
        mensaje = (
            f"'{llamada}' es una acción crítica: {consecuencia} "
            f"NO se ha ejecutado."
        )
    else:
        mensaje = (
            f"'{llamada}' no se deshace y la puerta ya no se fía del --yes "
            f"(o es una acción crítica): NO se ha ejecutado."
        )

    if esperado is not None:
        needs = [
            f"Repite la llamada con --confirm-string \"{esperado}\" (el nombre "
            f"exacto de lo que toca) para ejecutarla."
        ]
    else:
        needs = [
            f"Falta el objetivo en la llamada ({objetivo or 'kwarg'}): no se puede "
            f"verificar el nombre de lo que vas a tocar. Pásalo y repite con "
            f"--confirm-string."
        ]
    return mensaje, needs
