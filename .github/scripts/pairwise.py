"""
pairwise.py — Cobertura de todos los pares (all-pairs) para las combinaciones
de copier.yml.

El problema: 23 variables ramifican el template y su producto son ~4·10^10
combinaciones. Probarlas todas es imposible y elegir 20 a mano es una
corazonada — de hecho se nos escaparon 12 errores de lint que solo aparecían
con `use_optuna`, `use_monitoring` y `use_shap` a la vez.

La observación empírica detrás de all-pairs es que la mayoría de los fallos de
interacción se disparan con **dos** variables, no con siete. Cubrir todos los
pares (var_a=x, var_b=y) que pueden coexistir necesita decenas de casos en vez
de miles de millones, y es determinista: mismo copier.yml, misma matriz.

No cubre interacciones de 3+ variables. Por eso `PINNED` mantiene los combos
que ya han roto algo alguna vez: all-pairs es la red, no la única defensa.
"""

from __future__ import annotations

from typing import Any, Callable

Combo = dict[str, Any]


def _pairs(
    variables: dict[str, list],
    applies: Callable[[str, Combo], bool],
    orden: list[str],
    drivers: list[str],
) -> set[tuple]:
    """
    Los pares (var_a, val_a, var_b, val_b) que pueden coexistir DE VERDAD.

    Hay pares que no existen: `cluster_model=DBSCAN` exige `no_supervisado` y
    `nn_loss_fn=CrossEntropyLoss` exige `redes_neuronales`. Perseguirlos hunde
    la cobertura sin motivo — parecían un 40% de huecos y eran imposibles.

    `drivers` son las variables de las que dependen las condiciones `when:`
    (aquí `ml_type`). Un par es factible si existe algún valor de los drivers
    bajo el cual ambas variables aplican.
    """
    objetivo: set[tuple] = set()
    contextos = [{}] if not drivers else [{d: v} for d in drivers for v in variables[d]]
    for i, a in enumerate(orden):
        for b in orden[i + 1 :]:
            for va in variables[a]:
                for vb in variables[b]:
                    for ctx in contextos:
                        tentativa = {**ctx, a: va, b: vb}
                        # El propio driver no puede contradecir al par
                        if any(k in tentativa and tentativa[k] != v for k, v in ctx.items()):
                            continue
                        if applies(a, tentativa) and applies(b, tentativa):
                            objetivo.add((a, va, b, vb))
                            break
    return objetivo


def generate(
    variables: dict[str, list],
    defaults: Combo,
    applies: Callable[[str, Combo], bool] | None = None,
    drivers: list[str] | None = None,
) -> list[Combo]:
    """
    Devuelve la lista mínima-ish de combos que cubre todos los pares válidos.

    Greedy determinista: en cada caso nuevo se elige, variable a variable, el
    valor que cubre más pares pendientes; los empates se rompen por el orden de
    declaración. Sin `random`, así que la matriz solo cambia si cambia
    `copier.yml` — un diff de CI no puede sorprenderte.

    `applies(var, combo_parcial)` decide si una variable tiene sentido en ese
    contexto (las condiciones `when:` de copier). Las que no aplican toman su
    valor por defecto y no participan en la cobertura.
    """
    applies = applies or (lambda _var, _combo: True)
    orden = list(variables)
    pendientes = _pairs(variables, applies, orden, drivers or [])

    combos: list[Combo] = []
    # Cota de seguridad: el greedy siempre avanza, pero si un `applies` mal
    # escrito hiciera imposible cubrir un par, esto evita el bucle infinito.
    limite = len(pendientes) + len(orden) + 10

    while pendientes and len(combos) < limite:
        # Cada combo se SIEMBRA con un par pendiente. Sin esto el greedy
        # reconstruía combos ya vistos en cuanto se agotaban los pares
        # "fáciles" y se quedaba clavado en el 74% de cobertura. Sembrando,
        # cada iteración cubre al menos ese par y el proceso siempre avanza.
        a, va, b, vb = min(pendientes, key=_orden_estable)
        combo: Combo = {a: va, b: vb}
        for d in drivers or []:
            if d in combo:
                continue
            for dv in variables[d]:
                tentativa = {**combo, d: dv}
                if applies(a, tentativa) and applies(b, tentativa):
                    combo[d] = dv
                    break

        for var in orden:
            if var in combo:
                continue
            if not applies(var, combo):
                continue
            mejor, mejor_n = None, -1
            for val in variables[var]:
                # Cuentan los pares TODAVÍA ALCANZABLES con este valor: o su
                # pareja ya está fijada al valor que toca, o aún no se ha
                # decidido y podrá fijarse después. Contar solo los ya
                # emparejados daba 0 en la primera variable (el combo está
                # vacío) y el greedy elegía siempre el primer valor, dejando
                # el 55% de los pares sin cubrir.
                n = sum(
                    1
                    for (a, va, b, vb) in pendientes
                    if (a == var and va == val and combo.get(b, vb) == vb)
                    or (b == var and vb == val and combo.get(a, va) == va)
                )
                if n > mejor_n:
                    mejor, mejor_n = val, n
            combo[var] = mejor

        # Las variables que no aplican en este combo necesitan un valor igual:
        # el render usa StrictUndefined y fallaría si faltara alguna.
        completo = {**defaults, **combo}

        cubiertos = {
            (a, va, b, vb)
            for (a, va, b, vb) in pendientes
            if combo.get(a, _AUSENTE) == va and combo.get(b, _AUSENTE) == vb
        }
        pendientes -= cubiertos
        combos.append(completo)

    return combos


def _orden_estable(par: tuple) -> tuple:
    """Clave de orden total: los valores mezclan bool y str, y `sorted` no
    puede compararlos entre sí. Se ordena por su repr, que es determinista."""
    return tuple(repr(x) for x in par)


class _Ausente:
    """Centinela: `None` y `False` son valores legítimos de copier."""

    def __repr__(self) -> str:
        return "<ausente>"


_AUSENTE = _Ausente()


def etiqueta(combo: Combo, claves: list[str]) -> str:
    """Nombre corto y estable para identificar un combo en la salida de CI."""
    partes = []
    for k in claves:
        v = combo.get(k)
        if v is True:
            partes.append(k.replace("use_", ""))
        elif v not in (False, None, "", "no"):
            partes.append(str(v).replace(" ", "-")[:12])
    return "+".join(partes) or "base"


def _self_test() -> int:
    """
    Autotest sin dependencias: `python pairwise.py --self-test`.

    Este generador decide QUE se prueba en CI. Si se rompe en silencio, la
    cobertura cae y todo sigue en verde — el peor fallo posible en una
    herramienta de test. Por eso se verifica a si mismo.
    """
    fallos = []

    def check(nombre: str, ok: bool, detalle: str = "") -> None:
        print(f"  {'✔' if ok else '✘'} {nombre}{'  ' + detalle if detalle else ''}")
        if not ok:
            fallos.append(nombre)

    # 1. Sin restricciones: cobertura total de pares
    variables = {"a": [1, 2, 3], "b": ["x", "y"], "c": [True, False]}
    defaults = {"a": 1, "b": "x", "c": True}
    combos = generate(variables, defaults)
    objetivo = _pairs(variables, lambda v, c: True, list(variables), [])
    cubiertos = {
        (k1, c[k1], k2, c[k2])
        for c in combos
        for i, k1 in enumerate(variables)
        for k2 in list(variables)[i + 1 :]
    }
    check(
        "cubre todos los pares",
        not (objetivo - cubiertos),
        f"{len(objetivo)} pares en {len(combos)} combos",
    )

    # 2. Cota inferior teorica: nunca menos que el producto de los dos dominios
    #    mayores (aqui 3x2=6). Si sale menos, la cobertura miente.
    check("respeta la cota inferior", len(combos) >= 6, f"{len(combos)} >= 6")

    # 3. Determinismo
    check("determinista", generate(variables, defaults) == combos)

    # 4. Las restricciones se respetan
    vs = {"tipo": ["nn", "clasico"], "capas": [1, 2], "arbol": ["rf", "gb"]}
    ds = {"tipo": "nn", "capas": 1, "arbol": "rf"}

    def aplica(var, combo):
        t = combo.get("tipo")
        if t is None:
            return True
        return (var != "capas" or t == "nn") and (var != "arbol" or t == "clasico")

    con_restriccion = generate(vs, ds, aplica, ["tipo"])
    malos = [
        c
        for c in con_restriccion
        if (c["tipo"] == "clasico" and c["capas"] != ds["capas"])
        or (c["tipo"] == "nn" and c["arbol"] != ds["arbol"])
    ]
    check(
        "respeta las condiciones when",
        not malos,
        f"{len(con_restriccion)} combos, {len(malos)} invalidos",
    )

    # 5. Los pares imposibles no se persiguen
    imposibles = [
        p for p in _pairs(vs, aplica, list(vs), ["tipo"]) if {p[0], p[2]} == {"capas", "arbol"}
    ]
    check("descarta pares imposibles", not imposibles, "capas(nn) x arbol(clasico) nunca coexisten")

    print(f"\n{'PASS' if not fallos else 'FAIL: ' + ', '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    import sys as _sys

    if "--self-test" in _sys.argv:
        _sys.exit(_self_test())
    print(__doc__)
