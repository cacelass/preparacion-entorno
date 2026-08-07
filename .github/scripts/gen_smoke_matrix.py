"""
gen_smoke_matrix.py — La matriz del job `smoke`, generada en vez de escrita.

El job `smoke` es caro: genera el proyecto, instala dependencias de verdad y
ejecuta sus tests. Entre 8 y 13 minutos por combo, asi que no se puede hacer
all-pairs completo como en el render (ahi son 194 combos por 2 minutos en
total).

El problema no era el numero de combos, era CUALES. De los 14 flags
opcionales, los 10 combos escritos a mano solo ejercitaban UNO (`use_rag`, y
por venir en True por defecto, no por decision). `use_api`, `use_optuna`,
`use_monitoring`, `use_docker`, `use_mlflow`, `use_duckdb`, los boosters, SHAP,
conformal, calibration y graphify no se instalaban ni se ejecutaban jamas. Los
12 errores de lint que se colaron vivian exactamente ahi.

La idea: los flags viajan gratis. Cubrir `ml_type` x `task_type` x arquitectura
ya obliga a ~10 combos; encender en cada uno un subconjunto distinto de flags
no cuesta ni un job mas y convierte 0 cobertura de opcionales en cobertura
por pares.

Peso: `redes_neuronales` ya arrastra torch, que domina el tiempo de install.
Los flags pesados (monitoring, rag, docker) se cargan sobre los ml_type
ligeros para no dispararlo.

    python .github/scripts/gen_smoke_matrix.py          # legible
    python .github/scripts/gen_smoke_matrix.py --json   # para GITHUB_OUTPUT
"""

from __future__ import annotations

import json
import sys

#: Los combos base que no se negocian: cada ml_type con cada task_type, y las
#: arquitecturas de red que tienen codigo propio. Esto ya estaba y es correcto.
BASE: list[dict] = [
    {
        "label": "sup-clf",
        "ml_type": "supervisado",
        "task_type": "clasificacion",
        "model_type": "todos",
        "python-version": "3.10",
    },
    {
        "label": "sup-reg",
        "ml_type": "supervisado",
        "task_type": "regresion",
        "model_type": "todos",
        "python-version": "3.12",
    },
    {
        "label": "nosup",
        "ml_type": "no_supervisado",
        "task_type": "clasificacion",
        "cluster_model": "todos",
        "python-version": "3.12",
    },
    {
        "label": "hibrido-clf",
        "ml_type": "hibrido",
        "task_type": "clasificacion",
        "model_type": "todos",
        "python-version": "3.11",
    },
    {
        "label": "hibrido-reg",
        "ml_type": "hibrido",
        "task_type": "regresion",
        "model_type": "todos",
        "python-version": "3.13",
    },
    {
        "label": "nn-mlp-clf-adamw",
        "ml_type": "redes_neuronales",
        "task_type": "clasificacion",
        "nn_model": "MLP",
        "optimizer_type": "AdamW",
        "nn_loss_fn": "Auto",
        "python-version": "3.12",
    },
    {
        "label": "nn-mlp-reg-sgd-mse",
        "ml_type": "redes_neuronales",
        "task_type": "regresion",
        "nn_model": "MLP",
        "optimizer_type": "SGD",
        "nn_loss_fn": "MSELoss",
        "python-version": "3.13",
    },
    {
        "label": "nn-lstm-clf-rmsprop",
        "ml_type": "redes_neuronales",
        "task_type": "clasificacion",
        "nn_model": "LSTM",
        "optimizer_type": "RMSProp",
        "nn_loss_fn": "BCEWithLogitsLoss",
        "python-version": "3.11",
    },
    {
        "label": "nn-gru-reg-adagrad-l1",
        "ml_type": "redes_neuronales",
        "task_type": "regresion",
        "nn_model": "GRU",
        "optimizer_type": "Adagrad",
        "nn_loss_fn": "L1Loss",
        "python-version": "3.12",
    },
    {
        "label": "nn-transformer-clf",
        "ml_type": "redes_neuronales",
        "task_type": "clasificacion",
        "nn_model": "Transformer",
        "optimizer_type": "AdamW",
        "nn_loss_fn": "Auto",
        "python-version": "3.10",
    },
]

#: Flags que se reparten entre los combos base. Cada uno debe salir encendido
#: al menos una vez, y emparejado con ml_type distintos. `coste` es el peso
#: aproximado de instalacion: se evita apilar los caros sobre torch.
FLAGS: dict[str, dict] = {
    "use_mlflow": {"coste": 1},
    "use_api": {"coste": 1},
    "use_duckdb": {"coste": 1},
    "use_shap": {"coste": 1, "solo": {"supervisado", "hibrido"}},
    "use_conformal": {"coste": 0, "solo": {"supervisado", "hibrido", "redes_neuronales"}},
    "use_xgboost": {"coste": 1, "solo": {"supervisado", "hibrido"}},
    "use_lightgbm": {"coste": 1, "solo": {"supervisado", "hibrido"}},
    "use_catboost": {"coste": 2, "solo": {"supervisado", "hibrido"}},
    "use_calibration": {"coste": 0, "solo": {"redes_neuronales"}},
    "use_optuna": {"coste": 2},
    "use_monitoring": {"coste": 2},
    "use_docker": {"coste": 1},
    "use_rag": {"coste": 2},
    # use_sdd no instala dependencias nuevas (el mutador es stdlib): coste 0.
    "use_sdd": {"coste": 0},
    "graphify_mode": {"coste": 2, "valores": ["solo graphify", "graphify + obsidian vault"]},
}

#: Presupuesto de coste por combo. Los de red neuronal arrancan con torch ya
#: encima, asi que admiten menos carga adicional.
PRESUPUESTO = {"redes_neuronales": 2, "supervisado": 5, "hibrido": 5, "no_supervisado": 5}


def construir() -> list[dict]:
    """Reparte los flags sobre los combos base, en orden y sin aleatoriedad."""
    combos = [dict(c) for c in BASE]
    gastado = {c["label"]: 0 for c in combos}

    for flag, spec in FLAGS.items():
        solo = spec.get("solo")
        valores = spec.get("valores", [True])
        for valor in valores:
            # El combo elegible con menos carga acumulada: reparte parejo y es
            # determinista (los empates los rompe el orden de BASE).
            candidatos = [
                c
                for c in combos
                if (solo is None or c["ml_type"] in solo)
                and flag not in c
                and gastado[c["label"]] + spec["coste"] <= PRESUPUESTO[c["ml_type"]]
            ]
            if not candidatos:
                continue
            elegido = min(
                candidatos,
                key=lambda c: (
                    gastado[c["label"]],
                    BASE.index(next(b for b in BASE if b["label"] == c["label"])),
                ),
            )
            elegido[flag] = valor
            gastado[elegido["label"]] += spec["coste"]

    for c in combos:
        opciones = [k for k in c if k.startswith("use_") or k == "graphify_mode"]
        extra = "+".join(k.replace("use_", "")[:6] for k in sorted(opciones))
        c["label"] = c["label"] + (f"+{extra}" if extra else "")
        # JSON, no "key=val": `graphify + obsidian vault` lleva espacios.
        c["data"] = json.dumps(
            {k: v for k, v in c.items() if k not in ("label", "data", "python-version")},
            ensure_ascii=False,
        )
    return combos


def cobertura(combos: list[dict]) -> tuple[int, int, list[str]]:
    """Cuantos flags quedan encendidos alguna vez, y cuales no."""
    encendidos = {k for c in combos for k, v in c.items() if k in FLAGS and v not in (False, "no")}
    faltan = sorted(set(FLAGS) - encendidos)
    return len(encendidos), len(FLAGS), faltan


def main() -> int:
    combos = construir()
    matriz = [
        {"label": c["label"], "data": c["data"], "python-version": c["python-version"]}
        for c in combos
    ]
    if "--json" in sys.argv:
        print(json.dumps(matriz, ensure_ascii=False))
        return 0

    n, total, faltan = cobertura(combos)
    print(f"Combos: {len(matriz)}  ·  flags opcionales cubiertos: {n}/{total}")
    if faltan:
        print(f"SIN CUBRIR: {', '.join(faltan)}")
    print()
    for c in matriz:
        print(f"  {c['python-version']}  {c['label']}")
        print(f"      {c['data']}")
    return 1 if faltan else 0


if __name__ == "__main__":
    sys.exit(main())
