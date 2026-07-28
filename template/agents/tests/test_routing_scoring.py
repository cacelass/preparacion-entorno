"""
test_routing_scoring.py — Tests de `BaseAgent.can_handle`.

Protege las dos propiedades del ruteo que son fáciles de romper sin darse
cuenta al tocar `capabilities` de cualquier agente:

1. **Insensible a acentos.** Es un proyecto en español: "documentación" y
   "documentacion" son la misma palabra para el usuario, y deben serlo para
   el ruteo.
2. **La frase específica gana a la palabra genérica.** Si no, un agente con
   dos keywords genéricas se lleva consultas que pertenecen claramente a otro.
"""

from __future__ import annotations

import pytest

from agents.core.base_agent import BaseAgent, _fold


class _Fake(BaseAgent):
    """Agente de usar y tirar: solo existe para puntuar consultas."""

    name = "fake"
    description = "fake"

    def __init__(self, capabilities, **kwargs):
        super().__init__(**kwargs)
        self.capabilities = capabilities

    def actions(self) -> dict:
        return {}


def _score(context, capabilities: list[str], query: str) -> float:
    return _Fake(capabilities, context=context).can_handle(query)


# -- normalización ------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Documentación", "documentacion"),
        ("ANÁLISIS", "analisis"),
        ("sin tildes", "sin tildes"),
        ("Ñoño", "ñoño"),  # la eñe NO es un acento: es otra letra
    ],
)
def test_fold_quita_tildes_pero_conserva_la_ene(text, expected):
    assert _fold(text) == expected


def test_acento_en_la_consulta_no_impide_el_match(context):
    assert _score(context, ["documentacion"], "revisa la documentación") > 0


def test_acento_en_la_keyword_no_impide_el_match(context):
    assert _score(context, ["análisis"], "haz un analisis del dataset") > 0


def test_palabra_parcial_no_cuenta(context):
    """'ci' dentro de 'dependencias' no debe puntuar (bug real del pasado)."""
    assert _score(context, ["ci"], "actualiza las dependencias") == 0.0


def test_sin_capabilities_no_puntua(context):
    assert _score(context, [], "lo que sea") == 0.0


# -- especificidad ------------------------------------------------------------
def test_la_frase_larga_gana_a_dos_palabras_genericas(context):
    """
    Caso real: "busca en el grafo de conocimiento" debe ir al agente que
    tiene la frase completa, no al que acierta 'grafo' y 'conocimiento' por
    separado.
    """
    query = "busca en el grafo de conocimiento"
    especifico = _score(context, ["busca en el grafo"], query)
    generico = _score(context, ["grafo", "conocimiento"], query)
    assert especifico > generico


def test_mas_palabras_cubiertas_puntua_mas(context):
    query = "haz un tag release del proyecto"
    assert _score(context, ["tag release"], query) > _score(context, ["release"], query)


def test_el_score_esta_acotado_a_uno(context):
    query = "a b c d e f g h i j k l m n o p"
    assert _score(context, query.split(), query) <= 1.0


def test_query_irrelevante_puntua_cero(context):
    assert _score(context, ["docker", "dockerfile"], "entrena el modelo") == 0.0
