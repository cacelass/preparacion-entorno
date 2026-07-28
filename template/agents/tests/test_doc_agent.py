from __future__ import annotations


# -- navegacion del grafo: absorbida de `docsearch` -----------------------------
# `doc.graph_query` y `docsearch.search` consultaban el mismo grafo con cache.
# Se conservo la implementacion de docsearch (no cachea los fallos) y se
# trajeron sus acciones de navegacion.

def test_graph_query_falla_sin_grafo(context):
    from agents.agents.doc_agent import DocAgent

    result = DocAgent(context=context).graph_query(question="test")
    assert not result.success


def test_neighbors_falla_sin_grafo(context):
    from agents.agents.doc_agent import DocAgent

    result = DocAgent(context=context).neighbors(node="algo")
    assert not result.success


def test_list_references_falla_sin_grafo(context):
    from agents.agents.doc_agent import DocAgent

    result = DocAgent(context=context).list_references()
    assert not result.success
