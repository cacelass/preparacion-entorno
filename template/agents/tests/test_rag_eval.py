"""
Tests de la evaluación de recuperación (`agents/evals/rag_eval.py`).

La aritmética de las métricas se prueba sin chromadb a propósito: si el único
test de una métrica necesita el índice montado, deja de ejecutarse en cuanto
alguien genera el proyecto sin el extra `rag` — que es justo cuando un error
de cálculo pasaría desapercibido.
"""

from __future__ import annotations

import json

from agents.evals import rag_eval


# -- aciertos ------------------------------------------------------------------
def test_any_basta_con_una_fuente_esperada():
    assert rag_eval._acierta(["a.md", "b.md"], ["z.md", "a.md"], "any")


def test_all_exige_todas():
    assert not rag_eval._acierta(["a.md", "b.md"], ["a.md"], "all")
    assert rag_eval._acierta(["a.md", "b.md"], ["a.md", "b.md"], "all")


def test_la_comparacion_es_por_prefijo():
    """'harness/progress/' debe valer para cualquier fichero de esa carpeta."""
    assert rag_eval._acierta(["harness/progress/"], ["harness/progress/history.md"], "any")


def test_sin_aciertos_no_hay_posicion():
    assert rag_eval._posicion_primer_acierto(["a.md"], ["b.md", "c.md"]) is None


def test_la_posicion_es_1_indexada():
    assert rag_eval._posicion_primer_acierto(["b.md"], ["a.md", "b.md"]) == 2


# -- agregación ----------------------------------------------------------------
def test_agregar_promedia_las_metricas():
    casos = [
        {"success": True, "recall": 1.0, "reciprocal_rank": 1.0, "lexical_hits": 2, "returned": 4},
        {"success": False, "recall": 0.0, "reciprocal_rank": 0.0, "lexical_hits": 0, "returned": 4},
    ]
    m = rag_eval._agregar(casos, top_k=5)
    assert m["cases"] == 2
    assert m["hit_rate"] == 0.5
    assert m["recall_at_k"] == 0.5
    assert m["mrr"] == 0.5
    assert m["lexical_share"] == 0.25


def test_agregar_sin_casos_no_divide_por_cero():
    assert rag_eval._agregar([], top_k=5) == {"cases": 0}


def test_agregar_sin_resultados_devueltos_no_divide_por_cero():
    casos = [{"success": False, "recall": 0.0, "reciprocal_rank": 0.0,
              "lexical_hits": 0, "returned": 0}]
    assert rag_eval._agregar(casos, top_k=5)["lexical_share"] == 0.0


# -- juego de pruebas y umbrales -----------------------------------------------
def test_el_golden_del_template_es_json_valido_y_tiene_casos():
    casos = rag_eval.cargar_golden()
    assert casos, "el template debe traer un juego de pruebas de partida"
    for caso in casos:
        assert caso.get("query")
        assert caso.get("expected"), f"{caso['query']} no declara fuentes esperadas"
        assert caso.get("require", "any") in ("any", "all")


def test_los_umbrales_salen_del_fichero():
    umbrales = rag_eval.cargar_umbrales()
    assert 0.0 <= umbrales["min_hit_rate"] <= 1.0
    assert 0.0 <= umbrales["min_mrr"] <= 1.0


def test_un_golden_ilegible_no_revienta(tmp_path):
    roto = tmp_path / "roto.json"
    roto.write_text("{esto no es json", encoding="utf-8")
    assert rag_eval.cargar_golden(roto) == []
    # y los umbrales caen a los de por defecto, no a nada
    assert rag_eval.cargar_umbrales(roto) == rag_eval.UMBRALES


def test_un_golden_sin_umbrales_usa_los_de_por_defecto(tmp_path):
    parcial = tmp_path / "parcial.json"
    parcial.write_text(json.dumps({"cases": [], "thresholds": {"min_mrr": 0.9}}), encoding="utf-8")
    umbrales = rag_eval.cargar_umbrales(parcial)
    assert umbrales["min_mrr"] == 0.9
    assert umbrales["min_hit_rate"] == rag_eval.UMBRALES["min_hit_rate"]


# -- la suite no puede tumbar un proyecto sin índice ---------------------------
def test_sin_poder_medir_la_suite_sale_en_verde(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_eval, "evaluate",
                        lambda *a, **k: {"available": False, "reason": "índice vacío"})
    suite = rag_eval.suite(tmp_path)
    assert suite["failed"] == 0
    assert "no evaluado" in suite["results"][0]["message"]


def test_por_debajo_del_umbral_la_suite_falla(tmp_path, monkeypatch):
    informe = {
        "available": True, "index_up_to_date": True,
        "hybrid": {"cases": 2, "hit_rate": 0.0, "recall_at_k": 0.0, "mrr": 0.0,
                   "lexical_share": 0.0, "top_k": 5},
        "vector_only": {"cases": 2, "hit_rate": 0.0, "recall_at_k": 0.0, "mrr": 0.0,
                        "lexical_share": 0.0, "top_k": 5},
        "cases": [
            {"query": "a", "success": False, "message": "FALLA"},
            {"query": "b", "success": False, "message": "FALLA"},
        ],
    }
    monkeypatch.setattr(rag_eval, "evaluate", lambda *a, **k: informe)
    suite = rag_eval.suite(tmp_path)
    assert suite["failed"] == 2
    assert any("POR DEBAJO" in r["message"] for r in suite["results"])


def test_por_encima_del_umbral_los_fallos_sueltos_no_tinen_la_suite(tmp_path, monkeypatch):
    """
    Un caso que falla se ve, pero no pone la suite en rojo mientras la media
    aguante: si exigiera pleno, el juego de pruebas tendería a ser fácil.
    """
    informe = {
        "available": True, "index_up_to_date": True,
        "hybrid": {"cases": 4, "hit_rate": 0.75, "recall_at_k": 0.75, "mrr": 0.6,
                   "lexical_share": 0.5, "top_k": 5},
        "vector_only": {"cases": 4, "hit_rate": 0.5, "recall_at_k": 0.5, "mrr": 0.4,
                        "lexical_share": 0.0, "top_k": 5},
        "cases": [
            {"query": "a", "success": True, "message": "ok"},
            {"query": "b", "success": True, "message": "ok"},
            {"query": "c", "success": True, "message": "ok"},
            {"query": "d", "success": False, "message": "FALLA"},
        ],
    }
    monkeypatch.setattr(rag_eval, "evaluate", lambda *a, **k: informe)
    suite = rag_eval.suite(tmp_path)
    assert suite["failed"] == 0
    assert suite["passed"] == 3
    assert any(not r["success"] for r in suite["results"]), "el fallo suelto debe verse"
