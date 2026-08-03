"""
agents.evals.rag_eval — Evaluación de la recuperación del RAG.

Por qué existe
--------------
El RAG tiene parámetros que se tocan a ojo: el troceado, el embedder, el peso
del híbrido, el umbral. Sin una medida, cambiar cualquiera de ellos es fe:
«parece que ahora encuentra mejor». Esto convierte esa sensación en dos
números que se pueden comparar entre commits.

Cómo funciona
-------------
`rag_golden.json` es el juego de pruebas: cada caso es una pregunta en
lenguaje natural y las fuentes que deberían aparecer al buscarla. Se ejecuta
la búsqueda real contra el índice del proyecto y se mide:

- **hit_rate** — fracción de preguntas que devuelven alguna fuente esperada.
- **recall@k** — de las fuentes esperadas, cuántas aparecen en el top-k.
- **MRR** — 1/posición de la primera fuente correcta. Distingue acertar el
  primero de acertar el quinto, que `hit_rate` no ve.
- **lexical_share** — qué fracción de los aciertos entra por BM25. Es lo que
  dice si el híbrido se está ganando el sitio o solo cuesta tiempo.

Se mide en modo híbrido y en modo solo-vector, porque la comparación entre
ambos es el dato accionable: si el vector solo empata al híbrido, sobra
media herramienta.

El juego de pruebas es editable a mano — está pensado para crecer con las
preguntas que en tu proyecto real devuelven basura.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.tools.rag_tool import RagTool

#: Cuántos resultados se piden por pregunta. 5 y no 10 a propósito: lo que se
#: mide es lo que un agente va a leer de verdad, no lo que el índice contiene
#: en algún lugar del top-50.
TOP_K = 5

GOLDEN_FILE = "rag_golden.json"


#: Línea por debajo de la cual la recuperación se considera rota. No es un
#: pleno a propósito: ver `_thresholds` en el JSON.
UMBRALES = {"min_hit_rate": 0.5, "min_mrr": 0.2}


def _leer_golden(ruta: Path | None = None) -> dict[str, Any]:
    destino = ruta or (Path(__file__).parent / GOLDEN_FILE)
    if not destino.exists():
        return {}
    try:
        doc = json.loads(destino.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {"cases": doc}


def cargar_golden(ruta: Path | None = None) -> list[dict[str, Any]]:
    """Lee el juego de pruebas. Formato: [{query, expected:[...], require}]."""
    casos = _leer_golden(ruta).get("cases", [])
    return [c for c in casos if isinstance(c, dict) and c.get("query")]


def cargar_umbrales(ruta: Path | None = None) -> dict[str, float]:
    """Umbrales del veredicto, con los de por defecto como red de seguridad."""
    return {**UMBRALES, **(_leer_golden(ruta).get("thresholds") or {})}


def _acierta(esperadas: list[str], fuentes: list[str], modo: str) -> bool:
    """`any`: basta una de las esperadas. `all`: tienen que estar todas."""
    encontradas = {e for e in esperadas if any(f.startswith(e) for f in fuentes)}
    return len(encontradas) == len(esperadas) if modo == "all" else bool(encontradas)


def _posicion_primer_acierto(esperadas: list[str], fuentes: list[str]) -> int | None:
    for i, fuente in enumerate(fuentes):
        if any(fuente.startswith(e) for e in esperadas):
            return i + 1
    return None


def _evaluar_caso(root: Path, caso: dict, top_k: int, hybrid: bool) -> dict[str, Any]:
    esperadas = [e for e in caso.get("expected", []) if e]
    resultados = RagTool.search(root, caso["query"], top_k=top_k, hybrid=hybrid)
    if resultados and "error" in resultados[0]:
        return {
            "query": caso["query"], "success": False, "error": resultados[0]["error"],
            "message": resultados[0]["error"],
        }

    fuentes = [r["source"] for r in resultados]
    modo = caso.get("require", "any")
    acierto = _acierta(esperadas, fuentes, modo)
    posicion = _posicion_primer_acierto(esperadas, fuentes)
    cubiertas = sum(1 for e in esperadas if any(f.startswith(e) for f in fuentes))
    lexicos = sum(1 for r in resultados if r.get("match") in ("lexico", "ambos"))

    return {
        "query": caso["query"],
        "expected": esperadas,
        "got": fuentes,
        "success": acierto,
        "rank": posicion,
        "recall": round(cubiertas / len(esperadas), 3) if esperadas else 0.0,
        "reciprocal_rank": round(1 / posicion, 3) if posicion else 0.0,
        "lexical_hits": lexicos,
        "returned": len(resultados),
        "message": (
            f"top-{top_k}: {'ok' if acierto else 'FALLA'}"
            + (f" (pos {posicion})" if posicion else "")
            + f" — esperado {esperadas}, devuelto {fuentes[:3]}"
        ),
    }


def _agregar(casos: list[dict], top_k: int) -> dict[str, Any]:
    total = len(casos)
    if not total:
        return {"cases": 0}
    devueltos = sum(c.get("returned", 0) for c in casos)
    return {
        "cases": total,
        "hit_rate": round(sum(1 for c in casos if c.get("success")) / total, 3),
        "recall_at_k": round(sum(c.get("recall", 0.0) for c in casos) / total, 3),
        "mrr": round(sum(c.get("reciprocal_rank", 0.0) for c in casos) / total, 3),
        "lexical_share": round(
            sum(c.get("lexical_hits", 0) for c in casos) / devueltos, 3
        ) if devueltos else 0.0,
        "top_k": top_k,
    }


def evaluate(root: Path, *, top_k: int = TOP_K,
             golden: list[dict] | None = None) -> dict[str, Any]:
    """
    Ejecuta el juego de pruebas contra el índice real del proyecto.

    Devuelve `{available, reason}` en vez de métricas cuando no se puede
    medir (sin chromadb, sin índice o sin casos): un cero en esas condiciones
    no significaría que el RAG recupera mal, y confundir «no medido» con
    «mide cero» es la forma más fácil de volver inútil una métrica.
    """
    casos = golden if golden is not None else cargar_golden()
    if not RagTool.available():
        return {"available": False, "reason": "chromadb no instalado (uv sync --extra rag)"}
    if not casos:
        return {"available": False, "reason": f"sin casos en {GOLDEN_FILE}"}

    estado = RagTool.status(root)
    if not estado.get("available") or not estado.get("total_chunks"):
        return {"available": False, "reason": "índice vacío: ejecuta 'make index-rag'"}

    hibrido = [_evaluar_caso(root, c, top_k, True) for c in casos]
    vectorial = [_evaluar_caso(root, c, top_k, False) for c in casos]

    return {
        "available": True,
        "index_up_to_date": estado.get("up_to_date", True),
        "embedder": estado.get("embedder"),
        "total_chunks": estado.get("total_chunks"),
        "hybrid": _agregar(hibrido, top_k),
        "vector_only": _agregar(vectorial, top_k),
        "cases": hibrido,
    }


def suite(root: Path, *, top_k: int = TOP_K) -> dict[str, Any]:
    """
    Adaptador para `agents.evals.runner`: devuelve la suite ya resumida.

    Dos decisiones que no son obvias:

    1. **No poder medir no es fallar.** Un proyecto recién generado no tiene
       índice todavía; hacer que su CI arranque en rojo por eso enseña a
       ignorar la suite.
    2. **El veredicto va por umbral, no por pleno.** Los casos que fallan se
       ven uno a uno —son el mapa de dónde mejorar—, pero lo que pone la
       suite en rojo es caer por debajo de la línea. Exigir 12/12 obligaría a
       escribir un juego de pruebas fácil, que es justo lo contrario de lo
       que sirve.
    """
    informe = evaluate(root, top_k=top_k)
    if not informe.get("available"):
        return {
            "suite": "rag", "total": 1, "passed": 1, "failed": 0, "avg_duration_ms": 0,
            "results": [{"agent": "rag", "success": True,
                         "message": f"no evaluado — {informe.get('reason')}"}],
        }

    filas = [
        {"agent": f"rag:{c['query'][:40]}", "success": c["success"], "message": c["message"]}
        for c in informe["cases"]
    ]
    h, v = informe["hybrid"], informe["vector_only"]
    umbrales = cargar_umbrales()
    cumple = h["hit_rate"] >= umbrales["min_hit_rate"] and h["mrr"] >= umbrales["min_mrr"]

    filas.append({
        "agent": "rag:resumen", "success": cumple,
        "message": (
            f"híbrido hit={h['hit_rate']} recall={h['recall_at_k']} mrr={h['mrr']} "
            f"| solo-vector hit={v['hit_rate']} recall={v['recall_at_k']} mrr={v['mrr']} "
            f"| aporte léxico {h['lexical_share']} "
            f"| umbral hit>={umbrales['min_hit_rate']} mrr>={umbrales['min_mrr']}: "
            f"{'OK' if cumple else 'POR DEBAJO'}"
        ),
    })
    if not informe.get("index_up_to_date", True):
        filas.append({
            "agent": "rag:frescura", "success": True,
            "message": "el índice está desfasado: las métricas miden contenido viejo",
        })

    aciertos = sum(1 for c in informe["cases"] if c["success"])
    return {
        "suite": "rag",
        "total": len(informe["cases"]),
        "passed": aciertos,
        # Solo el umbral tiñe la suite de rojo. Ver el docstring.
        "failed": 0 if cumple else len(informe["cases"]) - aciertos,
        "avg_duration_ms": 0,
        "results": filas,
        "metrics": {"hybrid": h, "vector_only": v, "thresholds": umbrales},
    }
