"""
agents.evals.runner — Eval runner: ejecuta baterías de pruebas contra el sistema de agentes.

Uso:
    uv run python -m agents.evals.runner              # harness + smoke + routing + contracts
    uv run python -m agents.evals.runner --smoke       # solo smoke
    uv run python -m agents.evals.runner --routing     # solo routing
    uv run python -m agents.evals.runner --contracts   # solo contracts
    uv run python -m agents.evals.runner --harness     # solo piezas del arnés
    uv run python -m agents.evals.runner --json        # reporte JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from agents.context import get_context
from agents.core.base_agent import AgentResult
from agents.core.registry import agent_registry
from agents.orchestrator import Orchestrator


EvalReport = dict[str, Any]


def _smoke(orchestrator: Orchestrator) -> list[dict]:
    """
    Smoke test: cada agente ejecuta una acción sin argumentos obligatorios.
    Verifica que responde sin lanzar excepción.

    Se elige la primera acción auto-ejecutable, no la primera a secas: si la
    primera necesita un argumento (p. ej. `schedule.validate_cron(expression=)`)
    el smoke fallaba siempre con un TypeError que no dice nada sobre la salud
    del agente, solo sobre el orden de su diccionario `actions()`.
    """
    results = []
    for name in sorted(agent_registry.all()):
        agent = orchestrator._get_instance(name)
        actions = agent.actions()
        if not actions:
            continue
        primary = next(
            (a for a in actions if agent.can_auto_run(a)),
            None,
        )
        if primary is None:
            results.append({
                "agent": name,
                "action": "(ninguna)",
                "success": True,
                "duration_ms": 0,
                "message": "sin acciones ejecutables sin argumentos — nada que probar en smoke",
            })
            continue
        start = time.monotonic()
        try:
            result = agent.run(primary)
            duration = time.monotonic() - start
            results.append({
                "agent": name,
                "action": primary,
                "success": result.success,
                "duration_ms": round(duration * 1000),
                "message": result.message[:200],
            })
        except Exception as exc:
            duration = time.monotonic() - start
            results.append({
                "agent": name,
                "action": primary,
                "success": False,
                "duration_ms": round(duration * 1000),
                "message": f"EXCEPTION: {exc}",
            })
    return results


ROUTING_BENCHMARKS: list[tuple[str, str]] = [
    ("haz commit de los cambios", "git"),
    ("sugiere un mensaje de commit", "git"),
    ("ejecuta los tests", "test"),
    ("revisa el Dockerfile", "docker"),
    ("diagnóstico del proyecto", "doctor"),
    ("actualiza el changelog", "documentation"),
    ("analiza el dataset clientes.csv", "data"),
    ("revisa el código por duplicación", "review"),
    ("actualiza dependencias", "dependency"),
    ("escanea secretos hardcodeados", "secrets"),
    ("genera un workflow de CI", "cicd"),
    ("valida el Makefile", "make"),
    ("extrae las celdas del notebook", "notebook"),
    ("refactoriza los type hints", "refactor"),
    ("describe la expresión cron", "schedule"),
    ("audita al equipo de agentes", "audit"),
    ("busca papers sobre transformers", "research"),
    ("instala un agente externo", "installer"),
    ("sincroniza el grafo de conocimiento", "knowledge"),
    ("busca en el grafo de conocimiento", "docsearch"),
    ("verifica el estado de MLflow", "mlflow"),
    ("comprueba los endpoints de la API", "api"),
    ("compite dos estrategias", "supervisor"),
    ("verifica las figuras del reporte", "graph"),
    ("comprueba el entorno Python", "env"),
    ("guarda esto en la memoria", "memory"),
    ("planea el proyecto", "plan"),
    ("entrena el modelo", "ml"),
    ("busca en la documentación del proyecto", "rag"),
    ("indexa el código fuente", "rag"),
    ("busca en todas las fuentes de documentación", "doc"),
    ("dónde está documentado el módulo de datos", "doc"),
    ("cuál es la siguiente tarea pendiente del backlog", "harness"),
    ("abre la feature del arnés", "harness"),
]


def _routing(orchestrator: Orchestrator) -> list[dict]:
    """
    Routing test: frases conocidas deben rutear al agente correcto.
    """
    results = []
    for query, expected_agent in ROUTING_BENCHMARKS:
        start = time.monotonic()
        decision = orchestrator.select_agent(query)
        duration = time.monotonic() - start
        correct = decision.agent_name == expected_agent
        results.append({
            "query": query,
            "expected": expected_agent,
            "got": decision.agent_name,
            "confidence": round(decision.confidence, 3),
            "correct": correct,
            "duration_ms": round(duration * 1000),
        })
    return results


def _contracts() -> list[dict]:
    """
    Contract test: verifica que todos los agentes tienen contrato válido
    y no hay colisiones de recursos.
    """
    from agents.contracts import CONTRACTS, validate_contracts
    agent_registry.discover()
    problems = validate_contracts(set(agent_registry.all()))
    return [
        {"agent": c.role, "can": list(c.can), "cannot": list(c.cannot),
         "owns": list(c.owns), "collaborates": list(c.collaborates)}
        for c in CONTRACTS.values()
    ] + ([{"warning": p} for p in problems] if problems else [])


HARNESS_AGENTS = ("lider", "implementer", "reviewer", "explorer")
HARNESS_STATUS = ("pending", "in_progress", "done", "blocked")
HARNESS_REQUIRED = ("id", "title", "description", "acceptance_criteria", "status")


def _harness() -> list[dict]:
    """
    Harness test: verifica que las piezas del arnés existen y son coherentes.
    Es la contraparte en Python de lo que comprueba ./init.sh (ver AGENTS.md).
    """
    from pathlib import Path

    root: Path = get_context().root
    results: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append({"agent": name, "success": ok, "message": detail})

    gate = root / "init.sh"
    check("init.sh", gate.is_file(), "presente" if gate.is_file() else "FALTA la puerta del arnés")
    if gate.is_file():
        import os
        executable = os.access(gate, os.X_OK)
        check("init.sh:+x", executable, "ejecutable" if executable else "sin bit de ejecución (chmod +x init.sh)")

    for rel in ("AGENTS.md", "progress/current.md", "progress/history.md", "progress/README.md"):
        path = root / rel
        check(rel, path.is_file(), "presente" if path.is_file() else "FALTA")

    for name in HARNESS_AGENTS:
        path = root / ".opencode" / "agents" / f"{name}.md"
        check(f"agente:{name}", path.is_file(), "definido" if path.is_file() else "FALTA su definición")

    workflow = root / "agents" / "prompts" / "harness_workflow.md"
    check("harness_workflow", workflow.is_file(),
          "skill presente" if workflow.is_file() else "FALTA agents/prompts/harness_workflow.md")

    agents_md = root / "AGENTS.md"
    if agents_md.is_file():
        text = agents_md.read_text(encoding="utf-8", errors="replace")
        has_protocol = "Protocolo del arnés" in text and "init.sh" in text
        check("AGENTS.md:protocolo", has_protocol,
              "protocolo documentado" if has_protocol else "no documenta el protocolo del arnés")

    backlog = root / "featureslist.json"
    if not backlog.is_file():
        check("featureslist.json", False, "FALTA el backlog")
        return results

    try:
        doc = json.loads(backlog.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        check("featureslist.json", False, f"JSON inválido: {exc}")
        return results

    features = doc.get("features") if isinstance(doc, dict) else None
    if not isinstance(features, list) or not features:
        check("featureslist.json", False, "falta la clave 'features' con una lista no vacía")
        return results

    problems: list[str] = []
    ids: set[str] = set()
    for i, feat in enumerate(features):
        if not isinstance(feat, dict):
            problems.append(f"feature #{i} no es un objeto")
            continue
        missing = [k for k in HARNESS_REQUIRED if k not in feat]
        if missing:
            problems.append(f"feature #{i} sin campos: {', '.join(missing)}")
            continue
        if feat["id"] in ids:
            problems.append(f"id duplicado: {feat['id']}")
        ids.add(feat["id"])
        if feat["status"] not in HARNESS_STATUS:
            problems.append(f"{feat['id']}: status '{feat['status']}' no válido")
        if not isinstance(feat["acceptance_criteria"], list) or not feat["acceptance_criteria"]:
            problems.append(f"{feat['id']}: acceptance_criteria vacío o no es lista")

    for feat in features:
        if isinstance(feat, dict):
            for dep in feat.get("depends_on", []):
                if dep not in ids:
                    problems.append(f"{feat.get('id')}: depends_on '{dep}' no existe")

    running = [f["id"] for f in features if isinstance(f, dict) and f.get("status") == "in_progress"]
    if len(running) > 1:
        problems.append(f"{len(running)} features in_progress a la vez: {', '.join(running)}")

    if problems:
        for problem in problems:
            check("featureslist.json", False, problem)
    else:
        check("featureslist.json", True, f"{len(features)} features, esquema válido")

    return results


def _summarize(results: list[dict], label: str) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.get("success", r.get("correct", True)))
    avg_duration = sum(r.get("duration_ms", 0) for r in results) / max(total, 1)
    return {
        "suite": label,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "avg_duration_ms": round(avg_duration, 1),
        "results": results,
    }


def run_evals(smoke: bool = True, routing: bool = True, contracts: bool = True,
              harness: bool = True, json_output: bool = False) -> EvalReport:
    orchestrator = Orchestrator(context=get_context())
    agent_registry.discover()

    suites = {}
    if harness:
        suites["harness"] = _summarize(_harness(), "harness")
    if smoke:
        suites["smoke"] = _summarize(_smoke(orchestrator), "smoke")
    if routing:
        suites["routing"] = _summarize(_routing(orchestrator), "routing")
    if contracts:
        contract_results = _contracts()
        # contracts returns metadata + warnings; count warnings as failures
        warnings = [r for r in contract_results if "warning" in r]
        suites["contracts"] = {
            "suite": "contracts",
            "total": len(contract_results),
            "contracts": len(contract_results) - len(warnings),
            "warnings": len(warnings),
            "results": contract_results,
        }

    overall_pass = all(
        s.get("failed", 0) == 0 and s.get("warnings", 0) == 0
        for s in suites.values()
    )

    report: EvalReport = {
        "success": overall_pass,
        "timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "suites": suites,
    }

    if json_output:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        _print_report(report)

    return report


def _print_report(report: EvalReport) -> None:
    print(f"\n{'='*50}")
    print(f"  EVAL REPORT  {'✔' if report['success'] else '✘'}")
    print(f"{'='*50}")
    for name, suite in report["suites"].items():
        status = "✔" if suite.get("failed", 0) == 0 else "✘"
        print(f"\n  {status} {name.upper()}")
        print(f"     {suite.get('passed', 0)}/{suite['total']} passed"
              f"  |  avg {suite.get('avg_duration_ms', 0)}ms")
        for r in suite.get("results", []):
            if not r.get("success", r.get("correct", True)):
                print(f"     ✘ {r.get('agent', r.get('expected', '?'))}: "
                      f"{r.get('message', r.get('query', ''))[:120]}")
            if "warning" in r:
                print(f"     ⚠ {r['warning']}")
    print(f"\n{'='*50}")
    print(f"  {'PASS' if report['success'] else 'FAIL'}")
    print(f"{'='*50}\n")


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m agents.evals.runner")
    parser.add_argument("--smoke", action="store_true", help="Solo smoke tests")
    parser.add_argument("--routing", action="store_true", help="Solo routing tests")
    parser.add_argument("--contracts", action="store_true", help="Solo contracts")
    parser.add_argument("--harness", action="store_true", help="Solo harness (piezas del arnés)")
    parser.add_argument("--json", "-j", action="store_true", help="Salida JSON")
    args = parser.parse_args()

    selected = args.smoke or args.routing or args.contracts or args.harness

    report = run_evals(
        smoke=args.smoke or not selected,
        routing=args.routing or not selected,
        contracts=args.contracts or not selected,
        harness=args.harness or not selected,
        json_output=args.json,
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
