"""
agents.evals.runner — Eval runner: ejecuta baterías de pruebas contra el sistema de agentes.

Uso:
    uv run python -m agents.evals.runner              # smoke + routing + contracts
    uv run python -m agents.evals.runner --smoke       # solo smoke
    uv run python -m agents.evals.runner --routing     # solo routing
    uv run python -m agents.evals.runner --contracts   # solo contracts
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
    Smoke test: cada agente ejecuta su acción principal.
    Verifica que responde sin lanzar excepción.
    """
    results = []
    for name in sorted(agent_registry.all()):
        agent = orchestrator._get_instance(name)
        actions = agent.actions()
        if not actions:
            continue
        primary = list(actions.keys())[0]
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
              json_output: bool = False) -> EvalReport:
    orchestrator = Orchestrator(context=get_context())
    agent_registry.discover()

    suites = {}
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
    parser.add_argument("--json", "-j", action="store_true", help="Salida JSON")
    args = parser.parse_args()

    run_smoke = args.smoke or not (args.routing or args.contracts)
    run_routing = args.routing or not (args.smoke or args.contracts)
    run_contracts = args.contracts or not (args.smoke or args.routing)

    report = run_evals(
        smoke=run_smoke,
        routing=run_routing,
        contracts=run_contracts,
        json_output=args.json,
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
