"""
agents.cli — CLI para usar el sistema de agentes desde línea de comandos.

Uso:
    uv run python -m agents list
    uv run python -m agents describe git
    uv run python -m agents run git suggest_commit_message
    uv run python -m agents run data eda_report --filename dataset.csv --target-col target
    uv run python -m agents ask "genera el changelog desde el último tag"
    uv run python -m agents --json ask "revisa el Dockerfile"   # salida JSON para herramientas/agentes
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from agents.context import get_context
from agents.core.base_agent import AgentResult
from agents.core.registry import agent_registry
from agents.orchestrator import Orchestrator, RoutingDecision


def _print_result(result: AgentResult, *, json_mode: bool = False) -> None:
    if json_mode:
        print(json.dumps({
            "success": result.success,
            "agent": result.agent,
            "action": result.action,
            "message": result.message,
            "data": result.data,
            "warnings": result.warnings,
            "needs": result.needs,
        }, indent=2, ensure_ascii=False, default=str))
        return
    status = "✔" if result.success else "✘"
    print(f"{status} [{result.agent}.{result.action}] {result.message}")
    for warning in result.warnings:
        print(f"  ⚠ {warning}")
    if result.data is not None:
        try:
            print(json.dumps(result.data, indent=2, ensure_ascii=False, default=str))
        except TypeError:
            print(result.data)


def _print_routing(decision: RoutingDecision, *, json_mode: bool = False) -> None:
    if json_mode:
        print(json.dumps({
            "query": decision.query,
            "selected_agent": decision.agent_name,
            "confidence": round(decision.confidence, 3),
            "candidates": [(name, round(score, 3)) for name, score in decision.candidates],
        }, indent=2, ensure_ascii=False))
        return
    print(f"  Ruteo: '{decision.query}' → {decision.agent_name or 'ninguno'} "
          f"(confianza {decision.confidence:.2f})")
    print(f"  Candidatos: {[(n, f'{s:.2f}') for n, s in decision.candidates[:5]]}")


def _parse_kwargs(pairs: list[str]) -> dict[str, Any]:
    """Convierte ['--target-col', 'target', '--max-count', '10'] en {'target_col': 'target', 'max_count': 10}."""
    kwargs: dict[str, Any] = {}
    key = None
    for token in pairs:
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            kwargs[key] = True
        elif key is not None:
            value: Any = token
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.replace(".", "", 1).lstrip("-").isdigit():
                value = float(value) if "." in value else int(value)
            kwargs[key] = value
            key = None
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agents", description="Sistema de agentes del proyecto.")
    parser.add_argument("--json", "-j", action="store_true", help="Salida en JSON estructurado (para consumo por herramientas/agentes).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="Lista todos los agentes registrados.")

    describe_p = subparsers.add_parser("describe", help="Detalle de un agente: acciones y capacidades.")
    describe_p.add_argument("agent_name")

    run_p = subparsers.add_parser("run", help="Ejecuta una acción concreta de un agente.")
    run_p.add_argument("agent_name")
    run_p.add_argument("action")
    run_p.add_argument("kwargs", nargs=argparse.REMAINDER, help="--kwarg valor --otro-kwarg valor")

    ask_p = subparsers.add_parser("ask", help="Rutea una petición en lenguaje natural al agente más relevante.")
    ask_p.add_argument("query")

    pipeline_p = subparsers.add_parser("pipeline", help="Ejecuta un pipeline predefinido (develop|fix|release|cycle|analyze|data).")
    pipeline_p.add_argument("name", choices=["develop", "fix", "release", "cycle", "analyze", "data"])
    pipeline_p.add_argument("params", nargs=argparse.REMAINDER, help="--version 1.0.0 --filename data.csv")

    doctor_p = subparsers.add_parser("doctor", help="Diagnóstico completo del proyecto: entorno, código, tests, datos.")
    doctor_p.add_argument("--fix", action="store_true", help="Intenta corregir problemas automáticamente (usa auto_fix pipeline)")

    plan_p = subparsers.add_parser(
        "plan",
        help="Describe un encargo: el agente 'plan' lo descompone, pregunta lo que falte y delega. "
             "(atajo de `run plan intake --brief ...`; responde/ejecuta con `run plan answer/execute`)",
    )
    plan_p.add_argument("brief", help="El encargo en lenguaje natural (un paso por línea o separado por ';')")

    audit_p = subparsers.add_parser("audit", help="Audita al equipo de agentes con el log de ejecuciones.")
    audit_p.add_argument(
        "what", nargs="?", default="report", choices=["report", "failures", "suggest"],
        help="report (uso y tasas de éxito) | failures (fallos recientes) | suggest (mejoras propuestas)",
    )

    subparsers.add_parser("tools", help="Lista las herramientas registradas.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    json_mode = args.json
    orchestrator = Orchestrator(context=get_context())

    if args.command == "list":
        agents = orchestrator.list_agents()
        if json_mode:
            print(json.dumps(agents, indent=2, ensure_ascii=False))
        else:
            for info in agents:
                print(f"- {info['name']}: {info['description']}")
        return 0

    if args.command == "describe":
        agent_registry.discover()
        if args.agent_name not in agent_registry.all():
            msg = f"No existe el agente '{args.agent_name}'. Disponibles: {sorted(agent_registry.all())}"
            if json_mode:
                print(json.dumps({"error": msg, "available": sorted(agent_registry.all())}))
            else:
                print(msg)
            return 1
        agent = orchestrator._get_instance(args.agent_name)
        info = agent.describe()
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    if args.command == "run":
        kwargs = _parse_kwargs(args.kwargs)
        result = orchestrator.run(args.agent_name, args.action, **kwargs)
        _print_result(result, json_mode=json_mode)
        return 0 if result.success else 1

    if args.command == "ask":
        if json_mode:
            decision = orchestrator.select_agent(args.query)
            _print_routing(decision, json_mode=True)
        result = orchestrator.dispatch(args.query)
        _print_result(result, json_mode=json_mode)
        return 0 if result.success else 1

    if args.command == "pipeline":
        from agents.gstack.pipelines import run_pipeline
        pipe_kwargs = _parse_kwargs(args.params)
        result = run_pipeline(args.name, **pipe_kwargs)
        if json_mode:
            print(json.dumps({
                "success": result.success,
                "summary": result.summary,
                "steps": result.steps if hasattr(result, 'steps') else [],
            }, indent=2, ensure_ascii=False))
        else:
            print(result.summary)
        return 0 if result.success else 1

    if args.command == "doctor":
        if args.fix:
            from agents.gstack.pipelines import auto_fix
            result = auto_fix(auto_commit=True)
        else:
            from agents.gstack.pipelines import auto_analyze
            result = auto_analyze()
        if json_mode:
            print(json.dumps({
                "success": result.success,
                "summary": result.summary,
                "sections": result.sections if hasattr(result, 'sections') else [],
            }, indent=2, ensure_ascii=False))
        else:
            print(result.summary)
        return 0 if result.success else 1

    if args.command == "plan":
        result = orchestrator.run("plan", "intake", brief=args.brief)
        _print_result(result, json_mode=json_mode)
        for question in result.needs:
            if json_mode:
                continue
            print(f"  ? {question}")
        return 0 if result.success else 1

    if args.command == "audit":
        action = {"report": "report", "failures": "failures", "suggest": "suggest_improvements"}[args.what]
        result = orchestrator.run("audit", action)
        _print_result(result, json_mode=json_mode)
        return 0 if result.success else 1

    if args.command == "tools":
        from agents.tools.registry import tool_registry
        agent_registry.discover()
        names = sorted(tool_registry.all())
        if json_mode:
            print(json.dumps(names, indent=2))
        else:
            for name in names:
                print(f"- {name}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
