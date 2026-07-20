"""
agents.helpers — Funciones auxiliares para agentes.

Incluye el mecanismo canónico de delegación entre agentes (``delegate_to``).
Usa esto EN VEZ de importar directamente la clase de otro agente.
"""

from __future__ import annotations

from typing import Any

from agents.context import get_context
from agents.orchestrator import Orchestrator


_ORCH: Orchestrator | None = None


def _orch() -> Orchestrator:
    global _ORCH
    if _ORCH is None:
        _ORCH = Orchestrator(context=get_context())
    return _ORCH


def delegate_to(agent: str, action: str, **kwargs) -> Any:
    """
    Delega una tarea a otro agente.

    Es el mecanismo canónico de comunicación entre agentes.
    Usa esto en vez de importar ``SomeAgent`` y llamarlo directamente.

    Ejemplo::

        from agents.helpers import delegate_to

        result = delegate_to("git", "analyze_diff")
        files = result.data

        result = delegate_to("documentation", "update_changelog", version="2.0.0")
    """
    result = _orch().run(agent, action, **kwargs)
    if hasattr(result, "data"):
        return result
    return type("AgentResult", (), {"success": False, "data": None, "message": str(result)})()
