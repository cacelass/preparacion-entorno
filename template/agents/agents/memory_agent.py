"""
agents.agents.memory_agent — Proactive memory agent for long-horizon tasks.

Implements the architecture from "Remember When It Matters: Proactive Memory
Agent for Long-Horizon Agents" (arXiv 2607.08716):

  Phase 1 (observe): Watch agent trajectory via audit log, extract facts, state,
                     and procedural traces into a structured memory bank.

  Phase 2 (intervene): Decide whether to inject a reminder into the action
                      agent's context, based on recency, importance, and
                      relevance of stored memory entries.

The memory agent never modifies other agents — it only writes to its own memory
bank and, when triggered, surfaces context to the orchestrator or CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.memory_tool import MemoryTool


@register_agent
class MemoryAgent(BaseAgent):
    name = "memory"
    description = (
        "Proactive memory: observes agent trajectories, maintains a structured "
        "memory bank (facts, state, traces), and injects context reminders "
        "to combat behavioral state decay in long-horizon tasks."
    )
    capabilities = [
        "memoria", "memory", "recordar", "remember", "olvidar", "forget",
        "contexto", "context", "historial", "history", "traza", "trace",
        "recordatorio", "reminder", "inyectar", "inyección", "inyectar contexto",
    ]

    def actions(self) -> dict:
        return {
            "status": self.status,
            "note": self.note,
            "recall": self.recall,
            "forget": self.forget,
            "memory_edit": self.memory_edit,
            "search": self.search,
            "snapshot": self.snapshot,
            "inject": self.inject,
            "observe": self.observe,
            "decay": self.decay,
            "clear": self.clear,
        }

    # -- workspace ------------------------------------------------------------
    @property
    def _ws(self) -> Path:
        return self.ctx.agent_workspace("memory")

    # -- Phase 1: Observe & store ---------------------------------------------

    def observe(self, *, max_entries: int = 50) -> AgentResult:
        """
        Reads the latest audit log entries and extracts memory from agent
        trajectories: successful actions become traces, failures become
        warnings, decisions become facts.
        """
        audit_dir = self.ctx.workspace_dir / "audit"
        if not audit_dir.exists():
            return AgentResult(True, self.name, "observe", "No hay log de auditoría aún.")

        log_files = sorted(audit_dir.glob("*.jsonl"), reverse=True)
        if not log_files:
            return AgentResult(True, self.name, "observe", "No hay log de auditoría aún.")

        count = 0
        for log_file in log_files[:3]:
            for line in _tail(log_file, max_entries):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._ingest_record(record)
                count += 1

        MemoryTool.decay(self._ws)
        return AgentResult(
            True, self.name, "observe",
            f"Observados {count} registro(s) de auditoría. Memoria actualizada.",
            data={"observed": count, "kinds": MemoryTool.list_kinds(self._ws)},
        )

    def _ingest_record(self, record: dict) -> None:
        agent = record.get("agent", "unknown")
        action = record.get("action", "unknown")
        success = record.get("success", False)
        message = record.get("message", "")
        kwarg_names = record.get("kwarg_names", [])

        key = f"{agent}.{action}"
        if success:
            val = {"message": message, "params": kwarg_names}
            MemoryTool.write(self._ws, "traces", f"{key}:ok", val, ttl=86400 * 30)
        else:
            MemoryTool.write(self._ws, "traces", f"{key}:fail", message, ttl=86400 * 7)
            MemoryTool.write(self._ws, "state", f"warning:{agent}", message, ttl=86400)

    def note(self, *, key: str, value: str, kind: str = "facts", scope: str = "per-proyecto") -> AgentResult:
        kind = kind.strip().lower()
        if kind not in ("facts", "state", "traces"):
            return AgentResult(False, self.name, "note", f"kind inválido: '{kind}' — usa facts, state o traces.")
        try:
            MemoryTool.write(self._ws, kind, key, value, scope=scope)
        except ValueError as exc:
            return AgentResult(False, self.name, "note", str(exc))
        return AgentResult(
            True, self.name, "note",
            f"Memorizado '{kind}:{key}' ({scope}).",
            data={"kind": kind, "key": key, "scope": scope},
        )

    # -- Phase 2: Retrieve & inject -------------------------------------------

    def inject(self, *, context: str | None = None, max_entries: int = 10) -> AgentResult:
        """
        Generates a markdown context block for injection into another agent's
        context. This is Phase 2 of the proactive memory pattern: deciding
        what to surface as a reminder. If `context` is provided, it filters
        memory by relevance to that context.
        """
        ctx = MemoryTool.injectable_context(self._ws, max_entries=max_entries)
        if not ctx:
            return AgentResult(True, self.name, "inject", "No hay memoria relevante para inyectar.", data="")

        if context:
            filtered = self._filter_by_context(ctx, context)
            if not filtered:
                return AgentResult(True, self.name, "inject", "No hay memoria relevante para el contexto dado.", data="")
            ctx = filtered

        return AgentResult(
            True, self.name, "inject",
            f"Inyectando {max_entries} entrada(s) de memoria como contexto.",
            data=ctx,
        )

    def _filter_by_context(self, ctx: str, context: str) -> str:
        lines = ctx.split("\n")
        filtered = []
        context_lower = context.lower()
        keep = False
        for line in lines:
            if line.startswith("### "):
                keep = any(part in context_lower for part in line.lower().split())
                if keep:
                    filtered.append(line)
                continue
            if line.startswith("- **"):
                if keep and context_lower in line.lower():
                    filtered.append(line)
                elif keep:
                    filtered.append(line)
                continue
            if keep:
                filtered.append(line)
        return "\n".join(filtered)

    # -- Query helpers --------------------------------------------------------

    def recall(self, *, key: str) -> AgentResult:
        entry = MemoryTool.recall(self._ws, key)
        if entry is None:
            return AgentResult(False, self.name, "recall", f"'{key}' no encontrado en memoria.")
        return AgentResult(
            True, self.name, "recall",
            f"Encontrado '{key}' ({entry['kind']}).",
            data=entry,
        )

    def search(self, *, query: str, kind: str | None = None, scope: str | None = None, limit: int = 20) -> AgentResult:
        results = MemoryTool.search(self._ws, kind=kind, query=query, scope=scope, limit=limit)
        return AgentResult(
            True, self.name, "search",
            f"{len(results)} resultado(s)." + (f" (scope: {scope})" if scope else ""),
            data=results,
        )

    def forget(self, *, key: str) -> AgentResult:
        if MemoryTool.recall(self._ws, key):
            MemoryTool.delete(self._ws, "facts", key)
            MemoryTool.delete(self._ws, "state", key)
            MemoryTool.delete(self._ws, "traces", key)
            return AgentResult(True, self.name, "forget", f"Olvidado '{key}'.")
        return AgentResult(False, self.name, "forget", f"'{key}' no está en memoria.")

    def memory_edit(self, *, id: str, action: str = "update",
                    value: str | None = None, scope: str | None = None,
                    ttl: int | None = None) -> AgentResult:
        """
        Actualiza, olvida o invalida una entrada de memoria por su id.

        - ``update``: cambia value (y opcionalmente scope/ttl).
        - ``forget``: la borra del banco.
        - ``invalidate``: la expira (ttl=0); no puede resucitar.

        Los subagentes heredan el banco del agente que los lanzó (todos leen
        ``agents/workspace/memory/``), así que editar aquí es visible para todo
        el árbol de ejecución.
        """
        if action not in ("update", "forget", "invalidate"):
            return AgentResult(
                False, self.name, "memory_edit",
                f"Acción inválida '{action}' — usa update, forget o invalidate.",
                needs=["¿Qué acción aplico: update, forget o invalidate?"],
            )

        entry = MemoryTool.recall(self._ws, id)
        if entry is None:
            return AgentResult(False, self.name, "memory_edit", f"'{id}' no está en memoria.")

        if action == "forget":
            MemoryTool.delete(self._ws, entry["kind"], id)
            return AgentResult(True, self.name, "memory_edit", f"Memoria '{id}' olvidada.",
                               data={"id": id, "action": action})

        try:
            if action == "invalidate":
                edited = MemoryTool.edit(self._ws, id, ttl=0)
            else:
                edited = MemoryTool.edit(self._ws, id, value=value, scope=scope, ttl=ttl)
        except ValueError as exc:
            return AgentResult(False, self.name, "memory_edit", str(exc))

        if edited is None:
            return AgentResult(True, self.name, "memory_edit", f"Memoria '{id}' invalidada.",
                               data={"id": id, "action": action})
        return AgentResult(
            True, self.name, "memory_edit",
            f"Memoria '{id}' actualizada.",
            data={"id": id, "action": action, "entry": edited},
        )

    def snapshot(self) -> AgentResult:
        snap = MemoryTool.snapshot(self._ws)
        kinds = MemoryTool.list_kinds(self._ws)
        total = sum(kinds.values())
        return AgentResult(
            True, self.name, "snapshot",
            f"Snapshot de {total} entrada(s) ({', '.join(f'{k}={v}' for k, v in kinds.items())}).",
            data=snap,
        )

    def status(self) -> AgentResult:
        kinds = MemoryTool.list_kinds(self._ws)
        total = sum(kinds.values())
        scopes = self._scope_counts()
        bank_path = MemoryTool.bank_dir(self._ws) / "bank.json"
        size = bank_path.stat().st_size if bank_path.exists() else 0
        return AgentResult(
            True, self.name, "status",
            f"{total} entrada(s) en memoria ({size:,} bytes).",
            data={
                "entries_by_kind": kinds, "total": total,
                "entries_by_scope": scopes, "bank_size_bytes": size,
            },
        )

    def _scope_counts(self) -> dict[str, int]:
        import json

        path = MemoryTool.bank_dir(self._ws) / "bank.json"
        if not path.exists():
            return {}
        try:
            bank = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        counts: dict[str, int] = {}
        for entries in bank.values():
            if not isinstance(entries, dict):
                continue
            for entry in entries.values():
                scope = entry.get("scope", "per-proyecto")
                counts[scope] = counts.get(scope, 0) + 1
        return counts

    def decay(self) -> AgentResult:
        affected = MemoryTool.decay(self._ws)
        return AgentResult(
            True, self.name, "decay",
            f"Aplicado decaimiento a {affected} entrada(s).",
            data={"decayed": affected},
        )

    def clear(self, *, kind: str | None = None) -> AgentResult:
        removed = MemoryTool.clear(self._ws, kind=kind)
        return AgentResult(
            True, self.name, "clear",
            f"Eliminadas {removed} entrada(s)." + (f" (kind: '{kind}')" if kind else ""),
            data={"removed": removed},
        )


def _tail(path: Path, n: int) -> list[str]:
    """Reads the last n lines of a file efficiently."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [line.rstrip("\n\r") for line in lines[-n:]]
    except OSError:
        return []
