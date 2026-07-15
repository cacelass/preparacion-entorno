"""
agents.tools.memory_tool — Memory bank: structured persistent storage for agent state.

Follows the proactive memory pattern from "Remember When It Matters" (arXiv 2607.08716):
a separate memory store watches agent execution and decides when to inject reminders,
rather than waiting for passive retrieval.

The memory bank keeps three kinds of entries:
  - facts: persistent knowledge about the project (architecture, decisions, conventions)
  - state: current session state (ongoing tasks, last actions, agent positions)
  - traces: procedural history (what worked, what failed, patterns)

Each entry has a decay score: the memory agent uses this to decide what to
forget, what to reinforce, and what to inject as a reminder.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agents.tools.registry import register_tool


MEMORY_DIRNAME = "memory"
DECAY_INTERVAL = 3600  # 1 hour in seconds
DEFAULT_TTL = 86400 * 7  # 1 week
MAX_ENTRIES = 500


@register_tool("memory")
class MemoryTool:
    @staticmethod
    def bank_dir(workspace_dir: str | Path) -> Path:
        d = Path(workspace_dir) / MEMORY_DIRNAME
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- CRUD -----------------------------------------------------------------

    @staticmethod
    def write(workspace_dir: str | Path, kind: str, key: str, value: Any, *, ttl: int | None = None) -> dict:
        kind = _normalize_kind(kind)
        bank = _load_bank(workspace_dir)
        _prune(bank)
        entry = {
            "kind": kind,
            "key": key,
            "value": value,
            "created_at": time.time(),
            "accessed_at": time.time(),
            "ttl": ttl if ttl is not None else DEFAULT_TTL,
            "access_count": 0,
        }
        if kind not in bank:
            bank[kind] = {}
        bank[kind][key] = entry
        _save_bank(workspace_dir, bank)
        return entry

    @staticmethod
    def read(workspace_dir: str | Path, kind: str, key: str) -> dict | None:
        kind = _normalize_kind(kind)
        bank = _load_bank(workspace_dir)
        entry = bank.get(kind, {}).get(key)
        if entry is None:
            return None
        if _is_expired(entry):
            MemoryTool.delete(workspace_dir, kind, key)
            return None
        entry["accessed_at"] = time.time()
        entry["access_count"] += 1
        _save_bank(workspace_dir, bank)
        return dict(entry)

    @staticmethod
    def delete(workspace_dir: str | Path, kind: str, key: str) -> bool:
        kind = _normalize_kind(kind)
        bank = _load_bank(workspace_dir)
        if kind in bank and key in bank[kind]:
            del bank[kind][key]
            _save_bank(workspace_dir, bank)
            return True
        return False

    @staticmethod
    def search(workspace_dir: str | Path, *, kind: str | None = None, query: str | None = None, limit: int = 20) -> list[dict]:
        bank = _load_bank(workspace_dir)
        results = []
        kinds = [kind] if kind else list(bank)
        for k in kinds:
            for key, entry in bank.get(k, {}).items():
                if _is_expired(entry):
                    continue
                if query and query.lower() not in key.lower() and query.lower() not in str(entry.get("value", "")).lower():
                    continue
                results.append(dict(entry))
        results.sort(key=lambda e: e["accessed_at"], reverse=True)
        return results[:limit]

    @staticmethod
    def list_kinds(workspace_dir: str | Path) -> dict[str, int]:
        bank = _load_bank(workspace_dir)
        return {k: len(v) for k, v in bank.items() if isinstance(v, dict)}

    @staticmethod
    def snapshot(workspace_dir: str | Path) -> dict[str, list[dict]]:
        bank = _load_bank(workspace_dir)
        snapshot: dict[str, list[dict]] = {}
        for kind, entries in bank.items():
            if not isinstance(entries, dict):
                continue
            for key, entry in entries.items():
                snapshot.setdefault(kind, []).append({"key": key, "value": entry["value"]})
        return snapshot

    @staticmethod
    def clear(workspace_dir: str | Path, kind: str | None = None) -> int:
        kind = _normalize_kind(kind) if kind else None
        bank = _load_bank(workspace_dir)
        removed = 0
        if kind:
            if kind in bank:
                removed = len(bank[kind])
                bank[kind] = {}
        else:
            for k in list(bank):
                removed += len(bank[k])
                bank[k] = {}
        _save_bank(workspace_dir, bank)
        return removed

    # -- Decay / pruning -------------------------------------------------------

    @staticmethod
    def decay(workspace_dir: str | Path, *, factor: float = 0.5, min_access: int = 1) -> int:
        """Reduce TTL of rarely-accessed entries. Returns number of entries affected."""
        bank = _load_bank(workspace_dir)
        now = time.time()
        affected = 0
        for kind, entries in bank.items():
            if not isinstance(entries, dict):
                continue
            for key, entry in list(entries.items()):
                if entry["access_count"] < min_access and (now - entry["created_at"]) > DECAY_INTERVAL:
                    entry["ttl"] = max(300, int(entry["ttl"] * factor))
                    affected += 1
        _save_bank(workspace_dir, bank)
        return affected

    @staticmethod
    def injectable_context(workspace_dir: str | Path, *, max_entries: int = 10) -> str:
        """
        Returns a markdown-formatted string with the most relevant memory entries
        for injection as context to another agent. This is Phase 2 of the proactive
        memory pattern: deciding what to inject as a reminder.
        """
        bank = _load_bank(workspace_dir)
        candidates: list[dict] = []
        now = time.time()
        for kind, entries in bank.items():
            if not isinstance(entries, dict):
                continue
            for key, entry in entries.items():
                if _is_expired(entry):
                    continue
                recency = (now - entry["accessed_at"]) / max(1, entry["ttl"])
                importance = min(1.0, entry["access_count"] / 10)
                candidates.append({**entry, "_score": (1.0 - recency) * 0.6 + importance * 0.4})

        candidates.sort(key=lambda e: e["_score"], reverse=True)
        top = candidates[:max_entries]

        if not top:
            return ""

        parts = ["## Memory Context\n"]
        current_kind = None
        for entry in top:
            if entry["kind"] != current_kind:
                current_kind = entry["kind"]
                parts.append(f"\n### {current_kind.capitalize()}\n")
            val = entry["value"]
            val_str = val if isinstance(val, str) else json.dumps(val, indent=1, ensure_ascii=False)
            parts.append(f"- **{entry['key']}**: {val_str}")
        return "\n".join(parts)

    @staticmethod
    def note(workspace_dir: str | Path, key: str, value: Any) -> dict:
        """Quick shorthand: write a fact entry."""
        return MemoryTool.write(workspace_dir, "facts", key, value)

    @staticmethod
    def recall(workspace_dir: str | Path, key: str) -> dict | None:
        """Quick shorthand: read from any kind by key."""
        bank = _load_bank(workspace_dir)
        for kind, entries in bank.items():
            if not isinstance(entries, dict):
                continue
            if key in entries:
                return MemoryTool.read(workspace_dir, kind, key)
        return None


# -- Internal helpers ---------------------------------------------------------

def _normalize_kind(kind: str) -> str:
    kind = kind.strip().lower()
    if kind not in ("facts", "state", "traces"):
        raise ValueError(f"Invalid memory kind '{kind}': must be 'facts', 'state', or 'traces'.")
    return kind


def _bank_path(workspace_dir: str | Path) -> Path:
    return MemoryTool.bank_dir(workspace_dir) / "bank.json"


def _load_bank(workspace_dir: str | Path) -> dict:
    path = _bank_path(workspace_dir)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_bank(workspace_dir: str | Path, bank: dict) -> None:
    path = _bank_path(workspace_dir)
    path.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")


def _is_expired(entry: dict) -> bool:
    return time.time() - entry["created_at"] > entry.get("ttl", DEFAULT_TTL)


def _prune(bank: dict) -> None:
    now = time.time()
    for kind, entries in bank.items():
        if not isinstance(entries, dict):
            continue
        for key, entry in list(entries.items()):
            if now - entry["created_at"] > entry.get("ttl", DEFAULT_TTL):
                del entries[key]
    total = sum(len(v) for v in bank.values() if isinstance(v, dict))
    if total > MAX_ENTRIES:
        all_entries = []
        for kind, entries in bank.items():
            if not isinstance(entries, dict):
                continue
            for key, entry in entries.items():
                all_entries.append((kind, key, entry))
        all_entries.sort(key=lambda e: e[2]["accessed_at"])
        excess = total - MAX_ENTRIES
        for kind, key, _ in all_entries[:excess]:
            del bank[kind][key]
