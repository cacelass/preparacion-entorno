# Project-Level Instructions

## Memory system

- **`memory.md`** in the project root stores corrections, preferences, and things to remember across sessions.
- At the **start of every session**, read `memory.md` first.
- When the user corrects something and says to remember it, save it in `memory.md`.

## Assistant guidelines

- **Be concise.** Answer in 1-3 sentences when possible. No preamble or postamble.
- **Follow code conventions.** Match the project's style, imports, and patterns. Don't add comments unless asked.
- **Minimal changes.** Only modify what's necessary. Prefer editing existing files over creating new ones.
- **Read before edit.** Always read a file before editing it.
- **Verify your work.** Run lint/typecheck/tests after changes.
- **Commit discipline.** Only commit when explicitly asked. Don't force-push. Write concise commit messages matching repo style.
- **Security.** Never log, expose, or commit secrets or keys.
- **Proactive but not surprising.** Do the right thing, but don't take actions without explanation.

## OpenCode integration

This project has an opencode **subagent gateway** (`orquestador`) configured in `opencode.json`.
Press Tab in opencode to switch to it. The orquestador delegates to 27 Python agents
via `uv run python -m agents [ask|run|pipeline|doctor]`.

### Architecture

```
[opencode assistant]  ←  Tab  →  [orquestador subagent]
                                       │
                                  delegates via CLI (--json mode)
                                       │
                              [Python agent system]
                              ├── Orchestrator.dispatch() ← routing por keywords
                              ├── 27 agents (git, test, review, docker...)
                              ├── GStack pipelines (develop, fix, release...)
                              ├── Workflows por dominio (data, ml, dev, api, docker...)
                              └── audit trail + contracts
```

### Setup

```bash
make skills          # copy agent prompts → .opencode/skills/
make opencode-init   # verify orquestador agent is configured
make agents-eval     # smoke + routing + contracts
```

### Protocol

- Use `--json` flag for structured output (more reliable than parsing text)
- The orchestrator gateway prompt lives in `.opencode/agents/orquestador.md`
- **Workflow skills** (`.opencode/skills/*_workflow.md`) document pipelines completos por dominio; carga con `skill <name>` cuando la tarea abarca todo un dominio
- **Agent skills** (`.opencode/skills/*.md`) describen agentes individuales (~15 líneas); carga con `skill <name>` para acciones concretas
- Agent prompts in `template/agents/prompts/` are the source of truth for skills

### Maintenance

- `make skills` regenerates `.opencode/skills/` from `template/agents/prompts/`
- If you add a new agent, update `.opencode/agents/orquestador.md` with its entry
- If you add a workflow skill, register it in `.opencode/agents/orquestador.md`, `AGENTS.md`, and `agents/evals/runner.py`
- Run `uv run python -m agents list` to verify agents are discoverable
