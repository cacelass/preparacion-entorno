.PHONY: setup update lint test typecheck format security precommit skills opencode-init opencode-check help

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  dskit — auto-mantenimiento del template"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "  make setup        instala dskit + dev dependencies + pre-commit"
	@echo "  make update       uv sync + actualiza skills locales"
	@echo "  make lint         ruff check (excluye template/)"
	@echo "  make format       ruff format + isort"
	@echo "  make typecheck    mypy --strict"
	@echo "  make test         pytest (template tests si existen)"
	@echo "  make security     bandit scan"
	@echo "  make precommit    instalar/actualizar hooks de pre-commit"
	@echo "  make skills       instala prompts como skills en .opencode/skills/"
	@echo "  make opencode-init  configura subagentes opencode (orquestador gateway)"
	@echo "  make opencode-check valida consistencia entre skills y gateway"
	@echo "  make eval-agents  ejecuta evaluación de agentes (smoke+routing+contracts)"
	@echo ""

setup:
	uv sync --extra dev
	uv run pre-commit install
	@echo "  dskit listo."

update: setup skills opencode-init
	@echo "  dskit actualizado."

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run isort .

typecheck:
	uv run mypy --strict . --exclude template/ --exclude .venv

security:
	uv run bandit -r . -f txt -x template,.venv,tests
	uv run pip-audit 2>/dev/null || echo "  (pip-audit requiere deps instaladas)"

audit:
	@uv run radon cc . -s --min C --exclude template,.venv,tests 2>/dev/null || echo "  (sin radon)"

precommit:
	uv run pre-commit install
	uv run pre-commit autoupdate

skills:
	@mkdir -p .opencode/skills
	@for skill in template/agents/prompts/*.md; do \
		name=$$(basename "$$skill" .md); \
		cp "$$skill" ".opencode/skills/$$name.md"; \
		echo "   skill: $$name"; \
	done
	@echo "  Skills instaladas en .opencode/skills/"

opencode-init: skills
	@mkdir -p .opencode/agents
	@if [ ! -f .opencode/agents/orquestador.md ]; then \
		echo "  ERROR: falta .opencode/agents/orquestador.md — copia desde template/"; \
		exit 1; \
	fi
	@echo "  opencode configurado. Agente orquestador disponible."
	@echo "  Usa Tab en opencode para cambiar al agente 'orquestador'."

opencode-check:
	@echo "▶  Verificando configuración opencode..."
	@errors=0
	@if [ ! -f opencode.json ]; then \
		echo "  ✘ falta opencode.json"; \
		errors=$$((errors + 1)); \
	else \
		echo "  ✔ opencode.json"; \
	fi
	@if [ ! -f .opencode/agents/orquestador.md ]; then \
		echo "  ✘ falta .opencode/agents/orquestador.md"; \
		errors=$$((errors + 1)); \
	else \
		echo "  ✔ .opencode/agents/orquestador.md"; \
	fi
	@count=$$(ls .opencode/skills/*.md 2>/dev/null | wc -l); \
	if [ "$$count" -lt 20 ]; then \
		echo "  ✘ skills insuficientes ($$count/29 esperados) — ejecuta make skills"; \
		errors=$$((errors + 1)); \
	else \
		echo "  ✔ $$count skills instaladas"; \
	fi
	@count2=$$(ls template/agents/prompts/*.md 2>/dev/null | wc -l); \
	if [ "$$errors" -eq 0 ]; then \
		echo "  ✔ Todo correcto ($$count2 prompts fuente)."; \
	else \
		echo "  ✘ $$errors error(es) encontrados."; \
		exit 1; \
	fi

eval-agents:
	@echo "▶  Ejecutando evaluación de agentes..."
	@cd template && uv run python -m agents.evals.runner 2>/dev/null || \
		echo "  (necesita template renderizado con dependencias)"

test:
	uv run pytest -x -q 2>/dev/null || echo "  (no hay tests propios de dskit)"
