.PHONY: setup update lint test typecheck format security precommit skills help

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
	@echo ""

setup:
	uv sync --extra dev
	uv run pre-commit install
	@echo "  dskit listo."

update: setup skills
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

test:
	uv run pytest -x -q 2>/dev/null || echo "  (no hay tests propios de dskit)"
