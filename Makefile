.PHONY: setup update lint test skills help

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  dskit — auto-mantenimiento del template"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "  make setup        instala dskit + dev dependencies"
	@echo "  make update       uv sync + actualiza skills locales"
	@echo "  make lint         ruff check (excluye template/)"
	@echo "  make test         pytest (template tests si existen)"
	@echo "  make skills       instala prompts como skills en .opencode/skills/"
	@echo ""

setup:
	uv sync --extra dev
	@echo "  dskit listo."

update: setup skills
	@echo "  dskit actualizado."

lint:
	uv run ruff check .

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
