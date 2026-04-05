.PHONY: setup install-deps test docs lint clean run lab

# ─────────────────────────────────────────────────────────────────────────────
#  Variables
# ─────────────────────────────────────────────────────────────────────────────
MODULE   = {{ cookiecutter.project_module_name }}
ML_TYPE  = {{ cookiecutter.ml_type }}
PYTHON   = python

# ─────────────────────────────────────────────────────────────────────────────
#  setup  →  instala TODO: core + dev + extra del tipo de ML elegido
#
#  Uso:
#    make setup
#
#  Equivalente manual:
#    uv sync --extra dev --extra {{ cookiecutter.ml_type }}
# ─────────────────────────────────────────────────────────────────────────────
setup:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Instalando dependencias para ML tipo: $(ML_TYPE)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	uv sync --extra dev --extra $(ML_TYPE)
	@echo ""
	@echo "✅  Entorno listo. Activa con:  source .venv/bin/activate"

# ─────────────────────────────────────────────────────────────────────────────
#  install-deps  →  solo las dependencias del tipo ML (sin dev tools)
# ─────────────────────────────────────────────────────────────────────────────
install-deps:
	uv sync --extra $(ML_TYPE)

# ─────────────────────────────────────────────────────────────────────────────
#  Desarrollo
# ─────────────────────────────────────────────────────────────────────────────
test:
	uv run pytest tests -v

lint:
	uv run ruff check $(MODULE) tests
	uv run ruff format --check $(MODULE) tests

format:
	uv run ruff format $(MODULE) tests

# ─────────────────────────────────────────────────────────────────────────────
#  Jupyter
# ─────────────────────────────────────────────────────────────────────────────
lab:
	uv run jupyter lab --ip=* --port=8888 --no-browser

notebook:
	uv run jupyter notebook --ip=* --port=8888 --no-browser

# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────
run:
	uv run $(PYTHON) main.py

# ─────────────────────────────────────────────────────────────────────────────
#  Documentación
# ─────────────────────────────────────────────────────────────────────────────
docs:
	uv run sphinx-apidoc -o docs/source/ $(MODULE)
	make html -C docs

# ─────────────────────────────────────────────────────────────────────────────
#  Limpieza
# ─────────────────────────────────────────────────────────────────────────────
clean:
	rm -rf .pytest_cache docs/build
	find $(MODULE) tests -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

clean-models:
	rm -f models/*.joblib models/*.pt models/checkpoint-*.pt
	@echo "Modelos eliminados"

clean-figures:
	rm -f reports/figures/*.png reports/figures/*.html
	@echo "Figuras eliminadas"

clean-all: clean clean-models clean-figures

# ─────────────────────────────────────────────────────────────────────────────
#  Info del entorno
# ─────────────────────────────────────────────────────────────────────────────
info:
	@echo "Proyecto  : {{ cookiecutter.project_name }}"
	@echo "Módulo    : $(MODULE)"
	@echo "ML tipo   : $(ML_TYPE)"
	@echo "Python    : $(shell $(PYTHON) --version)"
	@uv pip list 2>/dev/null | head -30 || pip list | head -30