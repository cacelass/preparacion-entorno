.PHONY: setup install-deps test lint format docs clean clean-models clean-figures clean-all run lab notebook info help

# ─────────────────────────────────────────────────────────────────────────────
#  Variables
# ─────────────────────────────────────────────────────────────────────────────
MODULE   = {{ cookiecutter.project_module_name }}
ML_TYPE  = {{ cookiecutter.ml_type }}
PYTHON   = python

# ─────────────────────────────────────────────────────────────────────────────
#  help  →  muestra todos los targets disponibles (target por defecto)
# ─────────────────────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  {{ cookiecutter.project_name }}  ·  ML tipo: $(ML_TYPE)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "  Entorno"
	@echo "    make setup          instala core + dev + extras del tipo ML"
	@echo "    make install-deps   solo dependencias del tipo ML (sin dev)"
	@echo "    make info           muestra versiones y paquetes instalados"
	@echo ""
	@echo "  Desarrollo"
	@echo "    make run            ejecuta main.py"
	@echo "    make test           lanza pytest con verbose"
	@echo "    make lint           ruff check (sin modificar)"
	@echo "    make format         ruff format (modifica en sitio)"
	@echo ""
	@echo "  Jupyter"
	@echo "    make lab            JupyterLab  (puerto 8888)"
	@echo "    make notebook       Jupyter Notebook (puerto 8888)"
	@echo ""
	@echo "  Documentación"
	@echo "    make docs           sphinx-apidoc + html"
	@echo ""
	@echo "  Limpieza"
	@echo "    make clean          cachés y __pycache__"
	@echo "    make clean-models   borra .joblib y .pt de models/"
	@echo "    make clean-figures  borra figuras de reports/figures/"
	@echo "    make clean-all      todo lo anterior junto"
	@echo ""

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
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Instalando dependencias para ML tipo: $(ML_TYPE)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	uv sync --extra dev --extra $(ML_TYPE)
	@echo ""
	@echo "  Entorno listo. Activa con:  source .venv/bin/activate"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
#  install-deps  →  solo las dependencias del tipo ML (sin dev tools)
# ─────────────────────────────────────────────────────────────────────────────
install-deps:
	uv sync --extra $(ML_TYPE)

# ─────────────────────────────────────────────────────────────────────────────
#  Desarrollo
# ─────────────────────────────────────────────────────────────────────────────
run:
	uv run $(PYTHON) main.py

test:
	uv run pytest tests -v

# lint   → solo comprueba, no modifica nada (útil en CI)
lint:
	uv run ruff check $(MODULE) tests
	uv run ruff format --check $(MODULE) tests

# format → aplica los cambios en sitio
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
#  Documentación
# ─────────────────────────────────────────────────────────────────────────────
docs:
	uv run sphinx-apidoc -o docs/source/ $(MODULE)
	$(MAKE) html -C docs

# ─────────────────────────────────────────────────────────────────────────────
#  Limpieza
# ─────────────────────────────────────────────────────────────────────────────
clean:
	rm -rf .pytest_cache docs/build
	find $(MODULE) tests -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "  Cachés y __pycache__ eliminados"

# Elimina modelos entrenados (joblib sklearn + pesos PyTorch + checkpoints)
clean-models:
	rm -f models/*.joblib models/*.pt models/checkpoint-*.pt
	@echo "  Modelos eliminados"

# Elimina figuras generadas por visualize.py
clean-figures:
	rm -f reports/figures/*.png reports/figures/*.html
	@echo "  Figuras eliminadas"

clean-all: clean clean-models clean-figures
	@echo "  Limpieza completa"

# ─────────────────────────────────────────────────────────────────────────────
#  Info del entorno
# ─────────────────────────────────────────────────────────────────────────────
info:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Proyecto  : {{ cookiecutter.project_name }}"
	@echo "  Módulo    : $(MODULE)"
	@echo "  ML tipo   : $(ML_TYPE)"
	@echo "  Python    : $(shell $(PYTHON) --version)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@uv pip list 2>/dev/null | head -30 || pip list | head -30
	@echo ""