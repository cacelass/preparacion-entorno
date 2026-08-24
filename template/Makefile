.PHONY: setup install-deps \
        data features train predict pipeline \
        test smoke lint format typecheck security audit check lock \
        lab notebook tb \
        docs \
        profile \
        mlflow \
{% if use_sdd %}        mutation crap {% endif %}\
{% if use_integration %}        integration {% endif %}\
        monitor tune serve query \
{% if use_docker %}        docker-run docker-update docker-down {% endif %}\
{% if use_demo %}        demo-export demo-serve {% endif %}\
        agents-list agents-run agents-doctor agents-memory agents-eval \
        init harness-check backlog prompts-sync prompts-check assistants-sync \
        skills opencode-init opencode-check \
        index-rag index-rag-rebuild eval-rag \
        recommended-tools recommended-all \
        clean clean-models clean-figures clean-all \
        run info help

# ─────────────────────────────────────────────────────────────────────────────
#  Variables
# ─────────────────────────────────────────────────────────────────────────────
MODULE   = {{ project_slug }}
ML_TYPE  = {{ ml_type }}
PYTHON   = python

# ─────────────────────────────────────────────────────────────────────────────
#  help  →  target por defecto
# ─────────────────────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  {{ project_name }}  ·  ML: $(ML_TYPE)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "  Entorno"
	@echo "    make setup          instala core + dev + extras del tipo ML"
	@echo "    make install-deps   solo dependencias del tipo ML (sin dev)"
	@echo "    make info           versiones y paquetes instalados"
	@echo ""
	@echo "  Pipeline de datos y modelos"
	@echo "    make data           descarga/preprocesa datos crudos"
	@echo "    make features       construye features desde datos procesados"
	@echo "    make train          entrena el modelo"
	@echo "    make predict        genera predicciones con el modelo entrenado"
	@echo "    make pipeline       data → features → train → predict (todo)"
	@echo ""
	@echo "  Calidad"
	@echo "    make check          lint + typecheck + test (batería completa)"
	@echo "    make test           pytest -v - cov≥80% (todos los tests)"
	@echo "    make smoke          test de humo — verifica que el pipeline arranca"
	@echo "    make lint           ruff check (solo lectura, sin modificar)"
	@echo "    make format         ruff format (aplica cambios en sitio)"
	@echo "    make typecheck      mypy --strict (tipado estático)"
	@echo "    make security       bandit + pip-audit (vulnerabilidades)"
	@echo "    make audit          radon cc + agents audit (complejidad + equipo)"
{% if use_sdd %}	@echo "    make mutation       mutation testing de un módulo (spec-driven)"
	@echo "    make crap           métrica CRAP de un módulo (complejidad × cobertura)"
{% endif %}
{% if use_integration %}	@echo "    make integration    tests de integración contra servicios reales (Docker, sin mocks)"
{% endif %}
{% if use_demo %}	@echo ""
	@echo "  Demo"
	@echo "    make demo-export    exporta modelos a ONNX → demo/models/"
	@echo "    make demo-serve     sirve la demo en http://localhost:8000"
{% endif %}
	@echo ""
	@echo "  Jupyter"
	@echo "    make lab            JupyterLab  (puerto 8888)"
	@echo "    make notebook       Jupyter Notebook (puerto 8888)"
	@echo ""
	@echo "  Ejecución directa"
	@echo "    make run            ejecuta main.py"
	@echo "    make profile        cProfile de main.py → reports/profile.prof"
	@echo ""
{% if use_monitoring %}
	@echo "  Monitoring"
	@echo "    make monitor        drift detection + performance report"
	@echo ""
{% endif %}
{% if use_optuna %}
	@echo "  Optuna"
	@echo "    make tune           optimiza hiperparámetros (n_trials en main.py)"
	@echo ""
{% endif %}
{% if use_duckdb %}
	@echo "  DuckDB"
	@echo "    make query          Shell SQL interactivo sobre data/raw/"
	@echo ""
{% endif %}
{% if use_api %}
	@echo "  API REST"
	@echo "    make serve          FastAPI en localhost:8000  (docs en /docs)"
	@echo ""
{% endif %}
{% if ml_type == "redes_neuronales" %}
	@echo "  Monitorizacion"
{% if use_mlflow %}
	@echo "    make mlflow         MLflow UI en localhost:5000"
{% endif %}
	@echo "    make tb             lanza TensorBoard (puerto 6006)"
	@echo ""
{% endif %}
	@echo "  Documentación"
	@echo "    make docs           sphinx-apidoc + html"
	@echo ""
	@echo "  Arnés (harness)"
	@echo "    make init            puerta de entrada: ¿se puede trabajar? (init.sh)"
	@echo "    make harness-check   valida la estructura del arnés sin correr tests"
	@echo "    make backlog         muestra el estado de harness/featureslist.json"
	@echo ""
	@echo "  Agentes"
	@echo "    make agents-list     listar agentes disponibles"
	@echo "    make agents-run      ejecutar un agente"
	@echo "    make agents-doctor   diagnóstico completo con agentes"
	@echo "    make agents-memory   estado de la memoria de agentes"
	@echo "    make agents-eval     harness + rag + smoke + routing + contracts"
	@echo "    make eval-rag        mide la recuperacion del RAG (hit_rate/recall/MRR)"
	@echo "    make prompts-sync    regenera los prompts desde el codigo y contracts.py"
	@echo "    make prompts-check   falla si un prompt se ha desincronizado"
	@echo "    make assistants-sync espeja los subagentes a .claude/agents/"
	@echo ""
	@echo "  Skills del asistente"
	@echo "    make skills          instalar skills locales en .opencode/skills/"
	@echo "    make opencode-init   configurar subagentes opencode (orquestador)"
	@echo "    make opencode-check  validar consistencia de la configuración"
	@echo ""
	@echo "  Ecosistema"
	@echo "    make recommended-tools  mostrar herramientas recomendadas"
	@echo ""
	@echo "  Limpieza"
	@echo "    make clean          cachés y __pycache__"
	@echo "    make clean-models   borra .joblib y .pt de models/"
	@echo "    make clean-figures  borra figuras de reports/figures/"
	@echo "    make clean-all      todo lo anterior"
	@echo ""
{% if use_docker %}	@echo "  Docker"
	@echo "    make docker-run     construye imagen y lanza el chat en http://localhost:8080"
	@echo "    make docker-update  reconstruye la imagen con los cambios mas recientes"
	@echo "    make docker-down    para y elimina los contenedores"
	@echo ""
{% endif %}

# ─────────────────────────────────────────────────────────────────────────────
#  Entorno
# ─────────────────────────────────────────────────────────────────────────────
setup:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Instalando dependencias para ML tipo: $(ML_TYPE)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	uv sync --extra dev --extra $(ML_TYPE) \
{% if use_mlflow %}		--extra mlflow_tracking \
{% endif %}{% if graphify_mode != 'no' %}		--extra graphify \
{% endif %}{% if use_api %}		--extra api \
{% endif %}{% if use_duckdb %}		--extra duckdb \
{% endif %}{% if use_optuna %}		--extra optuna \
{% endif %}{% if use_monitoring %}		--extra monitoring \
{% endif %}{% if use_rag %}		--extra rag \
{% endif %}		&& echo "  Extras instalados."
	@uv run pre-commit install 2>/dev/null || echo "  (pre-commit no disponible)"
	@echo ""
	@echo "  Listo. Activa el entorno con:  source .venv/bin/activate"
	@echo ""

install-deps:
	uv sync --extra $(ML_TYPE)

# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline  data → features → train → predict
#
#  Cada step llama al script correspondiente dentro del módulo.
#  Estructura esperada:
#    $(MODULE)/data/make_dataset.py        → lee data/raw/, escribe data/processed/
#    $(MODULE)/features/build_features.py  → lee data/processed/, escribe data/interim/
#    $(MODULE)/models/train_model.py       → lee data/interim/, escribe models/
#    $(MODULE)/models/predict_model.py     → lee models/ + data/interim/, escribe reports/
# ─────────────────────────────────────────────────────────────────────────────
data:
	@echo "▶  Procesando datos crudos → data/processed/"
	uv run $(PYTHON) $(MODULE)/data/make_dataset.py

features: data
	@echo "▶  Construyendo features → data/interim/"
	uv run $(PYTHON) $(MODULE)/features/build_features.py

train: features
	@echo "▶  Entrenando modelo → models/"
	uv run $(PYTHON) $(MODULE)/models/train_model.py

predict: train
	@echo "▶  Generando predicciones → reports/"
	uv run $(PYTHON) $(MODULE)/models/predict_model.py

pipeline: predict
	@echo ""
	@echo "  Pipeline completo finalizado."
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
#  Ejecución directa
# ─────────────────────────────────────────────────────────────────────────────
run:
	uv run $(PYTHON) main.py

{% if use_monitoring %}
# ─────────────────────────────────────────────────────────────────────────────
#  Monitoring — drift detection y performance tracking
# ─────────────────────────────────────────────────────────────────────────────
monitor:
	@echo "▶  Ejecutando monitorización de drift y rendimiento..."
	uv run python -m tools.monitor

{% endif %}
{% if use_optuna %}
# ─────────────────────────────────────────────────────────────────────────────
#  Optuna — optimizacion de hiperparametros
# ─────────────────────────────────────────────────────────────────────────────
tune:
	@echo "▶  Lanzando optimizacion Optuna (OPTUNA_TRIALS en main.py)"
	uv run python -m tools.tune_model

{% endif %}
{% if use_duckdb %}
# ─────────────────────────────────────────────────────────────────────────────
#  DuckDB — exploración SQL interactiva sobre data/raw/
# ─────────────────────────────────────────────────────────────────────────────
query:
	@echo "▶  DuckDB shell — escribe .quit para salir"
	@echo "   Ejemplo: SELECT * FROM read_csv('data/raw/dataset.csv') LIMIT 10;"
	uv run duckdb

{% endif %}
{% if use_api %}
# ─────────────────────────────────────────────────────────────────────────────
#  API REST — servir el modelo con FastAPI
# ─────────────────────────────────────────────────────────────────────────────
serve:
	@echo "▶  Lanzando API REST en http://localhost:8000"
	@echo "   Documentación interactiva: http://localhost:8000/docs"
	uv run uvicorn api.main:app --reload --port 8000

{% endif %}
profile:
	@echo "▶  Profiling main.py → reports/profile.prof"
	uv run $(PYTHON) -m cProfile -o reports/profile.prof main.py
	@echo "   Visualiza con: uv run snakeviz reports/profile.prof"

# ─────────────────────────────────────────────────────────────────────────────
#  Calidad de código
# ─────────────────────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ -v

smoke:
	@echo "▶  Test de humo — pipeline con datos sintéticos"
	uv run pytest tests/ -v -m smoke --tb=short

check: lint typecheck test harness-check
	@echo ""
	@echo "  ✅ check pasado: lint + typecheck + test + arnés OK"

lint:
	uv run ruff check $(MODULE)/ tests/
	uv run ruff format --check $(MODULE)/ tests/

format:
	uv run ruff format $(MODULE)/ tests/

typecheck:
	uv run mypy --strict $(MODULE)/ tests/
	@echo "  ✅ typecheck OK"

security:
	uv run bandit -r $(MODULE)/ -f txt -x tests
	uv run pip-audit 2>/dev/null || echo "  (pip-audit requiere paquetes instalados — ejecuta make setup primero)"
	@echo "  ✅ security check OK"

audit:
	@echo "▶  Complejidad ciclomática (radon)"
	@uv run radon cc $(MODULE)/ -s --min C 2>/dev/null || echo "  (sin radon)"
	@echo ""
	uv run python -m agents audit suggest 2>/dev/null || echo "  (agente 'audit' no disponible — ejecuta make setup)"
	@echo "  ✅ audit OK"

{% if use_sdd %}
# ─────────────────────────────────────────────────────────────────────────────
#  Spec-driven: mutation testing y CRAP
#  Uso: make mutation TARGET={{ project_slug }}/features/build_features.py
# ─────────────────────────────────────────────────────────────────────────────
mutation:
	@echo "▶  Mutation testing de $(TARGET) — los tests deben «morder»"
	uv run python -m agents --json run mutation run_mutation_testing --target $(TARGET)

crap:
	@echo "▶  Métrica CRAP de $(TARGET) — riesgo de cambio"
	uv run python -m agents --json run mutation crap_report --target $(TARGET)
{% endif %}
{% if use_integration %}
# ─────────────────────────────────────────────────────────────────────────────
#  Tests de integración: servicios reales (Docker), sin mocks
# ─────────────────────────────────────────────────────────────────────────────
integration:
	@echo "▶  Tests de integración contra servicios reales (Docker, sin mocks)"
	uv run python -m agents --json run integration run_integration_tests
{% endif %}

# ─────────────────────────────────────────────────────────────────────────────
#  Jupyter
# ─────────────────────────────────────────────────────────────────────────────
lab:
	uv run jupyter lab --ip=* --port=8888 --no-browser

notebook:
	uv run jupyter notebook --ip=* --port=8888 --no-browser


{% if use_mlflow %}
# ─────────────────────────────────────────────────────────────────────────────
#  MLflow UI
# ─────────────────────────────────────────────────────────────────────────────
mlflow:
	@echo "Lanzando MLflow UI en http://localhost:5000"
	uv run mlflow ui --port 5000
{% endif %}
{% if ml_type == "redes_neuronales" %}
# ─────────────────────────────────────────────────────────────────────────────
#  TensorBoard — monitorización de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────
tb:
	@echo "▶  Lanzando TensorBoard en http://localhost:6006"
	uv run tensorboard --logdir runs/ --port 6006
{% endif %}

# ─────────────────────────────────────────────────────────────────────────────
#  Documentación
# ─────────────────────────────────────────────────────────────────────────────
docs:
	uv run sphinx-apidoc -o docs/source/ $(MODULE)/
	$(MAKE) html -C docs

# ─────────────────────────────────────────────────────────────────────────────
#  Arnés (harness) — ver el protocolo en AGENTS.md
# ─────────────────────────────────────────────────────────────────────────────
init:
	@chmod +x init.sh 2>/dev/null || true
	@./init.sh

harness-check:
	@chmod +x init.sh 2>/dev/null || true
	@./init.sh --quick

backlog:
	@uv run $(PYTHON) -c "import json; d=json.load(open('harness/featureslist.json')); \
	print(); print('  Backlog de {{ project_name }}'); print(); \
	[print('  [%-11s] %-10s %s' % (f['status'], f['id'], f['title'])) for f in d['features']]; \
	print()"

# ─────────────────────────────────────────────────────────────────────────────
#  Sistema de agentes
# ─────────────────────────────────────────────────────────────────────────────
agents-list:
	uv run python -m agents list

agents-run:
	uv run python -m agents run $(filter-out $@,$(MAKECMDGOALS))

agents-doctor:
	uv run python -m agents doctor

agents-memory:
	@echo "▶  Estado de la memoria de agentes"
	uv run python -m agents run memory status

agents-eval:
	@echo "▶  Evaluación del sistema de agentes (harness + rag + smoke + routing + contracts)"
	uv run python -m agents.evals.runner

prompts-sync assistants-sync:
	@echo "▶  Regenerando prompts y subagentes desde el código y los contratos..."
	uv run python -m agents.prompts_sync --write

prompts-check:
	@uv run python -m agents.prompts_sync

index-rag:
	@echo "▶  Indexando código y documentación del proyecto en ChromaDB..."
	@echo "   (incremental: solo se reindexa lo que cambió)"
	uv run python -m agents run rag index

# Necesario al cambiar DSKIT_RAG_EMBEDDER: los vectores del embedder anterior
# no son comparables con los del nuevo, y mezclarlos no da error — da basura.
index-rag-rebuild:
	@echo "▶  Reconstruyendo el índice RAG desde cero..."
	uv run python -m agents run rag index --rebuild

# Mide si la búsqueda encuentra lo que debería, contra
# agents/evals/rag_golden.json. Sin esto, tocar el troceado o el embedder es
# fe: "parece que ahora busca mejor" no es un dato.
eval-rag:
	@echo "▶  Evaluando la recuperación del RAG (hit_rate / recall@k / MRR)"
	uv run python -m agents run rag evaluate

# ─────────────────────────────────────────────────────────────────────────────
#  Skills — configuración del asistente de IA
# ─────────────────────────────────────────────────────────────────────────────
skills:
	@echo "▶  Instalando skills del proyecto..."
	@mkdir -p .opencode/skills
	@for skill in agents/prompts/*.md; do \
		name=$$(basename "$$skill" .md); \
		cp "$$skill" ".opencode/skills/$$name.md"; \
		echo "   skill: $$name"; \
	done
	@echo "   Skills instaladas en .opencode/skills/ (disponibles para el asistente)."
	@echo ""
	@echo "   Para soporte multi-asistente, también se pueden instalar como:"
	@echo "     npx autoskills -y    (si usas Node.js, instala skills de skills.sh)"

opencode-init: skills
	@echo "▶  Configurando subagentes de opencode..."
	@mkdir -p .opencode/agents
	@if [ ! -f .opencode/agents/orquestador.md ]; then \
		echo "  ERROR: falta .opencode/agents/orquestador.md — copia desde template/"; \
		exit 1; \
	fi
	@echo "   opencode configurado. Agente 'orquestador' disponible."
	@echo "   Usa Tab en opencode para cambiar al agente orquestador."
	@echo ""

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
	@for a in lider implementer reviewer explorer; do \
		if [ ! -f ".opencode/agents/$$a.md" ]; then \
			echo "  ✘ falta .opencode/agents/$$a.md (agente del arnés)"; \
		else \
			echo "  ✔ .opencode/agents/$$a.md"; \
		fi; \
	done
	@count=$$(ls .opencode/skills/*.md 2>/dev/null | wc -l); \
	if [ "$$count" -lt 20 ]; then \
		echo "  ✘ skills insuficientes ($$count/29+ esperados) — ejecuta make skills"; \
		errors=$$((errors + 1)); \
	else \
		echo "  ✔ $$count skills instaladas"; \
	fi
	@count2=$$(ls agents/prompts/*.md 2>/dev/null | wc -l); \
	echo "  ✔ $$count2 prompts fuente disponibles."
	@if [ "$$errors" -eq 0 ]; then \
		echo "  ✔ Todo correcto."; \
	else \
		echo "  ✘ $$errors error(es) encontrados."; \
		exit 1; \
	fi

# ─────────────────────────────────────────────────────────────────────────────
#  Recomendados — instala herramientas del ecosistema (agents/README.md)
# ─────────────────────────────────────────────────────────────────────────────
recommended-tools:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Herramientas recomendadas del ecosistema"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "  Skills (skills.sh):"
	@echo "    npx skills add obra/superpowers"
	@echo "    npx skills add mattpocock/skills"
	@echo "    npx skills add anthropics/skills/skill-creator"
	@echo ""
	@echo "  Frameworks de memoria (pip):"
	@echo "    uv pip install cognee"
	@echo "    uv pip install supermemory"
	@echo ""
	@echo "  Agentes de investigación (pip):"
	@echo "    uv pip install langchain-community langgraph"
	@echo "    git clone https://github.com/langchain-ai/local-deep-researcher"
	@echo ""
	@echo "  Fairness / bias (pip):"
	@echo "    uv pip install eticas-audit"
	@echo ""
	@echo "  Otros:"
	@echo "    npx autoskills -y           # skills automáticas por stack"
	@echo "    npm install -g @synsci/openscience  # AI workbench científico"
	@echo ""

recommended-all: recommended-tools

# ─────────────────────────────────────────────────────────────────────────────
#  Demo (ONNX)
#  Flujo: make train → make demo-export → make demo-serve (local)
#  No hay auto-deploy: si la quieres publicar, súbela tú cuando quieras.
# ─────────────────────────────────────────────────────────────────────────────
demo-export:
	@echo "▶  Exportando modelos a ONNX → demo/models/"
	uv run $(PYTHON) tools/export_onnx.py
	@echo ""
	@echo "  Demo lista. Para verla localmente: make demo-serve"
	@echo "  (directorio local: no hay auto-deploy. Los .onnx SÍ van al repo —"
	@echo "   son pequeños, KB a pocos MB.)"

demo-serve:
	@echo "▶  Sirviendo demo en http://localhost:8000"
	@cd demo && uv run $(PYTHON) -m http.server 8000

# ─────────────────────────────────────────────────────────────────────────────
#  Limpieza
# ─────────────────────────────────────────────────────────────────────────────
clean:
	rm -rf .pytest_cache docs/build
	find $(MODULE) tests -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	@echo "  Cachés y __pycache__ eliminados."

clean-models:
	rm -f models/*.joblib models/*.pkl models/*.pt models/checkpoint-*.pt
	@echo "  Modelos eliminados."

clean-figures:
	rm -f reports/figures/*.png reports/figures/*.svg reports/figures/*.html
	@echo "  Figuras eliminadas."

clean-all: clean clean-models clean-figures
	rm -f reports/*.csv
	rm -rf agents/workspace/
	@echo "  Limpieza completa."


{% if use_docker %}
# ─────────────────────────────────────────────────────────────────────────────
#  Docker — interfaz web de chat
# ─────────────────────────────────────────────────────────────────────────────
docker-run:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Construyendo y arrancando contenedor Docker"
	@echo "  Chat disponible en: http://localhost:8080"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	docker compose up -d --build
	@echo ""
	@echo "  Logs en tiempo real: docker compose logs -f"
	@echo ""

docker-update:
	@echo "  Reconstruyendo imagen con cambios recientes..."
	docker compose up -d --build --force-recreate
	@echo "  Imagen actualizada y contenedor reiniciado."

docker-down:
	@echo "  Parando contenedores..."
	docker compose down
	@echo "  Contenedores detenidos."
{% endif %}

# ─────────────────────────────────────────────────────────────────────────────
#  Dependencias
# ─────────────────────────────────────────────────────────────────────────────
lock:
	uv lock
	@echo "  uv.lock actualizado."

# ─────────────────────────────────────────────────────────────────────────────
#  Info del entorno
# ─────────────────────────────────────────────────────────────────────────────
info:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Proyecto  : {{ project_name }}"
	@echo "  Módulo    : $(MODULE)"
	@echo "  ML tipo   : $(ML_TYPE)"
	@echo "  Python    : $(shell uv run python --version 2>/dev/null || python --version)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@uv pip list 2>/dev/null | head -40 || pip list | head -40
	@echo ""
