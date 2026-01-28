.PHONY: setup test docs lint clean

setup:
	uv sync --extra dev --extra ml

test:
	uv run pytest tests

lint:
	uv run ruff {{ cookiecutter.project_module_name }} tests

docs:
	sphinx-apidoc -o docs/source/ {{ cookiecutter.project_module_name }}
	make html -C docs

clean:
	rm -rf .pytest_cache docs/build
	find {{ cookiecutter.project_module_name }} tests -type d -name "__pycache__" -exec rm -rf {} +
