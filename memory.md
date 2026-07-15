# Memory

Instrucciones, correcciones y preferencias guardadas por el usuario para recordar en futuras sesiones.

---

## Commits

- Usar mensajes convencionales (feat:, fix:, chore:, docs:)
- `ecommits` no está instalado en el entorno — hacer commits manuales
- Taggear tras commits grandes: `git tag -a vX.Y.Z -m "mensaje"`
- "comitea" significa commitear inmediatamente (no preguntar, no esperar cambio a code mode)

## Memory Agent (Jul 2026)

- Implementado `memory_agent.py` (proactive memory pattern, arXiv 2607.08716)
- Memory bank en `agents/workspace/memory/bank.json` con tres kinds: facts, state, traces
- `make skills` instala prompts como skills en `.opencode/skills/`
- `make agents-memory` muestra estado de la memoria de agentes
- Usar `npx autoskills -y` para skills del ecosistema (Node.js)

## Testing (Jul 2026)

- Hypothesis añadido a dev deps del template (`template/pyproject.toml`)
- `test_hypothesis.py` — property-based tests para invariantes de `AgentResult`, `can_handle`, `best_action`, `Contract`
- `test_audit_agent.py` — 12 tests (era 2). Cobertura: report vacío, agregación, tasa de fallo, failures con límite, suggest_improvements (fallo alto, lento, warnings, saludable), líneas corruptas, líneas vacías
- `test_doctor_agent.py` — 15 tests (era 2). Cobertura: todas las secciones, missing pyproject, missing slug, missing `__init__`, tests dir vacío, disk_usage inexistente/con archivos, summary, _human_size edge cases
- `test_docker_agent.py` — 14 tests (era 3). Linter: FROM sin tag, :latest, sin USER, ADD vs COPY, apt-get recommends, apt-get update separado, vacío, solo comentarios
- `test_test_agent.py` — 15 tests (era 4). Parsing JUnit XML (todos pasan, failures, errors, skipped, vacío, malformed, sin testsuite), coverage JSON (0%, 100%, sin totals), skeleton generation
- Los tests del template NO se pueden ejecutar directamente sobre el template sin renderizar (Jinja2 en source files rompe el parser de Python). Para probar: copiar `agents/` a tmp, quitar `{% raw %}` con sed, instalar deps.
- Bug encontrado: `docker_tool.py` comparaba `"apt-get update"` contra `line.upper()` (case mismatch). Arreglado.
- Bug encontrado: `memory` agent registrado sin contrato en `contracts.py`. Añadido.
- Abstracciones removidas: `base_tool.py` (ABC muerta), `prev()` reference resolution de `gstack/stack.py` (DSL no usado), `_discover_entry_points()` de `registry.py` (infraestructura de plugins sin paquetes externos)
- 219 tests pasan en agentes

## Code review checklist

Checklist de áreas clave a inspeccionar y herramientas a incorporar, según revisión exterior:
- **Código**: módulos bajo `{{ project_slug }}/` — data, features, models, utils, visualization. Verificar estructura de paquete Python (src/, `__init__.py`).
- **CI**: `.github/workflows/` — validar comandos existentes, eliminar redundancias.
- **Tests**: `tests/` — listar módulos sin cobertura. Meta ≥80%.
- **Docs**: README, LICENSE, CONTRIBUTING, SECURITY — imprescindibles.
- **Deps**: escaneo de vulnerabilidades (pip-audit), justificar dependencias.
- **Herramientas a integrar**: Black (formato), isort (imports), MyPy strict (tipado), Radon ≤10 (complejidad ciclomática), Bandit (seguridad), pre-commit hooks, pytest-cov --cov-fail-under=80.
- **Tests a añadir**: unitarios por módulo (data, features, modelos), integración (pipeline completo con dataset controlado), E2E (CLI/pipelines). Usar fixtures+parametrización en pytest.
- **Refactorizaciones**: eliminar sobreabstracción (funciones > clases encadenadas si solo hay una variante), eliminar código muerto, simplificar pipelines fragmentados, docstrings estilo Google en API pública.
- **Regla**: no abstracción anticipada. Solo extraer interfaz cuando haya ≥2 implementaciones reales.

## Self-maintenance (Jul 2026)

- dskit tiene su propio `Makefile` en la raíz para auto-mantenimiento
- `make setup` / `make update` / `make skills` desde la raíz de dskit
- `make recommended-tools` en proyectos generados muestra herramientas del ecosistema
- `make recommended-all` en proyectos generados como alias
- `uv pip install eticas-audit` para fairness/bias (ITACA)
- `npm install -g @synsci/openscience` para AI workbench científico
