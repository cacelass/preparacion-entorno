# Contribuir a dskit

Gracias por querer aportar. Esto es una plantilla de [copier](https://copier.readthedocs.io),
así que hay una regla que lo condiciona todo: **casi nada de lo que edites se
ejecuta tal cual**. Se renderiza primero con Jinja y luego corre dentro de un
proyecto generado. Esta guía es sobre todo cómo no tropezar con eso.

## Cómo está montado

```
copier.yml            Las 36 preguntas y su validación
template/             TODO se renderiza con Jinja (_templates_suffix: "")
  {{ project_slug }}/ El paquete del proyecto generado
  agents/             El arnés: 30 agentes, sus contratos y sus tests
  tests/              Los tests que hereda el proyecto generado
.github/scripts/      Los validadores de la plantilla (ver abajo)
```

`_templates_suffix: ""` significa que **cualquier** fichero de `template/` pasa
por Jinja: un `.py`, un `.yml`, un `.md`. Un `{{` accidental en un f-string
rompe la generación para todo el mundo.

## Antes de abrir un PR

```bash
uv run python .github/scripts/check_copier.py        # copier.yml bien formado
uv run python .github/scripts/pairwise.py --self-test # el generador de combos
uv run python .github/scripts/validate_template.py   # render + AST, 194 combos
```

`validate_template.py` es el que más atrapa: renderiza 194 combinaciones
all-pairs y valida el AST de los 343 ficheros resultantes sin instalar ni una
dependencia. Tarda menos de un minuto y no ensucia nada. **Ejecútalo siempre
que toques `template/`.**

## Probar cambios de verdad

Renderizar no basta: que un fichero parsee no quiere decir que funcione.

```bash
# Genera un proyecto real desde tu árbol de trabajo (--vcs-ref=HEAD es
# obligatorio: sin él copier usa el último tag y no verás tus cambios)
copier copy --trust --defaults --vcs-ref=HEAD . /ruta/con/espacio/proyecto

# Y ejecuta el CI que ese proyecto le entrega al usuario, sobre un clon limpio
python .github/scripts/run_generated_ci.py /ruta/con/espacio/proyecto
```

`run_generated_ci.py` commitea el proyecto, lo clona y ejecuta los pasos de su
`.github/workflows/ci.yml` en el clon. El clon es lo que ve GitHub Actions: si
el `.gitignore` se traga algo que el CI necesita, se cae ahí. Este script existe
porque dos bugs se colaron justo por ese hueco.

Genera en una partición con sitio (un proyecto con todos los extras pasa de
1 GB con su `.venv`). En `/tmp` es fácil quedarse corto.

## Tests del arnés

```bash
cd template
PYTHONPATH=. uvx --with pandas --with matplotlib --with scipy \
  --with scikit-learn --with chromadb \
  pytest agents/tests -c /dev/null --rootdir=agents
```

El `-c /dev/null` es necesario: el `pyproject.toml` de `template/` es Jinja y
pytest no puede leerlo.

**Sobre la plantilla sin renderizar hay ~36 fallos de base**, todos por
`agents/agents/test_agent.py`, que contiene `{% raw %}` y rompe el AST de
cualquier test que importe el registro completo de agentes. No son regresiones:
para saber si algo tuyo se ha roto, ejecuta solo los ficheros que tocaste.

## Estilo

- Ruff y mypy `--strict` (config en `template/pyproject.toml`). El paquete
  generado está limpio de errores de tipos: mantenlo así.
- Comentarios en español, explicando **por qué**, no qué. El CHANGELOG y los
  comentarios de este repo tienen ese registro; síguelo.
- Conventional Commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`.

## Añadir una opción a `copier.yml`

1. Añade la pregunta con `type`, `default` y `help`.
2. Si condiciona ficheros, añádelos a `_exclude` bajo `{% if not ... %}`.
3. Comprueba que `gen_smoke_matrix.py` la ejercita: falla a propósito si un
   flag opcional se queda sin cubrir en ningún combo.
4. Documéntala en la tabla de variables del `README.md`.

## Qué se agradece especialmente

- Bugs en el **proyecto generado**, no solo en la plantilla. Son los que peor se
  detectan y los que más molestan a quien la usa.
- Combinaciones de opciones que no habíamos probado.
- Migraciones para `copier update` cuando un cambio rompa proyectos existentes.
