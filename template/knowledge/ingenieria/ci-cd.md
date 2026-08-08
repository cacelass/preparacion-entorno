# CI/CD en proyectos de datos

## CI de código vs CI de ML

Un test de código es un veredicto binario: el código hace lo que la
especificación dice, o no. Un test de ML verifica que un sistema — dato +
pipeline + modelo — sigue produciendo lo que se espera cuando el dato y el
entorno cambian, y ese veredicto es gradual: el código se rompe, el modelo se
degrada. La diferencia cambia qué se vigila y con qué umbrales:

| Qué vigila | CI de código | CI de ML |
|-----------|--------------|----------|
| Pregunta | ¿Se rompe? | ¿Se degrada? |
| Fallo | binario, determinista | gradual, probabilístico |
| Contra qué se compara | la especificación | un baseline fijado y una referencia de datos |
| Garantía | ejecuta lo que dice | sigue produciendo lo esperado bajo cambio |
| Cobertura | líneas ejecutadas | mutantes muertos + goldens + slices |
| Determinismo | mismo input → mismo output | exige fijar semillas; la variación es un bug |

Consecuencia operativa: la suite de ML tiene capas (datos, modelo,
infraestructura, monitorización) y cada una responde una pregunta distinta.
"¿Falló el test?" no es una pregunta binaria sobre un commit: es la pregunta
por el estado del sistema. Detalle en `ml/testing-ml.md`.

## Etapas de un pipeline CI

El orden importa por dos reglas: lo barato primero, y lo que falla antes
primero. Un fallo de lint cuesta segundos; un fallo de golden eval cuesta
minutos y no debería bloquear un diff con un typo.

```
lint/format → typecheck → tests unitarios → (ML) tests de datos + tests de
modelo + eval golden → build/artefacto
```

| Etapa | Coste | Qué falla | Señal típica |
|-------|-------|-----------|--------------|
| lint/format | segundos | estilo, imports, secretos | diff de un `ruff check` |
| typecheck | segundos | contratos de firma, `Any` que se cuela | error de mypy en CI |
| tests unitarios | minutos | lógica rota, transformaciones | `pytest` en rojo |
| tests de datos | minutos | schema, invariantes, fugas | contrato pandera violado |
| tests de modelo | minutos | invariantes de salida, goldens | `pred == 0.5` fuera de rango |
| eval golden | minutos | recuperación degradada, regresión | `make eval-rag` cae |
| build/artefacto | variable | empaquetado, lock, imagen | `uv sync --frozen` falla |

Las capas de ML no son un "adicional" al final: van donde van porque son las
que convierten un test binario en una medida de degradación. Saltárselas es
un CI de código aplicado a un sistema ML.

## El workflow que ya trae el proyecto

El CI del template es de tres niveles, cada uno prueba una cosa distinta, y
el que decide qué se prueba es el **generador**, no el CI. En orden:

1. **`validate-template`** — render + AST de TODOS los combos de opciones.
   `check_copier.py` valida `copier.yml`, `pairwise.py --self-test` se
   autotesta, y `validate_template.py` renderiza cada combinación de flags y
   comprueba el AST (sintaxis, imports, código muerto). Si el generador se
   rompe en silencio, la cobertura de flags cae y todo sigue en verde: por eso
   el job lleva autotest. **Atrapa**: un template que ya no renderiza, un
   combo de opciones que produce código roto.
2. **`matriz` → `smoke`** — un proyecto generado por combo de flags. La
   matriz se construye con `gen_smoke_matrix.py`, que **falla si un flag
   opcional no se ejercita en ningún combo** (cobertura de flags obligatoria).
   Cada job `smoke` hace `copier copy`, `uv sync` con los extras del combo y
   `pytest -m smoke` (más ruff, bandit, pip-audit). `fail-fast: false`: un
   combo roto no cancela el resto. **Atrapa**: que un proyecto generado
   arranque y que cada extra tenga al menos un test.
3. **`ci-generado`** — ejecuta el workflow que se le entrega al usuario sobre
   un clon limpio del proyecto generado. Como el sync post-generación es
   opt-in, primero crea `uv.lock` y luego `run_generated_ci.py` corre el
   `ci.yml` del proyecto generado tal cual. **Atrapa**: que el CI del usuario
   pase. El workflow del usuario llevó roto dos veces (lock en `.gitignore`,
   mypy `--strict` con 193 errores) sin que ningún otro job lo notase; "lo que
   se rompa se rompe aquí, no en el repositorio de un usuario".

La regla que estructura todo: **el CI no decide qué se prueba — el generador
sí.** El CI solo ejecuta lo que el generador emite (combos, flags, extra
sufijo). Si quieres que algo nuevo se pruebe, se añade al generador; añadirlo
solo al workflow no basta.

## Puertas: init.sh y CI

Hay dos puertas, y ambas son código, no instrucciones:

| Puerta | Dónde | Qué decide | Coste |
|--------|-------|-----------|-------|
| `./init.sh` | local | ¿se puede trabajar? | segundos, antes de cada feature |
| CI (`ci.yml`) | remoto | ¿se puede mergear? | minutos, en cada push/PR |

`init.sh` es el gate local: verifica entorno, estructura, backlog y suite. En
CI se ejecuta como job `Harness gate` con `--quick` (los tests ya corrieron
arriba; sin `--quick` el fallo saldría dos veces). La regla del arnés aplicada
a CI:

> **Si un fallo se cuela dos veces, se automatiza.**

Un bug de datos que reaparece se convierte en test; un check que el lider
aplica de memoria se convierte en `init.sh`; un workflow entregado que no
pasa se convierte en `ci-generado`. Lo que no es código se viola siempre en
la semana 6.

## Pre-commit: los hooks locales

El proyecto trae `.pre-commit-config.yaml` con los hooks locales de ruff
(`--fix` + format), isort, bandit, mypy (`--strict`) y los genéricos
(trailing whitespace, EOF, YAML/TOML válidos, ficheros grandes, conflictos de
merge). bandit y mypy corren solo en `stages: [push]`.

Qué merece un hook (regla práctica):

- **Sí**: lint rápido (ruff, isort), formato (black/ruff-format), secretos
  (detect-secrets), ficheros rotos (YAML/TOML), conflictos de merge. Son
  baratos, deterministas y no requieren entorno.
- **No**: tests lentos, entrenamiento, eval de modelo, `uv sync`. Un hook que
  tarda minutos se desactiva; uno que necesita el venv falla en máquinas
  limpias.

Los hooks son la primera línea y son **locales**: lo que no se configura en
CI puede saltárselo cualquiera con `--no-verify`. Por eso el pre-commit es un
acelerador del feedback, y la puerta real sigue siendo el CI.

## Secretos en CI

- **GitHub Secrets** (`Settings → Secrets and variables → Actions`), nunca
  hardcodear tokens en el workflow ni en el código. Un secreto en el diff es
  un secreto comprometido: hay que rotarlo, no borrarlo.
- **Mínimo privilegio**: cada secreto va al job que lo necesita y con el
  scope mínimo (`permissions: contents: read` por defecto si se declara
  `permissions:` explícito). No dar `GITHUB_TOKEN` con write a jobs de lint.
- **El CI no es un entorno para datos sensibles.** Es un runner efímero,
  compartido, con salidas visibles en los logs. Los datasets que se montan
  ahí son fixtures pequeños, no datos de producción; si el CI necesita datos
  reales, los consume por artefacto/referencia con acceso restringido y se
  borran al terminar el job.
- **Logs**: las salidas de comandos que tocan secretos se redactan
  (`::add-mask::` o el mecanismo equivalente) antes de imprimir.

## Éxito y velocidad

Un CI que tarda 40 minutos no se ejecuta: se convierte en un merge directo
con `--no-verify` de mentalidad. El presupuesto real es que el feedback
llegue antes de que el desarrollador cambie de contexto (minutos, no horas).
Tres palancas:

- **Partición**: correr solo lo que cambió. Con rutas de trigger por job
  (`paths:`, `paths-ignore:`) el lint de `{{ project_slug }}/` no espera a los
  tests de `agents/`. En monorepos, afectación por directorio (targets
  afectados, no toda la suite).
- **Caché**: `astral-sh/setup-uv` cachea uv y su `uv.lock`; pip cache y
  wheels también. El coste dominante de un job Python es instalar
  dependencias, no ejecutar los tests.
- **No entrenar en CI, salvo smoke**: un job de entrenamiento de 30 minutos
  por push no es un gate, es una parálisis. El entrenamiento completo vive en
  pipelines bajo demanda (o en `make train`); en CI solo se entrena si es un
  smoke controlado (dataset mini, pocas epochs, semilla fija) para validar
  que el pipeline arranca. La eval golden usa artefactos precomputados.

Regla: el CI es un **gate de regresión**, no una fábrica de artefactos. Si un
job produce algo que no bloquea a nadie, está en el lugar equivocado.

## ML en CI

El pipeline de datos se valida como primera clase del CI: el contrato de
schema (pandera) corre en cada push y una violación deja de ser silenciosa;
los tests de datos (invariantes, sin fugas, slices) y de modelo (invariantes
de salida, goldens) corren con el resto de la suite. Detalle en
`data/calidad-datos.md` y `ml/testing-ml.md`. La eval golden del RAG mide si
la búsqueda encuentra lo que debería (hit_rate, recall@k, MRR) contra
`agents/evals/rag_golden.json` — es lo que convierte "parece que ahora busca
mejor" en un número comparable entre commits.

El **artefacto** que el CI debe proteger:

- `uv.lock` versionado y sincronizado con `--frozen` (reproducción exacta; si
  el lock no viaja con el código, "funciona en mi máquina" vuelve).
- Semillas fijas (`random`, `numpy`, `torch`) para que el determinismo sea
  testable; el test de pipeline determinista (mismo input → mismo output)
  corre en CI.
- El manifest del dato y el commit que lo produjo, registrados en el
  artefacto de modelo.

### RAG golden eval
{% if use_rag %}
```yaml
  rag-eval:
    runs-on: ubuntu-latest
    needs: quality
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: Install dependencies
        run: uv sync --extra dev --extra {{ ml_type }} --extra rag --frozen
      - name: RAG golden eval
        run: make eval-rag
```
La eval se ejecuta en un job separado, no en el job principal: su coste (indexar
+ consultar) no debe ralentizar el gate de regresión, y su umbral es una
propiedad del RAG, no del código.
{% endif %}

### Mutation / CRAP como gate
{% if use_sdd %}
```yaml
  mutation:
    runs-on: ubuntu-latest
    needs: quality
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: Install dependencies
        run: uv sync --extra dev --extra {{ ml_type }} --frozen
      - name: Mutation testing
        run: |
          uv run python -m agents --json run mutation run_mutation_testing \
            --target {{ project_slug }}/features/build_features.py
          uv run python -m agents --json run mutation crap_report \
            --target {{ project_slug }}/utils.py
```
La mutación mide que los tests "muerden", no solo que cubren líneas: un
mutante que sobrevive es código que los tests no protegen. Como gate de CI,
se aplica a módulos críticos y baratos de mutar; aplicarla a todo el repo
convierte el CI en un costo que nadie corre. Detalle en `ml/testing-ml.md` →
"Mutación aplicada a pipelines".
{% endif %}

### Docker build
{% if use_docker %}
```yaml
  docker-build:
    runs-on: ubuntu-latest
    needs: quality
    steps:
      - uses: actions/checkout@v4
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      - name: Build image
        uses: docker/build-push-action@v6
        with:
          context: .
          push: false
          tags: {{ project_slug }}:ci
```
Un job de build atrapa lo que los tests no: el Dockerfile roto, la imagen
que no se construye, el artefacto que no viaja al contenedor. `push: false`
en CI: publicar es decisión explícita del deploy, no del gate. Detalle en
`backend/docker.md`.
{% endif %}

## Fallos típicos

- **Tests flaky**: orden de ejecución no determinista, aleatoriedad sin
  semilla, `set`/`dict` iterados, test que depende del estado de otro. El
  síntoma es un CI que falla sin que nada cambiara; la solución es fijar
  semilla y orden, y si un test flakea, se arregla antes de seguir.
- **Jobs que dependen de red**: descargas de datasets, llamadas a APIs,
  `pip install` flotante. El runner puede no tener acceso, o la fuente puede
  no existir mañana. Los datos se fijan como fixture/artefacto; las
  dependencias se pin en el lock.
- **Datasets que no viajan**: el test pasa localmente porque el dataset está
  en `data/` de tu máquina, y en CI no existe. Los fixtures viven en el repo
  (`tests/data/`), pequeños y deterministas; los datos grandes entran por
  artefacto/referencia.
- **El anti-patrón: "CI en verde pero producción rota".** El CI prueba el
  código, no el servicio; sin un paso de smoke/serving (arrancar la app o el
  modelo y pegarle una petición real) el CI puede pasar mientras producción
  falla. El smoke mínimo: importar el paquete, cargar el modelo, hacer una
  predicción de ejemplo.

## Práctica: cómo extender el workflow

Reglas mínimas de un buen pipeline:

1. **Una cosa por job, nombre legible** (`lint`, `typecheck`, `rag-eval`):
   el fallo se lee en el título, no en el log.
2. **`fail-fast: false`** en matrices: un combo roto no oculta el resto.
3. **Lo barato y lo que falla antes, primero**: lint y tipos antes que eval.
4. **Gates bloqueantes, mediciones no**: un job que solo reporta
   (`continue-on-error: true`) no es una puerta; si no bloquea, no está
   protegiendo nada.
5. **El fallo se reproduce en verde**: cada regresión se convierte en un
   test, no en una nota en el commit.
6. **Sin entrenamiento en el gate** salvo smoke controlado; el artefacto
   (lock, seeds, manifest) se protege, no se fabrica en CI.

Para añadir un job: copia la estructura del job `quality` (checkout → setup-uv
→ `uv sync --frozen`) y añade tu paso. Tres extensiones típicas que ya vienen
esbozadas arriba y activas según el extra elegido: `rag-eval` (eval golden
del RAG, con `use_rag`), `mutation` (gate de mutación, con `use_sdd`) y
`docker-build` (con `use_docker`). Si añades un extra nuevo al template,
recuerda que
"el que decide qué se prueba es el generador": actualiza `gen_smoke_matrix.py`
para que el flag se ejercite, o la cobertura de flags del job `matriz`
empezará a fallar.

## Fuentes

- Sato, D., Wider, A., Windheuser, C., *Continuous Integration for Machine
  Learning* (ease.ml/ci). arXiv:1903.00278. https://arxiv.org/abs/1903.00278
- Breck, E., et al., *The ML Test Score: A Rubric for ML Production
  Readiness and Technical Debt Reduction*. arXiv:1706.08568.
  https://arxiv.org/abs/1706.08568
- Documentación de GitHub Actions (workflows, jobs, matrices, triggers,
  Secrets). https://docs.github.com/actions
- Documentación de DVC (pipelines y versionado de datos) y CML (CI para ML).
  https://dvc.org/doc y https://cml.dev
- Documentación de pre-commit (hooks, `stages`, `--no-verify`).
  https://pre-commit.com
