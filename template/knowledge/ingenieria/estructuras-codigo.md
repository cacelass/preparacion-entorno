# Estructura de código y del proyecto de datos

Referencia para el lider sobre cómo organizar código y proyecto: layout,
fronteras de módulos, pipelines, configuración, dependencias, reproducibilidad,
patrones y empaquetado.

## Layout canónico de un proyecto DS

**Principio.** La estructura importa por tres razones técnicas: importabilidad,
testeabilidad y ausencia de path-magic. El layout decide si tu código es una
librería o un montón de scripts que solo se ejecutan desde la carpeta
equivocada.

**Práctica.** `src-layout` por encima de flat: el paquete vive dentro de
`src/`, se instala con `pip install -e .` y se importa por nombre, nunca por
ruta relativa.

```text
src/{{ project_slug }}/
    config.py          # parámetros y rutas
    data/              # ingesta y limpieza
    features/          # construcción de features
    models/            # entrenamiento, evaluación, predicción
    utils/             # helpers sin dominio
data/                  # datasets (crudo/ → intermedio/ → final/)
models/                # artefactos de modelo (binarios, metadatos)
tests/                 # espejo de src/ (test_data/, test_features/, ...)
notebooks/             # exploración; nunca código de producción
```

**Por qué.**

- **Importabilidad**: `from {{ project_slug }}.features import build` funciona
  desde cualquier cwd; con scripts sueltos hay que manipular `sys.path`.
- **Sin path-magic**: no hay `Path("../..")` para llegar a los datos; las rutas
  vienen de `config.py`, y las escrituras a `data/` y `models/` están
  explícitas.
- **Testeabilidad**: `tests/` importa el paquete instalado; nada depende de
  "estar en el directorio donde está el notebook".
- **Notebooks fuera del paquete**: los notebooks son exploración con estado,
  no código versionable de producción. Si una celda debe repetirse, va a
  `src/` con sus tests.

**Cómo falla.** Flat layout donde `import utils` funciona solo desde la raíz y
rompe en CI; notebooks que se convierten en el pipeline de producción porque
"ya funciona" y nadie sabe extraerlo.

## Fronteras de módulos: una responsabilidad por módulo

**Principio.** Separa las etapas del pipeline en módulos con responsabilidad
única y una interfaz explícita entre ellas. La separación es la que permite
cambiar una etapa sin reescribir las demás.

**Práctica.**

| Módulo | Responsabilidad | NO hace |
|--------|-----------------|---------|
| `config` | parámetros, rutas, defaults | lógica de negocio |
| `data` | leer, limpiar, validar esquema | entrenar |
| `features` | transformar datos en features | escribir modelos |
| `models` | entrenar, evaluar, persistir | limpiar datos |
| `evaluation` | métricas, comparativas, reporting | entrenar |

La interfaz entre etapas es la firma de la función pública, no un DataFrame
compartido "por convención". Si `features` necesita saber cómo entrena
`models`, la frontera está rota.

**Cómo falla.** Un `utils.py` que crece hasta 2000 líneas mezclando parseo de
fechas, cálculo de métricas y logging: cualquier cambio en una parte obliga a
releer todo el archivo y cualquier import arrastra dependencias que no quiere.

## Pipelines de datos como transformaciones deterministas

**Principio.** Un pipeline es una cadena de transformaciones puras por etapa:
misma entrada → misma salida. Cada etapa versiona su entrada y solo escribe
dentro de `data/`.

**Práctica.**

- Cada etapa lee de un path versionado (`data/intermediate/features_v3.parquet`)
  y escribe el suyo; si cambia el código, cambia la versión, no la ruta.
- Sin efectos secundarios fuera de `data/` y `models/`: no escribir archivos
  temporales en la raíz, no tocar `tests/`, no mutar el dataset de entrada.
- Los steps se encadenan por nombres (`make` targets o un runner del
  proyecto), no por un mega-script que hace todo.
- Un step determinista sobre el mismo input produce el mismo output: fija
  seeds, no dependas de orden de iteración sobre sets, evita timestamps en
  salidas intermedias.

**Cómo falla.** Un pipeline que lee "el CSV más reciente" y escribe sobre el
mismo `processed.parquet`: un re-run con datos nuevos mezcla versiones y el
modelo entrena sobre algo que nadie puede reconstruir.

## Configuración

**Principio.** Los parámetros que cambian entre entornos (rutas, tamaños,
hiperparámetros, features activas) son configuración, no código duro.

**Práctica.**

- **Config en fichero + CLI**: `config.yaml`/`config.py` con defaults, y la
  CLI o `hydra`/`tyro` permiten sobrescribir por experimento.
- **Inyección de dependencias sobre globals**: la función recibe su config
  (`build_features(cfg, data)`), no lee una variable global `CFG` que alguien
  mutó en otro módulo.
- **Variables de entorno para secretos**: tokens y credenciales en `env` (vía
  `.env` no versionado o el gestor de secretos del entorno), nunca en `config`.
- Si hay un solo proyecto y un solo entorno, la config no justifica un
  framework: un `dataclass` en `config.py` basta.

**Cómo falla.** Hiperparámetros hardcodeados en el notebook que el siguiente
experimento no puede variar sin editar código; o un `config.yaml` global que
se lee en 10 módulos distintos y que nadie sabe quién lo puebla.

## Dependencias: pyproject.toml, extras y lockfiles

**Principio.** Las dependencias se declaran por rol (runtime, dev, extras) y
se fijan con un lockfile para que cada ejecución sea reproducible.

**Práctica.**

- **`pyproject.toml`** es la fuente única: `[project]` con
  `dependencies` (runtime), `[project.optional-dependencies]` con `dev` y
  grupos por extra (`api`, `docker`, `mlflow`, `rag`, `optuna`, `monitoring`,
  `duckdb`).
- **Lockfile (`uv.lock`)**: fija la resolución exacta; se regenera con
  `uv lock` cuando cambia la spec. Instalar siempre desde el lock
  (`uv sync`), nunca `pip install` suelto.
- **Dev vs prod separados**: lo que es solo para testear/lintear vive en
  `dev`, no infla el entorno de producción.
- **Pin vs rangos**: en la librería, rangos amplios (`numpy>=2,<3`); en la
  app, el lock es la verdad. Los rangos son contrato, el lock es reproducibilidad.

**Cómo falla.** `pip install` de "una librería que me faltaba" que nunca llega
al `pyproject.toml`, y el entorno de otro que no arranca; o depender de una
feature "documentada" en una versión sin fijar que se rompe en la siguiente
release.

## Reproducibilidad

**Principio.** Un experimento reproducible se puede reconstruir a partir de:
código versionado, datos versionados y parámetros registrados. Sin los tres,
"el resultado es reproducible" es una afirmación no verificable.

**Práctica.**

- **Seeds fijas** para todo componente aleatorio (numpy, python `random`,
  sklearn, torch, transformers): una semilla por run, registrada.
- **Versión de datos registrada** en cada run: hash del dataset o id de
  versión de la etapa, no "el que había en `data/`".
- **Parámetros y métricas grabadas con cada run**: hiperparámetros, split,
  semilla, versión de código (`git rev-parse HEAD`) y métricas de resultado,
  en `mlflow` o en un JSON/csv de experimentos en `models/`.
- El pipeline fijo de seeds y orden de iteración produce resultados estables
  entre ejecuciones del mismo commit.

**Cómo falla.** Un experimento cuyo resultado "no se puede reproducir" porque
el notebook guardó `random` sin seed y los datos crudos fueron sobrescritos;
o un leaderboard de modelos sin registrar con qué versión de features se
obtuvo cada número.

## Patrones de diseño con relevancia en DS (y cuándo NO aplicarlos)

**Principio.** Los patrones resuelven variación real; aplicarlos a código con
una sola variante es sobreabstracción. Regla del proyecto: solo se extrae una
interfaz cuando hay ≥ 2 implementaciones reales.

**Práctica.**

| Patrón | Uso en DS | Cuándo NO |
|--------|-----------|-----------|
| Strategy | Familia de modelos intercambiables en un parámetro | Un solo modelo → una función |
| Factory | Registro de modelos por nombre (`model_registry["xgboost"]()`) | Un único constructor |
| Repository | Abstraer acceso a datos (local, S3, DuckDB) | Un solo backend → la función lee directo |
| Pipeline | Encadenar transformaciones sobre datos | Una cadena fija → una función secuencial |
| Null object | "No model" con `predict` devolviendo baseline | Basta `None` + chequeo en un sitio |

**Cómo falla.** `BaseModel` + `XGBoostModel` + `RandomForestModel` + fábrica
+ Strategy cuando el proyecto usa un solo modelo: cada abstracción añade
indirección, y la interfaz que "facilita el cambio" hace más difícil leer el
flujo real.

## Empaquetado para despliegue

**Principio.** El código de producción se despliega como un paquete instalable
con un entry point y una frontera de servicio explícita, no como scripts que
corren desde una carpeta.

**Práctica.**

- El módulo es un paquete instalable (`pip install -e .` para dev).
- **Entry points** en `pyproject.toml` para la CLI (`[project.scripts]`):
  `{{ project_slug }}-train = "{{ project_slug }}.models.train:main"`.
- **Frontera de servicio**: si hay API, un endpoint (`/predict`) que valida
  entrada, llama a la función de predicción y serializa respuesta; el modelo
  se carga una vez en memoria, no por request.
- El entorno de despliegue instala desde el lock; el contenedor (si `use_docker`)
  incluye el lock y la version del código (`git rev-parse HEAD`).

**Cómo falla.** Un "servicio" que ejecuta el notebook con `papermill` por
request, o un endpoint que recarga el modelo de disco en cada llamada porque
"es lo que funciona en local".

## De notebook a paquete mantenido sin sobre-ingeniería

**Principio.** La ruta de madurez es incremental: cada paso se justifica con un
dolor real, no con arquitectura por adelantado.

**Práctica.**

1. **Notebook → funciones**: extrae las celdas que se repiten a funciones puras
   en `src/`, con el notebook llamándolas.
2. **Notebook → pipeline**: cuando hay más de una etapa o más de un
   experimento, estructura el pipeline en etapas con `make`/runner.
3. **Pipeline → paquete**: cuando el código se consume fuera del notebook
   (tests, CI, API), empaqueta con `pyproject.toml` y `pip install -e .`.
4. **Prueba de madurez**: si puedes borrar los notebooks y el proyecto sigue
   funcionando (pipeline corre, tests pasan), el código está maduro.

**Cómo falla.** El paso 3 a destiempo (empaquetar un script de 200 líneas
porque "es lo pro"), o el estancamiento en el paso 1 donde el "proyecto" es
un notebook de 800 celdas que nadie puede re-ejecutar de forma confiable.

## Fuentes

- PEP 518 / PEP 621 — `pyproject.toml` (build system y metadata).
- Documentación de `uv` (pyproject, `uv.lock`, grupos de dependencias).
- Documentación de pytest (estructura de tests, fixtures) y setuptools
  (`src-layout`).
- Martin Fowler, "Refactoring" y el catálogo de patrones de diseño
  (Gamma et al., "Design Patterns").
- Documentación de Hydra/tyro (config + CLI) y pydantic (validación).
- MLflow documentation (tracking, artifact logging).
- Documentación de DVC (versionado de datos) y `git` (submódulos/LFS).
- Martin Kleppmann, "Designing Data-Intensive Applications" (O'Reilly, 2017).
