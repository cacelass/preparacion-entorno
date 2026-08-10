# Calidad de datos

## La premisa: el dato es el límite

Un modelo está acotado por sus datos, no por su algoritmo. Ninguna
arquitectura ni hipertuning recupera la información que el conjunto de
entrenamiento no contiene: si una feature clave está corrupta en el 30% de
las filas o el 10% de las etiquetas es ruido, la métrica del mejor modelo
posible queda por debajo de la que se consigue con los mismos datos limpios.
"Garbage in, garbage out" no es una anécdota: es la afirmación de que la
calidad del dato fija el techo de la métrica, y el algoritmo solo decide a
qué distancia de ese techo llegas.

Consecuencias operativas:

- El mayor retorno por hora invertida está en los datos, no en el modelo.
- El mayor riesgo de producción no es un modelo mal entrenado, es un dato
  que cambió en silencio (columna renombrada, unidad cambiada, join con
  duplicados) y degradó las predicciones sin que nada falle.
- Por eso la validación de datos es un control continuo de pipeline, no un
  paso de EDA que se hace una vez al empezar.

**Cómo falla.** El equipo optimiza el algoritmo hasta el último punto de
métrica mientras entrena sobre un dataset con un bug de featurization que
invalidaría el mejor de sus experimentos; o detecta el drift de datos tres
semanas después del deploy, cuando la degradación ya es un incidente.

## Contratos de datos

Un contrato de datos es la especificación ejecutable de lo que un dataset
debe cumplir: esquema de columnas, tipos, rangos, unicidad, valores
permitidos y conteos de filas. Es la interfaz entre el productor del dato
(upstream: un feed, una API, otro equipo, una extracción) y el consumidor
(el pipeline de features, el modelo). Su función es convertir el breakage
silencioso en ruidoso y temprano.

| Regla | Qué expresa | Ejemplo |
|-------|-------------|---------|
| Tipos | dtype o coerción válida | `order_id` es int, `fecha` es datetime |
| Rangos | límites físicos o de negocio | `monto` ∈ [0, 10_000] |
| Unicidad | claves sin duplicados | `order_id` único |
| Valores | dominio discreto cerrado | `region` ∈ {N, S, E, O} |
| Conteos | tamaño y proporciones estables | 100_000 ± 5% filas, nulls < 5% |

Por qué es la interfaz que evita el breakage silencioso: sin contrato, un
cambio upstream (columna renombrada, unidad de precio cambiada, categorías
nuevas) atraviesa el pipeline y produce predicciones válidas pero falsas.
Con contrato, la violación falla en la frontera del dato, donde el mensaje
de error todavía identifica la causa.

Dónde vive el contrato:

- **`data/`**: la definición del esquema, junto al dato que valida
  (`data/contracts.py` o `data/contract.yaml`). Es la fuente de verdad de
  qué se le permite al dato ser.
- **`tests/`**: las aserciones ejecutables (`tests/test_data_contract.py`)
  que corren en CI y en la puerta del pipeline.

```python
# data/contracts.py — el contrato como código
import pandera as pa

schema_pedidos = pa.DataFrameSchema(
    columns={
        "order_id": pa.Column(int, unique=True),
        "monto":    pa.Column(float, pa.Check.in_range(0, 10_000)),
        "region":   pa.Column(str, pa.Check.isin(["N", "S", "E", "O"])),
        "fecha":    pa.Column(pa.DateTime),
    },
    checks=pa.Check(lambda df: len(df) > 0, name="no_vacio"),
)
```

**Cómo falla.** Contrato sin dueño que nadie actualiza cuando el negocio
cambia de verdad (y empieza a fallar en falso, o se ignora); o contrato solo
en un notebook que no corre ni en CI ni en el pipeline y por tanto no
protege nada.

## Herramientas de validación

| Herramienta | Modelo | Ventaja | Coste |
|-------------|--------|---------|-------|
| pandera | Esquemas como código | Tipos y checks, falla con SchemaError | Definirlo es tuyo |
| Great Expectations | Expectativas + data docs | Suites y docs HTML compartibles | Pesado de operar |
| pytest | Aserciones sobre el DataFrame | Cero deps, corre en CI | Manual, sin esquema |

Elige según el proyecto: pandera para contratos de esquema estrictos dentro
del código; Great Expectations si necesitas suites declarativas y data docs
compartidas; pytest para validaciones puntuales y tests de regresión de
bugs de datos. No son excluyentes: un contrato pandera puede envolverse y
llamarse desde un test pytest.

Qué assertar sobre un dataset, en orden de prioridad:

- **Shape**: filas y columnas esperadas (detecta truncado o join explotado).
- **dtypes**: tipos esperados por columna (detecta coerción rota).
- **Ratio de nulls**: por columna, contra umbral (detecta fuente caída).
- **Rangos de valores**: máximos, mínimos, dominios (detecta unidades rotas).
- **Cardinalidad**: número de valores únicos de claves y categorías (detecta
  duplicados o dict de lookup roto).
- **Drift vs referencia**: distribución de cada feature contra la del
  dataset de entrenamiento (detecta cambio de población).

```python
# tests/test_data_contract.py
import pandas as pd
from {{ project_slug }}.data.contracts import schema_pedidos
from {{ project_slug }}.utils.paths import PROCESSED_DATA_DIR


def test_contrato_procesado():
    df = pd.read_csv(PROCESSED_DATA_DIR / "dataset.csv")
    schema_pedidos.validate(df)                    # levanta SchemaError
    assert df.shape[0] > 0 and df.shape[1] == 4
    assert df["monto"].isna().mean() < 0.05
```

## Profiling: explorar antes de validar

El profiling produce resúmenes automáticos del dataset (pandas-profiling /
ydata-profiling: reporte HTML con missingness, estadísticos, correlaciones,
histogramas, detección de duplicados). Se genera una vez, se lee, se
descartan las tablas irrelevantes y se fijan los umbrales del contrato.

Qué mirar en el reporte:

- **Patrones de missingness**, no solo conteos: ¿el null es aleatorio
  (MCAR), depende de otras columnas (MAR), o del valor que falta (MNAR)?
  La respuesta decide si imputar es legítimo.
- **Skew**: variables asimétricas (ingresos, tiempos) sugieren log o
  winsorización antes del modelo.
- **Outliers**: extremos físicos vs errores de captura; se deciden en la
  política, no se borran del notebook.
- **Duplicados**: filas idénticas vs claves repetidas con valores distintos;
  cada caso tiene una respuesta distinta.
- **Cardinalidad**: categorías de alta cardinalidad (IDs en columna
  categórica), columnas constantes (sin información), categóricas con miles
  de valores (target encoding vs hashing).

La diferencia entre explorar y validar es de propósito y periodicidad:

| | Profiling | Validación |
|---|-----------|------------|
| Pregunta | ¿Qué hay aquí? | ¿Violó el contrato? |
| Periodicidad | Una vez, al conocer el dato | Cada vez que corre el pipeline |
| Salida | Reporte para humanos | Fallo/éxito para CI |
| Rol | Fija hipótesis y umbrales | Ejecuta los umbrales |

**Cómo falla.** Confundirlos: correr validación "una vez en el notebook" y
creer que el dato está protegido, o usar un profile report gigante como
sustituto del contrato ejecutable. El profiling descubre, la validación
protege.

## Missing, duplicados y outliers: una política, no improvisación

Cada decisión sobre missing, duplicados y outliers es una **decisión de
modelado**: cambia la distribución que el modelo aprende y la que verá en
producción. Por eso se decide, se registra y se codifica una vez, y se
aplica igual en train y en serve. Improvisar por notebook produce pipelines
incoherentes entre sí y entrenamiento/servicio desalineados.

- **Imputación de missing**: la estrategia se elige sobre train y se ajusta
  sobre train (media, mediana, moda, regresión, indicador de missingness).
  Nunca se calcula sobre test ni sobre datos de producción; si la imputación
  usa estadísticas, esas estadísticas son parte del artefacto y viajan con
  él. Imputar con la media global filtrada por el dataset de test es
  leakage en forma de sesgo.
- **Política de duplicados**: decidir qué cuenta como duplicado (clave,
  fila completa, ventana temporal) y si se conservan o se eliminan. Los
  duplicados inflan el peso de las filas repetidas en la función de pérdida
  y alteran el balance de clases: la decisión es del equipo, no del azar de
  un join.
- **Política de outliers**: winsorizar (recortar extremos a un percentil)
  frente a conservar. Winsorizar estabiliza los modelos sensibles a la
  escala (regresiones, distancias) a costa de perder las colas; conservar
  preserva los casos raros pero puede dominar la escala de las features.
  `sklego.preprocessing.Winsorizer` y `OutlierRemover` (scikit-lego)
  implementan ambas en un paso del pipeline.

Por qué cada una es una decisión de modelado, no de limpieza:

- Imputar mediana asume que el valor ausente no es informativo; imputar con
  indicador de missingness asume lo contrario. Las dos entrenan modelos
  distintos.
- Duplicados en la muestra de entrenamiento cambian el prior de la clase.
- Winsorizar o no cambia la cola de P(X) que el modelo ve, y por tanto qué
  pregunta puede responder sobre los casos extremos.

La política queda registrada (en el contrato, en el README de `data/` o en
`references/02-eda.md`) y codificada en el pipeline de features, para que
sea reproducible y revisable.

**Cómo falla.** Imputación calculada dentro del pipeline de train y
recalculada distinto en serve; o "me he encontrado outliers, los he
borrado a mano en el notebook" que nadie puede replicar ni defender.

## Versionado y reproducibilidad

Un dataset sin versión no es reproducible: el run de hoy y el de la semana
pasada entrenaron sobre cosas distintas y no puedes saber cuál. El mínimo
viable es un **manifest** por dataset: nombre, versión, hash, fecha,
conteo de filas/columnas, fuente y contrato aplicado.

```yaml
# data/manifests/dataset_raw.yaml
dataset: dataset.csv
version: "2024-03-01"
source: kaggle://owner/dataset
hash_sha256: 4f2a8c...
rows: 125_000
cols: 24
contract: pedidos_v1
```

Prácticas:

- **Raw inmutable**: `data/raw/` solo se escribe al descargar; nunca se
  modifica in-place. El procesado (`data/processed/`) se deriva y se puede
  regenerar desde raw; si cambias el pipeline, el artefacto cambia de
  versión.
- **Snapshot o hash**: un hash SHA-256 por archivo (o por carpeta) permite
  detectar que el dato cambió sin que nadie lo registrara. Barato y
  suficiente para la mayoría de los proyectos.
- **DVC-style** (git en el dato, `dvc add`/`dvc push`): necesario cuando el
  dataset es grande, vive en remoto o quieres versionar el dato junto al
  código con time-travel. Un manifest es la alternativa ligera.
- **Registrar la versión del dataset en cada run**: la versión de datos
  entra en el artefacto del modelo (con MLflow, en los parámetros del run, o
  en un `run_metadata.json`). Si no puedes decir qué datos alimentaron un
  run, los resultados no son reproducibles.

El dato sin tests acumula deuda técnica exactamente igual que el código sin
tests: se paga en debugging de una semana cuando upstream cambia y nadie se
entera. La validación del contrato es la forma de pagar esa deuda al día
(ver [deuda-tecnica.md](deuda-tecnica.md)).

**Cómo falla.** `data/processed/` sobrescrito en cada ejecución sin registro;
dos investigadores que entrenan el mismo código sobre versiones distintas
del dato y comparan métricas como si fueran comparables; el modelo en
producción sin constancia de qué versión de datos lo entrenó.

## Lineage

Lineage es la respuesta a tres preguntas sobre cada dataset: **de dónde
vino** (fuente), **qué lo transformó** (pipeline, código, versión) y **quién
lo produjo** (proceso o persona). Se registra en el manifest por campo de
origen y por paso de transformación.

Por qué el lineage es lo que hace mantenible un pipeline:

- Cuando una métrica aguas abajo cambia, el lineage permite decidir en
  minutos si la causa es un cambio de dato o un cambio de código, en vez de
  investigar toda la cadena.
- Cuando entra una fuente nueva, sabes exactamente qué datasets y modelos
  dependen de ella.
- Cuando el dato falla, la pregunta "¿de dónde salió esto?" tiene respuesta,
  que es lo que separa un pipeline mantenible de un misterio.

Mínimo registrable por artefacto de datos: `source` (URL o tabla), `transform`
(nombre y versión del script o función), `producer` (job/agente/run) y
`timestamp`. Los pipelines declarativos (DVC, dbt, workflows del arnés)
generan parte del lineage automáticamente: es gratis, no lo pagues dos
veces.

**Cómo falla.** Datasets huérfanos (nadie sabe de dónde vienen), transform
que fue borrado y regenerado sin registro, y la pregunta "¿por qué el
pipeline de hoy da distinto que el de ayer?" sin respuesta posible.

## Train/serve y train/future: consistencia

El modelo se entrena sobre la distribución histórica y predice sobre la que
llegue después. El desajuste entre ambas es el fallo más común y más
silencioso en producción. Dos tipos de drift, con causas y acciones
distintas:

| Drift | Qué cambia | Señal típica | Acción |
|-------|-----------|--------------|--------|
| Covariate | P(X): la distribución de las features | KS, PSI, distancia | Reentrenar / reweight |
| Concept | P(y|X) cambia | Degradación de métrica con ground truth | Re-etiquetar |

El **dataset de referencia** es el snapshot de la distribución de
entrenamiento: la referencia contra la que se compara cada ventana nueva.
Sin referencia fija, "hay drift" no significa nada: siempre lo hay respecto
a algo.

Señales de detección de drift:

- **Population Stability Index (PSI)**: mide el desplazamiento entre dos
  distribuciones sobre los mismos bins.

```
PSI = Σ_i (p_i − q_i) · ln(p_i / q_i)
```

  Regla práctica: PSI < 0.1 estable, 0.1–0.25 cambio moderado, > 0.25
  cambio significativo. Simple y la industria lo entiende.
- **Kolmogorov-Smirnov (KS)**: compara empíricamente dos distribuciones;
  sensible a forma y localización. Para numéricas.
- **Chi-cuadrado**: compara frecuencias de categorías. Para categóricas.
- **Distancias**: Wasserstein o MMD para detectar cambios finos entre
  distribuciones sin supuestos paramétricos.

{% if use_monitoring %}
Este proyecto incluye `monitoring/monitor.py` como hook de
monitorización: drift KS/chi² entre la referencia (X_train) y los datos
actuales, y degradación de métricas frente al baseline, vía `make monitor`.
Genera `reports/monitoring/drift_report.csv` y `drift_report.html`. Se
ejecuta sobre los mismos datos que consume la API o el pipeline, y su
salida es la señal para decidir reentrenar. Detectar no es reparar: la
alerta solo tiene valor si va seguida de una acción (retraer el modelo,
reentrenar, revisar la fuente).
{% endif %}

**Cómo falla.** Confundir covariate y concept drift (reentrenar un modelo
cuyas etiquetas ya no valen, o etiquetar de nuevo un modelo cuya población
cambió); o medir drift sin referencia fija y sin umbral calibrado, con
alertas que nadie lee.

## En la práctica

**Puerta de validación en el pipeline.** Validar justo tras cada salida de
datos (raw tras descargar, processed tras construir features) y **fallar
rápido**: el costo de un dataset roto crece con la profundidad del pipeline
(raw → features → modelo → deploy). Un `SchemaError` en `data/processed/`
cuesta minutos; el mismo dato roto detectado en producción cuesta días.
La puerta corre en CI y en el run del pipeline: no es un paso manual.

**Regla: añade un test de datos cada vez que un bug se repite.** Un bug de
datos que reaparece (fecha ilegible, encoding roto, join con duplicados)
significa que el contrato no lo cubría. Se escribe un test que lo
reproduzca y el contrato se amplía; la segunda vez ya no debe ocurrir.
Es la misma mecánica que "añade un test cuando encuentras un bug", aplicada
al dato: cada regresión de datos se convierte en parte permanente de la
suite.

```python
# tests/test_data_bugs.py — cada bug de datos repetido, aquí
def test_fechas_ilegibles_son_error():
    # bug visto 2 veces: "2024-13-01" pasaba el contrato de tipo
    fechas = pd.to_datetime(df["fecha"], errors="coerce")
    assert fechas.isna().mean() == 0
```

**El orden: contrato → profiling → política → versionado.**

1. **Contrato**: define el esquema y qué no puede cambiar en silencio.
2. **Profiling**: descubre qué hay en el dato y fija los umbrales realistas
   del contrato (no inventados).
3. **Política**: decide missing, duplicados y outliers; queda registrada y
   codificada.
4. **Versionado**: deja el registro reproducible de cada dataset y cada run.

En ese orden cada paso informa al siguiente; invertirlo produce contratos
inventados, políticas ad hoc y datasets sin historia.

{% if use_duckdb %}
Este proyecto carga `data/raw/` con DuckDB
(`{{ project_slug }}.data.make_dataset.load_data_duckdb`): SQL directo sobre
CSV/Parquet/JSON sin cargar todo en memoria. Sirve también como primera
capa de validación barata sobre archivos grandes: conteos, cardinalidades
y nulls por consulta antes del `SELECT *` que alimenta el contrato.

```python
from {{ project_slug }}.data.make_dataset import load_data_duckdb

df = load_data_duckdb(
    "dataset.csv",
    query="SELECT order_id, monto, region FROM datos WHERE monto IS NOT NULL",
)
```
{% endif %}

## Fuentes

- "Moving Fast With Broken Data" (Shreya Shankar, Labib Fawaz, Karl
  Gyllstrom, Aditya Parameswaran; validación de datos en pipelines de ML de
  Meta) — arXiv:2303.06094 — https://arxiv.org/abs/2303.06094
- "Operationalizing Machine Learning: An Interview Study" (Shankar, Garcia,
  Hellerstein, Parameswaran; velocity, validation y versioning como ejes
  del ML operativo) — arXiv:2209.09125 — https://arxiv.org/abs/2209.09125
- "Not Your Usual Type(s): Data contracts as types across languages and
  engines" (Montana, Marc, Bigon, Tagliabue; contratos de datos como
  tipos en el lakehouse) — arXiv:2607.13339 — https://arxiv.org/abs/2607.13339
- "Building a Correct-by-Design Lakehouse. Data Contracts, Versioning, and
  Transactional Pipelines for Humans and Agents" (Sheng, Wang, Barros,
  Montana, Tagliabue, Bigon) — arXiv:2602.02335 — https://arxiv.org/abs/2602.02335
- "The ML Test Score: A Rubric for ML Production Readiness and Technical
  Debt Reduction" (Breck, Cai, Nielsen, Salib, Sculley, IEEE BigData 2017)
  — DOI 10.1109/BigData.2017.8258038 — https://doi.org/10.1109/BigData.2017.8258038
- Documentación de pandera (schemas como código) — https://pandera.readthedocs.io/
- Documentación de Great Expectations (expectativas y data docs) — https://docs.greatexpectations.io/
- Documentación de scikit-lego (Winsorizer, OutlierRemover) — https://scikit-lego.readthedocs.io/
- Documentación de ydata-profiling (profiling automático) — https://docs.profiling.ydata.ai/
- Documentación de DVC (versionado de datos) — https://dvc.org/doc
