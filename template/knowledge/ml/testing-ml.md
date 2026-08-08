# Testing de sistemas ML

## Por qué testear ML es distinto de testear código

Un test de código verifica que el código hace lo que su especificación dice.
Un test de ML verifica que un sistema —dato + pipeline + modelo— sigue
produciendo lo que se espera de él cuando el dato y el entorno cambian. Tres
diferencias que cambian el diseño de la suite:

- **El dato es input.** El contrato de entrada no lo define el código, lo
  define la realidad upstream: una columna que se renombra, una unidad que
  cambia, un join que se duplica. La "especificación" del sistema incluye
  propiedades del dato que hay que descubrir y fijar, no leer de una
  interfaz. Por eso los tests de datos son tests de primera clase, no un
  subproducto del EDA.
- **El modelo no es determinista por naturaleza.** No hay una salida
  "correcta" esperada para una entrada genérica; la aserción se hace sobre
  propiedades y sobre distribuciones, no sobre valores. Un test que afirme
  `pred == 0.5` es en general incorrecto: lo que se puede afirmar es que la
  salida está en [0,1], que respeta monotonicidad, que las métricas sobre un
  conjunto fijo se mantienen dentro de un intervalo, o que dos run con la
  misma semilla coinciden.
- **La cobertura por líneas no prueba que la lógica de datos es correcta.**
  Un pipeline puede tener el 100% de sus líneas ejecutadas y estar
  procesando un dato equivocado (columna mal mapeada, imputación con
  leakage) sin que ningún test falle. La cobertura mide qué código se
  ejecutó, no si los datos que pasan por él son los que el negocio espera.

Consecuencia operativa: la suite de ML tiene **capas** — datos, modelo,
infraestructura, monitorización — y cada una se diseña con una pregunta
distinta. "¿Falló el test?" no es una pregunta binaria sobre un commit: es la
pregunta por el estado del sistema completo.

## La taxonomía del ML Test Score

Breck et al. (2017) proponen una rúbrica de preparación de producción con
cuatro categorías de tests y umbrales (v1: 20 puntos mínimos, v2: 37). La
estructura importa más que los umbrales:

| Categoría | Qué cubre | Pregunta que responde |
|-----------|-----------|------------------------|
| Tests de datos | Contratos, invariantes, fugas, etiquetas, slices | ¿El dato es el que el modelo espera? |
| Tests de modelo | Invariantes, goldens, calibración, paridad | ¿El modelo se comporta como se diseñó? |
| Tests de infraestructura | Determinismo, versionado, gates de CI | ¿El sistema es estable y reproducible? |
| Monitorización | Drift, degradación, alertas accionables | ¿El comportamiento cambia en producción? |

Las dos últimas son las que más se omiten y las que más deuda técnica evitan:
sin tests de infraestructura no se puede decir que un run sea reproducible, y
sin monitorización el primer síntoma de un problema es un incidente, no un
alerta. En este proyecto, los tests de infraestructura viven en la puerta
(`./init.sh` → suite + checks) y la monitorización depende de
`use_monitoring`.

## Tests de datos

Se apoyan en `data/calidad-datos.md` (contratos, validación, versionado,
drift). Aquí, la parte que es testing propiamente:

### Contrato de esquema

El contrato (`data/contracts.py`, schema pandera) es la especificación
ejecutable del dataset. Su test corre en CI y en la puerta del pipeline; una
violación deja de ser silenciosa. Detalle en
`data/calidad-datos.md` → "Contratos de datos".

### Invariantes de distribución

Fijar la referencia (la distribución de entrenamiento) y comparar cada
versión nueva del dato contra ella. Señales: KS para numéricas, chi-cuadrado
para categóricas, PSI con umbrales calibrados (estable < 0.1, moderado
0.1–0.25, fuerte > 0.25). Un test de distribución no debería tener umbrales
inventados: se fijan tras el EDA y se revisan cuando cambia el negocio.

```python
from scipy.stats import ks_2samp

def test_distribucion_estable():
    ref = pd.read_parquet("data/reference/monto.parquet")
    nuevo = pd.read_parquet("data/processed/monto.parquet")
    _, p = ks_2samp(ref, nuevo)
    assert p > 1e-3, "cambio de distribución en 'monto' sin registrar"
```

### Ausencia de fugas entre splits

La fuga clásica y testable: filas duplicadas o casi-duplicadas presentes a la
vez en train y en test. La intersección debe ser vacía.

```python
def test_no_hay_fuga_por_duplicados():
    train = pd.read_parquet(TRAIN_PATH)["id"]
    test = pd.read_parquet(TEST_PATH)["id"]
    assert len(set(train) & set(test)) == 0, "ids compartidos entre splits"
```

Cubre también las transformaciones ajustadas sobre todo el dataset: un test
que verifique que cada transform encaja dentro de un `Pipeline`/`ColumnTransformer`
(no se llama `fit_transform` fuera del fold) atrapa la fuga silenciosa #1
(ver `ml/validacion.md` → "Taxonomía de leakage").

### Calidad de etiquetas

- **Muestreo y revisión**: una muestra aleatoria (y otra estratificada por
  clase) de etiquetas revisada a mano cada N, con tasa de desacuerdo
  registrada. El test es la tasa misma: si el ruido de etiqueta supera un
  umbral acordado, el pipeline lo reporta.
- **Detección de etiquetas ruidosas**: votación cruzada de modelos (si tres
  modelos coinciden y la etiqueta difiere, es sospechosa) o *label noise
  detection*; los candidatos van a revisión, no a entrenamiento directo.

### Slices

El modelo puede fallar en un grupo aunque la métrica global esté bien. La
pregunta es "¿el modelo falla en un subgrupo?"; se responde con slicing de la
métrica por los grupos relevantes (segmento, región, percentil de una
feature):

```python
@pytest.mark.parametrize("slice_col,slice_val", [
    ("region", "N"), ("region", "S"),
    ("segmento", "pyme"), ("segmento", "corp"),
])
def test_metrica_por_slice(slice_col, slice_val, modelo, X_test, y_test):
    mask = X_test[slice_col] == slice_val
    assert len(X_test[mask]) >= MIN_N, "slice sin suficiente muestra"
    m = evaluar(modelo, X_test[mask], y_test[mask])
    assert m.recall >= UMBRAL_SLICE, f"recall bajo en {slice_col}={slice_val}"
```

## Tests de modelo

### Invariantes de salida

Propiedades que deben cumplirse siempre, para cualquier entrada válida:
rango, monotonicidad, probabilidades en [0,1], sumas que deben sumar 1.
No dependen de la calidad del modelo, solo de su corrección estructural.

```python
def test_salida_es_probabilidad():
    preds = modelo.predict_proba(X_valida)
    assert (preds >= 0).all() and (preds <= 1).all()
    assert np.allclose(preds.sum(axis=1), 1.0)

def test_monotonia_donde_se_exige():
    # para un modelo monótono por diseño, subir la feature no baja la score
    X_up = X_valida.copy(); X_up["ingreso"] *= 1.1
    assert (modelo.predict_proba(X_up)[:, 1] >= modelo.predict_proba(X_valida)[:, 1]).mean() > 0.99
```

### Ejemplos golden

Casos conocidos cuyo resultado no debe romperse: un cliente que siempre fue
default, un texto que siempre se clasificó como spam. Son la red de
seguridad frente a regresiones de refactor; el conjunto se mantiene pequeño,
representativo y curado a mano.

```python
GOLDEN = [
    ({"ingreso": 8_000, "deuda": 120_000}, "default"),
    ({"ingreso": 90_000, "deuda": 4_000}, "paga"),
]

@pytest.mark.parametrize("features,expected", GOLDEN)
def test_golden(features, expected, modelo):
    assert modelo.predict(pd.DataFrame([features]))[0] == expected
```

### Funciones puras del pipeline

Las transformaciones que son funciones puras (input fijo → output esperado)
se testean como funciones puras: entrada conocida, salida esperada, sin
modelo de por medio. Atrapan el grueso de los bugs de featurization.

```python
def test_imputer_deja_indicador():
    df = pd.DataFrame({"monto": [10.0, None, 30.0]})
    out = imputar_monto(df)
    assert out["monto_missing"].tolist() == [0, 1, 0]
    assert out["monto"].isna().sum() == 0
```

Regla: separar la lógica de transformación del I/O (leer/escribir) para que
sea testable sin archivos; el I/O se testea aparte, con fixtures.

### Property-based (Hypothesis)

Para invariantes que se mantienen en el espacio completo de entradas, no en
un puñado de ejemplos: Hypothesis genera entradas y busca el contraejemplo
que rompa la propiedad.

```python
from hypothesis import given, strategies as st
import numpy as np

@given(st.lists(st.floats(min_value=0, max_value=1e6), min_size=1))
def test_winsorizar_conserva_el_rango(valores):
    out = winsorizar(np.array(valores), lower=0.05, upper=0.95)
    assert (out >= 0).all() and (out <= 1e6).all()
    assert np.isfinite(out).all()
```

### Calibración

La calibración es una invariante probabilística: si el modelo dice p=0.8,
el 80% de los casos con esa score deben ser positivos. Se testea con el
grupo por bins y se acepta con tolerancia sobre la pendiente (calibration
curve, slope 1 ± δ), ver `ml/exprime-el-modelo.md`.

### Rendimiento por slice y paridad entre grupos

- **Rendimiento por slice**: la métrica de negocio desglosada por los
  subgrupos relevantes con umbral propio por slice (sección de datos
  arriba).
- **Paridad entre grupos**: métricas de fairness por atributo protegido —
  paridad demográfica, equalized odds, igualdad de oportunidad, calibración.
  La elección de la métrica es una decisión registrada (tensiones formales
  entre ellas), no un accidente. Detalle y mitigaciones en
  `ml/fairness-y-seguridad.md`.

```python
def test_paridad_tpr_por_grupo():
    tpr_a = tpr(modelo, X[X.grupo == "a"], y[X.grupo == "a"])
    tpr_b = tpr(modelo, X[X.grupo == "b"], y[X.grupo == "b"])
    assert abs(tpr_a - tpr_b) < 0.05, "TPR desigual entre grupos"
```

## Evaluación honesta

La evaluación no es un test de CI: es una medición que hay que blindar
contra el propio equipo. Reglas (ver `ml/metricas-y-evaluacion.md` →
"Disciplina del conjunto de test" y `ml/validacion.md`):

- **Un test set fijo y blindado.** Se aparta al principio, se usa una sola
  vez al final y nadie lo consulta durante el desarrollo. Cada uso para
  decidir (modelo, umbral, features) lo contamina.
- **No reutilizarlo para selección.** El test confirma, no selecciona; la
  selección se hace sobre validación. Reusar el test hasta que "pase" lo
  convierte en un conjunto de selección y su estimación deja de valer.
- **Seed fija y media sobre seeds.** Un split es una muestra: otro split da
  otro número. Reportar media ± desviación sobre R semillas
  (`RepeatedKFold`), nunca un solo run.
- **Reportar intervalos.** La métrica sin varianza no es reproducible: métrica
  ± intervalo (desviación sobre semillas o CI bootstrap), con el método y el
  número de repeticiones escritos junto al número.

## Infraestructura

### Test de determinismo del pipeline

Mismo input → mismo output. Si un run del pipeline produce algo distinto sin
que nada cambiara, es un bug, no una variación. El test ejecuta el pipeline
dos veces sobre el mismo dato y compara hashes de las salidas.

```python
def test_pipeline_determinista(tmp_path):
    d1 = ejecutar_pipeline(tmp_path / "out1")
    d2 = ejecutar_pipeline(tmp_path / "out2")
    for nombre in d1.keys():
        assert d1[nombre] == d2[nombre], f"{nombre} varió entre runs"
```

Fuentes de no-determinismo a fijar: semillas (`random`, `numpy`, `torch`),
orden de operaciones sobre diccionarios/sets, hilos en transformaciones
paralelas, y cualquier estadística calculada sobre un sample.

### Versionado de datos y modelo en cada run

Cada run loguea las cuatro coordenadas: entorno (versiones de paquetes),
código (commit), datos (hash/manifest, ver `data/calidad-datos.md`) y
parámetros. El test de infraestructura verifica que el artefacto de modelo
lleva registrado qué versión de datos lo produjo; sin eso, "reentrenar" no es
reproducible. En este proyecto lo cubren el manifest de datos y el registry
(MLflow si `use_mlflow`).

### El gate en CI

Los tests de las capas corren en cada commit; los de evaluación (test set
fijo) corren fuera del loop de desarrollo, en un step separado que no
produce "el número" en cada PR. La puerta `./init.sh` ejecuta la suite y
bloquea si está en rojo: nada entra sin tests verdes, y la deuda no se cuela
"de pasada".

## Mutación aplicada a pipelines

Los tests de datos y de funciones puras comparten el problema de la cobertura
por líneas: pueden pasar sin detectar lógica rota. La mutación verifica que
"muerden": se muta una transformación (invertir una comparación, cambiar un
`>` por `<`, True por False) y se comprueba que algún test lo pilla. Un
mutante que sobrevive es código que los tests no protegen.

{% if use_sdd %}
Este proyecto incluye el extra `use_sdd` con mutación nativa: `tools/mutate.py`
muta operadores de comparación/booleanos y ejecuta la suite por mutante
(resumen killed/survived/score), y `agents mutation` expone
`run_mutation_testing` y `crap_report` (CRAP = cc²·(1−cov/100)³+cc, umbral
30). El reviewer puede pedirlo antes del `finish` de una feature:

```bash
uv run python -m agents --json run mutation run_mutation_testing \
    --target {{ project_slug }}/features/build_features.py
uv run python -m agents --json run mutation crap_report \
    --target {{ project_slug }}/features/build_features.py
```

Ejemplo manual: mutar `monto > umbral` a `monto < umbral` en la
transformación y comprobar que el test de contrato de rangos (o un test de la
transformación) falla. Si no falla ninguno, falta un test.
{% else %}
Si `use_sdd` no está activo, el equivalente mínimo es manual: duplicar una
transformación, invertir a propósito una condición (`>` por `<`, True por
False) y verificar que al menos un test falla. Un mutante que sobrevive es
una transformación que los tests no protegen.
{% endif %}

## Práctica

### Estructura de una suite ML mínima

Los tests se organizan por módulo, como el código: uno por capa.

```
tests/
├── test_data_contract.py     # esquema, invariantes, sin fugas
├── test_data_distribucion.py # estabilidad de distribuciones, calidad de etiquetas
├── test_features.py          # funciones puras del pipeline de features
├── test_modelo_invariantes.py# rango, [0,1], monotonicidad, calibración
├── test_modelo_golden.py     # casos golden, slices, paridad
└── test_pipeline_determinismo.py  # mismo input → mismo output
```

Los tests de evaluación honesta (test set fijo, media sobre seeds) no son
unit tests: viven en `eval/` (o el equivalente del proyecto) y corren en un
step separado del CI.

### La regla: "cuando un bug se repite, se convierte en test"

Un bug de datos o de pipeline que reaparece significa que la suite no lo
cubría. Se escribe el test que lo reproduce y se amplía el contrato; la
segunda vez no debe ocurrir. La mecánica es la de TDD aplicada al dato: cada
regresión se convierte en parte permanente de la suite.

```python
def test_join_no_explota_duplicados():
    # bug visto 2 veces: el join de items duplicaba pedidos
    joined = pd.merge(pedidos, items, on="pedido_id")
    assert joined["pedido_id"].is_unique
```

### Qué no merece test

- **Goldens redundantes** que duplican cobertura sin añadir información (100
  casos para un problema donde 5 ya acotan el comportamiento).
- **Detalles de implementación del framework**: que sklearn o el modelo
  hagan lo que el framework garantiza no se testea; se testea tu interfaz con
  él.
- **El comportamiento de bibliotecas ajenas**: si un test solo verifica que
  `pandas.read_csv` leyó un CSV, está probando pandas.
- **Cobertura por sí misma**: una línea cubierta no es una línea protegida
  (ver mutación arriba); la cobertura sin criterio de calidad es un número
  decorativo.

## Fuentes

- Breck, E., Cai, S., Nielsen, E., Salib, M., Sculley, D., *The ML Test
  Score: A Rubric for ML Production Readiness and Technical Debt
  Reduction*. arXiv:1706.08568. https://arxiv.org/abs/1706.08568
- Sato, D., Wider, A., Windheuser, C., *Continuous Integration for Machine
  Learning* (ease.ml/ci). arXiv:1903.00278. https://arxiv.org/abs/1903.00278
- Zhang, J. M., Harman, M., Ma, L., Liu, Y., *Machine Learning Testing:
  Survey, Landscapes and Horizons*. arXiv:1808.04730.
  https://arxiv.org/abs/1808.04730
- Sculley, D., et al., *Hidden Technical Debt in Machine Learning Systems*
  (NIPS 2015). arXiv:1512.04256. https://arxiv.org/abs/1512.04256
