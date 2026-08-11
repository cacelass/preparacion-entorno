# Ingeniería de datos

## El problema: manipular datos correcto y rápido

Manipular datos en DS es un acto de equilibrio entre dos objetivos que suelen
tirar en direcciones opuestas: **correctez** (que el resultado sea
verdadero: sin filas duplicadas por un join, sin NULLs silenciosos, sin
unidades mezcladas) y **velocidad** (que la respuesta llegue a tiempo para
iterar). El fallo más caro no es la lentitud, es la correctez falsa: un
resultado que parece bien y no lo está cuesta días de debugging aguas abajo;
un resultado lento se arregla con mejor hardware o mejor plan de ejecución.

La herramienta correcta se elige por el **tamaño del dato, la forma de la
operación y el costo de fallar**:

| Motor | Cuándo | Por qué |
|-------|--------|---------|
| SQL (base) | Datos en servidor; joins, agrega, filtra | Se ejecuta junto a los datos; cero transferencia |
| pandas | Cabe en RAM; transformación imperativa por columna | Expresividad Python, ecosistema ML |
| polars/dask | No cabe en RAM; operaciones vectorizables masivas | Lazy/out-of-core, paralelismo multihilo |
| DuckDB | SQL sobre archivos o DataFrames sin mover datos | OLAP in-process, planeación de query real |
| Spark | Decenas de TB; cluster ya existente; shuffle necesario | Distribuido; casi nunca es la respuesta |

La jerarquía de decisión, en orden: **primero muévete hacia donde viven los
datos** (SQL en la base), **luego lee menos** (proyección, filtrado,
parquet en vez de CSV), **después vectoriza** (pandas/polars sobre columnas,
no bucles), y **solo al final escala horizontalmente** (Spark). La mayoría de
los pipelines de DS mueren por las dos primeras reglas ignoradas, no por
falta de cluster.

## SQL que todo DS debería dominar

SQL es el lenguaje con más apalancamiento del oficio: los datos productivos
viven en bases, y cualquier join que se haga en pandas sobre datos descargados
es un join que pudo hacerse en el servidor sin transferir nada. Dominar SQL no
es saber su sintaxis: es saber **qué va a hacer cada join con los datos
reales**.

### Joins: qué devuelve cada tipo y cuándo duplican filas

Un join empareja cada fila de A con cada fila de B que cumple la condición de
emparejamiento. Por eso **la cardinalidad del resultado depende de la
unicidad de las claves**, no del tipo de join:

| Join | Filas resultantes | Caso de uso |
|------|-------------------|-------------|
| INNER | A ∩ B (coincidencias) | Enriquecer con la fuente canónica |
| LEFT | Todo A + coincidencias de B | A es la tabla maestra, B es opcional |
| RIGHT | Todo B + coincidencias de A | Simétrico del anterior, poco usado |
| FULL OUTER | Todo A ∪ B | Conciliar dos fuentes parciales |
| ANTI (NOT IN / NOT EXISTS) | A sin coincidencia en B | Filas que fallaron el join / deben eliminarse |

```sql
-- LEFT JOIN: si una clave de A aparece 2 veces en B, la fila de A se duplica.
-- Siempre: SELECT COUNT(*) ANTES y DESPUÉS del join, y compara.
SELECT a.id, a.monto, b.categoria
FROM pedidos a
LEFT JOIN productos b ON b.producto_id = a.producto_id;
```

**Regla de oro**: antes de escribir un join, responde *"¿la clave es única
en ambas tablas?"* Si no lo es, el resultado multiplica filas y cada métrica
agregada sobre él queda sesgada. Verificación barata:

```sql
SELECT producto_id, COUNT(*) n
FROM productos GROUP BY producto_id HAVING COUNT(*) > 1;
```

El **anti-join** (`NOT EXISTS`) es la forma correcta de "las filas que no
emparejan"; `NOT IN` falla en silencio si B contiene NULLs (ver trampas).

### GROUP BY y agregaciones

`GROUP BY` reduce filas a una por valor de clave y aplica agregaciones. Lo que
los novatos ignoran: **todo lo que no esté en el GROUP BY debe ser un
agregado**; una columna "salvaje" rompe en la mayoría de motores y en los que
no (SQLite, MySQL por defecto) devuelve un valor arbitrario sin avisar.

```sql
SELECT cliente_id,
       COUNT(*)                        AS n_pedidos,
       SUM(monto)                      AS total,
       AVG(monto)                      AS ticket_medio,
       MIN(fecha)                      AS primer_pedido,
       COUNT(DISTINCT categoria)       AS n_categorias
FROM pedidos
WHERE monto > 0
GROUP BY cliente_id
ORDER BY total DESC;
```

Dos errores de agregación típicos:

- **Agregar después del join equivocado**: si el join duplicó filas, `SUM`
  cuenta dos veces el mismo monto. Agrega *antes* de unir cuando la agregación
  es por tabla fuente.
- **`COUNT(col)` vs `COUNT(*)`**: `COUNT(col)` excluye NULLs de la columna;
  `COUNT(*)` cuenta filas. `COUNT(col) < COUNT(*)` delata NULLs que quizá no
  deberías estar contando.

### Ventanas: ROW_NUMBER, LAG, particiones

Las ventanas agregan **sin reducir filas**: cada fila conserva su identidad y
gana una columna calculada sobre su partición. Se resuelven después del WHERE
y antes del ORDER BY final, así que no puedes filtrar su resultado en el WHERE
(necesitas una subconsulta/CTE).

```sql
-- Top 3 ventas por región, con la venta anterior por cliente.
WITH ventas_rnk AS (
    SELECT cliente_id, region, monto, fecha,
           ROW_NUMBER() OVER (PARTITION BY region ORDER BY monto DESC) AS rnk,
           LAG(monto)  OVER (PARTITION BY cliente_id ORDER BY fecha)   AS venta_anterior
    FROM ventas
)
SELECT * FROM ventas_rnk WHERE rnk <= 3;
```

| Función | Qué hace |
|---------|----------|
| `ROW_NUMBER()` | Ranking estricto 1..n; único dentro de la partición |
| `RANK()` / `DENSE_RANK()` | Empates comparten rango; `RANK` salta, `DENSE_RANK` no |
| `LAG(x)` / `LEAD(x)` | Valor de la fila anterior/siguiente en el orden — para deltas y rolling manual |
| `SUM(x) OVER (PARTITION BY ... ORDER BY ...)` | Suma acumulada por partición |
| `AVG(x) OVER (ORDER BY ... ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` | Media móvil de 7 filas |

`ROW_NUMBER` es la herramienta estándar para **dedupe en SQL**: numerar las
filas duplicadas y quedarte con la número 1 de cada clave.

### CTEs vs subconsultas

Las CTEs (`WITH`) son subconsultas con nombre. Preferirlas siempre:

- **Legibilidad**: dan nombre y encabezan el pipeline de arriba a abajo, en el
  orden en que se lee.
- **Reutilización**: la misma CTE puede referenciarse varias veces sin
  duplicar código.
- **Descomposición**: cada paso (limpiar → dedupe → agregar → join) es una CTE,
  verificable por separado.

```sql
WITH base AS (SELECT ... FROM raw WHERE ...),
     dedup AS (SELECT ... ROW_NUMBER() OVER (PARTITION BY k ORDER BY t DESC) = 1 ...)
SELECT ...
FROM base JOIN dedup USING (k);
```

El costo es el mismo (el optimizador suele inline las CTEs). La diferencia es
de mantenimiento: un pipeline SQL de 6 CTEs anidadas se revisa; un bloque de 6
subconsultas anidadas no.

### Trampas clásicas

- **Join en claves no únicas**: la causa número 1 de resultados incorrectos en
  pipelines de DS. Verifica unicidad antes; agrega antes de unir.
- **NULLs**:
  - `NULL = NULL` es NULL (falso), no verdadero: compara con `IS NULL`.
  - `NOT IN` devuelve *cero filas* si la subconsulta contiene cualquier NULL.
    Usar `NOT EXISTS`.
  - En agregaciones, NULLs se ignoran (excepto `COUNT(*)`). `SUM` de todo NULLs
    da NULL, no 0; usa `COALESCE` explícito si el 0 es semántico.
  - En el JOIN, claves NULL nunca emparejan, ni consigo mismas.
- **WHERE vs HAVING**: `WHERE` filtra filas *antes* de agregar; `HAVING`
  filtra grupos *después*. Un filtro sobre una agregación (`total > 1000`) no
  puede ir en `WHERE`. Filtrar en `WHERE` lo que podrías filtrar ahí reduce
  trabajo y correctamente excluye filas de las agregaciones.
- **Comparaciones implícitas de tipos**: fechas como strings, enteros como
  texto. El motor coacciona o falla; valida el esquema de la tabla antes de
  confiar en el resultado.
- **Flotantes como claves o en igualdades**: `0.1 + 0.2 <> 0.3`. Nunca juntes
  ni agrupes por flotantes.

## pandas idiomático

pandas es el entorno de experimentación de DS, no una base de datos. Su
rendimiento y su correctez dependen de respetar su modelo de cómputo:
**operaciones vectorizadas sobre columnas**, con índices alineados.

### Vectorización frente a bucles

Un `for` sobre filas de un DataFrame es casi siempre la señal de que el
código está luchando contra el modelo de datos. pandas está diseñado para
aplicar operaciones a columnas enteras; un bucle serializa el trabajo que las
operaciones vectorizadas hacen en un solo paso de C/Fortran.

| Patrón lento | Patrón idiomático |
|--------------|-------------------|
| `df.iterrows()` / bucle por fila | Operación sobre columna: `df["a"] + df["b"]` |
| `df.apply(f, axis=1)` sobre muchas filas | Vectorizar `f`: `np.where`, `np.select`, `pd.cut`, booleans |
| `apply` con `lambda` sobre elemento | Método nativo: `str.lower()`, `dt.year`, `.map()` con dict |
| `loc` en bucle para escribir | Asignación vectorizada: `df.loc[mascara, "col"] = valor` |

Regla práctica: **si el `apply` no cabe en una expresión, es un bucle
disfrazado**. La vectorización no es una micro-optimización: un `apply`
cruzando filas es O(n) en Python; la misma lógica vectorizada corre a
velocidad nativa. Cuando el `apply` es inevitable (función arbitraria por
fila), el siguiente paso de escala es mover el cálculo a SQL o a un
motor columna, no embeberlo más.

Alternativas exactas y más rápidas que `apply`:

- `df["tipo"] = np.select([m1, m2, m3], ["a", "b", "c"], default="d")`
- `df["bins"] = pd.cut(df["x"], bins=[0, 10, 100], labels=["bajo", "alto"])`
- `df["anio"] = df["fecha"].dt.year`
- `df["cat"] = df["id"].map(dict_id_a_categoria)` — búsqueda con dict

### groupby-transform vs merge

`groupby` admite dos modos de salida muy distintos:

- `groupby.agg(...)`: reduce el grupo a una fila (agregado).
- `groupby.transform(...)`: devuelve la misma forma que la entrada, cada fila
  con el agregado de *su* grupo. Sirve para normalizar, marcar, o calcular
  desviaciones respecto al grupo **sin join**.

```python
# z-score por categoría, sin unir nada:
df["z"] = (df["monto"] - df.groupby("categoria")["monto"].transform("mean")) \
          / df.groupby("categoria")["monto"].transform("std")
```

`transform` es más legible y menos propenso a explosión cartesiana que el
patrón "agrego, renombro, merge, colapso". La alternativa con merge es
equivalente cuando necesitas el agregado como *feature* del DataFrame grande:

```python
agg = df.groupby("cliente_id", as_index=False)["monto"].agg(total="sum")
df = df.merge(agg, on="cliente_id", how="left")
```

### Merge con claves duplicadas: explosión cartesiana

El error más caro de pandas es un `merge` donde la clave no es única en al
menos un lado. pandas no lo avisa: duplica filas en silencio y toda métrica
posterior queda inflada. Antes de mergear:

```python
assert df_a.duplicated("cliente_id").sum() == 0
assert df_b.duplicated("cliente_id").sum() == 0
```

o deduplica explícitamente lo que sepas que puede repetirse (uno a muchos):

```python
df_b = df_b.sort_values("fecha").drop_duplicates("cliente_id", keep="last")
df = df_a.merge(df_b, on="cliente_id", how="left")
```

Verificación post-merge obligatoria: `df.shape[0] == df_a.shape[0]` en todo
join que se pretenda uno-a-uno.

### dtypes: category y datetime

El dtype es el contrato del dato dentro del DataFrame. Dos que se ignoran
demasiado:

- **`category`**: reduce memoria y acelera `groupby`/`sort` en columnas
  categóricas. Conviértelo cuando la cardinalidad sea baja y el orden
  importe (`pd.Categorical` con `categories` explícito evita ordenaciones
  alfabéticas ilegibles).
- **`datetime64`**: las fechas como string no permiten `dt.year`, ventanas ni
  joins temporales correctos. Convertir una vez al leer:
  `df["fecha"] = pd.to_datetime(df["fecha"])`. Coerce con
  `errors="coerce"` y **verifica cuántos NULLs creaste** — eso detecta
  formatos rotos en la fuente.

```python
df["region"] = df["region"].astype("category")
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", format="%Y-%m-%d")
assert df["fecha"].isna().sum() == 0  # o un número documentado
```

### Views, copias y SettingWithCopyWarning

`SettingWithCopyWarning` aparece cuando pandas no puede probar si estás
escribiendo sobre una view o sobre el original. No es un error: es la señal
de que tu intención no está clara. La regla que elimina la clase entera de
bugs: **cada pipeline empieza con una copia explícita y las transformaciones
encadenadas no mutan input**.

```python
df = df.copy()
df = (df
      .assign(monto_neto=lambda d: d["monto"] - d["descuento"],
              total_ok=lambda d: d["monto"].fillna(0) > 0)
      .query("region != 'XX'"))
```

`assign` + `query` devuelven siempre un DataFrame nuevo: sin views, sin
advertencias, sin efectos laterales sobre el input. El `apply` que muta
in-place dentro de un bucle es la otra cara del mismo anti-patrón.

### Memoria: float32 y downcasting

El dataset no "no cabe en memoria" porque sea grande: cabe o no cabe según el
dtype. Un `float64` ocupa el doble que un `float32` con la misma precisión
relativa para la mayoría de modelos. Disciplina de memoria:

- `float32` para features continuas del modelo (la pérdida de precisión es
  irrelevante frente al ruido del dato).
- `float32` o `int32` tras leer CSV (pandas lee `int64`/`float64` por defecto).
- `category` para strings repetidos.
- Convertir solo lo que el pipeline toque; medir con
  `df.memory_usage(deep=True)` antes y después.

```python
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
for c in df.select_dtypes("int64").columns:
    df[c] = pd.to_numeric(df[c], downcast="integer")
```

El orden correcto de ataque a un problema de memoria es: **downcast →
category → filtrar/proyectar al leer → motor columna/out-of-core**. Nunca al
revés.

{% if use_duckdb %}
## DuckDB: SQL OLAP sobre tus archivos

Este proyecto incluye DuckDB (`data/`) como capa de SQL in-process. DuckDB es
un motor **OLAP embebido**: corre dentro de tu proceso, habla SQL estándar y
lee Parquet/CSV/JSON **sin copiar los datos a memoria de pandas** — y también
SQL directo sobre DataFrames de pandas sin convertir:

```python
import duckdb

# SQL sobre un DataFrame sin copiarlo:
res = duckdb.sql("""
    SELECT region, COUNT(*) n, SUM(monto) total
    FROM df
    WHERE monto IS NOT NULL
    GROUP BY region
""").df()
```

Su planeador de query (vectorizado, columnar, con estadísticas de Parquet)
hace que operaciones que en pandas son O(n) en Python —joins grandes, groupby
sobre muchas columnas, multi-filtros— corran con la ejecución de un motor
real. Ventajas frente a pandas para DS:

- **Joins y agregaciones grandes**: empuja el trabajo al motor; puedes unir
  tablas de cientos de millones de filas sin que estén en RAM.
- **SQL sobre Parquet/CSV sin `read_csv`**: `SELECT * FROM 'data/raw/*.parquet'`
  proyecta y filtra leyendo solo lo necesario.
- **Perfilado barato antes de cargar**: conteos, cardinalidades y nulls por
  query sobre archivos enormes, antes de decidir qué entra al DataFrame.
- **`INSERT INTO ... SELECT`** para filtros pesados sobre Parquet con
  reescritura de un archivo grande.

Cuándo NO usar DuckDB:

- **Operaciones imperativas por fila**: llamar Python por fila
  (`map`, `apply`, función arbitraria) mata la vectorización del motor.
- **Cómputos que necesitan el ecosistema pandas/sklearn**: si la salida es un
  DataFrame para ML, la query devuelve y ya; no intentes hacer feature
  engineering completo en SQL si vives en Python.
- **Concurrencia de escritura**: DuckDB es in-process; no es una base de
  datos de servicio para múltiples consumidores escribiendo a la vez.

El patrón ganador: **SQL para la pesadez (joins, agrega, filtra, dedupe) y
pandas para lo fino (features, modelado)**. Cada uno en su terreno.

```python
from {{ project_slug }}.data.make_dataset import load_data_duckdb

df = load_data_duckdb(
    "dataset.csv",
    query="SELECT region, monto FROM datos WHERE monto IS NOT NULL",
)
```
{% endif %}

## ETL vs ELT e idempotencia

**ETL** (Extract-Transform-Load) transforma fuera del destino: los datos se
limpian en el pipeline y se cargan ya listos. **ELT** (Extract-Load-Transform)
carga en el motor de datos (dbt, DuckDB, warehouse) y transforma ahí con SQL.
Para un proyecto de DS:

- ETL cuando la transformación es pesada, necesita pandas/Python o los datos
  de entrada no sobreviven en un motor (APIs, JSON anidado).
- ELT cuando el dato cabe en un motor SQL: la transformación queda declarada,
  auditable y re-ejecutable, y el pipeline reduce a "cargar y dejar que el
  SQL haga el resto". Menos código Python que mantener.

La propiedad que un pipeline de datos no puede negociar es la **idempotencia**:
mismo input → mismo output, sin importar cuántas veces corra. Un pipeline
idempotente se puede re-ejecutar tras un fallo, en CI, o después de un fix,
sin duplicar filas ni corromper agregaciones.

Cómo se garantiza:

- **Recrear en vez de incrementar**: escribir `data/processed/` de cero en
  cada corrida (contenido derivado, reproducible desde raw) en lugar de
  "añadir filas nuevas" sobre el archivo anterior.
- **Raw inmutable**: `data/raw/` se escribe una vez al descargar; nunca se
  edita in-place. El hash SHA-256 del manifest detecta cambios silenciosos
  (ver [calidad-datos.md](calidad-datos.md)).
- **Determinismo**: sin ordenes por defecto de sets/dicts, sin fechas de
  ejecución como datos, semillas fijas. La salida debe ser función del input,
  no del momento de la corrida.
- **Reintentos limpios**: si un paso falla a mitad, su efecto debe poder
  descartarse (archivos temporales con sufijo, commit atómico de escritura:
  escribir a `*.tmp` y renombrar).

```python
# data/make_dataset.py — idempotente: el output se deriva completo cada vez.
def build_processed(raw_dir, processed_dir):
    df = pd.read_parquet(raw_dir / "dataset.parquet")
    out = transform(df)                    # determinista: mismo input → mismo output
    tmp = processed_dir / "dataset.tmp.parquet"
    out.to_parquet(tmp)
    tmp.replace(processed_dir / "dataset.parquet")
```

### Schema evolution

Los datos cambian mientras el pipeline duerme: columnas nuevas, renombradas,
tipos que mutan, categorías que aparecen. El schema evolution es la disciplina
de **que ese cambio rompa ruidosamente en la frontera y no silenciosamente en
el modelo**:

- El contrato de esquema (pandera/Great Expectations) es la puerta: una
  columna nueva que no estaba no pasa; un renombrado se detecta como columna
  faltante en el primer test.
- Renombrar columnas es un cambio de contrato, no una decisión cosmética:
  actualiza contrato + pipeline + tests juntos, en el mismo cambio.
- Ante columnas nuevas, decidir explícitamente: ¿la ignoras (contrato la
  bloquea), la usas (añades al contrato), o toleras extra con regla
  `strict=False`? La decisión por defecto debe ser **fallar**, no tolerar.
- Ver `calidad-datos.md` para la mecánica de contratos y versionado.

### Leer menos datos como disciplina

La forma más barata de computar es **no leer**. Cada byte que atraviesa el
pipeline cuesta I/O, memoria y tiempo; el hábito de "cargar el CSV entero y
filtrar después" es el anti-patrón dominante en notebooks.

- **Proyección**: carga solo las columnas que necesitas (`pd.read_csv(..., usecols=...)`).
- **Filtrado temprano**: filtrar al leer o en la query, antes de cualquier
  transformación.
- **Formato columna**: Parquet almacena por columnas y permite _predicate
  pushdown_: el motor lee solo las columnas y los row-groups que la query
  toca. CSV se lee entero siempre.
- **Sampling honesto**: para EDA, leer una fracción (`nrows=`, `sample`)
  antes que el archivo completo; escalar después con datos reales.

```python
df = pd.read_parquet(
    "data/raw/dataset.parquet",
    columns=["order_id", "monto", "region", "fecha"],  # proyección
    filters=[("region", "in", ["N", "S"])],            # pushdown en Parquet
)
```

## ACID y transacciones

Cuando una escritura toca más de un recurso —varias filas, una tabla y un
índice, un archivo y su manifest— "escribir y rezar" no basta. Una
**transacción** agrupa esas operaciones y garantiza cuatro propiedades,
conocidas como ACID (Haerder & Reuter, 1983):

- **Atomicidad (A)**: todo o nada. Si una operación del grupo falla, ninguna
  deja efecto. Se implementa con un *write-ahead log* (WAL): se registra el
  cambio antes de aplicarlo y, ante un crash o error, se hace *rollback* a
  partir del log. Distíngela de la consistencia: la atomicidad es sobre
  *fallos*, no sobre *datos*.
- **Consistencia (C)**: la transacción lleva la base de un estado válido a
  otro válido — donde "válido" significa los invariantes **declarados**
  (PRIMARY KEY, FOREIGN KEY, CHECK, NOT NULL). La base no sabe nada de tu
  lógica de negocio: si el invariante no está declarado, la "consistencia"
  que esperas no existe. C de ACID no tiene relación con la *eventual
  consistency* de los sistemas distribuidos — homónimos, no parientes.
- **Aislamiento (I)**: las transacciones concurrentes no se ven entre sí. El
  grado se controla con el *isolation level*:

  | Nivel | Evita | Anomalía que queda |
  |-------|-------|--------------------|
  | Read uncommitted | nada | dirty reads (leer datos no commiteados) |
  | Read committed | dirty reads | non-repeatable reads (la misma fila cambia en la transacción) |
  | Repeatable read | + non-repeatable | phantom reads (filas nuevas aparecen en medio) |
  | Serializable | todo | — (equivale a ejecutar las transacciones en serie) |

  El aislamiento se paga: con *locks* en rendimiento (y *deadlocks* cuando dos
  transacciones se bloquean mutuamente), con *MVCC* (multiversion concurrency
  control) en memoria y complejidad. Nivel por defecto de los motores
  modernos: Read committed (Postgres, SQLite) o Repeatable read (MySQL).

- **Durabilidad (D)**: lo commiteado sobrevive a un crash. Requiere escribir
  el WAL a disco (fsync) antes de confirmar; la opción de "durabilidad
  relajada" de algunos motores (`synchronous = OFF` en SQLite, group commit)
  sacrifica la D a cambio de velocidad — válida solo donde la pérdida de los
  últimos segundos no importa.

### ACID en un proyecto de DS

- **DuckDB y SQLite dan ACID real sobre un solo archivo** (MVCC + WAL). Pero
  son in-process: no hay concurrencia de escritura entre procesos (ver
  "Cuándo NO usar DuckDB" más arriba). ACID de un solo nodo, no de servicio.
- El patrón **`tmp` + rename** (idempotencia, arriba) es atomicidad de **un**
  archivo: o existe el nuevo o existe el viejo. No cubre dos escrituras que
  deben commiteverse juntas. Cuando el pipeline necesita eso, usa una
  transacción real (SQLite/DuckDB en modo transaccional) en vez de emularla.
- **Parquet/CSV no son transaccionales**: una escritura a mitad muere con el
  archivo a medias. La mitigación es siempre escribir a un archivo temporal y
  renombrar — y asumir que "media escritura visible" es imposible solo si el
  rename es atómico (lo es dentro de un mismo filesystem).
- No confundas **idempotencia** con transaccionalidad: idempotente = se puede
  re-ejecutar sin duplicar; transaccional = las operaciones de un paso se
  commitean o se revierten juntas. Un pipeline necesita ambas, y son
  ortogonales.

### Cómo se rompe

- **Asumir ACID donde no lo hay**: escrituras directas a Parquet, `to_csv`
  sobre un archivo en uso, `INSERT` a medio de un batch. La base no te
  salva; la transacción o el tmp+rename sí.
- **Isolation mal elegido**: demasiado bajo → lecturas inconsistentes en
  procesos concurrentes (dirty reads); demasiado alto → deadlocks y lentitud
  inesperados. El nivel por defecto no es "el correcto", es "el razonable".
- **Durabilidad sin fsync**: el commit "confirma" pero un corte de luz lo
  pierde. Durability relajada solo donde sea aceptable perder el último
  tramo.
- **ACID distribuido**: la transacción entre dos bases no escala como la de
  una; exige *two-phase commit* o consenso, y el coste y los modos de fallo
  crecen. Kleppmann lo resume: los sistemas distribuidos son el último
  recurso, no el primero.

## Escala: cuándo es cuándo

"El dataset no cabe en memoria" tiene tres respuestas en orden de coste
creciente. Elegir bien es decidir **dónde** vive el problema, no echarle más
hardware a un problema de mala lectura.

### Chunking y out-of-core (polars/dask)

Cuando el dato no cabe en RAM pero cabe en disco y la operación es
paralelizable por partición (agregaciones, filtros, joins key-able):

- **polars lazy**: query optimizada y ejecutada por el motor (streaming con
  `collect(streaming=True)`); el equilibrio moderno entre expresividad Python
  y rendimiento de motor.
- **dask**: pandas con ejecución perezosa; `dask.dataframe` replica la API de
  pandas sobre particiones y paraleliza en threads/workers. Para datos que
  caben holgadamente en disco y operaciones que encajan en partición.
- **Chunking manual** (`pd.read_csv(..., chunksize=)`) como último recurso:
  mantenible solo para agregaciones acumulables; un join o un sort global
  requieren reordenar por chunks y dejan de valer la pena.

La señal de que estás en el régimen correcto: la operación es un filtro,
agregación o join reproducible por partición, y el bottleneck es I/O, no CPU.

### Spark y distribuido: cuándo de verdad (y casi nunca)

Spark resuelve un problema preciso: **datos que no caben en la memoria ni el
disco de una máquina, o shuffle distribuido de decenas de TB**. Si tu dato
cabe en un NVMe local, Spark es un costo (cluster, serialización, tuning,
debugging) sin beneficio.

Banderas honestas de "esto no es para Spark":

- El dataset cabe en disco local (decenas de GB) → polars/dask/duckdb.
- La operación es un groupby o un join con clave repartible → un motor
  single-node con I/O eficiente gana casi siempre.
- El pipeline es un prototipo de DS → el tiempo de iteración de Spark mata el
  experimento.

Distribuido se justifica cuando el shuffle es **irreducible** (joins y
ordenaciones sobre escalas multi-TB), el cluster ya existe y se paga igual, o
el SLA de tiempo lo exige. Si empiezas el proyecto pensando en Spark, plantéate
si el problema es de formato y lectura, que lo es en la mayoría de los casos.

### Formato columna vs filas, y compresión

El formato decide el costo de cada lectura:

| | Parquet | CSV |
|---|---------|-----|
| Almacenamiento | Columnar, comprimido (snappy/zstd) | Filas, texto plano |
| Lectura parcial | Proyección + predicate pushdown | Siempre completo |
| Tipos | Preserva dtype | Coerción a string, a adivinar |
| Compresión | Por columnas y por row-group | Ninguna (o del archivo completo) |
| Uso | DataFrames, pipelines, datos de entrada | Intercambio, feeds externos, humans |

Regla: **parquet es el formato de trabajo interno; CSV solo en la frontera**
(lo que una fuente te da o lo que entregas). Para archivos de texto pesados,
comprimir (gzip/zstd) reduce el I/O; parquet ya comprime por diseño.

Los tres mandamientos de escala, en orden:

1. **Lee menos**: proyección, filtrado, parquet, sampling.
2. **Mueve el cómputo a un motor**: SQL (DuckDB/db) para joins/agrega.
3. **Escala la herramienta**: polars/dask primero; Spark solo si el problema
   escala más allá de una máquina de verdad.

## Práctica: cinco recetas y el anti-patrón

### 1. Agregación por grupo (SQL y pandas)

```sql
SELECT cliente_id, COUNT(*) n, SUM(monto) total, AVG(monto) ticket
FROM pedidos GROUP BY cliente_id;
```

```python
df.groupby("cliente_id", as_index=False)["monto"] \
  .agg(n="count", total="sum", ticket="mean")
```

### 2. Join dedupe: quedarse con la última fila por clave

```sql
WITH rnk AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY cliente_id ORDER BY fecha DESC) r
    FROM pedidos
)
SELECT * FROM rnk WHERE r = 1;
```

```python
df.sort_values("fecha").drop_duplicates("cliente_id", keep="last")
```

### 3. Rolling / ventana temporal

```sql
SELECT fecha, monto,
       SUM(monto) OVER (ORDER BY fecha ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS suma_7d
FROM ventas;
```

```python
df["suma_7d"] = df.sort_values("fecha")["monto"].rolling(7).sum()
```

### 4. Pivote largo → ancho

```sql
SELECT cliente_id,
       MAX(CASE WHEN canal = 'web'  THEN monto END) AS web,
       MAX(CASE WHEN canal = 'app'  THEN monto END) AS app
FROM pedidos GROUP BY cliente_id;
```

```python
df.pivot_table(index="cliente_id", columns="canal", values="monto", aggfunc="sum")
```

### 5. Tratamiento de nulos (política, no improvisación)

```sql
SELECT cliente_id, COALESCE(monto, 0) AS monto, monto IS NULL AS sin_monto
FROM pedidos;
```

```python
df["monto"].fillna(0, inplace=False)
df["monto"].isna().sum()          # verifica, siempre
```

### Anti-patrón: "cargar todo y filtrar"

```python
# MÁL: el archivo entero a RAM, filtrar después.
df = pd.read_csv("data/raw/ventas.csv")
df = df[df["region"] == "N"]
# BIEN: filtra y proyecta al leer.
df = pd.read_parquet("data/raw/ventas.parquet",
                     columns=["region", "monto", "fecha"],
                     filters=[("region", "in", ["N"])])
```

La regla de cierre de la práctica: **la pipeline se escribe para el dato que
crece, no para el dato de hoy**. Todo lo que hoy cabe en memoria funcionará
peor mañana con el doble de filas; leer menos, proyectar y empujar el cómputo
al motor es lo que hace que el pipeline de mañana no se reescriba.

## Fuentes

- "Designing Data-Intensive Applications" (Martin Kleppmann, O'Reilly 2017) —
  replicación, particionado, formatos de almacenamiento, transacciones y
  por qué los sistemas distribuidos son el último recurso.
- Haerder, T. y Reuter, A., *Principles of Transaction-Oriented Database
  Recovery*, ACM Computing Surveys 15(4), 1983 — el paper que acuñó el
  término ACID. https://doi.org/10.1145/289.291
- Documentación de pandas (guía de estilo, merges, dtypes, rendimiento) —
  https://pandas.pydata.org/docs/
- "Enhancing Performance" y "Essential Basic Functionality" de pandas
  (vectorización, views/copies) — https://pandas.pydata.org/docs/
- Documentación de DuckDB (SQL embebido, Parquet, integración con DataFrames) —
  https://duckdb.org/docs/
- Documentación de polars (lazy/streaming, query optimization) —
  https://pola.rs/docs/
- Documentación de dask (out-of-core y ejecución perezosa) —
  https://docs.dask.org/
- Documentación de Apache Parquet (formato columnar) —
  https://parquet.apache.org/docs/
- Documentación SQLite (joins, window functions, trampas de NULL) —
  https://www.sqlite.org/lang.html
