# Eficiencia computacional en proyectos de datos

Referencia para el agente lider cuando un pipeline tarda, consume demasiada
memoria o un entrenamiento no escala. Cada sección: principio, práctica
concreta y cómo falla en un proyecto real de DS.

## Complejidad algorítmica

**Principio.** La complejidad asintótica (Big-O) predice cómo crece el costo
con el tamaño de entrada; domina sobre cualquier micro-optimización en cuanto
los datos crecen.

**Práctica.** Conoce las complejidades típicas de DS:

| Operación | Coste típico |
|-----------|--------------|
| Búsqueda en lista | O(n) |
| Búsqueda en dict/set/hash | O(1) esperado |
| Ordenación | O(n log n) |
| Join de dos tablas | O(n·m) sin índice; O(n log n) con hash join |
| Dot product / matmul | O(n³) naive, ~O(n^2.37) con algoritmos modernos |
| Entrenamiento kNN (predicción) | O(n·d) por consulta sin índice |

**Análisis amortizado.** Una operación puntual puede ser cara (apéndice de
lista con realloc, `snp.unique` sobre array creciente) pero el promedio por
operación es barato. Optimizar el peor caso aislado cuando el promedio ya es
bueno es desperdicio.

**Regla de oro.** Primero correcto, luego rápido. Un algoritmo O(n log n) con
un bug es infinitamente más lento que un O(n²) correcto. Mide antes de tocar
el algoritmo: la constante del intérprete a veces gana a la asintótica para
n pequeños.

**Cómo falla.** Construir un DataFrame fila a fila con `concat` en un bucle es
O(n²) de copias; el caso típico es 10⁵ filas que tardan minutos en vez de
milisegundos. El orden de los joins importa: cruzar primero lo grande con lo
chico puede ser el mismo resultado con 100× menos memoria.

## Medir antes de optimizar

**Principio.** Sin perfilado no hay optimización, solo adivinanza con la
bendición de parecer eficiente.

**Práctica.** La escalera de herramientas, de arriba a abajo:

| Nivel | Herramienta | Respuesta |
|-------|-------------|-----------|
| Tiempo aislado | `timeit` / `%timeit` | Cuánto tarda una operación puntual |
| Perfil por función | `cProfile -s cumulative` | Qué funciones acumulan el tiempo |
| Perfil por línea | `line_profiler` (`%lprun`) | Qué línea concreta cuesta |
| Memoria | `memory_profiler` (`%mprun`) | Cuánta RAM por línea / objeto |
| Peak real | `tracemalloc`, RSS de `/proc` | Consumo máximo del proceso |

**El bucle de perfilado.**

```text
hipótesis de cuello de botella → medir (profiler) → identificar causa raíz
→ arreglar → medir otra vez → confirmar mejora y que nada regresó
```

**Regla práctica.** El 90% del tiempo suele estar en el 10% de las líneas.
Perfila el pipeline completo una vez para encontrar dónde, y solo después usa
`timeit` para iterar fino sobre esa zona. Nunca optimices "por intuición" una
función que no ha salido en el perfil.

**Cómo falla.** Optimizar la limpieza de strings (2 ms) cuando el 40 s del
pipeline están en un join sin índice. O peor: medir con `time.time()` un
bucle caliente que amortiza el JIT/alloc, y "mejorar" la versión equivocada.

## Vectorización vs bucles Python

**Principio.** NumPy/Pandas delegan el cómputo a bucles C sobre memoria
contigua; un bucle Python paga la sobrecarga del intérprete por elemento.

**Por qué los ufunc ganan.** Cada iteración de un bucle Python implica:
interpretación del bytecode, dispatch dinámico, boxing/unboxing de objetos
y acceso no contiguo. Un ufunc recorre un buffer C homogéneo sin ese
overhead: típicamente 10-100× más rápido.

**Práctica.** Sustituye bucles por operaciones de array:

```python
# Lento: ~10⁵ iteraciones en el intérprete
result = [np.dot(w, x) for x in rows]          # type: ignore

# Rápido: un matmul en C
result = rows @ w
```

Broadcasting: `arr[:, None] - center` expande dimensiones sin copiar en
memoria; las operaciones se ejecutan sobre el array virtual.

**Cuándo el bucle gana.** Cuando el trabajo por fila es irreductiblemente
secuencial o no vectorizable (un recorrido de grafo, un estado que depende de
la fila anterior, parsing de texto irregular). Ahí el bucle es correcto;
vectorizarlo con trucos puede ser más lento y críptico. Alternativa real:
repartir el trabajo (ver Paralelismo).

**Cómo falla.** `df.apply(func, axis=1)` con una función Python convierte el
pipeline vectorizado en un bucle disfrazado, sin ganar nada. Y el patrón
inverso: forzar vectorización con `groupby.transform` anidados que hacen tres
pasadas sobre los datos cuando una sola operación bien elegida bastaba.

## Memoria y representación de datos

**Principio.** Los datos en memoria dominan coste, caché y swap; los dtypes
correctos cambian el juego antes que cualquier truco de algoritmo.

**Práctica.**

- **Dtypes.** `float32` (y a veces `float16`) reducen la RAM a la mitad que
  `float64` con pérdida tolerable de precisión para modelos lineales y GB.
  Categorías como `category` de Pandas comprimen strings repetidos.
- **Layout contiguo.** Las operaciones vectorizadas rinden mejor sobre arrays
  contiguos (C-order). Arrays en Fortran-order (`order='F'`) ayudan en
  cómputos column-por-columna; mezclarlos rompe la caché.
- **Evitar copias.** `df.copy()`, `pd.concat` y slicing que no es vista crean
  copias. Usa views, `inplace` donde aplique y `.to_numpy()` cuando el índice
  sobra.
- **Indexación.** Índices `int32`/`int64` explícitos en vez de `float64`
  (Pandas usa float64 para índices con NaN); ahorran memoria en joins y
  groupby.
- **Streaming / out-of-core.** Cuando el dataset no cabe: procesa por chunks
  (`pandas.read_csv(..., chunksize=)`), o pasa a Dask (DataFrame perezoso,
  agrega por particiones) o Polars (query engine en Rust, evaluación perezosa
  y por columnas).

**Cómo falla.** Un `pd.concat` de 50 archivos que duplica picos de memoria; un
DataFrame con columnas de strings Python-unicode que triplica el RSS; o
pensar que "necesitas una máquina más grande" cuando `downcast` + `category`
+ cargar solo lo necesario reduce el problema a un tercio.

## Entrada/salida de datos

**Principio.** Leer y escribir datos es a menudo el 60% del tiempo de un
pipeline; el formato y lo que se carga importan más que la transformación.

**Práctica.**

- **Parquet sobre CSV.** Parquet es columnar (solo lee las columnas pedidas),
  comprime por columna y guarda esquema y estadísticas. CSV es texto, lento
  de parsear y sin esquema. Regla: datos de trabajo en parquet; CSV solo como
  frontera de intercambio.
- **Cargar solo lo necesario.** `pd.read_parquet(..., columns=[...])` y
  filtros de partición (`filters=` en parquet, predicados en DuckDB/Polars)
  evitan traer columnas que no se usan.
- **Caché y precomputación.** Features caras de calcular (embeddings, agregados
  por grupo, normalizaciones) se calculan una vez y se persisten a parquet;
  el pipeline relector los consume. Nunca recalcules lo que ya está en disco.

**Cómo falla.** Un pipeline que relee el CSV crudo de 2 GB en cada ejecución
"para estar seguro de que no está stale", cuando lo correcto es un step de
ingesta que escribe parquet con versión y todo lo demás lee de ahí.

## Paralelismo: threads vs procesos

**Principio.** El GIL serializa la ejecución de bytecode Python; los threads
solo ganan cuando el trabajo libera el GIL (I/O, NumPy/Pandas en C). El
cómputo Python puro exige procesos.

**Práctica.**

| Escenario | Mecanismo |
|-----------|-----------|
| I/O (descargas, lecturas) | threads o asyncio |
| Cómputo NumPy/Pandas ya vectorizado | nada: ya usa hilos C y caché |
| Cómputo Python puro por unidad independiente | `multiprocessing` / `joblib.Parallel` |
| Misma función sobre muchas muestras | `joblib.Parallel(n_jobs=-1)` con `return_as="list"` |

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(delayed(compute)(x) for x in samples)
```

**Ley de Amdahl.** La ganancia está limitada por la parte serial:
speedup ≤ 1 / (s + (1-s)/N). Si el 20% es serial, el techo con infinitos
núcleos es 5×. El overhead de spawn, serialización (pickle) y trasvase de
resultados puede hacer que paralelizar una tarea de milisegundos la
ralentice.

**Cómo falla.** `ThreadPool` sobre cómputo Python puro (no gana nada, el GIL
lo serializa), o `Parallel` sobre un millón de llamadas diminutas que pagan
más en pickle que lo que ahorran en CPU. Regla: paraleliza el nivel grueso
(por archivo, por modelo, por partición), nunca el elemento suelto.

## GPU

**Principio.** La GPU gana con trabajo vectorizado masivo y alta densidad de
cómputo; el cuello de botella es mover datos entre CPU y GPU.

**Práctica.**

- **Batching.** Nunca mandes ejemplos de uno en uno: el overhead de
  transferencia PCIe (~ms) domina. Agrupa en batches grandes y consúmelos en
  streaming.
- **Transferencia como límite.** 1 GB/s a 10 GB/s de PCIe frente a 100+ GB/s
  de VRAM: si el trabajo por byte transferido es poco, la GPU pierde.
- **Mixed precision.** `float16` en cómputo y `float32` en acumuladores
  (amp en PyTorch/TensorFlow) multiplica el throughput en hardware moderno.
- **Cuándo NO merece.** Datos tabulares pequeños (< 10⁶ filas), modelos
  lineales o árboles (LightGBM/XGBoost en CPU suelen ganar), y cualquier cosa
  donde la transferencia sea el trabajo real.

**Cómo falla.** Un grid search de modelos pequeños en GPU cuyo tiempo lo domina
el `to(device)` de cada batch; o un dataset de 50 MB que tarda más en subir a
VRAM que en entrenarse en CPU.

## Trade-offs espacio-tiempo

**Principio.** Caché, memoización y estadísticas precomputadas compran tiempo a
cambio de memoria; la decisión correcta depende de la frecuencia de acceso.

**Práctica.**

- **Caché.** `functools.lru_cache` para funciones deterministas puras; caché a
  disco (parquet/feather) para resultados costosos entre ejecuciones.
- **Memoización.** En DP y recursión, guardar subresultados evita recomputar.
- **Estadísticas precomputadas.** Medias, min/max, cuantiles por grupo
  calculadas una vez y consultadas después: convierten un agregado O(n) por
  request en un lookup O(1).

**Cómo falla.** Cachear sin límite de memoria en un servidor de inferencia
(crece hasta el OOM), o precomputar agregados que solo se usan una vez y
mueren en disco ocupando espacio para nada.

## El 80/20 del rendimiento en DS

**Principio.** La mayoría del tiempo de un proyecto está en preparación de
datos y en entrenamiento; optimiza ahí, no en la línea del notebook que solo
ejecutas una vez.

**Práctica.**

- Mide dónde cae el tiempo del pipeline real (perfilado, no intuición).
- Orden de impacto típico: formato y volumen de datos → joins y agregados →
  features vectorizadas → entrenamiento → inferencia.
- Trata la inferencia como un presupuesto (latencia p50/p99 y throughput),
  no como un "bonus".

**Cómo fallar.** Premature optimization: reescribir un preprocesado de 10 ms
que corre una vez, mientras un `read_csv` repetido de 3 GB domina el flujo.
Optimiza lo que el profiler dice, y solo cuando un número real lo justifique.

## Fuentes

- Documentación de cPython profilers: `pstats`, `cProfile`, `tracemalloc`.
- Documentación de numpy (ufuncs, broadcasting, `order='C'/'F'`).
- Documentación de pandas (dtypes, `category`, `downcast`, `chunksize`).
- Dask y Polars: evaluación perezosa y out-of-core.
- Documentación de joblib (`Parallel`, backend `loky`/`threading`).
- PyTorch AMP / mixed precision y TensorFlow mixed precision.
- Gene Amdahl, "Validity of the Single Processor Approach" (1967).
- Martin Kleppmann, "Designing Data-Intensive Applications" (O'Reilly, 2017).
