# Visualización de datos

La figura es un argumento, no un adorno: comunica una afirmación sobre los
datos y un lector senior debe extraerla en segundos. Elegir el gráfico, la
escala y el color son decisiones de comunicación, y siguen una jerarquía
empírica de cómo el sistema visual humano decodifica cada codificación.

## La jerarquía perceptiva de Cleveland-McGill

Cleveland y McGill (1984) midieron con experimentos de psicofísica cuánto
cuesta al ojo extraer un valor de cada tipo de codificación:

| Precisión | Codificación | Uso típico |
|-----------|--------------|------------|
| Máxima | Posición en un eje común | Scatter, líneas, barras |
| Alta | Longitud | Barras (solo longitud codifica) |
| Media | Ángulo | Pie charts, donuts |
| Baja | Área | Burbujas, mapas de mosaico |
| Mínima | Color / tono | Relleno y matiz, no para valores finos |

La jerarquía no es estética: es cuánto error introduce el ojo al convertir el
píxel en número. Consecuencias prácticas:

- **Eje compartido antes que facetas apiladas**: comparar por posición en un
  eje común (barras lado a lado, líneas en la misma escala) es más preciso que
  comparar paneles apilados con ejes libres; los ejes independientes esconden
  diferencias de magnitud entre paneles.
- **Barras antes que pies**: el pie codifica por ángulo y área, dos niveles
  por debajo de la longitud. Para 2-4 categorías una barra (horizontal si las
  etiquetas son largas) se lee al instante; el pie solo aporta cuando el
  lector ya conoce el todo.
- **Nunca burbujas para comparar valores**: la percepción del área crece con
  el cuadrado del radio; para mostrar "el doble", el ojo exige el doble de
  diámetro y el área engaña. El área solo sirve cuando se quiere percibir
  magnitud agregada, no comparar.

## Elegir el gráfico según el dato

La pregunta primero, el gráfico después: ¿un valor a lo largo del tiempo, una
distribución, una relación entre variables, o las partes de un todo?

### Series temporales

La línea es el gráfico por defecto: codifica por posición (la más precisa) y
respeta la ordenación temporal. Reglas:

- **Ejes honestos**: el eje Y empieza en cero o lleva una rotura marcada; no
  se amplía la escala para dramatizar una tendencia sin señalarlo.
- Una línea por serie en el mismo panel cuando se comparan; el color las
  distingue (no la forma de punto).
- Series de magnitudes muy dispares: paneles con eje compartido o escala
  doble **marcada**, nunca escala doble silenciosa — el cruce de curvas es un
  artefacto, no una relación.
- No barras para series largas: la barra es para conteos discretos, no para
  la densidad de una trayectoria.

### Distribuciones

| Gráfico | Qué muestra | Cuándo |
|---------|-------------|--------|
| Histograma | Forma cruda; depende de `bins` | Primera vista; n grande; modas |
| KDE | Forma suavizada; depende del bandwidth | Comparar varias distribuciones |
| Boxplot | Cuartiles, IQR, outliers | Comparar muchas; resumir |
| Violín | Densidad + cuartiles | KDE y boxplot a la vez; n moderada |

- **Histograma vs KDE**: el KDE suaviza pero falsea donde no hay datos
  (estira a 0 con kernels anchos) y oculta la escasez; el histograma la
  muestra. Solapados: KDE. Resumir: boxplot.
- **El boxplot enmascara la multimodalidad**: dos modas caben dentro del
  mismo box. Si el dato es multimodal, violín o histograma; el boxplot solo
  se acompaña de la densidad.
- **Log para colas largas**: distribuciones asimétricas (ingresos, tiempos)
  se leen en escala log, con etiquetas en la unidad original.
- **Empirical CDF** para comparar colas y percentiles con n grande y
  sobretrazado severo.

### Relaciones

- **Scatter** para n pequeña-moderada. Con sobretrazado (miles de puntos que
  se pisan), en orden: transparencia (alpha) → jitter (desplazamiento
  aleatorio pequeño; obligatorio si una variable es discreta) → hexbin.
- **Hexbin**: agrega por celda y codifica por color el conteo; para
  n ≫ 10⁵. El color es la densidad, no un valor por punto.
- Correlación: el **scatter matrix (SPLOM)** es más honesto que la matriz de
  números: la matriz esconde forma (curvas, clusters, outliers). Un r de 0.9
  con forma de media luna es engañoso.
- Variables discretas: jitter siempre, con marginales.

### Partes de un todo

- **Barras apiladas** para comparar composiciones entre grupos: longitud
  total visible, segmentos por color.
- **NUNCA pie con más de 4 categorías**: el ojo no ordena 7 sectores por
  ángulo. Con >4: barras horizontales apiladas al 100 %.
- Proporciones sin total: barras al 100 %. La donut no arregla el pie: sigue
  codificando por ángulo.

## Ejes y escala

- **El cero importa**: para cantidades, un eje Y que no empieza en 0 exagera
  diferencias. No truncar sin marcar la rotura (`//`), y aun marcada es una
  decisión que se explica en el caption. Alternativa honesta: dos paneles o
  error bars, no un Y recortado.
- **Log-scale cuándo**: cuando el dato abarca varios órdenes de magnitud
  (precios, tráfico) o se comparan ratios. Ticks 1-2-5·10^k, etiquetas en la
  unidad original.
- **Ticks limpios**: redondos, sin decimales innecesarios; la densidad de
  ticks debe permitir leer la serie, no decorar el eje.
- **Unidades siempre** en ejes y leyenda; el lector no infiere multiplicadores
  (k, M) implícitos.

## Color

| Tipo de paleta | Cuándo | Ejemplos |
|----------------|--------|----------|
| Categórica | Grupos sin orden | Okabe-Ito, Tableau 10 |
| Secuencial | Magnitud ordenada (densidad, conteo) | Viridis, magma |
| Divergente | Desviación de un punto medio | RdBu, BrBG |

- **Colorblind-safe**: Okabe-Ito para categóricas, **viridis** para
  secuenciales (perceptualmente uniforme). Nunca rojo/verde como único par de
  contraste.
- **El color no codifica cuando la posición ya lo hace**: en un scatter la
  posición ya separa los datos; colorear por un tercer valor añade dimensión,
  colorear por la misma variable del eje es redundancia.
- El color solo codifica magnitud cuando es la única vía (heatmaps, hexbin);
  nunca para valores individuales comparables por ejes.
- Máximo ~8 categorías distinguibles; más, agrupar en "Otros".

## Estadística en la figura

- **Error bars**: muestran incertidumbre de la estimación (SEM, IC), no
  dispersión de los datos. Confundir `std` con `sem` produce figuras que
  parecen más precisas de lo que son: la nota dice qué se dibuja.
- Para comparar modelos, dibujar la **distribución de la métrica en CV**
  (boxplot/violín por modelo) además del punto: dos medias iguales con
  varianzas distintas no son comparables.
- **Intervalos en series**: la banda de IC en una evolución de pérdida muestra
  cuándo dos ejecuciones se separan de verdad; sin banda, ruido parece
  tendencia.
- No dibujar más de lo que se sabe: sin estimación de incertidumbre no se
  ponen barras de adorno.

## Storytelling

- **El título es el takeaway**: "Resultados" no dice nada; "El modelo B gana
  en 8 de 10 slices" dirige la lectura. El título hace la afirmación, la
  figura la respalda.
- **Anotaciones que señalan**: marcar el evento (deploy, drift, cambio de
  campaña) con una línea vertical y etiqueta; la anotación cuenta la historia
  sin explicación oral.
- **Small multiples**: en vez de un panel con 12 series ilegibles,
  mini-paneles con eje compartido y el mismo rango. Se compara por posición
  (alta precisión) manteniendo cada serie legible.
- **Figura autosuficiente**: caption con qué se ve, leyenda completa, unidades
  y fuente del dato al pie. Una figura que requiere explicación oral no es
  figura de informe.

## Reproducibilidad

La figura se genera por código desde `data/` a `reports/figures/`, con semilla
fija, en un script versionado. Una figura hecha a mano (Excel, dibujo,
recorte) no es evidencia: no se regenera, no refleja exactamente los datos ni
el momento del pipeline, y no se puede auditar.

```python
# figures/fig_03_modelos.py — una figura, un script, semilla fija
import matplotlib.pyplot as plt
import numpy as np
from {{ project_slug }}.utils.paths import FIGURES_DIR, PROCESSED_DATA_DIR

rng = np.random.default_rng(42)   # misma figura en cada ejecución
df = load_processed(PROCESSED_DATA_DIR)
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(df["epoch"], df["loss"], yerr=df["ci"], capsize=3)
fig.savefig(FIGURES_DIR / "fig_03_modelos.png", dpi=300, bbox_inches="tight")
```

- **data → figure**: la figura lee de `data/processed/` o del run de
  entrenamiento, nunca de valores copiados a mano.
- **Seed fija**: el ruido reproducible (jitter, bootstrap, CV) es idéntico
  entre ejecuciones; sin seed, la figura cambia y no se compara con la de
  ayer.
- **Script por figura**: un `figures/fig_XX_*.py` por gráfico, invocable con
  `make figures`; cada figura tiene dueño y regeneración.
- Registrar en el caption la versión del código y del dato: la figura es un
  snapshot del commit que la generó.

## Práctica: las 5 figuras de un proyecto DS

{% if ml_type == 'no_supervisado' %}
En clustering, las figuras 4 y 5 se sustituyen por el elbow/silhouette de
k y la proyección de los clusters; el resto aplica igual.
{% endif %}

1. **EDA inicial**: mapa de missingness + histogramas de las variables clave
   (log donde toque). Primera lectura honesta del dato.
2. **Distribución del target**: histograma/KDE del target y su transformación;
   justifica pérdida y transformación (ver [regresion.md](../ml/regresion.md)).
3. **Matriz de correlación honesta**: SPLOM, o matriz + scatter de las pares
   más correlacionadas. La matriz numérica sola miente sobre la forma.
4. **Comparación de modelos con error bars**: distribución de la métrica en CV
   por modelo (boxplot/violín) + mejor punto por modelo.
5. **Evolución del error**: pérdida por epoch en train/validación, con banda
   de intervalos si hay múltiples seeds. Overfitting y convergencia en una
   imagen.

## Fuentes

- **Graphical Perception: Theory, Experimentation, and Application to the
  Development of Graphical Methods** — W. S. Cleveland, R. McGill (1984).
  https://doi.org/10.1080/01621459.1984.10478080
- **The Visual Display of Quantitative Information** — E. R. Tufte (1983).
  Sin DOI estable — https://www.edwardtufte.com/tufte/books_vdqi
- **Storytelling with Data: A Data Visualization Guide for Business
  Professionals** — C. Knaflic (2015). Sin DOI estable —
  https://www.storytellingwithdata.com/
