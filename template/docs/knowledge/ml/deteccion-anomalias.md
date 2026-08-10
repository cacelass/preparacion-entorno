{% if ml_type == 'no_supervisado' %}
# Detección de anomalías: outlier, novelty y sus métodos

La detección de anomalías es el problema de marcar las observaciones que no
siguen el comportamiento mayoritario. No es un problema único: es una familia
con dos sabores, cada uno con su supuesto sobre los datos, su evaluación y su
modo de fallo. Este documento cubre la teoría y, sobre todo, las decisiones
que se toman mal en producción.

## El problema: dos sabores, no uno

- **Outlier detection.** Los datos de entrenamiento están **contaminados**:
  una fracción desconocida son anómalos. No hay etiquetas, y el modelo debe
  aprender "lo normal" aunque nunca se le dice qué es. Los outliers pueden
  estar dentro del conjunto de entrenamiento, contaminando la propia
  definición de normalidad que el método estima.
- **Novelty detection.** El entrenamiento contiene **solo datos normales**,
  garantizados o asumidos. El modelo aprende una descripción de lo normal y,
  en inferencia, marca como nuevo todo lo que se desvía de esa descripción.

No son intercambiables. La diferencia no es cosmética: la contaminación del
train es una fuente de error que el novato ignora. Un método entrenado con
datos contaminados (p.ej. autoencoder ajustado con outliers incluidos)
internaliza las anomalías como normales y las reproduce en reconstrucción. El
flujo correcto exige decidir cuál de los dos sabores aplica — si el conjunto
de entrenamiento puede contener anomalías, el método debe tolerarlas o se
necesita una limpieza previa.

Los dominios típicos: fraude (transacciones), intrusiones de red (tráfico),
control de calidad (piezas defectuosas) y novedades (sistema nuevo, sin
historial).

## Estadística clásica: z-score, MAD, Grubbs/ESD

Para unidimensionales, la anomalía es "lejos de la media". El **z-score**:

$$ z_i = \frac{x_i - \bar{x}}{s} $$

con $s$ la desviación estándar muestral. Marcar $|z_i| > 3$ (o el cuantil que
se decida). Problema: $\bar{x}$ y $s$ son sensibles a los outliers, que
inflan $s$ y esconden los extremos (enmascaramiento). Dos o más outliers
pueden hacerse pasar por normales mutualmente.

El **MAD** corrige la sensibilidad usando la mediana:

$$ \mathrm{MAD} = \mathrm{med}_i |x_i - \mathrm{med}_j x_j|, \qquad
   M_i = \frac{0.6745\,(x_i - \mathrm{med})}{\mathrm{MAD}} $$

El factor $0.6745$ hace que $M_i$ sea comparable al z-score bajo normalidad
(para la gaussiana, $E[\mathrm{MAD}] \approx 0.6745\,\sigma$). Los puntos con
$|M_i| > 3.5$ se consideran outliers. La mediana tiene un breakpoint del 50%:
resistente hasta que la mitad de los datos sea contaminación.

**Grubbs** es un test formal: $G = \max_i |x_i - \bar{x}| / s$, contrastado
contra la distribución t para decidir si el máximo es demasiado grande.
**ESD (generalized ESD)** generaliza a $r$ outliers detectándolos uno a uno
y eliminando cada máximo. Útil para fijar cuántos outliers hay con una tasa
de falso positivo controlada.

**Limitación estructural.** Todos estos métodos asumen una distribución
unimodal y aproximadamente simétrica. Con datos bimodales, multimodales o con
cola pesada, "lejos de la media" no significa "anómalo": la moda minoritaria
parece anómala y un punto legítimo en la cola de una distribución de
Pareto sale como outlier. Ante multimodalidad, modelar con mezclas (GMM) y
mirar el **score por componente**, o saltar a métodos sin supuesto de forma.

## Distancia y densidad: k-NN y LOF

La anomalía como punto **lejos de sus vecinos**. Para cada $x$, la distancia
a su $k$-ésimo vecino:

$$ \mathrm{score}(x) = d_k(x) = \|x - x_{(k)}\| $$

Elevada = anómala. No requiere distribución: sirve para formas arbitrarias.
Coste $O(n^2)$ si se computa toda la matriz; con índices (KD-tree, ball tree,
annoy, HNSW) se abarata. Problema clásico: **la densidad no es global**.
Un punto normal en una zona rala queda con $d_k$ grande y se marca anómalo
por vivir lejos de los demás, no por serlo.

**LOF (Local Outlier Factor)** corrige exactamente eso. La densidad local se
define a través de la *reachability distance*:

$$ \mathrm{reachdist}_k(p, o) = \max\{\mathrm{kdist}(o), d(p, o)\} $$

con $\mathrm{kdist}(o)$ la distancia al $k$-ésimo vecino de $o$. La
*local reachability density*:

$$ \mathrm{lrd}_k(p) = \left( \frac{1}{|N_k(p)|}
   \sum_{o \in N_k(p)} \mathrm{reachdist}_k(p, o) \right)^{-1} $$

y el factor:

$$ \mathrm{LOF}_k(p) = \frac{1}{|N_k(p)|}
   \sum_{o \in N_k(p)} \frac{\mathrm{lrd}_k(o)}{\mathrm{lrd}_k(p)} $$

LOF ≈ 1: la densidad local es comparable a la de sus vecinos (normal).
LOF ≫ 1: mucho menos denso que sus vecinos (outlier local). La división de
densidades es lo que **corrige la densidad variable**: un punto en una zona
rala tiene $\mathrm{lrd}$ pequeña, pero también sus vecinos; el ratio se
mantiene cerca de 1.

**El parámetro $k$.** Pequeño (3–10): captura outliers muy locales, pero
ruido; con $k$ demasiado pequeño y puntos aislados por azar, falsos
positivos. Grande: solo detecta desviaciones a escala global; pierde los
outliers locales. La elección de $k$ define *qué vecindario cuenta como
local* — es el análogo del $\varepsilon$ de DBSCAN y, como él, dominado por
la maldición de la dimensionalidad. Hay variantes (ALOCI) que lo promedian.

## Basado en árboles: Isolation Forest

**Intuición.** Cortar el espacio con particiones aleatorias recursivas
(árboles de aislamiento). Un punto anómalo está solo o es escaso: se aísla con
**pocos cortes** — la longitud del camino $h(x)$ hasta aislarlo es corta. Un
punto normal, en zona densa, necesita muchos cortes. No se modela densidad ni
distancia: solo la profundidad de aislamiento.

Para un punto $x$ se promedia la profundidad sobre $t$ árboles,
$E[h(x)]$, y se normaliza con la profundidad media esperada de un árbol:

$$ c(n) = 2H(n-1) - \frac{2(n-1)}{n}, \qquad H(i) = \ln(i) + \gamma $$

con $\gamma$ la constante de Euler–Mascheroni y $n$ el tamaño del conjunto de
entrenamiento. El score:

$$ s(x) = 2^{-E[h(x)] / c(n)} $$

- $s \to 1$: profundidad corta → anomalía clara.
- $s \approx 0.5$: profundidad igual a la esperada → normal.
- $s < 0.5$: camino largo → punto denso, inconfundiblemente normal.

**Por qué funciona sin asumir distribución.** No estima densidad ni ajusta
parámetros de forma; la única hipótesis es que los anómalos son *pocos y
distintos* (fácilmente separables con cortes aleatorios). Es lineal en
$O(n\,t\,\psi)$ con tamaño de submuestra $\psi$ (típicamente 256) por árbol:
escalable donde LOF y k-NN no lo son. Los cortes son **axis-parallel**:
estructuras rotadas o diagonales lo degradan (Extended Isolation Forest usa
hiperplanos con pendiente aleatoria para corregirlo).

**Parámetros.** `n_estimators` ($t$, p.ej. 100), `max_samples` ($\psi$) y
`contamination`: la fracción esperada de outliers, usada solo para fijar el
umbral del score. Si se subestima, se marcan menos de los que hay; si se
sobreestima, los "peores normales" se cuelan como anómalos. No es un
hiperparámetro del modelo — es una afirmación sobre el mundo.

## Reconstrucción: PCA residual y autoencoders

**PCA.** Proyectar a los $k$ componentes principales y reconstruir:
$\hat{x} = V_k V_k^\top (x - \bar{x}) + \bar{x}$. El error de reconstrucción:

$$ e(x) = \|x - \hat{x}\|_2^2 = \| (I - V_k V_k^\top)(x - \bar{x}) \|_2^2 $$

Los datos normales viven cerca del subespacio principal (varianza alta), así
que se reconstruyen bien. Un anómalo, alineado con direcciones de varianza
baja o fuera del subespacio, tiene $e(x)$ alto. Ventaja: la proyección es
barata y la interpretación directa (las direcciones residuales dicen *en qué*
se desvía). Es un modelo **lineal**: si la normalidad no es lineal, PCA la
capta mal.

**Autoencoder.** Mismo principio, no lineal: $f$ codifica a un espacio
latente (bottleneck) y $g$ reconstruye; $e(x) = \|x - g(f(x))\|_2^2$. La red
aprende a reconstruir la variedad de lo normal. Anomalía = error alto. Con
datos no lineales (imágenes, secuencias) supera a PCA; a cambio, requiere
mucho más datos y tuning (arquitectura, regularización, validación) y el
error de reconstrucción no tiene escala comparable entre dominios.

**Sensibilidad a la contaminación.** Ambos se entrenan minimizando el error
medio de reconstrucción. Si el train contiene outliers, el óptimo las
internaliza parcialmente: reconstruirlas mejor reduce la pérdida media, y el
umbral de anomalía sube. Con fracción de anomalías alta, el autoencoder
aprende "lo mayoritario" mal y produce reconstrucciones con error repartido.
La robustez se ataca limpiando el train antes o usando pérdidas robustas
(huber, cuantiles) que limiten la influencia de los peores puntos.

## One-class SVM ($\nu$-SVM)

Aprende una **frontera** alrededor de los datos normales, no una función de
densidad: maximiza la separación del origen en un espacio kernel con fracción
$\nu$ de los puntos fuera de la frontera (anómalos tolerados en train).

$$ \min_{w, \rho, \xi} \frac{1}{2}\|w\|^2 + \frac{1}{\nu n}
   \sum_i \xi_i - \rho \quad \text{s.a.} \quad
   w^\top \phi(x_i) \ge \rho - \xi_i, \; \xi_i \ge 0 $$

La decisión es $w^\top \phi(x) \ge \rho$ (dentro de la frontera = normal).
$\nu \in (0, 1]$ acota superiormente la fracción de outliers en train e
inferiormente la de support vectors. A diferencia de Isolation Forest o LOF,
devuelve una frontera *explicita*: útil cuando la frontera de lo normal es el
objeto, no solo un score.

**El límite: alta dimensionalidad.** El kernel gaussiano necesita un $\gamma$
adecuado; con muchas dimensiones, los puntos se vuelven equidistantes, la
frontera envuelve el ruido y el modelo sobreajusta el contorno de lo normal.
Funciona bien con datos tabulares de dimensión moderada y contaminación baja;
con alta dimensión conviene reducir previamente o usar Isolation Forest, que
no sufre del colapso de distancias. Notable: el $\nu$-SVM optimiza la
frontera, no la densidad — dos regiones normales muy separadas se envuelven
con la misma frontera y "lo nuevo" entre ellas puede quedar clasificado como
normal.

## Evaluación sin etiquetas

El score de anomalía es un **ranking**, no una decisión. Evaluar bien un
detector no supervisado sin etiquetas requiere montar un marco de referencia:

- **precision@k.** Marcar los $k$ puntos con score más alto y estimar por
  muestreo qué fracción son realmente anómalos (inspección, reglas de
  negocio). Es la métrica honesta cuando no hay ground truth: mide cuánto
  vale la "lista corta" que el detector entrega a un humano.
- **ROC con contaminación sintética.** Inyectar outliers conocidos
  (sintéticos o re-etiquetados) en un conjunto presumiblemente limpio y
  medir AUC: ¿el score los pone arriba? Poderoso, con una trampa: los
  outliers sintéticos deben parecerse a los reales, y eso casi nunca se sabe
  de antemano — miden el detector contra *esa* familia de anomalías, no
  contra las reales.
- **Curva detection rate vs false alarm.** No un punto, toda la curva: para
  cada umbral, tasa de detección (recall de anomalías) frente a tasa de
  falsas alarmas. Es la representación correcta para decidir en qué punto
  operar según el coste.

**Por qué AUC global engaña.** Cuando solo el 0.1% es anómalo, el AUC se
domina por los normales: un detector que marca todo como normal tiene AUC ≈
0.5 + pequeñísima ganancia, y mover el umbral a lo loco apenas se refleja. El
AUC pesa todos los pares por igual y los pares normales-normales son
millones. La evaluación correcta es la **curva PR** (o precision@k), que se
enfoca en la clase minoritaria: ahí un detector inútil queda cerca de la
línea de base $p/(p+n)$ y cualquier mejora es visible.

## Umbral

El score es un ranking; el **umbral es una decisión de negocio**. Fijarlo "a
ojo" (p.ej. el percentil 99) ignora el coste de los errores. La decisión
óptima minimiza el coste esperado: con $c_{FP}$ el coste de una falsa alarma
y $c_{FN}$ el coste de dejar pasar una anomalía, el umbral óptimo cae donde
$p(\text{FP}) \cdot c_{FP} = p(\text{FN}) \cdot c_{FN}$ — en la curva de
detection rate vs false alarm, en el punto de tangencia con la recta de coste.
Fraude de alto valor con investigaciones caras → umbral conservador;
intrusión de red con alarmas baratas → umbral agresivo.

**Contaminación tolerable por método** (orientativo, no ley):

| Método | Contaminación tolerable | Motivo |
|---|---|---|
| MAD / z-score robusto | hasta ~50% | mediana: breakpoint alto |
| Grubbs / ESD | ~5–10% | asume casi toda la muestra normal |
| LOF / k-NN | ~5–10% | vecinos contaminados deforman la densidad local |
| Isolation Forest | ~10–20% | los normales dominan el aislamiento |
| PCA / autoencoder | ~5–10% | la reconstrucción media absorbe outliers |
| $\nu$-SVM | acotada por $\nu$ | $\nu$ es la fracción máxima tolerada |

Si la contaminación real supera la tolerancia del método, el primer paso es
limpiar el train o cambiar de sabio a *novelty* (entrenar solo con normales).

## Trampas

1. **Colapso de densidad en alta dimensión.** La maldición pega más fuerte
   aquí: con muchas dimensiones las distancias se concentran (los puntos se
   vuelven equidistantes), el vecino $k$-ésimo deja de discriminar y LOF/k-NN
   colapsan. Isolation Forest no usa distancias y aguanta mejor; aún así,
   reducir dimensión (PCA/UMAP) antes de métodos basados en densidad es
   recomendable.
2. **Anomalías contextuales/temporales.** En series, "anómalo" se define
   contra el contexto: 200 transacciones en un minuto es normal un viernes y
   anómalo un lunes a las 4 a.m. El score global ignora el contexto; hay que
   modelar la normalidad **condicional** (por hora, por usuario, por
   segmento) o las anomalías contextuales se pierden en el agregado.
3. **Drift: las estadísticas normales envejecen.** Lo normal de ayer no es lo
   normal de hoy (campaña, cambio de estacionalidad, nueva población). Un
   detector entrenado una vez acumula falsos positivos con el tiempo; ver
   [ciclo-vida-mlops.md](ciclo-vida-mlops.md).
   {% if use_monitoring %}
   Este proyecto trae `monitoring/monitor.py` con `make monitor`: drift
   KS/chi² de features y degradación de métricas frente a baseline. Úsalo
   como señal para **reentrenar** el detector (o su referencia de "normalidad")
   cuando el score empiece a dispararse sin que cambie el negocio.
   {% endif %}
4. **Streaming.** En flujo, los datos llegan sin parar y la definición de
   normal evoluciona. El modelo offline (ajustado una vez) no sirve: hace
   falta un modelo **online** que actualice la referencia incrementalmente
   (half-space trees, iForest incrementales, o ventanas deslizantes) con
   coste por punto acotado.
5. **El mito de "detectar todo".** Ningún método no supervisado detecta todo
   tipo de anomalía. Cada método captura una familia: densidad (LOF),
   aislamiento (iForest), reconstrucción (AE), frontera ($\nu$-SVM). Lo que
   no cae en la familia se pierde. El problema real es definir *qué familia*
   importa para el negocio.

## Práctica: cómo enmarcar

1. **¿Hay etiquetas de fraude/anomalía?** Entonces es un problema
   **supervisado con desbalanceo severo**, no no-supervisado: modelos
   binarios con class weights, evaluación por PR y umbral por coste — ver
   [clasificacion.md](clasificacion.md) y
   [metricas-y-evaluacion.md](metricas-y-evaluacion.md). El encuadre como
   anomalía solo tiene sentido cuando las etiquetas no existen o no cubren la
   variedad de fraudes nuevos. Muchos equipos usan etiquetas para evaluar un
   detector no supervisado: legítimo, pero el detector no debe verlas.
2. **Combinar scores.** Ningún método es universal: estandarizar y combinar
   (media, mediana, rank-average) scores de familias distintas — densidad +
   aislamiento + reconstrucción — suele dar más robustez que el mejor solo.
   Los scores de métodos distintos tienen escalas distintas: combinar
   **rangos o cuantiles**, no valores crudos.
3. **Reporte honesto.** Toda métrica de detección va con su tasa de falsos
   positivos: "detectamos el 80% de los fraudes" sin decir a cambio de cuántas
   investigaciones inútiles es una cifra engañosa. Reportar un punto de la
   curva (detection rate, false alarm), la fracción de datos marcada y el
   coste operativo de cada error — no un AUC en abstracto.

## Fuentes

- **Isolation Forest** — F. T. Liu, K. M. Ting, Z.-H. Zhou (ICDM 2008).
  DOI 10.1109/ICDM.2008.17 — https://doi.org/10.1109/ICDM.2008.17
  (sin arXiv; el ID arXiv:1811.02141 que a veces se cita es **Extended
  Isolation Forest**, de Hariri et al., no el original).
- **Extended Isolation Forest** — S. Hariri, M. C. Kind, R. J. Brunner
  (TKDE 2021). arXiv:1811.02141 — https://arxiv.org/abs/1811.02141
- **LOF: Identifying Density-Based Local Outliers** — M. M. Breunig, H.-P.
  Kriegel, R. T. Ng, J. Sander (SIGMOD 2000).
  DOI 10.1145/342009.335388 — https://doi.org/10.1145/342009.335388
- **Anomaly Detection: A Survey** — V. Chandola, A. Banerjee, V. Kumar
  (ACM Computing Surveys 41(3), 2009). DOI 10.1145/1541880.1541882 —
  https://doi.org/10.1145/1541880.1541882
{% endif %}
