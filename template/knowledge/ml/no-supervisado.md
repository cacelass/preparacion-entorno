{% if ml_type == 'no_supervisado' %}
# Aprendizaje no supervisado: clustering y detección de anomalías

## Fundamentos: qué formaliza un "cluster"

Un clustering es una **partición** $C_1, \dots, C_K$ de los $n$ puntos tal que
la dispersión intra-cluster es baja y la separación entre clusters es alta.
No se descubre el "grupo real": se elige una partición que optimiza un
criterio fijado sobre una representación y una métrica concretas.

Dado el centroide de cada cluster $\mu_k = \frac{1}{n_k}\sum_{x_i \in C_k} x_i$,
las matrices de dispersión **within** y **between** son:

$$ S_W = \sum_{k=1}^{K} \sum_{x_i \in C_k} (x_i - \mu_k)(x_i - \mu_k)^\top $$

$$ S_B = \sum_{k=1}^{K} n_k (\mu_k - \bar{x})(\mu_k - \bar{x})^\top $$

con $\bar{x}$ la media global. Se cumple la descomposición
$S_T = S_W + S_B$: la dispersión total es fija y clusterizar reparte cuánto
de ella cae en $S_W$ (no explicado por los grupos) frente a $S_B$ (entre
grupos). Un buen clustering minimiza $S_W$ — o equivalentemente maximiza
$S_B$.

La suma de cuadrados intra-cluster (WCSS) es la versión escalar de $S_W$:

$$ \mathrm{WCSS} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2 =
   \mathrm{tr}(S_W) $$

Todo algoritmo de clustering particiona el espacio en regiones determinadas
por su criterio; entender qué geometría implícita impone cada criterio es lo
que permite elegir el algoritmo correcto para cada forma de datos.

{% if cluster_model == 'todos' or cluster_model == 'KMeans' %}
## KMeans

**Objetivo.** Minimizar WCSS sobre todas las particiones de $n$ puntos en
$K$ grupos. El problema es NP-duro en general; Lloyd es una heurística que
converge a un óptimo local.

**Algoritmo de Lloyd.**
1. Inicializar $K$ centroides (k-means++ o aleatorio).
2. **Asignación**: cada $x_i$ al centroide más cercano (partición de
   Voronoi): $C_k = \{ x_i : k = \arg\min_j \|x_i - \mu_j\|_2^2 \}$.
3. **Actualización**: $\mu_k \leftarrow \frac{1}{|C_k|}\sum_{x_i \in C_k} x_i$.
4. Repetir 2–3 hasta que la asignación no cambie (o el movimiento sea
   menor que una tolerancia).

**Convergencia.** WCSS decrece monótonamente en cada paso (asignación y
actualización solo pueden mejorarlo), así que el algoritmo converge a un
punto fijo. Ese punto fijo es un **óptimo local**, no necesariamente el
global: inicializaciones distintas dan particiones distintas.

**Semilla k-means++** (Arthur & Vassilvitskii, 2007):
1. Elegir $c_1$ uniformemente al azar entre los datos.
2. Para cada $x$, calcular $D(x)$ = distancia al cuadrado al centroide
   elegido más cercano.
3. Elegir el siguiente centroide $c_j$ con probabilidad
   $\frac{D(x)^2}{\sum_x D(x)^2}$.
4. Repetir 2–3 hasta tener $K$ centroides.

La probabilidad es proporcional a $D(x)^2$ (no uniforme): reparte las semillas
sobre las regiones más dispersas y evita inicializaciones malísimas que
amontonan todos los centroides en un punto. Garantía esperada
$$ E[\mathrm{WCSS}] \le 8\,(\ln k + 2)\,\mathrm{WCSS}_{OPT} $$
es decir, una aproximación $O(\log k)$ del óptimo en esperanza.

**Sensibilidad a escala y outliers.** WCSS es cuadrático en la distancia:
una feature con rango grande domina la métrica euclídea, por lo que hay que
estandarizar (media 0, varianza 1) o usar escalado robusto antes de
clusterizar. Un outlier lejano arrastra su centroide por el término
cuadrático; recortar datos o usar centroides robustos ayuda.

**Elegir $k$.**
- **Codo (elbow)**: WCSS($k$) frente a $k$; el "codo" es $k^*$. Subjetivo.
- **Silhouette**: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$, con
  $a(i)$ la distancia media intra-cluster de $i$ y $b(i)$ la media a su
  cluster vecino más cercano. Media de $s$ en $[-1, 1]$; máx. = $k^*$.
- **Gap statistic**: compara $\log\mathrm{WCSS}(k)$ con el valor esperado
  bajo una distribución nula de referencia; se elige el menor $k$ con
  $\mathrm{gap}(k) \ge \mathrm{gap}(k+1) - s_{k+1}$.
- **BIC/GMM**: ajustar una mezcla de gaussianas con penalización por
  parámetros (ver sección GMM).

**Modos de fallo.** KMeans impone células de Voronoi convexas con fronteras
planas. Falla con:

| Situación | Consecuencia |
|---|---|
| Clusters no esféricos (anillos, lunas) | Corta la estructura curva en mitades |
| Tamaños muy dispares | El cluster grande se parte para alimentar al pequeño |
| Densidades muy distintas | Puntos de zona densa robados por centroides de zona rala |
| Outliers | Centroides desplazados por el error cuadrático |
| Alta dimensión | Las distancias se concentran; WCSS pierde contraste |

Seleccionar el $k$ y la semilla sin validación externa convierte el resultado
en un artefacto de la inicialización.
{% endif %}

{% if cluster_model == 'todos' or cluster_model == 'AgglomerativeClustering' %}
## Clustering aglomerativo

**Idea.** Empieza con $n$ clusters singulares y fusiona iterativamente el par
con menor criterio de enlace (linkage) hasta quedarse con $k$ clusters (o
cortar el dendrograma a cierta altura).

**Funciones de enlace** (distancia entre clusters $A$, $B$):

| Enlace | Definición | Propiedad |
|---|---|---|
| Single | $\min_{x\in A, y\in B}\|x - y\|$ | Encadenamiento; formas alargadas; ruido puentea clusters |
| Complete | $\max_{x\in A, y\in B}\|x - y\|$ | Clusters compactos; sensible a outliers dentro de un cluster |
| Average (UPGMA) | $\frac{1}{\|A\|\|B\|}\sum_{x,y}\|x-y\|$ | Compromiso entre ambos |
| Ward | $\frac{\|A\|\|B\|}{\|A\|+\|B\|}\|\mu_A - \mu_B\|^2$ | Minimiza el incremento de WCSS |

**Ward exacto (mínima varianza).** La fórmula de Ward es exactamente el
incremento de la suma de cuadrados intra-cluster al fusionar $A$ y $B$:
$\Delta(A,B) = \frac{\|A\|\|B\|}{\|A\|+\|B\|}\|\mu_A - \mu_B\|^2$. Solo es
válida con distancias euclídeas al cuadrado; con otra métrica el resultado
deja de minimizar varianza. Ward tiende a producir clusters esféricos y de
tamaños parecidos.

**Dendrograma.** El árbol de fusiones muestra la jerarquía completa: cortar a
altura $h$ da $k$ clusters, y el orden del eje revela qué grupos son
subgrupos de qué. Útil cuando la estructura anidada es el objetivo
(taxonomías, detección de $k$ a posteriori).

**Complejidad.** $O(n^2)$ en memoria (matriz de distancias) y entre
$O(n^2)$ y $O(n^3)$ en tiempo según la implementación (búsqueda del mínimo
con heaps / cadena de vecinos). No escala a millones de puntos; para
grandes volúmenes usar BIRCH o MiniBatch k-means.

**Cuándo el árbol importa.** Si la pregunta es "¿cuántos grupos hay y qué
subestructura tienen?", el dendrograma da respuesta jerárquica. Si solo se
quiere una partición plana de datos grandes, otros métodos son mejores.
{% endif %}

{% if cluster_model == 'todos' or cluster_model == 'DBSCAN' %}
## DBSCAN

**Parámetros.** $\varepsilon$ (radio) y `min_pts` (número mínimo de vecinos).

**Definiciones** (dado $\varepsilon$, `min_pts`):
- $\varepsilon$-vecindario: $N_\varepsilon(p) = \{ q : d(p,q) \le \varepsilon \}$.
- **Core**: $|N_\varepsilon(p)| \ge \text{min\_pts}$ (incluye a sí mismo).
- **Border**: a distancia $\le \varepsilon$ de un core pero sin ser core.
- **Ruido**: ni core ni border.
- **Directamente densidad-alcanzable**: $q \in N_\varepsilon(p)$ con $p$ core.
- **Densidad-alcanzable**: cadena de puntos directamente alcanzables.
- **Densidad-conectados**: existe un $o$ que alcanza a ambos.
- **Cluster**: conjunto máximo de puntos densidad-conectados entre sí.

**Algoritmo.** Para cada punto no visitado: si es core, expandir el cluster
recorriendo todos sus puntos densidad-alcanzables; si no, marcarlo como ruido
(puede acabar siendo border). Un cluster se define por **conectividad de
densidad**, no por convexidad.

**Por qué maneja formas arbitrarias y outliers.** No hay supuesto de forma
ni $k$ fijo: basta una cadena de densidad para unir estructuras alargadas o
anulares, y los puntos en regiones ralas quedan como ruido — no se les fuerza
a pertenecer a ningún cluster. Esto resuelve los dos fallos principales de
k-means.

**Sensibilidad a $\varepsilon$.** Es el parámetro crítico. Método estándar:
**k-distance plot** — ordenar la distancia de cada punto a su vecino
$k$-ésimo (con $k$ = min_pts) y buscar el codo de la curva; ese valor es un
$\varepsilon$ razonable. Valores distintos de $\varepsilon$/min_pts cambian
los clusters, y puntos cerca de la frontera son inestables. min_pts se suele
fijar $\ge d+1$ (dimensión + 1) y, en la práctica, 2·d como recomendación.

**Colapso de densidad en alta dimensión.** La regla del volumen concentra la
masa cerca de la superficie y las distancias al vecino más cercano se
concentran: con dimensión alta o $\varepsilon$ fijo pasa de "todo es ruido" a
"todo es un cluster" sin punto intermedio estable. Mitigación: reducir
dimensión antes (PCA/UMAP), usar min_pts adaptativo o métodos basados en
ratio de densidad.
{% endif %}

{% if cluster_model == 'todos' or cluster_model == 'GaussianMixture' %}
## Mezclas de gaussianas (GMM)

**Modelo.** Cada punto se genera de uno de $K$ componentes gaussianos:

$$ p(x) = \sum_{k=1}^{K} \pi_k \,\mathcal{N}(x \mid \mu_k, \Sigma_k), \qquad
   \pi_k \ge 0,\; \sum_{k=1}^{K} \pi_k = 1 $$

Los parámetros son $\theta = \{\pi_k, \mu_k, \Sigma_k\}$. La variable latente
$z_{ik} \in \{0,1\}$ indica qué componente generó $x_i$; la inferencia es el
problema de máxima verosimilitud sobre $p(X \mid \theta)$ con latentes
ausentes, resuelto con EM.

**Algoritmo EM.** Partir de $\theta$ inicial (habitualmente k-means o
k-means++ con $K$ grupos). Iterar:

1. **E-step** — responsabilidades (probabilidad a posteriori de que $x_i$
   venga del componente $k$):
   $$ \gamma_{ik} = \frac{\pi_k \,\mathcal{N}(x_i \mid \mu_k, \Sigma_k)}
        {\sum_{j=1}^{K} \pi_j \,\mathcal{N}(x_i \mid \mu_j, \Sigma_j)} $$
2. **M-step** — reestimación:
   $$ N_k = \sum_{i=1}^{n} \gamma_{ik}, \qquad
      \pi_k = \frac{N_k}{n} $$
   $$ \mu_k = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} x_i $$
   $$ \Sigma_k = \frac{1}{N_k} \sum_{i=1}^{n}
      \gamma_{ik}(x_i - \mu_k)(x_i - \mu_k)^\top $$

3. Repetir hasta que la log-verosimilitud $\log p(X \mid \theta)$ converja
   (cambio < tolerancia).

La verosimilitud es monótona no decreciente en cada iteración, pero EM
converge a un **óptimo local**: la inicialización importa (usar varias
semillas, elegir la mejor log-verosimilitud). La estructura de covarianza se
puede restringir: `full`, `tied` (compartida), `diag`, `spherical`.

**Asignaciones suaves.** $\gamma_{ik}$ son probabilidades, no etiquetas duras:
cada punto aporta fraccionalmente a varios componentes. La etiqueta dura se
obtiene con $\arg\max_k \gamma_{ik}$. A diferencia de k-means, GMM modela
clusters **elípticos**, con tamaño y orientación por $\Sigma_k$, y solapa
clusters que se tocan.

**Elegir $k$ (BIC/ICL).** Con $\nu$ parámetros libres y $N = n$:

$$ \mathrm{BIC} = \log L(\hat\theta) - \frac{\nu}{2}\log n,
   \qquad \mathrm{ICL} = \mathrm{BIC} +
   \sum_{i,k} \gamma_{ik}\log\gamma_{ik} $$

Mayor es mejor. ICL añade a BIC la entropía de las responsabilidades:
penaliza componentes que solapan (responsabilidades cercanas a $1/K$).
BIC/ICL es la forma **model-based** de elegir $k$, comparable pero no
equivalente al codo de k-means.

**Cuándo usar GMM en vez de k-means.** Cuando los clusters son elípticos, de
densidad o covarianza distinta, o se quieren probabilidades de pertenencia en
vez de etiquetas. Cuando los datos tienen colas pesadas, mezclas de $t$ de
Student son más robustas a outliers que las gaussianas.
{% endif %}

{% if cluster_model == 'todos' or cluster_model == 'SpectralClustering' %}
## Clustering espectral

**Matriz de similitud.** Partir de una matriz de afinidad $S$ con
$s_{ij} \ge 0$ y construir un grafo de similitud; $W$ es su matriz de
adyacencia ponderada. Construcciones habituales: grafo $\varepsilon$-vecino,
grafo k-NN (y k-NN mutuo) y grafo completamente conectado con kernel
gaussiano $s(x_i, x_j) = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$. El parámetro
$\sigma$ juega el papel de $\varepsilon$: controla el tamaño del vecindario.

**Laplacianos del grafo.** Con $D$ la matriz de grados (diagonal con
$d_i = \sum_j w_{ij}$):

- No normalizado: $L = D - W$.
- Normalizado simétrico: $L_{sym} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}$.
- Normalizado random-walk: $L_{rw} = D^{-1} L = I - D^{-1} W$.

Propiedades clave (von Luxburg):
$$ f^\top L f = \frac{1}{2} \sum_{i,j} w_{ij} (f_i - f_j)^2 \ge 0 $$
$L$ es simétrica y semidefinida positiva. El autovalor 0 tiene multiplicidad
igual al número de componentes conexas, y su autoespacio lo generan los
vectores indicador de cada componente. En el caso ideal de $k$ componentes
desconectadas, los $k$ primeros autovectores son (casi) los indicadores de
cluster.

**Algoritmo — espectral no normalizado** (von Luxburg, sección 4):
1. Construir el grafo de similitud y su matriz de adyacencia $W$.
2. Calcular el Laplaciano no normalizado $L = D - W$.
3. Calcular los $k$ primeros autovectores $u_1, \dots, u_k$ de $L$
   (los de menor autovalor).
4. Formar $U \in \mathbb{R}^{n \times k}$ con $u_1, \dots, u_k$ como columnas.
5. Para $i = 1, \dots, n$, sea $y_i \in \mathbb{R}^k$ la fila $i$-ésima de $U$.
6. Clusterizar los puntos $(y_i)_{i=1,\dots,n}$ en $\mathbb{R}^k$ con
   k-means en clusters $C_1, \dots, C_k$.
**Salida**: clusters $A_1, \dots, A_k$ con $A_i = \{j \mid y_j \in C_i\}$.

**Algoritmo — normalizado, Shi–Malik (2000):** igual que el anterior pero
en el paso 3 se calculan los $k$ primeros autovectores **generalizados** del
problema $Lu = \lambda D u$ (equivalentemente, autovectores de $L_{rw}$);
los pasos 4–6 son idénticos (matriz $U$, filas $y_i$, k-means).

**Algoritmo — normalizado, Ng–Jordan–Weiss (2002):**
1. Construir el grafo; obtener $W$.
2. Calcular $L_{sym}$.
3. Calcular los $k$ primeros autovectores $u_1, \dots, u_k$ de $L_{sym}$.
4. Formar $U \in \mathbb{R}^{n \times k}$ con esos autovectores como columnas.
5. Normalizar las filas de $U$ a norma 1: $t_{ij} =
   u_{ij} / \big(\sum_{l} u_{il}^2\big)^{1/2}$; formar $T$.
6. Clusterizar las filas $y_i$ de $T$ con k-means en $C_1, \dots, C_k$.

**Por qué funciona.** El espectro relaja el problema de corte de grafo
(`RatioCut` para $L$, `Ncut` para los normalizados): los $k$ autovectores
definen una nueva representación donde la estructura de clusters está
"mejorada" y k-means la detecta trivialmente.

**Heurística del eigengap.** Elegir $k$ de modo que $\lambda_1, \dots,
\lambda_k$ sean muy pequeños y $\lambda_{k+1}$ sea relativamente grande
(brecha $|\lambda_{k+1} - \lambda_k|$ grande). Justificación: en el caso
ideal de $k$ clusters completamente desconectados, el autovalor 0 tiene
multiplicidad $k$ y hay una brecha hasta $\lambda_{k+1} > 0$. Funciona bien
cuando los clusters están bien pronunciados; se degrada con ruido y solape.

**Pitfalls.** Sensible a $\sigma$ (ancho del kernel) y al número de vecinos
del grafo; el paso final de k-means sigue siendo de óptimo local; la matriz
de similitud cuesta $O(n^2)$ memoria y la descomposición espectral es
costosa para $n$ grande. La elección de $k$ y de los parámetros de
conectividad están acopladas y no se resuelven de forma trivial.
{% endif %}

{% if cluster_model == 'todos' or cluster_model == 'Birch' %}
## BIRCH

**Resumen CF (clustering feature).** Cada subcluster se resume con la terna
$$ CF = (N, \mathrm{LS}, \mathrm{SS}) $$
donde $N$ es el número de puntos, $\mathrm{LS} = \sum_i x_i$ (suma lineal) y
$\mathrm{SS} = \sum_i x_i^2$ (suma cuadrática). La terna es **aditiva**:
$CF(a) + CF(b) = (N_a+N_b, \mathrm{LS}_a+\mathrm{LS}_b, \mathrm{SS}_a+\mathrm{SS}_b)$,
y de ella se derivan centroide, radio y diámetro sin retener los puntos.

**CF-tree.** Árbol de altura equilibrada: los nodos internos tienen factor de
ramificación $B$; las hojas guardan entradas CF restringidas por un umbral
$T$ (máximo diámetro de subcluster). Inserción incremental: bajar por el
árbol, encontrar la entrada hoja más cercana, absorber el punto si el
subcluster sigue bajo $T$; si no, crear entrada nueva (y dividir/reconstruir
nodos para mantener el límite de memoria). Diámetro de $N$ puntos:

$$ D = \sqrt{\frac{2 \sum_{i<j} \|x_i - x_j\|^2}{N(N-1)}} $$

**Umbrales.** $T$ limita el tamaño de cada subcluster: $T$ grande → pocos
subclusters grandes y toscos; $T$ pequeño → muchos pequeños y finos. El
factor $B$ limita el ancho del árbol. Los subclusters resumidos luego se
pueden refinar con un aglomerativo o k-means sobre los CF de hoja (cada CF es
un punto con peso $N$) y reasignar puntos para limpiar fronteras.

**Cuándo usarlo.** Una sola pasada sobre los datos con memoria acotada y
$O(n)$ en tiempo: ideal para streaming y datos muy grandes que no caben en
memoria. **Limitaciones**: asume que cada cluster se puede resumir con un
centroide (forma tipo Voronoi); malo con formas arbitrarias o densidades muy
heterogéneas. No reemplaza la elección del $k$ final, que ocurre en la fase
de refinado.
{% endif %}

## Reducción de dimensionalidad antes de clusterizar

**Supuesto de variedad.** Los datos de alta dimensión suelen vivir cerca de
una variedad de dimensión intrínseca mucho menor; las coordenadas
irrelevantes solo añaden ruido a las distancias y aplanan el contraste entre
clusters. Reducir dimensión antes de un método basado en distancias suele
mejorar el resultado.

- **PCA**: proyección lineal global; conserva la varianza, que no siempre
  coincide con la estructura de clusters.
- **UMAP**: proyección no lineal local; conserva la estructura de vecindades
  (no las distancias globales). Las distancias en la proyección UMAP no son
  una métrica del espacio original: "clusters vistos en UMAP" describen el
  embedding, no el espacio original.

**Cuándo el preprocesado cambia la respuesta.** El clustering no es
invariante a transformaciones de los datos: estandarizar, log-transformar o
rotar (PCA) cambia la métrica y, con ella, la partición óptima. No existe un
"cluster verdadero" independiente de la representación; el preprocesado es
parte del modelo y debe reportarse como tal. Elegir UMAP+PCA o solo PCA
cambia la respuesta, no solo su calidad.

## Validación de clusters

**Métricas internas** (sin etiquetas):

| Métrica | Fórmula | Orientación |
|---|---|---|
| Silhouette | $\frac{b(i)-a(i)}{\max(a(i),b(i))}$, media sobre $i$ | Máx. en $[-1,1]$ |
| Davies–Bouldin | $\frac{1}{K}\sum_k \max_{l\ne k}\frac{\sigma_k + \sigma_l}{d(\mu_k,\mu_l)}$ | Mín. |
| Calinski–Harabasz | $\frac{\mathrm{tr}(S_B)/(K-1)}{\mathrm{tr}(S_W)/(n-K)}$ | Máx. |

con $a(i)$ = distancia media intra-cluster de $i$, $b(i)$ = media a su cluster
vecino, $\sigma_k$ = distancia media al centroide $k$.

**Límites de las internas.** Cada una codifica supuestos geométricos:
favorecen clusters convexos, compactos y bien separados, y ninguna tiene
escala absoluta ni se puede comparar entre datasets. Un valor alto no es
evidencia de estructura real — es evidencia de que la partición encaja con el
criterio que la propia métrica penaliza. No validan significado semántico.

**Métricas externas** (solo con etiquetas disponibles):
- **Adjusted Rand Index (ARI)**: concordancia de pares corregida por azar;
  $1$ = perfecto, $0$ = azar. La corrección por azar es imprescindible: el
  Rand crudo es alto con solo coincidir en tamaños.
- **NMI**: información mutua normalizada; usar la versión ajustada (AMI)
  cuando los tamaños de cluster son dispares.

En producción no supervisada no hay etiquetas: las externas solo sirven en
benchmarks o cuando se etiqueta una muestra de control.

**"Los clusters se ven bien" no es evidencia.** Cualquier proyección 2D
puede fabricar una apariencia clusterizada (incluso sobre ruido i.i.d.); la
inspección visual no es una métrica. Exigir: estabilidad ante semillas y
submuestras, coherencia con validación externa cuando exista, y contraste
contra la hipótesis nula (datos sin estructura).

## Detección de anomalías como tarea no supervisada

- **Densidad**: los puntos en regiones de baja densidad son anómalos —
  los que DBSCAN etiqueta ruido, o la distancia al vecino $k$-ésimo elevada.
- **Isolation Forest**: particiones aleatorias recursivas; los anómalos se
  aíslan pronto → longitud media de camino corta. Score:
  $s(x) = 2^{-E[h(x)]/c(n)}$ (cerca de 1 = anomalía). Sin métrica de
  distancia, escalable a alta dimensión, aunque no inmune a ella.
- **Reconstrucción (autoencoders)**: entrenar a reconstruir el dato normal;
  un punto anómalo tiene error $\|x - g(f(x))\|$ alto. Requiere un conjunto
  de entrenamiento dominado por normales y un umbral sobre el cuantil del
  error.

La frontera entre "outlier" y "punto raro de un cluster" es una decisión de
modelo, no un hecho de los datos.

## Trampas prácticas

1. **Estandarizar antes de métodos basados en distancia.** k-means,
   aglomerativo, DBSCAN y el kernel gaussiano del espectral son dominados por
   la feature de mayor rango. `StandardScaler` (o `RobustScaler` con
   outliers) antes de clusterizar.
2. **Resultados inestables.** Semillas de k-means/EM, orden de los datos en
   el aglomerativo (desempates) y parámetros del grafo espectral cambian la
   partición. Correr varias semillas y quedarse con la mejor objetiva, o
   promediar (clustering de consenso); reportar la variabilidad.
3. **Clusters ≠ grupos reales.** La partición depende de la métrica y el
   preprocesado; describe la geometría elegida, no descubre entidades.
   Interpretar con cautela y validar externamente cuando se pueda.

## Fuentes

- **A Tutorial on Spectral Clustering** — U. von Luxburg (2007).
  arXiv:0711.0189 — https://arxiv.org/abs/0711.0189
- **k-means++: The Advantages of Careful Seeding** — D. Arthur,
  S. Vassilvitskii (2007). arXiv:math/0612022 —
  https://arxiv.org/abs/math/0612022
- **UMAP: Uniform Manifold Approximation and Projection for Dimension
  Reduction** — L. McInnes, J. Healy, J. Melville (2018).
  arXiv:1802.03426 — https://arxiv.org/abs/1802.03426
- **Isolation Forest** — F. T. Liu, K. M. Ting, Z.-H. Zhou (2012).
  arXiv:1811.10941 — https://arxiv.org/abs/1811.10941
- **BIRCH: An Efficient Data Clustering Method for Very Large Databases** —
  T. Zhang, R. Ramakrishnan, M. Livny (1996). Sin arXiv —
  https://doi.org/10.1145/233269.233324
- **A Density-Based Algorithm for Discovering Clusters in Large Spatial
  Databases with Noise** — M. Ester, H.-P. Kriegel, J. Sander, X. Xu (1996).
  Sin arXiv — https://cdn.aaai.org/KDD/1996/KDD96-037.pdf
- **Maximum Likelihood from Incomplete Data via the EM Algorithm** —
  A. P. Dempster, N. M. Laird, D. B. Rubin (1977). Sin arXiv —
  https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
- **Silhouettes: A Graphical Aid to the Interpretation and Validation of
  Cluster Analysis** — P. J. Rousseeuw (1987). Sin arXiv —
  https://doi.org/10.1016/0377-0427(87)90125-7
{% endif %}
