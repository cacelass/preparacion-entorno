# Álgebra lineal aplicada al aprendizaje automático

La contraparte operativa de `algebra-lineal.md`: qué descomposiciones se usan,
en qué problema exacto, y dónde se rompen. El `lider` debe consultar este
documento antes de recomendar PCA, factorización o regularización.

## PCA vía SVD

PCA busca la proyección ortogonal que maximiza la varianza, o
equivalentemente el subespacio de dimensión $k$ que minimiza el error de
reconstrucción. Ambas se resuelven con la SVD de la matriz centrada.

Sea $X \in \mathbb{R}^{n \times d}$ (filas = muestras, columnas = features),
$\tilde{X} = X - \bar{x}$ (columna centrada). Entonces, con
$\tilde{X} = U \Sigma V^\top$:

$$ \tilde{X}^\top \tilde{X} = V \Sigma^\top \Sigma V^\top
   \quad\Longrightarrow\quad \lambda_i = \frac{\sigma_i^2}{n-1},
   \quad PC_i = v_i $$

- Los **componentes principales** son las columnas de $V$ (autovectores de la
  covarianza muestral).
- Los **scores** (proyecciones) son $Z = \tilde{X} V_k$, de tamaño
  $n \times k$, con columnas incorreladas y
  $Var(Z_j) = \sigma_j^2/(n-1) = \lambda_j$.
- Vía SVD se evita formar $\tilde{X}^\top\tilde{X}$ (cuadra el
  condicionamiento y cuesta $O(nd^2)$ solo para el Gram). La vía por
  autovalores de la covarianza solo conviene cuando $d$ es pequeño.

Algoritmo exacto (convención de Shlens: restar la media, luego SVD):

```python
def pca(X, k=None, var_ratio=None):
    Xc = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)   # s: d valores
    ev = s**2 / (X.shape[0] - 1)                        # autovalores cov
    ratio = ev / ev.sum()                               # varianza explicada
    if k is None and var_ratio is not None:
        k = int(np.searchsorted(ratio.cumsum(), var_ratio) + 1)
    Vk = Vt[:k].T                                       # componentes (dxk)
    return Xc @ Vk, Vk, ev[:k], ratio                   # scores, loadings, EV
```

`sklearn.decomposition.PCA` hace exactamente esto (SVD de LAPACK, solver
`'auto'`) y además estandariza opcionalmente; sus `components_` son filas de
$V_k$ y aplica `svd_flip` para signos deterministas. El código de Shlens
normaliza por $1/\sqrt{n-1}$ en lugar de dividir por $n-1$: mismo espectro,
las escalas de $U$ y $\Sigma$ se reparten distinto.

## Varianza explicada, scree y elección de $k$

- **Varianza explicada por el componente $j$**: $\sigma_j^2 / \sum_i \sigma_i^2$.
  La acumulada por los $k$ primeros es la fracción de variabilidad total
  (que es $\mathrm{tr}(C) = \sum_i \lambda_i$) capturada por el subespacio.
- **Scree plot**: gráfico de $\lambda_j$ (o $\sigma_j$) en orden decreciente;
  la "rodilla" (codo) marca el punto donde la curva se aplana. El codo es un
  criterio visual, no un test: usarlo como número exacto es subjetivo.
- **Elección de $k$**:
  - umbral de varianza explicada acumulada (p.ej. 0.90–0.95);
  - criterio de Kaiser: retener $\lambda_j > 1$ (si la matriz es de
    correlación, autovalor medio = 1);
  - paralelo (comparar con autovalores de datos aleatorios permutados);
  - validación cruzada: elegir $k$ que minimiza el error de reconstrucción
    en datos de validación.
- La cola $\sum_{i>k}\sigma_i^2$ es exactamente el error de reconstrucción al
  cuadrado (Eckart–Young) → el scree es la curva de ese error.

**Cuándo NO**: PCA es una proyección **lineal**. En datos no gaussianos o
acostados sobre una variedad no lineal, el primer componente de mayor varianza
no captura la estructura de interés (Shlens, sección VI). Antes de reducir
con PCA, comprobar que la señal es aproximadamente lineal; si no, kernels o
embeddings no lineales.

## SVD para factorización de matrices y recomendación

Modelo de rango bajo: aproximar la matriz de interacciones
$R \in \mathbb{R}^{n_u \times n_i}$ (usuarios $\times$ ítems) por

$$ R \approx U_k \Sigma_k V_k^\top = P Q^\top, \quad
   P \in \mathbb{R}^{n_u \times k}, \; Q \in \mathbb{R}^{n_i \times k} $$

con $k$ la dimensión latente (capacidad del modelo). SVD truncada resuelve la
versión con todas las entradas observadas; en recomendación **no todas lo
están**, y entonces la SVD clásica no aplica.

| Aspecto | Feedback explícito | Feedback implícito |
|---|---|---|
| Dato | ratings/votos (escala ordinal) | conteos/clics/vistas (binarizables) |
| No observado | **missing**: no dice nada | **señal**: ausencia $\approx$ no interés |
| Pérdida | $\sum_{obs}(r_{ui}-p_u^\top q_i)^2+reg$ | $\sum_{u,i} c_{ui}(p_{ui}-p_u^\top q_i)^2$ |
| Optimización | SGD, ALS | ALS ponderado (Hu–Koren–Volinsky) |
| El rango $k$ | pocas decenas; la CV lo fija | puede ser mayor (señal densa) |

Con missing a discreción, el problema no es una SVD de la matriz completa sino
una **minimización sobre las entradas observadas** (bajo supuestos de
missing-at-random que en la práctica se violan: usuarios que puntúan son los
que ya consumen). El **ALS** alterna regresiones ridge por usuario y por ítem,
cada una $O(k^2)$ por fila; el ALS ponderado (implícito) reemplaza cada fila
por una solución en forma cerrada con pesos.

**Rango bajo como modelo**: $k \ll \min(n_u, n_i)$ codifica la hipótesis de
que pocos factores latentes explican las preferencias. Elegir $k$ grande
sobreajusta las entradas observadas y generaliza mal; $k$ se valida por error
en un holdout de interacciones (y con implícito, por ranking, no por MSE).

## SVD aleatorio para matrices grandes

Para matrices grandes y densas (o cuando el producto matriz-vector es barato),
la SVD completa es prohibitiva. El **range finder aleatorizado** de Halko et
al. captura el subespacio dominante con alto oversampling:

```python
def rsvd(A, k, q=1, p=5):
    Omega = np.random.randn(A.shape[1], k + p)   # test gaussiano, p≈5-10
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)                       # base ortonormal del rango
    for _ in range(q):                           # potencia: aplanar la cola
        Q, _ = np.linalg.qr(A.T @ Q)             # re-ortonormalizar SIEMPRE
        Q, _ = np.linalg.qr(A @ Q)
    B = Q.T @ A                                  # problema pequeño
    Uhat, s, Vt = np.linalg.svd(B, full_matrices=False)
    return Q @ Uhat[:, :k], s[:k], Vt[:k]
```

- **Error**: con oversampling $p = 5$–$10$,
  $\mathbb{E}\|A - QQ^\top A\| \lesssim \left(1 + \sqrt{\frac{k+p}{p-1}}\cdot
  \sqrt{\min(m,n)}\right) \sigma_{k+1}$, a un factor polinómico pequeño del
  óptimo de Eckart–Young. La probabilidad de fallar decae
  superexponencialmente con $p$.
- **Power iteration** ($q$ pasos): se aplica $B = (AA^\top)^q A$; los valores
  singulares de $B$ son $\sigma_j(B) = \sigma_j(A)^{2q+1}$, la cola decae
  mucho más rápido y $k$ muestras alcanzan aunque $\sigma$ decaiga lento.
  Cada paso cuesta $2q+1$ productos matriz-vector. Regla heurística: la
  brecha al óptimo se cierra como $C^{1/(2q+1)}$.
- **Fallo con $q > 0$ sin re-ortonormalizar**: el redondeo apaga la
  información de los modos con $\sigma_j \lesssim \mu^{1/(2q+1)}\|A\|$; hay que
  ortonormalizar entre cada aplicación de $A$ y $A^\top$ (Algorithm 4.4).
- **Muestreo de filas/columnas** (alternativa streaming): muestrear $\ell$
  columnas con probabilidad proporcional a sus normas al cuadrado o a los
  **leverage scores** (importancia de cada columna en los $k$ primeros
  vectores singulares derechos), y hacer la SVD truncada del submatriz.
  Garantía: $\|A - B\|_F \leq \|A - A_k\|_F + \varepsilon\|A\|_F$ con
  $\ell = \ell(k,\varepsilon)$, en una pasada. Elegir columnas óptimas es
  NP-duro; el muestreo aleatorio llega cerca con alta probabilidad.

Cuándo usar cada uno: Gaussian-sketch (rsvd) para densas donde matvec es
rápido en BLAS; muestreo de columnas cuando el acceso a la matriz es por
columnas o en streaming. El sketch gaussiano es el más robusto al espectro.

## Geometría de los datos en alta dimensión

- **Concentración de la medida**: en $\mathbb{R}^d$ con $d$ grande, la masa de
  una gaussiana se concentra en la cáscara de una esfera de radio
  $\sigma\sqrt{d}$, no en el centro. La mayoría de las direcciones son casi
  ortogonales entre sí: con datos i.i.d., $\langle x, y\rangle/\|x\|\|y\|$
  se concentra cerca de 0. Las distancias se vuelven indistinguibles.
- **Maldición de la dimensionalidad**: para puntos uniformes en $[0,1]^d$, el
  cociente entre la distancia máxima y la mínima a un punto tiende a 1 cuando
  $d \to \infty$: la **contraste relativo desaparece** y "vecino más cercano"
  deja de significar algo. k-NN, kernels con soporte fijo y la estimación de
  densidades degradan exponencialmente con $d$.
- Densidad: $n$ muestras uniformes en $\mathbb{R}^d$ dejan distancia media
  $\sim n^{-1/d}$ entre vecinos; llenar la bola unitaria requiere
  $n \sim \varepsilon^{-d}$ — inviable. Solo la estructura de rango bajo o la
  escasez salvan el problema.
- **Escasez**: con $d$ features, la mayoría de celdas del espacio están vacías;
  los datos son "dispersos" aunque cada feature individual no lo sea. La
  regularización $L_1$ explota que la señal vive en un subconjunto pequeño de
  features.
- Implicación práctica: reducir dimensión **antes** de métodos basados en
  distancias (k-NN, clustering, kernel) no es opcional cuando $d$ es grande.
  Y toda reducción asume que la estructura real es de baja dimensión — ver
  hipótesis de la variedad.

## Hipótesis de la variedad y datos dispersos

**Hipótesis de la variedad**: los datos reales en alta dimensión se concentran
cerca de una variedad (o unión de variedades) de **dimensión intrínseca**
$k \ll d$. Es el supuesto que justifica PCA, embeddings, autoencoders,
t-SNE/UMAP y el éxito de los adaptadores de rango bajo.

- La dimensión intrínseca se estima con la **cola del espectro** (autovalores
  de la covarianza: donde la cola se aplana, ahí termina la señal) o con
  métodos de vecinos (correlación de paquetes).
- Si los datos viven en una variedad **no lineal** (curva, esfera), PCA la
  "aplana" y mezcla puntos que la métrica intrínseca separa; ahí PCA subestima
  la estructura. La métrica de los datos no es la euclídea del espacio
  ambiente.
- Los datos **dispersos** (texto, grafo, one-hot, selección de genes) se
  describen bien con álgebra lineal estructurada: sparse matrices + métodos
  de Krylov en vez de SVD densa.

## Rango bajo en ML moderno: embeddings y matrices de pesos

- **Embeddings como matrices**: una capa de embedding es una tabla de
  búsqueda $E \in \mathbb{R}^{V \times d_e}$ (filas = tokens); cada fila es un
  vector entrenado. Históricamente, word2vec equivale a factorizar (de forma
  implícita) la matriz de PMI/coocurrencia — el vínculo original entre
  representaciones distribucionales y SVD.
- **Factorización de pesos**: descomponer una capa densa
  $W \in \mathbb{R}^{d_{out} \times d_{in}}$ como $W \approx A B$ con
  $A \in \mathbb{R}^{d_{out} \times r}$, $B \in \mathbb{R}^{r \times d_{in}}$,
  $r \ll \min(d_{out}, d_{in})$. Reduce parámetros de $d_{out}d_{in}$ a
  $r(d_{out}+d_{in})$ y es la base de los adaptadores de rango bajo y de la
  compresión.
- **Los pesos de redes grandes tienen rango intrínseco bajo**: el entrenamiento
  de sobreparametrizaciones se mueve en un subespacio de dimensión efectiva
  pequeña (Li et al., Aghajanyan et al.), lo que explica por qué actualizar
  solo un adaptador de rango bajo basta para especializar un modelo.

{% if ml_type == 'redes_neuronales' %}
## Matrices en redes neuronales: adaptadores de rango bajo (LoRA)

Al afinar un modelo pre-entrenado con pesos congelados $W_0$, LoRA aprende la
actualización como producto de rango bajo:

$$ W = W_0 + \Delta W = W_0 + B A, \qquad
   B \in \mathbb{R}^{d_{out} \times r}, \; A \in \mathbb{R}^{r \times d_{in}},
   \; r \ll \min(d_{out}, d_{in}) $$

- **Inicialización**: $B = 0$ (empieza en $W_0$, sin perturbar el
  pre-entrenado) y $A$ gaussiana. En inferencia se puede fusionar
  $W = W_0 + BA$ (sin coste extra) o mantener $BA$ aparte para servir varias
  tareas.
- **Memoria**: entrenar $r(d_{out}+d_{in})$ parámetros en vez de
  $d_{out}d_{in}$; no se necesita gradiente ni estado de optimizador para
  $W_0$. Con $r \in \{4, 8, 16, 64\}$ se recupera casi todo el rendimiento del
  fine-tune completo.
- **Por qué funciona**: la sobredimensión del pre-entrenado hace que la
  actualización efectiva viva en un subespacio de rango bajo; $\Delta W$
  captura la señal sin tocar el resto.
- **Cómo NO usarlo**: fijar $r$ demasiado grande anula la compresión; entrenar
  $BA$ con inicialización $A = B = 0$ no aprende (gradientes cero para $B$ si
  $A$ es cero solo si se usa la escala $\alpha/r$ correcta — la asignación de
  la escala es parte del método).
- **Regularización espectral en capas**: $\|W\|_2 = \sigma_{max}(W)$ es la
  constante de Lipschitz de la capa; normalizar el espectro estabiliza
  entrenamiento (normalización espectral, Miyato) y acota la sensibilidad a
  perturbaciones del input.
{% endif %}

## Normas y regularización espectral

- **$L_1$ por elemento** ($\sum |W_{ij}|$): promueve ceros exactos → selección
  de features (LASSO); es la norma que "encoge" a la frontera del poliedro.
- **$L_2$ (Frobenius) por elemento** ($\|W\|_F^2 = \sum_{ij} W_{ij}^2 =
  \sum_i \sigma_i^2$): encoge todos los valores singulares uniformemente;
  equivale a ridge en la capa y a un prior gaussiano sobre los pesos.
- **Norma espectral** $\|W\|_2 = \sigma_{max}(W)$: penaliza la amplificación
  máxima → controla la **Lipschitz** de la capa (GAN: normalización
  espectral; robustez adversarial). No castiga los modos pequeños.
- **Norma nuclear** $\|W\|_* = \sum_i \sigma_i$: relajación convexa del rango;
  minimizarla promueve matrices de rango bajo (matrix completion,
  recomendación). Su paso proximal es la **umbralización blanda** de los
  valores singulares: soft-threshold $\sigma_i \to \max(\sigma_i - \tau, 0)$
  vía SVD. Es el análogo matricial del soft-threshold de $L_1$ sobre vectores.

Elegir la norma según la estructura que quieras imponer: rango (nuclear),
magnitud de las salidas (espectral), dispersión de entradas (L1), energía
total (Frobenius).

## Condicionamiento y velocidad de optimización

En un mínimo cuadrático con Hessiana $H$ (simétrica SPD), el descenso de
gradiente con paso $\eta$ converge si $\eta < 2/\lambda_{max}(H)$, y la
velocidad peor caso está controlada por
$\kappa(H) = \lambda_{max}/\lambda_{min}$:

- Con $\kappa$ grande, los modos lentos ($\lambda_{min}$) apenas avanzan por
  paso y los rápidos oscilan: el error decae como
  $\left(\frac{\kappa-1}{\kappa+1}\right)^t$. Mal condicionamiento =
  cientos/miles de pasos extra o divergencia.
- En regresión, $\kappa$ de la matriz de diseño gobierna lo mismo: features
  correlacionadas → $\sigma_{min}$ pequeño → coeficientes inestables (varianza
  enorme) y descenso lento.
- Mitigaciones: **estandarizar** features (mejora las escalas, no corrige
  colinealidad), precondicionamiento (Jacobi, estimadores de curvatura),
  métodos de segundo orden (Newton/cuasi-Newton) que invierten el espectro, y
  optimizadores con **momentum/Adam** que compensan per-coordenada las
  curvaturas dispares.
- El número de condición de la matriz de features también aparece en el
  **error de generalización** de mínimos cuadrados: para el mismo ruido,
  $\kappa$ grande infla la varianza de los coeficientes
  ($\propto \sigma^2 (X^\top X)^{-1}$).

## Eficiencia: BLAS, disposición en memoria y matrices dispersas

- **Vectorizar**: un bucle Python paga interpretación y despacho por elemento;
  una llamada a `X @ Y` paga una sola vez y ejecuta una rutina BLAS nivel 3
  (GEMM) optimizada en caché y multithread. Para tamaños útiles, GEMM corre a
  una fracción alta del pico FLOPS; el bucle, a una fracción minúscula. El
  cuello de botella no son los FLOPS sino la memoria (bandwidth): las
  operaciones nivel 3 reutilizan bloques en caché y son las que valen la pena
  organizar (batch matmul, no matvec sueltos).
- **BLAS por niveles**: nivel 1 (axpy, dot) y 2 (matvec) son memory-bound; el
  nivel 3 (matmul) es compute-bound. Estructurar el cómputo para que haga
  GEMM grandes (p.ej. proyectar $X V_k$ de golpe) en vez de multiplicaciones
  por vector.
- **Disposición en memoria**: numpy es row-major (C), Fortran es column-major
  (F). `X.T` de una matriz C-contigua es F-contigua: pasar `X.T` a BLAS puede
  forzar una copia o recorridos no secuenciales. Iterar a lo largo del último
  eje es contiguo en C; en F, a lo largo del primero. Usar
  `np.ascontiguousarray`/`np.asfortranarray` explícitamente cuando se mide.
- **Matrices dispersas**: si $nnz \ll m\cdot n$ (texto TF-IDF, grafos,
  one-hot), guardar solo los no nulos (CSR/CSC) reduce memoria y el matvec a
  $O(nnz)$. `scipy.sparse`; nunca materializar a denso. Los métodos de Krylov
  y los matvec baratos son el caso natural de la SVD aleatoria.
- **dtype**: `float32` reduce memoria a la mitad y dobla el rendimiento en
  CPUs modernas (AVX) y GPUs; pero pierde ~7 dígitos decimales. La SVD/PCA en
  float32 puede apagar valores singulares pequeños (centrar en float64
  primero); el acumulado de productos en float32 introduce drift. Usar
  float64 para la factorización y float32 solo en el cómputo de paso si se
  acepta la precisión (mixed precision).

## Trampas prácticas

1. **Estandarizar antes de PCA/SVD**. PCA sobre datos sin estandarizar
   maximiza varianza **en las unidades originales**: una feature con unidades
   grandes domina los componentes. Estandarizar (z-score) hace que el análisis
   trabaje sobre la **matriz de correlación** en vez de la de covarianza.
   Decidir explícitamente: si las features comparten unidades y escala, la
   covarianza es defendible; si no, estandarizar. Estandarizar **cambia la
   geometría** — los loadings resultantes no son los de los datos crudos.
2. **Ambigüedad de signo de la SVD/eig**. Si $Av = \lambda v$, también
   $A(-v) = \lambda(-v)$: los signos de componentes, scores y loadings son
   arbitrarios y pueden **voltearse entre ejecuciones o máquinas**. No
   interpretar el signo de un loading; comparar por magnitud, y usar
   `sklearn.utils.extmath.svd_flip` (que fija un signo por columna) para
   reproducibilidad.
3. **Correlación ≠ causalidad**. Una matriz de correlación describe
   co-movimiento lineal, es simétrica y no tiene dirección; la causalidad es
   asimétrica y depende de intervenciones. Extraer "causas" de un heatmap de
   correlaciones (o de los loadings de PCA) es un error de inferencia. Véase
   el documento de inferencia causal del corpus.
4. **Efectos de dtype y cancelación**. Centrar $x_i - \bar{x}$ cuando
   $\bar{x}$ es enorme en float32 destruye los dígitos bajos; centrar en
   float64. Formar $X^\top X$ en float32 duplica los problemas del Gram.
   Calcular varianza con $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ cancela;
   usar la fórmula de dos pasadas.
5. **PCA vía eigen de la covarianza cuando debería ser SVD de $X$**. Formar
   $X^\top X$ cuesta $O(nd^2)$ y cuadra el condicionamiento; la SVD de
   $\tilde{X}$ es estable y da lo mismo por la relación
   $\sigma_i^2 = (n-1)\lambda_i$.
6. **Tratar missing en SVD/PCA como si fuera una matriz completa**. PCA
   clásico no soporta entradas ausentes; imputar la media distorsiona el
   espectro. Para missing real, usar PCA EM o factorización probabilística.

## Fuentes

- **A Tutorial on Principal Component Analysis** — J. Shlens (2014).
  arXiv:1404.1100 — https://arxiv.org/abs/1404.1100
- **Finding Structure with Randomness: Probabilistic Algorithms for
  Constructing Approximate Matrix Decompositions** — N. Halko,
  P.-G. Martinsson, J. A. Tropp (2011).
  arXiv:0909.4061 — https://arxiv.org/abs/0909.4061
{% if ml_type == 'redes_neuronales' %}
- **LoRA: Low-Rank Adaptation of Large Language Models** — E. Hu et al.
  (2021). arXiv:2106.09685 — https://arxiv.org/abs/2106.09685
{% endif %}
