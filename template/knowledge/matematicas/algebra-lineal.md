# Álgebra lineal: fundamentos para ciencia de datos

Referencia profunda para el agente `lider`: fórmulas exactas, propiedades
operativas y abusos numéricos frecuentes. Todo se conecta a cuándo aplicarlo
(y cuándo no) en un proyecto real de DS. Las secciones son autocontenidas.

## Espacios vectoriales, bases y dimensión

Un espacio vectorial $V$ es un conjunto cerrado bajo suma y producto por
escalar (cuerpo $\mathbb{R}$ o $\mathbb{C}$). Definiciones operativas:

- **Combinación lineal**: $v = c_1 v_1 + \dots + c_k v_k$.
- **Independencia lineal**: $c_1 v_1 + \dots + c_k v_k = 0 \Rightarrow$
  $c_i = 0$ para todo $i$. Ningún vector es combinación de los otros.
- **Span**: $span\{v_1,\dots,v_k\} = \{\sum_i c_i v_i\}$. Subespacio más
  pequeño que contiene a los vectores.
- **Base**: conjunto linealmente independiente que genera todo $V$. Toda base
  de $V$ tiene el mismo cardinal → **dimensión** $dim(V)$.
- **Rango-nulidad**: para $A \in \mathbb{R}^{m \times n}$,
  $rank(A) + nullity(A) = n$. La dimensión de la imagen más la del núcleo
  suman la del dominio.

**Teorema de la base incompleta**: todo conjunto independiente se extiende a
una base; es lo que permite "rellenar" matrices de autovectores con vectores
ortonormales arbitrarios cuando el rango es deficitario (técnica usada en la
construcción de la SVD y en el tutorial de Shlens).

Aplicación práctica: $dim(V)$ es el número mínimo de coordenadas que describen
sin pérdida un punto de $V$. En DS, estimar la **dimensión intrínseca** de los
datos (rango efectivo de la matriz de features, número de autovalores
relevantes de la covarianza) es el objeto de PCA y de la hipótesis de la
variedad — ver `matrices-app.md`.

## Normas vectoriales

Norma: función $\|\cdot\|: V \to \mathbb{R}_{\geq 0}$ con homogeneidad
$\|cv\| = |c|\|v\|$, desigualdad triangular y $\|v\| = 0 \Leftrightarrow v = 0$.

$$ \|x\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}, \quad
   p \in [1, \infty], \quad \|x\|_\infty = \max_i |x_i| $$

| Norma | Definición | Geometría | Uso en DS |
|---|---|---|---|
| $L_1$ | $\sum_i \vert x_i\vert$ | Rombo, esquinas en ejes | Ceros exactos (LASSO) |
| $L_2$ | $\sqrt{\sum_i x_i^2}$ | Esfera isotrópica | Euclídea; MSE |
| $L_\infty$ | $\max_i \vert x_i\vert$ | Cubo | Error máximo por coordenada |
| $L_p$, $p\in(1,\infty)$ | $\big(\sum_i \vert x_i\vert^p\big)^{1/p}$ | Intermedia | Puente $L_1$–$L_2$ |

La $L_1$ proyecta a la mediana y es robusta a outliers en la dirección de los
ejes (la $L_2$, al cuadrado, los amplifica). Para regularización, $L_1$ anula
coordenadas (vértices del rombo); $L_2$ encoge sin anular.

Cadenas útiles de orden (con $n$ dimensiones):

$$ \|x\|_\infty \leq \|x\|_2 \leq \|x\|_1 \leq \sqrt{n}\,\|x\|_2 \leq n\,\|x\|_\infty $$

La equivalencia de normas (existen $c, C > 0$ con $c\|x\|_a \leq \|x\|_b \leq
C\|x\|_a$) dice que la **topología** es la misma, pero las **constantes**
dependen de $n$: la brecha $\sqrt{n}$ entre $L_1$ y $L_2$ crece con la
dimensión y alimenta la maldición de la dimensionalidad.

Abuso frecuente: usar $L_2$ para "medir error" cuando la señal de interés es
por coordenadas (feature selection), o normalizar por norma que no corresponde
a la geometría del problema. Para regularización, $L_1$ produce soluciones
esparcidas en los vértices de la bola; $L_2$ encoge pero no anula.

## Producto interno, ortogonalidad y bases ortonormales

Producto interno en $\mathbb{R}^n$: $\langle x, y \rangle = x^\top y =
\sum_i x_i y_i$. La norma inducida es $\|x\|_2 = \sqrt{\langle x, x \rangle}$.

**Cauchy–Schwarz**: $|\langle x, y \rangle| \leq \|x\|\|y\|$, con igualdad si
y solo si $x$ e $y$ son colineales. Da la interpretación geométrica:
$\cos\theta = \langle x, y \rangle / (\|x\|\|y\|)$, que es la **correlación**
entre vectores centrados — la base del ángulo entre features.

- **Ortogonalidad**: $\langle x, y \rangle = 0$. Los vectores ortogonales son
  linealmente independientes (si todos son no nulos).
- **Base ortonormal**: $\langle e_i, e_j \rangle = \delta_{ij}$. En ella las
  coordenadas son $x_i = \langle x, e_i \rangle$: proyectar es sacar
  productos internos.
- **Proyección ortogonal** de $y$ sobre $span\{u\}$:
  $proj_u(y) = \dfrac{\langle y, u \rangle}{\langle u, u \rangle}\, u$.
- **Matriz de proyección** sobre $span\{U\}$ (columnas ortonormales):
  $P = U U^\top$. Propiedades: $P^2 = P$, $P^\top = P$; proyecta y no distorsiona
  lo que ya está en el subespacio.

Toda matriz ortogonal ($Q^\top Q = I$) es una **isometría**: $\|Qx\| = \|x\|$
para todo $x$. Por eso los pasos de Householder y las bases ortonormales son
numéricamente estables: no amplifican errores de redondeo. Perder
ortonormalidad (drift) equivale a introducir distorsión — ver sección de
trampas.

## Matriz de Gram

Dados vectores $x_1, \dots, x_n \in \mathbb{R}^m$ como columnas de
$X \in \mathbb{R}^{m \times n}$:

$$ G_{ij} = \langle x_i, x_j \rangle = x_i^\top x_j, \qquad G = X^\top X $$

- $G$ es **simétrica y semidefinida positiva** (PSD): $u^\top G u =
  \|Xu\|_2^2 \geq 0$.
- $rank(G) = rank(X)$. $G$ es definida positiva (SPD) si y solo si las columnas
  son linealmente independientes.
- $\det(G) > 0$ es el **cuadrado del volumen** del paralelepípedo generado por
  las columnas; $\det(G) = \prod_i \sigma_i^2$ con $\sigma_i$ los valores
  singulares de $X$.
- $G$ es la matriz de **similitud lineal** (o de kernel lineal) entre las
  observaciones: entrada de los métodos kernel y del cálculo de PCA en el
  espacio dual (kernel PCA sustituye $X X^\top$ por un kernel general $K$).

Aplicación: cuando $n < m$ (pocas muestras, muchas features), trabajar con
$G = X^\top X \in \mathbb{R}^{n \times n}$ es más barato que con $X X^\top \in
\mathbb{R}^{m \times m}$, y sus autovalores coinciden salvo ceros. Abuso:
formar $X^\top X$ en float32 o con datos sin centrar amplifica el
condicionamiento — ver trampas.

## Normas matriciales

| Norma | Definición | Interpretación |
|---|---|---|
| Frobenius | $\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\mathrm{tr}(A^\top A)}$ | Norma euclídea |
| Espectral ($L_2$) | $\|A\|_2 = \sigma_{max}(A) = \sqrt{\lambda_{max}(A^\top A)}$ | Mayor amplificación |
| $L_1$ (columna) | $\max_j \sum_i \vert a_{ij}\vert$ | Suma máxima por columnas |
| $L_\infty$ (fila) | $\max_i \sum_j \vert a_{ij}\vert$ | Suma máxima por filas |
| Nuclear | $\|A\|_* = \sum_i \sigma_i$ | Relajación convexa del rango |

Vale $\|A\|_F = \sqrt{\sum_i \sigma_i^2}$ y $\|Ax\|_2 \leq \|A\|_F\|x\|_2$
(consistencia). La $\|A\|_2$ es la mayor amplificación de cualquier vector
unitario: define la constante de Lipschitz de la capa lineal.

La norma espectral controla la amplificación: $\|A\|_2$ es el factor máximo
con el que $A$ estira cualquier vector unitario. En ML esto es la **constante
de Lipschitz** de una capa lineal — ver regularización espectral en
`matrices-app.md`.

## Rango, determinante y traza

- **Rango**: $rank(A)$ = dimensión del espacio columna (= espacio fila). Vale
  $rank(A) \leq \min(m, n)$; $rank(AB) \leq \min(rank\,A, rank\,B)$;
  desigualdad de Sylvester $rank(A) + rank(B) - n \leq rank(AB)$ con
  $A, B$ de tamaño $n$. El rango es el número de valores singulares no nulos.
- **Determinante**: $\det(A)$ es el **factor (con signo) de escalado de
  volúmenes** que aplica la transformación: el volumen del paralelepípedo
  imagen de la base canónica es $|\det(A)|$. Por eso $\det(A) = 0 \Leftrightarrow$
  $A$ singular $\Leftrightarrow$ rango deficiente $\Leftrightarrow$ núcleo no
  trivial. Propiedades: $\det(AB) = \det(A)\det(B)$,
  $\det(A^\top) = \det(A)$, $\det(cA) = c^n \det(A)$,
  $\det(A^{-1}) = 1/\det(A)$.
- **Traza**: $\mathrm{tr}(A) = \sum_i a_{ii}$. Ciclicidad:
  $\mathrm{tr}(AB) = \mathrm{tr}(BA)$ (clave para reordenar expresiones con
  Frobenius, p.ej. $\mathrm{tr}(A^\top B) = \langle A, B \rangle_F$).

Ambas conectan con el espectro (ver abajo): $\mathrm{tr}(A) = \sum_i \lambda_i$
y $\det(A) = \prod_i \lambda_i$. En DS: $\det$ aparece en la verosimilitud
gaussiana multivariante (factor de normalización $\propto \det(\Sigma)^{-1/2}$)
y la traza en la varianza total de la muestra (suma de autovalores de la
covarianza).

## Autovalores y autovectores

$$ A v = \lambda v, \quad v \neq 0 $$

- Autovalores de una matriz real **simétrica**: todos reales; autovectores de
  autovalores distintos son ortogonales; siempre diagonalizable
  ortogonalmente.
- Autovalores de $A^\top A$ y $AA^\top$: no negativos.
- **Suma = traza**: $\sum_i \lambda_i = \mathrm{tr}(A)$ (con multiplicidad
  algebraica). **Producto = determinante**: $\prod_i \lambda_i = \det(A)$.
- **Radio espectral**: $\rho(A) = \max_i |\lambda_i|$; gobierna la convergencia
  de la iteración de potencias y de métodos iterativos.
- Autovalores de $\alpha A + \beta I$: $\alpha\lambda_i + \beta$ (shift
  invariante), lo que permite estabilizar y desplazar espectros.

**Diagonalización**: $A = P D P^{-1}$ con $D$ diagonal si existe base de
autovectores (suma de multiplicidades geométricas = $n$). Para $A$ simétrica,
$P$ se toma ortogonal: $A = P D P^\top$.

**Teorema espectral** ($A$ simétrica real): existe base ortonormal de
autovectores y

$$ A = \sum_{i=1}^{n} \lambda_i u_i u_i^\top $$

suma de proyecciones de rango 1 ortogonales entre sí. Consecuencias: la
aplicación de $A$ sobre $x$ es $\sum_i \lambda_i (u_i^\top x) u_i$; permite el
cálculo funcional $f(A) = \sum_i f(\lambda_i) u_i u_i^\top$ (exponencial de
matrices, kernel de difusión, raíces de matrices para normalización de
similitudes). También da el límite del **cociente de Rayleigh**
$\rho_A(x) = x^\top A x / \|x\|^2 \in [\lambda_{min}, \lambda_{max}]$,
herramienta de la iteración de potencias y de la estimación de autovalores.

## Semidefinición positiva (PSD) y matrices simétricas

$A \in \mathbb{R}^{n \times n}$ simétrica es **SPD** si $x^\top A x > 0$ para
todo $x \neq 0$; **PSD** si $\geq 0$. Caracterizaciones equivalentes:

| Propiedad | SPD | PSD |
|---|---|---|
| Autovalores | todos $> 0$ | todos $\geq 0$ |
| Factorización | $\exists B$ invertible: $A = B^\top B$ | $\exists B$: $A = B^\top B$ |
| Cholesky | $A = L L^\top$, existe y es único | falla (usar LDL o eigen) |
| Diagonal dominante | $a_{ii}>\sum_{j\ne i}\vert a_{ij}\vert$ | $a_{ii}\ge\sum_{j\ne i}\vert a_{ij}\vert$ |
| Esquinas | $\det$(submatriz principal) $> 0$ | $\det$(submatriz principal) $\geq 0$ |

La matriz de covarianza muestral y toda matriz de Gram son PSD (de hecho SPD
si no hay colinealidad exacta). En la práctica el ruido de redondeo puede dar
autovalores negativos minúsculos: para Cholesky o muestreo gaussiano hay que
corregir (clip de autovalores) o usar descomposiciones que toleren el cero.

**Simétrica $\neq$ SPD**: la Hessiana de una pérdida no convexa es simétrica
pero indefinida (autovalores mixtos). Confundir ambas rompe Cholesky y hace
creer que un punto estacionario es mínimo — ver trampas.

## Por qué los autovalores se calculan de forma iterativa

El camino "ingenuo" es resolver el **polinomio característico**
$\det(A - \lambda I) = 0$. No se hace, por dos razones:

1. **No hay fórmula cerrada.** Por el teorema de Abel–Ruffini, para grado
   $\geq 5$ (matrices $\geq 5\times5$) no existen fórmulas por radicales; el
   problema de autovalores es inseparable en general.
2. **Inestabilidad numérica.** Los coeficientes del polinomio característico
   se obtienen por operaciones que pierden dígitos (el polinomio de Wilkinson
   $p(x) = \prod_{i=1}^{20}(x-i)$ tiene autovalores bien separados, pero
   perturbaciones de $10^{-14}$ en los coeficientes mueven raíces a $\pm i$
   imaginarios). Encontrar raíces de un polinomio con coeficientes mal
   condicionados amplifica el error; el polinomio no codifica los autovalores
   de forma estable.

Por eso los códigos reales (LAPACK `dgeev`, `numpy.linalg.eig`) usan **QR
con shifts** ($O(n^3)$, convergencia cúbica), y para matrices grandes o
dispersas **Lanczos/Arnoldi** (espacios de Krylov), que solo requieren
productos matriz-vector. En todos los casos el motor es la **iteración de
potencias**, no el polinomio característico.

## Iteración de potencias

Algoritmo iterativo para el par dominante $(\lambda_1, v_1)$ con
$|\lambda_1| > |\lambda_2|$:

```python
def power_iteration(A, x0, tol=1e-10, max_iter=1000):
    x = x0 / np.linalg.norm(x0)
    for _ in range(max_iter):
        y = A @ x
        lam = x @ y                 # cociente de Rayleigh
        x_new = y / np.linalg.norm(y)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, lam
```

- Convergencia del cociente de Rayleigh: $\propto |\lambda_2/\lambda_1|^{2k}$
  (el error del autovalor decae al cuadrado del error del autovector).
- Si $|\lambda_1| \approx |\lambda_2|$, la convergencia es lenta: el cociente
  manda. Con shifts $\sigma$ se trabaja con $A - \sigma I$ para alejar
  autovalores y acelerar.
- Solo da un par por ejecución: para todo el espectro se combina con
  deflación (proyectar fuera de $v_1$), o se usa directamente el QR-shifted.
- La variante **aleatorizada** (ver `matrices-app.md` y Halko 2009) repite el
  esquema con una matriz de prueba gaussiana para capturar todo un subespacio
  dominante de golpe.

## Descomposición en valores singulares (SVD)

Toda $A \in \mathbb{R}^{m \times n}$ admite

$$ A = U \Sigma V^\top $$

con $U \in \mathbb{R}^{m\times m}$ ortogonal, $V \in \mathbb{R}^{n\times n}$
ortogonal y $\Sigma \in \mathbb{R}^{m\times n}$ diagonal con
$\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_r > 0 = \sigma_{r+1} = \dots$,
$r = rank(A)$. Relación con el espectro:

$$ A^\top A = V \Sigma^\top \Sigma V^\top, \qquad
   AA^\top = U \Sigma \Sigma^\top U^\top $$

- Los **vectores singulares derechos** $V$ son los autovectores de $A^\top A$;
  los **izquierdos** $U$ los de $AA^\top$. Los **valores singulares** son
  $\sigma_i = \sqrt{\lambda_i(A^\top A)}$.
- Los valores singulares son **únicos** y siempre reales no negativos (no
  dependen de simetría), lo que hace a la SVD la descomposición más robusta.
- Si $A$ es simétrica, la SVD coincide con el espectro: $U = V$ y
  $\sigma_i = |\lambda_i|$.

**Formas**:

| Forma | Tamaños | Contenido |
|---|---|---|
| Completa | $m\times m$, $m\times n$, $n\times n$ | Bases completas |
| Fina (economy) | $m\times k$, $k\times k$, $n\times k$ | Pares singulares no nulos |
| Truncada (rango $k$) | $m\times k$, $k\times k$, $n\times k$ | Los $k$ mayores; compresión, PCA |

Tamaños (de $U$, $\Sigma$, $V$): $k = \min(m,n)$ en la forma fina (la que
devuelve `np.linalg.svd(A, full_matrices=False)`), $k < \min(m,n)$ en la
truncada.

**Interpretación geométrica**: la SVD descompone $A$ en rotación ($V^\top$),
escalado por ejes ($\Sigma$) y rotación ($U$). Las columnas de $U$ son la base
ortonormal del espacio columna; $V$ del espacio fila. El subespacio capturado
por las primeras $k$ columnas es el que mejor aproxima la acción de $A$.

## Teorema de Eckart–Young (mejor aproximación de rango $k$)

Sea $A_k = U_k \Sigma_k V_k^\top$ la SVD truncada a los $k$ mayores valores
singulares. Eckart–Young:

$$ \|A - A_k\|_2 = \sigma_{k+1}, \qquad
   \|A - A_k\|_F^2 = \sum_{i > k} \sigma_i^2 $$

y $A_k$ es el **mínimo global** de $\min_{rank(B) \leq k} \|A - B\|$ en la
norma espectral y en la Frobenius (de hecho en toda norma unitariamente
invariante). La cola de valores singulares $\sum_{i>k}\sigma_i^2$ mide
exactamente la varianza no capturada por el rango $k$ → fundamento de PCA,
compresión, denoising y factorización de matrices. La **velocidad de decaimiento**
de $\sigma_i$ decide qué $k$ basta: decaimiento rápido → pocos componentes
explican casi todo.

Aplicación: umbral de rango numérico $\sigma_{k+1} \leq tol \cdot \sigma_1$
define el "rango efectivo" de una matriz casi singular.

## Pseudoinversa de Moore–Penrose y mínimos cuadrados

La pseudoinversa $A^+$ existe siempre y es única; es la única matriz que
satisface las cuatro condiciones de Moore–Penrose:

$$ AA^+A = A, \quad A^+AA^+ = A^+, \quad (AA^+)^\top = AA^+, \quad
   (A^+A)^\top = A^+A $$

Vía SVD: $A^+ = V \Sigma^+ U^\top$, donde $\Sigma^+$ tiene
$1/\sigma_i$ en la diagonal para $\sigma_i > 0$ y $0$ en los ceros.

**Mínimos cuadrados**: la solución de norma mínima de
$\min_x \|Ax - b\|_2^2$ es

$$ x^* = A^+ b = \sum_{i: \sigma_i > 0} \frac{u_i^\top b}{\sigma_i} v_i $$

- Si $A$ tiene columnas linealmente independientes (rango columna completo):
  $A^+ = (A^\top A)^{-1} A^\top$ (mínimos cuadrados ordinarios).
- Si $A$ es de rango deficiente o rectangular: la fórmula SVD sigue dando la
  solución de norma mínima; la de las ecuaciones normales ni siquiera existe.
- Costo: dominado por la SVD, $O(mn^2)$ con $m \geq n$.
- Es exactamente lo que calcula `numpy.linalg.lstsq` (LAPACK `gelsd`, SVD
  dividida) y `scipy.linalg.pinv`.

Abuso frecuente: resolver mínimos cuadrados con las ecuaciones normales
$(A^\top A)^{-1}A^\top b$ — cuadra el condicionamiento (ver abajo) y pierde
precisión cuando las columnas están correlacionadas. Usar siempre SVD o QR.

## Condicionamiento y estabilidad numérica

**Número de condición** (norma 2):

$$ \kappa_2(A) = \frac{\sigma_{max}}{\sigma_{min}} = \|A\|_2 \|A^{-1}\|_2 $$

Para $A$ simétrica definida positiva, $\kappa_2(A) = \lambda_{max}/\lambda_{min}$.
Interpretación: cota de amplificación del error relativo al resolver
$Ax = b$,

$$ \frac{\|\Delta x\|}{\|x\|} \lesssim \kappa_2(A) \left(
   \frac{\|\Delta b\|}{\|b\|} + \frac{\|\Delta A\|}{\|A\|} \right) $$

$\kappa \approx 1$ → bien condicionado; $\kappa \gg 1$ → **mal condicionado**:
errores del orden de $\kappa \cdot \varepsilon_{maq}$ en la solución. $\kappa$
"grande" depende del problema: para $10^6$ ya es grave en regresión; para
$10^3$ puede ser aceptable en iterativos.

**Reglas que salen de aquí**:

- **Nunca computes $X^{-1}$** para resolver sistemas ni para regresión: es
  $3\times$ más caro que resolver directamente y, sobre todo,
  $\kappa(X^{-1}) = \kappa(X)$, mientras que las ecuaciones normales
  $\kappa(X^\top X) = \kappa(X)^2$ **cuadran** el condicionamiento.
- Usa `np.linalg.solve` (LU con pivoteo), o para regresión
  `np.linalg.lstsq` / `np.linalg.pinv` (SVD), o QR.
- El problema de **mínimos cuadrados** con $X^\top X$: formar el Gram es la
  fuente habitual de inestabilidad con features correlacionadas.
- `pinv` trunca valores singulares por debajo de una tolerancia relativa:
  da solución estable y de norma mínima en presencia de casi
  colinealidad; el ajuste con `lstsq` idéntico por debajo.

Detectar mal condicionamiento: mirar $\sigma_{min}$ relativo a $\sigma_1$, o
`np.linalg.cond`, no `det` (ver trampas).

## Cholesky

Si $A$ es simétrica definida positiva, existe única factorización

$$ A = L L^\top = U^\top U $$

con $L$ triangular inferior de diagonal positiva. Costo $n^3/3$ — la mitad que
LU — y **sin pivoteo** (la definición positiva garantiza estabilidad).

Cuándo usarla:
- **Resolver sistemas PSD**: $Ax = b$ → dos sustituciones triangulares
  ($O(n^2)$ tras factorizar). Es el método más rápido y estable para
  ecuaciones normales bien condicionadas y para núcleos (GPs).
- **Muestreo gaussiano**: si $z \sim \mathcal{N}(0, I)$ y $\Sigma = L L^\top$,
  entonces $x = \mu + L z \sim \mathcal{N}(\mu, \Sigma)$. Es el estándar para
  simular desde una multivariante sin factorizar $n \times n$ denso.

```python
import numpy as np
def gauss_sample(mu, Sigma, n=1):
    L = np.linalg.cholesky(Sigma)   # Sigma debe ser SPD
    z = np.random.randn(len(mu), n)
    return mu[:, None] + L @ z
```

- Si $A$ es PSD con ceros (rango deficiente), Cholesky falla; usar la
  factorización de **LDL** o el espectro con clipping de autovalores.
- Sirve como **test barato de SPD**: `np.linalg.cholesky` lanza
  `LinAlgError` si la matriz no es definida positiva.

## QR

$$ A = Q R, \qquad Q^\top Q = I, \quad R \text{ triangular superior} $$

Costo $O(2mn^2 - \frac{2}{3}n^3)$ (Householder, el estándar de
`numpy.linalg.qr`). El Gram–Schmidt clásico es inestable (pierde
ortonormalidad); los códigos usan reflectores de Householder o Givens.

Cuándo usarla:
- **Mínimos cuadrados estables**: $\min\|Ax-b\|_2$ con $A$ de rango columna
  completo → resolver $R x = Q^\top b$ por sustitución hacia atrás. Evita
  formar $A^\top A$ y cuadrar el condicionamiento. Es la elección cuando
  $m \gg n$ y no se teme rango deficiente (ahí, SVD).
- **Ortonormalización** en cualquier paso que requiera base ortonormal:
  es el paso 3 del randomized range finder de Halko (QR de $Y = A\Omega$) y
  del re-ortonormalizado entre potencias (Algorithm 4.4).
- **QR reveladora de rango** (RRQR, con pivoteo de columnas): estima el rango
  numérico y da bases de subespacios estables sin pagar una SVD completa.
- `numpy.linalg.svd` la usa internamente para matrices altas (Golub–Kahan
  arranca con una QR).

| Descomposición | Requisitos | Costo | Uso principal |
|---|---|---|---|
| LU | cuadrada no singular | $2n^3/3$ | resolver sistemas ($\texttt{solve}$) |
| Cholesky | SPD | $n^3/3$ | sistemas PSD, muestreo gaussiano |
| QR | rectangular/tall | $2mn^2$ | LS estable, ortonormalización |
| SVD | cualquiera | $O(mn^2)$ | rango, PCA, pseudoinversa, LS rango-deficiente |
| Eig (simétrica) | simétrica | $O(n^3)$ | espectro, teorema espectral |

## Matrices de covarianza y correlación; Hessiana

**Covarianza muestral** de $X \in \mathbb{R}^{n \times d}$ (filas = muestras)
centrada $\tilde{X} = X - \bar{x}$:

$$ C = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(x_i - \bar{x})^\top
     = \frac{1}{n-1} \tilde{X}^\top \tilde{X} $$

- Simétrica y PSD (SPD si $n > d$ y sin colinealidad). Diagonal = varianzas;
  fuera de la diagonal = covarianzas (descorrelación lineal).
- Autovectores = direcciones principales; autovalores = varianza a lo largo de
  cada dirección → es el objeto que diagonaliza PCA.
- La vía SVD evita formar el Gram: los valores singulares de $\tilde{X}$
  cumplen $\sigma_i^2 = (n-1)\lambda_i(C)$, y $V$ de la SVD son los
  autovectores. Ver `matrices-app.md`.

**Matriz de correlación**:

$$ R_{ij} = \frac{C_{ij}}{\sqrt{C_{ii} C_{jj}}} $$

- Adimensional, en $[-1, 1]$, PSD; invariante a reescalado por columna
  (unidades distintas, features de órdenes de magnitud dispares).
- $\cos\theta$ entre features centradas y normalizadas. Correlación cero
  $\Rightarrow$ incorrelación lineal, **no** independencia.

**Hessiana** de $f:\mathbb{R}^d \to \mathbb{R}$: $H_{ij} = \partial^2 f /
\partial x_i \partial x_j$, simétrica si $f$ es $C^2$ (teorema de Clairaut).
- En óptimo local de pérdida convexa, $H \succeq 0$; en sillas es indefinida.
- Los autovalores de $H$ son las curvaturas; $\kappa(H) = \lambda_{max} /
  \lambda_{min}$ controla la convergencia del descenso de gradiente (ver
  condicionamiento en `matrices-app.md`). $\lambda_{min} \approx 0$
  = dirección casi plana → estancamiento.
- El paso de Newton usa $H^{-1}$; en grandes modelos se reemplaza por
  aproximaciones (diagonal, Fisher, cuasi-Newton) porque $H$ es
  $d \times d$ intratable.

## Trampas numéricas y abusos comunes

1. **Cancelación catastrófica**. Restar dos números casi iguales elimina
   dígitos significativos: $(1 + \varepsilon) - 1 = \varepsilon$ con pérdida
   total de precisión relativa. Ejemplos en DS: varianza por
   $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ (usar la fórmula de dos pasadas o
   Welford); $\log(1+x)$ con $x$ pequeño (usar `np.log1p`); restar la media a
   datos de magnitud grande (centrar en float64). Regla: reorganizar para no
   restar cantidades casi iguales.

2. **Abuso de `np.linalg.det`**. $\det(A) = \prod \lambda_i$ se subdesborda a
   $0$ (o desborda a $\infty$) incluso para matrices perfectamente no
   singulares, y no tiene umbral interpretable: $\det \neq 0$ en coma
   flotante no implica bien condicionado. Para decidir singularidad/rank usar
   `np.linalg.cond` o el menor valor singular relativo a $\sigma_1$, no `det`.

3. **Deriva no ortogonal**. Tras muchas multiplicaciones o iteraciones
   numéricas, las columnas "ortonormales" dejan de serlo (errores de redondeo
   acumulados) y cualquier estimación basada en la base se distorsiona.
   Re-ortonormalizar con QR (o Gram–Schmidt modificado) entre pasos — es la
   razón del Algorithm 4.4 de Halko frente al 4.3. El Gram–Schmidt clásico
   sin re-ortonormalización falla igual.

4. **Simétrica ≠ SPD**. La Hessiana de pérdidas no convexas, las matrices de
   adyacencia simétricas o los Gram mal construidos pueden ser indefinidos.
   Cholesky requiere SPD; aplicarlo a una simétrica indefinida lanza error o
   produce basura. Verificar autovalores o usar LDL/defecto de la diagonal.

5. **Formar $X^\top X$ (o $A^\top A$) sin necesidad**. Cuadra el
   condicionamiento y pierde precisión con features correlacionadas. Preferir
   SVD/QR sobre la matriz original. Es la versión numérica de "no calcules la
   inversa".

6. **Truncar la SVD sin mirar la cola**. Guardar $k$ componentes sin revisar
   que $\sigma_{k+1}$ sea realmente despreciable frente a $\sigma_1$: la
   varianza descartada es $\sum_{i>k}\sigma_i^2$ (Eckart–Young).

7. **Mezclar convenciones de centrado/transpuesta**. La SVD de
   `X` con filas = muestras da $V$ (autovectores de $X^\top X$) como
   componentes; con filas = variables la respuesta es $U$. Los errores de
   signo y de orientación de componentes son el clásico bug silencioso de
   PCA — ver trampas en `matrices-app.md`.

## Fuentes

- **A Tutorial on Principal Component Analysis** — J. Shlens (2014).
  arXiv:1404.1100 — https://arxiv.org/abs/1404.1100
- **Finding Structure with Randomness: Probabilistic Algorithms for
  Constructing Approximate Matrix Decompositions** — N. Halko,
  P.-G. Martinsson, J. A. Tropp (2011).
  arXiv:0909.4061 — https://arxiv.org/abs/0909.4061
- **The Matrix Cookbook** — K. B. Petersen, M. S. Pedersen (2012).
  Sin arXiv — https://matrixcookbook.com
