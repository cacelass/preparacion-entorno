{% if ml_type == 'supervisado' or ml_type == 'hibrido' %}
# Aprendizaje supervisado: familias de modelos

Referencia densa de las familias que compiten en un pipeline tabular:
lineales, árboles/ensembles, máquinas de vector soporte y k-NN. Cada sección
da el núcleo matemático, cuándo conviene, cómo se malusa y qué hiperparámetros
mandan. La elección de modelo no se justifica por preferencia: se decide por
qué supuesto cumple (o deja de cumplir) el dato.

## Descomposición de sesgo-varianza

Sea el proceso generador $y = f(x) + \varepsilon$ con $E[\varepsilon] = 0$ y
$\mathrm{Var}(\varepsilon) = \sigma^2$. El riesgo cuadrático de un predictor
$\hat{f}$ (ajustado sobre una muestra de entrenamiento, evaluado en un $x$
fijo) se descompone en tres términos:

$$ E[(y - \hat{f}(x))^2] = \sigma^2 + \mathrm{bias}^2 +
   \mathrm{Var}(\hat{f}(x)) $$

**Derivación.**
1. Sustituir $y = f + \varepsilon$ y expandir
   $E[(f - \hat{f} + \varepsilon)^2]$. El término cruzado se anula porque
   $\varepsilon$ tiene media 0 y es independiente de $\hat{f}$:
   $$ E[(y - \hat{f})^2] = E[(f - \hat{f})^2] + \sigma^2 $$
2. Reescribir $f - \hat{f} = (f - E[\hat{f}]) - (\hat{f} - E[\hat{f}])$.
   Al elevar al cuadrado, el término cruzado se anula (la media de
   $\hat{f} - E[\hat{f}]$ es 0):
   $$ E[(f - \hat{f})^2] = \left(f(x) - E[\hat{f}(x)]\right)^2 +
      \mathrm{Var}(\hat{f}(x)) $$

Con $\mathrm{bias}^2 = (f(x) - E[\hat{f}(x)])^2$: el error esperado es el
ruido irreducible más el cuadrado del sesgo más la varianza.

**Lectura operativa.**

| Término | Qué captura | Palanca |
|---|---|---|
| $\sigma^2$ | Ruido intrínseco del proceso | Piso del error; no se puede bajar |
| $\mathrm{bias}^2$ | Lo no representable o distorsionado por el ajuste | Más capacidad, mejores features |
| $\mathrm{Var}(\hat{f})$ | Sensibilidad a la muestra | Regularización, más datos, ensembles |

- Subir capacidad baja el sesgo y sube la varianza: el óptimo está en el
  punto de cruce, no en el mínimo de error de entrenamiento.
- La regularización intercambia sesgo por varianza; el bagging reduce
  varianza a sesgo casi constante; el boosting reduce sesgo a costa de
  varianza si se sobre-itera.
- En modelos sobreparametrizados el error vuelve a caer tras el pico de
  varianza (*double descent*), pero en tabular la lectura clásica vale.

## Modelos lineales

### Regresión lineal: forma cerrada (OLS)

Minimizar $\mathrm{RSS}(\beta) = \|y - X\beta\|_2^2$ sobre $n$
observaciones y $p$ features da la forma cerrada:

$$ \hat{\beta} = (X^\top X)^{-1} X^\top y $$

Válida si $X$ tiene rango columna completo. Gauss-Márkov: entre los
estimadores lineales insesgados, OLS tiene varianza mínima; pero el sesgo
cero no minimiza el error de predicción. Contraer $\hat{\beta}$ hacia 0
(ridge) casi siempre reduce el MSE esperado fuera de muestra.

### Ridge: trayectoria de penalización

Minimizar $\mathrm{RSS} + \lambda\|\beta\|_2^2$, con solución cerrada
$\hat{\beta} = (X^\top X + \lambda I)^{-1} X^\top y$. Al recorrer
$\lambda$: con $\lambda = 0$ es OLS y con $\lambda \to \infty$,
$\hat\beta \to 0$. No hace selección (no anula coeficientes exactamente)
pero comparte información entre predictores correlacionados y estabiliza la
inversa. Invariante a rotaciones; útil con features correlacionadas y poca
señal por columna.

### Lasso: descenso por coordenadas y sparseza

Minimizar $\mathrm{RSS} + \lambda\|\beta\|_1$. La norma $L_1$ es no
diferenciable en 0, así que el óptimo anula coeficientes exactamente
(*sparsity*): con $\lambda$ grande entran pocas variables y al bajarlo se
añaden de una en una (la trayectoria es piecewise linear). Se resuelve con
descenso por coordenadas, actualizando un $\beta_j$ a la vez con su
soft-threshold operator:

$$ \beta_j \leftarrow \mathrm{S}_{t}\left(\frac{1}{n}x_j^\top r +
   \beta_j\right), \qquad \mathrm{S}_t(u) = \mathrm{sign}(u)\,(|u|-t)_+ $$

Con features correlacionadas el lasso elige de forma arbitraria; el elastic
net (mezcla de $L_1$ y $L_2$) agrupa correlacionadas y estabiliza la
selección.

### Regresión logística: IRLS

Modela $p = P(y=1 \mid x) = \sigma(x^\top\beta)$ con la logística
$\sigma(z) = 1/(1+e^{-z})$. La log-verosimilitud es la cross-entropy y la
función es convexa (sin forma cerrada). Se optimiza con IRLS (*iteratively
reweighted least squares*): Newton sobre la log-verosimilitud equivale a un
WLS con pesos $w_i = p_i(1-p_i)$ y respuesta de trabajo
$z_i = x_i^\top\beta + (y_i - p_i)/(p_i(1-p_i))$:

$$ \beta \leftarrow (X^\top W X)^{-1} X^\top W z, \qquad
   W = \mathrm{diag}(p_i(1-p_i)) $$

Converge en pocas iteraciones. La regularización $L_2$ (en sklearn, el
inverso de `C`) es imprescindible cuando $p$ es alto. La frontera es lineal
y las probabilidades salen razonablemente calibradas.

### Cuándo el baseline lineal es lo correcto

- $n$ pequeño frente a $p$: los lineales necesitan menos datos que los
  árboles y generalizan más estable.
- Fronteras aproximadamente lineales o aditivas: un lineal con interacciones
  explícitas las captura sin coste extra de varianza.
- Se necesitan coeficientes interpretables (dirección y tamaño del efecto)
  o inferencia estadística sobre los parámetros.
- Como **baseline obligatorio**: ningún modelo complejo se justifica si no
  gana de forma robusta al lineal bien regularizado con las mismas features.

## Árboles de decisión y ensembles

### CART: Gini, entropía y MSE

Un árbol es una partición binaria recursiva: en cada nodo con $n_m$ muestras
se elige el split $(j, t)$ que más reduce la impureza ponderada:

$$ \Delta(j,t) = i(\mathrm{padre}) - \frac{n_L}{n_m}\,i(hijo_L) -
   \frac{n_R}{n_m}\,i(hijo_R) $$

con criterios de impureza por tarea:

| Criterio | Fórmula | Uso |
|---|---|---|
| Gini | $1 - \sum_k \hat{p}_{mk}^2$ | Clasificación (default sklearn) |
| Entropía | $-\sum_k \hat{p}_{mk}\log \hat{p}_{mk}$ | Clasificación |
| MSE | $\frac{1}{n_m}\sum_{i\in m}(y_i - \bar{y}_m)^2$ | Regresión (reducción de varianza) |

En una hoja la predicción es la moda (clasificación) o la media (regresión)
de sus muestras.

### Por qué los árboles profundos sobreajustan

Un árbol sin poda sobre $n$ puntos logra error de entrenamiento 0 (una hoja
por región). El sesgo baja con la profundidad pero la varianza sube: las
hojas se quedan con pocas muestras y su estadístico es ruidoso. El control
de complejidad son la profundidad, los mínimos por nodo (`min_samples_split`,
`min_samples_leaf`) y la poda por coste-complejidad
($\mathrm{RSS}_\alpha = \mathrm{RSS} + \alpha|T|$ con $|T|$ hojas). Un
solo árbol sin poda casi nunca es el modelo final: sirve de baseline y de
lector de estructuras.

{% if model_type == 'todos' or model_type == 'RandomForest' %}
### Random Forest: bagging y descorrelación

Cada árbol se entrena sobre un *bootstrap* (muestra con reemplazo, ~63% de
puntos distintos) y, en cada split, solo sobre un subconjunto aleatorio de
features (`max_features`). La predicción es el promedio (o el voto). El
bagging reduce varianza promediando estimadores decorrelacionados:

$$ \mathrm{Var}\left(\frac{1}{B}\sum_{b=1}^{B}\hat{f}_b\right) =
   \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2
   \;\xrightarrow[B\to\infty]{}\; \rho\,\sigma^2 $$

La varianza residual queda dominada por $\rho$, la correlación entre
árboles: el muestreo aleatorio de features baja $\rho$ sin subir el sesgo,
por eso importa más que el bagging solo. Los árboles se crecen profundos
(sesgo bajo) y se confía en el promedio. El error *out-of-bag* estima la
generalización sin validación extra. Hiperparámetros que mandan:
`n_estimators` (converge), `max_features` (controla descorrelación),
`max_depth`/`min_samples_leaf` (complejidad por árbol). No requiere escalar
features, captura no linealidades e interacciones, y es robusto a outliers
en las features (los splits solo usan el orden).
{% endif %}

{% if model_type == 'todos' or model_type == 'ExtraTrees' %}
### ExtraTrees (extremely randomized trees)

Igual que Random Forest, pero el umbral del split se **muestrea al azar**
en vez de barrer todos los cortes. Añade varianza por umbral aleatorio que
el promedio reduce: varianza menor que RF con sesgo algo mayor, y
entrenamiento mucho más rápido (no ordena las features). Bueno con muchas
features ruidosas y como ensemble barato; suele rendir a la par que RF con
una fracción del coste.
{% endif %}

{% if model_type == 'todos' or model_type == 'AdaBoost' %}
### AdaBoost: reweighting y pérdida exponencial

AdaBoost mantiene una distribución $D_t$ sobre las muestras; en cada ronda
ajusta un aprendiz débil (típicamente un stump) y re-pondera:

$$ \alpha_t = \frac{1}{2}\ln\frac{1 - \mathrm{err}_t}{\mathrm{err}_t},
   \qquad D_{t+1}(i) \propto D_t(i)\,\exp(-\alpha_t y_i h_t(x_i)) $$

Las muestras mal clasificadas ganan peso exponencialmente. Se demuestra que
esto minimiza la pérdida exponencial $E[e^{-y f(x)}]$ mediante *forward
stagewise additive modeling*: cada round añade la pieza que más reduce esa
pérdida, no la 0-1. Por eso es tan sensible a etiquetas ruidosas y outliers
(el peso de un punto difícil explota y domina). Mitigaciones: no usarlo con
mucho ruido, `n_estimators` con early stopping, aprendices débiles de
verdad.
{% endif %}

### Gradient Boosting: descenso de gradiente funcional

El modelo es aditivo: $F_M(x) = \sum_{m=1}^{M} \gamma_m h_m(x)$. Cada paso
ajusta $h_m$ al **gradiente negativo** de la pérdida evaluado en las
predicciones actuales, los *pseudo-residuos*:

$$ r_i^{(m)} = -\left.\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}
   \right|_{F = F_{m-1}} $$

Para error cuadrático $r_i = y_i - F_{m-1}(x_i)$: se regresiona sobre los
residuos; luego un paso de línea fija $\gamma_m$. La regularización clave
es el *shrinkage* $\eta$: $F_m = F_{m-1} + \eta\,\gamma_m h_m$ con
$\eta \in (0,1]$ — mismas iteraciones con $\eta$ menor generalizan mejor
a costa de más árboles. Con árboles débiles (profundidad baja) el boosting
encaja interacciones por pasos; `subsample` (gradient boosting estocástico)
y early stopping sobre validación controlan la sobre-iteración. Un GBDT
bien tuneado sigue siendo el estándar de oro en tabular.

{% if model_type == 'todos' or model_type == 'XGBoost' or use_xgboost %}
### XGBoost: objetivo exacto y ganancia de split

XGBoost regulariza el objetivo aditivo. El modelo es
$\hat{y}_i = \sum_k f_k(x_i)$ con $f_k \in \mathcal{F}$ árboles CART y el
objetivo (Chen & Guestrin, Eq. 2):

$$ \mathcal{L}(\phi) = \sum_i l(\hat{y}_i, y_i) + \sum_k \Omega(f_k),
   \qquad \Omega(f) = \gamma T + \tfrac{1}{2}\lambda\|w\|^2 $$

con $T$ el número de hojas, $w$ los pesos de hoja, $l$ una pérdida convexa
diferenciable y $\gamma$, $\lambda$ las regularizaciones. En el paso $t$
se añade $f_t$ minimizando la aproximación de Taylor de segundo orden
(Eq. 3):

$$ \tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^{n}\left[ g_i f_t(x_i) +
   \tfrac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t) $$

con $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ y
$h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ los
estadísticos de gradiente de primer y segundo orden. Con
$I_j = \{ i : q(x_i) = j \}$, $G_j = \sum_{i \in I_j} g_i$ y
$H_j = \sum_{i \in I_j} h_i$ (Eqs. 4-6):

$$ w_j^* = -\frac{G_j}{H_j + \lambda}, \qquad
   \tilde{\mathcal{L}}(q) = -\tfrac{1}{2}\sum_{j=1}^{T}
   \frac{G_j^2}{H_j + \lambda} + \gamma T $$

La estructura óptima del split evalúa la ganancia exacta de partir el
conjunto $I$ en $I_L \cup I_R$ (Eq. 7):

$$ L_{\mathrm{split}} = \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} +
   \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R +
   \lambda} \right] - \gamma $$

Se elige el split de máxima ganancia; si es negativa se poda. Tres
regularizadores prácticos del paper:

- **Shrinkage**: los pesos nuevos se escalan por $\eta$ (learning rate);
  cada árbol aporta poco y deja margen a los siguientes.
- **Column subsampling** (`colsample_bytree`): el paper reporta que reduce
  sobreajuste más que el subsampling de filas y acelera el paralelismo.
- **Sparsity-aware splits**: los ausentes van a una *dirección por defecto*
  aprendida, con coste lineal en los no ausentes. La librería añade además
  $\alpha\|w\|_1$ (L1 sobre pesos de hoja) al $\Omega$ del paper. Los
  splits aproximados usan el *weighted quantile sketch* (buckets pesados
  por $h_i$).
{% endif %}

{% if model_type == 'todos' or model_type == 'LightGBM' or use_lightgbm %}
### LightGBM: GOSS y EFB

LightGBM hace boosting basado en histogramas (binning) y dos optimizaciones
clave:

- **GOSS** (*Gradient-based One-Side Sampling*): retiene todos los ejemplos
  con gradiente alto y una muestra aleatoria de los de gradiente bajo
  (re-ponderados). Los de gradiente alto son los que más error aportan; los
  de gradiente bajo ya están bien explicados.
- **EFB** (*Exclusive Feature Bundling*): agrupa features que nunca toman
  valor no nulo a la vez (típico de one-hot) en una sola, reduciendo la
  dimensión del histograma.

Crece los árboles **leaf-wise** (la hoja de mayor ganancia) en vez de
level-wise: con el mismo número de hojas ajusta mejor, pero sin limitar
`num_leaves` y `max_depth` sobreajusta. Hiperparámetros que mandan:
`num_leaves`, `min_data_in_leaf`, `learning_rate`, `n_estimators` con early
stopping, `feature_fraction` y `bagging_fraction`.
{% endif %}

{% if model_type == 'todos' or model_type == 'CatBoost' or use_catboost %}
### CatBoost: ordered target statistics y árboles simétricos

Para features categóricas la sustitución estándar es la *target statistic*
(TS). La greedy TS suavizada con prior $p$ y parámetro $a$ (Eq. 4):

$$ \hat{x}_k^i = \frac{\sum_{j=1}^{n} \mathbb{1}_{x_j^i = x_k^i}\, y_j
   + a p}{\sum_{j=1}^{n} \mathbb{1}_{x_j^i = x_k^i} + a} $$

**El problema es la filtración del target**: $\hat{x}_k^i$ usa $y_k$, lo
que produce un *conditional shift* — la distribución de $\hat{x}^i \mid y$
difiere entre train y test. Caso extremo del paper: feature categórica con
valores únicos y $P(y{=}1 \mid A) = 0.5$; la greedy TS separa el train con
un solo split pero predice $p$ en todo el test (accuracy 0.5). La propiedad
deseada es $E(\hat{x}_k^i \mid y) = E(\hat{x}^i \mid y)$. Leave-one-out
tampoco lo evita (falla con una feature constante). La **ordered TS**
(Eq. 5) introduce un orden artificial: una permutación aleatoria $\sigma$
de los datos y, para el ejemplo $k$, solo su historia
$D_k = \{ x_j : \sigma(j) < \sigma(k) \}$:

$$ \hat{x}_k^i = \frac{\sum_{x_j \in D_k} \mathbb{1}_{x_j^i = x_k^i}\, y_j
   + a p}{\sum_{x_j \in D_k} \mathbb{1}_{x_j^i = x_k^i} + a} $$

Para test, $D_k = D$ completo. Satisface P1 (sin conditional shift) y usa
todos los datos (P2). Con una sola permutación las primeras muestras tienen
TS de alta varianza, así que CatBoost usa permutaciones distintas por etapa
del boosting.

El paper identifica además la *prediction shift* del gradient boosting
clásico: los gradientes $g_t(x_k, y_k) \mid x_k$ se calculan sobre el mismo
$D$ con el que se ajusta cada $h_t$, lo que sesga el predictor base y la
generalización de $F_t$. La solución es el **ordered boosting**: mantener
modelos de soporte que predicen cada ejemplo sin haberlo visto en su propio
gradiente. La implementación práctica usa, por permutación, modelos
auxiliares $M_{r,j}$; el modo `Ordered` rinde más en datasets pequeños
(donde el shift se nota más) y cuesta ~1.7x más. Además, los árboles de
CatBoost son **simétricos** (oblivious): el mismo split en todas las hojas,
lo que regulariza, acelera la inferencia y reduce el sobreajuste a costa de
flexibilidad.
{% endif %}

{% if model_type == 'todos' or model_type == 'SVM' %}
## Máquinas de vector soporte (SVM)

### Primal y dual

La SVM de margen máximo busca $w$, $b$ tales que $y_i(w^\top x_i + b) \ge 1$
con norma mínima. Con slack $\xi_i$ para puntos no separables (soft margin,
pérdida hinge) el primal es:

$$ \min_{w,b,\xi} \; \tfrac{1}{2}\|w\|^2 + C\sum_i \xi_i, \quad
   \text{s.t. } y_i(w^\top x_i + b) \ge 1 - \xi_i, \; \xi_i \ge 0 $$

El dual de Lagrange convierte el problema en uno sobre $\alpha_i \ge 0$
(donde solo los puntos con $\alpha_i > 0$, los **vectores soporte**, pesan):

$$ \max_{\alpha} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i
   \alpha_j y_i y_j \, K(x_i, x_j), \quad \text{s.t. } \sum_i
   \alpha_i y_i = 0, \; 0 \le \alpha_i \le C $$

La decisión final es $f(x) = \mathrm{sign}(\sum_i \alpha_i y_i K(x_i, x)
+ b)$: solo los vectores soporte entran en la predicción. La dualidad
permite el **kernel trick** (cambiar el producto escalar por un kernel) sin
coste adicional de dimensión.

### Kernels y el RBF

| Kernel | Fórmula | Nota |
|---|---|---|
| Lineal | $x_i^\top x_j$ | Frontera lineal; interpretable |
| Polinómico | $(\gamma x_i^\top x_j + r)^d$ | Interacciones de grado $d$ |
| RBF | $\exp(-\gamma\|x_i - x_j\|^2)$ | Universal; kernel de infinitas dimensiones |

El RBF es el default útil en la práctica, pero $\gamma$ manda: mide el
ancho — con $\gamma$ grande cada punto queda aislado (sobreajuste) y con
$\gamma$ pequeño la frontera tiende a lineal. `C` controla los vectores
soporte permitidos fuera del margen (inverso de regularización).

### Hinge loss y vectores soporte

La SVM resuelve $\min \tfrac{1}{2}\|w\|^2 + C\sum_i
\max(0, 1 - y_i f(x_i))$: una **pérdida hinge** (convexa, cota superior de
la 0-1) regularizada con $L_2$. Los puntos con $y_i f(x_i) > 1$ no pagan
nada y no son soporte; los del margen o las violaciones son los que definen
el modelo. De ahí su robustez con datos separables y su ineficiencia con
mucho solape.

### Sensibilidad a escala

El RBF (y todo kernel) usa distancias euclídeas: una feature con rango
grande domina la distancia. Estandarizar (media 0, varianza 1) antes de
entrenar es obligatorio. La SVM no produce probabilidades nativas: para
calibrar hay que usar Platt scaling (ver `clasificacion.md`).
{% endif %}

{% if model_type == 'todos' or model_type == 'KNN' %}
## K vecinos más cercanos (KNN)

### Geometría

KNN clasifica o predice con la moda/media de los $k$ vecinos más cercanos
bajo una métrica $d$: la frontera es una partición de Voronoi (poligonal
piecewise lineal). No entrena nada (modelo no paramétrico perezoso): guarda
los datos y predice con búsqueda de vecinos — coste de predicción $O(nd)$
sin índices.

### Métricas de distancia

| Métrica | Fórmula | Uso |
|---|---|---|
| Euclídea (L2) | $\sqrt{\sum_j (x_{ij} - x_{kj})^2}$ | Default; sensible a escala |
| Manhattan (L1) | $\sum_j |x_{ij} - x_{kj}|$ | Robusta a outliers en features |
| Coseno | $\frac{x_i^\top x_k}{\|x_i\|\|x_k\|}$ | Texto/embeddings (solo dirección) |
| Mahalanobis | $\sqrt{(x_i - x_k)^\top \Sigma^{-1}(x_i - x_k)}$ | Cuenta la covarianza de los datos |

### La maldición de la dimensionalidad

En dimensión alta el volumen se concentra en la superficie y las distancias
al vecino más cercano se concentran: el contraste relativo
$(d_{\max} - d_{\min})/d_{\min} \to 0$, así que el "vecino más cercano" es
casi un punto aleatorio. Regla práctica: se necesitan $n \gg 10^d$ muestras
y aun así con $d \gtrsim 20$ el contraste colapsa. Mitigación: reducir
dimensión (PCA), seleccionar features, escalar siempre.

### Hiperparámetros y límites

- `k` pequeño -> alta varianza; `k` grande -> alto sesgo. Validar $k$ por
  CV (heurística $k \approx \sqrt{n}$).
- Pesos inversos a la distancia (`weights='distance'`) suavizan la frontera.
- Escalado obligatorio: cualquier métrica Lp está dominada por la feature de
  mayor rango. Con clases muy desbalanceadas KNN pierde: las clases
  mayoritarias rodean el espacio (ver `clasificacion.md`).
{% endif %}

## Práctica: cuándo cada familia

| Familia | Dónde rinde | Dónde falla |
|---|---|---|
| Lineal | $n$ pequeño, fronteras lineales, interpretación | No linealidad fuerte o interacciones |
| Árboles/GBDT | Tabular: mixtas, no linealidad, ausentes | No extrapola; alta dimensión, poca señal |
| SVM (RBF) | Datos medios, clases separadas | Sin escalar, $n$ grande, sin calibración |
| KNN | $n$ pequeño, dimensión baja, distancia útil | Dimensión alta, desbalance, memoria |

**Pitfalls que se repiten en producción:**

1. **Árboles sobre one-hot disperso**: one-hot convierte una categórica en
   muchas columnas débiles; los GBDT pierden señal y velocidad. Para
   categóricas de alta cardinalidad usar el soporte nativo (CatBoost,
   LightGBM) o target encoding con cuidado (ver
   [ingenieria-features.md](../ingenieria/ingenieria-features.md)).
2. **Modelos de distancia sin estandarizar**: SVM, KNN (y k-means) quedan
   dominados por la feature de mayor rango. `StandardScaler`/`RobustScaler`
   antes de entrenar.
3. **Boosting sobre-iterado**: más árboles de los que la validación
   justifica (o `eta` demasiado alto) sobreajusta el ruido. Early stopping
   sobre un conjunto de validación honesto y shrinkage bajo.
4. **Desbalance**: con clases raras, el accuracy es mentira; reweighting,
   sampling y métricas de minoría en `clasificacion.md`.
5. **Target con cola pesada en regresión**: log-transformar el target y
   pensar en intervalos, no solo en el punto medio — `regresion.md`.

## Fuentes

- **XGBoost: A Scalable Tree Boosting System** — T. Chen, C. Guestrin (2016).
  arXiv:1603.02754 — https://arxiv.org/abs/1603.02754
- **CatBoost: Unbiased Boosting with Categorical Features** —
  L. Prokhorenkova et al. (2018). arXiv:1706.09516 —
  https://arxiv.org/abs/1706.09516
- **LightGBM: A Highly Efficient Gradient Boosting Decision Tree** —
  G. Ke et al. (2017). arXiv:1703.09016 — https://arxiv.org/abs/1703.09016
- **Greedy Function Approximation: A Gradient Boosting Machine** —
  J. H. Friedman (2001). Sin arXiv — https://doi.org/10.1214/aos/1013203451
- **Random Forests** — L. Breiman (2001). Sin arXiv —
  https://doi.org/10.1023/A:1010933404324
- **A Decision-Theoretic Generalization of On-Line Learning and an
  Application to Boosting** — Y. Freund, R. E. Schapire (1997). Sin arXiv —
  https://doi.org/10.1006/jcss.1997.1504
- **Regression Shrinkage and Selection via the Lasso** — R. Tibshirani
  (1996). Sin arXiv — https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
- **A Training Algorithm for Optimal Margin Classifiers** — C. Cortes,
  V. Vapnik (1995). Sin arXiv — https://doi.org/10.1145/174130.174182
- **Nearest Neighbor Pattern Classification** — T. M. Cover, P. E. Hart
  (1967). Sin arXiv — https://doi.org/10.1109/TIT.1967.1053964
- **The Elements of Statistical Learning** — T. Hastie, R. Tibshirani,
  J. Friedman (2009). Sin arXiv — https://hastie.su.domains/ElemStatLearn/
{% endif %}
