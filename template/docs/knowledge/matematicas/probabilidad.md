# Probabilidad

Marco formal para la incertidumbre: axiomas, variables aleatorias, distribuciones
canónicas y la lógica de la inferencia (MLE, MAP, bayes). Todo lo siguiente se
aplica a cualquier proyecto DS: escoger la distribución de un objetivo, derivar
una pérdida, calibrar una confianza o reportar intervalos.

## Axiomas de Kolmogorov

Un espacio de probabilidad $(\Omega, \mathcal{F}, P)$ exige:

1. $P(A) \ge 0$ para todo evento $A$.
2. $P(\Omega) = 1$.
3. Aditividad numerable: si $A_1, A_2, \dots$ son disjuntos,
   $P(\cup_i A_i) = \sum_i P(A_i)$.

De aquí salen las reglas de uso diario. Probabilidad condicional ($P(B) > 0$):

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}.$$

Regla de la cadena (se generaliza a $n$ eventos):

$$P(A \cap B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A).$$

## Independencia e independencia condicional

$A$ y $B$ son independientes si $P(A \cap B) = P(A)\,P(B)$, equivalente a
$P(A \mid B) = P(A)$. Independencia condicional dado $C$:

$$P(A \cap B \mid C) = P(A \mid C)\,P(B \mid C).$$

**Trampa clásica:** independencia marginal NO implica independencia
condicional. Dos síntomas $X_1, X_2$ de una enfermedad $E$ pueden ser
marginalmente independientes pero condicionalmente dependientes dado $E$ (y al
revés). En ML: variables que "duplican" información parecen independientes
marginalmente y rompen los supuestos de un clasificador naive Bayes. Antes de
asumir independencia condicional en un diagnóstico, grafícalo o pruébalo
condicional.

## Ley de probabilidad total

Si $\{B_j\}$ particiona $\Omega$:

$$P(A) = \sum_j P(A \mid B_j)\,P(B_j).$$

Es el instrumento para marginalizar latentes: aparece al integrar variables
ocultas, al descomponer mezclas y al normalizar Bayes. **Abuso típico:** olvidar
que los $B_j$ deben cubrir todo el espacio; si dejas un hueco, las probabilidades
no suman 1.

## Teorema de Bayes

Sigue de la regla de la cadena y de la ley de probabilidad total:

$$P(B_i \mid A) = \frac{P(A \mid B_i)\,P(B_i)}{\sum_j P(A \mid B_j)\,P(B_j)}.$$

En inferencia, con $\theta$ el parámetro y $D$ los datos:

$$P(\theta \mid D) = \frac{P(D \mid \theta)\,P(\theta)}{P(D)}.$$

- Prior $P(\theta)$: conocimiento previo o regularización.
- Likelihood $P(D \mid \theta)$: cuán bien explica cada $\theta$ los datos.
- Evidence $P(D) = \int P(D \mid \theta)\,P(\theta)\,d\theta$: constante de
  normalización, independiente de $\theta$.
- Posterior $P(\theta \mid D)$: lo que hay que reportar, no solo un valor
  puntual.

Forma de odds: tomando el cociente entre dos hipótesis,

$$\frac{P(B_1 \mid A)}{P(B_2 \mid A)} =
\frac{P(A \mid B_1)}{P(A \mid B_2)} \cdot \frac{P(B_1)}{P(B_2)},$$

es decir, odds posteriores = razón de verosimilitud $\times$ odds previas. Los
datos solo aportan la razón de verosimilitud; el resto es prior.

**Uso en DS:** comparación de modelos por factor de Bayes, actualización
secuencial, calibración. **Mal uso:** leer "likelihood" como "probabilidad de
que el parámetro sea el correcto"; la verosimilitud no integra a 1 y no es una
densidad sobre $\theta$.

## Variables aleatorias

Una variable aleatoria $X$ es una función medible $X: \Omega \to \mathbb{R}$.
La describen tres objetos:

- PMF (discreta): $p(x) = P(X = x)$.
- PDF (continua): $f(x)$, con $P(a \le X \le b) = \int_a^b f(x)\,dx$.
- CDF: $F(x) = P(X \le x)$; en el caso continuo $f(x) = F'(x)$.

### Esperanza y varianza

$$E[X] = \sum_x x\,p(x) \quad \text{o} \quad E[X] = \int x\,f(x)\,dx,$$

$$\mathrm{Var}(X) = E\big[(X - E[X])^2\big] = E[X^2] - E[X]^2.$$

Propiedades esenciales:

- Linealidad: $E[aX + bY] = aE[X] + bE[Y]$ (no requiere independencia).
- $\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X)$.
- $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X,Y)$.
- Si $X \perp Y$: $E[XY] = E[X]\,E[Y]$ y $\mathrm{Cov}(X,Y) = 0$.

**Independencia de variables:** $F_{X,Y} = F_X F_Y$ (o $f_{X,Y} = f_X f_Y$).
iid = independientes e idénticamente distribuidas; es el supuesto estándar de
entrenamiento, y su violación (series temporales, datos agrupados, leakage) es
la causa más común de errores estándar optimistas y de cross-validation roto.

### Ley de esperanza total y ley de varianza total

$$E[X] = E\big[E[X \mid Y]\big],$$

$$\mathrm{Var}(X) = E\big[\mathrm{Var}(X \mid Y)\big] + \mathrm{Var}\big(E[X \mid Y]\big).$$

La segunda descompone la varianza en variación dentro de los grupos
($E[\mathrm{Var}(X \mid Y)]$) y entre grupos ($\mathrm{Var}(E[X \mid Y])$). Es
la base de los modelos jerárquicos y de la intuición del $R^2$ en regresión:
cuánta variación "explica" un predictor al condicionar. **Mal uso:** asumir iid
con datos correlacionados o anidados; los intervalos de confianza quedan
subestimados y el modelo parece mejor de lo que es.

## Distribuciones de referencia

Elige distribución por soporte, mecanismo generador y colas. Se da el PMF/PDF
exacto, media y varianza; después, cuándo aparece y cómo se mal usa.

### Bernoulli($p$)

$X \in \{0,1\}$, un ensayo con probabilidad de éxito $p$:

$$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0,1\}.$$

$E[X] = p$, $\mathrm{Var}(X) = p(1-p)$. Es la unidad de éxito/fallo: churn,
fraude, conversión. **Mal uso:** modelar como Bernoulli algo con dependencia
temporal o entre unidades.

### Binomial($n, p$)

Suma de $n$ Bernoulli iid:

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, \dots, n.$$

$E[X] = np$, $\mathrm{Var}(X) = np(1-p)$. Conteo de éxitos en un número fijo de
ensayos independientes.

### Poisson($\lambda$)

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k \ge 0.$$

$E[X] = \lambda$, $\mathrm{Var}(X) = \lambda$. Límite de la Binomial con
$n \to \infty$, $p \to 0$, $np \to \lambda$: eventos raros en una ventana
(clics, llegadas, siniestros, fallos). **Mal uso:** usarla con sobredispersión
($\mathrm{Var} > E[X]$); ahí corresponde la negativa binomial o un modelo de dos
parámetros (GLM quasi-Poisson).

### Uniforme($a, b$)

$$f(x) = \frac{1}{b-a}, \quad x \in [a, b].$$

$E[X] = \frac{a+b}{2}$, $\mathrm{Var}(X) = \frac{(b-a)^2}{12}$. Prior no
informativo acotado; base de la transformada inversa para simulación.

### Exponencial($\lambda$)

$$f(x) = \lambda e^{-\lambda x}, \quad x \ge 0.$$

$E[X] = 1/\lambda$, $\mathrm{Var}(X) = 1/\lambda^2$. Tiempos entre eventos de un
proceso de Poisson. **Falta de memoria:**

$$P(X > s + t \mid X > s) = P(X > t).$$

**Uso:** survival simple, priors de decaimiento, tasas de llegada.
**Mal uso:** tiempos cuya tasa cambia con el tiempo (no es exponencial) o con
colas pesadas.

### Gamma($\alpha, \beta$)

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)}\,x^{\alpha-1} e^{-\beta x},
\quad x > 0,$$

con $\Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t}\,dt$. $E[X] =
\alpha/\beta$, $\mathrm{Var}(X) = \alpha/\beta^2$. Generaliza la Exponencial
($\alpha = 1$) y es conjugada al Poisson: modela el tiempo de espera de la suma
de $\alpha$ exponenciales y sirve de prior para tasas y precisiones.

### Beta($\alpha, \beta$)

$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}, \quad x \in [0,1],$$

con $B(\alpha,\beta) = \frac{\Gamma(\alpha)\,\Gamma(\beta)}{\Gamma(\alpha+\beta)}$.

$$E[X] = \frac{\alpha}{\alpha+\beta}, \qquad
\mathrm{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}.$$

Distribución sobre probabilidades; conjugada a Bernoulli/Binomial. **Uso:** prior
de tasas de éxito, métricas acotadas (precisión, cobertura, conversión). **Mal
uso:** modelar con Beta métricas continuas no acotadas, o con suporte truncado
que no es $[0,1]$ (usar Beta-cuádruple o escalada).

### Normal / Gaussiana($\mu, \sigma^2$)

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).$$

$E[X] = \mu$, $\mathrm{Var}(X) = \sigma^2$. Central por el Teorema del Límite
Central: la suma (promedio) de variables iid con varianza finita converge a una
Normal. Por eso promedios y residuos de regresión son tratables. **Mal uso:**
asumir normalidad en datos positivos o de colas pesadas (ingresos, tiempos,
recuentos); usa Laplace, t de Student o transformaciones (log, Box-Cox).

### Laplace($\mu, b$)

$$f(x) = \frac{1}{2b}\exp\left(-\frac{|x-\mu|}{b}\right).$$

$E[X] = \mu$, $\mathrm{Var}(X) = 2b^2$. Colas más pesadas que la Normal;
conecta con la pérdida L1 y la mediana. Modelo de errores robusto: el MLE de
$\mu$ es la mediana muestral.

### Categórica y multinomial

Categórica($\mathbf p$): $P(X = k) = p_k$ con $\sum_k p_k = 1$; es el objetivo de
la clasificación multiclase. Multinomial($n, \mathbf p$) con $\sum_k x_k = n$:

$$P(\mathbf x) = \frac{n!}{x_1! \cdots x_K!}
\prod_{k=1}^{K} p_k^{x_k}.$$

$E[X_k] = n p_k$, $\mathrm{Var}(X_k) = n p_k (1-p_k)$,
$\mathrm{Cov}(X_k, X_l) = -n p_k p_l$ (¡las cuentas son negativamente
dependientes!). Da la forma de la pérdida de cross-entropy en clasificación.

### Normal multivariante $\mathcal{N}(\boldsymbol\mu, \boldsymbol\Sigma)$

$$f(\mathbf x) = \frac{1}{(2\pi)^{D/2} |\boldsymbol\Sigma|^{1/2}}
\exp\left(-\frac{1}{2}(\mathbf x - \boldsymbol\mu)^\top
\boldsymbol\Sigma^{-1}(\mathbf x - \boldsymbol\mu)\right).$$

Partiendo $\mathbf x = (\mathbf x_a^\top, \mathbf x_b^\top)^\top$,
$\boldsymbol\mu = (\boldsymbol\mu_a^\top, \boldsymbol\mu_b^\top)^\top$ y

$$\boldsymbol\Sigma =
\begin{pmatrix} \boldsymbol\Sigma_{aa} & \boldsymbol\Sigma_{ab} \\
                \boldsymbol\Sigma_{ba} & \boldsymbol\Sigma_{bb} \end{pmatrix},$$

la condicional es Gaussiana (clave en GMMs, kriging, procesos gaussianos,
Kalman):

$$\mathbf x_a \mid \mathbf x_b \sim
\mathcal{N}(\boldsymbol\mu_{a|b}, \boldsymbol\Sigma_{a|b}),$$

$$\boldsymbol\mu_{a|b} = \boldsymbol\mu_a +
\boldsymbol\Sigma_{ab}\,\boldsymbol\Sigma_{bb}^{-1}
(\mathbf x_b - \boldsymbol\mu_b),$$

$$\boldsymbol\Sigma_{a|b} = \boldsymbol\Sigma_{aa} -
\boldsymbol\Sigma_{ab}\,\boldsymbol\Sigma_{bb}^{-1}\boldsymbol\Sigma_{ba}.$$

La condicional reduce varianza frente a la marginal (los $\mathbf x_b$ aportan
información); $\boldsymbol\Sigma_{a|b}$ es el complemento de Schur. La marginal
es $\mathbf x_a \sim \mathcal{N}(\boldsymbol\mu_a, \boldsymbol\Sigma_{aa})$ y
$\mathbf A\mathbf x + \mathbf b \sim \mathcal{N}(\mathbf A\boldsymbol\mu +
\mathbf b, \mathbf A\boldsymbol\Sigma\mathbf A^\top)$: por eso las proyecciones
PCA de un Gaussiano son Gaussianas.

## Funciones generatrices de momentos

$$M_X(t) = E\big[e^{tX}\big].$$

Derivando en $t = 0$ se recuperan los momentos: $E[X^k] = M_X^{(k)}(0)$. Dos
variables con la misma MGF en un entorno de $0$ tienen la misma distribución:
la MGF caracteriza la distribución. Por eso la suma de normales independientes
es normal y el CLT se demuestra con MGFs. **Uso práctico:** identificar la
distribución de combinaciones lineales y derivar momentos de mezclas.
**Límite:** la MGF puede no existir (Cauchy, colas pesadas); ahí se usa la
función característica $\varphi(t) = E[e^{itX}]$.

## Máxima verosimilitud (MLE)

Dados datos iid $\{x_1, \dots, x_n\}$ de $p(x; \theta)$,

$$\hat\theta_{ML} = \arg\max_\theta \prod_{i=1}^{n} p(x_i; \theta)
= \arg\max_\theta \sum_{i=1}^{n} \log p(x_i; \theta).$$

### Trabajado: Bernoulli

Con $\ell(p) = \sum_i x_i \log p + \big(n - \sum_i x_i\big)\log(1-p)$, se deriva
e iguala a cero:

$$\hat p = \frac{1}{n}\sum_{i=1}^{n} x_i.$$

El MLE de la probabilidad de éxito es la frecuencia empírica; se obtiene donde
se anula la derivada del log-likelihood, sin optimización numérica.

### Trabajado: Gaussiana

Con $\mu$ y $\sigma^2$ desconocidos:

$$\hat\mu = \bar x = \frac{1}{n}\sum_i x_i, \qquad
\hat\sigma^2 = \frac{1}{n}\sum_i (x_i - \bar x)^2.$$

Nota: el MLE de la varianza está sesgado (divide por $n$); el estimador
insesgado divide por $n-1$. Primer ejemplo de que MLE optimiza verosimilitud,
no sesgo.

### Invariencia

Si $\hat\theta$ es el MLE de $\theta$, entonces $g(\hat\theta)$ es el MLE de
$g(\theta)$ para cualquier transformación $g$: se estima $\sigma$ o $\log\sigma$
a partir de $\hat\sigma^2$ sin re-estimar.

### Normalidad asintótica y errores estándar

$$\sqrt{n}\big(\hat\theta_{ML} - \theta_0\big)
\xrightarrow{d} \mathcal{N}\Big(0,\ \mathcal{I}(\theta_0)^{-1}\Big),$$

con información de Fisher (las dos formas son equivalentes):

$$\mathcal{I}(\theta) = -E_\theta\Big[\tfrac{\partial^2}{\partial\theta^2}
\log p(x;\theta)\Big]
= E_\theta\Big[\Big(\tfrac{\partial}{\partial\theta}\log p(x;\theta)\Big)^2\Big].$$

En la práctica, el SE de $\hat\theta_j$ es la raíz del elemento $jj$ de la
inversa del Hessiano del negativo del log-likelihood en el óptimo (matriz de
información observada); con ello construyes intervalos y tests.
**Mal uso:** la normalidad asintótica falla en muestras pequeñas, en fronteras
del espacio de parámetros ($p = 0$) y cuando el modelo no está identificado
(Hessiano singular → SEs infinitos, señal de colinealidad).

## MAP y MLE regularizado

El MAP maximiza la posterior:

$$\hat\theta_{MAP} = \arg\max_\theta
\big[\log P(D \mid \theta) + \log P(\theta)\big].$$

Es MLE + log-prior, y ahí está el puente con la regularización:

- Prior Gaussiano sobre pesos $\Leftrightarrow$ penalización L2 (ridge):
  minimizar $-\log P(D \mid w) + \lambda \|w\|_2^2$.
- Prior de Laplace $\Leftrightarrow$ L1 (lasso): $-\log P(D \mid w) + \lambda
  \|w\|_1$; la cúspide en 0 induce ceros exactos (selección de variables).

**Mal uso:** MAP no es "bayes completo": devuelve la moda, no incertidumbre, y
no es invariante a reparametrizaciones (la moda de $g(\theta)$ no es $g$ de la
moda). Para cuantificar incertidumbre necesitas la posterior completa (muestreo
o aproximación de Laplace).

## Inferencia bayesiana y priors conjugados

Actualización secuencial: posterior $\propto$ likelihood $\times$ prior; la
posterior de hoy es el prior de mañana (aprendizaje online). Un prior es
conjugado si la posterior pertenece a la misma familia.

| Likelihood | Prior | Posterior | Uso |
|---|---|---|---|
| Bernoulli($p$) | Beta($\alpha,\beta$) | Beta($\alpha+\sum x_i,\ \beta+n-\sum x_i$) | tasas de éxito |
| Poisson($\lambda$) | Gamma($a, b$) | Gamma($a + \sum x_i$, $b + n$) | tasas de llegada |
| Normal($\mu$), $\sigma^2$ conocida | Normal($\mu_0, \tau_0^{-1}$) | Normal (ver abajo) | medias |
| Multinomial($\mathbf p$) | Dirichlet($\alpha$) | Dirichlet($\alpha+\mathbf n$) | probs. de clase |

En el caso Gaussiano con precisión $\tau = 1/\sigma^2$:

$$\tau_n = \tau_0 + n\tau, \qquad
\mu_n = \frac{\tau_0\mu_0 + n\tau \bar x}{\tau_0 + n\tau}.$$

La precisión posterior se obtiene sumando precisiones, y la media posterior es
un promedio ponderado por precisión entre prior y datos (la actualización del
Dirichlet es el smoothing tipo Laplace: añadir pseudo-conteos $\boldsymbol\alpha$).

**Aplicación:** estimar tasas con datos escasos sin caer en $0$ o $1$; A/B
testing secuencial; priors sobre probabilidades de clase. **Mal uso:** elegir
conjugados por conveniencia sin justificar el prior; con pocos datos la
posterior queda dominada por el prior y la respuesta es "prior escondido".

## Desigualdad de Jensen

Para $\phi$ convexa,

$$E\big[\phi(X)\big] \ge \phi\big(E[X]\big),$$

con el sentido invertido si $\phi$ es cóncava. Consecuencias concretas: la
cross-entropy es $\ge$ entropía (la pérdida de clasificación no baja de la
entropía del problema), $\log E[X] \ge E[\log X]$ (media aritmética $\ge$
geométrica) y es el paso clave del EM (cota inferior de la evidencia).
**Uso:** saber en qué dirección sesga promediar funciones no lineales — promediar
logits y luego softmax es distinto de promediar probabilidades ya convertidas.

## Práctica

- **Elegir distribución:** primero el soporte (binario, conteo, positivo,
  continuo acotado/no), luego el mecanismo (eventos raros → Poisson, tiempos →
  Exponencial/Gamma, proporción → Beta, residuos de regresión → Normal salvo
  colas pesadas). Ante sobredispersión usa negativa binomial; ante exceso de
  ceros, modelos inflados a cero.
- **Likelihood vs probabilidad:** la verosimilitud es una función del parámetro
  con los datos fijos; como función de $\theta$ no integra a 1 y no es una
  densidad. Un likelihood alto no es "probabilidad de que $\theta$ sea cierto".
- **Log-likelihood:** monótona, convierte productos en sumas (estabilidad
  numérica) y da la forma natural de la pérdida: el negativo del log-likelihood
  es cross-entropy en clasificación y MSE en regresión Gaussiana. Suma
  log-probabilidades con log-sum-exp:

```python
def log_sum_exp(logs):
    m = max(logs)                     # evita overflow
    return m + np.log(np.sum(np.exp(np.array(logs) - m)))
```

- **Calibración:** un modelo está calibrado si $P(Y{=}1 \mid \hat p = s) = s$
  para todo $s$. Mídelo con diagrama de fiabilidad, ECE y Brier; corrige con
  temperature/Platt scaling o isotónico. Una log-loss baja no garantiza buena
  calibración (modelos sobremojados o bajoentrenados la desvirtúan).
- **Conformal:** para cobertura con garantía sin supuestos de distribución
  (solo exchangeability), usa predicción conformal sobre un score: cuantil del
  score de un calibration set.

{% if use_calibration %}
**Con calibración activa en este proyecto:** temperature/Platt scaling asumen
que el ranking del modelo es correcto y solo ajustan la probabilidad de salida;
si la log-loss de entrenamiento no baja, la calibración no arregla un modelo
mal entrenado. Reporta ECE y Brier en validación, nunca en train.
{% endif %}

{% if use_conformal %}
**Con conformal activo en este proyecto:** la cobertura $1-\alpha$ es sobre
observaciones nuevas exchangeables, no sobre el train; reutilizar el calibration
set para otra cosa (early stopping, selección) rompe la garantía. Los intervalos
son distribution-free pero heredan el sesgo del predictor base.
{% endif %}

## Fuentes

- C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
- T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*,
  Springer, 2009.
- G. Casella, R. L. Berger, *Statistical Inference*, 2nd ed., Duxbury, 2002.
- D. P. Bertsekas, J. N. Tsitsiklis, *Introduction to Probability*, Athena
  Scientific, 2008.
