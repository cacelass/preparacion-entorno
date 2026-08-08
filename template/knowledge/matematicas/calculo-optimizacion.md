# Cálculo diferencial y optimización

Referencia profunda para el agente `lider`: gradientes, convexidad, métodos
de primer y segundo orden, schedules y la geometría de la regularización.
Cada sección conecta la teoría con cuándo aplicarla en un proyecto DS real y
con los abusos más frecuentes. Se asume cálculo multivariable y álgebra
lineal (ver `algebra-lineal.md`).

## Gradiente, Jacobiano y Hessiana

Para $f: \mathbb{R}^d \to \mathbb{R}$ diferenciable, el **gradiente** es el
vector de derivadas parciales:

$$ \nabla f(x) = \left( \frac{\partial f}{\partial x_1}, \ldots,
   \frac{\partial f}{\partial x_d} \right)^\top $$

- Apunta a la dirección de máximo crecimiento local; $-\nabla f(x)$ a la de
  máximo descenso.
- En todo mínimo (local o global) de $f$ suave, $\nabla f = 0$ (condición
  necesaria, no suficiente).

**Jacobiano**: para $f: \mathbb{R}^n \to \mathbb{R}^m$,
$J \in \mathbb{R}^{m \times n}$ con $J_{ij} = \partial f_i / \partial x_j$.
Es la mejor aproximación lineal de $f$ cerca de $x$:
$f(x + \delta) \approx f(x) + J(x)\,\delta$. Generaliza el gradiente
($m = 1$ da $\nabla f = J^\top$) y es la matriz que propaga errores,
varianzas y gradientes a través de una capa.

**Hessiana**: para $f: \mathbb{R}^d \to \mathbb{R}$ dos veces diferenciable,
$H_{ij} = \partial^2 f / \partial x_i \partial x_j$. Si $f$ es $C^2$,
$H$ es simétrica (Clairaut) y sus autovalores son las curvaturas por
dirección. $H(x) \succeq 0$ en todo punto es equivalente a que $f$ sea
convexa (ver sección de convexidad).

**Derivada direccional**: $D_v f(x) = \lim_{t \to 0} \frac{f(x+tv) - f(x)}{t}
= \nabla f(x)^\top v$. Entre direcciones unitarias, la de mayor incremento
es $v \parallel \nabla f(x)$. La **segunda** derivada direccional,
$v^\top H(x) v$, mide la curvatura a lo largo de $v$: positiva → la
pendiente crece en esa dirección; negativa → dirección de curvatura negativa
(relevante en puntos de silla, ver trampas).

Abuso: usar diferencias finitas como método de cómputo (y no solo para
verificar un gradiente analítico en un punto de prueba) cuesta $O(d)$
evaluaciones y arrastra error de truncamiento y cancelación — nunca para
entrenar.

## Regla de la cadena y backpropagación

Dado el grafo computacional de la pérdida $L(\theta)$, backpropagation es la
regla de la cadena evaluada en **modo reverso**: el paso atrás multiplica los
Jacobianos locales (adyacentes a cada nodo) en orden inverso a la pasada
directa, reutilizando las activaciones guardadas.

$$ \frac{\partial L}{\partial \theta} =
   \frac{\partial L}{\partial z_k} \cdot
   \frac{\partial z_k}{\partial z_{k-1}} \cdots
   \frac{\partial z_1}{\partial \theta} $$

| Modo | Gradiente ($m{=}1$) | Jacobiano $m \times n$ | Memoria |
|---|---|---|---|
| Forward | $O(d)$ pases | $O(n)$ pases | baja |
| Reverse (backprop) | $1$ pase | $O(m)$ pases | activaciones por nodo |

Cuando $f$ es escalar (una pérdida), reverse mode da el gradiente exacto en
**un** pase, frente a los $O(d)$ pases del modo forward o de diferencias
finitas. El precio es la memoria: reverse guarda activaciones intermedias,
lo que en redes profundas y batches grandes domina el consumo — de ahí el
gradient checkpointing (recalcular en vez de guardar).

## Convexidad

$f$ es convexa si su dominio es convexo y, para todo $x, y$ y
$\lambda \in [0, 1]$:

$$ f(\lambda x + (1-\lambda) y) \le \lambda f(x) + (1-\lambda) f(y) $$

Equivalentes para $f$ diferenciable: $f(y) \ge f(x) + \nabla f(x)^\top(y-x)$
(el gráfico queda por encima de sus tangentes) y, si $f$ es $C^2$,
$H(x) \succeq 0$ en todo punto.

**Por qué convexo es tratable**: todo mínimo local es global; no hay puntos
de silla; cualquier punto con $\nabla f = 0$ es solución; los métodos de
primer orden convergen con garantías sin conocer el óptimo; y las condiciones
KKT son necesarias y suficientes. Fuera de la convexidad, la optimización es
búsqueda no lineal sin garantía de mínimo global: en DS la convexidad es la
frontera entre "resuelvo de forma fiable" (logística, ridge, SVM lineal, EM
para familias exponenciales) y "rezo con el paisaje" (redes profundas).

**Convexidad fuerte**: existe $\mu > 0$ tal que

$$ f(y) \ge f(x) + \nabla f(x)^\top(y-x) + \frac{\mu}{2}\|y - x\|^2 $$

(para $f$ $C^2$, $H \succeq \mu I$). Aporta minimizador único y convergencia
lineal del gradiente: el error se multiplica por un factor < 1 en cada
iteración.

**Número de condición**: $\kappa = L / \mu$, con $L$ la constante de
suavidad ($\|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|$; para $C^2$,
$H \preceq L I$). $\kappa$ grande → curvas de nivel elípticas largas y
delgadas → convergencia lenta (ver descenso de gradiente).

| Clase de $f$ | Tasa de $f(\theta_k) - f^*$ | Iteraciones para $\epsilon$ |
|---|---|---|
| $L$-suave, convexa | $O(1/k)$ | $O(L/\epsilon)$ |
| $L$-suave, $\mu$-fuerte | $O((1-1/\kappa)^k)$ | $O(\kappa \log(1/\epsilon))$ |
| $\mu$-fuerte + Nesterov | $O((1-1/\sqrt{\kappa})^k)$ | $O(\sqrt{\kappa}\log(1/\epsilon))$ |
| Suave, no convexa | $\min_k \|\nabla f(\theta_k)\|^2 \le O(1/k)$ | $O(L/\epsilon^2)$ |

## Descenso de gradiente (GD)

Actualización con paso $\alpha_t$:

$$ \theta_{t+1} = \theta_t - \alpha_t \nabla f(\theta_t) $$

Para $f$ convexa $L$-suave y $\alpha = 1/L$:

$$ f(\theta_k) - f^* \le \frac{L}{2k}\|\theta_0 - \theta^*\|_2^2 = O(1/k) $$

Para $f$ fuertemente convexa la convergencia es **lineal**:

$$ f(\theta_k) - f^* \le \frac{L}{2}\|\theta_0 - \theta^*\|^2
   \left(1 - \frac{\mu}{L}\right)^k $$

**Paso**: constante $\alpha \le 2/L$ converge en el caso suave; por encima
diverge. Para cuadrática fuertemente convexa el óptimo es
$\alpha = 2/(\lambda_{max} + \lambda_{min})$. Si $L$ no se conoce, usar
backtracking/line search. En la práctica (objetivos de ML), el paso es un
hiperparámetro dominante: demasiado grande diverge, demasiado pequeño no
converge en presupuesto.

**Por qué el mal condicionamiento produce zig-zag**: con curvas de nivel
elípticas de semiejes $\sqrt{\lambda_{max}}$ y $\sqrt{\lambda_{min}}$, el
gradiente es casi perpendicular al eje largo del valle. GD oscila a través
del valle (sobrepasa en la dirección de curvatura grande) y avanza despacio
a lo largo del eje largo; el número de pasos para cerrar el valle escala con
$\kappa$. Soluciones: precondicionar (Newton), o métodos adaptativos que
normalizan por coordenada (AdaGrad/Adam).

## SGD (descenso de gradiente estocástico)

Gradiente de un minibatch $B$ de tamaño $b$:

$$ g_t = \frac{1}{b} \sum_{i \in B} \nabla \ell(\theta_t, x_i), \qquad
   \mathbb{E}[g_t] = \nabla f(\theta_t) $$

El gradiente del minibatch es un estimador **insesgado** del gradiente
completo: en expectación el paso apunta a la dirección correcta, pero con
ruido de varianza $\propto \sigma^2 / b$ (el ruido además suaviza el paisaje
y ayuda a escapar de sillas pequeñas). Coste por paso cae de $O(n)$ a $O(b)$.

Con pasos decrecientes $\alpha_t$ con $\sum_t \alpha_t = \infty$ y
$\sum_t \alpha_t^2 < \infty$ (p.ej. $\alpha_t \propto 1/\sqrt{t}$), para
$f$ convexa suave:

$$ \mathbb{E}[f(\theta_k)] - f^* = O(1/\sqrt{k}) $$

El ruido degrada el $O(1/k)$ del GD completo a $O(1/\sqrt{k})$: los
optimizadores estocásticos pagan el mismo presupuesto en muchas más
iteraciones baratas. Con paso fijo, SGD no converge al minimizador: fluctúa
en una banda de tamaño $\sim \alpha \sigma$.

**Schedules de learning rate** (el schedule es parte del modelo):
- **Step**: multiplicar por $\gamma \in (0, 1)$ cada $T$ épocas. Clásico.
- **Exponencial**: $\alpha_t = \alpha_0 \gamma^{t}$; decae rápido y exige
  buen $\alpha_0$.
- **Coseno**:
  $\alpha_t = \alpha_{min} + \frac{\alpha_0 - \alpha_{min}}{2}(1 + \cos(\pi t/T))$;
  decae suave hasta $\alpha_{min}$ sin "fuga", típico en visión.
- **Warmup + decay**: subir de $\approx 0$ a $\alpha_0$ en pocas épocas
  (necesario con Adam en transformadores) y luego decaer.

{% if use_optuna %}
Con `use_optuna` activo en este proyecto, barre en log-escala: LR en
$[10^{-4}, 10^{-1}]$, weight decay en $[10^{-6}, 10^{-1}]$, momentum en
$[0.85, 0.99]$, $\beta_2$ en $[0.99, 0.9999]$; evalúa en validación con el
schedule completo, no con un recorte de épocas, o el ranking de
hiperparámetros queda sesgado.
{% endif %}

## Momentum y Nesterov

**Momentum** (Polyak): acumula una velocidad que promedia direcciones
consistentes y cancela oscilaciones perpendiculares:

$$ v_{t+1} = \mu v_t + \nabla f(\theta_t), \qquad
   \theta_{t+1} = \theta_t - \alpha v_{t+1}, \qquad \mu \in [0, 1) $$

**Nesterov** (momentum acelerado): evalúa el gradiente en el punto
"adelantado" $\theta_t + \mu v_t$ (lookahead):

$$ v_{t+1} = \mu v_t + \nabla f(\theta_t + \mu v_t), \qquad
   \theta_{t+1} = \theta_t - \alpha v_{t+1} $$

En el régimen convexo fuerte acelera de $(1 - 1/\kappa)$ a
$(1 - 1/\sqrt{\kappa})$, que es la tasa **óptima** para métodos de primer
orden. En deep learning, Sutskever et al. (2013) muestran que el momentum
bien inicializado y combinado con decay de LR es clave para entrenar redes
profundas. Abuso: momentum alto ($\mu \to 1$) con gradientes ruidosos
provoca overshoot y divergencia; en transformadores suele necesitar warmup.

## Métodos adaptativos: AdaGrad, RMSProp, Adam, AdamW

**AdaGrad** acumula el cuadrado histórico y normaliza por coordenada:

$$ G_t = G_{t-1} + g_t^2, \qquad
   \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t} + \epsilon} g_t $$

El LR efectivo decae de forma monótona (la suma crece): excelente para
gradientes dispersos (NLP, embeddings), malo para entrenamientos largos (el
paso muere antes de converger).

**RMSProp** sustituye la suma por una media móvil exponencial
($\rho \in [0, 1)$):

$$ v_t = \rho v_{t-1} + (1-\rho) g_t^2, \qquad
   \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_t} + \epsilon} g_t $$

El paso sobrevive a regímenes largos y no estacionarios. Todas las
operaciones son element-wise: cada parámetro tiene su propia escala y la
magnitud de la actualización es invariante a reescalados diagonales del
gradiente.

### Adam (Algoritmo 1, Kingma & Ba 2015)

Combina la media móvil del gradiente ($m_t$, primer momento) con la del
cuadrado ($v_t$, segundo momento) y corrige el sesgo de la inicialización en
cero:

$$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 $$

$$ \hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad
   \hat v_t = \frac{v_t}{1 - \beta_2^t}, \qquad
   \theta_t = \theta_{t-1} - \frac{\alpha \hat m_t}{\sqrt{\hat v_t} + \epsilon} $$

```text
# Algoritmo 1 del paper de Kingma & Ba (ICLR 2015) — operaciones element-wise
Requiere: α (stepsize), β1, β2 ∈ [0, 1), f(θ) estocástica, θ0 inicial
m0 ← 0        # primer momento (media del gradiente)
v0 ← 0        # segundo momento bruto (media del gradiente²)
t ← 0
mientras θ_t no converja:
    t ← t + 1
    g_t ← ∇θ f_t(θ_{t-1})                 # gradiente en el instante t
    m_t ← β1·m_{t-1} + (1 − β1)·g_t       # media móvil del gradiente
    v_t ← β2·v_{t-1} + (1 − β2)·g_t^2     # media móvil del gradiente²
    m̂_t ← m_t / (1 − β1^t)                # corrección de sesgo del 1er momento
    v̂_t ← v_t / (1 − β2^t)                # corrección de sesgo del 2º momento
    θ_t ← θ_{t-1} − α·m̂_t / (√v̂_t + ε)    # actualización
devuelve θ_t
```

Defaults del paper: $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$,
$\epsilon = 10^{-8}$; $\beta_1^t$ y $\beta_2^t$ denotan las potencias
$t$-ésimas. La corrección de sesgo importa en los primeros pasos y con
gradientes dispersos: sin ella, $m_t$ y $v_t$ parten de cero y subestiman los
momentos.

**AdamW (weight decay desacoplado)**: la penalización se resta directamente
del parámetro en vez de sumarse al gradiente (Loshchilov & Hutter 2017):

$$ \theta_t = \theta_{t-1} - \alpha \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
   + \lambda \theta_{t-1} \right) $$

En Adam, la $L_2$ acoplada se distorsiona: la normalización por $\sqrt{v_t}$
reescala la penalización por coordenada y hace que el $\lambda$ efectivo
dependa de $\alpha$ y de la historia de gradientes. El decay desacoplado
restaura un $\lambda$ con interpretación de regularización independiente del
optimizador y suele generalizar mejor.

{% if ml_type == 'redes_neuronales' %}
### Elección de optimizador en redes neuronales

Para `ml_type = redes_neuronales`, Adam con los defaults del paper es el
punto de partida robusto; AdamW si se usa weight decay (transformers, redes
grandes). En visión, SGD+Nesterov con momentum $\mu = 0.9$ y schedule cosine
suele alcanzar mejor test que Adam al final del entrenamiento. El tamaño de
batch fija la varianza del paso (ruido $\propto 1/\sqrt{b}$): al subir el
batch, sube también el LR (linear scaling rule) y acopla warmup. Nunca
comparemos dos optimizadores con el mismo número de épocas sin fijar también
el schedule: son hiperparámetros del mismo problema.
{% endif %}

## Newton y L-BFGS

Newton usa la curvatura:

$$ \theta_{t+1} = \theta_t - H(\theta_t)^{-1} \nabla f(\theta_t) $$

Es invariante a cambios de base lineales (elimina el efecto del número de
condición) y converge en **un paso** para cuadráticas; cerca del óptimo la
convergencia es cuadrática. Coste: $O(d^2)$ de memoria y $O(d^3)$ por paso
(resolver con Cholesky en vez de invertir); requiere $H$ definida positiva —
en zonas no convexas se regulariza ($H + \lambda I$, Levenberg–Marquardt).

**Cuándo la Hessiana paga**: problemas suaves con $d$ moderado, Hessiana SPD
y necesidad de precisión alta o de pocas iteraciones: MLE en familias
exponenciales, optimización de hyperparámetros sobre un GP, regresión
robusta. Ahí Newton (o cuasi-Newton) domina por iteración a los métodos de
primer orden. No paga en redes profundas: $d$ enorme, $H$ indefinida y
objetivo estocástico — una Hessiana de $10^7$ parámetros ni se forma.

**L-BFGS**: cuasi-Newton que aproxima $H^{-1}$ con los últimos $m$ pares de
gradientes ($m \approx 10$); memoria $O(md)$, convergencia superlineal, sin
matrices densas. Es el estándar para objetivos convexos suaves con suma
finita y batch completo o mini-batch grande: regresión logística y lineal
regularizadas, factorización de matrices. No es robusto con minibatches
pequeños (el gradiente ruidoso arruina la aproximación de curvatura) — para
eso están SGD/Adam.

| Situación | Método |
|---|---|
| $d$ moderado, $H$ SPD, precisión alta | Newton |
| Convexo suave, batch completo, $d \lesssim 10^6$ | L-BFGS |
| Objetivo estocástico, $d$ enorme, no convexo | SGD / Adam |

## Multiplicadores de Lagrange y KKT

Para $\min f(x)$ con restricciones de igualdad $g_i(x) = 0$, todo punto
estacionario satisface $\nabla f(x) = \sum_i \lambda_i \nabla g_i(x)$: el
gradiente de la función es combinación lineal de los gradientes de las
restricciones (la restricción "tira" tangente a las curvas de nivel).

Para desigualdades $g_i(x) \le 0$ y $h_j(x) = 0$, las condiciones **KKT**
(necesarias bajo regularidad de las restricciones) son:

1. **Factibilidad primal**: $g_i(x) \le 0$, $h_j(x) = 0$.
2. **Factibilidad dual**: $\lambda_i \ge 0$.
3. **Estacionariedad**:
   $\nabla f(x) + \sum_i \lambda_i \nabla g_i(x) + \sum_j \mu_j \nabla h_j(x) = 0$.
4. **Complementary slackness**: $\lambda_i g_i(x) = 0$ — si la restricción
   no está activa, su multiplicador es cero.

Si $f$ y las $g_i$ son convexas (y hay punto estrictamente factible,
Slater), KKT son **necesarias y suficientes**: la solución existe si y solo
si hay multiplicadores que las satisfacen. Es la maquinaria de los SVM (los
support vectors son las restricciones activas), del primal-dual y de los
métodos de penalización.

Abuso: en problemas no convexos (deep learning) un punto KKT es solo
candidato a estacionario — puede ser silla o máximo, y los multiplicadores no
prueban optimalidad global. Tampoco vale ignorar la factibilidad dual: un
"óptimo" con $\lambda_i < 0$ es un artefacto del solver, no una solución.

## Geometría de la regularización

Regularizar resuelve $\min_\beta \|y - X\beta\|_2^2 + \text{pen}(\beta)$.

**$L_2$ (ridge)**: $\text{pen}(\beta) = \lambda\|\beta\|_2^2$. Solución
cerrada $\hat\beta = (X^\top X + \lambda I)^{-1} X^\top y$: encoge todos los
coeficientes (factor uniforme) y **nunca anula ninguno**; añade $\lambda$ a
los autovalores de $X^\top X$ (mejora el condicionamiento). Equivale a un
prior gaussiano $\beta \sim \mathcal{N}(0, \sigma^2/\lambda \cdot I)$.

**$L_1$ (lasso)**: $\text{pen}(\beta) = \lambda\|\beta\|_1$. La bola
$\|\beta\|_1 \le t$ es un **poliedro con vértices sobre los ejes** (el
"rombo"): la solución cae en un vértice → coeficientes **exactamente cero** →
selección de variables. Con features correlacionadas el lasso elige
arbitrariamente una del grupo (y el resto a cero). No hay forma cerrada: se
resuelve con subgradiente o proximal (ver abajo).

**Elastic net**: $\lambda_1\|\beta\|_1 + \lambda_2\|\beta\|_2^2$. Mantiene la
esparsidad del $L_1$ y la estabilidad del $L_2$ con grupos de variables
correlacionadas (selecciona el grupo y luego encoge).

| Penalización | Bola | Efecto | Ceros exactos | Uso típico |
|---|---|---|---|---|
| $L_2$ | esfera | encoge | no | colinealidad, ridge |
| $L_1$ | rombo | esparsifica | sí | selección (lasso) |
| Elastic net | intermedia | grupo + sparse | sí | features correlacionadas |

**Operador proximal** (objetivos compuestos suave + no suave):

$$ \mathrm{prox}_{\lambda f}(z) = \arg\min_x \left( f(x) +
   \frac{1}{2\lambda}\|x - z\|_2^2 \right) $$

Para $f(x) = \|x\|_1$ es la **umbralización blanda** (soft-threshold),
$S_\lambda(z) = \mathrm{sign}(z)\max(|z| - \lambda, 0)$. El descenso
proximal (ISTA) alterna un paso de gradiente en la parte suave y un prox en
la no suave:

$$ \theta_{k+1} = \mathrm{prox}_{\alpha\lambda\|\cdot\|_1}
   \left( \theta_k - \alpha \nabla g(\theta_k) \right) $$

converge a $O(1/k)$ (FISTA lo acelera a $O(1/k^2)$). Abuso: aplicar gradiente
a secas a la $L_1$ (sin prox) oscila alrededor de los ceros; y una
penalización mal escalada anula señal real si las features no están
normalizadas.

## Trampas y abusos

- **Gradientes que desaparecen o explotan**: al componer muchas capas, la
  norma del gradiente se multiplica por productos de normas de Jacobianos;
  si el producto < 1, muere (las capas tempranas no aprenden); si > 1,
  explota. Mitigaciones: init Glorot/He, batch/layer norm, conexiones
  residuales, gradiente clipping (por norma global) y activaciones con
  derivada acotada. Síntoma clásico: las primeras capas "congeladas"
  mientras las últimas aprenden.
- **Puntos de silla**: en alta dimensión, los puntos estacionarios típicos
  de pérdidas no convexas son sillas con autovalores mixtos, no mínimos
  locales profundos (Dauphin et al.; Pascanu et al.). El gradiente es cero
  pero hay direcciones de curvatura negativa por las que escapar. SGD escapa
  gracias al ruido; las sillas con valle casi plano se confunden con
  convergencia. Diagnóstico: autovalores de la Hessiana. Remedios: ruido
  (SGD), momentum, o pasos en dirección de curvatura negativa.
- **Objetivos no suaves**: hinge, ReLU, $L_1$ y quantile loss no son
  diferenciables en ciertos puntos; el "gradiente" es un subgradiente
  (convergencia más lenta, sin garantías en el kink) y la verificación
  numérica falla exactamente en esos puntos. Para $L_1$ usar prox; para ReLU
  el gradiente de una rama es válido casi siempre.
- **Confundir "la pérdida bajó" con "convergió"**: con paso fijo, SGD fluctúa
  en una banda y los criterios de parada por cambio de pérdida engañan.
- **Comparar configuraciones sin fijar el schedule** o con distinto
  presupuesto de épocas: el ranking de optimizadores sale sesgado.

## Fuentes

- **Adam: A Method for Stochastic Optimization** — D. P. Kingma, J. Ba
  (ICLR 2015). arXiv:1412.6980 — https://arxiv.org/abs/1412.6980
- **Decoupled Weight Decay Regularization** — I. Loshchilov, F. Hutter
  (2017). arXiv:1711.05101 — https://arxiv.org/abs/1711.05101
- **Adaptive Subgradient Methods for Online Learning and Stochastic
  Optimization** — J. Duchi, E. Hazan, Y. Singer (2011).
  arXiv:1103.0377 — https://arxiv.org/abs/1103.0377
- **On the importance of initialization and momentum in deep learning** —
  I. Sutskever, J. Martens, G. Dahl, G. Hinton (2013).
  arXiv:1302.43820 — https://arxiv.org/abs/1302.43820
- **Optimization Methods for Large-Scale Machine Learning** — L. Bottou,
  F. E. Curtis, J. Nocedal (2018).
  arXiv:1606.04838 — https://arxiv.org/abs/1606.04838
- **An overview of gradient descent optimization algorithms** — S. Ruder
  (2016). arXiv:1609.04747 — https://arxiv.org/abs/1609.04747
- **On the Convergence of Adam and Beyond** — S. Reddi, S. Kale, S. Kumar
  (ICLR 2018). arXiv:1904.09237 — https://arxiv.org/abs/1904.09237
- **Identifying and attacking the saddle point problem in high-dimensional
  non-convex optimization** — Y. N. Dauphin et al. (2014).
  arXiv:1406.2572 — https://arxiv.org/abs/1406.2572
- **On the saddle point problem for non-convex optimization** — R. Pascanu,
  Y. N. Dauphin, S. Ganguli, Y. Bengio (2014).
  arXiv:1405.4604 — https://arxiv.org/abs/1405.4604
- **On the difficulty of training recurrent neural networks** — R. Pascanu,
  T. Mikolov, Y. Bengio (2013).
  arXiv:1211.5063 — https://arxiv.org/abs/1211.5063
- S. Boyd, L. Vandenberghe, *Convex Optimization*, Cambridge Univ. Press,
  2004. Sin arXiv — https://web.stanford.edu/~boyd/cvxbook/
