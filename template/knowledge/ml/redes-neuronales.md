{% if ml_type == 'redes_neuronales' or ml_type == 'hibrido' %}
# Redes neuronales profundas

Referencia densa de los mecanismos que sostienen una red entrenada con
backpropagation: el paso de gradientes, las activaciones, la pérdida, la
inicialización, la normalización, los optimizadores y las familias de
arquitecturas que usa este proyecto según `nn_model`.

## Forward pass y backpropagation

La red es un grafo computacional acíclico. El forward evalúa cada nodo en
orden topológico; el backward distribuye el gradiente de la pérdida escalar
con la regla de la cadena en orden inverso. Cada nodo local recibe el gradiente
de su salida respecto a la pérdida y emite: (a) el gradiente respecto a sus
entradas y (b) el gradiente respecto a sus parámetros.

### MLP de 2 capas, paso exacto

Notación: $x \in \mathbb{R}^d$, oculta $h = \phi(z_1)$ con
$z_1 = W_1 x + b_1$, $W_1 \in \mathbb{R}^{m \times d}$; salida
$\hat{y} = W_2 h + b_2$, $W_2 \in \mathbb{R}^{c \times m}$; pérdida escalar
$L(\hat{y}, y)$.

1. Forward: $z_1 = W_1 x + b_1$; $h = \phi(z_1)$; $\hat{y} = W_2 h + b_2$.
2. $\delta_2 = \nabla_{\hat{y}} L$ (vector $c$-dimensional).
3. Gradientes de la capa de salida:
   $\nabla_{W_2} L = \delta_2 h^\top$, $\nabla_{b_2} L = \delta_2$.
4. Propagación a la oculta:
   $\delta_1 = (W_2^\top \delta_2) \odot \phi'(z_1)$.
5. Gradientes de la capa oculta:
   $\nabla_{W_1} L = \delta_1 x^\top$, $\nabla_{b_1} L = \delta_1$.

Un paso de SGD aplica $W_i \leftarrow W_i - \eta \nabla_{W_i} L$ con el
$\eta$ (learning rate) que elija el optimizador. El punto 4 es la regla de la
cadena: el gradiente que llega a la capa anterior es el de la capa siguiente
por la transpuesta de la matriz de pesos y por la derivada local de la
activación ($\odot$ = producto elementwise).

### Por qué backprop es eficiente

Backprop es **modo reverso de diferenciación automática**: una pasada forward
guarda los valores intermedios y una pasada reverse acumula
$\bar{z} = \partial L / \partial z$ en cada nodo. El coste total es unas 2-3
veces el del forward, independientemente del número de parámetros. El modo
directo costaría una pasada completa por parámetro (o por salida), inviable
para millones de pesos. La misma operación sobre la misma gráfica, pero con la
pérdida como entrada y las salidas como parámetros, da todas las parciales en
un solo recorrido.

### Desvanecimiento y explosión del gradiente

El gradiente hacia una capa temprana es un **producto de matrices jacobianas**
de las capas intermedias:

$$
\frac{\partial L}{\partial W_1} \sim \prod_{l} J_l, \qquad
J_l = \frac{\partial h_{l}}{\partial h_{l-1}}
$$

Si el radio espectral de los $J_l$ es < 1, el producto decae exponencialmente
(vanishing); si es > 1, crece exponencialmente (exploding). Los activaciones
saturantes lo empeoran: sigmoid tiene derivada máxima 0.25, así que cada capa
multiplica el gradiente por ≤ 0.25. Las consecuencias prácticas: capas
tempranas no aprenden (vanishing) o divergen (exploding), y son la razón de
existir de inicialización cuidada, BatchNorm, skip connections y optimizadores
adaptativos.

## Activaciones

| Activación | $f(z)$ | $f'(z)$ | Notas |
|---|---|---|---|
| Sigmoid | $\dfrac{1}{1+e^{-z}}$ | $f(z)(1-f(z)) \leq 0.25$ | satura en ambos extremos; no centrada en cero |
| tanh | $\dfrac{e^z-e^{-z}}{e^z+e^{-z}}$ | $1 - f(z)^2$ | centrada; satura; gradiente máx 1 |
| ReLU | $\max(0, z)$ | $1$ si $z>0$, $0$ si $z<0$ | barata; neuronas muertas |
| LeakyReLU | $\max(\alpha z, z)$, $\alpha \approx .01$ | $\alpha$ si $z<0$; $1$ si $z>0$ | evita muertas |
| ELU | $z$ si $z>0$; $\alpha(e^z-1)$ si $z<0$ | $1$; $f(z)+\alpha$ | suave, negativa, satura |
| GELU | $z\,\Phi(z)$ | $\Phi(z) + z\,\phi(z)$ | suave; estándar en transformers |
| Softmax | $e^{z_i}/\sum_j e^{z_j}$ | $\sigma_i(\delta_{ij}-\sigma_j)$ | vectorial; Jacobiana propia |

- **Saturación**: fuera de la región casi-lineal, la derivada se aplana y el
  gradiente muere. Sigmoid/tanh saturan rápido con pesos grandes.
- **Neuronas muertas (ReLU)**: si una unidad queda en $z<0$ para todos los
  datos, su gradiente es 0 siempre y nunca se recupera. Causas: init con bias
  negativa, LR alto, gradientes explosivos. LeakyReLU/ELU existen por esto.
- **Softmax** no es una activación de capa: convierte un vector en una
  distribución de probabilidad sobre clases. Su gradiente es la Jacobiana
  $\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j)$,
  no diagonal.

Regla: ReLU (o variantes) en ocultas salvo que el problema pida rangos acotados;
softmax en la cabeza de clasificación; sigmoid en la última capa de clasificación
binaria (junto con BCEWithLogits). GELU donde el modelo ya lo traiga (transformer).

## Funciones de pérdida

Para clasificación multiclase, cross-entropy sobre las logits:

$$
L = -\sum_{c} y_c \log p_c, \qquad p = \operatorname{softmax}(z)
$$

Para regresión, MSE (o MAE/L1 para robustez frente a outliers). Para binaria
con logits, BCEWithLogits combina sigmoid + BCE en un solo op estable
numéricamente:

$$
L = -\big(y \log\sigma(z) + (1-y)\log(1-\sigma(z))\big)
$$

### Por qué softmax + cross-entropy combinan limpio

La derivada de la composición cancela el denominador de softmax:

$$
\frac{\partial L}{\partial z_i} = p_i - y_i
$$

Es el gradiente más limpio de la red: *predicción menos objetivo*. MSE sobre
softmax no cancela nada, produce un gradiente con la Jacobiana de softmax
entre medias, satura cuando $p_i \to y_i$ y penaliza relativamente poco los
errores grandes (la pérdida está acotada en $[0,1]$). La práctica estándar es
cruzar logits sin normalizar con cross-entropy; el propio softmax va en la
función de pérdida, nunca precomputado aparte.

### Label smoothing

Sustituye el one-hot por
$y'_c = (1-\varepsilon)y_c + \varepsilon/K$ con $K$ clases. Efectos: evita que
el modelo persiga probabilidades 1.0 (logits infinitos, confianza
descalibrada), mejora generalización y calibración, y suaviza la superficie de
pérdida. Típico $\varepsilon = 0.1$.

{% if nn_loss_fn == 'Auto' %}
Este proyecto deja `nn_loss_fn = Auto`: la pérdida se infiere del tipo de
etiqueta (CrossEntropy para clasificación, MSELoss para regresión).
{% endif %}
{% if nn_loss_fn in ['CrossEntropyLoss', 'BCEWithLogitsLoss'] %}
Este proyecto usa `nn_loss_fn = {{ nn_loss_fn }}`, consistente con el
gradiente "predicción menos objetivo" de arriba.
{% endif %}

## Inicialización de pesos

### Por qué la inicialización simétrica falla

Si todos los pesos valen lo mismo, todas las unidades de una capa reciben la
misma preactivación, computan la misma función y reciben **el mismo gradiente**:
la simetría nunca se rompe y la capa se comporta como una sola unidad. Además,
la escala importa: pesos muy grandes saturan activaciones (gradiente muerto);
muy pequeños hacen que los productos del forward se colapsen a 0 (gradiente
que nunca llega).

### El argumento de varianza (Xavier/Glorot)

Para una capa lineal $z_i = \sum_j w_{ij} x_j$ con $E[x]=0$ y pesos
independientes de las entradas:

$$
\operatorname{Var}(z_i) = n_{in} \cdot \operatorname{Var}(w) \cdot
\operatorname{Var}(x)
$$

Para que la varianza de la señal **se conserve** entre capas se necesita
$\operatorname{Var}(w) = 1/n_{in}$ (fan-in). Glorot & Bengio (2010) exigen que
se conserven a la vez forward y backward, lo que promedia con el fan-out:

$$
\operatorname{Var}(w) = \frac{2}{n_{in} + n_{out}}
$$

En la práctica uniforme con $\pm\sqrt{6/(n_{in}+n_{out})}$ (o normal con esa
varianza).

### He (para ReLU)

ReLU descarta la mitad de las unidades, lo que divide la varianza por 2; para
compensarlo se dobla:

$$
\operatorname{Var}(w) = \frac{2}{n_{in}}
$$

Es decir, normal $N(0, 2/n_{in})$ o uniforme $\pm\sqrt{6/n_{in}}$. Regla: He
con ReLU y variantes; Xavier/Glorot con tanh/sigmoid; la última capa lineal de
clasificación admite valores más pequeños o el mismo esquema.

## Normalización y dropout

### BatchNorm

Normaliza cada canal **a través del batch**: para cada feature calcula la media
y varianza del minibatch y aplica

$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}},
\qquad y = \gamma \hat{x} + \beta
$$

con $\gamma, \beta$ aprendibles. Diferencia train/eval: en train $\mu_B,
\sigma_B^2$ son los del minibatch actual (y el ruido resultante actúa de
regularizador); en eval se usan **medias móviles acumuladas durante el
entrenamiento**, sin dependencia del batch. Beneficios: permite learning rates
más altos, reduce la sensibilidad a la inicialización, estabiliza la
distribución de preactivaciones. Coste: en eval depende de las estadísticas
acumuladas; con batches pequeños o datos que cambian de distribución degrada.
Requiere un minibatch razonable; no funciona bien con batch size 1.

### LayerNorm

Normaliza por **muestra a lo largo de la dimensión de features**
(per-token en transformers):

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}},
\qquad \mu = \frac{1}{D}\sum_j x_j, \quad
\sigma^2 = \frac{1}{D}\sum_j (x_j - \mu)^2
$$

con $\gamma, \beta$ por feature. No depende del batch, así que vale con batch
size 1, es invariante a la longitud de secuencia y no arrastra estadísticas
acumuladas. Es la normalización estándar en transformers.

### Dropout (inverted)

En train, cada unidad se **apaga con probabilidad $q$** (keep prob $p=1-q$) y
las supervivientes se escalan por $1/p$:

$$
\tilde{h}_i = \begin{cases} h_i / p & \text{con prob } p \\ 0 & \text{con prob } q \end{cases}
$$

Por eso se llama *inverted dropout*: el escalado ocurre en train, así que en
eval la red corre sin dropout y **sin corrección**. Mecanismo: impide que las
unidades co-adapten (cada unidad debe ser útil sin depender de otras), fuerza
representaciones redundantes y equivale a promediar un conjunto de sub-redes
(muestreo de arquitectura). Se aplica típicamente 0.1-0.5 en capas densas y
menos en conv.

## Optimizadores

El proyecto elige el optimizador con `optimizer_type`
(`{{ optimizer_type }}`). Actualización completa de cada familia:

### SGD + momentum (heavy ball)

$$
v_{t+1} = \mu v_t + g_t, \qquad
\theta_{t+1} = \theta_t - \eta v_{t+1}
$$

con $g_t = \nabla_\theta L(\theta_t)$ y $\mu \approx 0.9$. El momentum acumula
la dirección de los gradientes pasados, suaviza el ruido estocástico del
minibatch y acelera en valles de pendiente baja pero consistente. Nesterov
evalúa el gradiente en el punto adelantado $\theta_t - \eta \mu v_t$.

### RMSProp

$$
v_{t+1} = \rho v_t + (1-\rho) g_t^2, \qquad
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_{t+1}} + \varepsilon} g_t
$$

Mantiene un promedio exponencial del cuadrado del gradiente ($\rho \approx
0.9$—0.99) y escala el paso por su raíz: coordenadas con gradiente grande
avanzan menos. Bueno en problemas no estacionarios y secuenciales.

### Adam — Algoritmo 1 (Kingma & Ba, ICLR 2015)

Adaptive moment estimation: momentos primero y segundo con corrección de sesgo.
Defaults del paper: $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$,
$\varepsilon = 10^{-8}$.

```
Algoritmo 1: Adam (del paper arXiv:1412.6980)
Requiere: α (stepsize); β1, β2 ∈ [0,1): tasas de decaimiento exponencial de
  los momentos; f(θ): función objetivo estocástica; θ0: parámetro inicial.
m0 ← 0                      # 1er vector de momento
v0 ← 0                      # 2º vector de momento (momento crudo)
t ← 0                       # paso de tiempo
mientras θ_t no converja:
    t ← t + 1
    g_t ← ∇_θ f_t(θ_{t−1})                       # gradiente en el paso t
    m_t ← β1 · m_{t−1} + (1 − β1) · g_t           # momento 1º sesgado
    v_t ← β2 · v_{t−1} + (1 − β2) · g_t²          # momento 2º sesgado
    m̂_t ← m_t / (1 − β1^t)                       # corrección de sesgo
    v̂_t ← v_t / (1 − β2^t)                       # corrección de sesgo
    θ_t ← θ_{t−1} − α · m̂_t / (√v̂_t + ε)         # actualización
devuelve θ_t
```

Propiedades clave: la magnitud de la actualización es invariante a re-escalar
el gradiente, el paso está acotado aproximadamente por $\alpha$, trabaja bien
con gradientes dispersos y hace annealing de la tasa por su cuenta.

### AdamW — weight decay desacoplado

El weight decay clásico suma $\lambda \theta$ al gradiente; en Adam eso divide
el término por $\sqrt{\hat{v}} + \varepsilon$, de modo que el decay queda
escalado por el mismo precondicionador que domina las coordenadas con gradiente
ruidoso. AdamW aplica el decay **directamente a los parámetros**, fuera del
paso adaptativo:

$$
\theta_t \leftarrow \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}
+ \varepsilon} - \lambda \theta_{t-1}
$$

(Loshchilov & Hutter, arXiv:1711.05101). Más simple y empíricamente mejor que
L2-in-Adam; es la variante por defecto en la mayoría de frameworks modernos.

### Programas de learning rate

- **Step**: $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$, multiplicar
  por $\gamma$ cada $s$ epochs. Simple, hay que calibrar los escalones.
- **Cosine**: $\eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})
  (1 + \cos(\pi t/T))$ con $T$ el número total de pasos. Decae suave hasta
  $\eta_{\min}$; suele acabar mejor que step en el mismo número de epochs.
- **Warmup**: rampa lineal de ~0 a $\eta_{\max}$ en los primeros $W$ pasos
  (típico 5-10% del total), seguida de cosine. Necesario en transformers y con
  Adam: al inicio los momentos son ruidosos y un paso grande desestabiliza
  BatchNorm/LayerNorm.

### Precisión mixta (fp16/fp32) y AMP

Los pesos maestros viven en fp32; forward y backward corren en fp16; el
gradiente se escala por un factor $s$ antes del backward para evitar underflow
de fp16 (rango dinámico corto), y se des-escala al actualizar los pesos
maestros fp32. AMP (automatic mixed precision, `torch.cuda.amp` /
`torch.autocast`) decide automáticamente qué operaciones van en fp16 y qué
mantener en fp32, con loss scaling dinámico. Ganancia: ~1.5-2x en GPUs con
tensor cores y la mitad de memoria de activaciones. Requiere normalización
(loss scaling) y cuidado con reducciones de precisión.

### Acumulación de gradientes

Sumar gradientes de $K$ minibatches y aplicar el paso del optimizador una sola
vez, dividiendo por $K$ (o escalando la pérdida por $1/K$):

```
grad_acum = 0
para i en 1..K:
    loss = L(batch_i) / K
    loss.backward()          # acumula en grad_acum
optimizer.step()             # un solo paso cada K minibatches
optimizer.zero_grad()
```

Simula un batch efectivo de $K \times N$ sin pagar su memoria (las activaciones
se liberan por minibatch). El LR debe pensarse para el batch efectivo; los
framework modernos suelen recalcular con la regla de scaling lineal.

## Regularización

| Técnica | Qué hace | Coste |
|---------|----------|-------|
| Weight decay (L2) | empuja los pesos a 0; penaliza complejidad | solo $\lambda$ nuevo |
| Dropout | apaga unidades; anti co-adaptación | $q$ a calibrar; eval sin cambio |
| Early stopping | vigila la métrica de validación, guarda el mejor modelo | paciencia |
| Data augmentation | genera variantes realistas del input | dominio-dependiente |
| Ruido | en entradas o etiquetas; regulariza (ej. label noise ~0.05) | ajustar magnitud |

El objetivo común es reducir la **brecha train/validación**: aumentar la
capacidad solo la empeora si el modelo sobreajusta. Las técnicas se combinan:
AdamW ya trae weight decay desacoplado; dropout suele ir en las capas densas;
early stopping es la red de seguridad universal.

## Arquitecturas

{% if nn_model == 'MLP' %}
### MLP

Perceptrón multicapa: capas densas con activación no lineal. **Cuándo**: datos
estructurados/tabulares de tamaño moderado, secuencias cortas ya vectorizadas,
o como baseline contra el que comparar modelos más complejos.

- **Profundidad vs anchura**: con el mismo número de parámetros, más capas
  componen funciones de orden superior (jerárquicas) y suelen generalizar
  mejor; más anchura expresa más features por nivel. La profundidad se paga en
  dificultad de optimización (por eso existen init, norm y skip).
- **Teorema de aproximación universal** (statement): una capa oculta con
  suficientes unidades y una activación no lineal puede aproximar cualquier
  función continua sobre un compacto con precisión arbitraria. No dice nada de
  la anchura requerida, ni de la aprendibilidad, ni de la generalización.

Para MLP vale todo lo de arriba: He init, BatchNorm/LayerNorm, AdamW, dropout,
early stopping.
{% endif %}
{% if nn_model in ['CNN1D', 'ResNet'] %}
### CNN1D (y ResNet)

Convolución 1D sobre secuencias: un kernel de tamaño $k$ se desliza con stride
$s$ y padding $p$; cada posición de salida es un producto escalar entre el
kernel y una ventana. Tamaño de salida para secuencia de longitud $L$:

$$
L_{out} = \left\lfloor \frac{L + 2p - d(k-1) - 1}{s} \right\rfloor + 1
$$

($d$ = dilation; con $d=1$: $\lfloor (L + 2p - k)/s \rfloor + 1$). Se usa el
mismo kernel en todas las posiciones (weight sharing) y un kernel por canal de
entrada/salida.

- **Equivarianza traslacional**: desplazar la entrada desplaza la salida, pero
  la operación es la misma en todas las posiciones. Por eso una conv detecta un
  patrón (pico, motif, forma local) sin importar dónde aparezca — el supuesto
  correcto para señales, audio, texto y series temporales.
- **Campo receptivo**: tras $l$ capas conv con stride $s_j$ y kernel $k_j$,

$$
R = 1 + \sum_{l} (k_l - 1) \prod_{j<l} s_j
$$

  crece más rápido con capas conv que con stride. Dilación aumenta el campo sin
  coste de parámetros.
- **Pooling**: reduce dimensión, aumenta el campo receptivo y añade invariancia
  local (max-pooling toma el máximo de una ventana; para 1D típico k=2, s=2).
  Útil en capas tempranas; en la práctica moderna las conv con stride
  sustituyen a veces al pooling.

### ResNet y el bloque residual

El bloque residual del paper (He et al., arXiv:1512.03385) define, con
$F(x, \{W_i\})$ la pila de capas conv + ReLU,

$$
y = F(x, \{W_i\}) + x
$$

La **shortcut de identidad** ($+x$) es sin parámetros (se usan proyecciones
solo al cambiar dimensiones). Por qué funciona: el paper observa un **problema
de degradación** — redes llanas más profundas tienen *mayor error de
entrenamiento*, y demuestra que no es overfitting (el error de train sube). Si
las capas añadidas debieran comportarse como la identidad, a un solucionador
le resulta más fácil **empujar el residual a cero** que aprender a copiar la
entrada con una pila de no-linealidades; y el gradiente fluye directamente por
la shortcut sin atravesar la pila, lo que mitiga el vanishing gradient en redes
muy profundas (100+ capas).

Convolución + residual + BatchNorm (conv → BN → ReLU → conv → BN → `+x` →
ReLU) es el patrón que hizo factibles las redes profundas de visión y de señal.
{% endif %}
{% if nn_model in ['LSTM', 'GRU'] %}
### LSTM y GRU

La RNN vanilla $h_t = \tanh(W x_t + U h_{t-1})$ sufre vanishing gradient
recurrente: el gradiente respecto a $h_0$ es un producto de $U^\top \cdot
\operatorname{diag}(\tanh')$ por paso; con radio espectral < 1 el gradiente
muere y la red no aprende dependencias largas. Las celdas con compuertas
(gated) lo arreglan con una **vía aditiva** que deja pasar el estado.

LSTM (Long Short-Term Memory), con $\odot$ producto elementwise y
$\sigma$ = sigmoid:

$$
\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) \\
\tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

Las compuertas **forget** $f_t$, **input** $i_t$ y **output** $o_t$ están en
$[0,1]$ y modulan cuánto del estado pasado se conserva, cuánta información
nueva entra y qué se expone a la salida. La actualización de la celda $c_t$ es
una **combinación convexa** (suma ponderada), no un producto: el gradiente
atraviesa la línea $c_{t-1} \to c_t$ aditivamente y solo se modula por
$f_t \in (0,1)$, lo que controla el vanishing.

GRU simplifica a dos compuertas (update y reset), sin celda separada:

$$
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) \\
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) \\
\tilde{h}_t &= \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$

**Cuándo son el problema**: cuando la señal es secuencial y el orden importa —
series temporales, audio, texto como secuencia. En la práctica moderna, para
secuencias largas y datasets grandes los transformers los superan; las RNN
gated siguen siendo buenas con datos escasos, secuencias cortas o latencia
estricta (decodificación paso a paso sin atender a toda la secuencia).
{% endif %}
{% if nn_model == 'Transformer' %}
### Transformer

#### Self-attention (scaled dot-product)

Sobre una secuencia, cada posición genera **Q** (query), **K** (key) y **V**
(value) por proyecciones lineales aprendidas $Q = XW^Q$,
$K = XW^K$, $V = XW^V$. La atención del paper (Vaswani et al.,
arXiv:1706.03762) es:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}} \right) V
$$

$QK^\top$ son productos escalares entre queries y keys (compatibilidad); la
división por $\sqrt{d_k}$ evita que los productos escalares crezcan con $d_k$
y empujen softmax a regiones de gradiente casi nulo (con componentes de media
0 y varianza 1, el producto escalar tiene varianza $d_k$); softmax convierte
las compatibilidades en pesos y se pondera $V$.

#### Multi-head

No una atención sino $h$ en paralelo, cada una en un subespacio de menor
dimensión:

$$
\operatorname{MultiHead}(Q, K, V) = \operatorname{Concat}(
\operatorname{head}_1, \dots, \operatorname{head}_h) W^O
$$

con $\operatorname{head}_i = \operatorname{Attention}(QW_i^Q, KW_i^K,
VW_i^V)$, proyecciones $W_i^Q, W_i^K \in \mathbb{R}^{d_{model} \times d_k}$,
$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, $W^O \in
\mathbb{R}^{h d_v \times d_{model}}$. Cada cabeza puede atender a relaciones
distintas (sintaxis, co-referencia, posición) que una sola atención promedia
perdería. Con $d_k = d_v = d_{model}/h$, el coste total es similar al de una
atención de dimensión completa.

#### Positional encoding

La atención es invariante a permutaciones: sin señal de posición, "el gato
come al pez" y "el pez come al gato" dan lo mismo. Se inyectan sinusoides fijas
(o embeddings aprendidos), sumadas a las entradas:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

#### Por qué atención arregla el largo alcance, y a qué precio

Cualquier par de posiciones se conecta con **un solo paso** de atención (path
length O(1)), sin producto recurrente ni vanishing gradient; además es
paralelizable sobre toda la secuencia. El precio es cuadrático en $n$: cada
query atiende a las $n$ keys, así que tiempo y memoria son $O(n^2 \cdot d)$
(tabla $n \times n$ de compatibilidades). Para secuencias largas esto domina y
motiva ventanas, attention esparsa o linealizaciones.

#### Capa: LayerNorm + residual

Cada sub-capa (atención y FFN) se envuelve como
$y = \operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))$ (Pre-LN, el
patrón moderno) o $\operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))$
(Post-LN, el del paper original). La shortcut residual es lo que permite el
gradiente fluir a través de decenas de capas; LayerNorm estabiliza las
preactivaciones por muestra sin depender del batch.
{% endif %}

## Dinámica de entrenamiento

- **Diagnóstico de overfitting**: se compara train vs validación *en la misma
  escala*. Si train baja y validación se estanca o sube mientras crece la
  brecha → overfitting (más regularización, más datos, menos capacidad). Si
  ambas suben o se estancan → underfitting (más capacidad, más epochs, mejor
  init). La brecha es el dato, no el valor absoluto de la métrica.
- **Learning-rate finding** (LR range test, Smith 2015): en unos cientos de
  pasos, subir $\eta$ exponencialmente y anotar la pérdida; el $\eta$ óptimo
  está en la región de mayor pendiente de descenso, justo antes de que la
  pérdida diverja. Sin este paso, cualquier conclusión sobre arquitectura o
  regularización es sospechosa.
- **Batches pequeños como regularizador**: el gradiente del minibatch es un
  estimador ruidoso del gradiente real; ese ruido actúa como regularización y
  ayuda a escapar de mínimos agudos. Por eso "batch pequeño ≈ más
  regularización" y batches muy grandes tienden a generalizar peor a igual
  capacity (o requieren ajustar LR/regularización).
- **Sensibilidad a la semilla y reproducibilidad**: el entrenamiento es
  estocástico (init, orden de datos, dropout). Una semilla distinta cambia el
  resultado final. Para reproducir: fijar `seed` en torch/numpy/random, forzar
  determinismo (`torch.use_deterministic_algorithms`), fijar el número de
  workers de DataLoader y registrar semilla + versión de librerías con cada
  run. Comparar modelos es comparar distribuciones sobre semillas, no una
  carrera single-run.

## Práctica

- **GPU: memoria vs batch size**: las activaciones de cada capa se guardan para
  el backward, así que la memoria crece con el batch size y con la profundidad
  (y con precisión: fp16 ≈ mitad). Si no cabe un batch grande: acumular
  gradientes, bajar a fp16/AMP, o reducir la secuencia. La utilización de GPU
  sube con el batch, pero el LR y la regularización se reajustan.
- **Checkpoints**: guardar periódicamente `state_dict` del modelo + estado del
  optimizador + epoch + mejor métrica; permitir reanudar exactamente el
  entrenamiento. Guardar siempre el mejor modelo por métrica de validación,
  no el último.
- **Precisión mixta**: primera palanca de velocidad/memoria en GPU moderna;
  activar AMP y medir, no asumir.
- **Cuándo el DL NO merece la pena**: datos tabulares pequeños o medianos con
  miles de filas → los árboles con boosting (XGBoost, LightGBM, CatBoost)
  ganan con una fracción del esfuerzo y del coste de cómputo. El DL rinde
  cuando hay mucha data, estructura espacial/secuencial/gráfica, o cuando un
  modelo preentrenado transfiere (texto, imagen, audio). Antes de montar una
  red, hay que perder contra un GBDT con features bien hechas y contra el
  baseline del proyecto.

## Fuentes

- Kingma, D. & Ba, J., "Adam: A Method for Stochastic Optimization".
  arXiv:1412.6980. https://arxiv.org/abs/1412.6980
- Vaswani, A. et al., "Attention Is All You Need". arXiv:1706.03762.
  https://arxiv.org/abs/1706.03762
- He, K. et al., "Deep Residual Learning for Image Recognition".
  arXiv:1512.03385. https://arxiv.org/abs/1512.03385
- Hochreiter, S. & Schmidhuber, J., "Long Short-Term Memory". arXiv:1503.04069.
  https://arxiv.org/abs/1503.04069
- Ioffe, S. & Szegedy, C., "Batch Normalization: Accelerating Deep Network
  Training by Reducing Internal Covariate Shift". arXiv:1502.03167.
  https://arxiv.org/abs/1502.03167
- Srivastava, N. et al., "Dropout: A Simple Way to Prevent Neural Networks from
  Overfitting". arXiv:1207.0580. https://arxiv.org/abs/1207.0580
- He, K. et al., "Delving Deep into Rectifiers". arXiv:1502.01852.
  https://arxiv.org/abs/1502.01852
- Glorot, X. & Bengio, Y., "Understanding the difficulty of training deep
  feedforward neural networks". AISTATS 2010.
  https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
- Loshchilov, I. & Hutter, F., "Decoupled Weight Decay Regularization".
  arXiv:1711.05101. https://arxiv.org/abs/1711.05101
- Smith, L. N., "Cyclical Learning Rates for Training Neural Networks".
  arXiv:1506.01186. https://arxiv.org/abs/1506.01186
{% endif %}
