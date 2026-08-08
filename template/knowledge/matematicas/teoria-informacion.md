# Teoría de la información

Cuantifica incertidumbre y dependencia estadística con un lenguaje que las
correlaciones clásicas no ven. Es la maquinaria exacta detrás de: el criterio de
split de un árbol, por qué log-loss es la pérdida canónica de clasificación, la
selección de features sin supuestos de linealidad, la perplejidad de un LM y la
compresión. Este fichero va de la definición a la derivación y de ahí a cuándo
funciona y cuándo engaña; la referencia canónica es Cover & Thomas, y MacKay
trata lo mismo con énfasis en inferencia.

## Entropía de Shannon

La entropía de una variable discreta $X \sim p$ es

$$H(X) = -\sum_{x \in \mathcal{X}} p(x)\log p(x) = \mathbb{E}[-\log p(X)].$$

La base del logaritmo fija la unidad: base 2 → bits, base $e$ → nats, base 10 →
dits. Un bit equivale a $\ln 2 \approx 0.693$ nats; la conversión es
$H_{\text{bits}} = H_{\text{nats}} / \ln 2$.

Tres lecturas equivalentes:

- **Incertidumbre:** el número esperado de preguntas binarias bien diseñadas
  para determinar $x$. $H = 0$ si $X$ es determinista; $H = \log|\mathcal{X}|$
  si es uniforme (máximo).
- **Longitud de código óptima:** la longitud esperada de un código de prefijo
  óptimo (Huffman) cumple $H(X) \le \bar L < H(X) + 1$. La entropía es el
  límite inferior fundamental de la compresión sin pérdidas (teorema de
  codificación de fuente).
- **Sorpresa promedio:** el evento $x$ cuesta $-\log p(x)$; la entropía es su
  esperanza. Es el origen de log-loss.

**Caso Bernoulli:** con $p(X=1) = p$,

$$H_b(p) = -p\log p - (1-p)\log(1-p),$$

es cóncava, simétrica en $p \leftrightarrow 1-p$, máxima en $p = 1/2$ (valor
1 bit) y nula en $p \in \{0, 1\}$. La concavidad es el motor de casi todo lo
que sigue: promediar distribuciones no puede aumentar la entropía. En un
proyecto DS aparece directamente en la ganancia de un split binario y en la
entropía de una respuesta 0/1.

```python
import numpy as np

def hb(p):  # entropía de Bernoulli en bits
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
```

## Entropía conjunta y condicional

$$H(X,Y) = -\sum_{x,y} p(x,y)\log p(x,y), \qquad
H(Y \mid X) = \sum_x p(x)\,H(Y \mid X = x).$$

**Regla de la cadena:**

$$H(X,Y) = H(X) + H(Y \mid X).$$

La incertidumbre conjunta se descompone en la de $X$ más la de $Y$ dado que ya
sabemos $X$. Se generaliza a $H(X_1,\dots,X_n) = \sum_i H(X_i \mid X_{<i})$.

**Condicionar nunca aumenta la entropía:**

$$H(Y \mid X) \le H(Y),$$

con igualdad si y solo si $X$ y $Y$ son independientes. Es consecuencia directa
de la concavidad; operativamente, información adicional no puede aumentar la
incertidumbre. Advertencia para debugging: con datos finitos, la $H(Y \mid X)$
muestral cae siempre que añades variables aunque no tengan señal. El resultado
asintótico no protege contra el sobreajuste — es exactamente la razón por la que
un split con demasiadas categorías "parece" perfecto (véase árboles).

## Divergencia KL

La divergencia KL (o entropía relativa) mide cuánta información se pierde al
usar $q$ para aproximar la verdadera $p$:

$$D_{KL}(p \| q) = \sum_x p(x)\log\frac{p(x)}{q(x)}.$$

**No es una métrica.** No es simétrica ($D_{KL}(p\|q) \ne D_{KL}(q\|p)$ en
general) ni satisface la desigualdad triangular. Es una medida *orientada*, y la
dirección importa: minimizar $D_{KL}(p\|q)$ sobre $q$ tiende a cubrir todo el
soporte de $p$ (moment matching en la familia exponencial); minimizar
$D_{KL}(q\|p)$ concentra $q$ en los modos de $p$ (mode seeking). Elegir la
dirección equivocada produce modelos con comportamiento opuesto.

**Desigualdad de Gibbs:** $D_{KL}(p\|q) \ge 0$, con igualdad si y solo si
$p = q$, por Jensen sobre el logaritmo. Requiere que $p$ sea absolutamente
continua respecto a $q$: si $p(x) > 0$ donde $q(x) = 0$, la divergencia es
infinita (véase Trampas).

**Entropía cruzada:**

$$H(p,q) = -\sum_x p(x)\log q(x) = H(p) + D_{KL}(p \| q).$$

Minimizar la cross-entropía respecto a $q$ es minimizar la KL, porque $H(p)$ no
depende de $q$. Es el origen de log-loss como criterio de clasificación.

## Información mutua

$$I(X;Y) = H(X) - H(X \mid Y) = H(X) + H(Y) - H(X,Y)
= D_{KL}\big(p(x,y) \,\|\, p(x)p(y)\big).$$

Mide cuánto reduce conocer $Y$ la incertidumbre sobre $X$; es simétrica.
Propiedades que importan en la práctica:

- $I(X;Y) \ge 0$; $I(X;Y) = 0$ **si y solo si** $X$ y $Y$ son independientes.
- **Invariante a transformaciones invertibles:** $I(f(X); g(Y)) = I(X;Y)$ con
  $f, g$ biyectivas. Las correlaciones no la tienen: Pearson se destruye con un
  reescalado no lineal, la MI no.
- **Desigualdad de procesamiento de datos:** si $X \to Y \to Z$ es una cadena de
  Markov, $I(X;Z) \le I(X;Y)$. Ningún post-procesado de $Y$ añade información
  sobre $X$; no se "recupera" lo perdido. Recuerda que una pipeline de features
  no crea información, solo la preserva o la destruye.
- **Caso gaussiano bivariado** con correlación $\rho$: $I = -\frac12
  \log(1-\rho^2)$ nats. Para $|\rho|$ pequeño, $I \approx \rho^2/2$ — ahí está
  el vínculo con la correlación; pero la MI captura además dependencias no
  lineales.

**MI vs correlación.** Pearson mide dependencia lineal y Spearman, monótona; la
MI mide dependencia arbitraria. Ejemplo canónico: $X \sim \mathcal N(0,1)$,
$Y = X^2$. Pearson = 0, pero $I(X;Y) > 0$: conocer $X^2$ deja solo el signo,
así que $H(X \mid Y) = 1$ bit exacto y $I = H(X) - 1 > 0$. Cuando la correlación
da cero y la MI no, hay estructura que un modelo lineal jamás verá.

## Ganancia de información en árboles

El criterio de split de un árbol es la información mutua empírica entre el
target $Y$ y la partición inducida por $X$:

$$IG(Y; X) = H(Y) - \sum_{v \in \text{val}(X)} \frac{|S_v|}{|S|}\,H(Y \mid X = v).$$

Como $H(Y)$ es constante dentro del nodo, maximizar IG es minimizar la entropía
condicional residual — exactamente $\hat I(Y;X)$ en la muestra.

**Sesgo hacia alta cardinalidad.** Para una variable con tantos valores como
filas, el extremo es una partición donde cada $S_v$ tiene un elemento,
$H(Y \mid X = v) = 0$ y el IG es máximo por construcción, sin señal real. Es la
razón por la que IDs, timestamps o textos crudos "ganan" todos los splits.
Mitigaciones:

- **Gain ratio** (C4.5): divide IG por la entropía de la partición
  $IV(X) = -\sum_v \frac{|S_v|}{|S|}\log\frac{|S_v|}{|S|}$, penalizando splits
  con muchos valores casi uniformes.
- límites de cardinalidad o encodings (frecuencia, ordinal con sentido) y
  validación cruzada honesta: si la variable de alta cardinalidad solo gana en
  training, es IG muestral, no información.

Los árboles binarios (CART) evitan parte del problema eligiendo umbral, pero un
umbral puede aislar un punto en un continuo ruidoso y dar IG máximo espurio; se
controla con profundidad mínima y poda, no con el criterio.

## Principio de máxima entropía

Entre todas las distribuciones compatibles con unas restricciones, elige la de
mayor entropía: la que hace menos supuestos. Es la respuesta formal a "¿qué
distribución uso si solo conozco estos momentos?"; los multiplicadores de
Lagrange dan la familia exponencial.

| Restricciones | Distribución de máxima entropía |
|---|---|
| ninguna, soporte finito | uniforme |
| media fija, soporte $[0, \infty)$ | exponencial |
| media fija, soporte $\{0,1,2,\dots\}$ | geométrica |
| media y varianza fijas, soporte $\mathbb{R}$ | gaussiana |

**Por qué la normal es la de máxima entropía con media/varianza fijas:** entre
todas las densidades con $E[X]=\mu$ y $Var(X)=\sigma^2$, la gaussiana maximiza
la entropía diferencial, cuyo valor es $\frac12\log(2\pi e\sigma^2)$. Eso la
convierte en el supuesto "de mínima información" por defecto para errores y
priors, y explica su ubicuidad: es la distribución que expresa exactamente lo
que sabes (dos momentos) y nada más. **Cuidado:** la entropía diferencial no es
invariante a cambios de variable (a diferencia de la MI); nunca compares
entropías de variables continuas con escalas o varianzas distintas.

## Log-loss como cross-entropía empírica

Log-loss es la entropía cruzada estimada en la muestra. Con targets one-hot,

$$\ell(y, \hat p) = -\sum_c \mathbb 1[y = c]\log \hat p_c = -\log \hat p_y,$$

y el promedio es la log-verosimilitud negativa normalizada. Minimizar log-loss
es maximizar verosimilitud (MLE); con regularización, MAP. El vínculo es
exacto: **cuando el modelo emite una distribución, la pérdida correcta es la
cross-entropía**; cualquier otra (0/1, hinge) es una aproximación con otras
propiedades. Es una scoring rule propiamente estricta: su esperanza se minimiza
solo en la probabilidad verdadera, así que no se puede engañar emitiendo
confianza mal ubicada.

En regresión, la contrapartida es la NLL gaussiana
$\frac12\log(2\pi\sigma^2) + \frac{(y-\mu)^2}{2\sigma^2}$: minimizarla ajusta
media y varianza a la vez, no solo el error cuadrático. Focal loss es cross-
entropía re-ponderada por $(1-p_t)^\gamma$, que amortigua los puntos fáciles.

## ELBO y variational inference

En inferencia variacional se maximiza la cota inferior de la evidencia:

$$\log p(x) \ge \mathbb{E}_{q(z)}\big[\log p(x \mid z)\big] - D_{KL}\big(q(z) \,\|\, p(z)\big).$$

Los dos términos son funcionales de esta teoría: el primero es un log-likelihood
esperado (cross-entropía), el segundo la KL entre la aproximación y el prior.
Maximizar ELBO equilibra ajuste a datos y costo de información de $q$ respecto
al prior; es un *minimum description length* práctico.

**Dónde aparece la MI.** En un VAE, la regularización cumple la identidad
$\mathbb{E}_x\big[D_{KL}(q(z|x) \| p(z))\big] = I(X;Z) + D_{KL}(q(z) \| p(z))$:
el término penaliza la información mutua entre datos y código, más la
divergencia del posterior agregado al prior. En InfoNCE / contrastive learning,
el objetivo es una cota inferior de $I(X;Z)$ entre representaciones y entradas.
En el information bottleneck se maximiza $I(Z;Y)$ sujeto a $I(X;Z) \le \beta$.
Lección práctica: todo objetivo de representación con un término KL o un InfoNCE
está trayendo entropía y MI a la mesa aunque el framework no lo diga.

## Aplicaciones prácticas

### Selección de features por información mutua

- **Filtros:** ordenar por $\hat I(X_j; Y)$ y quedarse con las $k$ primeras.
  Ventajas: captura dependencias no lineales, invariante a reescalado, sirve
  igual para continuo/discreto/categórico. Desventaja: ignora interacciones — un
  par de features puede aportar MI conjunta que ninguna aporta por separado (XOR
  es el ejemplo canónico).
- **mRMR:** maximiza relevancia $\hat I(X_j; Y)$ mientras minimiza redundancia
  $\hat I(X_j; X_{\text{seleccionadas}})$; corrige el defecto del filtro simple.
- **Por qué es difícil en continuas:** estimar $I$ exige estimar densidades
  conjuntas; el histograma sufre la maldición de la dimensionalidad y el sesgo
  depende de las celdas. En la práctica se usan estimadores basados en vecinos
  (KSG), siempre contra una baseline por permutación (véase Trampas).

```python
from sklearn.feature_selection import mutual_info_classif

mi = mutual_info_classif(X, y, discrete_features="auto", random_state=0)
# nats (log e); KSG local con n_neighbors=3 por defecto
# baseline: permuta y, recalcula, y solo cree valores >> permutado
```

### Diseño de loss y métricas

- cross-entropy/softmax para clasificación; focal loss para desbalance (CE
  re-ponderada); NLL gaussiana para regresión con incertidumbre.
- **Perplejidad:** $\text{perplexity}(p) = 2^{H(p)}$ con $H$ en bits (o $e^{H}$
  en nats). En NLP es $\exp\!\big(-\tfrac1N \sum_i \log p(w_i \mid
  \text{ctx})\big)$, la media geométrica de $1/p$: "tamaño efectivo del
  vocabulario por el que el modelo sigue dudando". Duplicar la perplejidad es
  duplicar la incertidumbre media; solo compara perplejidades con el mismo
  vocabulario y partición del corpus.

### Compresión

- La entropía limita toda compresión sin pérdidas; Huffman (óptimo) y aritmético
  (más cerca del límite). La lección presupuestal en DS: una columna categórica
  con entropía baja se comprime sola y no justifica ingeniería; una con entropía
  alta no debe encodearse por frecuencia si el valor es información (el encoding
  sube la entropía residual). Un target-based encoding introduce leak cuando
  $\hat I(X_j; Y)$ es alta — estima la MI antes de decidir el encoding (véase
  `ml/ingenieria-features.md`).

{% if ml_type == 'no_supervisado' %}
### Clustering con MI

- Las versiones normalizadas de la MI (NMI, AMI) son métricas *externas* de
  clustering: comparan particiones sin exigir etiquetas coincidentes en nombre,
  solo en estructura. NMI divide por $\sqrt{H(U)H(V)}$ y corrige el sesgo de
  cardinalidad; AMI además descuenta lo esperable por azar.
- La MI sirve como criterio de agrupación jerárquica (MI entre la variable de
  cluster y las features) y para elegir $k$: busca el $k$ a partir del cual
  añadir un cluster no aporta MI sobre los datos.
{% endif %}

{% if use_rag %}
### Recuperación y entropía (bloque RAG)

El RAG es teoría de la información aplicada: un buen `rag search` minimiza la
entropía condicional de la respuesta dado el chunk recuperado,
$H(\text{respuesta} \mid \text{chunk}) < H(\text{respuesta})$, es decir,
maximiza $I(\text{pregunta}; \text{chunk})$. Consecuencias:

- Un chunk redundante (entropía condicional residual alta) aporta poco aunque su
  similitud coseno con la query sea alta: la utilidad es reducir incertidumbre.
- El tamaño de chunk es un trade-off: más contexto reduce la entropía condicional
  si cabe en la ventana, pero diluye la señal. La compresión de contexto
  (LLMLingua) corta lo redundante: reordena el contenido para maximizar la MI
  entre el contexto y la pregunta.
- Medir calidad con hit_rate/recall@k es estimar empíricamente esa MI sobre el
  golden set: un índice que no baja la perplejidad de la respuesta no está
  recuperando información, solo parecido.
{% endif %}

## Trampas

- **Estimar MI en continuos.** El histograma con celdas fijas sesga: celdas
  grandes pierden dependencia (MI → 0 espuria), celdas pequeñas la inflan (cada
  punto en su celda → MI → H(X), arbitraria). El estimador de vecinos KSG
  (Kraskov–Stögbauer–Grassberger) adapta la resolución local y es el estándar de
  facto; aun así converge muy lento en dimensiones altas. Siempre compara contra
  una baseline de permutación antes de creer un valor.
- **KL infinita.** Si $q(x) = 0$ donde $p(x) > 0$, $D_{KL}(p\|q) = \infty$:
  aparece al comparar un modelo con una empírica de soporte mayor o con log-probs
  saturadas a cero. Mitigación: smoothing, clipping o $f$-divergencias
  alternativas.
- **Unidades confusas.** `np.log` da nats, `np.log2` da bits. Mezclar unidades
  invalida comparaciones de orden de magnitud; documenta la base del log en todo
  código que reporte entropías o perplejidades.
- **"MI alta" no es causalidad.** La MI es dependencia estadística, simétrica y
  no dirigida: un confusor común crea MI sin efecto causal, y un split con IG
  alto no prueba que manipular $X$ cambie $Y$ (véase `causalidad.md`).
- **Entropía diferencial y escala.** La entropía de variables continuas cambia
  con la parametrización; solo compárala dentro de la misma variable y escala.
- **IG muestral vs información.** La entropía condicional empírica decrece
  monótonamente al añadir variables con datos finitos. Toda "ganancia" por
  encima del ruido necesita validación cruzada o corrección (gain ratio, AMI,
  que además descuenta lo esperable por azar).

## Fuentes

- T. M. Cover, J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley,
  2006.
- D. J. C. MacKay, *Information Theory, Inference, and Learning Algorithms*, CUP,
  2003.
- C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006
  (caps. 1–2 y apéndice B: probabilidad, entropía, KL, familia exponencial).
- A. Kraskov, H. Stögbauer, P. Grassberger, "Estimating Mutual Information",
  *Phys. Rev. E* 69, 066138, 2004.
- E. T. Jaynes, *Probability Theory: The Logic of Science*, CUP, 2003 (máxima
  entropía).
