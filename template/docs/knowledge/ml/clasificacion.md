{% if task_type == 'clasificacion' %}
# Clasificación: de la pérdida a la decisión

El eslabón que separa un clasificador de un "modelo que devuelve una clase" es
la cadena completa: pérdida correcta, umbral explícito, manejo honesto del
desbalance y probabilidades que se puedan usar. Este documento cubre esa
cadena y sus modos de fallo habituales.

## La pérdida: cross-entropy

Para clasificación binaria con $y \in \{0, 1\}$ y $p = P(y=1 \mid x)$, la
pérdida es la cross-entropy (equivalente a la log-verosimilitud negativa de
una Bernoulli):

$$ \mathcal{L} = -\frac{1}{n}\sum_i \left[ y_i \log p_i + (1 - y_i)\log(1 -
   p_i) \right] $$

Para multiclase con $K$ clases y $y_k \in \{0, 1\}$, la categórica (con
salida softmax $p_k = e^{z_k}/\sum_j e^{z_j}$):

$$ \mathcal{L} = -\frac{1}{n}\sum_i \sum_{k=1}^{K} y_{ik} \log p_{ik} $$

**Por qué no usar MSE.** Sobre una salida sigmoide, MSE tiene dos
problemas:

1. **Gradiente que se desvanece**: $\frac{\partial}{\partial z}
   \mathrm{MSE} \propto (p - y)\,\sigma'(z)$, y $\sigma'(z) \to 0$ en los
   extremos. En cross-entropy la derivada es $\propto (p - y)$: aprende
   rápido justo donde el modelo está más equivocado.
2. **Sin interpretación probabilística**: MSE no es la verosimilitud de
   ningún modelo generativo Bernoulli/categórico. El mínimo de log-loss es
   la probabilidad condicionada $P(y \mid x)$; el mínimo de MSE sobre
   etiquetas 0/1 con salida sigmoide converge a lo mismo *en el límite
   infinito de datos*, pero con muestras finitas converge más lento y
   exagera las confianzas.

La log-loss mide lo único que importa: qué tan bien las probabilidades
estiman $P(y \mid x)$. Es la métrica de optimización; la métrica de
evaluación (accuracy, F1, etc.) es otra capa.

## Umbralización

La salida de un clasificador son probabilidades; la clase predicha depende de
un umbral. El default de 0.5 asume costes simétricos: FP y FN cuestan lo
mismo. Si no es así, el umbral óptimo deriva de la matriz de costes
($c_{FP}$, $c_{FN}$):

$$ \text{predice 1 si } p \ge \frac{c_{FP}}{c_{FP} + c_{FN}} $$

Mover el umbral recorre la curva ROC: cada umbral es un punto
$(FPR, TPR)$. Se elige el punto que minimiza el coste esperado
(o maximiza Youden $J = TPR - FPR$, o satisface un objetivo de recall
mínimo). Cuidado: elegir el umbral sobre el test es data leakage de decisión
— se optimiza sobre validación y el test se evalúa una sola vez con el
umbral ya fijo.

## Datos desbalanceados

**Por qué el accuracy es una mentira.** Con 99% negativo y 1% positivo, el
clasificador trivial "siempre negativo" obtiene 99% de accuracy. Toda métrica
agregada sobre las clases sin pesar (y el accuracy lo es) queda dominada por
la mayoría. La evaluación correcta es sobre la clase minoritaria:
precisión/recall/F1 de la clase positiva, curva PR (precision-recall) o
métricas macro.

| Técnica | Mecanismo | Caveats |
|---|---|---|
| Resampling (SMOTE) | Sintéticos interpolando vecinos | Solo en train; puede amplificar ruido |
| Class weights | $w_k = n/(K n_k)$ en la pérdida | Conserva distribución; recalibrar umbral después |
| Encuadre como anomalía | One-class / isolation forest | Solo si la minoría es estructuralmente distinta |

La curva PR es preferible a la ROC bajo desbalance fuerte: la ROC es
insensible a la ratio entre clases y puede pintar un modelo con precisión
minúscula como excelente. Sobre la evaluación por clases ver
[metricas-y-evaluacion.md](../modelos/metricas-y-evaluacion.md).

## Multiclase

- **OvR** (one-vs-rest): $K$ clasificadores binarios (clase $k$ contra el
  resto). Fronteras más simples; las probabilidades se normalizan y quedan
  descalibradas si no se corrigen.
- **OvO** (one-vs-one): $\binom{K}{2}$ clasificadores. Más costoso;
  mejora cuando cada par es fácilmente separable.
- **Softmax** directo: $p_k = e^{z_k} / \sum_j e^{z_j}$ — el modelo
  multiclase nativo; la cross-entropía con softmax tiene gradiente
  $\nabla_{z_k} \mathcal{L} = p_k - y_k$ (muy limpio).
- **Label encoding**: el target se codifica como entero (índice de clase),
  no como ordinal — las etiquetas 0,1,2 no tienen orden y la pérdida no debe
  tratarlas como numéricas. La salida no es un ranking, es una distribución
  sobre $K$ categorías.

## Calibración

{% if use_calibration %}
Las probabilidades de un modelo no son automáticamente "probabilidades": un
modelo está **calibrado** si entre las predicciones con confianza $p$, la
fracción real de positivos es $p$. Un clasificador sobreconfiado predice
0.9 en puntos donde solo el 60% son positivos.

**Por qué se descalibran.** El optimizador empuja las probabilidades a los
extremos (0 y 1) para reducir log-loss; con datos de test desplazados, las
confianzas exageran. SVM produce scores no probabilísticos; los árboles
devuelven frecuencias de hoja (cuantizadas); los GBDT mezclan hojas. El
resultado: la media global puede ser razonable y las confianzas individuales
mal.

**Métodos post-hoc** (ajustan un mapeo score -> probabilidad sobre
validación):

| Método | Mapeo | Cuándo |
|---|---|---|
| Platt | $p = 1/(1 + \exp(A f(x) + B))$ | Scores con forma de logit; SVM; pocos parámetros |
| Isotónico | Regresión isotónica no paramétrica | Muchos datos; sobreajusta con pocos |
| Temperature scaling | $p = \mathrm{softmax}(z/T)$ | Redes: un solo $T$; no cambia el ranking |

Con $T > 1$ las probabilidades se vuelven más planas (menos confiadas) y con
$T < 1$ más extremas. La temperatura se ajusta maximizando la
log-verosimilitud sobre validación y **no cambia el accuracy** (el argmax
es invariante a $T$): solo arregla la confianza.

**Evaluar calibración.** Brier score:
$\mathrm{BS} = \frac{1}{n}\sum_i (p_i - y_i)^2$, que se descompone en
*reliability* (calibración), *resolution* y *uncertainty*; y el diagrama de
confianza (ECE: error de calibración esperado, media de
$|$confianza$-$frecuencia$|$ por bin). Una log-loss baja ya castiga la
descalibración; BS la aísla.
{% endif %}

## Conformal prediction

{% if use_conformal %}
Un clasificador puntual no cuantifica la incertidumbre. La conformal
prediction convierte un predictor cualquiera en **conjuntos de predicción**
con garantía de cobertura, sin supuestos sobre el modelo:

$$ P(Y \in C(X)) \ge 1 - \alpha $$

**Garantía.** La cobertura es *marginal* sobre los datos nuevos, no
condicional a cada $x$: el error se reparte entre instancias. Es
*distribution-free*: solo exige que las muestras sean **exchangeables**
(permutables), un supuesto más débil que i.i.d.

**Algoritmo split-conformal:**
1. Entrenar sobre un subconjunto; reservar $n_{cal}$ muestras de
   calibración.
2. Definir el score de no-conformidad: típicamente $s(x, y) = 1 - p_y(x)$
   (probabilidad de la clase verdadera) o $s = 1 - \max_k p_k(x)$ con
   $y \ne \arg\max$.
3. Calcular $q = $ cuantil $\lceil (n_{cal}+1)(1-\alpha)\rceil / n_{cal}$
   de los scores de calibración.
4. Para un nuevo $x$: $C(x) = \{ y : s(x, y) \le q \}$.

Con $\alpha = 0.1$ esperamos que la clase verdadera esté en el conjunto el
90% de las veces. Cuando el modelo está seguro, los conjuntos son
singletons; cuando no, crecen o se vacían — el **tamaño del set es la medida
de incertidumbre**. No hay garantía condicional: en regiones de datos raras
la cobertura puntual puede ser peor. Para regresión el mecanismo es análogo
(ver `regresion.md`).
{% endif %}

## Trampas prácticas

1. **Encoding antes del split**: ajustar `StandardScaler`, one-hot,
   target-encoding o SMOTE sobre el dataset completo antes de partir
   filtra información del test en el entrenamiento (leakage). El orden
   correcto es: partir primero y ajustar todo transformador solo con el
   train. El target encoding es el más peligroso — usa el target para
   construir la feature (ver
   [ingenieria-features.md](../ingenieria/ingenieria-features.md)).
2. **Umbral afinado sobre el test**: probar varios umbrales sobre el test
   y quedarse con el mejor sobre-estima el rendimiento real. El umbral se
   elige en validación; el test solo valida el pipeline completo una vez.
3. **Macro vs micro**: `macro` promedia la métrica por clase (cada clase
   pesa igual) — correcto cuando importa la minoría; `micro` agrega
   globalmente y queda dominado por la mayoría. Reportar ambas con su
   interpretación y la elección explícita (ver
   [metricas-y-evaluacion.md](../modelos/metricas-y-evaluacion.md)).
4. **Desbalance + calibración**: los class weights cambian la distribución
   de las probabilidades; recalibrar (o re-umbralizar) después de
   reweighting.
5. **Clases nuevas en producción**: una categoría que no existía en train
   produce confianzas altas en la clase vecina; con conformal, sets vacíos
   son la señal de OOD.

## Fuentes

- **Probabilistic Outputs for Support Vector Machines and Comparisons to
  Regularized Likelihood Methods** — J. C. Platt (1999). Sin arXiv —
  https://doi.org/10.1007/978-1-4612-4924-6_7
- **Transforming Classifier Scores into Accurate Multiclass Probability
  Estimates** — B. Zadrozny, C. Elkan (2002). Sin arXiv —
  https://doi.org/10.1145/775047.775151
- **On Calibration of Modern Neural Networks** — C. Guo et al. (2017).
  arXiv:1706.04599 — https://arxiv.org/abs/1706.04599
- **A Gentle Introduction to Conformal Prediction and Distribution-Free
  Uncertainty Quantification** — A. N. Angelopoulos, S. Bates (2021).
  arXiv:2107.07511 — https://arxiv.org/abs/2107.07511
- **Algorithmic Learning in a Random World** — V. Vovk, A. Gammerman,
  G. Shafer (2005). Sin arXiv — https://doi.org/10.1007/b106715
- **SMOTE: Synthetic Minority Over-sampling Technique** — N. V. Chawla et
  al. (2002). Sin arXiv — https://doi.org/10.1613/jair.953
- **A Mathematical Theory of Communication** — C. E. Shannon (1948).
  Sin arXiv — https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
{% endif %}
