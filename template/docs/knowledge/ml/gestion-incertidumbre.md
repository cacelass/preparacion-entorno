# Gestión de la incertidumbre

## Qué es la incertidumbre y por qué importa

Una predicción sin su incertidumbre no es una decisión, es un número. Toda
decisión compara opciones bajo incertidumbre; si el modelo devuelve solo un
punto, el decisor lo trata como exacto y el error se cobra entero en
producción. La diferencia operativa no es estar mal, es **saber que estás
mal**: un sistema que se equivoca y lo avisa admite rechazo, escalado a humano
y reentrenamiento; uno que se equivoca callado destruye confianza. Gestionar
la incertidumbre es convertir una puntuación en un conjunto o distribución
que el proceso de decisión pueda consumir.

## Incertidumbre aleatoria vs epistémica

La incertidumbre total se descompone en dos fuentes con propiedades y remedios
distintos.

### Aleatoria (aleatoric)

Es el ruido irreducible del proceso que genera los datos: el resultado de un
dado, el sexo de un recién nacido, la latencia bajo condiciones no observadas.
No desaparece con más datos — es propiedad del mundo, no del modelo. Se modela
como la varianza del target condicionada a las features,
$\sigma^2(x) = \mathrm{Var}[y \mid x]$, y por eso puede ser heterocedástica
(cambiar con $x$). Es el límite inferior del error: ningún modelo la supera,
marca el mejor error alcanzable (el *ruido irreducible* de Kendall & Gal).

### Epistémica (epistemic)

Es lo que el modelo no sabe por falta de datos, y **sí desaparece con más
datos**: regiones del espacio de features sin cobertura, parámetros mal
identificados, arquitectura insuficiente. Es incertidumbre sobre los parámetros
dado el dataset, $p(\theta \mid \mathcal{D})$, y es reducible. La capturan los
métodos bayesianos y los ensembles; en la práctica se manifiesta como
dispersión entre modelos o entre muestras de la posterior.

### Cuándo importa cada una

- **Safety (aplicaciones de riesgo)**: ambas. La aleatoria limita lo que se
  puede predecir; la epistémica limita lo que se cree saber. Ocultar la
  epistémica en producción es la fuente clásica de sobreconfianza fuera del
  dominio de entrenamiento.
- **Decisión de reentrenar**: solo la epistémica. Si el modelo falla por falta
  de datos en una región, más datos lo arreglan; si falla por ruido irreducible,
  más datos no cambian nada y el coste del reentrenamiento es dinero perdido.
  Detectar drift de covariables (regiones nuevas) es un problema de
  incertidumbre epistémica.

### Separarlas en la práctica (a escala)

Sin inferencia bayesiana exacta, las dos fuentes se modelan con mecanismos
distintos y complementarios:

- **Epistémica → deep ensemble**: entrenar $J$ seeds independientes y usar la
  dispersión entre sus predicciones (Lakshminarayanan et al.). Es un baseline
  fuerte de calibración y escala a modelos grandes sin tocar el entrenamiento.
- **Aleatoria → perturbación funcional**: en vez de añadir ruido en la salida
  (que rompe la coherencia espacial), muestrear una **función**: un vector de
  ruido gaussiano de baja dimensión entra en capas de normalización compartidas
  y reparametriza el paso. Cada muestra es una alternativa dinámicamente
  coherente, no ruido punto a punto.

FGN (arXiv:2506.10772, WeatherNext 2) es el caso canónico a escala: 4 seeds
para la epistémica y un vector de 32 dimensiones perturbando las capas de
normalización para la aleatoria, sobre un campo global de ~$10^8$ variables. La
lección no es la arquitectura: es que **la fuente de la aleatoriedad y la
dimensionalidad del ruido determinan si el ensemble es un conjunto de
alternativas plausibles o un saco de ruido independiente.** (El CRPS como
función de pérdida y la evaluación de la estructura conjunta están en
`series-temporales.md`, sección "Forecasting probabilístico".)

## Calibración vs cobertura

Dos propiedades distintas que se confunden porque ambas cuantifican "qué tan
fiable es la incertidumbre".

### Calibración

La **calibración** dice: la frecuencia observada del evento coincide con la
probabilidad predicha. Si el modelo dice $0.8$, entre todas las predicciones
con $p \approx 0.8$ el evento debe ocurrir el $\sim 80\%$ de las veces. Se
evalúa con el **reliability diagram** (frecuencia observada vs confianza
predicha por bin), el **Brier score**
$$\mathrm{BS} = \frac{1}{n}\sum_i (p_i - y_i)^2$$
y el **ECE** (expected calibration error): media ponderada por bin de
$|$confianza $-$ frecuencia$|$ (ver [clasificacion.md](clasificacion.md) y
[metricas-y-evaluacion.md](metricas-y-evaluacion.md)).

### Cobertura

La **cobertura** dice: el intervalo o conjunto de predicción contiene el valor
real el $(1-\alpha)\%$ de las veces, en promedio. Es propiedad de **conjuntos**,
no de probabilidades: mide si la región que declara el modelo encierra la
verdad. Para un intervalo $C(x)$ a nivel $\alpha$,
$$P(Y \in C(X)) \ge 1 - \alpha$$

### No son lo mismo

Calibración habla de probabilidades, cobertura habla de conjuntos. Un modelo
perfectamente calibrado puede tener intervalos mal construidos (sistemáticamente
cortos); un intervalo con cobertura exacta puede venir de probabilidades
descalibradas. No se sustituyen: la decisión sobre la clase necesita
calibración, la decisión sobre el rechazo necesita cobertura. Medir una y
declarar la otra es un error de reporting.

{% if use_calibration %}
## Calibración en este proyecto

Con `use_calibration` activo, `models/calibrate.py` ajusta post-hoc el factor
de temperatura $T$ en validación (ver *temperature scaling* en Redes
sobreconfiadas) y expone reliability diagram, Brier score y ECE antes/después.
Reglas de uso:

- El $T$ se entrena en el **split de validación**, nunca en test ni train.
- Calibrar es monótono: no cambia el ranking ni el AUC, solo la lectura de las
  probabilidades como confianza. Recalibrar tras cambiar los class weights.
- Si el ajuste no reduce el ECE de forma medible, la descalibración no es
  temperatura-dependiente: prueba isotonic (no monótono) en su lugar.
{% endif %}

{% if use_conformal %}
## Conformal prediction

Convierte un predictor cualquiera (clasificador, regresor, red neuronal) en
**sets** o **intervalos** con una garantía de cobertura finita, sin supuestos
sobre el modelo. Es la columna de la incertidumbre en este corpus: la aplican
[clasificacion.md](clasificacion.md) (sets), [regresion.md](regresion.md)
(intervalos por residuos), [series-temporales.md](series-temporales.md)
(bloques temporales y ACI) y [metricas-y-evaluacion.md](metricas-y-evaluacion.md)
(cómo medirla). En este proyecto `models/conformal.py` implementa split
conformal sobre el modelo entrenado.

### Por qué es distribution-free

La garantía no usa la distribución de los datos ni la forma del predictor. Solo
exige que las muestras sean **exchangeables** (intercambiables: cualquier
permutación del conjunto tiene la misma distribución conjunta), un supuesto más
débil que i.i.d. No hay asintótica: la cobertura se sostiene con $n_{cal}$
finito, con un factor de corrección finito. Vale para cualquier modelo entrenado
por cualquier método, sin reentrenar nada en la fase de calibración.

### El nonconformity score

Un score $s(x, y)$ que mide cuán *raro* es ver la etiqueta $y$ junto con $x$:
pequeño = compatible con el modelo, grande = inusual. Ejemplos:

| Tarea | Score típico | Notas |
|---|---|---|
| Clasificación | $s(x, y) = 1 - p_y(x)$ | Probabilidad de la clase verdadera |
| Clasificación | $s(x, y) = \mathrm{APS}(x, y)$ | Suma acumulada de probs hasta $y$; mejor multiclase |
| Regresión | $s(x, y) = \lvert y - \hat{y}(x) \rvert$ | Residuo absoluto |
| Regresión | $s(x,y)=|y-\hat y(x)|/\hat\sigma(x)$ | Normalizado; anchos adaptativos (CQR) |

### Split conformal

Procedimiento exacto con un conjunto de calibración aparte:

1. Partir los datos: entrenar en una parte, reservar $n_{cal}$ muestras **de
   calibración** que el modelo no haya visto.
2. Calcular los scores de no-conformidad $s_i = s(x_i, y_i)$ en calibración.
3. Tomar el cuantil empírico
   $$q = \mathrm{quantile}\left(\{s_i\},\ \frac{\lceil (n_{cal}+1)(1-\alpha)\rceil}{n_{cal}}\right)$$
4. Para un nuevo $x$: clasificación $C(x) = \{ y : s(x, y) \le q \}$;
   regresión $C(x) = [\hat{y}(x) - q,\ \hat{y}(x) + q]$.

No hay reentrenamiento: el modelo base se entrena una vez y la calibración es
un solo pase sobre el conjunto de validación.

### Teorema de cobertura marginal

Para una muestra nueva $(X_{n+1}, Y_{n+1})$ exchangeable con la calibración,
$$P\left(Y_{n+1} \in C(X_{n+1})\right) \ge 1 - \alpha$$
La cobertura es **marginal** sobre los datos nuevos: el error se reparte entre
instancias. No garantiza cobertura **condicional** a cada $x$ (ver límites).
Con datos i.i.d. en producción, fallar en el $\alpha$ de los casos es el coste
pactado, no un bug.

### Sets vs intervalos

- **Clasificación**: $C(x)$ es un conjunto de etiquetas. Modelo seguro →
  singleton; ambigüedad → varias clases; datos fuera del dominio → conjunto
  vacío (la señal de OOD de la que habla [clasificacion.md](clasificacion.md)).
- **Regresión**: $C(x)$ es un intervalo. Con residuo absoluto el ancho es
  constante en todo el espacio; para anchos adaptativos hay que usar scores
  normalizados o CQR (ver [regresion.md](regresion.md)).

### Límites

- La cobertura **condicional** por $x$ no está garantizada; en regiones de
  datos raras la cobertura puntual puede caer muy por debajo de $1-\alpha$.
- El **tamaño del set/intervalo varía por instancia**: es la medida de
  incertidumbre, y se evalúa con la sharpness (ver Evaluación).
- Con **drift** entre calibración y producción la exchangeabilidad se rompe y
  la cobertura marginal deja de valer (ver Trampas); en series temporales el
  remedio es recalibrar periódicamente (ACI, ver
  [series-temporales.md](series-temporales.md)).
{% endif %}

## Métodos bayesianos

Modelar la incertidumbre como una **distribución a posteriori** sobre los
parámetros:
$$p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\, p(\theta)$$
La posterior cuantifica todo: qué parámetros son compatibles con los datos y
con cuánta masa, y de ahí salen las predicciones con su propagación de
incertidumbre.

### Posterior, prioris y credible intervals

- **Posterior**: $p(\theta \mid \mathcal{D})$. Es la incertidumbre *epistémica*
  sobre los parámetros después de ver los datos.
- **Priori**: $p(\theta)$, lo que se sabe antes de ver los datos. La priori
  débil (poca información) deja que los datos dominen; la fuerte regulariza
  pero sesga. Elegirla a priori es declarar la respuesta, no descubrirla.
- **Credible interval**: región de la posterior que contiene $1-\alpha$ de la
  probabilidad: "con probabilidad $0.95$ el parámetro está aquí, dados los
  datos y la priori".

### Intervalos creíbles vs de confianza

- **Creíble (bayesiano)**: probabilidad sobre el parámetro dados los datos:
  $P(\theta\in C\mid\mathcal{D})=1-\alpha$; "con probabilidad $1-\alpha$ el
  parámetro está aquí".
- **Confianza (frecuentista)**: probabilidad sobre el *procedimiento*, no sobre
  el parámetro: repitiendo la recogida de datos, el $1-\alpha$ de los
  intervalos así construidos contendrían el parámetro.

No son intercambiables. Con priori plana y modelos regulares tienden a
coincidir numéricamente; la lectura es distinta y en muestras pequeñas pueden
divergir de forma material.

### Coste de la inferencia

La posterior exacta es intratable salvo casos cerrados (conjugados, GP). Las
aproximaciones —MCMC, variacional, o las aproximaciones de red abajo— añaden
coste de cómputo y de validación de la propia aproximación. El coste real no
es el entrenamiento, es saber cuándo la aproximación miente.

### Gaussian Processes

La posterior es **cerrada** (gaussiana): media y varianza analíticas. La
varianza predicha crece monótonamente con la distancia a los datos de
entrenamiento — lejos de las observaciones la incertidumbre explota, justo el
comportamiento deseado para detectar OOD. Coste $O(n^3)$ en el número de
puntos; con aproximaciones inducidas (SVGP) escala a datasets grandes a cambio
de incertidumbre menos fiel en las regiones no vistas.

{% if ml_type == 'redes_neuronales' %}
### MC-dropout

Una red entrenada con dropout aproxima la posterior dejando el dropout
**activo en test**: cada forward pass es una muestra; la media de $T$ pasadas
es la predicción y la varianza entre pasadas es la incertidumbre. Es la
aproximación de Gal & Ghahramani: sin coste de entrenamiento adicional, a
coste de $T$ forward passes por predicción. Barata y razonablemente calibrada;
menos precisa que ensembles cuando importa la cola de la distribución.

### Ensembles como estimador de incertidumbre

Entrenar $M$ redes (distintas semillas, submuestras de datos o
inicializaciones) y leer la **dispersión entre modelos** como incertidumbre:
la varianza entre predicciones captura la incertidumbre epistémica. Las deep
ensembles de Lakshminarayanan et al. son un baseline fuerte de calibración y
detección de OOD, a menudo superior a bayesianos aproximados al mismo coste.
El precio es $M \times$ el coste de entrenamiento y de inferencia.
{% endif %}

## Redes sobreconfiadas

### Por qué las softmax probabilities no son confianza

Guo et al. muestran que las redes modernas (sin recalibración) están
sistemáticamente **sobreconfiadas**: declaran $0.9$ donde la frecuencia real
es $0.7$. La softmax es una normalización de logits, no una medida de
incertidumbre; con más capacidad que datos la red memoriza y la confianza
sube con el overfit. La descalibración además depende del dataset y la
arquitectura: no es un defecto del modelo, es el estado por defecto.

### Correcciones post-hoc

- **Temperature scaling**: $p = \mathrm{softmax}(\mathrm{logits} / T)$ con $T$
  ajustado por máxima verosimilitud en validación. Un solo parámetro, monótono,
  barato; corrige la sobreconfianza media sin cambiar el ranking.
- **Isotonic regression**: reasigna las probabilidades con una función
  monótona por tramos ajustada en validación. Más potente con muchos datos,
  no monótono por defecto, puede sobreajustar la calibración a la muestra.

Ambas corrigen la **calibración**, no la cobertura ni la incertidumbre
epistémica: no le dan al modelo conciencia de lo que no sabe, solo ajustan la
lectura de sus probabilidades (ver Calibración en este proyecto, si aplica).

## De la incertidumbre a la decisión

### Umbrales de rechazo

Rechazar (refuse) cuando la incertidumbre supera un umbral $\tau$: la
predicción no se emite y se delega. El umbral se elige con la **matriz de
coste**, no a ojo: rechazar cuesta $c_{reject}$; equivocarse cuesta $c_{error}$;
el punto de equilibrio compara las dos curvas de coste esperado. Un sistema con
conformal + rechazo sabe cuándo no sabe (ver
[exprime-el-modelo.md](exprime-el-modelo.md)).

### Matriz de coste

| Acción | Clase correcta | Clase equivocada |
|---|---|---|
| Predecir | $0$ | $c_{error}$ |
| Rechazar | $c_{reject}$ | $c_{reject}$ |

El óptimo no maximiza exactitud, minimiza coste esperado. Si $c_{reject}
\ll c_{error}$, rechaza ante cualquier incertidumbre relevante; si $c_{error}$
es pequeño, se predice casi siempre. La matriz convierte la incertidumbre en
una cantidad monetaria comparable.

### Predicción + intervalo, no punto

Entregar el par $(\hat{y},\, C(x))$: el intervalo es parte del artefacto de
decisión. El decisor puede comparar contra umbrales de negocio ("¿$y$ puede
superar el mínimo?"), algo imposible con un punto. La decisión se toma sobre la
banda, no sobre la media.

### Cuándo escalar al humano

Escalar cuando la incertidumbre es alta y el error es costoso o irreversible
(diagnóstico, crédito, moderación). Regla práctica: el sistema automático
decide cuando la incertidumbre es baja y el coste del error es aceptable; el
humano recibe la predicción, su intervalo y el motivo de la incertidumbre. El
escalado es un diseño, no una excusa: sin un protocolo de escalado definido, la
incertidumbre alta no lleva a ningún sitio.

## Evaluación de la incertidumbre

- **Cobertura empírica**: fracción de $y_i$ dentro de $C(x_i)$ sobre un
  conjunto de evaluación. Debe rondar $1-\alpha$ (conformal, cuantiles). Una
  cobertura superior a la prometida no es mejor: es un intervalo inútilmente
  ancho.
- **Sharpness**: qué tan estrechos son los intervalos/sets a cobertura fija.
  Cobertura $1-\alpha$ con intervalos enormes es trivial; el modelo útil
  minimiza la anchura media condicionada a cumplir la cobertura. Métricas:
  anchura media, CRPS para distribuciones, tamaño medio del set en
  clasificación.
- **Calibración por bin**: agrupar por confianza predicha y comparar frecuencia
  observada vs predicha (reliability diagram, ECE). Complementa a la cobertura:
  cobertura correcta no implica calibración correcta.
- **Tradeoff sharpness/coverage**: a mayor $1-\alpha$, más ancho el intervalo.
  Evaluar a un $\alpha$ de negocio fijo y reportar ambos números juntos:
  cobertura sin sharpness no discrimina modelos, sharpness sin cobertura no es
  fiable.
- **CRPS como scoring rule propia**: para una distribución predictiva $F$ y un
  valor observado $y$,
  $\operatorname{CRPS}(F,y) = \int (F(z)-\mathbf{1}[y \le z])^2\, dz$ combina
  calibración y sharpness en una cifra; es **propia** (se minimiza con la
  distribución verdadera) y **diferenciable**, así que sirve como pérdida de
  entrenamiento (FGN, WeatherNext 2). Sobre muestra finita se usa el estimador
  fair, que penaliza ensembles demasiado estrechos. Detalle y estructura
  conjunta en `series-temporales.md`.

## Trampas

1. **Confundir calibración con cobertura**: se miden con métricas distintas y
   ninguna implica la otra (ver arriba).
2. **Intervalos simétricos cuando el error no lo es**: si $y \mid x$ es
   asimétrica (caudales, precios, tiempos), un intervalo simétrico alrededor de
   la media cubre mal un lado. Usar cuantiles, conformal con score asimétrico o
   CQR.
3. **Conformal bajo drift**: si la producción no es exchangeable con la
   calibración, la cobertura marginal se rompe y puede caer muy por debajo de
   $1-\alpha$. Detectar drift y recalibrar periódicamente (ACI, bloques
   temporales).
4. **Falsa seguridad de "baja incertidumbre" con datos fuera de distribución**:
   un modelo puede dar incertidumbre baja fuera de su dominio (softmax
   sobreconfiado, posterior degenerada). La incertidumbre interna no prueba
   pertenencia al dominio: complementar con detección de OOD y monitoreo de
   drift.

## Fuentes

- Vovk, V., Gammerman, A., Shafer, G., *Algorithmic Learning in a Random
  World*, Springer 2005 (conformal prediction). Sin arXiv —
  https://doi.org/10.1007/b106715
- Angelopoulos, A. N., Bates, S., *A Gentle Introduction to Conformal
  Prediction and Distribution-Free Uncertainty Quantification*, 2021.
  arXiv:2107.07511 — https://arxiv.org/abs/2107.07511
- Kendall, A., Gal, Y., *Uncertainties in Deep Learning*, 2017.
  arXiv:1703.04977 — https://arxiv.org/abs/1703.04977
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q., *On Calibration of Modern
  Neural Networks*, ICML 2017. arXiv:1706.04599 —
  https://arxiv.org/abs/1706.04599
- Brier, G. W., *Verification of Forecasts Expressed in Terms of Probability*,
  Monthly Weather Review, 1950. Sin arXiv —
  https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2
- Gal, Y., Ghahramani, Z., *Dropout as a Bayesian Approximation*, 2016.
  arXiv:1506.02142 — https://arxiv.org/abs/1506.02142
- Lakshminarayanan, B., Pritzel, A., Blundell, C., *Simple and Scalable
  Predictive Uncertainty Estimation using Deep Ensembles*, 2017.
  arXiv:1612.01474 — https://arxiv.org/abs/1612.01474
- Alet, F., Price, I., El-Kadi, A., Masters, D., Markou, S., Andersson, T. R.,
  Stott, J., Lam, R., Willson, M., Sanchez-Gonzalez, A., Battaglia, P.,
  *Skillful joint probabilistic weather forecasting from marginals* (FGN),
  2025. arXiv:2506.10772 — https://arxiv.org/abs/2506.10772
