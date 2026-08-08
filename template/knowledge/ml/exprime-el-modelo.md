# Exprimir el modelo

Un modelo entrenado es un punto de partida, no un techo. El arte de exprimirlo
es saber en qué orden tocar cada palanca: primero los datos, luego las
features, después los hiperparámetros, y solo al final la familia de modelo.

## Palancas de datos antes que las del modelo

El impacto no es uniforme. El ranking de palancas, de mayor a menor efecto
esperado:

1. **Calidad de las etiquetas**: ¿los labels están bien? Errores de etiqueta
   al 5-10% son comunes y limitan todo lo demás. Auditar una muestra de y
   (releer, doble etiquetado, confusión entre clases) suele rendir más que
   semanas de tuning. Un modelo no puede aprender lo que la etiqueta no sabe.
2. **Más datos**: más ejemplos de los slices difíciles, no más datos
   redundantes. Con curvas de aprendizaje en saturación, los datos dejan de
   ayudar.
3. **Datos más limpios**: outliers explicados, valores inconsistentes,
   duplicados, desfases temporales. Limpiar corrige errores sistemáticos que
   el modelo aprende como patrones.
4. **Calidad de features**: features de dominio, bien construidas, con
   significado causal. Sustituir features ruidosas por versiones correctas.
5. **Modelo e hiperparámetros**: lo último. Con datos mal etiquetados o
   features pobres, ningún algoritmo las arregla.

La lección operativa: antes de tocar `eta` o la arquitectura, responde "¿por
qué falla?" con datos — etiquetas, slicing de errores y curvas de aprendizaje.
El modelo perfecto sobre datos sucios pierde contra un modelo mediocre sobre
datos buenos.

## Exprimir el boosting (árboles)

Los hiperparámetros de boosting no actúan solos: forman un sistema. Lo que
importa es la interacción.

### La pareja eta/learning_rate y n_estimators

- `eta` (XGBoost/CatBoost) o `learning_rate` (LightGBM) escala el peso de cada
  árbol nuevo (shrinkage). Bajar `eta` obliga a subir `n_estimators`:
  capacidad y regularización viajan juntas.
- **Por qué más rondas + eta menor suele ganar**: el shrinkage deja espacio
  para que árboles posteriores corrijan de forma fina; la pérdida no se
  consume de golpe. En general, `eta = 0.01-0.05` con más estimadores supera a
  `eta = 0.3` con pocos, a costa de tiempo de entrenamiento.
- La relación no es ilimitada: por debajo de cierto `eta`, ganar más rondas no
  reduce el error y solo sube el cómputo. El punto óptimo se encuentra
  barriendo el par, no cada uno por separado.

### Early stopping sobre validación, nunca sobre train

El error en train desciende siempre: parar por él es elegir el modelo sobre su
propia medida. Se observa una métrica en un conjunto de validación aparte
(holdout o fold) y se guarda el mejor checkpoint; si no mejora en
`early_stopping_rounds` consecutivos, se corta. El número de rondas importa
poco (50-200); lo que decide es que la decisión use validación limpia. En CV,
el early stopping se hace dentro de cada fold.

### subsample y colsample_bytree

- `subsample` (o `bagging_fraction`): fracción de filas por iteración. Añade
  estocasticidad y regulariza; valores útiles 0.5-0.8.
- `colsample_bytree` (o `feature_fraction`): fracción de columnas por árbol.
  El paper de XGBoost señala que el submuestreo de columnas evita el
  sobreajuste incluso más que el de filas y acelera el cómputo paralelo. Es la
  palanca estocástica de mayor valor en boosting.

### max_depth vs min_child_weight

- `max_depth` limita la profundidad (interacciones que puede capturar cada
  árbol). Demasiado profundo → sobreajuste; muy poco → sin interacciones.
- `min_child_weight` (suma mínima de pesos/hessianos en una hoja) es el
  regularizador estructural más potente: poda hojas con poco soporte. Subirlo
  reduce varianza con coste mínimo de sesgo.
- Se compensan: `max_depth` sube y `min_child_weight` sube juntos cuando el
  modelo sobreajusta; ambos bajan cuando underfit.

### Regularización: reg_alpha / reg_lambda

El objetivo regularizado del boosting incluye

$$\Omega(f)=\gamma T + \tfrac{1}{2}\lambda\lVert w\rVert^2 + \alpha|w|$$

con $T$ hojas, $w$ pesos de hoja, $\lambda$ penalización L2 (smooth) y
$\alpha$ L1 (sparsity). `lambda` suaviza los pesos de las hojas — la
regularización que menos daña y que más veces ayuda; `alpha` empuja hojas a
peso cero y sirve para tablas muy grandes o ruido. `gamma` penaliza añadir
cada hoja (complejidad estructural). Subir `lambda`/`alpha` es más limpio que
bajar `max_depth` cuando solo sobra varianza.

### Restricciones de monotonicidad y de interacción

- **Monotonic constraints**: fijar que la predicción sea monótona en una
  feature (mayor edad → mayor riesgo). Inyecta conocimiento de dominio,
  estabiliza regiones con pocos datos y mantiene el modelo coherente con la
  teoría del negocio.
- **Interaction constraints**: permitir interacciones solo entre grupos de
  features declarados. Reduce el espacio de modelos, evita interacciones sin
  sentido y mejora la robustez fuera del soporte de entrenamiento.

### scale_pos_weight para desbalance

En clasificación binaria desbalanceada, `scale_pos_weight` re-ponderá la clase
positiva: aproximadamente `negativos/positivos`. Es la palanca más directa
para subir recall de la clase minoritaria sin tocar la métrica; el trade-off
se cierra con el umbral de decisión (ver calibración), no solo con el peso.

### Objetivos y métricas custom

Boosting permite objetivos propios: clases con costes distintos, regresión
con pérdidas asimétricas (cuantil, Huber), ranking con métricas custom. La
regla: el **objetivo** (lo que se optimiza) y la **métrica** (lo que se mide)
son cosas distintas; la métrica de negocio debe estar alineada con la métrica
de validación, y esta a su vez guiar el objetivo.

{% if use_xgboost %}
### Exprimir XGBoost

- Maneja valores ausentes nativamente: aprende la "dirección por defecto" de
  cada split (sparsity-aware, arXiv:1603.02754). **No imputes por imputar**:
  si el hueco es informativo, XGBoost lo explota mejor que una imputación
  previa; la imputación solo si hay razón de dominio.
- Hiperparámetros de mayor impacto por defecto: `eta` bajo + `n_estimators`
  alto con `early_stopping_rounds`; `max_depth` 4-8; `colsample_bytree`; y
  `reg_lambda`/`reg_alpha` cuando la varianza domina.
- Usar `monotone_constraints` y `interaction_constraints` (clave de diccionario
  o lista) para inyectar conocimiento de dominio.
- El árbol se aprende por splits con pérdida de segundo orden; para datasets
  grandes, `tree_method="hist"` es más rápido con calidad casi idéntica.
{% endif %}

{% if use_lightgbm %}
### Exprimir LightGBM

- Crece los árboles **leaf-wise** (hoja con mejor ganancia), no level-wise:
  puede sobreajustar antes que XGBoost. La protección es `max_depth` (cap),
  `min_data_in_leaf` y `lambda_l2`.
- `num_leaves` es la palanca de capacidad: subirlo captura más interacciones;
  acompañarlo de `min_data_in_leaf` (soporte mínimo por hoja) para que la
  capacidad extra no se gaste en ruido.
- `feature_fraction` y `bagging_fraction` para estocasticidad; `bagging_freq`
  para el intervalo de remuestreo.
- Categóricas nativas: pasar `categorical_feature` evita el one-hot y deja que
  el algoritmo ordene las categorías por gradiente; en datos de cardinalidad
  alta ahorra memoria y tiempo sin perder calidad.
{% endif %}

{% if use_catboost %}
### Exprimir CatBoost

- Categóricas nativas con **ordered target statistics**: cada categoría se
  codifica con la media del target *sin usar el propio ejemplo*
  (arXiv:1706.09516), eliminando el leakage del target encoding clásico. Es su
  ventaja diferencial: no hace falta one-hot ni preparación previa.
- `cat_features` explícito; combina categorías de splits previos
  (feature combinations), lo que captura interacciones de alto orden.
- **Ordered boosting** reduce el sesgo de predicción shift frente al GBDT
  clásico; `bootstrap_type="Bayesian"` mejora la estimación de incertidumbre
  interna (`prediction_type="TotalUncertainty"`).
- Árboles simétricos y `depth` pequeña (2-8) mantienen el modelo ligero y
  estable; `l2_leaf_reg` es su regularización equivalente a `lambda`.
{% endif %}

{% if ml_type == 'redes_neuronales' %}
## Exprimir redes neuronales

- **LR schedules con warmup**: arrancar con LR pequeña que sube linealmente en
  las primeras iteraciones estabiliza el entrenamiento; luego decay (coseno o
  por pasos). La búsqueda del rango de LR (LR finder) vale más que el
  scheduler: si el rango es malo, ningún schedule lo arregla.
- **AdamW con weight decay**: el weight decay desacoplado regulariza los pesos
  sin ensuciar el gradiente del momento. `weight_decay` 1e-4-1e-2 según
  tamaño de datos.
- **Label smoothing**: suaviza la distribución objetivo (p.ej. 0.9/0.05/0.05
  en 3 clases); mejora calibración y generalización, sobre todo con pérdidas
  de entropía cruzada.
- **EMA de pesos**: promediar los pesos de los últimos pasos (media móvil
  exponencial) da un modelo más estable que el último checkpoint; coste cero
  en inferencia.
- **Gradient clipping**: recorta la norma del gradiente; imprescindible con
  LSTM/GRU (explosión de gradientes), útil siempre para estabilidad.
- **AMP** (mixed precision): entrena en float16/float32; ahorra memoria y
  velocidad, no precisión, si el loss se estabiliza.
- **Batch mayor + escalado de LR**: doblar el batch permite doblar la LR
  (regla de escala); es la vía más barata de acelerar el entrenamiento sin
  perder calidad.
- **Data augmentation**: más barato que más datos; solo si genera ejemplos
  plausibles de la distribución real.
- **Dropout bien colocado**: tras las capas de mayor capacidad, no en la
  entrada; `dropout` alto es regularización fuerte que conviene acompañar de
  más épocas.
- **Weight tying / ensamblados en tiempo de entrenamiento**: SWA (promediar
  puntos de la trayectoria) y snapshot ensembles (checkpoints de diferentes
  mínimos) multiplican la eficacia sin entrenar más.
- **Destilación de conocimiento**: entrenar un modelo pequeño contra las
  salidas (soft) de uno grande; el estudiante aprende la estructura de las
  probabilidades, no solo la clase.
- **Test-time augmentation (TTA)**: promediar predicciones sobre
  transformaciones del input; sube robustez, multiplica el costo de
  inferencia.
{% endif %}

## Exprimir el ensamblado

- **Bagging**: muestrear filas (y opcionalmente columnas) y promediar
  predicciones (random forest es el caso canónico). Reduce varianza sin tocar
  sesgo; mejora siempre que el modelo tenga varianza que cortar.
- **Blend vs stack**:
  - Blend: promediar (o ponderar) las predicciones de varios modelos sobre la
    misma validación. Simple, estable, sin entrenar un metamodelo.
  - Stacking: un metamodelo aprende a combinar las predicciones de los base.
    Vale la complejidad cuando los modelos base capturan errores
    complementarios y hay suficiente validación para entrenar el meta sin
    sobreajustar; con pocos datos, el meta sobreajusta al blend.
- **La trampa de leakage del stacking**: el metamodelo debe entrenarse sobre
  predicciones **out-of-fold** de los base. Si se entrena sobre predicciones
  del mismo conjunto donde los base se entrenaron, el meta aprende el
  sobreajuste de los base y el rendimiento medido es optimista. Regla: las
  predicciones que alimentan al meta nunca salen de filas vistas por los base.
- **Diversidad**: el ensamblado gana cuando los errores no están
  correlacionados. Diversidad de familias (lineal + árboles + NN) o de
  semillas/particiones suele rendir más que tres variantes del mismo boosting.
  La ganancia de mezclar es proporcional a la decorrelación de errores, no a
  la calidad individual.

## Calibración para valor de decisión

Un clasificador puede acertar la clase y fallar la probabilidad. Si la decisión
depende de umbrales, costes o rechazo, la probabilidad debe ser *correcta*, no
solo la clase.

- **Platt scaling**: regresión logística sobre las puntuaciones crudas.
- **Temperature scaling**: un único parámetro $T$ escala los logits,
  $p_i = \sigma(z_i/T)$, optimizado sobre validación (cross-entropy). Simple,
  preserva el ranking, ideal cuando el modelo ya ordena bien.
- **Isotonic**: regresión isotónica ajustada a las salidas del modelo.
  Flexible, pero necesita más datos de validación y puede sobreajustar el
  ranking.

{% if use_calibration %}
Este proyecto incluye calibración con temperature scaling en
`models/calibration.py`: entrena $T$ sobre validación y lo aplica al modelo
final (`make train --calibrate`). Las predicciones del modelo ya salen
calibradas para el corte de umbral.
{% endif %}

- **Optimización de umbral con matriz de coste**: en clasificación binaria
  cada celda tiene un coste (TP/TN/FP/FN):

```
          pred = 0      pred = 1
real 0    C00            C01
real 1    C10            C11
```

  El umbral óptimo es donde el coste esperado se minimiza: predecir 1 si
  $p_1\cdot C_{10} \leq p_0\cdot C_{01}$ (con $p_0 = 1 - p_1$). Con
  probabilidades bien calibradas, un umbral por coste es matemáticamente
  correcto; con probabilidades mal calibradas, es arbitrario. Por eso
  calibración y umbral se optimizan juntos sobre validación, no sobre test.

## Incertidumbre

- **Conjuntos/intervalos conformales**: distribution-free; garantizan que el
  set o intervalo cubra la verdadera respuesta con probabilidad
  $1-\alpha$ (p.ej. 90%). Se calibran sobre un split aparte y son válidos para
  cualquier modelo; el costo es decidir *el tamaño* del set o intervalo.
- **Cuándo rechazar por baja confianza**: si el modelo emite una probabilidad
  por debajo de un umbral de rechazo, se deriva la decisión a un humano, a una
  regla o a un modelo más caro. El coste de acertar con baja confianza se mide
  contra el coste de rechazar. Un sistema con conformal + rechazo sabe cuándo
  no sabe.

{% if use_conformal %}
Este proyecto trae `use_conformal`: `models/conformal.py` produce intervalos
o conjuntos de predicción calibrados sobre validación. Se aplica después del
modelo final y antes de la decisión, con el nivel $\alpha$ definido por el
coste del error.
{% endif %}

## Exprimir la evaluación

- **Métricas alineadas con el coste**: si el error cuesta de forma asimétrica,
  la métrica debe reflejarlo (F-beta, coste esperado, AUC-PR en desbalance).
  Optimizar la métrica equivocada optimiza el modelo hacia la decisión
  equivocada.
- **Nested CV para comparar con honestidad**: elegir hiperparámetros y
  comparar modelos sobre la misma muestra que evalúa infla la diferencia. La
  capa outer estima el rendimiento real (ver `formas-de-aplicar.md`).
- **Comparación estadística de modelos**: dos modelos no se comparan por un
  número mayor: se comparan con un test sobre las predicciones pareadas.
  Test de McNemar (clasificación, tablas de discordancia) o t pareada sobre
  los folds. La diferencia debe superar el ruido.
- **Seed-averaging**: entrenar y evaluar con varias semillas (3-5) y reportar
  media y desviación. Convierte "el modelo A ganó" en "A gana por X ± Y con
  solapamiento nulo/real". Sin semillas repetidas, el resultado de una semilla
  concreta no es reproducible.

{% if use_monitoring %}
En producción la exprimición no termina en offline: `monitoring/monitor.py`
(KS/chi² para drift, rendimiento vs baseline) avisa cuándo el modelo
exprimido dejó de serlo. La calibración y los umbrales se revalidan con los
datos nuevos.
{% endif %}

## La escalera de exprimido

La secuencia concreta y el criterio de parada:

1. **Baseline** ordenado y honesto (ver `formas-de-aplicar.md`).
2. **Palancas de datos**: etiquetas, más datos de los slices difíciles,
   limpieza.
3. **Palancas de features**: nuevas features de dominio, validación y
   selección.
4. **Presupuesto de tuning** de hiperparámetros (Optuna o barrido guiado por
   las interacciones de este documento), con nested CV o validación intacta.
5. **Calibración e incertidumbre**: umbral por coste, conformal si la decisión
   lo requiere.
6. **Nueva familia de modelo** o ensamblado solo si las anteriores se
   agotaron.

{% if use_optuna %}
Este proyecto trae `tuning/` con Optuna (`make tune`): define el presupuesto
de búsqueda (trials), explora con TPE y deja el mejor trial en
`models/best_params.json`. El escalón 4 se hace aquí, sobre el pipeline
ordenado.
{% endif %}

**Criterio de parada**: cuando la ganancia de métrica es menor que el ruido.
Cuantifica el ruido con seed-averaging (media ± desviación sobre semillas): si
el siguiente escalón entra dentro del intervalo de variación de las semillas,
el experimento no demuestra mejora — se detiene. Exprimir es saber cuándo la
siguiente palanca ya no se oye.

## Fuentes

- Chen, T., Guestrin, C., "XGBoost: A Scalable Tree Boosting System" —
  arXiv:1603.02754 — https://arxiv.org/abs/1603.02754
- Prokhorenkova, L., et al., "CatBoost: unbiased boosting with categorical
  features" — arXiv:1706.09516 — https://arxiv.org/abs/1706.09516
- Ke, G., et al., "LightGBM: A Highly Efficient Gradient Boosting Decision
  Tree" — arXiv:1707.09026 — https://arxiv.org/abs/1707.09026
- Guo, C., et al., "On Calibration of Modern Neural Networks" —
  arXiv:1706.04599 — https://arxiv.org/abs/1706.04599
- Angelopoulos, A. N., Bates, S., "A Gentle Introduction to Conformal
  Prediction and Distribution-Free Uncertainty Quantification" —
  arXiv:2107.07511 — https://arxiv.org/abs/2107.07511
- Cawley, G. C., Talbot, N. L. C., "On Over-fitting in Model Selection and
  Subsequent Selection Bias in Performance Evaluation" — arXiv:1006.3282 —
  https://arxiv.org/abs/1006.3282
- Izmailov, P., et al., "Averaging Weights Leads to Wider Optima and Better
  Generalization" — arXiv:1803.05407 — https://arxiv.org/abs/1803.05407
- Hinton, G., Vinyals, O., Dean, J., "Distilling the Knowledge in a Neural
  Network" — arXiv:1503.02531 — https://arxiv.org/abs/1503.02531
