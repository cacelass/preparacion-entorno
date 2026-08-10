# Métricas y evaluación

## Clasificación

### Accuracy y cuándo miente

Accuracy $= (\mathrm{TP}+\mathrm{TN})/N$ pondera igual acertar la clase
mayoritaria que la minoritaria. Con desbalance, el clasificador trivial que
siempre predice la clase dominante alcanza accuracy $=$ prevalencia de esa
clase y puede superar a un modelo útil. Con un ratio de clases por debajo de
~80/20, accuracy deja de informar: reporta además precisión/recall por clase,
la matriz de confusión y la tasa base. Solo es la métrica correcta cuando los
errores cuestan lo mismo y las clases están balanceadas.

### Precisión, recall y F-β

- **Precisión** $P=\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$: de lo que el modelo
  predice positivo, cuánto es realmente positivo.
- **Recall** $R=\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$: de lo positivo real,
  cuánto capturó.
- Trade-off: bajar el umbral sube recall y baja precisión; subirlo, al revés.
  Solo mejorar el modelo mueve ambas a la vez.

$$F_\beta = \frac{(1+\beta^2)\,P\,R}{\beta^2 P + R}$$

con $\beta$ el peso relativo del recall frente a precisión ($\beta{=}1 \to$
F1, $\beta{>}1$ favorece recall, $\beta{<}1$ favorece precisión). Es una media
armónica: penaliza que cualquiera de las dos caiga y no permite compensar una
con la otra. Elige $\beta$ por coste: si el falso negativo es más caro que el
falso positivo, $\beta>1$. Con clases raras, precisión alta con recall bajo
significa "acierta cuando se atreve, casi nunca se atreve": léelo junto a la
prevalencia de la clase.

### ROC-AUC vs PR-AUC

- **ROC-AUC**: área bajo TPR vs FPR sobre todos los umbrales. Mide
  ordenamiento y es invariante a la proporción de clases por construcción, pero
  con desbalance fuerte el área se concentra en umbrales donde la clase rara
  apenas aparece y el número deja de discriminar.
- **PR-AUC**: área bajo precisión-recall. El baseline no es 0.5 sino la
  prevalencia $p$: una PR-AUC de 0.7 con $p=0.01$ es excelente. Es la métrica
  correcta en desbalance y cuando los positivos son raros y caros.
- Regla: desbalance fuerte o positivos raros $\to$ PR-AUC; balance o FPR
  crítico (spam, moderación) $\to$ ROC-AUC. Ambas miden ranking; ninguna mide
  calibración.

### Log-loss

$$\mathrm{logloss} = -\frac{1}{N}\sum_i \big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]$$

Penaliza la confianza errónea: una predicción segura y equivocada cuesta mucho
más que una insegura. Dos modelos con el mismo AUC pueden tener logloss
distinto porque el logloss castiga la mala calibración además del ordenamiento.
Úsala cuando la decisión depende de la probabilidad (umbral, rechazo, coste),
no solo de la clase.

### Calibración

Una probabilidad está calibrada si de los ejemplos con $\hat p \approx 0.7$,
~70 % resultan positivos.

- **Brier score**: $\mathrm{BS}=\frac{1}{N}\sum_i(p_i-y_i)^2$. Combina
  calibración, resolución e incertidumbre; menor es mejor. No es absoluto:
  compáralo contra el del modelo trivial $\bar p(1-\bar p)$.
- **Reliability diagram**: agrupa las predicciones en bins (p.ej. deciles);
  en el eje $x$ la $\hat p$ media del bin y en el $y$ la fracción real de
  positivos. Sobre la diagonal = calibración perfecta; curvas por debajo =
  sobreconfianza.
- **ECE**: $\mathrm{ECE}=\sum_b \frac{|B_b|}{N}\,|\mathrm{acc}(B_b)
  - \mathrm{conf}(B_b)|$, desviación ponderada por el tamaño del bin. Útil como
  resumen, frágil al número de bins: no lo uses como única evidencia.

Calibración y ranking son ortogonales: temperature scaling e isotonic
preservan el ranking (AUC) y cambian las probabilidades. AUC alto con
calibración mala es habitual.

### Selección de umbral con matriz de costes

Sea $C_{ij}$ el coste de predecir $j$ siendo la clase real $i$. El umbral
óptimo minimiza el coste esperado: predecir 1 si

$$C_{01}\,p_0 + C_{11}\,p_1 \;\le\; C_{00}\,p_0 + C_{10}\,p_1,$$

es decir, cuando el coste esperado de predecir 1 no supera al de predecir 0
(con $p_0 = 1 - p_1$). Con $C_{00}=C_{11}=0$, predecir 1 si $C_{01}\,p_0 \le
C_{10}\,p_1$. Con probabilidades calibradas el umbral óptimo es el ratio de
costes y es matemáticamente correcto; con probabilidades mal calibradas se
barre el umbral sobre validación minimizando el coste empírico. El umbral se
fija siempre sobre validación, nunca sobre test.

### Métricas top-k y ranking

- **MAP** (Mean Average Precision): por cada query, promedio de la precisión en
  las posiciones donde hay relevantes; MAP es la media sobre todas las queries.
- **NDCG@k**: $\mathrm{DCG}@k=\sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$
  normalizado por el DCG del orden ideal ($\mathrm{NDCG}=\mathrm{DCG}/\mathrm{IDCG}$).
  Acepta relevancia graduada y castiga que un relevante quede abajo.

Evalúan el orden, no la calibración ni el umbral; se usan en recomendación y
ranking. MAP asume relevancia binaria; NDCG tolera relevancias graduadas.

### Promedios macro, micro y ponderado

- **Macro**: media de la métrica por clase sin ponderar. Da peso igual a las
  clases raras; es la lectura honesta en desbalance porque el error en la clase
  rara se ve y no se diluye.
- **Micro**: agrega los conteos globales (ΣTP, ΣFP, ΣFN) y calcula la métrica.
  En multiclase equivale a accuracy; lo domina la clase mayoritaria.
- **Weighted**: media por clase ponderada por el soporte. Engaña: un modelo que
  acierta la mayoría y falla el resto saca weighted alto.
- Regla: macro cuando importa la clase rara; micro/weighted solo si la
  mayoritaria es la clase de negocio.

## Regresión

### MSE, RMSE y sensibilidad a outliers

$$\mathrm{MSE}=\frac{1}{n}\sum_i(y_i-\hat y_i)^2,\qquad \mathrm{RMSE}=\sqrt{\mathrm{MSE}}$$

El error cuadrático da peso $e^2$ a cada residuo: un error de magnitud 10 pesa
100. RMSE mantiene la unidad de $y$. Ambos priorizan la cola de errores
grandes: si los outliers son datos reales y caros, RMSE es correcto; si son
ruido o errores de registro, pocos puntos mandan en la métrica y en el modelo.

### MAE

$$\mathrm{MAE}=\frac{1}{n}\sum_i |y_i - \hat y_i|$$

Lineal: cada residuo pesa lo que vale; más robusto al extremo que RMSE. La
diferencia RMSE − MAE cuantifica la cola: si RMSE ≈ MAE los residuos son
homogéneos; si RMSE ≫ MAE hay pocos errores grandes dominando. MAE es la
pérdida correcta cuando la predicción óptima es la mediana, no la media.

### R² y sus modos de fallo

$$R^2 = 1 - \frac{\mathrm{SS}_{res}}{\mathrm{SS}_{tot}}$$

mide cuánta varianza de $y$ reduce el modelo frente a predecir siempre la
media. Modos de fallo habituales:

- Depende de la varianza de $y$: con $y$ poco variable, un modelo casi perfecto
  da $R^2$ bajo; con $y$ muy variable, $R^2$ alto con errores absolutos grandes.
- No mide el error absoluto: no es comparable entre datasets ni entre
  transformaciones de $y$ (log, cuadrado).
- Outliers en $y$ inflan $\mathrm{SS}_{tot}$ y abultan $R^2$.
- $R^2$ fuera de muestra negativo es válido y común con sobreajuste o datos
  fuera de distribución: el modelo empeora a predecir la media. No es un bug.

### MAPE y la trampa de la división por cero

$$\mathrm{MAPE}=\frac{1}{n}\sum_i \frac{|y_i-\hat y_i|}{|y_i|}$$

Divide por el valor real: con $y_i=0$ el término explota y, con $y_i$ pequeño,
un error pequeño da un porcentaje enorme. Además es asimétrico (sobreestimar
con $y$ pequeña castiga más). Alternativas: excluir los $y=0$ (sesga la
métrica), usar wMAPE $=\frac{\sum|y_i-\hat y_i|}{\sum|y_i|}$ (sin división por
cero, estable) o SMAPE.

### SMAPE

$$\mathrm{SMAPE}=\frac{1}{n}\sum_i \frac{2\,|y_i-\hat y_i|}{|y_i|+|\hat y_i|}$$

Simétrico por construcción y acotado; evita la división por cero salvo
$y_i=\hat y_i=0$. No es perfecto: con errores grandes de signos opuestos el
denominador no compensa del todo y sigue penalizando de forma asimétrica. Útil
como métrica robusta de reporting, no como función de pérdida de negocio.

### Pérdida pinball y cuantiles

$$\rho_q(u) = \begin{cases} q\,u & u \ge 0 \\ (q-1)\,u & u < 0 \end{cases}, \qquad u = y - \hat y$$

Minimizar $\rho_q$ entrena un modelo para el cuantil $q$, no para la media.
Entrenando $q=0.5$, $0.1$ y $0.9$ se obtiene un intervalo (10–90 %) sin
supuestos de distribución. Evalúa cada cuantil con la pérdida pinball media y
con la cobertura empírica: un cuantil bien calibrado cubre ~$q$ de los puntos.

{% if use_conformal %}
### Cobertura de intervalos conformales

Con conformal prediction la cobertura está garantizada en distribución
($1-\alpha$). En la práctica evalúa dos números: la **cobertura empírica**
(fracción de $y_i$ dentro del intervalo, debe rondar $1-\alpha$) y la
**anchura media** (el intervalo más corto que cumple la cobertura). Cobertura
$1-\alpha$ con intervalos enormes es trivial; el modelo útil minimiza la
anchura a cobertura fija. El calibrado se hace sobre un split de validación
aparte, nunca sobre train ni test.
{% endif %}

{% if ml_type == 'no_supervisado' %}
## Clustering

### Validez interna

Sin etiquetas, la validez interna premia la geometría que el propio algoritmo
busca:

- **Silhouette**: $s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}$, con $a(i)$ la
  distancia media intra-cluster y $b(i)$ la media al cluster vecino más
  cercano. En $[-1,1]$; media alta $\to$ clusters compactos y separados. Asume
  clusters convexos y métrica euclídea; con formas alargadas o anidadas, engaña.
- **Davies–Bouldin**: $\mathrm{DB}=\frac{1}{K}\sum_k \max_{j\ne k}
  \frac{\sigma_k+\sigma_j}{d(C_k,C_j)}$, con $\sigma$ la dispersión media y $d$
  la distancia entre centroides. Menor es mejor; favorece clusters esféricos y
  castiga el ruido.
- **Calinski–Harabasz**: $\mathrm{CH}=\frac{\mathrm{tr}(S_B)/(K-1)}
  {\mathrm{tr}(S_W)/(n-K)}$, ratio de dispersión entre/dentro. Mayor es mejor;
  asume variabilidad homogénea intra-cluster y se rompe con densidades
  desiguales.

Cada índice codifica una definición distinta de "cluster"; todos asumen formas
convexas y sufren con densidades heterogéneas y ruido. Además son criterios de
optimización: KMeans minimiza WCSS y cualquier índice basado en dispersión
tiende a premiar la geometría de KMeans, no la estructura real.

### Validez externa (con etiquetas)

- **ARI** (Adjusted Rand Index): $\mathrm{ARI}=\frac{\mathrm{RI}-E[\mathrm{RI}]}
  {\max(\mathrm{RI})-E[\mathrm{RI}]}$; corregido por azar, esperado 0, perfecto
  1, puede ser negativo. Robusto a tamaños de cluster desiguales; elección por
  defecto.
- **NMI** (Normalized Mutual Information): $\mathrm{NMI}=\frac{2\,\mathrm{MI}}
  {H(U)+H(V)} \in [0,1]$; mide la información compartida entre particiones. No
  corregido por azar (el MI esperado no es 0); útil cuando los tamaños difieren
  mucho.

Si existen etiquetas de referencia, son la evidencia más fuerte: miden contra
una verdad externa, no contra el criterio que el algoritmo optimiza.

### Por qué no hay una métrica interna perfecta

La validez interna optimiza el mismo criterio que el algoritmo satisface, así
que no hay juicio independiente: medir KMeans con silhouette premia
particiones convexas aunque la estructura real sea otra. La vía honesta:
(1) fijar qué forma de grupo importa para el negocio, (2) usar 2–3 índices
internos coherentes con la métrica del algoritmo, (3) contrastar con etiquetas
externas si existen y (4) sanidad visual (proyección 2D de muestras). Ninguna
cifra sustituye el juicio de si el cluster tiene sentido de negocio.

{% endif %}

## Comparar modelos con honestidad

### La falacia de "A ganó a B en una corrida"

Una corrida es una realización del ruido. Con varianza de entrenamiento, A
puede ganar el 60 % de las veces y aun así la corrida concreta decir lo
contrario. Un solo número no separa señal de ruido; las diferencias se reportan
con su incertidumbre: media ± desviación sobre repeticiones y un contraste
estadístico.

### Contrastes sobre resultados pareados

- **McNemar** (clasificación): sobre la tabla de discordancia $n_{01}$ (A
  acierta, B falla) y $n_{10}$ (B acierta, A falla):
  $\frac{(|n_{01}-n_{10}|-1)^2}{n_{01}+n_{10}}\sim\chi^2_1$. Detecta si A y B
  difieren sobre la misma muestra; no cuantifica la magnitud.
- **t pareado / Wilcoxon** sobre scores repetidos: evalúa A y B sobre los
  mismos $k$ folds (o semillas) y contrasta las diferencias pareadas. El t
  pareado asume normalidad aproximada de las diferencias; con pocas
  repeticiones, Wilcoxon signed-rank. El emparejamiento (mismos folds, mismas
  semillas, mismo preprocesado) es lo que da poder al contraste.

### Seed-averaging y tamaño del efecto

Entrenar 3–5 semillas por modelo y reportar media ± desviación convierte "A
ganó" en "A gana por $0.03 \pm 0.02$". La significación sin tamaño del efecto
no decide: un $p<0.01$ con efecto 0.001 es irrelevante para el negocio.
Reporta la diferencia normalizada (la diferencia sobre su desviación conjunta)
con su intervalo.

### La trampa de las comparaciones múltiples

Comparar $M$ modelos y quedarse con el mejor infla la probabilidad de un falso
descubrimiento. Si varios candidatos compiten contra el mismo baseline, usa el
test de **Friedman** sobre los rankings por fold/semilla y después **Nemenyi**
para comparaciones por pares, que controlan el error de familia. Regla
práctica: si un tuner probó 100 configuraciones, el "mejor" no es evidencia;
solo la validación anidada (ver `validacion.md`) da un número honesto.

## Disciplina del conjunto de test

- El test es **confirmación final, no selección**. Cada uso del test para
  decidir (modelo, umbral, features) lo contamina: la estimación se vuelve
  optimista en proporción a cuántas decisiones se tomaron con él.
- **Peeking repetido**: ver el número del test, ajustar y volver es memorizar
  su ruido; a los pocos reusos, el test estima el ajuste a ese conjunto, no la
  generalización.
- La métrica que se optimiza en validación no es la que se reporta en test: son
  estimaciones de cosas distintas. Se reporta la métrica de negocio en test una
  sola vez, con su intervalo.

## Práctica

- **Alinear la métrica con el coste de negocio**: cada error cuesta (dinero,
  tiempo, riesgo). Si el coste es asimétrico, usa F-β, coste esperado con
  matriz de costes o pérdida cuantil; optimizar otra métrica optimiza el modelo
  hacia otra decisión.
- **Reportar barras de error**: métrica ± desviación sobre semillas/folds, con
  el método y el número de repeticiones. Un número sin varianza no es
  reproducible.
- **Elegir el umbral con coste**: en clasificación, fijar el umbral sobre
  validación minimizando el coste esperado, nunca el 0.5 por defecto. Con
  probabilidades calibradas el umbral óptimo es el ratio de costes; con las mal
  calibradas, se barre sobre validación.

## Fuentes

- Saito, T., Rehmsmeier, M., *The Precision-Recall Plot Is More Informative
  than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*,
  PLoS ONE 2015. arXiv:1405.4084. https://arxiv.org/abs/1405.4084
- Boyd, K., Eng, K.-H., Page, C. D., *Area under the Precision-Recall Curve:
  Point Estimates and Confidence Intervals*. arXiv:1506.05390.
  https://arxiv.org/abs/1506.05390
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q., *On Calibration of Modern
  Neural Networks*. arXiv:1706.04599. https://arxiv.org/abs/1706.04599
- Wang, Y., et al., *A Theoretical Analysis of NDCG Type Ranking Measures*.
  arXiv:1304.6480. https://arxiv.org/abs/1304.6480
- Koenker, R., Bassett, G., *Regression Quantiles*, Econometrica 1978. Sin
  arXiv. https://doi.org/10.2307/1913643
- Botchkarev, A., *Performance Metrics (Error Measures) in Machine Learning
  Regression, Forecasting and Prognostics: Properties and Typology*.
  arXiv:1809.03006. https://arxiv.org/abs/1809.03006
- Corani, G., Benavoli, A., Demšar, J., Mangili, F., Zaffalon, M.,
  *Statistically Significant Comparisons of Learning Algorithms: a Frequentist
  or a Bayesian Approach?*. arXiv:1811.12808. https://arxiv.org/abs/1811.12808
- Demšar, J., *Statistical Comparisons of Classifiers over Multiple Data Sets*,
  JMLR 2006. Sin arXiv. https://jmlr.org/papers/v7/demsar06a.html
- Domingos, P., *A Few Useful Things to Know about Machine Learning*.
  arXiv:1206.5533. https://arxiv.org/abs/1206.5533
- Rousseeuw, P. J., *Silhouettes: A Graphical Aid to the Interpretation and
  Validation of Cluster Analysis*. Sin arXiv.
  https://doi.org/10.1016/0377-0427(87)90125-7
- Hubert, L., Arabie, P., *Comparing Partitions*, Journal of Classification
  1985. Sin arXiv. https://doi.org/10.1007/BF01908075
