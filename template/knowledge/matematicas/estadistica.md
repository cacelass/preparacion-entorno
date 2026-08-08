# Estadística

Referencia profunda para el agente `lider`: estimación, intervalos,
contraste de hipótesis, bootstrap y regresión desde el punto de vista
estadístico. Cada sección conecta la teoría con cuándo aplicarla en un
proyecto DS real y con los abusos más frecuentes.

## Población, muestra y estimadores

Población: la distribución generadora $F$, desconocida. Muestra:
$X_1, \dots, X_n$ i.i.d. $\sim F$. **Parámetro** $\theta = \theta(F)$ (lo que
queremos aprender); **estimador** $\hat\theta = T(X_1, \dots, X_n)$ (un
estadístico, función de la muestra). Punto clave: $\hat\theta$ es aleatorio,
$\theta$ es fijo.

- **Sesgo**: $\mathrm{Bias}(\hat\theta) = \mathbb{E}[\hat\theta] - \theta$.
- **Varianza**: $\mathrm{Var}(\hat\theta)$ — la precisión del estimador.
- **MSE**: $\mathrm{MSE}(\hat\theta) = \mathbb{E}[(\hat\theta - \theta)^2] =
  \mathrm{Var}(\hat\theta) + \mathrm{Bias}(\hat\theta)^2$: sesgo y varianza
  compiten; sacrificar algo de sesgo por mucha varianza puede bajar el MSE.
- **Consistencia**: $\hat\theta_n \xrightarrow{p} \theta$ (convergencia en
  probabilidad). Insesgado no implica consistente, y al revés.
- **Eficiencia**: menor varianza entre los insesgados; la cota de
  Cramér–Rao marca el mínimo alcanzable (lo satura el MLE asintóticamente).

| Estimador | Sesgo | Nota |
|---|---|---|
| $\bar X$ para $\mu$ | 0 | varianza $\sigma^2/n$; óptimo lineal por Gauss–Markov |
| $s^2 = \frac{1}{n-1}\sum (X_i-\bar X)^2$ para $\sigma^2$ | 0 | insesgado (corrección de Bessel) |
| $\hat\sigma^2 = \frac{1}{n}\sum (X_i-\bar X)^2$ | $-\sigma^2/n$ | sesgado pero con MENOR MSE (normales) |

Ejemplo de trade-off: en regresión, la contracción (ridge) introduce sesgo
para reducir varianza y el MSE total baja. Abuso: creer que "insesgado es
siempre mejor", o reportar solo la varianza de un estimador sin su sesgo.

## Ley de los grandes números (LLN) y teorema central del límite (CLT)

**LLN** (débil): si $\mathbb{E}|X_1| < \infty$, $\bar X_n \xrightarrow{p}
\mathbb{E}[X_1]$. (La versión fuerte es convergencia casi segura.) Justifica
usar medias empíricas como estimadores de esperanzas — la base del riesgo
empírico en ML.

**CLT**: si $\mathbb{E}[X_1] = \mu$ y $\mathrm{Var}(X_1) = \sigma^2 <
\infty$, con $X_1, \dots, X_n$ i.i.d.,

$$ \frac{\bar X_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d}
   \mathcal{N}(0, 1) $$

es decir $\bar X_n \approx \mathcal{N}(\mu, \sigma^2/n)$. El error estándar
deca como $1/\sqrt{n}$: **para dividir el error por 2 hay que cuadruplicar
$n$**. Es una convergencia en distribución, asintótica: no garantiza nada
para $n$ fijo.

**Cuándo falla o es lento**:
- **Colas pesadas**: sin varianza finita (Cauchy, Pareto con
  $\alpha \le 2$) no hay CLT estándar; con varianza finita pero colas muy
  pesadas la convergencia es extremadamente lenta ($n$ del orden de miles).
- **Dependencia**: correlaciones temporales o espaciales rompen el i.i.d.;
  la tasa efectiva $n_{\text{eff}} < n$ y el SE asintótico queda corto.
- **Skew extremo y outliers**: la normal tarda en aparecer; un único valor
  extremo domina $\bar X$.
- **Funciones no lisas de medias**: usar delta method o bootstrap.

Práctica: el CLT justifica SE e intervalos asintóticos en regresión y tests;
nunca asumas normalidad "porque $n$ es grande" sin comprobarlo.

## Intervalos de confianza

Un IC al $1 - \alpha$ es un **procedimiento**: repetido sobre muestras
independientes, una fracción $1-\alpha$ de los intervalos construidos así
contiene a $\theta$.

$$ \bar X \pm z_{\alpha/2}\,\frac{\sigma}{\sqrt{n}} \qquad
   \text{($\sigma$ conocida)} $$

$$ \bar X \pm t_{n-1,\alpha/2}\,\frac{s}{\sqrt{n}} \qquad
   \text{($\sigma$ estimada por $s$)} $$

- **z vs t**: con varianza conocida, z; si se estima, t — bajo normalidad
  $\frac{\bar X - \mu}{s/\sqrt{n}} \sim t_{n-1}$. Para $n$ grande ambas
  coinciden; con $n$ pequeño la t da intervalos más anchos (colas más
  gordas), que es lo correcto.
- **Interpretación correcta**: "el procedimiento captura $\theta$ el 95% de
  las veces", no "$\mathbb{P}(\theta \in [a, b]) = 0.95$": en el enfoque
  frecuentista $\theta$ es fijo y el intervalo es la variable aleatoria.
- El ancho decae como $1/\sqrt{n}$; un IC ancho significa muestra
  insuficiente, no "ausencia de efecto".

Abuso: dar un IC sin decir el método (normal, t, bootstrap) ni si se
verificaron los supuestos; o usar z cuando la varianza se estima y $n$ es
pequeño.

## Contraste de hipótesis

$H_0$: hipótesis nula (sin efecto, modelo restringido). $H_1$: alternativa
(efecto presente). Se decide comparando un estadístico con su distribución
bajo $H_0$.

| Decisión | $H_0$ verdadera | $H_1$ verdadera |
|---|---|---|
| No rechazar $H_0$ | correcta | **error tipo II** ($\beta$) |
| Rechazar $H_0$ | **error tipo I** ($\alpha$) | correcta — potencia $1-\beta$ |

- **Error tipo I** ($\alpha$): rechazar $H_0$ cuando es verdadera (falso
  positivo). Se fija a priori (0.05, 0.01).
- **Error tipo II** ($\beta$): no detectar un efecto real. La **potencia**
  $1 - \beta$ crece con $n$, con el tamaño del efecto y con la precisión del
  diseño.
- **p-value**: probabilidad, bajo $H_0$, de observar un estadístico tan
  extremo o más que el observado. NO es la probabilidad de que $H_0$ sea
  cierta, ni el tamaño del efecto, ni la probabilidad de replicar.

**Abusos y malentendidos**:
- **p-hacking**: probar muchas hipótesis, decidir el análisis tras ver los
  datos o parar "cuando sale significativo" infla el error tipo I por encima
  de $\alpha$; las correcciones de comparaciones múltiples existen para esto.
- **"No significativo" ≠ "no hay efecto"**: ausencia de evidencia no es
  evidencia de ausencia. Para afirmar equivalencia se usan tests de
  equivalencia (TOST): fija un margen y comprueba que el efecto cabe dentro.
- **"Significativo" ≠ "importante"**: con $n$ enorme, efectos irrelevantes
  salen significativos. Reporta siempre el tamaño del efecto y su IC.
- **Dicotomizar el p**: $0.049$ y $0.051$ no son cualitativamente
  distintos; reporta el valor exacto y el IC.
- Un p-value aislado no mide replicabilidad: entre réplicas varía mucho.

## Tests y supuestos

| Test | Pregunta | Supuestos | Cuándo usarlo |
|---|---|---|---|
| t de Welch | dif. de medias, 2 grupos | independencia, normalidad aprox. | default 2 grupos, var. libres |
| t de Student | idem | además varianzas iguales | solo si se justifica (Levene) |
| ANOVA (F) | igualdad de medias, 3+ grupos | independ., norm., homoced. | grupos; post hoc corregido |
| chi-cuadrado | independencia / bondad de ajuste | esperados $\ge 5$ (aprox.) | tablas de contingencia |
| Mann–Whitney U | dominancia estocástica, 2 grupos | ordinal o continuo; sin normal. | no paramétrico |

**Cuándo pasar a no paramétricos**: normalidad claramente violada (outliers,
colas pesadas), datos ordinales, escalas con ceros arbitrarios. Coste:
pierden potencia si los datos eran normales (eficiencia relativa ~0.95 de la
t de Student) pero ganan mucho con colas pesadas. Welch es robusto a
varianzas desiguales; ANOVA no lo es — con heterocedasticidad, Welch o
Kruskal–Wallis.

## Bootstrap

Idea: la distribución empírica $\hat F_n$ es la mejor aproximación a $F$;
remuestrear con reemplazo aproxima la distribución muestral de $\hat\theta$.
Repite $B$ veces (para cuantiles, $B \ge 1000$):

1. Muestrear $X^*_1, \dots, X^*_n$ con reemplazo de la muestra original.
2. Recalcular el estadístico $\hat\theta^*_b$.
3. Usar la distribución de $\{\hat\theta^*_b\}$ como aproximación de la
   distribución de $\hat\theta$.

- **Percentile CI**: $[\hat\theta^*_{(\alpha/2)},\;
  \hat\theta^*_{(1-\alpha/2)}]$.
- **BCa** (bias-corrected and accelerated): corrige sesgo y asimetría;
  preferible al percentile en general.
- Variantes: bootstrap de residuos en regresión (diseño fijo), block
  bootstrap para series temporales, bootstrap por grupos si la unidad de
  muestreo no es la fila.

**Cuándo falla**:
- Estadísticos no suaves (max, min, cuantiles extremos): la empírica no
  captura la cola y el CI colapsa.
- Dependencia temporal/espacial: sin bloquear, el bootstrap subestima la
  varianza.
- Muestras pequeñas con outliers: el remuestreo amplifica valores extremos.
- Extrapolación: el bootstrap estima incertidumbre de $\hat\theta$ sobre la
  muestra observada, no fuera de su rango.
- Selección de modelos: el bootstrap del "índice del mejor modelo" no da
  comparaciones válidas entre candidatos.

## Comparaciones múltiples

Con $m$ tests, el error tipo I por test se acumula. Dos familias de control:

- **FWER** (family-wise error rate): probabilidad de al menos un falso
  positivo entre todos los rechazos. Control fuerte con **Bonferroni**:
  rechazar $H_0^{(i)}$ si $p_i \le \alpha/m$. Simple y válido siempre, pero
  conservador: con $m$ grande pierde casi toda la potencia.
- **FDR** (false discovery rate): proporción esperada de falsos positivos
  entre los rechazados. **Benjamini–Hochberg**: ordena
  $p_{(1)} \le \dots \le p_{(m)}$ y rechaza hasta el mayor $i$ con
  $p_{(i)} \le \frac{i}{m}\,\alpha$. Controla el FDR bajo independencia o
  dependencia positiva (PRDS) y tiene mucha más potencia que Bonferroni.

| Método | Controla | Conservador | Uso |
|---|---|---|---|
| Bonferroni | FWER | sí | resultado confirmatorio (el hallazgo principal) |
| Benjamini–Hochberg | FDR | no | screening exploratorio (features, correlaciones) |

Abuso: aplicar Bonferroni a exploraciones masivas (miles de features) mata
todo el análisis; aplicar FDR sin declarar el nivel ni el supuesto de
dependencia. Reporta siempre el número de tests realizados.

## Tamaño del efecto vs significancia

El p-value mezcla dos cosas: la magnitud del efecto y la precisión con la que
se estima ($n$). Un efecto diminuto con $n$ gigantesco sale "significativo";
un efecto grande con $n$ pequeño no. Por eso:

- Reporta el estimador del efecto y su intervalo de confianza, no solo p.
- **d de Cohen** para diferencias de medias
  ($\approx (\bar x_1 - \bar x_2)/s_p$), correlación, odds ratio, AUC:
  adimensionales o en unidades del problema.
- Un IC estrecho alrededor de un efecto pequeño dice "efecto pequeño y bien
  estimado"; el p dirá "significativo" y no distinguirá ambos casos.
- "Significativo" es una decisión estadística; la relevancia práctica la
  decide el dominio. Con $n$ enorme todo es significativo.

## Regresión lineal como estadística

Modelo: $y = X\beta + \varepsilon$, con $\mathbb{E}[\varepsilon \mid X] = 0$
y $\mathrm{Var}(\varepsilon) = \sigma^2 I$.

**OLS** (mínimos cuadrados ordinarios) minimiza $\|y - X\beta\|_2^2$; su
solución son las **ecuaciones normales**:

$$ \hat\beta = (X^\top X)^{-1} X^\top y $$

Numéricamente se resuelve con QR/SVD, nunca formando $(X^\top X)^{-1}$: se
cuadra el condicionamiento (ver `algebra-lineal.md`).

**Teorema de Gauss–Markov**: bajo linealidad, $\mathbb{E}[\varepsilon \mid
X] = 0$, homocedasticidad y no autocorrelación, OLS es el mejor estimador
lineal insesgado (BLUE). No requiere normalidad; la normalidad solo se añade
para los tests t/F exactos y los intervalos.

**Diagnósticos** (sobre los residuos $e = y - X\hat\beta$):
- **Residuos vs ajuste**: patrones (embudo, curva) indican
  heterocedasticidad o no linealidad; puntos extremos, outliers e influencia
  (leverage, distancia de Cook).
- **Heterocedasticidad** (varianza no constante): los SE y los tests dejan
  de ser válidos; se detecta con gráficos o Breusch–Pagan; se corrige con
  errores robustos (White/HC), WLS o transformando la respuesta.
- **Normalidad de residuos**: Q-Q plot o Shapiro; afecta a la inferencia
  exacta, no a la consistencia del estimador.
- **Multicolinealidad**: **VIF** $= 1/(1 - R^2_j)$ con $R^2_j$ la regresión
  de la feature $j$ sobre las demás; VIF > 10 (algunos usan > 5) →
  coeficientes inestables y SE inflados. No sesga la predicción ni el $R^2$;
  se mitiga con ridge o eliminando variables redundantes.

**$R^2$ y $R^2$ ajustado**:

$$ R^2 = 1 - \frac{\sum_i e_i^2}{\sum_i (y_i - \bar y)^2}, \qquad
   \bar R^2 = 1 - (1 - R^2)\,\frac{n - 1}{n - p - 1} $$

$R^2$ crece con cada variable añadida; $\bar R^2$ penaliza el número de
parámetros $p$. Compara modelos con AIC/BIC y validación cruzada, no solo
con $R^2$.

**Abusos**: interpretar $\hat\beta_j$ como efecto causal (hace falta
identificación — ver `causalidad.md`); extrapolar fuera del rango de $X$;
usar el $R^2$ para "explicar causalidad"; leer coeficientes individuales con
multicolinealidad fuerte.

## Práctica: A/B testing

- **Peeking / optional stopping**: observar p-values mientras corre el
  experimento y parar "cuando sale significativo" infla el error tipo I por
  encima de $\alpha$. Soluciones: tamaño de muestra fijo pre-registrado, o
  métodos secuenciales con p-values always-valid (Johari et al.).
- **CUPED**: usa covariables pre-experimento (el valor de la métrica antes
  del experimento) en una regresión para reducir la varianza del estimador;
  gana potencia sin más muestra (Deng et al., KDD 2013). Requisito: la
  covariable debe ser independiente del tratamiento y correlacionada con el
  outcome.
- **Múltiples métricas y variantes**: pre-especifica la métrica primaria o
  aplica FDR; la multiplicidad también cuenta.
- **Asociación vs causalidad**: un RCT bien ejecutado identifica el efecto
  porque la aleatorización rompe la confusión. Un análisis observacional
  describe asociaciones; la asociación estadística no establece causa
  (confundidores, selección). Detalle en `causalidad.md`.

{% if use_calibration %}
**Con calibración activa en este proyecto:** las probabilidades del modelo
entran en decisiones umbral; la calibración no arregla un modelo mal
entrenado. Mide ECE/Brier en validación y separa el calibration set del
train y de la selección de hiperparámetros.
{% endif %}

{% if use_conformal %}
**Con conformal activo en este proyecto:** los intervalos predichos tienen
cobertura $1-\alpha$ marginal bajo exchangeability, sin supuestos de
distribución; reutilizar el calibration set para otra tarea (early stopping,
selección de modelos) rompe la garantía. Son intervalos de predicción, no de
parámetros: no los confundas con los IC de un coeficiente.
{% endif %}

## Fuentes

- G. Casella, R. L. Berger, *Statistical Inference*, 2nd ed., Duxbury, 2002.
- R. Durrett, *Probability: Theory and Examples*, 5th ed., Cambridge, 2019.
- A. W. van der Vaart, *Asymptotic Statistics*, Cambridge, 1998.
- B. Efron, R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman &
  Hall/CRC, 1993.
- Y. Benjamini, Y. Hochberg, *Controlling the false discovery rate: a
  practical and powerful approach to multiple testing*, JRSS-B 57(1), 1995.
  Sin arXiv.
- R. Johari, P. Pekelis, D. J. Walsh, *Always Valid Inference: Continuous
  Monitoring of A/B Tests* (2015).
  arXiv:1512.04922 — https://arxiv.org/abs/1512.04922
- A. Deng, Y. Xu, R. Kohavi, T. Walker, *Improving the Sensitivity of Online
  Controlled Experiments by Utilizing Pre-Experiment Data*, KDD 2013.
  Sin arXiv.
- R. Kohavi, D. Tang, Y. Xu, *Trustworthy Online Controlled Experiments*,
  Cambridge Univ. Press, 2020.
- T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical
  Learning*, 2nd ed., Springer, 2009.
