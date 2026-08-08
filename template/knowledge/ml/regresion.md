{% if task_type == 'regresion' %}
# Regresión: pérdidas, transformaciones e intervalos

En regresión "el modelo" son tres decisiones acopladas: qué pérdida define
el óptimo, qué transformación se aplica al target y cómo se reporta la
incertidumbre. Elegirlas a ciegas (MSE + media + sin intervalos) produce
modelos correctos para un problema que no era el suyo.

## Pérdidas

| Pérdida | Fórmula | Óptimo de $c$ | Comportamiento |
|---|---|---|---|
| MSE | $(y - \hat{y})^2$ | Media condicional | Outliers penalizados al cuadrado; domina errores grandes |
| MAE | $|y - \hat{y}|$ | Mediana condicional | Robusta a outliers; no diferenciable en 0 |
| Huber | cuadrática en $|u|\le\delta$ | Intermedio | Suave; $\delta$ = escala del ruido normal |
| Pinball $\rho_\tau$ | $u(\tau - \mathbb{1}_{u<0})$ | Cuantil $\tau$ | Estima cuantiles; base de intervalos |

Con $u = y - \hat{y}$. La elección de la pérdida fija **qué estadístico de la
distribución condicional** se estima:

- **MSE** predice la media: óptimo cuando el error se distribuye simétrico y
  sin colas, y cuando la pérdida de negocio es cuadrática.
- **MAE/Huber** predicen cerca de la mediana: correctas con colas pesadas o
  outliers; Huber necesita $\delta$ (escala del ruido "normal").
- **Pinball** con $\tau$ estima el cuantil $\tau$-ésimo: dos de ellas
  ($\tau = \alpha/2$ y $\tau = 1 - \alpha/2$) dan un intervalo
  $[q_{\alpha/2},\, q_{1-\alpha/2}]$ sin supuestos de distribución.

Una pista de que la pérdida está mal elegida: un modelo que predice media
sobre un target con cola larga devuelve valores raros para la mayoría de
observaciones (la media la tiran pocos puntos extremos).

## Transformaciones del target

- **Log target**: modelar $\log y$ hace el problema más simétrico y positivo.
  Al invertir, la predicción $\exp(\hat{\mu})$ es la **media geométrica**,
  no la media aritmética: si $\log y \sim \mathcal{N}(\mu, \sigma^2)$,
  $E[y] = \exp(\mu + \sigma^2/2)$ y la mediana es $\exp(\mu)$. Decidir si
  reportar media (con la corrección) o mediana es una decisión de negocio,
  no un detalle técnico.
- **Box-Cox**: $y^{(\lambda)} = (y^\lambda - 1)/\lambda$ (límite
  $\lambda \to 0$ es el log); $\lambda$ se estima por máxima verosimilitud
  sobre un rango. Requiere $y > 0$; Yeo-Johnson extiende a valores
  negativos. Objetivo: normalizar el error, no la target per se — el supuesto
  real es sobre la distribución de los residuos.
- **Media vs mediana como predicción puntual**: con pérdida asimétrica
  (sobre-predictir es peor que sub-predictir) el óptimo puntual no es la
  media: es un cuantil. "El modelo predice mal" muchas veces es "se está
  evaluando con la métrica equivocada para el decisor".

## Heterocedasticidad y regresión ponderada

OLS es insesgado aunque $\mathrm{Var}(\varepsilon \mid x) = \sigma^2(x)$ no
sea constante, pero sus errores estándar son incorrectos y pondera por igual
zonas de distinta fiabilidad. Si la varianza depende de $x$:

- **WLS** (mínimos cuadrados ponderados): minimizar
  $\sum_i w_i (y_i - x_i^\top\beta)^2$ con $w_i = 1/\sigma^2(x_i)$ — dar más
  peso a las regiones de menor varianza. Requiere estimar $\sigma^2(x)$
  (p. ej. ajustando un modelo de residuos al cuadrado).
- **Errores estándar robustos** (Huber-White, sandwich) para inferencia sin
  modelar la varianza.
- Modelar la varianza por separado (mean-variance modeling) cuando la
  heterocedasticidad es la señal de negocio (p. ej. predicciones de riesgo).

**Diagnóstico de residuos** (imprescindible antes de confiar en el modelo):
residuos vs. ajustados (patrón de embudo = heterocedasticidad), QQ plot
(normalidad — necesaria para intervalos, no para consistencia), residuos vs.
cada feature (estructura no capturada). Un modelo con $R^2$ alto y residuos
con forma está mal especificado.

## Regularización en regresión

- **Ridge** encoge los coeficientes hacia 0 sin anularlos: estabiliza la
  inversa de $X^\top X$ cuando las features están correlacionadas (la forma
  cerrada es $(X^\top X + \lambda I)^{-1}X^\top y$).
- **Lasso** selecciona features (norma $L_1$, solución sparse por descenso
  de coordenadas) pero encoge también los sobrevivientes: tras seleccionar,
  reajustar OLS sobre el conjunto elegido (relaxed lasso) reduce el sesgo.
- La regularización no es "una opción más": en regresión con $p$ alto o
  features correlacionadas, el modelo sin regularizar predice peor aunque
  ajuste mejor en train.

## Modos de fallo de las métricas

- **MAPE** $\frac{100}{n}\sum |y - \hat{y}|/|y|$: división por cero cuando
  $y = 0$, y cuando $y$ es pequeño el error relativo explota aunque el error
  absoluto sea chico. Además es asimétrica (sobre-predictir a $y$ pequeño
  penaliza desproporcionadamente). Alternativas: SMAPE, o MAPE ponderada.
- **RMSE vs MAE**: RMSE pondera los errores grandes cuadráticamente; si la
  pérdida real es lineal, el RMSE infla unos pocos outliers. Comparar ambos:
  si difieren mucho, la cola de errores grandes domina — decidir si eso es
  el negocio o un artefacto.
- **Error relativo vs absoluto**: cuando la escala del target varía entre
  grupos (precio de 3€ vs 3000€), el error absoluto mide mal; usar error
  relativo o métricas por grupo. La elección macro/micro y las métricas de
  evaluación están en
  [metricas-y-evaluacion.md](../modelos/metricas-y-evaluacion.md).

## Intervalos de predicción

- **Quantile regression**: ajustar dos modelos pinball ($\tau = \alpha/2$,
  $\tau = 1-\alpha/2$) da el intervalo de predicción directamente. Es
  distribution-free pero la cobertura es aproximada (los dos cuantiles se
  estiman por separado y no se garantiza $1-\alpha$ conjunta).
- **Conformal para regresión**: mismo mecanismo que en clasificación, con
  score de no-conformidad $s(x, y) = |y - \hat{y}(x)|$.

{% if use_conformal %}
### Conformal regression (split-conformal)

1. Entrenar el regresor en una parte; reservar $n_{cal}$ muestras.
2. Calcular los residuos absolutos $s_i = |y_i - \hat{y}(x_i)|$ en
   calibración.
3. Tomar $q = $ cuantil $\lceil (n_{cal}+1)(1-\alpha)\rceil / n_{cal}$ de
   los $s_i$.
4. Para un nuevo $x$: $C(x) = [\hat{y}(x) - q,\; \hat{y}(x) + q]$.

Garantía: $P(Y \in C(X)) \ge 1-\alpha$, marginal y distribution-free (solo
exchangeabilidad). El intervalo es de **ancho constante** en todo el espacio
(no se adapta a la heterocedasticidad); para anchos adaptativos hay que usar
conformal con regresión de cuantiles (CQR) o scores normalizados. Que un
punto quede fuera del intervalo en producción no es un error del modelo: es
lo que la cobertura marginal permite a razón de $\alpha$.
{% endif %}

## Práctica

- **Los modelos no extrapolan.** Un árbol/GBDT predice la media de las hojas:
  fuera del rango de entrenamiento la predicción se aplana en la última hoja.
  Un lineal extrapola linealmente (linealmente mal, si la verdad es
  no lineal). Un lineal sobre features transformadas extrapola en la escala
  transformada. Antes de predecir, comprobar si $x$ cae dentro del rango de
  entrenamiento (detección de OOD).
- **Clipping del target en producción**: recortar las predicciones a
  $[\min y, \max y]$ observados en train protege contra absurdos por
  extrapolación y errores aguas arriba. El clip no es cosmético: es la
  decisión de qué valores son físicamente posibles.
- **Drift en la target**: la distribución del target cambia con el tiempo
  (concept drift). Vigilar la distribución de $y$ y de los residuos en
  producción y re-entrenar cuando se desvía; el sistema de monitorización
  del proyecto cubre esto (ver el workflow de monitorización).
- **No predecir solo el punto**: si el decisor actúa sobre el riesgo de la
  predicción, entregar intervalo o distribución, no media.

## Fuentes

- **Regression Quantiles** — R. Koenker, G. Bassett (1978). Sin arXiv —
  https://doi.org/10.2307/1913643
- **Robust Estimation of a Location Parameter** — P. J. Huber (1964).
  Sin arXiv — https://doi.org/10.1214/aoms/1177703732
- **An Analysis of Transformations** — G. E. P. Box, D. R. Cox (1964).
  Sin arXiv — https://doi.org/10.1111/j.2517-6161.1964.tb00560.x
- **Another Look at Measures of Forecast Accuracy** — R. J. Hyndman,
  A. B. Koehler (2006). Sin arXiv —
  https://doi.org/10.1016/j.ijforecast.2006.03.001
- **A Gentle Introduction to Conformal Prediction and Distribution-Free
  Uncertainty Quantification** — A. N. Angelopoulos, S. Bates (2021).
  arXiv:2107.07511 — https://arxiv.org/abs/2107.07511
- **Ridge Regression: Biased Estimation for Nonorthogonal Problems** —
  A. E. Hoerl, R. W. Kennard (1970). Sin arXiv —
  https://doi.org/10.1080/00401706.1970.10488634
- **The Elements of Statistical Learning** — T. Hastie, R. Tibshirani,
  J. Friedman (2009). Sin arXiv — https://hastie.su.domains/ElemStatLearn/
{% endif %}
