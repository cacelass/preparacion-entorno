# Series temporales

## Qué hace distinta a una serie temporal

Una serie temporal es una muestra ordenada $\{y_1, \dots, y_n\}$ donde las
observaciones **no son iid**: el valor en $t$ correlaciona con los valores en
$t-1, t-2, \dots$ (autocorrelación). Consecuencias operativas:

- **La estructura temporal es información.** El orden de las filas es un
  regresor; barajar el dataset destruye la señal. La autocorrelación positiva
  (típica en demanda, tráfico, sensores) hace que la predicción del próximo
  paso sea casi trivialmente mejor que la media global.
- **La causalidad va en una dirección.** Pasado $\to$ futuro. Todo método
  que evalúe o entrene con datos del futuro mide optimismo, no capacidad
  predictiva.
- **El horizonte y la frecuencia lo gobiernan todo.** Un modelo pensado para
  un horizonte de 1 paso a frecuencia horaria no vale para un horizonte de 30
  días. El horizonte $h$ se mide en unidades de la frecuencia: $h$ pequeños
  (< 10) son "cortos" y explotables por inercia local; $h$ grandes requieren
  modelar nivel y tendencia en vez de transiciones. Horizonte ≥ frecuencia
  estacional (predecir toda una semana) obliga a predecir patrones, no ruido.

Nivel mínimo de seriedad: define **frecuencia** (horaria, diaria...),
**horizonte** $h$ y **paso de refit** antes de elegir modelo. Los tres son
parámetros de la evaluación, no del modelo.

## Estacionariedad

### Definiciones

- **Estacionariedad fuerte**: la distribución conjunta de cualquier bloque
  $[y_{t_1}, \dots, y_{t_k}]$ no cambia al desplazarlo en el tiempo. Exigible
  casi nunca en la práctica.
- **Estacionariedad débil (de segundo orden)**: media constante
  $\mathbb{E}[y_t] = \mu$ y autocovarianza $\gamma(k) = \operatorname{Cov}(y_t,
  y_{t-k})$ que depende solo del lag $k$, no de $t$. Es la que usan los
  modelos ARMA: los parámetros (medias, varianzas, correlaciones) son los
  mismos en cualquier punto de la muestra.

### Por qué importa

Los modelos clásicos (ARIMA, ARMA, regresión con errores correlacionados)
asumen estacionariedad débil. Sin ella, los coeficientes estimados no son
estables en el tiempo, los errores estándar y los tests de hipótesis pierden
validez, y la extrapolación al futuro extrapola un estado que ya no se
sostiene. Un nivel con drift (paseo aleatorio) no es estacionario: su varianza
crece con $t$ y predecir su media futura es predecir un objetivo móvil.

### Detección

- **Test ADF (Augmented Dickey-Fuller)**: $H_0$ = raíz unitaria (no
  estacionario). $p < 0.05$ $\Rightarrow$ rechazar $H_0$ y tratar la serie como
  estacionaria. Es un test, no una prueba: con muestras cortas su potencia es
  baja y el resultado depende del número de lags auxiliares que se incluya.
- **Correlograma ACF / PACF**: la ACF de una serie estacionaria decae
  rápidamente; si decae muy lentamente (prácticamente lineal), hay raíz
  unitaria o tendencia. La PACF aísla la correlación del lag $k$ sin los lags
  intermedios y sirve para orientar el orden $p$ de un AR: cortes abruptos de
  la PACF apuntan a $p$ = último lag significativo, mientras cortes de la ACF
  apuntan a $q$.

### Diferenciación

La transformación $\Delta y_t = y_t - y_{t-1}$ elimina una raíz unitaria:
convierte un paseo aleatorio en ruido blanco. Para tendencias estocásticas
conviene diferenciar ($d=1$ suele bastar; un segundo $\Delta^2$ para doble
integración es raro en series reales). La diferenciación estacional
$\Delta_s y_t = y_t - y_{t-s}$ elimina la no estacionariedad estacional.
Reglas de oro: **no sobrediferenciar** (introduce correlación negativa y
destruye señal, empeorando las predicciones) y **diferenciar por
visualización + ADF, no por receta**; un modelo con $d$ correcto y orden
equivocado sobreajusta, pero un modelo con $d$ de más pierde la señal.

## Descomposición

Modela la serie como combinación de componentes:

$$y_t = T_t + S_t + R_t \quad \text{(aditiva)}, \qquad
y_t = T_t \cdot S_t \cdot R_t \quad \text{(multiplicativa)},$$

con $T$ tendencia, $S$ estacionalidad (período $m$) y $R$ residuo. La
descomposición multiplicativa equivale a una aditiva en $\log y_t$ (o a una
aditiva con estacionalidad proporcional al nivel); se usa cuando la amplitud
de la estacionalidad crece con el nivel (demanda, ventas, tráfico).

- **Clásica**: media móvil para la tendencia, estacionalidad como promedio de
  los desvíos por posición estacional. Sencilla, pero la media móvil asimétrica
  en los bordes y la interacción entre componentes degradan el residuo.
- **STL (Seasonal-Trend decomposition using Loess)**: ajusta las tres
  componentes por regresión local. Robusta a outliers, permite estacionalidad
  que cambia lentamente con el tiempo (periodo $m$ fijo, amplitud variable) y
  admite residuos no gaussianos. Es el estándar práctico.

El test de calidad de una descomposición: **el residuo debe parecer ruido**.
Si $R_t$ tiene estructura visible (dependencia serial, ciclos, saltos
asociados a eventos), la descomposición no capturó todo y quedó señal en el
residuo. Comprueba la ACF del residuo y su distribución: residuo con
autocorrelación o colas demasiado gruesas es un modelo incompleto, no "ruido".

## Modelos clásicos

### Media móvil exponencial / ETS

La familia ETS (Error-Trend-Seasonality) extiende el suavizado exponencial.
Holt-Winters (aditivo) con nivel $\ell_t$, tendencia $b_t$ y estacionalidad
$s_t$ de período $m$:

$$\ell_t = \alpha\,(y_t - s_{t-m}) + (1-\alpha)\,(\ell_{t-1} + b_{t-1}),$$
$$b_t = \beta\,(\ell_t - \ell_{t-1}) + (1-\beta)\,b_{t-1},$$
$$s_t = \gamma\,(y_t - \ell_t) + (1-\gamma)\,s_{t-m},$$
$$\hat{y}_{t+h} = \ell_t + h\,b_t + s_{t-m+\lceil h/m\rceil m}.$$

Los parámetros $\alpha, \beta, \gamma \in [0,1]$ controlan cuánto pesa la
observación nueva frente al pasado; se estiman por máxima verosimilitud. ETS
modela directamente el nivel, la tendencia y la estacionalidad, y genera
predicciones por intervalos razonables. Poco flexible (estructura fija de
componentes), pero barato y robusto.

### ARIMA / SARIMA

$ARIMA(p, d, q)$ combina un autorregresivo AR($p$), diferenciación de orden
$d$ y medias móviles MA($q$):

$$\phi_p(B)\,(1-B)^d y_t = c + \theta_q(B)\,\varepsilon_t,$$

con $B$ el operador de rezago. El modelo estacional es
$SARIMA(p,d,q)(P,D,Q)_S$: añade polinomios AR/MA estacionales con período $S$.
La selección del orden es el punto delicado:

- **Box-Jenkins**: identificar con ACF/PACF, estimar, diagnosticar residuos.
  Práctico en la práctica porque explota el conocimiento de dominio.
- **Automática por AIC**: probar una rejilla de ordenes y elegir el de menor
  AIC (o AICc para muestras cortas). El AIC penaliza la verosimilitud con el
  número de parámetros: $2k$ de penalización, contra la mejora de ajuste.
  Funciona, con una trampa debajo.

**Peligro de sobreajustar el orden.** Cada $(p,q,P,Q)$ añadido gana un punto
de AIC casi gratis si reduce el error en la muestra; en muestras cortas la
rejilla puede elegir ordenes altos que memorizan el ruido. Consecuencias:
parámetros estimados con varianza enorme y predicciones que se degradan fuera
de la muestra. Mitigación: acota la rejilla (valores máximos pequeños: $p,q
\leq 5$, $P,Q \leq 1$ salvo evidencia), compara con AICc, y valida la
selección con walk-forward en vez de con el ajuste en muestra. Un ARIMA(0,1,0)
honesto suele ganar a un ARIMA(7,1,9) que "ajusta" mejor.

### Por qué siguen siendo el baseline

ETS y ARIMA son rápidos, deterministas, explican el comportamiento (nivel,
tendencia, estacionalidad explícitos), necesitan pocos datos y dan intervalos
de predicción razonables por construcción. En la práctica **superan a modelos
ML complejos en horizontes cortos y series con poca señal**. Son el punto de
partida obligatorio: un modelo ML que no bate a ETS/SARIMA sobre walk-forward
no aporta nada que justifique su complejidad.

## ML para forecasting

### Features de lag y ventanas

El forecasting con ML reencuadra la serie como regresión: cada fila es
$(x_t, y_t)$ con $y_t$ el valor futuro y $x_t$ las features conocidas en $t$:

- **Lags**: $y_{t-1}, y_{t-2}, \dots, y_{t-k}$. Es el grueso de la señal en
  horizontes cortos. Los lags se construyen con `shift(k)` y **nunca** con
  ventanas centradas.
- **Ventanas rodantes**: media, desviación, min, max, percentiles de
  $\{y_{t-w}, \dots, y_{t-1}\}$. Codifican nivel y volatilidad reciente; la
  media móvil de 7 días suele superar al lag único de 7 días porque suaviza.
- **Calendario y estacionalidad**: día de la semana, mes, hora, semana del
  año, día festivo (binaria), día del año como codificación circular
  ($\sin(2\pi\,d/365)$, $\cos(2\pi\,d/365)$). Para demanda y tráfico, el
  calendario es a menudo la feature más importante después de los lags.
- **Regresores exógenos**: precios, clima, eventos — solo si se conocen en el
  momento de predecir (sin lookahead).

Regla de construcción: toda feature debe ser computable en el instante $t$
con información estrictamente anterior o contemporánea conocida.

### Boosting sobre lags y su riesgo

Gradient boosting (LightGBM, XGBoost, CatBoost) sobre lags + calendario es el
ML más efectivo en forecasting tabular: captura interacciones entre lags y
regresores sin modelar explícitamente la dependencia serial. Riesgo real:

- **Overfitting al ruido.** Con $k$ lags grandes y series cortas, el boost
  memoriza la autocorrelación del ruido muestral; en validación temporal el
  error sube, aunque en el train el ajuste sea perfecto. El boosting
  interpola pero **no extrapola** más allá del rango observado: si el nivel
  futuro supera todo lo visto, el modelo predice el máximo histórico.
- **Lags directos vs recursivos.** Para horizonte $h$, o se entrena un modelo
  por paso (directo, más caro y más robusto al error acumulado) o se itera la
  predicción de 1 paso (recursivo, propaga el error). La estrategia directa
  con $h$ modelos separados es la habitual en la práctica.

### TCN y LSTM

Las redes recurrentes (LSTM/GRU) y las convoluciones causales (TCN) modelan
la dependencia temporal de forma endógena, sin features de lag manuales. Piden
miles de muestras, son sensibles a la escala (se normalizan **solo con el
pasado**), y su ventaja sobre boosting aparece sobre todo en series largas con
dependencias de largo alcance o entradas multi-serie. En la práctica, con
series de demanda/tráfico y menos de unos miles de puntos, un LightGBM sobre
lags bate a la LSTM con una fracción del esfuerzo.

### El debate estadístico vs ML

No hay ganador universal. Regla empírica aceptada:

| Señal | Estadísticos (ETS/ARIMA) | ML sobre lags (boosting, LSTM) |
|-------|--------------------------|-------------------------------|
| Datos escasos ($n \lesssim 200$) | gana | sobreajusta |
| Dependencia puramente serial | gana (estructura explícita) | compite con lags bien puestos |
| Regresores exógenos ricos | requiere extensión manual | gana por construcción |
| Estacionalidad compleja + festivos | estacionalidad fija | gana (features de calendario) |
| Heterocedasticidad / cambio de régimen | parametrizable | gana si hay features que lo señalen |

La decisión se toma con walk-forward honesto, no con el dogma del paradigma.

## Validación y backtest

### Walk-forward / expanding window

Entrenar con el pasado y validar con un bloque futuro contiguo, desplazando el
punto de corte:

```
|───── train ─────|val|              fold 1
|────── train ──────|val|            fold 2
|─────── train ───────|val|          fold 3   (expanding)
|── train ──|val|  |── train ──|val|           (rolling, ventana fija)
```

- **Expanding**: crece el train; aprovecha todos los datos, más lento.
- **Rolling**: ventana fija deslizante; refleja mejor un despliegue que se
  reentrena solo con lo reciente y acota el coste.
- El tamaño del paso del val y su número de pasos definen cuánto stress-test
  haces: pocos folds cortos = evaluación ruidosa; muchos folds largos = caro.

### Cadencia de refit

¿Cada cuánto se reentrena? Es un hiperparámetro del despliegue, no una
decisión estética. Opciones y coste:

- **Refit por paso de validación** (cada fold): lo más honesto; replica que
  en producción reentrenas tras cada batch nuevo. Caro si el modelo tarda.
- **Refit cada $k$ pasos**: entre refit y refit el modelo envejece; el
  backtest debe reentrenar **exactamente con la misma cadencia que se
  desplegará**, o la métrica mide otro sistema.
- **En producción**: refit disparado por calendario (semanal, mensual), por
  caída de métrica monitorizada, o por drift detectado. El backtest que no
  simula esa cadencia (p. ej. no reentrena nunca y asume el modelo congelado)
  subestima el coste real del mantenimiento.

### Purged y embargo

En ML financiero y con etiquetas de ventana (o lags de respuesta), el bloque
de validación y el de entrenamiento se contaminan mutuamente:

- **Purging**: eliminar del train todas las filas cuya ventana de etiqueta se
  solapa temporalmente con el bloque de validación; sin purgar, información
  del futuro se filtra al train vía la construcción de $y$.
- **Embargo**: descartar además unas filas inmediatamente posteriores al train,
  donde la autocorrelación del borde hace que validación "recuerde" el final
  del train. El tamaño del embargo escala con la persistencia de la serie.

### Fugas típicas en series

1. **Normalizar/escalar con el futuro.** `StandardScaler().fit(X)` sobre todo
   el dataset usa la media y la varianza de los datos futuros. El scaler se
   ajusta **solo con el train del fold** (mismo patrón que en validación
   general, pero aquí el futuro es explícito y el error, silencioso).
2. **Features con lookahead.** Ventanas centradas (`rolling(7, center=True)`),
   lags con `shift(-k)`, agregados que incluyen el valor actual sin desplazar,
   o regresores conocidos solo después de $t$. La feature "ve" el futuro que
   se intenta predecir.
3. **Datos del futuro en el train.** Filas posteriores al corte en el
   entrenamiento (split aleatorio, purga mal hecha, deduplicación por
   timestamp ausente). El modelo aprende valores que en producción no tendrá.
4. **Test reutilizado.** Correr el backtest una y otra vez hasta que "pase"
   convierte la validación en selección; la métrica se infla.

Síntoma común: error en backtest sorprendentemente bajo y colapso en
producción.

## Evaluación

| Métrica | Definición | Sensibilidad |
|---------|------------|--------------|
| MAE | $\frac{1}{n}\sum \|y_t - \hat{y}_t\|$ | robusta, escala-absoluta |
| MSE | $\frac{1}{n}\sum (y_t - \hat{y}_t)^2$ | penaliza cuadráticamente los errores grandes |
| RMSE | $\sqrt{\mathrm{MSE}}$ | misma penalización, en unidades de $y$ |
| sMAPE | $\frac{1}{n}\sum \frac{2\|y_t-\hat{y}_t\|}{\|y_t\|+\|\hat{y}_t\|}$ | acotada, inestable cerca de 0 |
| MASE | $\mathrm{MAE} / \mathrm{MAE}_{\text{naive}}$ | escala-free, depende del naive de referencia |

- **MAE vs RMSE**: si el RMSE ≫ MAE, hay errores grandes poco frecuentes; si
  la aplicación lo penaliza, usa RMSE; si no, MAE. El RMSE es más sensible a
  outliers y a la escala; con errores gaussianos, RMSE ≈ 1.25·MAE.
- **sMAPE y la división por cero**: con $y_t = 0$ o $\hat{y}_t = 0$ el
  denominador se anula y el término explota o se descarta según la
  implementación; además la medida es asimétrica en el signo del error. Evita
  sMAPE cuando la serie tiene ceros frecuentes.
- **MASE**: mide cuántas veces mejor que un **naive** (persistencia:
  $\hat{y}_t = y_{t-1}$; estacional: $\hat{y}_t = y_{t-m}$) es el modelo.
  MASE < 1: el modelo pierde contra no hacer nada. Al ser escala-free,
  comparas series de magnitudes distintas. Es la métrica recomendada por
  defecto para comparar modelos entre datasets.
- **La métrica se agrega sobre todos los folds**, no se reporta por fold como
  si fuera evidencia: media y desviación del error de walk-forward, o el
  error total acumulado. Un fold bueno y otro malo no son "dos resultados".

### Predicción por intervalos

- **Quantile regression**: estimar los cuantiles $\tau = \alpha/2$ y
  $1-\alpha/2$ con la pérdida pinball; da intervalos que se adaptan a la
  heterocedasticidad (más anchos en volatilidad alta). La cobertura es
  aproximada: los dos cuantiles se ajustan por separado.
- **Conformal aplicado a series**: el conformal clásico asume
  exchangeabilidad, que las observaciones temporales violan. Se adapta con
  bloques (calibrar sobre bloques temporales no solapados), scores
  normalizados por la volatilidad local, o **ACI (Adaptive Conformal
  Inference)**, que ajusta $\alpha$ con el historial de cobertura en línea.

{% if use_conformal %}
### Conformal en este proyecto

Con `use_conformal` activo, `models/conformal.py` produce intervalos
split-conformal sobre un conjunto de calibración **temporal** (el bloque final
del train, jamás intercalado), calculando el cuantil empírico de los residuos
absolutos del modelo. Para series:

1. Entrenar en `[1, t_0]$, calibrar en $[t_0, t_0+n_{cal}]$ (bloque contiguo,
   no aleatorio).
2. Residuos absolutos $s_i = |y_i - \hat{y}(x_i)|$ en calibración.
3. $q$ = cuantil $\lceil (n_{cal}+1)(1-\alpha)\rceil / n_{cal}$ de los $s_i$.
4. $C(x) = [\hat{y}(x) - q,\; \hat{y}(x) + q]$.

La garantía marginal $P(Y \in C(X)) \ge 1-\alpha$ depende de que la
calibración y el futuro sean intercambiables; en series con drift, la
cobertura real se desvía y el remedio es calibrar periódicamente (ACI) o
usar residuos normalizados por la volatilidad de la ventana.
{% endif %}

## Estacionalidad, festivos y datos irregulares

- **Estacionalidad**: el período $m$ se identifica con el espectro (FFT) o con
  el conocimiento de dominio (24 h, 7 días, 52 semanas, 12 meses). Con
  estacionalidad múltiple (hora + semana), los modelos clásicos se quedan
  cortos y los ML sobre features de calendario dominan. La estacionalidad que
  cambia de amplitud con el nivel (multiplicativa) se codifica mejor en $\log$.
- **Festivos**: los días festivos rompen la estacionalidad semanal (efecto
  calendario) y además tienen efecto de arrastre (días previos y posteriores,
  puentes). Como binarias por festivo + día relativo (días antes/después) son
  features de alto valor para ML. Los modelos clásicos no los modelan por
  defecto: hay que pasarlos como regresores o aceptar el error.
- **Datos irregulares**: huecos por sensores caídos o días no operativos. Un
  gap en un lag rompe la cadena de features. Opciones: remuestrear a una
  frecuencia fija (imputando con interpolación o el valor anterior), o dejar
  los huecos y que el lag se calcule sobre el índice, no sobre la fila. Nunca
  uses un lag que "salte" el hueco sin marcar: crea una serie sintética
  inexistente. El índice debe ser un DatetimeIndex explícito, no posiciones.
- **Horizonte corto vs largo**: corto (inercia y lags dominan, modelos
  simples compiten) vs largo (tendencia, estacionalidad y regresores
  dominan; el intervalo de predicción crece y la incertidumbre estructural
  importa más que el error del método). Un intervalo para un horizonte largo
  que no crece con $h$ está mintiendo.

## Forecasting probabilístico

Predecir un valor puntual $\hat{y}_{t+h}$ oculta que el futuro es una
distribución. El objetivo probabilístico es la **distribución sobre la
trayectoria futura** $p(X^{1:T} \mid X^0, X^{-1})$ y su salida natural es un
**ensemble de trayectorias**: muestrear $M$ caminos completos, cada uno
coherente —no ruido independiente punto a punto—, de modo que la dispersión
del ensemble a horizonte $h$ sea la incertidumbre del modelo sobre ese
horizonte.

### Por qué un ensemble, no un intervalo por paso

Un intervalo por paso predice cada $X^t$ con su banda pero ignora que los
pasos están correlacionados: la trayectoria es un objeto, no $T$ objetos
independientes. Un ensemble de trayectorias conserva esa estructura conjunta —
preguntas como "¿probabilidad de que el agregado del horizonte supere un
umbral?" solo se responden sobre caminos completos, no sobre marginales por
paso. La dispersión debe crecer con $h$: un ensemble cuyos miembros convergen
a largo plazo miente, igual que el intervalo plano del apartado anterior.

### CRPS: la métrica de una predicción probabilística

Para una distribución predictiva $F$ y un valor observado $y$, el **CRPS**
(Continuous Ranked Probability Score) mide a la vez calibración y sharpness:

$$\operatorname{CRPS}(F, y) = \int_{-\infty}^{\infty}
\bigl(F(z) - \mathbf{1}[y \le z]\bigr)^2 \, dz.$$

Es una **scoring rule propia**: su valor esperado se minimiza cuando $F$ es la
distribución verdadera, así que optimizarla no recompensa mentir. Frente al
log-score es más robusta a colas y outliers (castigo cuadrático, no
logarítmico) y se computa y compara punto a punto. La estructura conjunta se
evalúa aparte — CRPS sobre agregados espaciales o temporales, o sobre
magnitudes derivadas.

Sobre una muestra finita de $M$ miembros se estima con el **estimador fair**:

$$\widehat{\operatorname{CRPS}}(F_M, y) = \frac{1}{M}\sum_{m=1}^{M}|y-x_m|
- \frac{1}{2M^2}\sum_{m=1}^{M}\sum_{k=1}^{M}|x_m-x_k|,$$

que penaliza un ensemble demasiado estrecho (miembros casi iguales) frente a
uno que de verdad cubre el rango. Es el análogo a cobertura+sharpness de
`gestion-incertidumbre.md`, en una sola cifra y —clave para ML—
**diferenciable**: se puede usar como función de pérdida de entrenamiento.

### FGN (Functional Generative Networks): un ejemplo a escala

FGN (Alet et al., Google DeepMind, arXiv:2506.10772) es la arquitectura de
WeatherNext 2 y la traducción directa de la descomposición aleatoria/epistémica
al espacio de funciones:

- **Epistémica**: un deep ensemble de $J$ seeds entrenadas independientemente.
- **Aleatoria**: por paso, un vector de ruido gaussiano $\epsilon_t \in
  \mathbb{R}^{32}$ entra en las capas de normalización compartidas y
  reparametriza el paso — se muestrea una **función**, no se añade ruido a la
  salida. Los miembros del ensemble son alternativas dinámicamente coherentes.

Entrenado solo sobre **marginales** (CRPS por ubicación, variable y nivel),
captura estructura conjunta espacial: como un único vector de 32 dimensiones
influencia el campo entero, la única manera de bajar el CRPS en todas partes es
codificar correlaciones físicas reales a lo largo de ese subespacio. La lección
es portátil a cualquier forecasting multivariado: **la baja dimensionalidad del
ruido compartido fuerza a aprender la estructura conjunta sin supervisión
explícita sobre ella.**

### Cómo se rompe

- **Rollouts inestables a lead largo.** Entrenar solo el paso 1 produce
  trayectorias que degeneran a estados no físicos en horizontes largos; FGN lo
  corrige con un rollout autoregresivo corto (~8 pasos) con backprop a través
  del rollout. Lección: al evaluar un modelo de trayectorias se valida la
  **estabilidad del rollout**, no solo el error del paso 1 — el error corto
  puede estar bien mientras la trayectoria ya no es plausible.
- **CRPS solo mide marginales.** Ganar en CRPS no garantiza estructura
  conjunta correcta; hay que evaluarla aparte con agregados y magnitudes
  derivadas (en el paper, ciclones tropicales).

## Práctica

### El pipeline correcto

```
1. split por tiempo          → train + validación contigua, sin futuro en train
2. por cada fold             → ajustar transformaciones SOLO en el train
                              → entrenar el modelo
                              → predecir el bloque de validación
3. métrica agregada          → MASE/MAE sobre todos los folds, con su cadencia de refit
```

Reglas no negociables: el scaler y cualquier transformador se ajustan dentro
de cada fold; las features de lag se construyen con `shift` positivo; el
embargo y la purga se aplican según la construcción de etiquetas; el test
final se toca una vez.

### Cuándo un naive es difícil de batir

La persistencia ($\hat{y}_{t+1} = y_t$) y el naive estacional
($\hat{y}_{t+h} = y_{t+h-m}$) son los rivales reales, no un formalismo. En
series con autocorrelación fuerte y baja señal, batirlos de forma estable en
walk-forward es difícil; un modelo que los bate por < 1 % en MASE no justifica
complejidad operativa. El naive estacional es además un test de coherencia: si
el backtest no lo vence, hay un problema de features, no de método.

### El mito de "más datos siempre ayuda"

En series temporales, **más datos históricos no implica mejor predicción**:
la señal relevante suele estar en lo reciente, y los datos viejos pueden
arrastrar un régimen obsoleto (cambio de producto, de mercado, de proceso).
Más datos sí ayudan a estimar con menos varianza y a entrenar modelos más
ricos (ML, redes), pero el punto de corte óptimo del train (¿3 años? ¿10?)
se decide en walk-forward, no por intuición. El "más datos" que de verdad
ayuda es **más frecuencia y más regresores conocidos en $t$**, no más
profundidad histórica.

{% if use_optuna %}
### Tuning de lags en este proyecto

Con `use_optuna` activo, `make tune` puede explorar el espacio de
preprocesado de la serie: número de lags, tamaño de las ventanas rodantes
(media/desv), período estacional de referencia y, si el modelo lo permite, los
ordenes $p, q$ de una pieza ARIMA. El trial valida con walk-forward fijo
(misma partición para todos los trials), nunca con el error en muestra: un
lag extra que solo reduce el ajuste es exactamente el sobreajuste al ruido que
el tuning debe descartar. Los lags finales se fijan en el modelo de
producción como hiperparámetros, no como parte de la búsqueda por trial
(evita que cada trial reordene la construcción de features).
{% endif %}

## Fuentes

- Hyndman, R. J. y Athanasopoulos, G., *Forecasting: Principles and Practice*
  (3rd ed.), OTexts, 2021. https://otexts.com/fpp3/
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C. y Ljung, G. M., *Time Series
  Analysis: Forecasting and Control* (5th ed.), Wiley, 2015.
- Hyndman, R. J. y Khandakar, Y., *Automatic Time Series Forecasting: The
  forecast Package for R*, JSS 2008. https://doi.org/10.18637/jss.v027.i03
- Cleveland, R. B., Cleveland, W. S., McRae, J. E. y Terpenning, I., *STL: A
  Seasonal-Trend Decomposition Procedure Based on Loess*, JOSS 1990.
  https://www.wessa.net/download/stl.pdf
- Makridakis, S., Spiliotis, E. y Assimakopoulos, V., *The M4 Competition:
  100,000 time series and 61 forecasting methods*, IJF 2020.
  https://doi.org/10.1016/j.ijforecast.2019.04.014
- Cerqueira, V., Torgo, L. y Mozetič, I., *Evaluating time series forecasting
  models: An empirical study on performance estimation methods*.
  arXiv:1905.11744. https://arxiv.org/abs/1905.11744
- Bergmeir, C. y Benítez, J. M., *On the Use of Cross-Validation for Time
  Series Predictor Evaluation*. arXiv:1503.05341. https://arxiv.org/abs/1503.05341
- López de Prado, M., *Advances in Financial Machine Learning* (purged CV,
  embargo), Wiley 2018. https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
- Vovk, V., Gammerman, A. y Shafer, G., *Algorithmic Learning in a Random
  World*, Springer 2005. https://doi.org/10.1007/b106715
- Gibbs, I. y Candès, E., *Adaptive Conformal Inference Under Distribution
  Shift*. arXiv:2108.09717. https://arxiv.org/abs/2108.09717
- Prophet: Taylor, S. J. y Letham, B., *Forecasting at Scale* (AAAI 2018).
  https://doi.org/10.1609/aaai.v32i1.11654
- Alet, F., Price, I., El-Kadi, A., Masters, D., Markou, S., Andersson, T. R.,
  Stott, J., Lam, R., Willson, M., Sanchez-Gonzalez, A., Battaglia, P.,
  *Skillful joint probabilistic weather forecasting from marginals* (FGN),
  2025. arXiv:2506.10772 — https://arxiv.org/abs/2506.10772
