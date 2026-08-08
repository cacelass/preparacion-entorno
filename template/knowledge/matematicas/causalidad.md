# Causalidad

Inferencia causal para proyectos DS: qué distingue predecir de intervenir, cómo
identificar un efecto en datos observacionales y por qué la importancia de
features de un modelo predictivo no es un efecto causal. Referentes: Pearl
(do-calculus, DAGs) y Rubin (resultados potenciales).

## Correlación vs causalidad

"Correlación no implica causalidad" no es una coletilla: es el problema
fundamental de la inferencia causal. $P(Y \mid X)$ describe la asociación en una
población; $P(Y \mid do(X = x))$ describe lo que pasaría si interviniéramos
fijando $X = x$. Difieren siempre que haya confusores, colisionadores o
mecanismos intermedios que distorsionen la asociación.

El problema fundamental: para cada unidad solo observamos uno de los dos mundos
posibles (tratada o no). El efecto causal del tratamiento $D$ sobre $Y$ para la
unidad $i$ es

$$\tau_i = Y_i(1) - Y_i(0),$$

y nunca observamos ambos potenciales; lo observado es
$Y_i = D_i\,Y_i(1) + (1 - D_i)\,Y_i(0)$. Como $\tau_i$ es inobservable, cualquier
estimación causal requiere supuestos de identificación.

## Confusión, sesgo de selección, sesgo de colisionador

- **Confusión:** una causa común $C \to X$ y $C \to Y$ induce asociación espuria
  entre $X$ e $Y$. Ej. clásico: helados y ahogados, ambos empujados por la
  temperatura. Es el obstáculo central de los datos observacionales.
- **Sesgo de selección:** muestrear condicionando al resultado o a un
  post-tratamiento. Ej.: encuestar solo a usuarios activos subestima el churn;
  estudiar supervivientes subestima la letalidad.
- **Colisionador:** si $C$ es causado por $X$ y por $Y$ ($X \to C \leftarrow Y$),
  condicionar en $C$ abre el camino $X \to C \leftarrow Y$ y crea asociación
  entre $X$ e $Y$ que no existe. Paradoja de Berkson: en un hospital, dos
  enfermedades sin relación parecen asociadas porque los pacientes entran por
  la que tienen. **Regla:** "controlar por todo" puede crear el sesgo que
  quieres evitar; no ajustes por colisionadores ni por mediadores
  post-tratamiento.

## DAGs

Un DAG codifica supuestos causales: los nodos son variables, las flechas
$A \to B$ dicen "A causa directamente a B", y no hay ciclos.

- **Camino trasero (back-door):** todo camino que entra a $X$ por una flecha
  ($X \leftarrow \cdots \to Y$); la asociación espuria viaja por ahí.
- **Camino delantero (front-door):** $X \to M \to Y$, a través de mediadores.
- **d-separación:** dos nodos quedan d-separados si el condicionamiento bloquea
  todos los caminos entre ellos; en un modelo no condicionado, variables
  independientes quedan d-separadas, y condicionar en un colisionador las
  d-conecta (véase arriba). Si un DAG implica que $X$ y $Y$ no están
  d-separados, el modelo es compatible con una asociación; el DAG no la prueba.

### Criterio back-door

Un conjunto $Z$ es admisible si (1) bloquea todos los caminos traseros de $X$ a
$Y$ y (2) no contiene descendientes de $X$. Entonces

$$P(y \mid do(x)) = \sum_z P(y \mid x, z)\,P(z).$$

Es la justificación formal de "ajusta por los confusores": estratificar o
ponderar por $Z$ reproduce la distribución intervencional sin intervenir.

### Criterio front-door

Cuando $X$ está confundido sin remedio pero existe un mediador $M$ que captura
todo el efecto y es a su vez no confundido, el efecto es identificable como

$$P(y \mid do(x)) =
\sum_m P(m \mid x) \sum_{x'} P(y \mid x', m)\,P(x').$$

**Aplicación:** el DAG decide qué ajustar y qué NO ajustar. Si el modelo omite
el confusor relevante o ajusta por un colisionador, la estimación del efecto es
sesgada aunque el modelo predictivo sea perfecto.

## Intervenciones vs condicionamiento

$P(Y \mid X = x)$ describe los subgrupos que ya tienen $X = x$ (selección).
$P(Y \mid do(X = x))$ describe lo que pasaría si forzáramos $X = x$ en toda la
población. En el formalismo de Pearl, intervenir equivale a cortar las flechas
entrantes a $X$ (gráfico mutilado $G_{\bar X}$). Por eso
$P(y \mid do(x)) = P(y \mid x)$ solo cuando no hay caminos traseros abiertos.

### Reglas del do-calculus

Sea $G_{\bar X}$ el grafo sin flechas entrantes a $X$, $G_{\underline Z}$ el
grafo sin flechas salientes de $Z$, y $\overline{Z(W)}$ el conjunto de nodos de
$Z$ que no son ancestros de $W$:

1. **Inserción/eliminación de observaciones:** si $Y \perp Z \mid X, W$ en
   $G_{\bar X}$, entonces $P(y \mid do(x), z, w) = P(y \mid do(x), w)$.
2. **Intercambio acción/observación:** si $Y \perp Z \mid X, W$ en
   $G_{\bar X \underline Z}$, entonces $P(y \mid do(x), do(z), w) =
   P(y \mid do(x), z, w)$.
3. **Eliminación de acciones:** si $Y \perp Z \mid X, W$ en
   $G_{\bar X \overline{Z(W)}}$, entonces $P(y \mid do(x), do(z), w) =
   P(y \mid do(x), w)$.

Aplicando estas reglas algebraicamente se decide si un efecto causal es
identificable a partir de distribuciones observables. Es lo que distingue
"cuánto subiría $Y$ si subiéramos $X$" de "cuánto sube $Y$ con $X$ alto".

## Resultados potenciales (Rubin)

- **ATE:** $E[Y(1) - Y(0)]$ — efecto promedio sobre toda la población.
- **ATT:** $E[Y(1) - Y(0) \mid D = 1]$ — efecto sobre los tratados; suele ser
  el relevante en políticas (¿qué ganó quien recibió el programa?).
- **SUTVA:** no interferencia entre unidades (el tratamiento de $i$ no afecta a
  $j$) y consistencia ($Y_i = D_i\,Y_i(1) + (1-D_i)\,Y_i(0)$).
- **Ignorabilidad / no confusión:** $(Y(1), Y(0)) \perp D \mid X$; condicional
  en $X$, la asignación es como aleatoria.
- **Overlap / positividad:** $0 < P(D=1 \mid X) < 1$; sin unidades comparables
  en ambos brazos no hay estimador creíble.

El RCT es el patrón de oro porque la aleatorización garantiza ignorabilidad por
diseño: $D$ no depende de ninguna característica, conocida o no. Pero incluso en
un RCT quedan amenazas: incumplimiento de asignación, atrición, violaciones de
SUTVA y efectos de comportamiento (Hawthorne).

## Estrategias observacionales

- **Estratificación:** estimar el efecto dentro de cada nivel de $Z$ y promediar.
  Se rompe con muchos confusores (maldición de dimensionalidad).
- **Matching:** emparejar tratados y controles con $X$ similar (vecino más
  próximo, exacto, con caliper) y comparar; se asume ignorabilidad dentro de los
  pares. Revisar balance después de matchear (SMD, test de balance).
- **Propensity score:** $e(X) = P(D = 1 \mid X)$. Propiedad de balance: dado
  $e(X)$, $D \perp X$. Colapsa el ajuste a una dimensión; sirve para matching,
  estratificación y ponderación.
- **Ponderación por probabilidad inversa (IPW):** los pesos
  $w_i = \frac{D_i}{e(X_i)} + \frac{1-D_i}{1-e(X_i)}$ reconstruyen la población.
  Chequea siempre overlap: pesos enormes delatan $e$ cerca de 0/1 (inestabilidad;
  usa pesos estabilizados). Estimador tipo Horvitz-Thompson:

```python
w = d / e + (1 - d) / (1 - e)              # e = propensity score estimado
ate = (d * y / e - (1 - d) * y / (1 - e)).mean()
```

- **Diferencias en diferencias (DiD):** con dos periodos y dos grupos,
  $\hat\delta = (\bar Y_{T,post} - \bar Y_{T,pre}) - (\bar Y_{C,post}
  - \bar Y_{C,pre})$; supuesto de tendencias paralelas (no de niveles
  comparables). Con más periodos: event study con leads/lags.
- **Instrumental variables (IV):** una variable $Z$ que causa $D$ (relevancia),
  no tiene efecto directo sobre $Y$ (exclusión) y es independiente de los
  confusores. Identifica el efecto local en los "compliers" (LATE). Justificar
  exclusión es difícil; sin ella, el IV es sesgado.
- **Ajuste por regresión:** incluir confusores como covariables estima el efecto
  condicional si el modelo es correcto. Falla por forma funcional equivocada,
  extrapolación fuera del soporte y no overlap. La regresión ajustada sigue
  asumiendo que no hay confusores no medidos.

## Paradoja de Simpson y hallazgos espurios

**Ejemplo exacto (piedras renales, Charig 1986).** El tratamiento A tiene éxito
en 273/350 casos (78.0%) y el B en 289/350 (82.6%). Por agregado, B parece
mejor. Pero estratificando por el tamaño de la piedra, A gana en ambos estratos:

| Estrato | Tratamiento A | Tratamiento B |
|---|---|---|
| Piedra pequeña | 81/87 = 93.1% | 234/270 = 86.7% |
| Piedra grande | 192/263 = 73.0% | 55/80 = 68.8% |

A es mejor dentro de cada estrato y peor en total porque los médicos asignan A a
los casos graves (confusión por gravedad). El resultado agregado depende de la
composición: agrega, desagrega y entiende el mecanismo antes de decidir.

**Casualidad (hallazgos por azar):** con $m$ tests a nivel $\alpha = 0.05$, el
número esperado de falsos positivos es $0.05 m$; con mil hipótesis, ~50 son
ruido. El efecto file-drawer (publicar solo resultados significativos) infla la
literatura, y por eso muchos efectos publicados no replican. **Antídotos:**
pre-registrar, corregir múltiples tests (Bonferroni, BH-FDR), reportar
intervalos completos y réplicas; en un mismo experimento, no "cazar" la
submuestra que da significativo.

## ML y causalidad

Los modelos predictivos estiman $P(Y \mid X)$ bajo la distribución de
entrenamiento: responden "¿qué valor de $Y$ acompaña a $X$?", no "¿qué le
pasará a $Y$ si cambiamos $X$?". La importancia de features de un modelo —
SHAP, permutación, ganancia de árboles — mide contribución a la predicción
dentro del modelo, es sensible a correlaciones y no mide efectos causales. Un
modelo con AUC excelente puede estar aprendiendo confusores (el médico asigna
tratamiento por gravedad) y ser inútil para decidir el tratamiento.

**Cuándo el ML sí ayuda a responder lo causal:** como estimador de funciones de
riesgo (propensity, regresión de resultado) dentro de estimadores de doble ML o
TMLE (ortogonalidad: el error del ML no domina el del estimador causal), o como
modelo de predicción cuando el objetivo es intervencional pero bien identificado
(predicción contrafactual sobre un cambio planificado). El modelo predictivo
solo, por bueno que sea, no identifica efectos.

{% if use_shap %}
**Con SHAP activo en este proyecto:** usa los summary/beeswarm plots como
diagnóstico de consistencia del modelo, no como ranking causal. Con features
correlacionadas, SHAP reparte el crédito según la estructura del árbol y
valores pequeños no implican ausencia de efecto causal; no tomes decisiones
"porque la feature X causa Y" a partir del plot. Si necesitas un efecto, plantea
un diseño causal (identificación) y estima con doble ML/TMLE.
{% endif %}

## Práctica

- **Diseñar la pregunta causal:** fija la intervención (¿qué cambiamos y sobre
  quién?), la población objetivo, el outcome y el estimando (ATE, ATT, LATE). Si
  no puedes decir cuál es la intervención, la pregunta no es causal.
- **Elegir identificación:** dibuja el DAG con el dominio. Con confusores
  medidos → back-door (ajuste, matching, IPW, regresión); sin ellos → diseño
  (RCT, natural experiment, DiD, IV, front-door). Sé explícito sobre los
  supuestos que no puedes probar con datos (exclusión, tendencias paralelas,
  ignorabilidad) y cómo se violarían.
- **Comunicar incertidumbre:** intervalos de confianza y análisis de
  sensibilidad (e.g. E-value: qué magnitud de confusión no medida haría
  desaparecer el efecto). Reporta balance y overlap, no solo el efecto.
- **Pitfalls:** ajustar por un colisionador; condicionar post-tratamiento
  (sesgo de mediación); sobre-ajustar por mediadores; extrapolar fuera del
  soporte; olvidar SUTVA en datos de red; concluir causalidad de una asociación
  significativa sin identificación.

## Fuentes

- J. Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed., CUP, 2009.
- J. Pearl, M. Glymour, N. P. Jewell, *Causal Inference in Statistics: A
  Primer*, Wiley, 2016.
- G. W. Imbens, D. B. Rubin, *Causal Inference for Statistics, Social, and
  Biomedical Sciences*, CUP, 2015.
- M. A. Hernán, J. M. Robins, *Causal Inference: What If*, Chapman & Hall/CRC,
  2020.
- S. L. Morgan, C. Winship, *Counterfactuals and Causal Inference*, 2nd ed.,
  CUP, 2015.
