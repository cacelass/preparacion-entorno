# Modelos bayesianos aplicados

La disciplina de construir, ajustar y confiar en un modelo bayesiano real.
Asume la teoría de [probabilidad.md](../matematicas/probabilidad.md) (teorema
de Bayes, priors conjugados, MAP) y las aproximaciones bayesianas para redes
([gestion-incertidumbre.md](gestion-incertidumbre.md): MC-dropout, Laplace,
credible vs confidence). Este fichero es el flujo completo de la práctica:
prior + likelihood → posterior, cómo muestrear esa posterior, cómo
diagnosticar el muestreo, cómo leer la predicción y cuándo no usar Bayes.

## El modelo bayesiano aplicado

$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\, p(\theta)}
{p(\mathcal{D})}$$

Todo modelo bayesiano es esa ecuación: una posterior que combina la
verosimilitud de los datos con el conocimiento previo. El arte está en
construir $p(\mathcal{D} \mid \theta)$ (el modelo generativo) y $p(\theta)$
(lo que se sabe antes). El resultado no es un $\hat\theta$ puntual sino una
distribución completa sobre $\theta$ que se propaga a cualquier cantidad que
dependa de los parámetros.

### Por qué modelar bayesiano

- **Incertidumbre coherente.** La posterior cuantifica la incertidumbre
  epistémica (ver gestion-incertidumbre.md): con pocos datos es ancha, con
  muchos se concentra. Un punto más un intervalo creíble es un artefacto de
  decisión; un punto solo no lo es.
- **Regularización vía priors.** El prior encoge los parámetros; un prior
  Gaussiano equivale a penalización L2 y uno de Laplace a L1 (ver abajo). La
  regularización deja de ser un truco de optimización y pasa a ser una
  declaración explícita de plausibilidad.
- **Datos pequeños.** El régimen donde el MLE diverge o explota (frecuencias
  $0/n$, grupos con pocas observaciones) es donde el prior y el pooling
  jerárquico mantienen estimaciones razonables y con varianza honesta.
- **Pooling jerárquico.** Cuando las observaciones se agrupan (clínicas,
  regiones, usuarios), el modelo comparte información entre grupos: los
  pequeños "se apoyan" en los grandes sin anular su identidad (ver Modelos
  jerárquicos).
- **Estructura y mecanismo.** El modelo generativo obliga a declarar cómo se
  producen los datos, lo que hace el error de especificación explícito y
  diagnosticable (posterior predictive checks) en lugar de implícito.

## Inferencia posterior

La posterior $p(\theta \mid \mathcal{D})$ casi nunca es una densidad conocida:
solo lo es en los casos conjugados y en un puñado de modelos. Todo lo demás
exige aproximación numérica. Hay tres familias, con distinto precio y
fidelidad.

### Conjugada

Existe cuando prior y likelihood pertenecen a la misma familia y la posterior
es cerrada (tabla en probabilidad.md: Beta/Binomial, Gamma/Poisson,
Dirichlet/Multinomial). Es el único caso donde la posterior es exacta y
barata.

**En la práctica casi nunca existe.** Los conjugados cubren likelihoods
exponenciales simples; un GLM con link logit o una jerarquía con varianzas
desconocidas ya no tiene posterior cerrada. Usar conjugados fuera de su
alcance significa deformar el modelo para que tenga solución analítica, que es
la elección equivocada. Se usan como bloques (ej. el prior Gamma sobre una
tasa) y como punto de partida didáctico, no como el motor de un modelo real.

### MCMC: Metropolis-Hastings

Construye una cadena de Markov cuya distribución estacionaria es la posterior.
El procedimiento:

1. Estar en $\theta^{(t)}$. Proponer $\theta^* \sim q(\theta^* \mid
   \theta^{(t)})$.
2. Aceptar con probabilidad
   $$\alpha = \min\left(1,\ \frac{p(\theta^*)p(\mathcal{D}\mid\theta^*)}
   {p(\theta^{(t)})p(\mathcal{D}\mid\theta^{(t)})} \cdot
   \frac{q(\theta^{(t)}\mid\theta^*)}{q(\theta^*\mid\theta^{(t)})}\right)$$
3. Si se acepta, $\theta^{(t+1)} = \theta^*$; si no, $\theta^{(t+1)} =
   \theta^{(t)}$.

**Por qué funciona:** el cociente cancela la constante de normalización
$p(\mathcal{D})$ (desconocida); solo se evalúan verosimilitud y prior. La
regla de aceptación es la condición de balance detallado, que garantiza que la
cadena converge en distribución a la posterior. Funciona para cualquier $q$
razonable, lo que la hace universal pero lenta: en dimensiones altas la región
de alta probabilidad es pequeña, las propuestas se rechazan y la cadena camina
(random walk), tardando en mezclarse.

### Gibbs

Caso particular de MCMC que muestrea cada coordenada de su distribución
condicional completa $p(\theta_j \mid \theta_{-j}, \mathcal{D})$ — cerrada
cuando el modelo es conjugado por bloques (típico en modelos de mezclas y
jerárquicos con priors conjugados). No hay tasa de rechazo (aceptación 1), lo
que la hace mucho más eficiente que MH en modelos con estructura conjugada.
Su límite es el mismo de la conjugación: los condicionales completos deben ser
muestreados, y cuando no lo son se degrada a MH por bloque o a HMC.

### HMC y NUTS

El estándar moderno para modelos continuos. HMC trata $\theta$ como la
posición de una partícula en un campo de energía potencial $U(\theta) =
-\log p(\theta \mid \mathcal{D})$ y le añade un momento $r \sim
\mathcal{N}(0, M)$; el estado $(r, \theta)$ sigue una dinámica hamiltoniana

$$\frac{d\theta}{dt} = M^{-1} r, \qquad
\frac{dr}{dt} = -\nabla_\theta U(\theta)$$

Integrar (leapfrog) y aceptar/rechazar con Metropolis sobre el hamiltoniano
conservado da una cadena que explora la posterior **dirigida por el gradiente**
en lugar de caminar al azar: en dimensiones altas el tiempo de mezcla escala
mucho mejor que MH. NUTS (Hoffman & Gelman) elimina el problema de la longitud
de trayectoria: la extiende automáticamente hasta que dobla hacia atrás, sin
paso aleatorio que configurar. Solo requiere gradientes (obtenidos por
autodiff); por eso las librerías modernas lo exponen como default.

### Variational inference

Convierte la inferencia en optimización: se busca la $q(\theta) \in
\mathcal{Q}$ que minimiza la divergencia de Kullback-Leibler a la posterior,

$$\mathrm{KL}(q \,\|\, p) = E_q[\log q(\theta)] -
E_q[\log p(\theta, \mathcal{D})] + \log p(\mathcal{D})$$

Como $\log p(\mathcal{D})$ (la evidencia) no depende de $q$, minimizar KL
equivale a maximizar el **ELBO**,

$$\mathrm{ELBO}(q) = E_q[\log p(\theta, \mathcal{D})] - E_q[\log q(\theta)]
\le \log p(\mathcal{D})$$

En **mean-field** la familia factoriza $q(\theta) = \prod_j q_j(\theta_j)$:
cada variable es independiente a posteriori, una aproximación fuerte que
subestima la covarianza. ADVI (Kucukelbir et al.) automatiza el ajuste
transformando las variables a soporte real, estimando los gradientes del ELBO
por Monte Carlo y subiendo por gradiente en un solo bucle de optimización.

**Tradeoff velocidad vs aproximación.** VI es órdenes de magnitud más rápido
que MCMC y escala a millones de parámetros, pero la familia $\mathcal{Q}$
restringe la forma de la posterior (mean-field pierde correlaciones) y el ELBO
subestima la varianza: los intervalos creíbles tienden a ser optimistas.
Regla: VI para iterar, prototipar y para modelos grandes; NUTS para el modelo
final cuando el coste lo permite. Ante la duda, valida VI contra una corrida
corta de NUTS: el prior, los datos y el modelo son los mismos, y la
discrepancia entre ambas es la medida del error de la aproximación.

## Tooling práctico

- **numpyro / pymc.** Ambos exponen HMC/NUTS sobre autodiff (JAX y
  PyTensor respectivamente). numpyro compila con JIT y paraleliza en CPU/GPU;
  pymc tiene integración con arviz y un ecosistema de GLMs y priors más
  declarativo. Ninguna requiere escribir derivadas: solo el modelo y los datos.
- **arviz** para diagnóstico (R-hat, ESS, divergencias, PPC) y plots.

### Flujo estándar

1. **Definir el modelo.** Declarar priors y likelihood como variables
   simbólicas; cada parámetro con su prior y su dominio.
2. **Muestrear.** `sample()` con NUTS: varias cadenas en paralelo (4 por
   defecto), warm-up de ~1000 iteraciones y ~1000–2000 de post-warmup.
3. **Diagnosticar.** Antes de leer cualquier resultado: cadenas mezcladas
   ($\hat R < 1.01$), ESS suficiente en los parámetros que importan y cero (o
   casi) divergencias. Un muestreo no convergido no es un resultado, es basura.
4. **Posterior predictiva.** Simular réplicas del dataset desde la posterior,
   $y_{rep} \sim p(y \mid \theta)$, y compararlas con los datos reales (PPC).

```python
import numpyro.distributions as dist
from numpyro import sample

def modelo(x, y):
    alpha = sample("alpha", dist.Normal(0, 1))
    beta = sample("beta", dist.Normal(0, 0.5))
    sigma = sample("sigma", dist.HalfNormal(1))
    mu = alpha + beta * x
    sample("y", dist.Normal(mu, sigma), obs=y)
```

El objetivo del flujo no es solo tener una posterior, es saber que se puede
confiar en ella (paso 3) y que el modelo genera datos plausibles (paso 4)
antes de usarla para decidir.

## GLMs bayesianos

Un GLM bayesiano es la regresión de siempre con priors sobre los coeficientes
y una posterior que los cuantifica. Regresión lineal:

$$\beta \sim \mathcal{N}(0, \sigma_\beta^2), \qquad
y_i \sim \mathcal{N}(\mathbf x_i^\top \boldsymbol\beta, \sigma^2)$$

Regresión logística (likelihood Bernoulli con link logit):

$$y_i \sim \mathrm{Bernoulli}\big(\sigma(\mathbf x_i^\top
\boldsymbol\beta)\big)$$

### El prior como regularizador

La posterior de $\boldsymbol\beta$ es proporcional a $\exp(-\mathrm{NLL} -
\|\boldsymbol\beta\|_2^2 / (2\sigma_\beta^2))$: maximizar la posterior (MAP)
es exactamente la pérdida L2 de ridge con $\lambda = 1/(2\sigma_\beta^2)$, y un
prior de Laplace equivale a L1/lasso (ver probabilidad.md). La diferencia
bayesiana es que no se queda en la moda: la **distribución completa** de
$\boldsymbol\beta$ da el error estándar de cada coeficiente, la correlación
entre ellos y la incertidumbre de cualquier combinación lineal. La
regularización se elige como una declaración sobre la escala plausible de los
efectos (priors weakly informative), no como un $\lambda$ que se tunea a
ciegas.

### Coeficientes con incertidumbre

El output no es un vector $\hat{\boldsymbol\beta}$ sino una posterior
conjunta. De ahí se lee:

- La **media posterior** y el **intervalo creíble** de cada coeficiente: un
  intervalo que no cruza cero es evidencia bayesiana de efecto, no un test.
- La **probabilidad posterior de signo** $P(\beta_j > 0 \mid \mathcal{D})$,
  más informativa que el p-valor y sin el falso dilema de significación.
- Los **efectos predichos**: se propaga la posterior por el modelo y se
  reporta el efecto marginal esperado de mover una feature, con su banda.

Para que los priors sobre pendientes sean interpretables, estandariza las
features (media 0, desviación 1); entonces $\beta_j$ es "el cambio por una
desviación típica de $x_j$, manteniendo lo demás fijo" y un prior
$\mathcal{N}(0, 1)$ o $\mathcal{N}(0, 0.5)$ es genuinamente débil para esa
escala.

## Modelos jerárquicos / multinivel

Cuando las observaciones se agrupan, un modelo simple las trata como
independientes. El modelo multinivel trata los efectos de grupo como variables
aleatorias que comparten un prior común (hiperparámetros). Para un intercepto
por grupo $j$:

$$y_{ij} \sim \mathcal{N}(\alpha_{j[i]} + \mathbf x_{ij}^\top
\boldsymbol\beta,\ \sigma^2), \qquad
\alpha_j \sim \mathcal{N}(\mu_\alpha, \tau_\alpha^2)$$

con $\mu_\alpha, \tau_\alpha$ estimados de los datos. La notación anida: las
observaciones $i$ dentro de grupos $j$, y los $\alpha_j$ dentro de una
distribución común.

### Partial pooling

El modelo comparte fuerza entre grupos a través de $\mu_\alpha$ y
$\tau_\alpha$:

- **No-pooling** (un $\alpha_j$ libre por grupo): con $n_j$ pequeño,
  $\alpha_j$ se estima con varianza enorme o diverge (frecuencias $0/n_j$).
- **Full-pooling** (un solo $\alpha$ para todos): ignora las diferencias entre
  grupos; si los grupos difieren de verdad, sesga sistemáticamente.
- **Partial pooling**: el $\alpha_j$ posterior es un promedio ponderado entre
  la estimación del grupo ($n_j$ grande → domina el grupo) y la media común
  ($n_j$ pequeño → se tira hacia $\mu_\alpha$). Los grupos pequeños "se
  apoyan" en la información del resto sin perder su estimación: es el mismo
  shrinkage que los priors hacen sobre coeficientes, aplicado a grupos.

El hiperparámetro $\tau_\alpha$ controla cuánto pooling hay y se estima, no se
fija: si es grande, los grupos se tratan casi como independientes; si es
pequeño, se encogen fuertemente a la media. Comparado con no-pooling, los
intervalos de los grupos pequeños son más estrechos **y honestos**: la
varianza entre-grupos estimada es parte de la incertidumbre, no ruido
ignorado.

## Predicción

### Distribución predictiva posterior

La predicción de un punto nuevo $y^*$ con features $x^*$ integra la
incertidumbre sobre los parámetros:

$$p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \theta)\,
p(\theta \mid \mathcal{D})\, d\theta$$

En la práctica se simula: para cada muestra $\theta^{(s)}$ de la posterior, se
extrae $y^{(s)} \sim p(y \mid x^*, \theta^{(s)})$. La media de las muestras es
la predicción puntual; el cuantil 2.5%/97.5% es el intervalo creíble del 95%.
Un intervalo creíble captura a la vez la varianza del likelihood (ruido
aleatorio) y la dispersión de la posterior (incertidumbre epistémica). Con un
likelihood no simétrico (Poisson, lognormal), la predictiva es asimétrica y el
intervalo creíble lo refleja — no se fuerza simetría alrededor de la media.

### Posterior predictive checks (PPC)

El test de cordura del modelo: simular réplicas $y_{rep}$ del dataset completo
condicionado a la posterior y comparar la distribución de estadísticos de las
réplicas con el valor observado. Si el modelo genera datos plausibles, las
réplicas se parecen a los reales; si un estadístico cae sistemáticamente fuera
(media, desviación, ceros, máximos), el modelo está mal especificado para ese
aspecto de los datos. Un PPC fallido no "rechaza" el modelo como un test: dice
dónde el modelo y los datos no coinciden, y por dónde mejorarlo (otra
distribución, otra jerarquía, interacciones).

### Credible vs confidence interval

El intervalo creíble responde "dados los datos, con probabilidad $1-\alpha$ el
parámetro está aquí". El de confianza responde sobre el procedimiento: si
repites la recogida de datos, el $1-\alpha$ de los intervalos así construidos
contienen el parámetro (ver gestion-incertidumbre.md). Con priors débiles y
mucha muestra tienden a coincidir numéricamente; en muestras pequeñas y con
priors informativos divergen. No son intercambiables ni en lectura ni en
interpretación: el creíble es la afirmación que un decisor quiere hacer.

## Elección de priors

### Weakly informative

Un prior débilmente informativo fija la **escala plausible** del parámetro sin
fijar su valor: $\mathcal{N}(0, 1)$ sobre coeficientes estandarizados,
$\mathrm{HalfNormal}(1)$ sobre desviaciones, $\mathrm{Normal}(0, 1)$ o
$\mathrm{Cauchy}(0, 2.5)$ sobre pendientes. Cumple dos funciones: mantiene el
muestreo estable (no deja que la posterior explore regiones numéricamente
degeneradas) y estabiliza estimaciones con datos escasos, todo sin sesgar el
resultado cuando hay suficiente muestra.

### Por qué no priors "no informativos" extremos

La uniforme o la normal de varianza infinita no son "sin prior": son priors
que ponen masa en valores absurdos. Con datos suficientes la likelihood los
domina y no pasa nada; con datos escasos la posterior hereda las regiones
degeneradas, el muestreo diverge y los intervalos son irreales. El caso
clásico es la prior uniforme sobre $\sigma$ (o sobre $\log\sigma$): asignan
masa a varianzas que no generan los datos y el MCMC los persigue. Un prior
plano en una transformación es muy informativo en otra (uniforme en $\sigma$
es fuertemente informativa en $\sigma^2$).

### Sensibilidad al prior

La conclusión de un modelo bayesiano no es completa sin verificar que no
depende críticamente del prior. Se hace con un **prior sensitivity check**:
reajustar con priors más anchos o más estrechos y comparar la posterior de las
cantidades que importan (no todos los parámetros). Si el efecto principal
cambia de signo o de magnitud material, el dato no soporta la conclusión por
sí solo: el modelo está dominado por el prior y hay que decirlo, no
esconderlo. La sensibilidad alta es información: marca dónde los datos son
escasos y dónde un prior informativo legítimo (literatura, un estudio previo)
aporta de verdad.

## Comparación de modelos

### WAIC y LOO

Ambos estiman el error predictivo esperado de un modelo, ponderando ajuste y
complejidad. WAIC usa el log-predictivo puntual sobre la posterior con una
corrección de número efectivo de parámetros. LOO-CV aproxima la validación
cruzada leave-one-out con pesos de importancia (Pareto-smoothed importance
sampling, PSIS): cada punto se predice dejándolo fuera y el error se agrega.
Se reporta como $\mathrm{elpd}_{loo}$ (expected log predictive density) y la
diferencia entre modelos con su error estándar; la regla práctica es que
diferencias menores que ~4 unidades de elpd (o un peso de importancia $k >
0.7$ en PSIS) no son concluyentes. Como la CV frecuentista, compara modelos
por capacidad predictiva, no por verosimilitud cruda (que siempre favorece al
modelo más complejo).

### Por qué no test de hipótesis clásico

- El p-valor no cuantifica el tamaño ni la dirección del efecto, y mezcla
  tamaño muestral con magnitud: con $n$ grande cualquier diferencia trivial es
  "significativa".
- El p-valor condiciona a la hipótesis nula y asume que el modelo es correcto;
  no produce una afirmación sobre el parámetro dados los datos.
- La comparación bayesiana es de **modelos en competencia** (WAIC/LOO,
  predictiva posterior) o de **efectos con su incertidumbre** (intervalos
  creíbles, probabilidad posterior de signo): reporta la magnitud y su banda,
  que es lo que una decisión necesita. Un intervalo creíble ancho que cruza
  cero dice "datos insuficientes", no "sin efecto".

## Cuándo NO usar un modelo bayesiano

- **Escala.** Millones de filas con miles de parámetros: MCMC no converge en
  un tiempo útil. Ahí toca VI (ADVI) o una aproximación frecuentista/GP, o se
  degrada el problema (features escasas, muestreo) para que la posterior sea
  tratable.
- **Cuando el frecuentista da la misma respuesta.** Con datos abundantes,
  likelihood regular y priors débiles, la media posterior converge al MLE y el
  intervalo creíble al de confianza. Si el decisor solo necesita el punto y el
  modelo no tiene estructura que aprovechar, Bayes añade coste sin valor.
- **Coste computacional.** El muestreo con NUTS, la validación del muestreo y
  los PPC son una carga real de ingeniería y runtime; en un modelo que se
  reentrena cada noche sobre millones de filas, no es gratis.
- **Sin presupuesto de validación.** Un modelo bayesiano mal diagnosticado
  (muestreo sin converger, posterior no revisada) es peor que un modelo
  puntual: da una falsa precisión con intervalo.

### Dónde Bayes gana de calle

**Datos pequeños con estructura.** Grupos con pocas observaciones (pooling
jerárquico), eventos raros (tasas $0/n$), incertidumbre de parámetros que
importa para decidir, conocimiento previo de dominio que debe entrar en el
modelo, y la necesidad de una banda honesta en un régimen donde la asintótica
frecuentista no vale. Ahí el modelo bayesiano no es "otra herramienta", es la
única que da respuestas no degeneradas.

## Trampas

1. **Muestreo no convergido.** El R-hat (Gelman-Rubin) compara la varianza
   entre cadenas con la varianza dentro de cada cadena; $\hat R \ge 1.01$ en
   cualquier parámetro significa que las cadenas no se han mezclado y la
   posterior no es fiable. Las **divergencias** (NUTS abandona la trayectoria)
   señalan regiones de curvatura extrema que el muestreo no resuelve; suelen
   ser síntoma de un prior demasiado plano o de mala parametrización (centrar
   vs no centrar los efectos jerárquicos). Diagnosticar antes de leer
   resultados, siempre.
2. **Priors que dominan la likelihood.** Con pocos datos la posterior es
   esencialmente el prior, y un prior elegido por conveniencia se convierte en
   la respuesta disfrazada. El antídoto es el prior sensitivity check y
   declarar el prior con su justificación.
3. **Leer intervalos creíbles como frecuentistas.** Un creíble del 95% no
   garantiza que el 95% de los intervalos repetidos contengan el parámetro; es
   una afirmación sobre el parámetro dados estos datos. Reportarlo como
   cobertura frecuentista es un error de interpretación.
4. **Seleccionar modelos sin validar aparte.** Comparar muchos modelos con
   WAIC/LOO sobre el mismo dato selecciona por azar; como en CV frecuentista,
   el optimismo inflado se evita validando la selección en un split de
   retención.

{% if use_optuna %}
## Conexión con la optimización bayesiana

Con `use_optuna` activo, `optimizacion-hiperparametros.md` describe el
surrogate GP de Optuna. Es el mismo razonamiento bayesiano aplicado a otro
objetivo: en lugar de muestrear la posterior de los parámetros de un modelo de
datos, se modela $f(\mathbf x) = \mathrm{loss}(\mathbf x)$ (una función negra
sobre los hiperparámetros) con un proceso gaussiano, y el siguiente punto a
evaluar maximiza la **expected improvement**, balanceando explotar las
regiones prometedoras y explorar las inciertas. La diferencia con este fichero
es el objetivo (una función de coste, no una verosimilitud) y que el surrogate
es auxiliar: lo que se usa es su media/varianza para escoger evaluaciones, no
su posterior para decidir. La conexión conceptual es directa: el GP
cuantifica la incertidumbre sobre el paisaje de pérdida y la usa para decidir,
exactamente como un modelo bayesiano usa su posterior.
{% endif %}

{% if use_calibration %}
## Calibración bayesiana

Con `use_calibration` activo, `models/calibrate.py` hace post-hoc
temperature/isotonic scaling (ver gestion-incertidumbre.md). La perspectiva
bayesiana refina la lectura: la incertidumbre que aporta la posterior no
garantiza calibración frecuentista, y calibrar la media posterior no corrige
una posterior mal especificada. Si el reliability diagram de un modelo
bayesiano sale descalibrado, sospecha primero de la likelihood y del modelo
generativo (PPC), no solo de la temperatura; el temperature scaling puede
parchear el número sin arreglar el mecanismo. Con un likelihood adecuado y
PPC limpio, la posterior predictiva es razonablemente calibrada por
construcción.
{% endif %}

## Fuentes

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A.,
  Rubin, D. B., *Bayesian Data Analysis*, 3rd ed., CRC Press, 2013 (BDA3).
- McElreath, R., *Statistical Rethinking: A Bayesian Course with Examples in R
  and Stan*, 2nd ed., CRC Press, 2020.
- Hoffman, M. D., Gelman, A., *The No-U-Turn Sampler: Adaptively Setting Path
  Lengths in Hamiltonian Monte Carlo*, 2014. arXiv:1111.4246 —
  https://arxiv.org/abs/1111.4246
- Kucukelbir, A., Ranganath, R., Gelman, A., Blei, D. M., *Automatic
  Variational Inference in Stan*, 2017. arXiv:1603.00788 —
  https://arxiv.org/abs/1603.00788
- Vehtari, A., Gelman, A., Gabry, J., *Practical Bayesian Model Evaluation
  Using Leave-One-Out Cross-Validation and WAIC*, 2017. arXiv:1507.04544 —
  https://arxiv.org/abs/1507.04544
