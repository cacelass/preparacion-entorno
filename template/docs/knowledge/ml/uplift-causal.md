# Uplift y efectos heterogéneos

Uplift modeling estima *para quién* funciona un tratamiento, no solo si
funciona en promedio. Es la pieza de ML que conecta causalidad con targeting:
dada una campaña, ¿a qué clientes la mando? Este documento cubre la notación,
los metalearners, la evaluación sin ground truth y las trampas que invalidan
un modelo de uplift construido sobre datos históricos.

## El problema: los promedios esconden la heterogeneidad

El efecto promedio del tratamiento (ATE) resume, pero puede no describir a
nadie: un ATE de +1 % en conversión puede esconder un +7 % en un segmento y un
−5 % en otro. La pregunta de negocio no es "¿funciona?", sino "¿a quién?": a
quién mando la campaña, a quién le subo el precio, a qué paciente trato. Un
modelo predictivo de respuesta contesta otra cosa (¿quién compra?), no la
decisión de tratar.

## Recapitulación: contrafactuales (ver `matematicas/causalidad.md`)

El problema fundamental de la inferencia causal: para cada unidad solo
observamos uno de los dos mundos posibles, $Y_i(1)$ (tratada) o $Y_i(0)$
(control), nunca ambos. El efecto individual $\tau_i = Y_i(1) - Y_i(0)$ es
inobservable por construcción. El uplift no estima los $\tau_i$: estima su
esperanza condicionada a $X$.

## Uplift y notación

Con resultados potenciales $Y(1), Y(0)$, tratamiento $W \in \{0, 1\}$ y
covariables $X$, el uplift es el efecto condicionado:

$$\tau(x) = E[Y(1) - Y(0) \mid X = x]$$

{% if task_type == 'clasificacion' %}
Con resultado binario ($Y \in \{0, 1\}$, conversión), es la diferencia de
probabilidades:

$$\tau(x) = P(Y(1) = 1 \mid x) - P(Y(0) = 1 \mid x)$$
{% else %}
Con resultado continuo ($Y \in \mathbb{R}$), es la diferencia de medias
condicionadas:

$$\tau(x) = E[Y(1) \mid x] - E[Y(0) \mid x]$$
{% endif %}

**Identificación**: con asignación aleatoria del tratamiento, $Y(w) \perp W
\mid X$ y el efecto se identifica con datos observados: $E[Y \mid W = 1, X] -
E[Y \mid W = 0, X]$ es un estimador insesgado de $\tau(x)$. Con datos
observacionales el espejismo se rompe: si $W$ depende de covariables que
también influyen en $Y$ (confundimiento), esa diferencia mezcla efecto con
selección (ver trampas más abajo). El uplift sobre datos observacionales exige
o bien un experimento como base, o bien supuestos de identificación fuertes que
casi nunca se cumplen en logs históricos.

## Enfoques

Todos se construyen sobre estimadores estándar de scikit-learn; lo que cambia
es cómo se combinan.

- **T-learner**: dos modelos, $\hat{\mu}_1 = E[Y \mid W=1, X]$ y $\hat{\mu}_0 =
  E[Y \mid W=0, X]$; $\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$. Simple y
  robusto al desbalance de tratamiento. Problema: la varianza del estimador es
  la suma de las varianzas de ambos modelos; la diferencia de dos errores es
  ruidosa, sobre todo donde el dato es escaso (p.ej. pocos tratados en un
  segmento).
- **S-learner**: un solo modelo con $W$ como feature. La varianza es menor
  (comparte regularización), pero arriesga que el modelo *ignore* $W$ si el
  efecto es pequeño o si $X$ predice fuerte: con regularización fuerte, $\tau$
  se encoge hacia 0 y el modelo acaba aprendiendo solo respuesta, no uplift.
- **X-learner**: usa un modelo de propensión, estima efectos imputados en cada
  brazo y combina. Mejor comportamiento con desbalance y con datos pequeños;
  más piezas que calibrar.
- **Causal forest** (GRF): bosques "honestos" que parten el espacio buscando
  heterogeneidad de efecto, no de respuesta. Las hojas agrupan unidades con
  $\tau$ similar y dan intervalos de confianza por hoja. Disponible en
  econml/econml (GRF) y causalml.

Regla práctica: empieza con T-learner como baseline (mínimo supuestos, fácil de
depurar), y usa causal forest o X-learner cuando haya desbalance o se necesiten
intervalos.

## Evaluación SIN ground truth individual

No existe $\tau_i$ observado: las métricas de clasificación estándar (AUC,
precisión) no aplican, porque no hay etiqueta de "efecto individual". Un
"buen" modelo de uplift no es el que clasifica bien, sino el que ordena bien a
las unidades por beneficio de tratar.

Métricas de orden:

- **Curva de uplift**: ordenar las unidades por $\hat{\tau}$ descendente; en el
  top $k$, acumular la diferencia tratado−control observada. La curva perfecta
  sube rápido (los más beneficiados primero); la aleatoria es una recta.
- **Qini coefficient**: área entre la curva de uplift y la recta aleatoria,
  normalizada contra la curva perfecta. Es el análogo del AUC, pero sobre el
  orden por efecto incremental.
- **AUUC** (area under the uplift curve): área bajo la curva de uplift,
  relativa al modelo perfecto.

Se estiman con un holdout donde se conoce el tratamiento asignado; si los
datos vienen de un experimento, la curva es insesgada. Sin experimento, la
curva hereda el sesgo de selección de los datos (ver trampas).

## Targeting y beneficio esperado

Con un modelo $\hat{\tau}(x)$ y coste $c$ por unidad tratada y beneficio $b$
por unidad de resultado, el beneficio esperado de tratar a la unidad $x$ es
$b\,\hat{\tau}(x) - c$. Regla de decisión: tratar si

$$\hat{\tau}(x) > \frac{c}{b}$$

El umbral sale del negocio, no del modelo. La curva de uplift se lee también
como curva de beneficio: el punto donde la ganancia marginal cruza $c$ fija el
porcentaje de población a tratar.

**¿Cuándo el uplift justifica su complejidad frente a un modelo de propensión +
score de respuesta?**

- Uplift gana cuando el ATE es pequeño o cercano a 0 pero la heterogeneidad es
  alta: hay segmentos con efecto grande. Un modelo de respuesta los ordena por
  *quién compra* (muchos de ellos comprarían igual, sin tratamiento), no por
  *quién compra por el tratamiento*.
- Un score de respuesta + propensión funciona si el efecto es uniformemente
  positivo y grande: ahí "quién responde" y "quién responde al tratamiento"
  correlacionan y el modelo simple basta.
- El coste es real: T/X-learners duplican el entrenamiento, la evaluación es
  más ruidosa y necesitan datos de experimento limpios. Si no hay heterogeneidad
  que explotar, el uplift añade varianza y no valor — se comprueba mirando si
  la curva de uplift se separa de la recta aleatoria.

## Trampas

1. **Sesgo de selección en datos históricos**: si en el pasado solo se trató a
   quien ya se creía buen candidato, tratados y no tratados difieren en $X$ de
   forma sistemática y la diferencia de medias observada mezcla efecto con
   selección. El modelo "aprende" que el efecto del tratamiento es el efecto de
   la elegibilidad. La única cura sólida es basar el uplift en datos de un
   experimento aleatorizado; sobre logs históricos, cualquier corrección
   (propensity weighting, matching) depende de que el confundimiento sea
   observado.
2. **Tratamiento no observado**: si los logs no registran de forma fiable quién
   recibió la intervención (atribución rota, canales mezclados), el
   brazo "tratado" es una mentira y el $\tau$ estimado no es efecto de nada.
3. **$\hat{\tau}$ calibrado al revés**: un modelo entrenado para predecir
   respuesta (¿quién compró?) no es un modelo de uplift; si el grupo tratado ya
   tenía baseline alto, el efecto estimado es espurio. También pasa a la
   inversa: un modelo de uplift correcto puede reportar $\hat{\tau}$ con signo o
   escala erróneos si el estimador base está mal calibrado; valida contra el
   orden (curva de uplift), no contra el valor absoluto.
4. **Confundir propensión con uplift**: $P(W=1 \mid x)$ (quién fue tratado) no
   es $\tau(x)$. Con tratamiento dirigido a buenos candidatos, la propensión
   correlaciona con respuesta, no con efecto; usarla como score de targeting
   replica el sesgo histórico.

## Fuentes

- Athey, S., Imbens, G. W., *The Econometrics of Randomized Experiments*.
  Handbook of Economic Field Experiments, 2017. Sin arXiv.
  https://doi.org/10.1016/bs.hefe.2016.10.004
- Athey, S., Imbens, G. W., *Recursive Partitioning for Heterogeneous Causal
  Effects* (causal trees). PNAS 2016. Sin arXiv.
  https://doi.org/10.1073/pnas.1510489113
- Wager, S., Athey, S., *Estimation and Inference of Heterogeneous Treatment
  Effects using Random Forests* (causal forest). JASA 2018. arXiv:1510.04342.
  https://arxiv.org/abs/1510.04342
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., Yu, B., *Metalearners for
  estimating heterogeneous treatment effects using machine learning* (T, S, X
  learners). PNAS 2019. arXiv:1706.03461. https://arxiv.org/abs/1706.03461
- Nie, X., Wager, S., *Quasi-Oracle Estimation of Heterogeneous Treatment
  Effects* (R-learner). Biometrika 2021. arXiv:1712.04912.
  https://arxiv.org/abs/1712.04912
- Radcliffe, N. J., Surry, P. D., *Real-World Uplift Modelling with
  Significance-Based Uplift Trees* (curva de uplift, Qini). 2011. Sin arXiv.
  https://www.stochasticsolutions.com/pdf/Radcliffe_Qini.pdf
- Devriendt, F., Berrevoets, J., Verbeke, W., *Uplift Modelling: From Causal
  Inference to Personalised Decisions* (revisión de trampas y evaluación).
  arXiv:2108.12879. https://arxiv.org/abs/2108.12879
