# Diseño de experimentos (A/B testing)

El A/B test es la máquina de inferencia causal más limpia que existe en
producto: la aleatorización identifica el efecto de una intervención sobre una
métrica sin confusores. Este documento cubre la cadena completa y los modos de
fallo más caros (potencia, peeking, multiplicidad), no la mecánica de una
herramienta concreta.

## El marco completo

Cadena de pasos, en orden:

1. **Hipótesis** — efecto esperado, dirección y magnitud, población objetivo.
   Sin hipótesis el experimento no tiene respuesta de "sí/no".
2. **Métrica primaria** — una sola métrica preespecificada que decide. Las
   demás van como guardarraíles o secundarias.
3. **Power analysis** — fijar $\alpha$, $\beta$ (potencia $1-\beta$) y el
   efecto mínimo detectable.
4. **Tamaño de muestra** — $n$ por brazo calculado con la unidad de
   aleatorización correcta.
5. **Unidad de aleatorización** — usuario, sesión, dispositivo o clúster.
6. **Análisis** — preespecificado: estimador, corrección de multiplicidad,
   regla de parada.
7. **Decisión** — lanzar / rechazar / iterar, según primaria y guardarraíles.

Qué falla cuando se salta un paso:

| Paso saltado | Síntoma |
|---|---|
| Hipótesis | No hay criterio: cualquier resultado sirve para "validar" |
| Métrica primaria | Se elige después la métrica que quedó bien (selección) |
| Power analysis | Experimento subpotenciado: "no encontramos efecto" cuando nunca se pudo detectar |
| Tamaño de muestra | IC anchísimos, decisiones sobre ruido, falsos nulos |
| Unidad de aleatorización | Correlación intragrupo ignorada: $n$ efectivo menor del planeado |
| Análisis | Peeking, múltiples comparaciones, sobre el mismo dato |
| Decisión | Lanzar un efecto que no se sostiene en el tiempo |

Saltarse los pasos 3–4 es el más caro: la potencia se planifica antes de
recoger datos y no se puede "arreglar" después con análisis. Si el experimento
sale sin el $n$ necesario, la única salida honesta es repetirlo.

## Power y tamaño de muestra

Hipótesis nula $H_0: \Delta = 0$, potencia $1-\beta$ = probabilidad de rechazar
$H_0$ cuando el efecto real es $\Delta$. Para dos brazos balanceados:

$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot 2\sigma^2}{\Delta^2}$$

con $z$ los cuantiles de la normal estándar y $\sigma^2$ la varianza de la
métrica por observación (para proporción binaria, $\sigma^2 = p(1-p)$). El
**efecto mínimo detectable** (MDE) es el $\Delta$ que un experimento con esos
$n$, $\alpha$, $\beta$ puede detectar:

$$\Delta = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \sqrt{\frac{2\sigma^2}{n}}$$

El MDE es lo que hay que reportar al decir "no encontramos efecto": no implica
que no exista, solo que el experimento no podía detectar efectos menores.

Tradeoffs:

- Bajar $\alpha$ o subir la potencia sube los $z$ y $n$ crece con su cuadrado:
  no hay manera de "ganar" los tres a la vez.
- Reducir el MDE objetivo multiplica $n$ por $1/\Delta^2$: detectar efectos 2×
  menores exige 4× la muestra.
- Bajar la varianza es la única palanca que no castiga la estadística: $n$
  cae proporcional a $\sigma^2$.

### Cuándo $n$ es inalcanzable

- **Reducir varianza** (CUPED, estratificación): la palanca con más retorno;
  ver más abajo.
- **Relajar el MDE**: aceptar detectar solo efectos grandes. Decisión de
  negocio explícita.
- **Test de una cola**: solo si la dirección del efecto es segura; da ~15 % de
  ahorro en $n$. El coste: un efecto en la dirección contraria no se detecta.
- **Outcome binario raro** (conversión 0.1 %): un MDE en puntos porcentuales
  exige $n$ enormes; considerar métricas continuas (gasto, clicks) o de
  intensidad.
- **Más exposición**: más usuarios o más tiempo, con cuidado de estacionalidad
  y drift.

Si ni con eso el $n$ cabe, el experimento no se puede hacer bien: no se hace, o
se hace declarando "no concluyente". Un A/B subpotenciado con $n$ insuficiente
es ruido caro, no evidencia.

## Peeking y alpha spending

Mirar $p$ cada día y parar cuando cruza 0.05 infla el error tipo I. Cada
inspección es una prueba adicional; con observación continua, la probabilidad
de cruzar el umbral en algún momento tiende a 1 aunque el efecto real sea 0. El
$p$ reportado tras peeking no es el $p$ nominal de 0.05: es un p-hacking
involuntario y estructural.

Soluciones:

- **Paradas predefinidas**: fijar el número y el calendario de análisis antes
  de empezar, con $\alpha$ repartido entre ellos (alpha spending, p.ej.
  O'Brien–Fleming).
- **mSPRT** (mixture sequential probability ratio test): "always valid";
  permite inspeccionar en cualquier momento sin inflar el error tipo I, a costa
  de algo de potencia respecto al test de muestra fija.
- **e-values**: estadísticos que bajo $H_0$ tienen esperanza $\le 1$; se
  combinan multiplicando y soportan parada continua con un umbral fijo. Son la
  base moderna del always-valid inference.
- **Regla práctica**: declarar la duración (fecha de parada o $n$) ANTES de
  empezar y cumplirla. Si el efecto "llega" antes de la fecha, es un resultado
  tentador, no un resultado: no se lanza. El testing secuencial permite
  adaptarse, pero solo si se planificó como secuencial.

## Reducción de varianza

**CUPED**: ajustar la métrica con una covariable pre-experimento $X$ (la misma
métrica en el periodo previo, u otra correlacionada):

$$Y' = Y - \theta\,(X - \bar{X}), \qquad \theta = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)}$$

La varianza de $Y'$ es $\operatorname{Var}(Y)\,(1 - \rho^2)$, con $\rho =
\operatorname{Corr}(X, Y)$. Un $\rho = 0.5$ reduce la varianza un 25 % (~33 %
menos $n$ para el mismo MDE); con métricas correlacionadas con el pasado
(ingresos, conversión) la ganancia es grande.

**Estratificación / blocking**: dividir la población en estratos
pre-experimento (país, segmento, dispositivo) y aleatorizar dentro de cada uno.
Elimina la componente de varianza entre estratos y garantiza balance por
covariable, lo que además hace el análisis más robusto a desequilibrios
accidentales.

Por qué **no** reducen sesgo, solo ruido: si la aleatorización es correcta no
hay sesgo que reducir — estas técnicas estrechan el IC del mismo estimador. Si
hay sesgo (asignación no aleatoria, selección), CUPED no lo arregla: ajusta por
la covariable observada, no por el confundimiento no observado. Son mejoras de
precisión, no de identificación.

## Aleatorización por clusters

Cuando la unidad de tratamiento es un grupo (tienda, ciudad, cuenta
corporativa) y las observaciones dentro del clúster correlan. El tamaño de
muestra efectivo:

$$n_{eff} = \frac{n}{1 + (m - 1)\,\rho}$$

con $m$ = tamaño medio del clúster y $\rho$ = correlación intragrupo (ICC).
Con $m = 100$ y $\rho = 0.02$, $n_{eff} = n/2.98$: hace falta ~3× más
observaciones para la misma potencia. Poco importa el número de filas: el
número de clústeres independientes es lo que cuenta; con pocos clústeres la
varianza entre ellos domina y la potencia se hunde.

Implicaciones:

- Calcular $n$ en clústeres, no en usuarios, o el diseño sale subpotenciado.
- La rampa de despliegue se hace por oleadas de clústeres o de porcentaje de
  usuarios, monitoreando guardarraíles antes de escalar. Cada escalón es una
  decisión y debe estar planificada: rampar sin regla es peeking a escala de
  despliegue.

## Múltiples métricas y multiplicidad

- **Guardarraíles**: métricas que no deben degradarse (latencia, churn,
  ingresos). No deciden el lanzamiento, lo bloquean.
- **Primaria**: la que decide. **Secundarias**: hipótesis para el siguiente
  experimento, exploración declarada como tal.

Cada métrica adicional infla el error tipo I conjunto. Con $m$ métricas
independientes, la probabilidad de al menos un falso positivo es
$1 - (1-\alpha)^m$: con 10 métricas y $\alpha = 0.05$, ~40 % de "significativo"
falso. Corrección: controlar el FDR con Benjamini–Hochberg (ver
`matematicas/estadistica.md`), declarando $m$ de antemano.

El peligro de "hacer 100 métricas y quedarse con las 3 que dieron" no es
descubrimiento, es selección de ruido: esas 3 son, con alta probabilidad,
falsos positivos. Regla: las métricas se listan antes de empezar. Las que no
estaban en la lista son hipótesis para el próximo experimento, no resultados.

## Efectos a largo plazo vs corto plazo

La métrica primaria suele ser de corto plazo (conversión en 7 días); el efecto
de negocio es de largo (retención, LTV). Un proxy imperfecto:

- Si el proxy anticipa mal el largo plazo, se lanzan cambios con efecto de
  corto plazo y daño de largo (p.ej. promociones que canibalizan demanda
  futura).
- Medir el largo plazo es caro (esperar), pero a veces imprescindible: el
  periodo de observación debe cubrir el ciclo de decisión completo, no solo la
  primera respuesta.

El experimento es la **única vía causal limpia** para estimar el efecto de una
intervención (ver `matematicas/causalidad.md`): la aleatorización rompe los confusores. Sin
aleatorización, las comparaciones pre/post o tratado/control observacionales
están contaminadas por selección y tendencia temporal.

### Cuándo NO se puede A/B

- **Intervención irreversible**: onboarding, contratos, infraestructura donde
  "no tratar" no es viable.
- **Redes sociales / spillover**: si el tratamiento de un brazo contamina al
  otro (amigos, mercados, anuncios competitivos), se viola la SUTVA y la
  estimación es sesgada. El A/B por clúster puede mitigarlo.
- **Ética o regulación**: asignar un tratamiento que daña no es aceptable.
- **Muestra insuficiente**: el $n$ necesario no es alcanzable (ver arriba).

Alternativas observacionales (con limitaciones severas): diferencias en
diferencias, regression discontinuity, variables instrumentales, matching y
ponderación por propensión. Todas dependen de supuestos de identificación que
no se pueden verificar en los datos — ver `matematicas/causalidad.md`. Sirven como
evidencia para una decisión, no como estimación de un efecto con el estatus de
un experimento.

## Fuentes

- Kohavi, R., Tang, D., Xu, Y., *Trustworthy Online Controlled Experiments: A
  Practical Guide to A/B Testing*. Cambridge University Press, 2020. Sin arXiv.
  https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/
- Kohavi, R., Longbotham, R., Sommerfield, D., Henne, R. M., *Controlled
  Experiments on the Web: Survey and Practical Guide* (A/B testing a escala).
  Data Mining and Knowledge Discovery, 2009. Sin arXiv.
  https://doi.org/10.1007/s10618-008-0114-1
- Deng, A., Xu, Y., Kohavi, R., Walker, T., *Improving the Sensitivity of
  Online Controlled Experiments by Utilizing Pre-Experiment Data* (CUPED).
  WSDM 2013. Sin arXiv. https://doi.org/10.1145/2433396.2433413
- Johari, R., Pekelis, L., Walsh, D. J., *Always Valid Inference: Continuous
  Monitoring of A/B Tests* (mSPRT). arXiv:1812.00103.
  https://arxiv.org/abs/1812.00103
- Howard, S. R., Ramdas, A., McAuliffe, J., Sekhon, J., *Always-Valid
  Sequential Testing* (e-values). arXiv:2006.05751.
  https://arxiv.org/abs/2006.05751
- Athey, S., Imbens, G. W., *The Econometrics of Randomized Experiments*.
  Handbook of Economic Field Experiments, 2017. Sin arXiv.
  https://doi.org/10.1016/bs.hefe.2016.10.004
