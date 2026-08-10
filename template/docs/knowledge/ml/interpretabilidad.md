# Interpretabilidad

Qué hace un modelo y por qué, con métodos específicos, agnósticos, globales y
locales; qué puede y qué no puede afirmar cada uno, y qué reportar. Complementa
a `matematicas/causalidad.md`: aquí la pregunta es sobre el modelo, allá sobre el mecanismo
que genera los datos.

## Por qué interpretar

- **Depuración:** un modelo que acierta por confusores o fugas de datos
  ("leakage") suele tener explicaciones raras; la explicación es el primer
  detector de que el modelo aprendió lo que no debías.
- **Confianza:** quien decide (producto, clínica, negocio) no firma sobre una
  métrica promedio; firma sobre casos. Una explicación por instancia convierte
  "el modelo va bien" en "el modelo se comporta así en este caso".
- **Cumplimiento:** GDPR (derecho a explicación, art. 22), AI Act (transparencia
  por nivel de riesgo) y la regulación financiera (BCBS/ECB: explicabilidad de
  modelos crediticios) exigen una traza de por qué se decidió.
- **Importancia ≠ causalidad:** todo ranking de features —SHAP, permutación,
  ganancia— mide contribución a la predicción *dentro del modelo* bajo la
  distribución de entrenamiento. Con correlaciones, el crédito se reparte de
  forma arbitraria y sensible al método; no mide efectos causales. Ver
  `matematicas/causalidad.md`: explicar el modelo no identifica intervenciones.

## Específicos del modelo

### Coeficientes lineales

$f(x) = \beta_0 + \sum_j \beta_j x_j$: $\beta_j$ es el cambio en el objetivo por
unidad de $x_j$ manteniendo el resto constante. Dos problemas:

- **Escala:** $\beta_j$ depende de la unidad de $x_j$. Con $x_1$ en años y $x_2$
  en días, $\beta_2$ es ~365 veces menor sin que implique menos relevancia.
  Comparar coeficientes crudos entre features exige variables estandarizadas
  (z-scores) o reportar el efecto por desviación estándar.
- **Multicolinealidad:** con $x_j$ correlacionadas, los $\beta_j$ individuales
  son inestables y con varianza alta; el ajuste conjunto es bueno y el reparto
  individual no. Intervalos de confianza anchos delatan el problema.

### Odds ratios en logística

En $P(y=1) = \sigma(\beta_0 + \beta^T x)$, $\exp(\beta_j)$ es el *odds ratio* de
subir $x_j$ una unidad: el odds $p/(1-p)$ se multiplica por $\exp(\beta_j)$. Es
una razón, no una diferencia de probabilidad: para $p$ cerca de 0 o 1, el mismo
OR mueve la probabilidad muy poco. Para comunicar efectos en probabilidad,
reporta el cambio en $P$ con el resto fijo — dependerá del punto de partida.

### Importancia de features en árboles

La importancia por ganancia suma, sobre todos los splits, la reducción de
impureza (Gini, entropía o MSE) atribuida a cada feature. Sesgos conocidos:

- **Alta cardinalidad / continuas:** las variables con muchos cortes posibles
  tienen más oportunidades de split y su ganancia se reparte en muchos nodos
  poco profundos; las categóricas de baja cardinalidad quedan infravaloradas.
  Los cortes favorables por azar inflan la ganancia.
- **Qué no mide:** ni la dirección del efecto (la ganancia es no negativa) ni
  la forma funcional; no distingue interacción de efecto principal y subestima
  features correlacionadas (la primera "se lleva" la ganancia). Es una
  heurística de ranking, no una medida de contribución puntual.

## Globales agnósticos

Funcionan sobre el modelo entrenado sin mirar su estructura; solo usan
predicciones.

### Importancia por permutación

Procedimiento:

1. Fijar la métrica de referencia $L(y, f(X))$ en datos de evaluación.
2. Para cada feature $j$: permutar sus valores al azar (romper la asociación
   con $y$), volver a predecir y medir la caída de la métrica.
3. Importancia = caída de rendimiento al romper $x_j$:

$$\text{PI}_j = \mathbb{E}\left[L(y, f(X^{(j)}_{perm}))\right]
- \mathbb{E}\left[L(y, f(X))\right].$$

Caveats:

- **Correlación:** si $x_1$ y $x_2$ están correlacionadas, permutar solo $x_1$
  empareja valores que no coexisten en la distribución conjunta, el modelo
  extrapola y la caída mezcla "pérdida de $x_1$" con "fila imposible". Reporta
  el *model class reliance* (MCR: rango de importancias sobre modelos
  reentrenados con subsets) si te importa la estabilidad del ranking.
- **Qué mide:** degradación predictiva. No hay dirección, y una feature puede
  tener importancia ~0 por redundancia total (su información está duplicada)
  sin ser causalmente irrelevante.
- **Repetibilidad:** con pocas muestras o features raras, el valor varía;
  permuta varias veces y reporta la media ± desviación.

```python
def permutation_importance(model, X, y, metric, rng=None, n_repeats=10):
    base = metric(y, model.predict(X))
    out = {}
    for j in range(X.shape[1]):
        vals = []
        for _ in range(n_repeats):
            Xp = X.copy(); Xp[:, j] = rng.permutation(X[:, j])
            vals.append(base - metric(y, model.predict(Xp)))
        out[j] = (np.mean(vals), np.std(vals))
    return out
```

### PDP e ICE

Dependencia parcial (PDP) de la feature $S$: promedia las predicciones variando
$x_S$ y manteniendo el resto en sus valores observados:

$$\hat f_S(x_S) = \frac{1}{n}\sum_{i=1}^n f(x_S, x^{(i)}_{C}), \qquad
x^{(i)}_{C} = \text{resto de la fila } i.$$

Las ICE son las curvas individuales $x_S \mapsto f(x_S, x^{(i)}_{C})$ antes de
promediar: revelan heterogeneidad que el promedio oculta (efectos opuestos que
se cancelan, interacciones).

**Riesgo de extrapolación:** la PDP marginaliza sobre la distribución del resto;
si $x_S$ se aleja de su soporte o de la conjunta con $x_C$, el modelo evalúa
regiones sin datos. Con features correlacionadas escribe relaciones imposibles.
Alternativa: ALE (efectos locales acumulados), que integra la derivada
*condicional* $\mathbb{E}[f'_j \mid X_j]$ y solo promedia donde hay datos — no
extrapola, pero es más caro y su lectura es local.

### SHAP

Valores de Shapley (teoría de juegos cooperativos): dado un juego $v(S)$ que
asigna el pago al conjunto $S \subseteq N$ de jugadores, el valor del jugador
$i$ es el reparto equitativo de la contribución marginal:

$$\phi_i(v) = \sum_{S \subseteq N\setminus\{i\}}
\frac{|S|!(|N|-|S|-1)!}{|N|!}
\left[v(S \cup \{i\}) - v(S)\right].$$

En ML los "jugadores" son features y $v(S)$ es la predicción esperada
condicionando al subconjunto $S$ (de ahí la necesidad de definir qué pasa con
las ausentes: integrarlas sobre una distribución de fondo). Propiedades:

- **Aditividad (eficiencia):** $f(x) = \phi_0 + \sum_j \phi_j$, con $\phi_0$ la
  predicción base (media del fondo); los $\phi_j$ suman exactamente la
  desviación de la predicción sobre la base.
- **Consistencia:** si un modelo cambia de modo que la contribución marginal de
  $i$ no disminuye para ningún subconjunto, $\phi_i$ no disminuye. Es la
  propiedad que la importancia por ganancia de árboles *no* cumple. La
  interpretación local resultante es una *explicación lineal local*: cerca de la
  instancia, el modelo se aproxima por un hiperplano con pendientes $\phi_j$.
- **Simetría y jugador nulo:** features con contribuciones marginales idénticas
  reciben el mismo valor; una que no cambia la predicción en ningún subconjunto
  recibe 0.

**KernelSHAP vs TreeSHAP:** KernelSHAP es model-agnostic: estima los $\phi_i$
con una regresión lineal ponderada sobre subconjuntos aleatorios (muestreo +
background de ~50-100 filas); sirve para cualquier modelo pero es lento y con
varianza. TreeSHAP es específico de árboles y calcula los $\phi_i$
**exactamente** en tiempo polinomial recorriendo la estructura del árbol (con
perturbación "interventional" o "path-dependent"); es el default práctico para
RandomForest/XGBoost/LightGBM.

**Modos de fallo:**

- **Features correlacionadas:** el reparto depende de la estrategia de
  integración (path-dependent vs interventional) y del background; con
  correlación fuerte, los $\phi_i$ individuales se vuelven arbitrarios aunque
  la suma (la predicción) sea correcta. Interpreta el conjunto, no el detalle.
- **Dependencia del modelo, no de los datos:** SHAP explica el modelo, no la
  verdad subyacente. Dos modelos con el mismo rendimiento pueden tener SHAP
  opuestos; SHAP solo es fiel al modelo que lo generó.
- **Fondo:** cambiar el background cambia $\phi_0$ y todos los $\phi_i$.
- **Multiclase:** hay un vector de valores por clase; resumir con $|\cdot|$
  promedio es un resumen, no el valor de una clase.

**PDP vs SHAP:** el PDP muestra *cómo cambia la predicción* con la feature
(cuánto); SHAP muestra *cuánto contribuye* a cada instancia concreta (qué
parte del delta sobre la base). Son complementarios: SHAP da la importancia y
la dirección media; PDP/ICE dan la forma de la relación (umbrales, no
linealidad, saturación).

### Efectos globales (GFE)

Efecto global de la feature $j$: $G_j(x_j) = \int f(x_j, x_{-j})\,dP(x_{-j})$
es la respuesta promedio; la PDP es su estimador empírico. La variabilidad
alrededor de $G_j$ (la dispersión de las ICE) mide interacción: si las ICE se
cruzan o se dispersan, no hay efecto principal limpio y el promedio engaña.
Reporta el efecto junto con su dispersión, no el promedio solo.

## Locales

### LIME

Sustituto local: para la instancia $x$, (1) muestrear vecinos perturbándola;
(2) predecir con el modelo negro; (3) ajustar un modelo interpretable (lineal o
árbol pequeño) ponderando los vecinos por proximidad a $x$; (4) los
coeficientes del sustituto son la explicación:

$$\xi(x) = \arg\min_{g \in G} \sum_z \pi_x(z)\,\left[g(z) - f(z)\right]^2
+ \Omega(g).$$

**Problema de estabilidad:** el sustituto depende de la semilla, del tamaño de
la perturbación, del kernel de proximidad y del número de features del
sustituto ($K$). Dos ejecuciones pueden dar explicaciones distintas; estabiliza
con muestreos repetidos y reporta la variabilidad.

### SHAP local

Los $\phi_j$ de una instancia son la explicación local: cada feature aporta una
parte aditiva del desplazamiento de la predicción sobre la base. Es una única
explicación globalmente consistente (todos los $\phi_j$ vienen del mismo valor
de Shapley), a diferencia de LIME, que entrena un sustituto por instancia. El
waterfall es la lectura: base → suma de $\phi_j$ → predicción final.

### Contrafactuales

Explicación por cambio mínimo: para $x$ con predicción $f(x)$, encontrar $x'$
con predicción deseada $y'$ y costo mínimo:

$$\min_{x'} d(x, x') \quad \text{s.a.} \quad f(x') = y'.$$

Dicen "qué tendría que haber sido distinto", que es lo que un humano entiende:
"le denegaron el crédito; se habría aprobado con plazo ≥ 36 meses". Reglas:
que $x'$ sea realista (dentro del soporte), próximo (perturbación mínima),
esparso (pocos cambios) y que se ofrezcan varias alternativas, no una.

## Modelos sustitutos

Sustituto global: entrenar un modelo interpretable $g$ (regresión logística,
árbol pequeño) sobre las predicciones del modelo negro $f$ y reportar $g$ como
aproximación. Solo es fiel si la fidelidad $P(g(X) \approx f(X))$ es alta *en
las regiones que importan*; si $f$ es muy no lineal, ningún sustituto global es
fiel y reporta una caricatura. Útil como resumen ejecutivo, no como prueba.
Complementa a LIME: el sustituto global dice "en promedio el modelo se parece a
esto"; LIME dice "para esta instancia, esto".

## Atención y embeddings

- **Atención ≠ explicación:** los pesos de atención son internos del modelo,
  no están entrenados para justificar decisiones y no hay referencia de verdad
  para validarlos; con pocas instancias generan pesos no identificables.
  Reportar atención como explicación es una afirmación no verificada. Para
  atribución en texto/redes, métodos de gradient/perturbación (gradient ×
  input, oclusión, integrated gradients) son más fiables.
- **Embeddings:** los vectores aprendidos tienen estructura geométrica
  (analogías, vecinos semánticos), pero interpretar una dimensión individual es
  ilusorio: son direcciones combinadas, no unidades. Interpreta el *espacio*
  (vecinos, UMAP de un subconjunto, prototipos), no coordenadas. Las capas
  intermedias aprenden conceptos emergentes sin etiquetar; usa análisis de
  comportamiento (activaciones, saliency) y no afirmes que "la neurona X
  representa Y".

## Práctica

**Informe de interpretabilidad estándar:**

| Sección | Contenido |
|---|---|
| Resumen global | ranking de importancia (permutación o SHAP bar) con método, semilla, estabilidad |
| Top local | waterfall/force plots de casos: 3-5 predicciones altas, 3-5 bajas, 3 errores |
| Slices de fallo | subgrupos (edad, región, clase) donde el error sube; explica esos slices |
| Límites | correlaciones entre features top, dependencia del background, advertencia de no-causalidad |

**Importancia global vs SHAP:** para priorizar features (ingeniería, recorte,
monitoreo) basta permutación o ganancia de árboles, son baratos. Necesitas SHAP
(con explicaciones por instancia) cuando se comunica a un humano la decisión de
un caso (rechazo crediticio, diagnóstico), se audita, o se quiere cuantificar
la contribución puntual con la propiedad de consistencia. No uses SHAP para
decidir qué feature eliminar: usa validación cruzada con el modelo reentrenado.

**Costo de las explicaciones.** Permutación y PDP/ICE son del orden de
$O(\text{predict} \times n \times p)$; TreeSHAP es barato incluso en tabular
grande. KernelSHAP sobre KNN o redes es caro: por instancia resuelve una
regresión con muestreo de subconjuntos sobre un background (50-100 filas); en
tabular usa TreeSHAP cuando puedas y reserva KernelSHAP para unas pocas docenas
de instancias. SHAP es model-agnostic de verdad solo con KernelSHAP; los
explainers "rápidos" son específicos del tipo de modelo.

{% if use_shap %}
## SHAP en este proyecto (`use_shap` activo)

`explain_models` (en `{{ project_slug }}/models/predict_model.py`) genera dos
gráficas por modelo en `reports/figures/`:

- **`shap_bar_{modelo}.png`** — importancia global media (mean $|\text{SHAP}|$):
  ranking de arriba (más importante) abajo. Úsalo como resumen ejecutivo y para
  priorizar features; el ranking refleja el modelo entrenado, no un efecto
  causal.
- **`shap_beeswarm_{modelo}.png`** — un punto por (instancia, feature) sobre el
  eje $x$ de $\phi_j$; el color codifica el valor de la feature (rojo = alto,
  azul = bajo). Lee: anchura = importancia, dirección = signo del impacto,
  gradiente de color = forma de la relación (si rojo y azul quedan en lados
  opuestos, el efecto cambia de signo con el valor).

El explainer se elige por tipo de modelo: `TreeExplainer` (exacto y rápido)
para RandomForest/DecisionTree/XGBoost/LightGBM; `LinearExplainer` para
regresiones; `KernelExplainer` (aproximado y lento) para KNN y otros, con 50
muestras de fondo y 100 filas a explicar por defecto. En binario se explica la
clase positiva; en multiclase, los valores se promedian en valor absoluto.

Reglas de lectura: interpreta el *conjunto* de features (con correlaciones, el
reparto individual es inestable); la dirección importa (una feature con $\phi_j$
medios ~0 no mueve la predicción aunque la barra la muestre); y nunca leas
causalidad de estos plots — véase `matematicas/causalidad.md`.
{% endif %}

## Fuentes

- M. T. Ribeiro, S. Singh, C. Guestrin, "Why Should I Trust You? Explaining the
  Predictions of Any Classifier", 2016. arXiv:1602.04938.
  https://arxiv.org/abs/1602.04938
- S. M. Lundberg, S.-I. Lee, "A Unified Approach to Interpreting Model
  Predictions", 2017. arXiv:1705.07874. https://arxiv.org/abs/1705.07874
- S. M. Lundberg, G. G. Erion, S.-I. Lee, "Consistent Individualized Feature
  Attribution for Tree Ensembles", 2018. arXiv:1802.03888.
  https://arxiv.org/abs/1802.03888
- A. Fisher, C. Rudin, F. Dominici, "All Models are Wrong, but Many are Useful:
  Learning a Variable's Importance by Studying an Entire Class of Prediction
  Models Simultaneously", 2018. arXiv:1801.01489. https://arxiv.org/abs/1801.01489
- A. Goldstein, A. Kapelner, J. Bleich, E. Pitkin, "Peeking Inside the Black
  Box: Visualizing Statistical Learning with Plots of Individual Conditional
  Expectation", 2013. arXiv:1309.6392. https://arxiv.org/abs/1309.6392
- D. W. Apley, J. Zhu, "Visualizing the Effects of Predictor Variables in Black
  Box Supervised Learning Models", 2016. arXiv:1612.08468.
  https://arxiv.org/abs/1612.08468
- S. Wachter, B. Mittelstadt, C. Russell, "Counterfactual Explanations without
  Opening the Black Box: Automated Decisions and the GDPR", 2017.
  arXiv:1711.00399. https://arxiv.org/abs/1711.00399
- Z. C. Lipton, "The Mythos of Model Interpretability", 2016. arXiv:1606.03490.
  https://arxiv.org/abs/1606.03490
- C. Rudin, "Stop Explaining Black Box Machine Learning Models for High Stakes
  Decisions and Use Interpretable Models Instead", 2018. arXiv:1811.10154.
  https://arxiv.org/abs/1811.10154
- S. Jain, B. C. Wallace, "Attention is not Explanation", 2019. arXiv:1902.10186.
  https://arxiv.org/abs/1902.10186
- I. E. Kumar, S. Venkatasubramanian, C. Scheidegger, S. Friedler, "Problems
  with Shapley-value-based explanations as feature importance measures", 2020.
  arXiv:2002.11097. https://arxiv.org/abs/2002.11097
