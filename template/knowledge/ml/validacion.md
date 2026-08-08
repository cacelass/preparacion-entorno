# Validación

## Holdout

### Tamaños de split

Repartos habituales 70/15/15 o 80/10/10. Reglas:

- El split de validación debe ser lo bastante grande para estimar la métrica
  con precisión: para una proporción $p$, el error estándar es
  $\sqrt{p(1-p)/n_{val}}$; con $n_{val}=100$, ±5 puntos. Una métrica no puede
  separar modelos si su propio ruido es mayor que la diferencia que se quiere
  detectar.
- Con datasets enormes (millones de filas), el holdout basta: la varianza de la
  estimación es despreciable y el coste del CV no se justifica.

### Estratificación

En clasificación, repartir manteniendo la proporción de clases en cada split
evita que un split salga sin clase minoritaria y la métrica salte. En
regresión, se puede estratificar por cuantiles de $y$. El objetivo es que cada
split represente la distribución real.

### Varianza de un split único

Un split es una muestra de la distribución de los datos: otro split da otro
número. La varianza del holdout viene de qué filas cayeron en validación; con
$n$ grande la estimación converge, con $n$ pequeño el resultado de un split es
casi aleatorio. Por eso el holdout de un solo split sirve como test final (dato
grande), pero no como evidencia para comparar modelos (dato pequeño).

## K-fold CV

### Sesgo y varianza de la estimación

CV estima la generalización con menos sesgo que un holdout único: cada fila
entrena y valida. Pero la varianza del estimador no baja como $1/k$: los folds
no son independientes porque sus conjuntos de entrenamiento se solapan (~89 %
con $k=10$) y la varianza la domina la partición, no $k$.

- $k$ pequeño (3–5): más datos de entrenamiento por fold, menos puntos de
  validación por fold (más ruido), menor coste.
- $k$ grande (10, o LOO con $k=n$): menos sesgo, más varianza y más coste. LOO
  tiene la varianza más alta y casi nunca paga.
- $k=5$–10 es el rango práctico; elige según el coste y el tamaño de muestra.

### Estratificado y repetido

- **StratifiedKFold**: mantiene la proporción de clases por fold;
  imprescindible con desbalance.
- **Repeated K-fold**: repetir el CV con $R$ semillas distintas y promediar.
  Reduce la varianza debida a la partición y da media ± desviación sobre
  $R \times k$ evaluaciones; es la base del seed-averaging y de los contrastes
  pareados entre modelos.

### Splits por grupos

Cuando las filas no son independientes —mismo usuario, paciente, sesión o
dispositivo— el CV aleatorio filtra: el mismo grupo aparece en train y en
validación, y el modelo memoriza el grupo en lugar de generalizar.

- **GroupKFold**: asigna cada grupo entero a un fold; ninguna fila de un grupo
  en train tiene "amigos" en validación.
- **LeaveOneGroupOut**: entrena con todos los grupos menos uno y valida en el
  grupo excluido; la estimación honesta cuando cada grupo es un contexto de
  despliegue distinto (un hospital, una tienda).
- Regla: si los grupos existen, el número de muestras efectivamente
  independientes es el número de grupos, no el de filas; la varianza del CV se
  mide en grupos.

## Series temporales

### Por qué el CV aleatorio filtra el futuro

El CV aleatorio entrena con filas posteriores a las de validación: el modelo
"aprende" valores que en producción no tendrá. La estructura causal
(pasado $\to$ futuro) se rompe y la métrica sale optimista.

### Walk-forward

Entrenar sobre el pasado e ir avanzando: cada paso entrena hasta $t$ y valida
$[t,\, t+\Delta)$; después el horizonte se desplaza. Reproduce el despliegue
real. Variantes: ventana fija (rolling) o creciente (expanding).

### Blocked, purged y embargo

- **Blocked CV** (`TimeSeriesSplit`): cortes cronológicos; cada fold entrena
  solo con el pasado del bloque que valida.
- **Purge**: eliminar de train los puntos cuya ventana de etiqueta se solapa
  con el bloque de validación (esencial con etiquetas de ventana móvil o lag de
  respuesta).
- **Embargo**: descartar unas filas inmediatamente posteriores a train para que
  la autocorrelación en el borde no contamine la validación.

### Cuándo los splits aleatorios siguen valiendo (iid)

Si las filas son intercambiables —mediciones independientes, sin dependencia
temporal ni de grupo y sin drift— el CV aleatorio es válido y eficiente (usa
todos los datos). El riesgo: la iid no se declara, se detecta. Si los datos se
recogieron en el tiempo, hay que justificar que no hay drift de covariables ni
dependencia serial. Además, si el despliegue es predictivo (entrenar el pasado,
servir el futuro), la evaluación que simula el despliegue (walk-forward) es la
que mide lo que de verdad importa, aunque los datos sean iid.

## Validación anidada (nested CV)

- **Outer loop**: estima la generalización del procedimiento completo
  (selección + entrenamiento). **Inner loop**: selecciona los hiperparámetros
  dentro de cada fold outer. El fold outer nunca participa en la selección, así
  que la estimación es honesta (Cawley & Talbot).
- Sin anidación, tunear y evaluar sobre los mismos folds infla el número: el
  optimizador memoriza el ruido de esa partición (sesgo de selección).
- **Coste**: $O(k_{out} \times k_{in})$ entrenamientos (con 5×5, 25 veces el
  coste de un fold, más el refit final). Con presupuesto corto, al menos
  reserva un holdout intacto para la selección y reporta el sesgo de no anidar.
- La anidación estima el error del *procedimiento*, no del modelo refit con
  todos los datos; el modelo final (refit con el mejor $\theta$ sobre todo)
  suele ser ligeramente mejor que lo que el outer reporta.

## Evaluación con bootstrap

- Remuestrear con reemplazo y, por cada muestra, entrenar y evaluar; o usar el
  estimador **.632**: entrenar en la muestra bootstrap, evaluar en las filas
  out-of-bag y combinar con el error de train, $e_{.632} = 0.368\,e_{train} +
  0.632\,e_{oob}$. El **.632+** corrige además el sobreajuste.
- El bootstrap da una **distribución** completa de la métrica: intervalos de
  confianza directos de los cuantiles, asimétricos si hace falta.
- Cuándo supera al CV: muestras pequeñas (un fold de CV deja muy poco
  entrenamiento), cuando se necesita un intervalo más que un punto, y con
  modelos estables. El bootstrap plano (entrenar en la muestra bootstrap y
  evaluar en OOB) es optimista para modelos que sobreajustan: la muestra
  bootstrap no ve todas las filas; el .632 lo compensa. Para selección de
  modelo, CV; para intervalo de confianza, bootstrap.

## Taxonomía de leakage

Síntoma común de todo leakage: métrica optimista en validación que colapsa en
producción. El modelo "aprendió" información que en el despliegue no existirá.

### 1. Transformaciones ajustadas sobre todo el dataset

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)      # fit sobre train + test
X_train, X_test = train_test_split(X, y)
```

El escalador "ve" los estadísticos de test: la media y la varianza de las
filas de test se filtran al modelo. **Síntoma**: score alto y estable, y
producción peor de lo que el score sugiere.

**Correcto**:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)
```

### 2. Features derivadas del target

Feature construida con el valor de $y$ o con información del futuro:

```python
df["venta_pasada"] = df.groupby("sku")["venta"].shift(-1)   # futuro
```

O mean encoding ajustado sobre el dataset completo. **Síntoma**: AUC > 0.95 con
features de pinta inocente; el modelo no sirve.

### 3. Selección de features antes del split

```python
selector = SelectKBest(k=20).fit(X, y)   # ve train y test
X = selector.transform(X)
X_train, X_test = train_test_split(X, y)
```

El selector "eligió" columnas con el test en la mano. **Síntoma**: CV brillante
y selección que no se sostiene con datos nuevos. La selección es parte del
pipeline y va dentro de cada fold.

### 4. Filas duplicadas o grupos compartidos entre train y test

El mismo usuario, paciente o transacción en ambos lados, o casi-duplicados sin
deduplicar. **Síntoma**: los folds salen fantásticos porque el modelo reconoce
filas vistas. Se arregla con deduplicación y con GroupKFold (ver arriba).

### 5. Leakage temporal

Train contiene filas posteriores al test, o features del futuro:

```python
df["media_movil_centrada"] = df["venta"].rolling(7, center=True).mean()
```

En el momento $t$ no conoces el promedio centrado de los próximos 3 días.
**Síntoma**: el modelo "predice" demasiado bien en validación y falla en línea.

### 6. Leakage hacia las etiquetas

La etiqueta se construye con datos que solo se conocerán después, o contiene
información ya derivada de las features (p.ej. el target incluye el valor del
regresor). **Síntoma**: el error es demasiado bajo para ser real; revisa cómo
se generó $y$ antes de culpar al modelo.

### 7. Reuso del test

Evaluar el mismo test repetidamente hasta que "pasa" convierte al test en un
conjunto de selección: su estimación se vuelve optimista y al desplegar con
datos nuevos el número no se sostiene. El test se toca una vez.

## Orden estándar del pipeline

```
split → fit transforms (solo train) → transform → entrenar modelo → evaluar
```

Por qué ajustar las transformaciones sobre el dataset completo es el bug
silencioso #1: `StandardScaler`, `SimpleImputer`, encodings y PCA aprenden
parámetros de los datos (media, varianza, categorías, componentes). Ajustarlos
antes del split hace que las filas de test participen en esos parámetros →
flujo de información test → modelo → métrica optimista. El código no da error y
el número se ve bien: por eso es silencioso. Usa `Pipeline` +
`ColumnTransformer` para que el `fit` de cada transform quede acotado al fold
de entrenamiento; `GridSearchCV`/`cross_validate` con Pipeline reajustan las
transformaciones dentro de cada fold automáticamente.

## Checklist de validación

| Estructura de los datos | Split que respeta esa estructura |
|---|---|
| Filas independientes, sin orden, sin grupos | StratifiedKFold / RepeatedKFold |
| Filas agrupadas (usuario, paciente, sesión) | GroupKFold / LeaveOneGroupOut |
| Serie temporal con lag o ventanas | Walk-forward + purge + embargo |
| Poca muestra, se necesita intervalo | Bootstrap (.632) |
| Se tunearán hiperparámetros | Nested CV (inner selecciona, outer evalúa) |
| Dataset enorme | Holdout estratificado único |

Regla: el split replica la estructura de dependencia de los datos, no la
inventa. Pregunta primero "¿qué haría que dos filas no fueran intercambiables?"
(grupo, tiempo, duplicado) y elige el split que lo respete.

## Fuentes

- Kohavi, R., *A Study of Cross-Validation and Bootstrap for Accuracy
  Estimation and Model Selection*, IJCAI 1995. Sin arXiv.
  https://dl.acm.org/doi/10.5555/1643031.1643047
- Cawley, G. C., Talbot, N. L. C., *On Over-fitting in Model Selection and
  Subsequent Selection Bias in Performance Evaluation*. arXiv:1006.3282.
  https://arxiv.org/abs/1006.3282
- Bergmeir, C., Benítez, J. M., *On the Use of Cross-Validation for Time Series
  Predictor Evaluation*. arXiv:1503.05341. https://arxiv.org/abs/1503.05341
- Cerqueira, V., Torgo, L., Mozetič, I., *Evaluating time series forecasting
  models: An empirical study on performance estimation methods*.
  arXiv:1905.11744. https://arxiv.org/abs/1905.11744
- Nadeau, C., Bengio, Y., *Inference for the Generalization Error*, Machine
  Learning 2003. Sin arXiv. https://doi.org/10.1023/A:1024068626366
- Efron, B., Tibshirani, R., *Improvements on Cross-Validation: The .632+
  Bootstrap Method*, JASA 1997. Sin arXiv.
  https://doi.org/10.1080/01621459.1997.10474007
- Kaufman, S., Rosset, S., Perlich, C., Stitelman, O., *Leakage in Data
  Mining: Formulation, Detection, and Avoidance*, TKDD 2012. Sin arXiv.
  https://doi.org/10.1145/2060028.2060051
- Kapoor, S., Narayanan, A., *Leakage and the Reproducibility Crisis in
  Machine Learning-based Science*. arXiv:2207.07048.
  https://arxiv.org/abs/2207.07048
- López de Prado, M., *Advances in Financial Machine Learning* (purged CV,
  embargo), Wiley 2018. Sin arXiv.
  https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
