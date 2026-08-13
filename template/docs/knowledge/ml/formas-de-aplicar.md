# Formas de aplicar un modelo

La diferencia entre un modelo que engaña y uno que predice es el protocolo de
aplicación: cómo se divide, dónde se ajusta cada transformación y cómo se
estima el error. Este documento fija la disciplina.

## El orden del pipeline y por qué el orden es corrección

El orden no es cosmético: es lo que decide si la métrica de validación es
honesta. La secuencia correcta es siempre:

1. **Dividir primero** (train/validation/test).
2. **Ajustar el preprocesado solo en train** (fit de scaler, imputer, encoder,
   selección de features, one-hot).
3. **Transformar** con los parámetros ajustados: `train.fit_transform(X)`,
   `test.transform(X)` — nunca `fit_transform` sobre test.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])
pipe.fit(X_train, y_train)          # el scaler aprende media/varianza de train
y_pred = pipe.predict(X_test)       # y transforma test con esos parámetros
```

Un transformador ajustado sobre train+test ya ha "visto" la distribución de
test; el error de validación queda inflado de forma optimista y la
generalización real es peor de lo medido. Con pocos transformadores el sesgo
es numéricamente pequeño; con target encoding, imputación de grupos o
selección de features se vuelve catastrófico.

### Taxonomía de leakage

| Tipo | Qué lo causa | Magnitud típica |
|------|--------------|-----------------|
| Preprocesado global | scaler/imputer/encoder ajustados sobre todos los datos | pequeña a media |
| Target leakage | features construidas con el target (estadísticas de y, valores futuros de y) | alta |
| Temporal | features con información del futuro; shuffle en series temporales | alta |
| Via validación | early stopping o tuning sobre test; features elegidas sobre el dataset completo | media |
| Selección antes del split | filtrado de features con el dataset completo, luego CV | media |

- **Target leakage**: cualquier feature que se pueda derivar de y (media de y
  por grupo, lags del target sin desfase, one-hot del índice de la fila de y).
  El modelo "aprende" a mirar la respuesta. Se detecta cuando las features más
  importantes según el modelo no tienen explicación causal.
- **Temporal**: features calculadas con datos posteriores al instante de la
  predicción, o validación con shuffle en series. Para series, el test siempre
  es posterior al train y toda feature se computa solo con el pasado.
- **Via validación**: si se repite early stopping, búsqueda de umbral o ajuste
  de hiperparámetros sobre el mismo validation fold, ese fold deja de medir y
  pasa a ser parte del entrenamiento. El resultado se debe medir en un test
  intacto.
- **Feature selection antes del split**: calcular varianza, correlación o
  importancia sobre todo el dataset para decidir qué entra, y luego validar,
  informa la selección con datos de test.

## Estrategias de remuestreo

| Estrategia | Cuándo | Riesgo / nota |
|------------|--------|----------------|
| Holdout | Baseline rápido, datasets grandes | Alta varianza; gasta datos en validación |
| K-fold (CV) | Estándar; estima media y varianza del error | Repetir el ajuste K veces |
| Stratified | Clasificación desbalanceada | Mantiene la proporción de clases por fold |
| Repeated CV | Reducir varianza del estimador de error | Costo × número de repeticiones |
| GroupKFold / LeaveOneGroupOut | Agrupados (paciente, sesión) | No mezclar filas de un grupo en splits |
| TimeSeriesSplit (walk-forward) | Series temporales | Train siempre anterior al test |
| Blocked / purged CV | Series con dependencia cercana | Bloques con gap entre train y test |
| Nested CV | Elegir hiperparámetros y estimar el error | Inner selecciona, outer estima |
| Bootstrap | Intervalos de confianza, datasets pequeños | Filas con reemplazo; optimista para tuning |

- **GroupKFold / LeaveOneGroupOut**: si varias filas provienen de la misma
  entidad (el mismo paciente en medicina, el mismo autor en texto), las filas
  de un grupo no son independientes: mezclarlas entre train y test infla el
  rendimiento porque el modelo ha visto "variantes" del mismo individuo. El
  grupo es la unidad de split.
- **Series temporales**: el split es por tiempo, no aleatorio. Walk-forward:
  cada paso entrena con todo lo anterior y valida la ventana siguiente.
  Blocked/purged: separa train y test con un hueco (gap) para que información
  de los bordes (features con lags) no se filtre.
- **Nested CV** (CV anidada) es la única forma honesta de optimizar
  hiperparámetros y reportar un error:

```
dataset → outer folds (estiman el error real)
            └─ cada fold: inner CV (elige hiperparámetros) → entrena → evalúa
```

  Sin la capa exterior, el error del mejor modelo sobre la búsqueda es
  optimista: se ha optimizado sobre la misma muestra que evalúa.
- **Bootstrap**: muestrear N filas con reemplazo, entrenar, evaluar sobre las
  no muestreadas (out-of-bag). Bueno para intervalos de confianza del modelo;
  no para comparar hiperparámetros (el error OOB es optimista en selección).

## Escalado

| Método | Fórmula | Propiedad |
|--------|---------|-----------|
| Standard | $z=(x-\mu)/\sigma$ | Media 0, varianza 1; sensible a outliers |
| Robust | $z=(x-\text{mediana})/\text{IQR}$ | Robusto a outliers |
| MinMax | $z=(x-\min)/(\max-\min)$ | Rango fijo $[0,1]$; sensible a outliers |

Qué modelos lo necesitan:

- **Los que miden distancias o kernels**: KNN, SVM, k-means, PCA, LDA,
  regularización L1/L2 (la penalización asume escalas comparables), redes
  neuronales (convergencia del optimizador).
- **Los que NO lo necesitan**: árboles y ensembles de árboles (random forest,
  gradient boosting). Son invariantes a transformaciones monótonas; escalar
  solo añade cómputo y ruido de redondeo.
- Regla: si el modelo usa distancia, gradiente o norma, escala; si corta por
  umbrales de feature, no. La excepción es la interpretabilidad: con
  regresión logística o lineal regularizada, coeficientes comparables solo
  existen con features escaladas.

## Imputación

- **Media/mediana**: no introducen valores nuevos, pero distorsionan la
  varianza (la reducen al colapsar múltiples valores al centro) y rompen
  correlaciones y momentos superiores. Aceptables como baseline; malas cuando
  la magnitud de la feature importa (distancias, boosting con splits sobre
  valores).
- **MICE** (imputación por ecuaciones encadenadas): modela cada columna con
  las demás, iterativamente. Preserva correlaciones; caro y estocástico —
  usar un seed fijo.
- **KNN**: imputa con los vecinos más cercanos según las demás features.
  Preserva estructura local; se degrada en dimensiones altas y con outliers.
- **Model-based**: el valor faltante se predice con un modelo auxiliar
  entrenado sobre filas completas. Flexible, pero puede ocultar el patrón de
  missingness.
- **Missingness como feature**: añadir un indicador binario `is_missing` por
  columna con huecos. Gratis y a menudo informativa: si el dato no está porque
  un proceso falló o un usuario no llegó ahí, la ausencia es señal.
- **Cuidado con MNAR** (missing not at random): si la probabilidad de faltar
  depende del valor faltante en sí (un sensor que falla con valores altos),
  ninguna imputación lo recupera; imputar introduce sesgo sistemático. Es un
  problema del proceso de datos, no del algoritmo.
- Todos los imputadores se ajustan en train y se aplican con `transform` en
  test; el imputador guarda sus parámetros como parte del pipeline.

## El bucle de ingeniería de features

```
hipótesis → construir → validar → seleccionar → repetir
```

Cada feature es una hipótesis sobre el mundo que el modelo no ve aún. El
bucle es: proponer la feature, validarla (¿reduce el error de validación?
¿aporta varianza no explicada por las existentes?), y seleccionar.

Selección de features:

- **Varianza**: features constantes o casi constantes no aportan; el modelo
  puede gastar splits en ruido.
- **Correlación**: features altamente correlacionadas entre sí duplican
  información; en modelos lineales inflan varianza de los coeficientes (VIF).
- **Importancia por permutación**: importa la caída de rendimiento al permutar
  una feature; más fiable que la importancia por impurity en boosting.
- **Lasso**: regularización L1 empuja coeficientes a cero — selección
  incorporada en modelos lineales.
- **RFE** (recursive feature elimination): elimina las menos importantes,
  reajusta, repite.

El valor de las features de dominio: una feature de negocio bien construida
(ratio, descomposición del problema, conocimiento del proceso) suele superar
a cien features genéricas. La literatura de boosting de la sección de fuentes
lo confirma: en soluciones ganadoras de Kaggle, la ingeniería de features de
dominio marca más diferencia que el algoritmo.

El peligro de muchas features inútiles: aumentan la varianza del modelo,
alimentan el sobreajuste de la selección (con 1000 features se encuentra
siempre alguna que parece importar), encarecen el entrenamiento y hacen
inestable la selección (pequeños cambios de datos cambian qué features entran).

## Consistencia train/serve

El código que transforma en entrenamiento debe ser el mismo que transforma en
producción. La regla de oro: no existe una función `preprocess_train` y otra
`preprocess_serve`; existe una sola transformación parametrizada.

- Usar `sklearn.Pipeline` de punta a punta (imputación → escalado → encoding →
  modelo) y serializar el pipeline entero con el modelo. El pipeline conoce
  sus parámetros ajustados: `model.predict` ya aplica todo el preprocesado.
- En NN y frameworks, el preprocesado va como capa del modelo o como función
  compartida importada por ambos caminos (training y serving), nunca copiada.
- **Feature stores**: repositorio centralizado de features con su código de
  transformación y su versión. Garantizan que la feature que se sirve es la
  misma que se entrenó (misma agregación, mismo ventana temporal, mismos
  defaults para valores ausentes). Un drift entre train y serve que cambie
  silenciosamente la definición de una feature es un bug que ninguna métrica
  offline detecta.

## Diagnósticos

- **Curvas de aprendizaje** (error train/validación vs tamaño de dataset):
  - Ambas altas y cercanas → underfitting: más datos no basta, hace falta un
    modelo más flexible o mejores features.
  - Train bajo, validación alta, brecha que no se cierra con datos →
    overfitting: regularizar, reducir complejidad, más datos.
  - Curvas convergentes y bajas → zona sana; ahí el límite es el ruido
    irreducible, no el modelo.
- **Análisis de residuos** (regresión): los residuos $\hat{y}-y$ deben ser
  ruido sin patrón. Patrones en función de $\hat{y}$ (embudo: varianza
  creciente), de una feature (efecto no modelado), o en el tiempo (deriva)
  indican qué falta capturar.
- **Chequeo de calibración**: en clasificación, ¿los 100 casos con
  probabilidad ~0.8 se cumplen un 80%? Binned confidence vs accuracy. Modelos
  descalibrados dan malas decisiones con umbrales.
- **Análisis de error por slice**: partir los errores por segmento (feature
  categórica, rango de valores, grupo de usuarios) y encontrar dónde concentra
  el modelo sus fallos. Responder "dónde falla y por qué" dirige la siguiente
  iteración (más datos de ese slice, features nuevas para ese grupo) mucho
  mejor que una métrica global.

## Receta práctica para un primer baseline sólido

La escalera del baseline — de lo simple a lo complejo, en ese orden:

1. **Lineal** (regresión logística/lineal regularizada): rápido, estable,
   interpretable, define el suelo del problema. Con buena ingeniería de
   features sorprende.
2. **Árboles**: un árbol o random forest para capturar interacciones no
   lineales que el lineal no ve.
3. **Ensemble de boosting**: gradient boosting (XGBoost/LightGBM/CatBoost)
   con ajuste básico.
4. **Ajuste fino** de hiperparámetros solo después de tener el pipeline
   ordenado y las features validadas.

{% if use_optuna %}
Este proyecto trae `tools/tune_model.py` con Optuna (`make tune`): optimiza los
hiperparámetros sobre el pipeline ya ordenado, con CV anidada o holdout de
validación, y deja el mejor trial en `models/best_params.json`. Se usa en el
escalón 4, nunca antes de tener un baseline.
{% endif %}

Solo entonces añadir complejidad (modelos nuevos, features caras,
ensamblados). La regla que se repite: **ordenar el pipeline, validar las
features y tener un baseline honesto rinde más que el mejor algoritmo sin
disciplina**.

## Fuentes

- Chen, T., Guestrin, C., "XGBoost: A Scalable Tree Boosting System" —
  arXiv:1603.02754 — https://arxiv.org/abs/1603.02754
- Prokhorenkova, L., et al., "CatBoost: unbiased boosting with categorical
  features" — arXiv:1706.09516 — https://arxiv.org/abs/1706.09516
- Cawley, G. C., Talbot, N. L. C., "On Over-fitting in Model Selection and
  Subsequent Selection Bias in Performance Evaluation" — arXiv:1006.3282 —
  https://arxiv.org/abs/1006.3282
- Kapoor, S., Narayanan, A., "Leakage and the Reproducibility Crisis in
  Machine-Learning-Based Science" — arXiv:2207.07048 —
  https://arxiv.org/abs/2207.07048
