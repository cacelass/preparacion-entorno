# Optimización de hiperparámetros

## El problema y el presupuesto de búsqueda

HPO (Hyperparameter Optimization): dado un modelo $f(x;\theta)$ con
hiperparámetros $\theta \in \Theta$ y una métrica de validación $V$, encontrar
$\theta^* = \arg\max_{\theta \in \Theta} V(\theta)$. Cada evaluación cuesta un
entrenamiento completo.

Se enfrentan dos cantidades: la precisión del modelo y el presupuesto
computacional. La utilidad marginal de cada trial decrece; el objetivo no es
"el mejor $\theta$ posible" sino "el mejor $\theta$ dentro de un presupuesto".

Regla práctica: **tune late, tune little**. Primero un pipeline correcto y una
métrica honesta; después, pocos parámetros y pocos trials. Tunar antes de
tener un baseline confiable es construir sobre ruido: cada cambio de features
o de pipeline invalida los hipers encontrados.

## Grid vs random search

Grid evalúa el producto cartesiano: $|A_1| \times |A_2| \times \dots$. En $d$
dimensiones con $k$ valores por eje requiere $k^d$ evaluaciones. Con 6+ ejes,
inviable.

Resultado de Bergstra y Bengio (2012): en la práctica **solo un puñado de
hiperparámetros importa**; el resto apenas mueve la métrica. El grid reparte
las evaluaciones a lo largo de ejes irrelevantes y deja sin cubrir los que
importan; random, con $n$ evaluaciones, muestrea $n$ valores distintos de cada
parámetro.

| $n$ | grid ($d{=}2$) | grid ($d{=}10$) | random (cualquier $d$) |
|-----|----------------|-----------------|------------------------|
| 100 | 10 por eje     | 1.58 por eje    | 100 por eje            |

Regla: salvo espacios de 1–2 dimensiones, **usa random o, mejor, una búsqueda
bayesiana**. El grid solo gana cuando el espacio es chico y quieres enumerarlo
de forma reproducible.

## Bayesian optimization

BO sostiene dos piezas: un **surrogate** que aproxima $V(\theta)$ barato, y
una **acquisition function** que decide dónde evaluar a continuación.

Con un GP como surrogate, la adquisición estándar es Expected Improvement
(EI):

$$EI(x) = \mathbb{E}\big[\max\big(f^* - f(x),\; 0\big)\big],$$

donde $f^*$ es el mejor valor visto y el esperado se toma sobre la posterior
del GP. EI balancea explotar (media alta cerca de $f^*$) y explorar (varianza
alta donde aún no has mirado): el GP aporta $\mu(x)$ y $\sigma^2(x)$ en cada
punto, y $EI$ crece con ambas.

### TPE

Tree-structured Parzen Estimator modela dos densidades sobre los puntos: $l(x)$
sobre los buenos (métrica por encima de un cuantil) y $g(x)$ sobre los malos.
El ratio $l(x)/g(x)$ juega el papel de adquisición:

$$\mathrm{EI}(x) \propto \frac{l(x)}{g(x)}.$$

Ventaja práctica sobre el GP: no estima un modelo global del espacio, trata
cada parámetro con un árbol unidimensional y por eso **maneja espacios mixtos**
(continuos + enteros + categóricos) y condicionales sin fricción. Los GP con
kernel de cuadrado exponencial sufren con categóricos y con huecos del espacio;
TPE no.

## Optuna en la práctica

API Study/Trial:

```python
import optuna

def objective(trial):
    lr    = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    depth = trial.suggest_int("max_depth", 2, 20)
    fam   = trial.suggest_categorical("family", ["linear", "gbm"])
    return -train_and_eval(lr, depth, fam)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

- **Study**: el problema. Guarda historial y mejor trial; persiste en SQLite
  para continuar sesiones entre ejecuciones.
- **Trial**: una evaluación. `trial.suggest_*` declara el espacio y devuelve
  un valor; el sampler decide qué devolver.

### Samplers

- **TPESampler**: por defecto. Buena elección general, barato, soporta todo
  tipo de parámetros.
- **CMA-ES**: espacios continuos, cuando TPE se estanca o hay presupuesto
  grande. Peor con categóricos puros.
- **NSGA-II**: multiobjetivo. En vez de `direction` único, lista de objetivos;
  `study.best_trials` devuelve el frente de Pareto ("maximizar AUC
  minimizando latencia"). Para una sola métrica, no lo uses.

### Pruning

Si dentro de un trial la métrica se reporta por epochs/folds, el pruner corta
trials condenados y libera presupuesto:

```python
study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
```

- **MedianPruner**: corta si el mejor valor parcial del trial está por debajo
  de la mediana histórica en el mismo step. Robusto y barato.
- **Hyperband**: asigna presupuestos crecientes y elimina a los peores en cada
  ronda; muy eficiente con trials caros.
- El modelo debe llamar `trial.report(valor, step)` y comprobar
  `trial.should_prune()`; sin eso, el pruner no tiene de qué comer.

### Espacios condicionales

Cuando un parámetro solo existe para parte de las configuraciones, decláralo
dentro del `if`:

```python
fam = trial.suggest_categorical("family", ["linear", "gbm"])
if fam == "gbm":
    depth = trial.suggest_int("max_depth", 2, 20)
else:
    depth = 0  # no aplica
```

TPE y el modelo de árbol de Optuna manejan el hueco de forma natural; un grid
no puede: hay que fijar valores por rama a mano. Es otra razón para preferir
BO con árboles.

### Multi-fidelity

Succesive halving evalúa muchos candidatos con pocos recursos, descarta la
mitad y repite con el doble; Hyperband encadena rondas de sucesive halving con
presupuestos distintos. En Optuna: `optuna.pruners.HyperbandPruner` y reportar
por epoch. Con XGBoost/LightGBM, `n_estimators` se resuelve con early stopping
dentro del trial, no como parámetro fijo del espacio.

### Warm-start

`study.enqueue_trial({...})` inyecta configuraciones conocidas como primeros
trials (baselines, configuraciones de papers). Optimizar un estudio ya
optimizado continúa desde su historial.

### Semillas y el error de reproducibilidad

Un trial con una semilla puntualiza una realización; el ruido de la semilla
puede superar la diferencia entre dos configuraciones. Buenas prácticas:

- Evaluar cada configuración con **media sobre $k$ semillas** (3–5) cuando el
  coste lo permite; con pocas semillas, el "mejor trial" puede ser un
  artefacto.
- Fijar `TPESampler(seed=...)` para que la *búsqueda* sea reproducible, no
  solo el entrenamiento.
- Guardar por trial: params, valor y semilla. Solo el mejor no basta para
  auditar.

**Trampa de reproducibilidad**: si al re-ejecutar con otra semilla cambia el
mejor trial, el tuning es ruido. Los candidatos finales se validan con semillas
nuevas antes de promocionar.

## Diseño del espacio de búsqueda

La escala del muestreo importa tanto como el rango:

| Tipo de parámetro | Escala | Optuna |
|-------------------|--------|--------|
| Magnitudes (lr, regularización) | log-uniform | `suggest_float(..., log=True)` |
| Tasas / proporciones | uniform | `suggest_float("dropout", 0.0, 0.6)` |
| Conteos (capas, árboles, vecinos) | integer | `suggest_int("max_depth", 2, 20)` |
| Elección discreta | categorical | `suggest_categorical("criterion", [...])` |

Con log-uniform, `suggest_float(..., log=True)` reparte muestras por décadas.
En $[10^{-4}, 10^{-1}]$ hay 3 órdenes de magnitud; muestreando en escala lineal,
el 90 % de los puntos cae en la última década y la región de lr pequeños casi
no se toca.

Distinguir dos formas de dependencia de la métrica:

- **Efecto monótono**: la métrica sube/baja suave con el parámetro (más
  `n_estimators` → mejor hasta saturar). Rango generoso + log-uniform bastan;
  el óptimo exacto no es crítico.
- **Pico sensible**: la métrica tiene un máximo estrecho (típico en `lr`,
  `min_child_weight`, `gamma`). Con rango ancho, casi todos los trials caen en
  valles y el resultado depende del muestreo. Log-uniform ayuda; si el pico es
  muy estrecho, restringe el rango tras una primera pasada (búsqueda en dos
  etapas).

## El sesgo de tunear sobre la misma validación

Cada vez que un conjunto se usa para elegir entre configuraciones, su
estimación deja de ser honesta para el modelo final: se selecciona sobre el
mismo dato que evalúa. Con muchos trials, se memoriza el ruido de la
validación.

La estimación honesta es **validación anidada** (nested CV): el CV interno
selecciona la configuración y el externo estima el modelo ganador. Ese número
es el que se compara contra el baseline y se reporta. Ver `validacion.md` para
la mecánica; aquí el punto es que el tuning nunca se reporta con la validación
que lo guió, y el test se toca una sola vez al final.

## Presupuesto y parada

- **Early stopping dentro del trial**: XGBoost/LightGBM con
  `early_stopping_rounds` y `trial.report` por epoch cortan trials malos antes
  de gastar el presupuesto; el pruner hace lo mismo entre trials.
- **Cuando deja de pagar**: los rendimientos son decrecientes. Si 20 trials
  nuevos no mejoran a los 20 previos, la búsqueda está saturada. Regla
  práctica: **tunea los 3 parámetros que más mueven la métrica y deja el
  resto por defecto**. Un 4º y 5º parámetro rara vez compensan el coste y el
  riesgo de sobreajuste al espacio.
- Fija un presupuesto *a priori* (tiempo o trials) y córtalo sin remordimiento:
  la mejor configuración de 50 trials rara vez es mucho peor que la de 500.

## Cuándo NO tunear

- **Datos muy pequeños**: la métrica de validación es ruido; el "mejor" óptimo
  no generaliza.
- **Baseline fuerte por defecto**: XGBoost o un HistGradientBoosting con
  defaults razonables suelen estar a 1–2 puntos del óptimo; el tuning fino no
  paga frente a mejorar features o datos.
- **Sin señal**: si el modelo no supera al baseline trivial (mayoría, media)
  tras un intento razonable, tunear no arregla un problema de features, de
  datos o de planteamiento. Busca señal antes de gastar presupuesto.

{% if use_optuna %}
## Optuna en este proyecto

Este proyecto se generó con `use_optuna` activo: `make tune` ejecuta
`uv run python -m tools.tune_model`. El módulo declara objetivos por tipo de
modelo (regresión, clasificación, KNN...) con `trial.suggest_*`, usa
`TPESampler(seed=42)` y `MedianPruner` por defecto, y reporta el mejor trial y
su configuración. Ajusta `n_trials` (default 30) desde la CLI:

```bash
make tune
uv run python -m tools.tune_model --n-trials 100
```

Reglas al tocar `tools/tune_model.py`: mantén fijas las semillas de sampler y
de entrenamiento para que un mismo trial sea comparable entre ejecuciones;
loguea el historial de cada estudio (SQLite, o mlflow si está activo); y
recuerda que la métrica del estudio es de validación, no de test — el modelo
final se entrena con la mejor configuración y se evalúa una sola vez en test.
{% endif %}

## Fuentes

- Bergstra, J. y Bengio, Y., *Random Search for Hyper-Parameter
  Optimization*, JMLR 2012. arXiv:1306.4711. https://arxiv.org/abs/1306.4711
- Snoek, J., Larochelle, H. y Adams, R. P., *Practical Bayesian Optimization
  of Machine Learning Algorithms*, NeurIPS 2012. arXiv:1206.2944.
  https://arxiv.org/abs/1206.2944
- Akiba, T. et al., *Optuna: A Next-generation Hyperparameter Optimization
  Framework*, KDD 2019. arXiv:1907.10902. https://arxiv.org/abs/1907.10902
- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. y Talwalkar, A.,
  *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*,
  JMLR 2018. arXiv:1603.06560. https://arxiv.org/abs/1603.06560
