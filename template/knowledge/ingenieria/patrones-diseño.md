# Patrones de diseño para código ML

Referencia para el lider sobre cuándo un patrón resuelve un problema real de
código ML y cuándo es sobreabstracción. Cada sección: principio, práctica
concreta y el fallo típico en un proyecto DS.

## Por qué importan los patrones en ML (y la regla del proyecto)

**Principio.** Los patrones no son decoración: son nombres compartidos para
decisiones de estructura que reaparecen una y otra vez. En código ML el coste
de la sobreabstracción es doble —indirección que nadie sigue al depurar un
pipeline— y el beneficio aparece solo cuando hay variación real: varios
modelos, varios backends de datos, varios esquemas de validación.

**Regla del proyecto.** No abstracción anticipada. Solo se extrae una
interfaz cuando hay ≥ 2 implementaciones reales en el código, no "por si
acaso". Una interfaz con una sola implementación no abstrae nada: añade un
salto que hay que leer y no permite cambiar nada que no se cambiaría igual.

**Cómo falla.** Un proyecto que empieza con `BaseModel`, `ModelFactory` y
`ModelRegistry` para un único `XGBClassifier`: cada commit de features exige
tocar cinco capas y el flujo real queda sepultado bajo indirección.

## Patrones creacionales

### Factory — registro de modelos por nombre

**Principio.** Elegir una familia de modelos por nombre en configuración, no
por ramas `if/elif` en el script de entrenamiento.

**Práctica.** Un registro plano de nombre → clase y una única función de
creación:

```python
MODEL_REGISTRY: dict[str, type] = {
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
    "linear": LogisticRegression,
}

def make_model(name: str, **params) -> BaseEstimator:
    try:
        return MODEL_REGISTRY[name](**params)
    except KeyError:
        raise ValueError(f"modelo desconocido: {name}") from None
```

El nombre del modelo vive en config y viaja con el run; el factory es el único
sitio que sabe instanciar.

**Cómo falla.** Cadenas `if name == "xgboost": ... elif ...` replicadas en
train y en serve: el segundo modelo obliga a tocar dos lugares y la lista de
modelos soportados queda dispersa.

### Builder — objetos de configuración complejos

**Principio.** Cuando un objeto de configuración tiene muchos campos con
dependencias entre ellos (defaults que cambian según el split, el modelo o la
tarea), un constructor con 15 kwargs obliga al llamador a saberlo todo y
esconde combinaciones inválidas.

**Práctica.** El builder acumula estado con métodos que devuelven `self` y
valida al construir:

```python
class PipelineConfigBuilder:
    def __init__(self):
        self._split = "nested"
        self._target = "y"

    def with_split(self, split: str):
        self._split = split
        return self

    def with_target(self, target: str):
        self._target = target
        return self

    def build(self) -> PipelineConfig:
        if self._split == "timeseries" and self._target is None:
            raise ValueError("series temporales exigen target")
        return PipelineConfig(split=self._split, target=self._target)
```

**Cómo falla.** Un `PipelineConfig(...)` con 12 kwargs posicionales: los
experimentos dejan de registrar qué opciones activaron y dos configs "iguales"
en el código producen runs distintos porque un default mutó.

### Singleton — el modelo cargado en memoria

**Principio.** El modelo entrenado se carga una vez por proceso y se comparte;
recargarlo por petición o por re-fit es el error clásico de serving.

**Práctica.** En Python el singleton no es una clase con `__new__` guardado:
es un estado de proceso, normalmente colgado del framework.
{% if use_api %}
```python
from fastapi import FastAPI, Request, Depends

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model("models/current.joblib")
    yield
    app.state.model = None

app = FastAPI(lifespan=lifespan)

def get_model(request: Request):
    return request.app.state.model

@app.post("/predict")
def predict(req: PredictRequest, model=Depends(get_model)):
    return model.predict(req.features)
```
{% else %}
```python
_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model("models/current.joblib")
    return _MODEL
```
{% endif %}

El acceso pasa por un único punto (`app.state` o un módulo), no por una
variable global escrita desde cualquier sitio.

**Cómo falla.** Cargar el modelo dentro de cada handler: el primer pico de
tráfico se come N cargas de disco y el p99 explota. O dos workers cargan dos
copias y la RAM se duplica sin aviso.

### Prototype — copiar configs de modelos

**Principio.** Los experimentos se derivan de configs base; el error es mutar
la config compartida "para este run".

**Práctica.** Copia profunda + un campo distinto por experimento:

```python
import copy
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    name: str
    params: dict = field(default_factory=dict)
    seed: int = 42

base = ModelConfig(name="xgboost", params={"max_depth": 6})
exp1 = copy.deepcopy(base)
exp1.params["max_depth"] = 10   # base queda intacta
```

**Cómo falla.** Dos experimentos que comparten el mismo dict y lo mutan in
place: el segundo run hereda los parámetros del primero y el leaderboard
registra configs que nadie puede reconstruir.

## Patrones estructurales

### Adapter — envolver librerías

**Principio.** El código de negocio no debe acoplarse a la API de una
librería concreta: un adapter aísla el cambio de versión o de librería a un
solo fichero.

**Práctica.** Una clase fina que traduce tu vocabulario al de la librería:

```python
class CatBoostScorer:
    def __init__(self, model_path: str):
        self._model = CatBoostClassifier().load_model(model_path)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)
```

**Cómo falla.** `catboost` llamada desde 12 módulos: la actualización de la
API o el cambio a `lightgbm` toca 12 sitios y rompe en producción antes de
terminar la migración. El adapter se justifica desde la segunda versión de la
librería, no desde la primera.

### Facade — CLI fina sobre un pipeline

**Principio.** Un pipeline de varias etapas se expone con una interfaz
pequeña y estable (CLI o funciones top-level), no dejando que el llamador
conozca las etapas.

**Práctica.**

```text
make train    # ingesta → features → fit → evaluate, en orden
make predict  # carga modelo + transforma + predice
```

**Cómo falla.** Un Makefile que crece a 30 targets porque cada experimento
añade uno: la fachada se convierte en el dios del proyecto. La fachada es
estable; la variación de experimentos vive en config, no en targets nuevos.

### Repository — acceso a datos tras una interfaz

**Principio.** El código de entrenamiento no debe saber de dónde salen los
datos (local, S3, DuckDB, una versión concreta): detrás de una interfaz se
puede cambiar el backend o versionar sin tocar al consumidor.

**Práctica.**

```python
class DatasetRepo(Protocol):
    def load(self, version: str | None = None) -> pd.DataFrame: ...

class LocalParquetRepo:
    def load(self, version: str | None = None):
        path = f"data/final/dataset_{version or 'latest'}.parquet"
        return pd.read_parquet(path)
```

**Cómo falla.** Con un único backend, el repo añade un salto sin variación
que cambiar: la regla del proyecto manda leer el parquet directo hasta que
exista un segundo backend o una necesidad real de versionado.

### Inyección de dependencias vs globals

**Principio.** Una función recibe sus dependencias (config, repo, modelo)
como parámetros; no las lee de una variable global que otro módulo pudo mutar.

**Práctica.**

```python
def evaluate(model, X, y, cfg) -> dict:   # DI: testeable
    return {"auc": roc_auc_score(y, model.predict_proba(X)[:, 1])}

def evaluate_bad():                        # globals: frágil
    return {"auc": roc_auc_score(Y, MODEL.predict_proba(X)[:, 1])}
```

**Cómo falla.** Un test que muta `MODEL` o `DATASET` global para simular un
caso envenena los tests siguientes y los tests en paralelo se pisan. Con DI
el test pasa un fake explícito y el acoplamiento queda a la vista.

## Patrones de comportamiento

### Strategy — modelos y validación intercambiables

**Principio.** Cuando la variación real está en "qué algoritmo" o "qué
esquema de validación", se selecciona por nombre, no con ramas.

**Práctica.**

```python
CV_SCHEMES = {
    "kfold": lambda seed: KFold(n_splits=5, shuffle=True, random_state=seed),
    "timeseries": lambda seed: TimeSeriesSplit(n_splits=5),
}

def cross_validate(make_model, X, y, scheme, seed=42):
    for train_idx, val_idx in CV_SCHEMES[scheme](seed).split(X):
        ...
```

**Cómo falla.** Un `cross_validate` con un booleano `use_timeseries` que
ramifica dentro del bucle: la tercera variante (grupos, estratificación)
obliga a reescribir el cuerpo y el código acumula banderas que se combinan
mal.

### Template Method — andamiaje fijo con hooks

**Principio.** Cuando el esqueleto entrenar→evaluar→reportar es idéntico entre
experimentos y solo cambian algunos pasos, el esqueleto vive en un método y
los pasos variables son hooks.

**Práctica.**

```python
class Experiment:
    def run(self, X, y):
        X_train, X_val = self.split(X, y)    # hook
        model = self.build_model()           # hook
        model.fit(X_train, y.loc[X_train.index])
        return self.report(model, X_val, y.loc[X_val.index])
```

**Cómo falla.** El hook equivocado: `split`, `build_model` y `report` se
declaran abstractos, pero 4 de 5 subclases sobreescriben también `run` porque
el esqueleto no encaja — el Template Method queda como burocracia. Si cada
experimento quiere su propio flujo, no hay plantilla.

### Observer — vigilancia de drift y monitorización

**Principio.** Cuando la acción ante un evento (drift, datos raros) debe
propagarse a varios consumidores sin acoplarlos al productor, un registro de
observadores permite añadir vigías sin tocar el pipeline.

**Práctica.** Un `DriftBus` simple: el pipeline publica métricas y los
observadores (alerta, log, reentrenador) se suscriben.

**Cómo falla.** Implementar pub/sub con un solo consumidor ("por si mañana").
El patrón se paga solo con dos observadores reales; antes, un callback
directo en el punto de publicación es más legible.

### Command — acciones de CLI como objetos

**Principio.** Cada acción de una CLI es un objeto con `run(args)` y, cuando
aplica, un `--dry-run`/`undo`. Permite listar, validar y auditar acciones sin
un `main()` gigante.

**Práctica.** El arnés del proyecto lo hace: `git commit_feature`,
`harness finish`, `rag search` son comandos con contrato y puerta de permisos
(`agents/contracts.py`).

**Cómo falla.** Un `argparse main()` de 300 líneas con `if args.action == ...`
anidados: añadir una acción obliga a releer todo y las acciones destructivas
no se distinguen de las inocuas — de ahí la puerta `destructive` del arnés.

### Iterator/Generator — datos en streaming

**Principio.** Los datos no caben o no conviene cargarlos enteros: un
generador produce lotes bajo demanda y mantiene el pipeline perezoso.

**Práctica.**

```python
def batch_iter(df: pd.DataFrame, size: int):
    for start in range(0, len(df), size):
        yield df.iloc[start:start + size]
```

**Cómo falla.** `pd.read_csv` del fichero entero "porque el proyecto es
pequeño" hasta que el dataset crece 100× y el entrenamiento muere de RAM; o
generadores con estado oculto (posición global) que no son re-ejecutables.

## Patrones específicos de ML

### Train/serve split

**Principio.** El código que produce features en entrenamiento y el que las
produce en inferencia debe ser el mismo. Si se duplica, divergen en silencio
(skew de train/serve).

**Práctica.** `{{ project_slug }}/features/build_features.py` es la única
fuente; el endpoint de predicción la importa, nunca la reimplementa.

**Cómo falla.** El notebook define `def feats(df)` y la API lo copia con un
bug de redondeo: el modelo en producción recibe features que nunca vio en
entrenamiento y las métricas caen sin que el código "falle".

### Checkpoint

**Principio.** Un entrenamiento largo debe poder reanudarse: se persiste no
solo el modelo final, sino el estado (pesos, optimizador, época, config,
métricas parciales) en intervalos.

**Práctica.**

```python
trainer.save_checkpoint({
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "config": cfg,
    "best_metric": best,
    "seed": SEED,
}, f"models/checkpoints/epoch_{epoch:03d}.pt")
```

**Cómo falla.** Guardar solo los pesos finales: un crash en la época 40 de 50
reinicia desde cero y 40 horas de GPU se pierden. Guardar sin config convierte
el checkpoint en inútil para reanudar (ver "config como dato").

### Config como dato

**Principio.** Un run queda totalmente descrito por su config: hiperparámetros,
features, split, seed, versión de datos y de código. El config ES el
experimento; el log guarda el config, no la narrativa.

**Práctica.** Un `dataclass`/dict serializable, guardado junto a las métricas
(`models/runs/<id>/config.json` o en mlflow).

**Cómo falla.** El experimento descrito como "cambié max_depth y añadí una
feature" en un chat: nadie puede reconstruir el número del leaderboard y el
siguiente run "mejora" comparando contra configs fantasma.

### Registry

**Principio.** Los modelos se versionan y promocionan con nombre y etiqueta
(staging/production), nunca como `model_v2_final_really.pt` sobrescrito.

**Práctica.** Registro por nombre+versión con metadatos (métricas, fecha, git
sha); el serve apunta a una versión promocionada, no a "el archivo más nuevo".

**Cómo falla.** Deployar `model_final_v3.pt` mientras `model_final_v3(final).pt`
sigue en disco: dos artefactos con el mismo nombre lógico y el rollback es
adivinar cuál era el bueno.

### Reproducibilidad

**Principio.** Por run se registran seed + versión de datos + versión de
código. Sin los tres, "reproducible" no es verificable.

**Práctica.**

```python
def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

run_meta = {"seed": 42, "data_version": "v3_2026-08", "code": git_head()}
```

**Cómo falla.** Un run cuyo resultado no se puede reproducir porque el dataset
fue sobrescrito y no hay hash ni versión anotada: la métrica del leaderboard
queda huérfana.

### Anti-patrones

| Anti-patrón | Síntoma | Contramedida |
|-------------|---------|--------------|
| Glue code | Scripts que solo encadenan llamadas | Encapsular cada etapa (ver `deuda-tecnica.md`) |
| Pipeline jungle | DAG de transformaciones disperso | Etapas deterministas con fronteras explícitas |
| God object | Un `config.py` o `utils.py` que lo sabe todo | Una responsabilidad por módulo |
| Config drift | El código dice una cosa y la config otra | Config como dato + validación en arranque |

## Cuándo NO usar un patrón

**Principio.** Un patrón se usa cuando la variación es real y actual, no
posible y futura. La señal de sobreabstracción es medible:

- Una interfaz con una única implementación.
- Clases abstractas con un solo hijo.
- Un factory que solo crea un tipo.
- Flags de configuración que ningún código lee.
- "Lo necesitaremos cuando lleguemos a..." — YAGNI: se extrae cuando el
  segundo caso llega, y entonces el refactor es barato si el código era simple.

**Práctica.** Antes de añadir una capa, escribe la segunda implementación en
un lado del folio. Si no existe, el patrón no resuelve nada hoy: es deuda
anticipada que se paga en indirección cada vez que alguien lee el flujo.

**Cómo falla.** Un proyecto que vive para su arquitectura: `BaseModel`,
`AbstractValidator`, `DataAccessInterface` para un modelo, un validador y un
CSV. Los tests tardan el doble, los commits tocan cinco capas y la regla
"código mínimo que resuelve el problema" (AGENTS.md) se incumple por diseño.

## Fuentes

- Gamma, Helm, Johnson, Vlissides, "Design Patterns" (GoF, 1994).
- Martin Fowler, "Refactoring" y el catálogo de refactorings.
- Fowler, "Patterns of Enterprise Application Architecture" (Repository,
  Registry).
- Documentación de FastAPI (lifespan, `app.state`, `Depends`) para el
  Singleton en serving.
- Documentación de `dataclasses` y `copy` (stdlib de Python).
- Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (NIPS
  2015) — glue code y pipeline jungles.
- Las reglas del proyecto: `AGENTS.md` (simplicidad primero, YAGNI, no
  abstracción anticipada) e `ingenieria/estructuras-codigo.md`.
