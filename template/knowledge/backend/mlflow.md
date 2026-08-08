# MLflow: tracking, registry y serving

## Tracking: qué captura un run

Un run registra un experimento concreto. Cada componente de lo que define un
modelo tiene su sitio:

| Qué | Dónde | Para qué |
|-----|-------|----------|
| Parámetros | `log_param` / `log_params` | hipers, datos usados, versiones |
| Métricas | `log_metric` / `log_metrics` | accuracy, loss, RMSE por step o al final |
| Artefactos | `log_artifact` / `log_model` | pesos, scaler, encoders, figuras, JSONs |
| Tags | `set_tag` | metadata de búsqueda: dataset, autor, experimento, git sha |

Regla de oro de la comparabilidad: **un experimento distinto = un run
distinto, y cada run loguea todo lo que lo define**. Si dos runs no comparten
la misma definición de parámetros y métricas, la comparación es ruido.

```python
import mlflow

with mlflow.start_run(run_name="xgboost_lr01"):
    mlflow.log_params({"lr": 0.1, "max_depth": 6, "dataset": "v2"})
    mlflow.log_metrics({"acc": 0.91, "auc": 0.87})
    mlflow.log_artifact("models/artifacts/scaler.joblib")
```

- **Logging automático** (`mlflow.autolog()`): captura params, métricas y el
  modelo para frameworks conocidos (sklearn, lightgbm, torch) sin escribir
  nada a mano. Conveniente, pero captura lo que la librería decide: para
  comparar, loguear además las métricas de negocio explícitamente.
- **Estructura de experimentos**: uno por problema/proyecto, no uno por
  intento. La comparación dentro de un experimento es la que tiene sentido;
  cruzar experimentos mezcla problemas distintos.

**Fallo en producción**: run sin `log_params` → no se puede reproducir qué se
entrenó; dos experimentos para el mismo problema → la comparación se hace a
mano con post-its. El tracking no sirve si el nombre del run no dice nada.

## Gestión de runs y stores

- **Tags y lifecycle**: los runs se etiquetan (dataset, sha de código, tipo) y
  pasan a `deleted`/`active`; un run terminado es inmutable en la práctica.
- **Comparar runs**: la UI de MLflow compara dos runs lado a lado o como tabla;
  la clave es que las métricas se llamen igual (misma definición, misma
  dirección: mayor=mejor o menor=mejor explícito).
- **Backend store vs artifact store**: el backend (SQL) guarda metadatos,
  params y métricas; el artifact store (S3, disco, MLflow server) guarda los
  binarios (modelos, scaler, figuras). Un backup de solo el backend deja los
  artefactos sin recuperar. Ambos se configuran por separado y ambos se deben
  respaldar.

**Fallo en producción**: el artifact store en disco local del servidor de
MLflow → los artefactos desaparecen al recrear el servidor; o el backend
SQLite en un solo nodo que se pierde en el redeploy. Para equipo, backend
Postgres + artifact store en objeto o red compartida.

## Model Registry: la fuente de verdad

El registry eleva un run (o varios) a modelo versionado. Es el contrato entre
experimentación y serving:

- **Model versions**: cada registro crea una versión inmutable (v1, v2...)
  ligada a su run y su artefacto.
- **Stages**: `Staging` (validación/ensayo), `Production` (sirviendo),
  `Archived` (retirado). Las transiciones quedan auditablemente logueadas y se
  controlan con permisos.
- El servicio que sirve debe **leer la versión en `Production`**, no un fichero
  suelto: el registry es la única fuente de verdad de qué modelo es el activo.

```python
client = mlflow.tracking.MlflowClient()
version = client.transition_model_version_stage(
    name="modelo_credito", version=2, stage="Production"
)
```

**Fallo en producción**: la API carga `models/xgb.joblib` que alguien
sobrescribió por ssh, mientras el registry dice otra cosa. Cuando el registry
es la fuente de verdad, el deploy se define como "mover el stage a
Production", no como "copiar un fichero".

## Flavors: serialización por framework

Cada framework se serializa en su flavor:

- **sklearn**: `mlflow.sklearn.log_model` guarda el pipeline completo
  (estimador + preprocesado si está dentro) y la versión de sklearn que lo
  serializó.
- **torch**: `mlflow.pytorch.log_model` guarda `model.state_dict()` (o el
  módulo) + el entorno; al cargar se necesita la arquitectura.
- **pyfunc**: el flavor portátil. `log_model(python_model=...)` o
  `load_model(..., model_type="pyfunc")` expone `predict()`. Cualquier
  framework se reduce a pyfunc, que es lo que consume un servicio genérico.

Cargar un modelo con el entorno correcto: MLflow guarda `conda.yaml`/`uv`
y puede crear un entorno aislado (`env_manager="conda"` o `virtualenv`).
En producción, servir dentro del entorno ya construido en la imagen Docker;
no re-resolver el entorno en cada arranque.

**Fallo en producción**: cargar un modelo con un flavor distinto al que lo
logueó (sklearn guardado, torch cargado) o con otra versión de la librería →
excepción o salida cambiada en silencio. El flavor es un contrato: respetarlo.

## Reproducibilidad

Un run es reproducible si loguea las cuatro coordenadas:

1. **Entorno**: `conda.yaml` / `uv.lock` (o `mlflow.log_artifact("uv.lock")`).
2. **Código**: git sha (`mlflow.set_tag("git_sha", subprocess...)`).
3. **Datos**: hash o versión del dataset (`md5` de `data/processed/X_train.csv`).
4. **Parámetros**: todo lo que cambia la salida, en `log_params`.

Con esas cuatro, "reproduce el run 47" es un comando, no una pesquisa.

**Fallo en producción**: un run con la misma semilla y mismos params que otro
pero con otro dataset produce otro modelo; sin la coordenada de datos, el
misterio de "por qué este modelo es distinto" se investiga días.

## Serving desde el registry e integración con FastAPI

Servir desde el registry desacopla experimentación y deployment:

```python
import mlflow

model = mlflow.pyfunc.load_model(
    model_uri="models:/modelo_credito/Production"
)

@app.post("/predict")
def predict(req: PredictRequest):
    return {"prediction": float(model.predict(req.features)[0])}
```

- `models:/<name>/Production` resuelve en el arranque a la versión activa; un
  cambio de stage no requiere cambiar código.
- Cargar el modelo una vez en `lifespan` (igual que en `api.md`): el coste de
  descarga y deserialización no se repite por petición.
- El pyfunc sirve el preprocesado si el pipeline se logueó entero; si no, el
  preprocesado vive fuera y hay que mantenerlo en sincronía con el training.

**Fallo en producción**: cargar el modelo en cada petición (descarga + load
por request) o servir un modelo del registry sin probarlo contra el pipeline
de features real; el registry no garantiza que el input del servicio sea el
input del modelo.

## Fuentes

- Documentación de MLflow: tracking, logging automático, Model Registry,
  flavors y pyfunc.
- MLflow Model Registry docs: stages y transiciones.
- Documentación de MLflow sobre deployments: carga desde `models:/`.
