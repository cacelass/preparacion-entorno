# Changelog

Todos los cambios relevantes de esta plantilla se documentan aquí.  
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).

---

## [1.9.0] — 2026-07-06

### Sistema de agentes y release automático

- Se añadió y documentó la carpeta `agents/` como capa de agentes especializados para git, documentación, CI/CD, pruebas, dependencias, API, secretos, notebooks e instalación de agentes externos.
- `GitAgent` ahora puede coordinar `update_changelog`, `bump_version`, `commit_with_changelog` y `tag_release` en un flujo único de release.
- `BaseAgent` y `Orchestrator` mejoraron el ruteo determinista: además de escoger agente, ahora también resuelven la acción con alias y validación de argumentos.
- Se documentó el nuevo workspace por agente y la colaboración entre agentes para evitar ciclos de import y duplicación de lógica.

---

## [1.8.2] — 2026-05-22

### Corrección de bugs — auditoría completa

Auditoría automática con Jinja2 StrictUndefined sobre **17 combinaciones × 59 archivos**
(1 003 checks de renderizado + AST) y suite semántica con 80+ aserciones de contenido.
Resultado final: **0 bugs** en renderizado, AST y semántica.

---

#### `main.py` — 3 bugs críticos (`redes_neuronales + regresion`)

- **`output_dim` incorrecto**: `len(y_train.unique())` devuelve el número de clases únicos,
  que en regresión es el número de valores distintos del target continuo, no `1`.
  Corregido con bloque `{% if task_type == 'regresion' %}` que asigna `output_dim = 1`.
- **`evaluate_models` llamada con `num_classes=output_dim`** en regresión: la firma de
  `evaluate_models` para regresión no acepta ese parámetro y lanzaba `TypeError`.
  Corregido condicionando la llamada con `{% if task_type == 'regresion' %}`.
- **`best["Accuracy"]` y `best["F1"]` en el print final** de regresión: esas columnas no
  existen en el DataFrame de regresión (que tiene `RMSE`, `MAE`, `R2`).
  Corregido mostrando `RMSE`/`MAE`/`R²` para regresión y `Accuracy`/`F1` para clasificación.

---

#### `tuning/tune_model.py` — 5 bugs críticos (NN)

- **`_OBJECTIVES = {"{{ nn_model }}": None}`**: el sentinel `None` hacía que `tune_models()`
  saltara el objetivo real con `if objective_fn is None: continue`. Nunca se ejecutaba
  ningún trial para redes neuronales. Corregido a `_objective_nn`.
- **Optimizador `AdamW` hardcodeado**: ignoraba `optimizer_type`. Para SGD, RMSProp y Adagrad
  se generaba código con `AdamW` en lugar del optimizador elegido. Corregido con el mismo
  bloque `{% if optimizer_type %}` que usa `train_model.py`.
- **Loss `CrossEntropyLoss` hardcodeada**: ignoraba `nn_loss_fn`. Para MSELoss, L1Loss y
  BCEWithLogitsLoss se generaba `CrossEntropyLoss`. Corregido con bloque `{% if nn_loss_fn %}`.
- **`dtype=torch.long` para targets en regresión**: `MSELoss` y `L1Loss` requieren
  `torch.float32`. Los `TensorDataset` y `val_y` creaban tensores `long` en ambos casos,
  causando `RuntimeError` en el primer backward. Corregido con bloque `{% if task_type %}`.
- **`criterion(model(Xb), yb)` sin `.squeeze()`** en regresión: la salida del modelo tiene
  shape `(batch, 1)` y el target `(batch,)`. Sin `.squeeze()`, MSELoss/L1Loss lanzan
  `ValueError: shape mismatch`. Corregido a `criterion(model(Xb).squeeze(), yb)`.

---

#### `{{ project_slug }}/features/build_features.py` — 2 bugs críticos

- **`process_input()` ausente en bloque `no_supervisado`**: la API, el chat y `try_model()`
  llaman a `process_input()` para cualquier `ml_type`. Al no existir, el import fallaba con
  `ImportError`. Añadida implementación completa con `scaler.joblib` y `encoders.joblib`.
- **`process_input()` ausente en bloque `hibrido`**: mismo problema. Añadida implementación
  que además detecta y aplica automáticamente la transformación dimensional guardada en
  `artifacts/` (PCA, UMAP, KMeans-features o IsolationForest).

---

#### `{{ project_slug }}/models/train_model.py` — 2 bugs

- **`joblib` e `ARTIFACTS_DIR` no importados** en el bloque NN: el bloque `supervisado`
  los importaba pero el bloque `redes_neuronales` no. Cualquier llamada a `joblib.dump`
  lanzaba `NameError`. Añadidos al bloque de imports NN.
- **`output_dim.joblib` no se guardaba tras entrenar**: la API infería `output_dim=2` por
  defecto al no encontrar el artefacto. Roto para regresión (`output_dim` debería ser `1`)
  y para clasificación multiclase (3+ clases). Añadida llamada `joblib.dump(output_dim, ...)`
  al final de `train_models()`.

---

#### `tests/conftest.py` — 3 bugs

- **`tuning.tune_model` no parcheado** con `monkeypatch` cuando `use_optuna=True`:
  los tests de tuning usaban rutas reales del sistema de ficheros.
- **`monitoring.monitor` no parcheado** cuando `use_monitoring=True`: ídem.
- **`api.main` no parcheado** cuando `use_api=True`: ídem.
  Los tres módulos añadidos a `candidate_modules` condicionados con `{% if %}`.

---

#### `tests/test_predict_model.py` — 5 bugs (NN)

- **`"MLP"` hardcodeado** en todos los asserts: fallaba para CNN1D, LSTM, GRU, Transformer.
  Corregido a `MODEL_NAME` (importado de `train_model`).
- **Sin tests de regresión**: solo había tests de clasificación. Para `task_type=regresion`
  el bloque NN quedaba con 0 tests de evaluación. Añadidos 5 tests de regresión
  (`RMSE`, `MAE`, `R2`, scatter PNG, predicciones float).
- **`train_models()` sin `val_split=`**: la función requiere el parámetro; sin él usaba
  el default de `0.1` que podía reducir los datos de entrenamiento por debajo del
  `batch_size`, causando un crash en el `DataLoader`.
- **Columna `"Accuracy"` en tests de regresión**: `evaluate_models` regresión devuelve
  `RMSE`/`MAE`/`R2`. El assert `"Accuracy" in df_res.columns` siempre fallaba.
- **`num_classes=` en `evaluate_models()` de regresión**: la firma de regresión no acepta
  ese argumento. Eliminado del bloque `{% else %}` (regresión).

---

#### `tests/test_train_model.py` — 3 correcciones (NN)

- **`train_models()` sin `val_split=0.2`** en los 3 calls NN: añadido en todos.

---

#### `tests/test_tuning.py` — 2 bugs (NN regresión)

- **`output_dim=int(y_train.nunique())`** en los tests NN de regresión: devuelve el número
  de valores únicos del target continuo, no `1`. Corregido a `output_dim=1` condicionado
  con `{% if task_type == 'regresion' %}`.

---

#### `pyproject.toml` — 1 bug

- **Extra `monitoring` sin `evidently`**: al seleccionar `use_monitoring=True`, el entorno
  se generaba sin la dependencia principal. Añadido `evidently` y `scipy` al extra.

---

#### `Makefile` — 1 bug

- **Target `lock:` ausente**: no había forma de regenerar `uv.lock` tras cambiar
  dependencias en `pyproject.toml`. Añadido `lock: uv lock` y declarado en `.PHONY`.

---

#### `README.md` — reescrito

- Badge de versión actualizado a `1.8.2`
- Tabla de variables completa con `optimizer_type`, `nn_loss_fn`, `cluster_model`,
  `use_catboost`, `use_docker`, `use_shap`
- Tabla de módulos opcionales con flag → descripción → make target
- Sección "Notas por tipo de ML" con detalles de `output_dim.joblib`, `process_input()`,
  y comportamiento del optimizador/pérdida en NN
- Makefile — referencia completa con todos los targets incluyendo `make lock`
- Estructura de directorios actualizada con `models/artifacts/`

---

## [1.8.1] — 2026-05-20

### Añadido

#### Notebooks educativos adaptativos (condicionados por `ml_type == 'redes_neuronales'`)

- **`0-0-DescargaDatos.ipynb`**: análisis de rangos de features pre-normalización y
  checklist automático de preparación NN (muestras, nulos, balance, varianza).
- **`0-1-ProcesamientoDatos.ipynb`**: forma exacta del tensor de entrada por arquitectura
  y verificación de dtypes/balance de clases.
- **`0-2-Ejecucion.ipynb`**: demo interactiva de autograd, curva de convergencia del
  optimizador elegido, comparativa de funciones de pérdida y evaluación con TorchMetrics.

#### Nuevas opciones de configuración NN

- **`optimizer_type`**: `AdamW` (default) · `Adam` · `SGD` · `RMSProp` · `Adagrad`
- **`nn_loss_fn`**: `Auto` · `CrossEntropyLoss` · `MSELoss` · `L1Loss` · `BCEWithLogitsLoss`

#### TorchMetrics integrado en `train_model.py`

- Clasificación: `MulticlassAccuracy`, `F1Score`, `Precision`, `Recall` (macro)
- Regresión: `MAE`, `RMSE`, `R²`
- Métricas train/val logueadas en TensorBoard (`Train/*`, `Val/*`)
- Degradación silenciosa si `torchmetrics` no está instalado

#### `predict_model.py` NN regresión

- `evaluate_models` con rama completa para regresión: RMSE, MAE, MAPE, R²
- `_plot_regression_scatter` — scatter predicho vs real con línea y=x
- `_plot_residuals` — histograma de residuos + Q-Q plot
- TorchMetrics como fuente primaria, sklearn como fallback (`_HAS_TM`)

#### Extra `monitoring` en `pyproject.toml`

- Añadidos `evidently` y `scipy` cuando `use_monitoring=True`
- Verificación en `_tasks` con `import evidently`

#### Otros

- `make lock` — target `uv lock` en Makefile
- `api/__init__.py` — el directorio `api/` ya es paquete Python importable
- `docs/source/conf.py` — `sys.path` corregido de `../../src/` a `../..`
- `make test` — añadido `--cov={{ project_slug }}` con report `term-missing` y HTML