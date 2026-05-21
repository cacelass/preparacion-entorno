# Changelog

Todos los cambios relevantes de esta plantilla se documentan aquí.  
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).

---

## [1.8.1] — 2026-05-20

### Añadido / Mejorado 

Los notebooks de la plantilla ahora integran contenido educativo **directamente en el flujo de trabajo**
del proyecto, condicionado con Jinja a `ml_type == 'redes_neuronales'` y a la arquitectura (`nn_model`),
el optimizador (`optimizer_type`), la función de pérdida (`nn_loss_fn`) y el tipo de tarea (`task_type`).

#### `0-0-DescargaDatos.ipynb`
- **§3b — Rangos de features** (nuevo): barras de rango/media/std pre-normalización; detecta
  features con varianza ≈ 0 y rangos extremos (>100× mediana) antes del scaler.
- **§12 — Checklist NN** (nuevo al final): evalúa automáticamente muestras, nulos, infinitos,
  encoding del target, balance de clases / outliers extremos en regresión, varianza de features,
  y aviso temporal para LSTM/GRU/Transformer. Muestra semáforo ✓/! /✗ con consejo de acción.

#### `0-1-ProcesamientoDatos.ipynb`
- **§3b — Forma del tensor de entrada** (nuevo): muestra la forma exacta que espera cada
  arquitectura (`MLP` → 2D, `CNN1D` → unsqueeze canal, `LSTM`/`GRU` → unsqueeze seq_len,
  `Transformer` → transponer). Detecta NaNs/infs en `X_train` con mensaje de corrección.
- **§3c — Dtypes y balance** (nuevo): verifica que las clases son enteros desde 0
  (`CrossEntropyLoss` requiere `torch.long`), detecta desbalance con ratio, muestra estadísticas
  del target en regresión y recomienda escalar si `std > 10`.

#### `0-2-Ejecucion.ipynb`
- **§3b — Autograd** (nuevo, solo NN): demo interactiva del grafo de cómputo con tensor simple;
  explica el ciclo `forward → backward → step → zero_grad`; muestra el `grad_fn` del modelo real.
- **§3c — Optimizador** (nuevo, solo NN): curva de convergencia del `optimizer_type` elegido
  en un problema juguete de regresión 1D; parámetros aprendidos vs esperados.
- **§3d — Función de pérdida** (nuevo, solo NN): tabla comparativa MSE/L1/Huber en regresión;
  demo CrossEntropy/BCE con logits en clasificación; visualización de probabilidades post-softmax.
  Condicionado a `task_type` y al `nn_loss_fn` elegido.
- **§3e — TorchMetrics** (nuevo, solo NN): calcula Accuracy/F1/Precision/Recall/AUROC
  (clasificación) o MAE/RMSE/R² (regresión) sobre el conjunto de test completo usando
  `MetricCollection`; AUROC solo disponible para `MLP` y `CNN1D`.


### Añadido
- **`optimizer_type`** — nueva pregunta en `copier.yml` (solo `redes_neuronales`).
  Permite elegir el optimizador PyTorch: `AdamW` (default) | `Adam` | `SGD` | `RMSProp` | `Adagrad`.
  El `train_model.py` y `load_checkpoint` se generan con el optimizador elegido.
- **`nn_loss_fn`** — nueva pregunta en `copier.yml` (solo `redes_neuronales`).
  Permite elegir la función de pérdida: `Auto` (CrossEntropy/MSE según `task_type`) |
  `CrossEntropyLoss` | `MSELoss` | `L1Loss` | `BCEWithLogitsLoss`.
- **TorchMetrics** integrado en el bucle de entrenamiento (`redes_neuronales`).
  - Clasificación: `MulticlassAccuracy`, `F1Score`, `Precision`, `Recall` (macro).
  - Regresión: `MAE`, `RMSE`, `R²`.
  - Métricas train/val logueadas en TensorBoard (escalares `Train/*` y `Val/*`) y
    mostradas en la consola cada 10 épocas.
  - Degradación silenciosa si `torchmetrics` no está disponible.
- **Notebooks PyTorch educativos** (solo `ml_type = redes_neuronales`):
  - `1-0-Autograd.ipynb` — mecanismo autograd, DAG de cómputo, gradientes.
  - `1-1-Optimizadores.ipynb` — SGD, Adagrad, Adam, RMSProp, Adadelta con CIFAR-10.
  - `1-2-FuncionesDePerdida.ipynb` — MSE, L1, MBE, SVM Loss, CrossEntropy.
  - `1-3-Scores.ipynb` — `torchmetrics`: F1, Accuracy, Precision, Recall, AUROC,
    `MetricCollection`, HammingDistance.
  - Excluidos automáticamente en otros `ml_type` vía `_exclude`.
- **`api/__init__.py`** añadido (BUG-031): el directorio `api/` ahora es un paquete Python válido.
- **`make lock`** (BUG-032): nuevo target que ejecuta `uv lock`.
- **`make docs`** ahora depende de `setup` (BUG-014): evita que sphinx-autodoc falle sin entorno.
- **Extra `monitoring`** en `pyproject.toml` (BUG-029): añade `evidently` y `scipy` cuando
  `use_monitoring = yes`; incluido en `uv sync` de `_tasks` y verificado con `import evidently`.

### Corregido
- **BUG-002** — `make test` ahora pasa `--cov={{ project_slug }} --cov-report=term-missing
  --cov-report=html:htmlcov` para cobertura real del paquete.
- **BUG-022** — `docs/source/conf.py`: `sys.path` apuntaba a `../../src/` inexistente;
  corregido a `../..` (layout plano del proyecto).
- **BUG-015** — `.gitignore`: `__pycache__/` solo cubría la raíz; cambiado a
  `**/__pycache__/` para recursividad completa. Añadido `*.pyo` explícito.
- **BUG-016** — Notebooks sin indicación de entorno: todos los notebooks
  (`.ipynb`) reciben ahora una celda markdown de advertencia al inicio con el
  comando `make lab` / `make notebook` para garantizar que se lanzan con `uv run`.
- **BUG-024** — Celdas de carga de datos sin guardia: `load_data(DATA_FILE)` en
  `0-0-DescargaDatos` y `0-1-ProcesamientoDatos`, y `pd.read_csv(X_train.csv)` en
  `0-2-Ejecucion`, ahora están envueltas en `try/except FileNotFoundError` con
  mensaje de ayuda que indica el comando `make data && make features` o el paso
  previo requerido.
- **BUG-023** — Combinaciones de opciones inviables: añadido `_message_before_copy`
  en `copier.yml` que advierte al usuario antes de generar. Añadido texto de ayuda
  detallado en `use_api` explicando el requisito de `make train` previo.
- **BUG-022** — `docs/source/conf.py`: `sys.path` apuntaba a `../../src/` inexistente.
  Corregido a `../..` (ya en v1.8.1, confirmado en auditoría).


---


## [1.8.0] — 2026-05-20
### Añadido
- `use_catboost` — integración completa de CatBoost (supervisado + híbrido).
  - `copier.yml`: variable `use_catboost: bool` (default false) + choice `CatBoost` en `model_type`.
  - `pyproject.toml`: extra `catboost` en las dependencias opcionales de `supervisado` e `hibrido`.
  - `train_model.py`: imports `CatBoostClassifier` / `CatBoostRegressor`, entradas en
    `_build_models()` para supervisado (clf + reg, con y sin Optuna) e híbrido (clf + reg).
  - `tune_model.py`: función `_objective_catboost` (trials: iterations, depth, learning_rate,
    l2_leaf_reg, border_count) + entrada en `_OBJECTIVES`.
  - `README.md`: badge `· CatBoost ✓` junto a XGBoost y LightGBM.
  - `_tasks`: verificación `import catboost` tras `uv sync` si `use_catboost=true`.
- `use_docker` — configuracion Docker completa con interfaz web de chat.
  - `Dockerfile` con imagen Python slim, `uv`, `figlet` y banner ASCII DSKIT al arrancar.
  - `docker-compose.yml` con volumenes para `models/`, `data/` y `apuntes/`.
  - `docker/app.py` — servidor FastAPI + WebSocket que expone los modelos entrenados.
  - `docker/static/index.html` — interfaz de chat con tema oscuro, markdown renderizado,
    mensajes de usuario a la derecha y bot a la izquierda, chips de comandos rapidos.
  - `docker/entrypoint.sh` — arranca banner, comprueba modelos (entrena si no hay ninguno)
    y lanza uvicorn.
  - `.dockerignore` ajustado al proyecto.
  - `make docker-run`, `make docker-update`, `make docker-down` en el Makefile.
- `README.md` — seccion Docker, badge dskit y enlace a `https://github.com/cacelass/dskit`.
- Herramientas instaladas en el contenedor: `markitdown`, `graphify`.

---

## [1.7.0] — 2026-05-09

### Añadido
- `use_monitoring` — módulo de monitorización sin dependencias externas (solo scipy).
  Genera `monitoring/monitor.py` con `check_drift` (KS para numéricas, chi² para
  categóricas), `check_performance` (métricas vs baseline JSON) y `run_monitoring`
  que produce `drift_report.csv` y `drift_report.html`.
  `make monitor` ejecuta el análisis completo.
  11 tests en `test_monitoring.py`.
- `model_type: SVM` — SVC (clasificación) y SVR (regresión) disponibles como opción
  en `model_type`. Incluye Pipeline con StandardScaler interno para no depender del
  scaler global. Objetivo Optuna incluido cuando `use_optuna=true`.
- `.copier-answers.yml` añadido como placeholder en `template/` — Copier 9.x requiere
  que el archivo exista físicamente para generarlo en el proyecto destino y habilitar
  `copier update`.

### Corregido
- `model_type: SVM` — import de `Pipeline` en bloque Optuna usaba alias incorrecto.

---

## [1.6.0] — 2026-05-09

### Añadido
- `use_optuna` — optimización de hiperparámetros con Optuna.
  Genera `tuning/tune_model.py` con objetivos por modelo y `task_type`.
  `train_models()` carga `best_params_<modelo>.joblib` automáticamente si existen.
  `make tune` lanza la optimización. `OPTUNA_TRIALS` configurable en `main.py`.
  4 tests en `test_tuning.py`.

---

## [1.5.0] — 2026-05-09

### Añadido
- `use_duckdb` — carga de datos con DuckDB sobre CSV, Parquet y JSON sin servidor.
  Genera `load_data_duckdb()` con query SQL opcional y muestreo aleatorio,
  y `query_duckdb()` para SQL arbitrario.
  `make query` lanza el shell DuckDB interactivo.
  6 tests en `test_make_dataset.py`.

---

## [1.4.0] — 2026-05-09

### Añadido
- `use_api` — API REST con FastAPI: `/health`, `/info` y `/predict`.
  Adaptado a los 4 `ml_type` y ambos `task_type`. `make serve` en `localhost:8000`.
  8 tests en `test_api.py`.

### Corregido
- `redes_neuronales` — modelo guardado como `MLP_final.pt` (antes `MLP.pt`).
- `pyarrow` añadido al extra `redes_neuronales`.

---


### Añadido
- `use_duckdb` — carga de datos con DuckDB sobre CSV, Parquet y JSON sin servidor.
  Genera `load_data_duckdb()` con soporte de query SQL opcional y muestreo aleatorio
  (`sample_n`), y `query_duckdb()` para SQL arbitrario con alias `datos`.
- `make query` — shell DuckDB interactivo sobre `data/raw/` (solo si `use_duckdb=true`).
- Extra `duckdb` en `pyproject.toml` con `pyarrow` incluido para soporte Parquet.
- 6 tests en `test_make_dataset.py` cubriendo CSV, Parquet, query, sample y errores.

---

## [1.4.0] — 2026-05-09

### Añadido
- `use_api` — API REST con FastAPI para servir el modelo entrenado.
  Genera `api/main.py` con endpoints `/health`, `/info` y `/predict`,
  y `api/schemas.py` con modelos Pydantic V2.
  Adaptado a los 4 tipos de ML y ambos `task_type`.
- `make serve` — lanza uvicorn en `localhost:8000` (docs interactivos en `/docs`).
- Extra `api` en `pyproject.toml` con `fastapi`, `uvicorn` y `httpx`.
- 8 tests en `test_api.py` cubriendo health, info, predict, 422 y 503.
- Con `use_api=false` la carpeta `api/` y `test_api.py` se eliminan automáticamente.

### Corregido
- `redes_neuronales` — modelo guardado como `MLP_final.pt` (antes `MLP.pt`) para
  consistencia entre `train_model.py`, `predict_model.py` y la API.
- `redes_neuronales` — `pyarrow` añadido al extra `redes_neuronales` (requerido por polars).
- `test_build_features.py` — assert PCA con dimensión hardcodeada a 4 corregido.

---

## [1.2.0] — 2026-05-05

### Añadido
- `redes_neuronales` — **early stopping** con `PATIENCE` configurable en `main.py`.
  Cuando `PATIENCE > 0`, el entrenamiento se detiene si `val_loss` no mejora durante
  N épocas consecutivas y restaura automáticamente los mejores pesos (`*_best.pt`).
- `redes_neuronales` — **validation split** en el loop de entrenamiento (`VAL_SPLIT = 0.1`).
  Separa una fracción de `X_train` para validación y registra `Loss/val` en TensorBoard
  junto a `Loss/train`, permitiendo detectar overfitting visualmente.
- `redes_neuronales` — semilla global de PyTorch (`torch.manual_seed` +
  `torch.cuda.manual_seed_all`) para resultados reproducibles entre ejecuciones.
- `redes_neuronales` — `torch.cuda.memory_reserved()` mostrado junto a
  `memory_allocated()` en el log de inicio (útil para diagnosticar fragmentación de VRAM).

### Corregido
- `redes_neuronales` — `torch.backends.cudnn.allow_tf32 = True` estaba en el bloque
  MPS por error; movido al bloque CUDA donde corresponde (cuDNN es exclusivo de CUDA).
- `supervisado` — `y` tras `LabelEncoder.fit_transform` ahora se devuelve como
  `pd.Series` preservando índice y nombre, evitando `AttributeError: 'numpy.ndarray'
  object has no attribute 'value_counts'` en notebooks.
- `supervisado` — eliminada variable `le = LabelEncoder()` creada pero nunca usada.
- `redes_neuronales` — `torch.load(..., weights_only=True/False)` explícito para
  silenciar el `FutureWarning` de PyTorch 2.x.

---

## [1.1.3] — 2026-05-01

### Corregido
- Sustituidos todos los caracteres Unicode (`─`) en comentarios de `copier.yml`
  por ASCII puro — causaban que copier ignorara silenciosamente las preguntas
  adyacentes (`use_shap`, `_answers_file`)
- `_answers_file: .copier-answers.yml` ya funciona correctamente y se genera
  en cada proyecto, habilitando `copier update`
  
---

## [1.1.2] — 2026-04-27

### Añadido
- `use_decision_tree` como flag opcional en `copier.yml` — mismo patrón que `use_xgboost`/`use_lightgbm`. Solo aparece en el wizard cuando `ml_type` es `supervisado`/`hibrido` y `task_type == clasificacion`

### Corregido
- `train_model.py`: `DecisionTreeClassifier` ya no se incluía siempre — import, docstring e instanciación en `_build_models()` ahora condicionados a `use_decision_tree`

---

## [1.1.1] — 2026-04-26

### Corregido
- Ajustes menores de estabilidad tras la publicación de v1.1.0
- Corrección de referencias internas en `copier.yml`

---

## [1.1.0] — 2026-04-26

### Mejorado
- `build_features.py`: limpieza y robustez en el bloque de preparación de datos de usuario
- `main.py`: mensajes de consola más descriptivos al elegir modo pipeline vs. test
- Consistencia en los nombres de artefactos `.joblib` exportados entre módulos

---

## [1.0.0] — 2026-04-26

### Añadido
- **`find_best_threshold` en `predict_model.py`** — calcula el umbral óptimo maximizando F1 sobre la curva precision-recall. Incluido como bloque comentado; para clasificación binaria simplemente descomentar
- **Modo interactivo en `main.py`** — elige entre ejecutar el pipeline completo (`0`) o probar el modelo ya entrenado (`1`); entrada inválida ejecuta el pipeline por defecto
- **Preparación de datos de usuario en `build_features.py`** — bloque dedicado para transformar datos de entrada personalizados, alineado con las transformaciones de entrenamiento
- **Persistencia de artefactos `.joblib`** para reproducibilidad entre entornos:
  - `scaler.joblib`, `encoders.joblib`, `pca.joblib`, `threshold.joblib`
