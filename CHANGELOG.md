# Changelog

Todos los cambios relevantes de esta plantilla se documentan aquí.  
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).

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