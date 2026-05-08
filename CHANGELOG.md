# Changelog

Historial de versiones de dskit. Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).

---

## [1.3.0] — 2026-05-06

### Añadido
- `redes_neuronales` — validation split configurable (`VAL_SPLIT = 0.1`) con `Loss/val` en TensorBoard junto a `Loss/train`
- `redes_neuronales` — early stopping con `PATIENCE` configurable en `main.py`; restaura los mejores pesos al parar
- `redes_neuronales` — semillas globales (`torch.manual_seed` + `cuda.manual_seed_all`) para reproducibilidad entre ejecuciones
- `redes_neuronales` — `torch.cuda.memory_reserved()` mostrado en el log de inicio junto a `memory_allocated()`
- Tests de forward pass para `LSTMClassifier`, `GRUClassifier` y `TransformerClassifier`
- Tests de `train_models`, `save` y `load_model` para `redes_neuronales`

### Corregido
- `redes_neuronales` — `torch.backends.cudnn.allow_tf32` estaba en el bloque MPS; movido a CUDA donde corresponde
- `redes_neuronales` — deteccion de dispositivo en `predict_model.py` actualizada a CUDA > MPS > CPU
- `redes_neuronales` — `torch.load` con `weights_only` explicito para silenciar `FutureWarning` de PyTorch 2.x
- `supervisado` — `y = le_target.fit_transform(...)` ahora devuelve `pd.Series` con `index=y.index`; elimina `AttributeError: 'ndarray' has no attribute 'value_counts'`
- `supervisado` — eliminada variable `le = LabelEncoder()` creada pero nunca usada
- `supervisado` — `stratify=y` condicionado a `task_type == clasificacion`; evita error con targets continuos
- `hibrido` — mismos tres fixes de `le_target`, `le` sin usar y `stratify` condicional
- `hibrido` — `DECISION_THRESHOLD` e import de `test_model` condicionados a `clasificacion`; evita `ImportError` en regresion
- `hibrido` — `evaluate_models` sin `threshold` en regresion; `sort_values` por `RMSE_test` en lugar de `Acc_test`
- `hibrido` — `trained = {}` estaba pegado al tag `{% endif %}` en la misma linea; generaba `SyntaxError`
- `redes_neuronales` — `index=y.index` añadido en `build_features.py` al envolver `le_target` en `pd.Series`
- `supervisado` y `redes_neuronales` — `sort_values('Acc_test')` en `main.py` condicionado a `clasificacion`
- `supervisado` — `test_model()` en el menu interactivo condicionado a `clasificacion`
- `train_model.py` — `use_label_encoder=False` eliminado de `XGBClassifier` (eliminado en XGBoost 2.0)
- `train_model.py` — `DecisionTreeRegressor` añadido a imports e instanciacion para `task_type == regresion`
- `visualize.py` no_supervisado — guard ante `plot_distributions` con 0 columnas numericas
- `predict_model.py` — `XGBClassifier` en hibrido sin `use_label_encoder=False`
- `conftest.py` — `pd.cut(...).astype(int)` reemplazado por `.astype('Int64').fillna(0).astype(int)`
- `paths.py` — `RUNS_DIR` y su `mkdir` condicionados a `ml_type == redes_neuronales`
- `copier.yml` — em dash unicode en primera linea sustituido por ASCII
- `pyproject.toml` — `black` eliminado del grupo `dev` (conflicto con `ruff format`); `shap` añadido condicionalmente a `supervisado`; `pytest-cov` añadido a `dev`

---

## [1.2.0] — 2026-05-05

### Añadido
- `redes_neuronales` — deteccion de Apple Silicon GPU (MPS) en `train_model.py` y `predict_model.py`: CUDA > MPS > CPU

---

## [1.1.3] — 2026-05-01

### Corregido
- Sustituidos todos los caracteres Unicode (`─`) en comentarios de `copier.yml` por ASCII puro — causaban que Copier ignorara silenciosamente las preguntas adyacentes (`use_shap`, `_answers_file`)
- `_answers_file: .copier-answers.yml` funciona correctamente y se genera en cada proyecto, habilitando `copier update`

---

## [1.1.2] — 2026-04-27

### Añadido
- `use_decision_tree` como flag opcional en `copier.yml` — mismo patrón que `use_xgboost` / `use_lightgbm`. Solo aparece en el wizard cuando `ml_type` es `supervisado` / `hibrido` y `task_type == clasificacion`

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
- **Persistencia de artefactos `.joblib`** para reproducibilidad entre entornos: `scaler.joblib`, `encoders.joblib`, `pca.joblib`, `threshold.joblib`