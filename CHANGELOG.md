# Changelog

Todos los cambios relevantes de esta plantilla se documentan aquí.  
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).

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