# Changelog

Todos los cambios relevantes de esta plantilla se documentan aquí.  
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).

---

## [Unreleased]

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