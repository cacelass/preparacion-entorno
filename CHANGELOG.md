# Changelog

Todos los cambios relevantes de esta plantilla se documentan aquí.  
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).

---

## [Unreleased]

### Por añadir
- Soporte para regresión (además de clasificación)
- Integración opcional con MLflow para tracking de experimentos

---

## [0.3.0] — 2025-04

### Añadido
- **Redes neuronales — múltiples arquitecturas** vía `nn_model`:
  - `MLP` — Perceptrón multicapa para datos tabulares
  - `CNN1D` — Red convolucional 1-D para patrones locales entre features
  - `LSTM` — Long Short-Term Memory para dependencias temporales
  - `GRU` — Gated Recurrent Unit, alternativa ligera a LSTM
  - `Transformer` — Encoder con multi-head attention y positional encoding
- **XGBoost y LightGBM** como opcionales en `supervisado` e `híbrido`
  (`use_xgboost` / `use_lightgbm` en `json`)
- **`hooks/post_gen_project.py`** — instala dependencias automáticamente
  con `uv sync` tras generar el proyecto
- **`make tb`** — lanza TensorBoard en `localhost:6006` (solo `redes_neuronales`)
- **`make smoke`** — ejecuta tests marcados con `@pytest.mark.smoke`
- **`predict_model.py` (redes_neuronales)** — exporta predicciones a CSV:
  - `reports/predicciones_{MODEL_NAME}.csv` con `y_true`, `y_pred`, `proba_*`, `correcto`
  - `reports/resultados_{MODEL_NAME}.csv` ordenado por F1
  - `predict_new()` con parámetro `export_csv=True`
- **`.env.example`** — plantilla de variables de entorno documentada
- **`CHANGELOG.md`** — este archivo

### Mejorado
- `train_model.py` (redes_neuronales): AdamW + CosineAnnealingLR + gradient clipping
- `main.py` (redes_neuronales): muestra arquitectura activa y métricas finales al terminar
- Tests de `test_train_model.py` ampliados para cubrir las 5 arquitecturas
- `README.md` del proyecto generado muestra arquitectura de red y flags de boosting

---

## [0.2.0] — 2025-02

### Añadido
- Bloque `redes_neuronales` con MLP + PyTorch + TensorBoard
- `tasks.py` con pipeline invoke (`data`, `features`, `train`, `predict`)
- Tests en `tests/` para los 4 tipos de ML
- Soporte completo para `uv` como gestor de dependencias

### Mejorado
- `Makefile` unificado con targets para todos los tipos de ML
- `pyproject.toml` con grupos de dependencias opcionales por `ml_type`

---

## [0.1.0] — 2024-11

### Añadido
- Estructura inicial de proyecto con copier
- Soporte para `supervisado`, `no_supervisado` e `híbrido`
- Módulos: `make_dataset`, `build_features`, `train_model`, `predict_model`, `visualize`
- Paths centralizados en `utils/paths.py`
- Integración con Sphinx para documentación