# ML Workflow — Ciclo de modelo

## Pipeline
```
make train  →  make predict
data/interim/   models/ + reports/
```

## Según tipo de ML

{% if ml_type == "supervisado" or ml_type == "hibrido" %}
### {{ ml_type|capitalize }} — {{ task_type|capitalize }}
| Paso | Comando | Agente |
|------|---------|--------|
| Entrenar | `make train` | `ml` (inspect_model tras entrenar) |
| Evaluar | `make predict` | `ml` (overfitting, comparación) + `graph` (figuras) |
{% if use_optuna %}| Tuning | `make tune` | Optuna + `ml` (analiza estudios) |{% endif %}
{% if use_mlflow %}| Tracking | `make mlflow` | MLflow UI + `mlflow` (list_runs, best_run) |{% endif %}

{% if task_type == "clasificacion" %}
  Métricas: accuracy, F1, precision, recall, ROC-AUC, matriz confusión
  {% if use_calibration %}Calibración: Temperature Scaling{% endif %}
  {% if use_conformal %}Predicción conforme: intervalos con cobertura garantizada{% endif %}
{% elif task_type == "regresion" %}
  Métricas: RMSE, MAE, MAPE, R²
  {% if use_conformal %}Predicción conforme: intervalos con cobertura garantizada{% endif %}
{% endif %}
{% if use_shap %}Explainability: SHAP (feature importance, dependence plots){% endif %}

Modelos disponibles: {{ model_type }}

{% elif ml_type == "no_supervisado" %}
### No supervisado — Clustering
| Paso | Comando | Agente |
|------|---------|--------|
| Entrenar | `make train` | `ml` (inspect_model) |
| Evaluar | `make predict` | `ml` (silhouette, DB, CH) + `graph` (figuras) |

Métricas: silhouette score, Davies-Bouldin, Calinski-Harabasz
Visualización: UMAP, dendrograma, proyección PCA 2D
Algoritmos: {{ cluster_model }}

{% elif ml_type == "redes_neuronales" %}
### Redes neuronales — {{ task_type|capitalize }}
| Paso | Comando | Agente |
|------|---------|--------|
| Entrenar | `make train` | `ml` (inspect_model) |
| Evaluar | `make predict` | `ml` (overfitting, comparación) + `graph` (figuras) |
| Monitor | `make tb` | TensorBoard (`runs/`) |
{% if use_optuna %}| Tuning | `make tune` | Optuna + `ml` |{% endif %}
{% if use_mlflow %}| Tracking | `make mlflow` | MLflow UI + `mlflow` |{% endif %}

Framework: PyTorch ({{ nn_model }})
Optimizador: {{ optimizer_type }}
Loss: {{ nn_loss_fn }}
{% if use_calibration %}Calibración: Temperature Scaling{% endif %}
{% if use_conformal %}Predicción conforme: intervalos con cobertura garantizada{% endif %}
{% if task_type == "clasificacion" %}
  Métricas: accuracy, F1, precision, recall vía torchmetrics
  Explainability: Captum
{% elif task_type == "regresion" %}
  Métricas: RMSE, MAE, MAPE, R² vía torchmetrics
  Explainability: Captum
{% endif %}
{% endif %}

## Agente `ml` — acciones clave
- `inspect_model` — tipo de estimador, parámetros, features
- `check_overfitting` — gap train/test, necesita threshold
- `feature_importance` — ranking de features (modelos árbol)
- `model_comparison` — ranking por tamaño, tipo, params
- `list_models` — descubre modelos en `models/`
{% if use_optuna %}- `analyze_study` — analiza estudios de Optuna{% endif %}

## Agente `graph` — figuras
- `audit_figures` — detecta figuras vacías, aspect ratio incorrecto
- Se ejecuta automáticamente en pipelines develop/fix

## Problemas comunes
- Overfitting → gap train/test > threshold. Solución: regularización, más datos, early stopping
- Modelo no encontrado → `make train` no ejecutado o falló
- Métricas malas → revisar features, más ingeniería, probar otros modelos
- {% if use_optuna %}Optuna lento → reducir n_trials, usar pruning{% endif %}
