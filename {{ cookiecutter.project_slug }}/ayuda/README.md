# Ayuda

Carpeta para recursos de referencia del proyecto. Papers, cheatsheets, notas
metodológicas o cualquier documentación de apoyo que no forme parte del código.

---

## Comandos esenciales

```bash
# Entorno
uv sync --extra dev --extra {{ cookiecutter.ml_type }}
source .venv/bin/activate

# Pipeline
make run          # main.py completo
make data         # solo carga/preproceso de datos
make train        # solo entrenamiento
make predict      # solo predicciones → reports/

# Calidad
make test         # pytest completo
make smoke        # tests de humo (rápidos)
make lint         # ruff check
make format       # ruff format

# Debug de rendimiento
make profile      # cProfile → reports/profile.prof
                  # luego: snakeviz reports/profile.prof

# Limpieza
make clean        # __pycache__ y cachés
make clean-models # borra .joblib / .pt
make clean-all    # todo
{% if cookiecutter.ml_type == "redes_neuronales" %}
# TensorBoard
make tb           # http://localhost:6006
{% endif %}
```

---

## Tipo de ML: `{{ cookiecutter.ml_type }}`{% if cookiecutter.ml_type == "redes_neuronales" %} · Arquitectura: `{{ cookiecutter.nn_model }}`{% endif %}

{% if cookiecutter.ml_type == "supervisado" %}
### Modelos disponibles{% if cookiecutter.model_type == "todos" %} (todos){% else %} (activo: `{{ cookiecutter.model_type }}`){% endif %}

| Modelo | Cuándo usar |
|---|---|
| KNN | Lazy learner, buena línea base. Escalar features antes |
| LogisticRegression | Clasificación binaria, interpretable, probabilidades calibradas |
| DecisionTree | Caja blanca, útil para explicabilidad |
| RandomForest | Robusto, feature importance, buen por defecto |
{% if cookiecutter.use_xgboost == "si" %}| XGBoost | Gradient boosting optimizado. Referencia en Kaggle |{% endif %}
{% if cookiecutter.use_lightgbm == "si" %}| LightGBM | Leaf-wise boosting. Más rápido que XGBoost en datos grandes |{% endif %}

Cambiar el modelo activo: edita `model_type` en `cookiecutter.json` y regenera,
o descomenta/comenta modelos directamente en `_build_models()` de `train_model.py`.
{% endif %}

{% if cookiecutter.ml_type == "redes_neuronales" %}
### Arquitecturas de red neuronal

| Modelo | Cuándo usar |
|---|---|
| MLP | Datos tabulares sin estructura temporal |
| CNN1D | Patrones locales entre features (señales, sensores) |
| LSTM | Dependencias temporales largas |
| GRU | Como LSTM, más ligero y rápido |
| Transformer | Relaciones globales, alta dimensionalidad |

**Cambiar arquitectura:** regenerar el proyecto con `cookiecutter` eligiendo otro `nn_model`.

### Checkpoints y reanudación de entrenamiento

```python
from {{ cookiecutter.project_slug }}.models.train_model import load_checkpoint

model, optimizer, epoch_inicio = load_checkpoint(
    input_dim=..., output_dim=...,
    checkpoint_path="models/checkpoint-10.pt",
)
# Continuar desde epoch_inicio + 1
```
{% endif %}

{% if cookiecutter.ml_type == "no_supervisado" %}
### Elegir el número de clusters k

1. `make run` → genera dendrograma, codo y silhouette en `reports/figures/`
2. Dendrograma: cuenta las ramas en el corte horizontal más largo
3. Codo: busca el "codo" en la curva de inercia
4. Silhouette: maximizar (rango [-1, 1])
5. Edita `N_CLUSTERS` en `main.py` con el k elegido
{% endif %}

{% if cookiecutter.ml_type == "hibrido" %}
### Estrategias híbridas

| Estrategia | Descripción |
|---|---|
| `pca_clf` | PCA → reduce dimensiones → clasificador |
| `umap_clf` | UMAP (no lineal) → clasificador |
| `kmeans_features` | Distancias a centroides → features adicionales → clasificador |
| `iso_feature` | Anomaly score (IsolationForest) → feature → clasificador |
| `semi_supervisado` | LabelPropagation/LabelSpreading con datos parcialmente etiquetados |

Edita `STRATEGY` en `main.py` para cambiar la estrategia.
{% endif %}

---

## Estructura de outputs

```
reports/
├── figures/
│   ├── cm_{{ cookiecutter.nn_model if cookiecutter.ml_type == "redes_neuronales" else "<modelo>" }}.png        # matriz de confusión
│   └── proba_dist_*.png     # distribución de probabilidades (binario)
{% if cookiecutter.ml_type == "redes_neuronales" %}├── predicciones_{{ cookiecutter.nn_model }}.csv   # y_true, y_pred, proba_*, correcto
└── resultados_{{ cookiecutter.nn_model }}.csv     # métricas por modelo, ordenadas por F1
{% else %}└── resultados.csv            # métricas comparativas
{% endif %}```

---

> Esta carpeta no se publica ni se incluye en el paquete. Es solo un espacio de trabajo local.