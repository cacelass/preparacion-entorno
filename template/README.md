# {{ project_name }}

![Python](https://img.shields.io/badge/Python-{{ python_version }}+-blue?logo=python&logoColor=white)
{% if ml_type == 'supervisado' %}![ML Type](https://img.shields.io/badge/ML-Supervised%20{{ task_type | capitalize }}-orange)
{% elif ml_type == 'no_supervisado' %}![ML Type](https://img.shields.io/badge/ML-Unsupervised%20Clustering-orange)
{% elif ml_type == 'redes_neuronales' %}![ML Type](https://img.shields.io/badge/ML-Neural%20Networks%20{{ nn_model }}-orange)
{% elif ml_type == 'hibrido' %}![ML Type](https://img.shields.io/badge/ML-Hybrid-orange)
{% endif %}{% if use_mlflow %}![Tracking](https://img.shields.io/badge/Experiment%20Tracking-MLflow-blue?logo=mlflow)
{% endif %}![Version](https://img.shields.io/badge/Version-{{ project_version }}-green)
![Author](https://img.shields.io/badge/Author-{{ project_author_name | replace(" ", "%20") | replace("-", "--") }}-blueviolet)

> {{ project_description }}

**Tipo de ML:** `{{ ml_type }}`{% if ml_type == "redes_neuronales" %} — arquitectura: `{{ nn_model }}`{% endif %}  
**Autor:** {{ project_author_name }}  
**Versión:** {{ project_version }}{% if use_xgboost %} · XGBoost ✓{% endif %}{% if use_lightgbm %} · LightGBM ✓{% endif %}

---

## Estructura del proyecto

```
{{ project_slug }}/
├── data/
│   ├── raw/            ← datos originales (nunca modificar)
│   ├── interim/        ← datos en proceso
│   └── processed/      ← datos listos para modelar
├── models/             ← modelos entrenados (.joblib / .pt)
│   └── artifacts/      ← encoders, scalers, etc.
├── notebooks/
│   ├── 0-0-...-Descargadatos.ipynb
│   ├── 0-1-...-ProcesamientoDatos.ipynb
│   └── 0-2-...-Ejecucion.ipynb
├── reports/figures/    ← gráficos generados
├── {{ project_slug }}/
│   ├── data/           make_dataset.py
│   ├── features/       build_features.py
│   ├── models/         train_model.py · predict_model.py
│   ├── visualization/  visualize.py
│   └── utils/          paths.py
├── tests/
├── main.py             ← pipeline completo
├── Makefile
└── pyproject.toml
```

## Inicio rápido

```bash
# 1. Instalar dependencias
make setup

# 2. Activar entorno
source .venv/bin/activate

# 3. Colocar datos en data/raw/ y editar DATA_FILE / TARGET_COL en main.py

# 4. Explorar con notebooks
invoke lab

# 5. Pipeline completo
python main.py
```

Consulta el archivo `ayuda` para más detalles.