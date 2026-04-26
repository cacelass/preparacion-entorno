{% if ml_type == "supervisado" %}
import numpy as np
import joblib

{% if task_type == "clasificacion" %}
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
{% else %}
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
{% endif %}

{% if use_xgboost == "si" %}
{% if task_type == "clasificacion" %}
from xgboost import XGBClassifier
{% else %}
from xgboost import XGBRegressor
{% endif %}
{% endif %}
{% if use_lightgbm == "si" %}
{% if task_type == "clasificacion" %}
from lightgbm import LGBMClassifier
{% else %}
from lightgbm import LGBMRegressor
{% endif %}
{% endif %}

{% if use_mlflow %}
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
{% endif %}

from {{ project_slug }}.utils.paths import MODELS_DIR


# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------

def _build_models() -> dict:
    """
    Define los modelos a entrenar.
    Tarea: {{ task_type }}

{% if task_type == "clasificacion" %}
    KNN                → lazy learner. Requiere features escaladas.
    LogisticRegression → modelo base binario. Interpretable, probabilidades calibradas.
    DecisionTree       → caja blanca. Regularizar con max_depth y min_samples_leaf.
    RandomForest       → ensemble robusto con feature importances.
    XGBoost (opc.)     → gradient boosting optimizado. Referencia en Kaggle.
    LightGBM (opc.)    → leaf-wise boosting. Más rápido en datasets grandes.
{% else %}
    LinearRegression   → modelo base. Rápido e interpretable.
    Ridge              → regresión lineal con regularización L2.
    Lasso              → regularización L1, útil para selección de variables.
    KNN                → regresión no paramétrica. Sensible a dimensionalidad.
    RandomForest       → ensemble robusto. feature_importances_ disponible.
    SVR                → potente en alta dimensión. Lento en datasets grandes.
    XGBoost (opc.)     → gradient boosting para regresión.
    LightGBM (opc.)    → leaf-wise boosting para regresión.
{% endif %}
    """
    models = {}

{% if task_type == "clasificacion" %}
    models["KNN"] = KNeighborsClassifier(n_neighbors=7, weights="distance")
    models["LogisticRegression"] = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42,
    )
    models["DecisionTree"] = DecisionTreeClassifier(
        max_depth=7, min_samples_leaf=5, class_weight="balanced", random_state=42,
    )
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200, max_depth=10, max_features="sqrt",
        max_samples=0.8, class_weight="balanced", random_state=42, n_jobs=-1,
    )
{% else %}
    models["LinearRegression"] = LinearRegression()
    models["Ridge"] = Ridge(alpha=1.0)
    models["Lasso"] = Lasso(alpha=0.1, max_iter=2000)
    models["KNN"] = KNeighborsRegressor(n_neighbors=7, weights="distance")
    models["RandomForest"] = RandomForestRegressor(
        n_estimators=200, max_depth=10, max_features="sqrt",
        max_samples=0.8, random_state=42, n_jobs=-1,
    )
    # models["SVR"] = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("reg", SVR(kernel="rbf", C=1.0, gamma="scale")),
    # ])
{% endif %}

{% if use_xgboost == "si" %}
{% if task_type == "clasificacion" %}
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )
{% else %}
    models["XGBoost"] = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="rmse",
        random_state=42, n_jobs=-1,
    )
{% endif %}
{% endif %}

{% if use_lightgbm == "si" %}
{% if task_type == "clasificacion" %}
    models["LightGBM"] = LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1,
    )
{% else %}
    models["LightGBM"] = LGBMRegressor(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
{% endif %}
{% endif %}

    return models


def _find_best_k(X_train, y_train, k_range=range(1, 21)) -> int:
    """Busca el k óptimo para KNN por cross-validation."""
{% if task_type == "clasificacion" %}
    scoring = "f1_weighted"
{% else %}
    scoring = "neg_root_mean_squared_error"
{% endif %}
    scores = {
        k: cross_val_score(
            KNeighborsClassifier(n_neighbors=k) if "{{ task_type }}" == "clasificacion"
            else KNeighborsRegressor(n_neighbors=k),
            X_train, y_train, cv=5, scoring=scoring,
        ).mean()
        for k in k_range
    }
    best_k = max(scores, key=scores.get)
    print(f"    KNN mejor k={best_k} ({scoring}={scores[best_k]:.3f})")
    return best_k


def train_models(
    X_train,
    y_train,
    tune_knn: bool = True,
    cv_evaluate: bool = True,
) -> dict:
    """
    Entrena modelos de {{ task_type }} y los guarda en models/.

{% if task_type == "clasificacion" %}
    Métrica CV: F1_weighted (5-fold).
{% else %}
    Métrica CV: RMSE negativo (5-fold) — más bajo (menos negativo) es mejor.
{% endif %}
{% if use_mlflow %}
    MLflow: cada modelo se loguea como un run independiente dentro del
    experimento '{{ project_slug }}'. Los artifacts (.joblib) se registran
    en el Model Registry bajo el nombre del modelo.
{% endif %}

    Returns
    -------
    dict : {nombre_modelo: modelo_entrenado}
    """
    print("--> Entrenando modelos de {{ task_type }}...")
    models = _build_models()

{% if model_type != "todos" %}
    selected = "{{ model_type }}"
    if selected in models:
        models = {selected: models[selected]}
    else:
        print(f"    model_type='{selected}' no encontrado. Entrenando todos.")
{% endif %}

    if tune_knn and "KNN" in models:
        best_k = _find_best_k(X_train, y_train)
{% if task_type == "clasificacion" %}
        models["KNN"] = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
{% else %}
        models["KNN"] = KNeighborsRegressor(n_neighbors=best_k, weights="distance")
{% endif %}

{% if use_mlflow %}
    mlflow.set_experiment("{{ project_slug }}")
{% endif %}

    trained = {}
    for name, model in models.items():
        print(f"    [{name}] entrenando...")

{% if use_mlflow %}
        with mlflow.start_run(run_name=name):
            # ── Parámetros ────────────────────────────────────────────────
            params = {}
            if hasattr(model, "get_params"):
                params = {k: v for k, v in model.get_params().items()
                          if v is not None and not callable(v)}
            mlflow.log_params(params)
            mlflow.log_param("task_type", "{{ task_type }}")
            mlflow.log_param("model_name", name)
{% endif %}

        model.fit(X_train, y_train)

        if cv_evaluate:
{% if task_type == "clasificacion" %}
            cv_score = cross_val_score(
                model, X_train, y_train, cv=5, scoring="f1_weighted"
            ).mean()
            print(f"      F1_weighted 5-fold CV: {cv_score:.3f}")
{% else %}
            cv_score = -cross_val_score(
                model, X_train, y_train, cv=5,
                scoring="neg_root_mean_squared_error",
            ).mean()
            print(f"      RMSE 5-fold CV: {cv_score:.4f}")
{% endif %}

{% if use_mlflow %}
            mlflow.log_metric("cv_score", cv_score)
{% endif %}

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"      Guardado → {name}.joblib")

{% if use_mlflow %}
        mlflow.sklearn.log_model(
            model, artifact_path=name,
            registered_model_name=f"{{ project_slug }}_{name}",
        )
        mlflow.log_artifact(str(MODELS_DIR / f"{name}.joblib"))
{% endif %}

        trained[name] = model

    print(f"--> {len(trained)} modelos guardados en {MODELS_DIR}")
    return trained


def load_models(model_names: list = None) -> dict:
    """Carga modelos desde disco."""
    if model_names is None:
        model_names = [p.stem for p in MODELS_DIR.glob("*.joblib")]
    models = {}
    for name in model_names:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
            print(f"    Cargado: {name}")
        else:
            print(f"    No encontrado: {path}")
    return models
    """
    Define los modelos a entrenar.

    KNN            → lazy learner, sin suposiciones sobre los datos.
                     Requiere features escaladas. Sensible a k y a dimensiones altas.

    LogisticReg    → modelo base en clasificación binaria. Rápido, interpretable
                     y genera probabilidades calibradas.

    DecisionTree   → caja blanca, fácil de interpretar. Propenso a overfitting
                     → regularizar con max_depth, min_samples_leaf.

    RandomForest   → ensemble de árboles. Robusto y buen rendimiento por defecto.
                     Permite calcular importancia de variables (feature_importances_).

    GradBoost      → mayor precisión que RF en muchos casos, pero más lento
                     y más sensible a hiperparámetros.

    SVM (RBF)      → potente en espacios de alta dimensión. Lento en datasets grandes.
                     El pipeline incluye StandardScaler propio.

    XGBoost        → gradient boosting optimizado. Muy rápido con histogramas,
                     soporte nativo de valores nulos, regularización L1/L2.
                     Referencia en tabular ML (Kaggle).

    LightGBM       → gradient boosting basado en hojas (leaf-wise). Más rápido que
                     XGBoost en datos grandes. Excelente con features categóricas
                     nativas. Menor consumo de memoria.

    Pipelines con PCA integrado:
      PCA_LogReg   → PCA(95%) → LogReg. Útil con muchas features correladas.
      PCA_SVM      → PCA(95%) → SVM RBF. Acelera SVM enormemente en alta dim.

    ¿Cuándo incluir los pipelines PCA?
      - Muchas features muy correladas entre sí (|r| > 0.8)
      - Alta dimensionalidad (>50 features)
      - SVM o KNN muy lentos sin reducción previa
      - Quieres comparar directamente si PCA mejora o no el rendimiento
    """
    models = {
        # --- KNN: el valor de n_neighbors se optimiza automáticamente en train_models() ---
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),

        # --- Regresión Logística: modelo base rápido e interpretable ---
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),

        # --- Árbol de Decisión: ajustar max_depth para evitar overfitting ---
        "DecisionTree": DecisionTreeClassifier(
            max_depth=7,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),

        # --- Random Forest: robusto, con importancia de variables ---
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            max_features="sqrt",   # sqrt(n_features) por árbol
            max_samples=0.8,       # bootstrap sample del 80%
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        # --- Gradient Boosting: alta precisión, mayor coste computacional ---
        # "GradientBoosting": GradientBoostingClassifier(
        #     n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
        # ),

        # --- SVM RBF: muy eficaz en dimensiones altas ---
        # El pipeline escala internamente; no necesita X ya escalado.
        # "SVM": Pipeline([
        #     ("scaler", StandardScaler()),
        #     ("clf", SVC(
        #         kernel="rbf", C=1.0, gamma="scale",
        #         class_weight="balanced", probability=True, random_state=42,
        #     )),
        # ]),

        # --- Pipeline PCA + Regresión Logística ---
        # "PCA_LogReg": Pipeline([
        #     ("pca", PCA(n_components=0.95, random_state=42)),
        #     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        # ]),

        # --- Pipeline PCA + SVM RBF ---
        # SVM es O(n²~n³); PCA lo acelera mucho. probability=True para ROC-AUC.
        # "PCA_SVM": Pipeline([
        #     ("pca", PCA(n_components=0.95, random_state=42)),
        #     ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
        #                 class_weight="balanced", probability=True, random_state=42)),
        # ]),
    }

{% if use_xgboost == "si" %}
    # --- XGBoost: gradient boosting con regularización nativa ---
    # scale_pos_weight: ratio clases negativas/positivas (para desbalanceo)
    # use_label_encoder=False evita deprecation warnings en versiones antiguas
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,                    # min loss reduction para hacer split
        reg_alpha=0.1,              # L1 regularización
        reg_lambda=1.0,             # L2 regularización
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        # scale_pos_weight=neg/pos,  # descomentar si clases muy desbalanceadas
    )
{% endif %}

{% if use_lightgbm == "si" %}
    # --- LightGBM: leaf-wise boosting, muy eficiente en memoria ---
    # num_leaves: controla complejidad del árbol (aumentar con cuidado → overfitting)
    # min_child_samples: mínimo de muestras por hoja (regularización)
    models["LightGBM"] = LGBMClassifier(
        n_estimators=300,
        max_depth=-1,               # -1 = sin límite (controlar con num_leaves)
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,                 # silencia output de entrenamiento
    )
{% endif %}

    return models



{% elif ml_type == "no_supervisado" %}
import joblib
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from {{ project_slug }}.utils.paths import MODELS_DIR


# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------

def _build_models(n_clusters: int = 3) -> dict:
    """
    Define los modelos de clustering a ajustar.

    KMeans            → rápido y escalable. Asume clusters esféricos.
                        Inicialización k-means++ reduce el riesgo de mínimos locales.

    AgglomerativeClustering → clustering jerárquico aglomerativo (bottom-up).
                               No requiere reinicializaciones. Permite usar un dendrograma
                               para elegir k antes de ajustar.
                               linkage: 'ward' (minimiza varianza intraclúster, mejor general),
                               'complete', 'average', 'single'.

    MiniBatchKMeans   → versión acelerada de KMeans para datasets grandes.
                        Usa mini-lotes; ligeramente peor calidad, mucho más rápido.

    DBSCAN            → basado en densidad; detecta clusters de cualquier forma
                        y es robusto a outliers. No necesita especificar k,
                        pero requiere ajustar eps y min_samples.
    """
    return {
        "KMeans": KMeans(
            n_clusters=n_clusters,
            init="k-means++",   # mejor inicialización que random
            n_init=10,
            max_iter=300,
            random_state=42,
        ),

        "AgglomerativeClustering": AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",     # 'ward' | 'complete' | 'average' | 'single'
        ),

        # "MiniBatchKMeans": MiniBatchKMeans(
        #     n_clusters=n_clusters, n_init=10, random_state=42
        # ),

        # "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    }


def find_optimal_k(X, k_range=range(2, 11)) -> dict:
    """
    Calcula el método del codo (inercia), el Silhouette Score,
    el Davies-Bouldin Score y el Calinski-Harabasz Score para cada k.

    Devuelve un diccionario con:
      - 'k_range'    : lista de k probados
      - 'inertias'   : inercia (WCSS) por k — buscar el codo
      - 'silhouettes': Silhouette Score por k — mayor es mejor (+1 máximo)
      - 'db_scores'  : Davies-Bouldin por k   — menor es mejor (0 mínimo)
      - 'ch_scores'  : Calinski-Harabasz por k — mayor es mejor

    Uso típico:
      metrics = find_optimal_k(X)
      plot_elbow_and_silhouette(metrics)   # en visualize.py
    """
    print("--> Calculando métricas para selección de k...")
    inertias, silhouettes, db_scores, ch_scores = [], [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(X)
        labels = km.labels_

        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels, metric="euclidean")
        db  = davies_bouldin_score(X, labels)
        ch  = calinski_harabasz_score(X, labels)

        silhouettes.append(sil)
        db_scores.append(db)
        ch_scores.append(ch)

        print(f"    k={k}  inercia={km.inertia_:.1f}  silhouette={sil:.3f}  "
              f"davies-bouldin={db:.3f}  calinski-harabasz={ch:.1f}")

    return {
        "k_range":     list(k_range),
        "inertias":    inertias,
        "silhouettes": silhouettes,
        "db_scores":   db_scores,
        "ch_scores":   ch_scores,
    }


def train_models(X, n_clusters: int = 3) -> dict:
    """
    Ajusta todos los modelos definidos en _build_models() y los guarda en models/.

    ⚠ AgglomerativeClustering no tiene método .predict() — usa .labels_ para
    asignar clusters a los datos de entrenamiento.

    Parameters
    ----------
    X          : datos de entrenamiento (array-like)
    n_clusters : número de clusters (ajústalo tras analizar el codo y silhouette)

    Returns
    -------
    dict : {nombre_modelo: modelo_ajustado}
    """
    print(f"--> Ajustando modelos de clustering (k={n_clusters})...")
    models = _build_models(n_clusters)
    fitted = {}

    for name, model in models.items():
        print(f"    [{name}] ajustando...")
        model.fit(X)

        # Silhouette Score (no aplicable a DBSCAN con un solo cluster)
        labels   = model.labels_ if hasattr(model, "labels_") else model.predict(X)
        n_unique = len(set(labels)) - (1 if -1 in labels else 0)
        if n_unique > 1:
            sil = silhouette_score(X, labels)
            db  = davies_bouldin_score(X, labels)
            print(f"      Silhouette Score  : {sil:.3f}  (mejor → +1)")
            print(f"      Davies-Bouldin    : {db:.3f}   (mejor → 0)")

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"      Guardado → {name}.joblib")
        fitted[name] = model

    return fitted


def train_kmeans_pipeline(X_train, y_train, n_clusters: int = 50):
    """
    Pipeline KMeans → LogisticRegression.
    Usa el clustering como reducción de dimensionalidad antes de un clasificador.

    Útil cuando se dispone de etiquetas (semisupervisado):
      la distancia a cada centroide se usa como features para el clasificador.

    Parameters
    ----------
    n_clusters : número de centroides KMeans (actúan como nuevas features)

    Returns
    -------
    pipeline entrenado
    """
    print(f"--> Entrenando pipeline KMeans({n_clusters}) + LogisticRegression...")
    pipeline = Pipeline([
        ("kmeans",  KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)),
        ("log_reg", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODELS_DIR / "KMeansPipeline.joblib")
    print("    Guardado → KMeansPipeline.joblib")
    return pipeline


def load_models(model_names: list = None) -> dict:
    """
    Carga modelos desde disco.

    Parameters
    ----------
    model_names : lista de nombres sin extensión.
                  Si None, carga todos los .joblib disponibles en models/.

    Returns
    -------
    dict : {nombre_modelo: modelo_cargado}
    """
    if model_names is None:
        model_names = [p.stem for p in MODELS_DIR.glob("*.joblib")]

    models = {}
    for name in model_names:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
            print(f"    Cargado: {name}")
        else:
            print(f"    ⚠ No encontrado: {path}")
    return models


{% elif ml_type == "redes_neuronales" %}
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from {{ project_slug }}.utils.paths import MODELS_DIR, RUNS_DIR


# ---------------------------------------------------------------------------
# Detección de dispositivo (CPU / CUDA)
# ---------------------------------------------------------------------------
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# MLP — Perceptrón Multicapa
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """
    Red neuronal densa (MLP) con capas ocultas configurables.

    Cuándo usarla:
      - Datos tabulares de tamaño medio/grande.
      - No hay estructura temporal ni espacial en los datos.
      - Buena línea base antes de probar arquitecturas más complejas.

    Parameters
    ----------
    input_dim   : número de features de entrada
    output_dim  : número de clases (clasificación) o 1 (regresión)
    hidden_dims : lista con el tamaño de cada capa oculta, e.g. [128, 64]
    dropout     : tasa de dropout aplicada tras cada capa oculta (regularización)
    """
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# CNN1D — Red Convolucional 1-D
# ---------------------------------------------------------------------------
class CNN1D(nn.Module):
    """
    Red convolucional sobre secuencias 1-D (features tabulares tratadas como canal único).

    Cuándo usarla:
      - Datos tabulares con patrones locales entre features adyacentes
        (e.g. señales de sensores, series temporales de longitud fija).
      - Más rápida que LSTM para secuencias largas.
      - Menos parámetros que MLP cuando las features tienen estructura local.

    Arquitectura:
      Conv1D(1→32, k=3) → BN → ReLU → MaxPool →
      Conv1D(32→64, k=3) → BN → ReLU → AdaptiveAvgPool →
      FC(64→output_dim)

    Parameters
    ----------
    input_dim  : número de features de entrada (= longitud de la secuencia)
    output_dim : número de clases de salida
    dropout    : dropout antes de la capa final
    """
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # → (batch, 64, 1) independientemente de input_dim
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        # x: (batch, input_dim) → (batch, 1, input_dim) para Conv1d
        x = x.unsqueeze(1)
        x = self.conv(x).squeeze(-1)  # (batch, 64)
        return self.head(x)


# ---------------------------------------------------------------------------
# LSTM — Long Short-Term Memory
# ---------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    """
    LSTM para clasificación sobre datos secuenciales o tabulares.

    Cuándo usarla:
      - Datos con dependencias temporales / secuenciales largas.
      - Series temporales multivariadas donde el orden importa.
      - Las features representan pasos de tiempo consecutivos.

    Arquitectura:
      LSTM(input_dim→hidden_dim, num_layers, bidireccional opcional) →
      último estado oculto → Dropout → FC(hidden_dim→output_dim)

    Parameters
    ----------
    input_dim    : número de features en cada paso de tiempo
                   (para datos tabulares: input_dim=n_features, seq_len=1)
    output_dim   : número de clases de salida
    hidden_dim   : tamaño de las celdas LSTM (por dirección)
    num_layers   : capas LSTM apiladas (1-3 normalmente)
    bidirectional: si True, procesa la secuencia en ambas direcciones
    dropout      : dropout entre capas LSTM (si num_layers > 1) y antes de FC
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2,
                 bidirectional=False, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        fc_in = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_in, output_dim),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # Para datos tabulares sin dimensión temporal: unsqueeze(1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        # Tomamos el último estado de la última dirección
        out = h_n[-1]
        return self.head(out)


# ---------------------------------------------------------------------------
# GRU — Gated Recurrent Unit
# ---------------------------------------------------------------------------
class GRUClassifier(nn.Module):
    """
    GRU para clasificación sobre datos secuenciales o tabulares.

    Cuándo usarla vs LSTM:
      - Menos parámetros que LSTM → más rápido en entrenamiento.
      - Rendimiento similar a LSTM en la mayoría de tareas.
      - Preferible cuando el dataset es pequeño o la GPU es limitada.

    Parameters
    ----------
    input_dim  : features por paso de tiempo
    output_dim : clases de salida
    hidden_dim : tamaño de las celdas GRU
    num_layers : capas GRU apiladas
    dropout    : dropout entre capas y antes de FC
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        _, h_n = self.gru(x)
        return self.head(h_n[-1])


# ---------------------------------------------------------------------------
# Transformer — Encoder para clasificación tabular/secuencial
# ---------------------------------------------------------------------------
class _PositionalEncoding(nn.Module):
    """Codificación posicional sinusoidal (Vaswani et al., 2017)."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Transformer Encoder para clasificación tabular o secuencial.

    Cuándo usarlo:
      - Relaciones globales entre features/tokens son relevantes.
      - Datos tabulares de alta dimensionalidad con interacciones complejas.
      - Secuencias donde la atención posicional es más informativa que LSTM.
      - Requiere más datos y más VRAM que MLP/LSTM.

    Arquitectura:
      Embedding lineal → Positional Encoding →
      N × TransformerEncoderLayer(d_model, nhead, dim_ff) →
      [CLS] token promediado → FC(d_model→output_dim)

    Parameters
    ----------
    input_dim  : features de entrada (= longitud del vector x por paso)
    output_dim : clases de salida
    d_model    : dimensión interna del Transformer (múltiplo de nhead)
    nhead      : número de cabezas de atención
    num_layers : capas de Transformer apiladas
    dim_ff     : dimensión de la FFN interna (normalmente 4 * d_model)
    dropout    : dropout en atención y FFN
    """
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4,
                 num_layers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model debe ser múltiplo de nhead"
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_enc   = _PositionalEncoding(d_model, dropout=dropout)
        enc_layer      = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head    = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch, input_dim) → (batch, 1, input_dim) si es tabular
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.embedding(x)      # (batch, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)          # Global Average Pooling sobre tokens
        return self.head(x)


# ---------------------------------------------------------------------------
# Fábrica de modelos — selección por copier
# ---------------------------------------------------------------------------
def _build_model(input_dim: int, output_dim: int) -> nn.Module:
    """
    Devuelve la arquitectura seleccionada en nn_model.

    MLP         → datos tabulares sin estructura temporal.
    CNN1D       → patrones locales entre features (señales, sensores).
    LSTM        → dependencias temporales largas.
    GRU         → como LSTM pero más ligero.
    Transformer → relaciones globales, alta dimensionalidad.
    """
{% if nn_model == "MLP" %}
    return MLP(input_dim=input_dim, output_dim=output_dim,
                hidden_dims=[256, 128, 64], dropout=0.3)
{% elif nn_model == "CNN1D" %}
    return CNN1D(input_dim=input_dim, output_dim=output_dim, dropout=0.3)
{% elif nn_model == "LSTM" %}
    return LSTMClassifier(input_dim=input_dim, output_dim=output_dim,
                          hidden_dim=64, num_layers=2, bidirectional=False, dropout=0.3)
{% elif nn_model == "GRU" %}
    return GRUClassifier(input_dim=input_dim, output_dim=output_dim,
                         hidden_dim=64, num_layers=2, dropout=0.3)
{% elif nn_model == "Transformer" %}
    return TransformerClassifier(input_dim=input_dim, output_dim=output_dim,
                                 d_model=64, nhead=4, num_layers=2, dim_ff=256, dropout=0.1)
{% endif %}


MODEL_NAME = "{{ nn_model }}"


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def train_models(
    X_train,
    y_train,
    input_dim: int,
    output_dim: int,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    checkpoint_every: int = 10,
) -> dict:
    """
    Entrena la red neuronal seleccionada ({{ nn_model }}) con PyTorch.

    Características:
    - CUDA automático si está disponible
    - TensorBoard: loss por época en runs/
    - Checkpoints periódicos en models/checkpoint-{epoch}.pt
    - Guardado final de pesos en models/{{ nn_model }}.pt

    Parameters
    ----------
    input_dim        : número de features de entrada
    output_dim       : número de clases de salida
    epochs           : épocas de entrenamiento
    batch_size       : tamaño de mini-lote
    lr               : learning rate para Adam
    checkpoint_every : cada cuántas épocas guardar checkpoint (0 = desactivado)

    Returns
    -------
    dict : {'{{ nn_model }}': modelo_entrenado}
    """
    print(f"--> Entrenando red neuronal: {{ nn_model }}...")

    X_t    = torch.tensor(X_train.values if hasattr(X_train, "values") else X_train,
                          dtype=torch.float32)
    y_t    = torch.tensor(y_train.values if hasattr(y_train, "values") else y_train,
                          dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model     = _build_model(input_dim=input_dim, output_dim=output_dim).to(device)
    # model   = torch.compile(model)   # PyTorch ≥ 2.0: descomentar para mayor velocidad
    print(f"    Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()  # cambiar a MSELoss para regresión
    tb        = SummaryWriter(log_dir=str(RUNS_DIR))
    print(f"    TensorBoard → tensorboard --logdir {RUNS_DIR}")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # evita exploding gradients
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        tb.add_scalar("Loss/train", avg_loss, epoch)
        tb.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss":                 avg_loss,
            }, MODELS_DIR / f"checkpoint-{epoch+1}.pt")

    tb.close()
    out_path = MODELS_DIR / f"{MODEL_NAME}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"    Guardado: {out_path}")
    return {MODEL_NAME: model}


def load_model(input_dim: int, output_dim: int, weights_path: str = None):
    """Carga pesos finales y devuelve el modelo en modo eval."""
    if weights_path is None:
        weights_path = f"{MODEL_NAME}.pt"
    path  = MODELS_DIR / weights_path if not str(weights_path).startswith("/") else weights_path
    model = _build_model(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"    Modelo cargado desde {path}")
    return model


def load_checkpoint(input_dim: int, output_dim: int, checkpoint_path: str):
    """Carga un checkpoint para continuar el entrenamiento."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model      = _build_model(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer  = torch.optim.AdamW(model.parameters())
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_inicio = checkpoint["epoch"]
    print(f"    Checkpoint cargado: epoch {epoch_inicio}")
    return model, optimizer, epoch_inicio


{% elif ml_type == "hibrido" %}
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

{% if use_xgboost == "si" %}
from xgboost import XGBClassifier
{% endif %}
{% if use_lightgbm == "si" %}
from lightgbm import LGBMClassifier
{% endif %}

from {{ project_slug }}.utils.paths import MODELS_DIR


# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------

def _build_models(strategy: str) -> dict:
    """
    Clasificadores adaptados a la estrategia híbrida elegida.

    'pca_clf' / 'umap_clf'
        Espacio reducido (pocas dimensiones). SVM y LogReg son muy competitivos.
        RandomForest pierde algo de ventaja cuando hay pocas features.

    'kmeans_features' / 'iso_feature'
        Espacio original + features extra. GradientBoosting aprovecha bien
        las distancias a centroides o el anomaly score.

    'semi_supervisado'
        Espacio original con etiquetas propagadas. Todos los modelos válidos.
    """
    base = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            max_features="sqrt",
            max_samples=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    }

    if strategy in ("pca_clf", "umap_clf"):
        # SVM con escalado propio: muy eficaz en espacios reducidos
        base["SVM_RBF"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", probability=True, random_state=42,
            )),
        ])

    if strategy in ("kmeans_features", "iso_feature"):
        # GradBoost aprovecha bien distancias a centroides o anomaly scores
        base["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

{% if use_xgboost == "si" %}
    # XGBoost: robusto ante outliers, regularización nativa, muy competitivo
    # en espacios reducidos (pca/umap) y con features de distancia (kmeans).
    base["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
{% endif %}

{% if use_lightgbm == "si" %}
    # LightGBM: leaf-wise, más rápido que XGBoost con datasets grandes.
    # Especialmente bueno cuando hay features categóricas o alta cardinalidad.
    base["LightGBM"] = LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
{% endif %}

    return base


def _find_best_k(X_train, y_train, k_range=range(1, 21)) -> int:
    """
    Busca el mejor k para KNN por validación cruzada (5-fold, métrica F1_weighted).
    Devuelve el k con mayor F1 medio, priorizando k más alto en empates.
    """
    print("    Buscando mejor k para KNN...")
    best_k, best_score = 1, 0.0
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        score = cross_val_score(knn, X_train, y_train, cv=5, scoring="f1_weighted").mean()
        if score >= best_score:   # >= → preferimos k más alto en empates
            best_k, best_score = k, score
    print(f"    Mejor k = {best_k}  (F1_weighted CV = {best_score:.3f})")
    return best_k


def train_models(
    X_train,
    y_train,
    strategy: str = "pca_clf",
    tune_knn: bool = True,
    cv_evaluate: bool = True,
) -> dict:
    """
    Entrena clasificadores sobre el espacio transformado por la estrategia híbrida.

    Parameters
    ----------
    X_train      : features de entrenamiento ya transformadas (array-like)
    y_train      : etiquetas de entrenamiento (array-like)
    strategy     : estrategia híbrida usada para generar X_train.
                   Valores: 'pca_clf' | 'umap_clf' | 'kmeans_features' |
                             'iso_feature' | 'semi_supervisado'
    tune_knn     : si True, optimiza k de KNN por cross-validation antes de entrenar.
    cv_evaluate  : si True, muestra F1_weighted (5-fold CV) de cada modelo.

    Returns
    -------
    dict : {nombre_modelo: modelo_entrenado}
    """
    print(f"--> Entrenando modelos híbridos (estrategia='{strategy}')...")
    models = _build_models(strategy)

    if tune_knn and "KNN" in models:
        best_k = _find_best_k(X_train, y_train)
        models["KNN"] = KNeighborsClassifier(n_neighbors=best_k, weights="distance")

    trained = {}
    for name, model in models.items():
        print(f"    [{name}] entrenando...")
        model.fit(X_train, y_train)

        if cv_evaluate:
            cv_score = cross_val_score(
                model, X_train, y_train, cv=5, scoring="f1_weighted"
            ).mean()
            print(f"      F1_weighted 5-fold CV: {cv_score:.3f}")

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"      Guardado → {name}.joblib")
        trained[name] = model

    print(f"--> {len(trained)} modelos guardados en {MODELS_DIR}")
    return trained


def load_models(model_names: list = None) -> dict:
    """
    Carga modelos desde disco.

    Parameters
    ----------
    model_names : lista de nombres sin extensión, e.g. ["RandomForest", "KNN"].
                  Si None, carga todos los .joblib disponibles en models/.

    Returns
    -------
    dict : {nombre_modelo: modelo_cargado}
    """
    if model_names is None:
        model_names = [p.stem for p in MODELS_DIR.glob("*.joblib")]

    models = {}
    for name in model_names:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
            print(f"    Cargado: {name}")
        else:
            print(f"    ⚠ No encontrado: {path}")
    return models
{% endif %}