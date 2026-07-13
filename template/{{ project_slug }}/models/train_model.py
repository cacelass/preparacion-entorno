{% macro load_models_macro() %}
def load_models(model_names: list | None = None) -> dict:
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
            print(f"    ⚠ No encontrado: {path}")
    return models
{% endmacro %}
{% if ml_type == "supervisado" %}
import numpy as np
import joblib
from contextlib import nullcontext

{% if task_type == "clasificacion" %}
{% if model_type == "todos" or model_type == "RandomForest" %}
from sklearn.ensemble import RandomForestClassifier
{% endif %}
{% if model_type == "todos" or model_type == "LogisticRegression" %}
from sklearn.linear_model import LogisticRegression
{% endif %}
{% if model_type == "todos" or model_type == "DecisionTree" %}
from sklearn.tree import DecisionTreeClassifier
{% endif %}
{% if model_type == "todos" or model_type == "KNN" %}
from sklearn.neighbors import KNeighborsClassifier
{% endif %}
{% if model_type == "todos" or model_type == "SVM" %}
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler as _SVMScaler
{% endif %}
from sklearn.model_selection import cross_val_score
{% else %}
{% if model_type == "todos" %}
from sklearn.linear_model import LinearRegression, Ridge, Lasso
{% endif %}
{% if model_type == "todos" or model_type == "RandomForest" %}
from sklearn.ensemble import RandomForestRegressor
{% endif %}
{% if model_type == "todos" or model_type == "KNN" %}
from sklearn.neighbors import KNeighborsRegressor
{% endif %}
{% if model_type == "todos" or model_type == "DecisionTree" %}
from sklearn.tree import DecisionTreeRegressor
{% endif %}
{% if model_type == "todos" or model_type == "SVM" %}
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline as _SVMPipeline
from sklearn.preprocessing import StandardScaler as _SVMScaler
{% endif %}
from sklearn.model_selection import cross_val_score
{% endif %}

{% if use_xgboost or model_type == "XGBoost" %}
{% if task_type == "clasificacion" %}
from xgboost import XGBClassifier
{% else %}
from xgboost import XGBRegressor
{% endif %}
{% endif %}
{% if use_lightgbm or model_type == "LightGBM" %}
{% if task_type == "clasificacion" %}
from lightgbm import LGBMClassifier
{% else %}
from lightgbm import LGBMRegressor
{% endif %}
{% endif %}
{% if use_catboost or model_type == "CatBoost" %}
{% if task_type == "clasificacion" %}
from catboost import CatBoostClassifier
{% else %}
from catboost import CatBoostRegressor
{% endif %}
{% endif %}

{% if use_mlflow %}
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
{% endif %}

from {{ project_slug }}.utils.paths import MODELS_DIR, ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------

def _load_best_params(model_name: str) -> dict:
    """Carga los mejores hiperparámetros de Optuna si existen."""
    path = ARTIFACTS_DIR / f"best_params_{model_name}.joblib"
    if path.exists():
        params = joblib.load(path)
        print(f"    [{model_name}] best_params cargados desde Optuna: {params}")
        return params
    return {}


def _build_models() -> dict:
    """
    Define los modelos a entrenar.
    Tarea: {{ task_type }}

{% if task_type == "clasificacion" %}
{% if model_type == "todos" or model_type == "KNN" %}
    KNN                → lazy learner. Requiere features escaladas.
{% endif %}
{% if model_type == "todos" or model_type == "LogisticRegression" %}
    LogisticRegression → modelo base binario. Interpretable, probabilidades calibradas.
{% endif %}
{% if model_type == "todos" or model_type == "DecisionTree" %}
    DecisionTree       → caja blanca. Regularizar con max_depth y min_samples_leaf.
{% endif %}
{% if model_type == "todos" or model_type == "SVM" %}
    SVM                → margen máximo con kernel RBF. Incluye escalado interno.
{% endif %}
{% if model_type == "todos" or model_type == "RandomForest" %}
    RandomForest       → ensemble robusto con feature importances.
{% endif %}
{% if use_xgboost or model_type == "XGBoost" %}
    XGBoost            → gradient boosting optimizado. Referencia en Kaggle.
{% endif %}
{% if use_lightgbm or model_type == "LightGBM" %}
    LightGBM           → leaf-wise boosting. Más rápido en datasets grandes.
{% endif %}
{% if use_catboost or model_type == "CatBoost" %}
    CatBoost           → boosting nativo para categoricas. Poco tunning necesario.
{% endif %}
{% else %}
{% if model_type == "todos" %}
    LinearRegression   → modelo base. Rápido e interpretable.
    Ridge              → regresión lineal con regularización L2.
    Lasso              → regularización L1, útil para selección de variables.
{% endif %}
{% if model_type == "todos" or model_type == "KNN" %}
    KNN                → regresión no paramétrica. Sensible a dimensionalidad.
{% endif %}
{% if model_type == "todos" or model_type == "RandomForest" %}
    RandomForest       → ensemble robusto. feature_importances_ disponible.
{% endif %}
{% if use_xgboost or model_type == "XGBoost" %}
    XGBoost            → gradient boosting para regresión.
{% endif %}
{% if use_lightgbm or model_type == "LightGBM" %}
    LightGBM           → leaf-wise boosting para regresión.
{% endif %}
{% if use_catboost or model_type == "CatBoost" %}
    CatBoost           → boosting nativo para categoricas, regresión.
{% endif %}
{% endif %}
    """
    models = {}

{% if use_optuna %}
    _best = {name: _load_best_params(name) for name in [
{% if model_type == "todos" or model_type == "KNN" %}
        "KNN",
{% endif %}
{% if model_type == "todos" or model_type == "LogisticRegression" %}
        "LogisticRegression",
{% endif %}
{% if model_type == "todos" or model_type == "DecisionTree" %}
        "DecisionTree",
{% endif %}
{% if model_type == "todos" or model_type == "RandomForest" %}
        "RandomForest",
{% endif %}
{% if use_xgboost or model_type == "XGBoost" %}
        "XGBoost",
{% endif %}
{% if use_lightgbm or model_type == "LightGBM" %}
        "LightGBM",
{% endif %}
{% if use_catboost or model_type == "CatBoost" %}
        "CatBoost",
{% endif %}
    ]}
{% endif %}

{% if task_type == "clasificacion" %}
{% if model_type == "todos" or model_type == "KNN" %}
{% if use_optuna %}
    models["KNN"] = KNeighborsClassifier(**{
        "n_neighbors": 7, "weights": "distance",
        **_best.get("KNN", {}),
    })
{% else %}
    models["KNN"] = KNeighborsClassifier(n_neighbors=7, weights="distance")
{% endif %}
{% endif %}
{% if model_type == "todos" or model_type == "LogisticRegression" %}
{% if use_optuna %}
    models["LogisticRegression"] = LogisticRegression(**{
        "max_iter": 1000, "class_weight": "balanced", "random_state": 42,
        **_best.get("LogisticRegression", {}),
    })
{% else %}
    models["LogisticRegression"] = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42,
    )
{% endif %}
{% endif %}
{% if model_type == "todos" or model_type == "DecisionTree" %}
{% if use_optuna %}
    models["DecisionTree"] = DecisionTreeClassifier(**{
        "max_depth": 7, "min_samples_leaf": 5, "class_weight": "balanced", "random_state": 42,
        **_best.get("DecisionTree", {}),
    })
{% else %}
    models["DecisionTree"] = DecisionTreeClassifier(
        max_depth=7, min_samples_leaf=5, class_weight="balanced", random_state=42,
    )
{% endif %}
{% endif %}
{% if model_type == "todos" or model_type == "RandomForest" %}
{% if use_optuna %}
    models["RandomForest"] = RandomForestClassifier(**{
        "n_estimators": 200, "max_depth": 10, "max_features": "sqrt",
        "max_samples": 0.8, "class_weight": "balanced", "random_state": 42, "n_jobs": -1,
        **_best.get("RandomForest", {}),
    })
{% else %}
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200, max_depth=10, max_features="sqrt",
        max_samples=0.8, class_weight="balanced", random_state=42, n_jobs=-1,
    )
{% endif %}
{% endif %}
{% if model_type == "todos" or model_type == "SVM" %}
    # SVM con escalado interno en Pipeline para no depender del scaler global
    models["SVM"] = Pipeline([
        ("scaler", _SVMScaler()),
        ("svc", SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=42,
        )),
    ])
{% endif %}
{% else %}
{% if model_type == "todos" %}
    models["LinearRegression"] = LinearRegression()
    models["Ridge"] = Ridge(alpha=1.0)
    models["Lasso"] = Lasso(alpha=0.1, max_iter=2000)
{% endif %}
{% if model_type == "todos" or model_type == "KNN" %}
{% if use_optuna %}
    models["KNN"] = KNeighborsRegressor(**{
        "n_neighbors": 7, "weights": "distance",
        **_best.get("KNN", {}),
    })
{% else %}
    models["KNN"] = KNeighborsRegressor(n_neighbors=7, weights="distance")
{% endif %}
{% endif %}
{% if model_type == "todos" or model_type == "DecisionTree" %}
{% if use_optuna %}
    models["DecisionTree"] = DecisionTreeRegressor(**{
        "max_depth": 7, "min_samples_leaf": 5, "random_state": 42,
        **_best.get("DecisionTree", {}),
    })
{% else %}
    models["DecisionTree"] = DecisionTreeRegressor(
        max_depth=7, min_samples_leaf=5, random_state=42,
    )
{% endif %}
{% endif %}
{% if model_type == "todos" or model_type == "RandomForest" %}
{% if use_optuna %}
    models["RandomForest"] = RandomForestRegressor(**{
        "n_estimators": 200, "max_depth": 10, "max_features": "sqrt",
        "max_samples": 0.8, "random_state": 42, "n_jobs": -1,
        **_best.get("RandomForest", {}),
    })
{% else %}
    models["RandomForest"] = RandomForestRegressor(
        n_estimators=200, max_depth=10, max_features="sqrt",
        max_samples=0.8, random_state=42, n_jobs=-1,
    )
{% endif %}
{% endif %}
{% if model_type == "todos" or model_type == "SVM" %}
    models["SVM"] = _SVMPipeline([
        ("scaler", _SVMScaler()),
        ("svr", SVR(kernel="rbf", C=1.0, gamma="scale")),
    ])
{% endif %}
{% endif %}

{% if use_xgboost or model_type == "XGBoost" %}
{% if task_type == "clasificacion" %}
{% if use_optuna %}
    models["XGBoost"] = XGBClassifier(**{
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "eval_metric": "logloss", "random_state": 42, "n_jobs": -1,
        **_best.get("XGBoost", {}),
    })
{% else %}
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
{% endif %}
{% else %}
{% if use_optuna %}
    models["XGBoost"] = XGBRegressor(**{
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "eval_metric": "rmse", "random_state": 42, "n_jobs": -1,
        **_best.get("XGBoost", {}),
    })
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
{% endif %}

{% if use_lightgbm or model_type == "LightGBM" %}
{% if task_type == "clasificacion" %}
{% if use_optuna %}
    models["LightGBM"] = LGBMClassifier(**{
        "n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "class_weight": "balanced",
        "random_state": 42, "n_jobs": -1, "verbose": -1,
        **_best.get("LightGBM", {}),
    })
{% else %}
    models["LightGBM"] = LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1,
    )
{% endif %}
{% else %}
{% if use_optuna %}
    models["LightGBM"] = LGBMRegressor(**{
        "n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "n_jobs": -1, "verbose": -1,
        **_best.get("LightGBM", {}),
    })
{% else %}
    models["LightGBM"] = LGBMRegressor(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
{% endif %}
{% endif %}
{% endif %}
{% if use_catboost or model_type == "CatBoost" %}
{% if task_type == "clasificacion" %}
{% if use_optuna %}
    models["CatBoost"] = CatBoostClassifier(**{
        "iterations": 300, "depth": 6, "learning_rate": 0.05,
        "l2_leaf_reg": 3.0, "border_count": 254,
        "loss_function": "Logloss", "eval_metric": "Accuracy",
        "random_seed": 42, "verbose": 0,
        **_best.get("CatBoost", {}),
    })
{% else %}
    models["CatBoost"] = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        l2_leaf_reg=3.0, border_count=254,
        loss_function="Logloss", eval_metric="Accuracy",
        random_seed=42, verbose=0,
    )
{% endif %}
{% else %}
{% if use_optuna %}
    models["CatBoost"] = CatBoostRegressor(**{
        "iterations": 300, "depth": 6, "learning_rate": 0.05,
        "l2_leaf_reg": 3.0, "border_count": 254,
        "loss_function": "RMSE",
        "random_seed": 42, "verbose": 0,
        **_best.get("CatBoost", {}),
    })
{% else %}
    models["CatBoost"] = CatBoostRegressor(
        iterations=300, depth=6, learning_rate=0.05,
        l2_leaf_reg=3.0, border_count=254,
        loss_function="RMSE",
        random_seed=42, verbose=0,
    )
{% endif %}
{% endif %}
{% endif %}

    return models


{% if model_type == "todos" or model_type == "KNN" %}

def _find_best_k(X_train, y_train, k_range=range(1, 21)) -> int:
    """Busca el k óptimo para KNN por cross-validation."""
{% if task_type == "clasificacion" %}
    scoring = "f1_weighted"
{% else %}
    scoring = "neg_root_mean_squared_error"
{% endif %}
    scores = {
        k: cross_val_score(
{% if task_type == "clasificacion" %}
            KNeighborsClassifier(n_neighbors=k),
{% else %}
            KNeighborsRegressor(n_neighbors=k),
{% endif %}
            X_train, y_train, cv=5, scoring=scoring,
        ).mean()
        for k in k_range
    }
    best_k = max(scores, key=scores.get)
    print(f"    KNN mejor k={best_k} ({scoring}={scores[best_k]:.3f})")
    return best_k

{% endif %}

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
    en el Model Registry bajo el nombre del modelo. fit, cross-validation,
    el guardado en disco y el registro del modelo ocurren TODOS dentro del
    mismo run — así el run queda con params + métricas + modelo juntos.
{% endif %}

    Returns
    -------
    dict : {nombre_modelo: modelo_entrenado}
    """
    print("--> Entrenando modelos de {{ task_type }}...")
    models = _build_models()

{% if model_type == "todos" or model_type == "KNN" %}
    if tune_knn and "KNN" in models:
        best_k = _find_best_k(X_train, y_train)
{% if task_type == "clasificacion" %}
        models["KNN"] = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
{% else %}
        models["KNN"] = KNeighborsRegressor(n_neighbors=best_k, weights="distance")
{% endif %}

{% endif %}
{% if use_mlflow %}
    mlflow.set_experiment("{{ project_slug }}")
{% endif %}

    trained = {}
    for name, model in models.items():
{% if use_mlflow %}
        mlflow.end_run()  # cerrar cualquier run activo antes de abrir uno nuevo
{% endif %}
        print(f"    [{name}] entrenando...")

        # run_ctx es un run real de MLflow si use_mlflow está activo, o un
        # context manager "vacío" (nullcontext) si no — así el cuerpo de abajo
        # tiene SIEMPRE la misma indentación y fit/CV/guardado/log_model
        # quedan dentro del mismo run, en vez de fuera de él.
{% if use_mlflow %}
        run_ctx = mlflow.start_run(run_name=name)
{% else %}
        run_ctx = nullcontext()
{% endif %}

        with run_ctx:
{% if use_mlflow %}
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


{{ load_models_macro() }}


{% elif ml_type == "no_supervisado" %}
import joblib
{% if cluster_model == "todos" or cluster_model == "KMeans" %}
from sklearn.cluster import KMeans
{% endif %}
{% if cluster_model == "todos" or cluster_model == "AgglomerativeClustering" %}
from sklearn.cluster import AgglomerativeClustering
{% endif %}
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
{% if cluster_model == "todos" or cluster_model == "KMeans" %}
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
{% endif %}

from {{ project_slug }}.utils.paths import MODELS_DIR


# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------

def _build_models(n_clusters: int = 3) -> dict:
    """
    Define los modelos de clustering a ajustar.
{% if cluster_model == "todos" or cluster_model == "KMeans" %}
    KMeans            → rápido y escalable. Asume clusters esféricos.
                        Inicialización k-means++ reduce el riesgo de mínimos locales.
{% endif %}
{% if cluster_model == "todos" or cluster_model == "AgglomerativeClustering" %}
    AgglomerativeClustering → clustering jerárquico aglomerativo (bottom-up).
                               linkage: 'ward' minimiza varianza intraclúster.
{% endif %}
    """
    models = {}
{% if cluster_model == "todos" or cluster_model == "KMeans" %}
    models["KMeans"] = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
    )
{% endif %}
{% if cluster_model == "todos" or cluster_model == "AgglomerativeClustering" %}
    models["AgglomerativeClustering"] = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
    )
{% endif %}
    return models


{% if cluster_model == "todos" or cluster_model == "KMeans" %}
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

{% endif %}

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


{% if cluster_model == "todos" or cluster_model == "KMeans" %}
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

{% endif %}

{{ load_models_macro() }}


{% elif ml_type == "redes_neuronales" %}
import os
import math
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from {{ project_slug }}.utils.paths import MODELS_DIR, RUNS_DIR, ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# Detección de dispositivo (CPU / CUDA)
# ---------------------------------------------------------------------------
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Dispositivo: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM allocated: {round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB")
    print(f"  VRAM reserved:  {round(torch.cuda.memory_reserved(0)/1024**3,  1)} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True   # acelera convoluciones en Ampere+
elif device.type == "mps":
    print("  Apple Silicon GPU (MPS)")

# Semillas para reproducibilidad
_SEED = 42
torch.manual_seed(_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_SEED)


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
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) debe ser múltiplo de nhead ({nhead})")
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
# ResNet — Red Residual con bloques FC
# ---------------------------------------------------------------------------
class _ResidualBlock(nn.Module):
    """Bloque residual para datos tabulares: Linear → BN → ReLU → Linear → BN → skip."""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResNet(nn.Module):
    """
    Red residual con bloques FC apilados, conexiones skip y normalización.

    Cuándo usarla:
      - Más profunda que MLP (10+ capas) sin degradación.
      - Datos tabulares donde MLP se estanca al añadir capas.
      - Aprendizaje de representaciones más ricas que MLP simple.

    Arquitectura:
      Linear(input_dim→hidden_dim) → BN → ReLU → Dropout →
      N × ResidualBlock(hidden_dim) →
      Linear(hidden_dim→output_dim)

    Parameters
    ----------
    input_dim      : número de features de entrada
    output_dim     : número de clases (clasificación) o 1 (regresión)
    hidden_dim     : tamaño interno de los bloques residuales
    num_blocks     : número de bloques residuales apilados
    dropout        : dropout en bloques residuales
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_blocks=4, dropout=0.2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_blocks):
            layers.append(_ResidualBlock(hidden_dim, dropout=dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Autoencoder — Codificación/decodificación no supervisada
# ---------------------------------------------------------------------------
class Autoencoder(nn.Module):
    """
    Autoencoder profundo para representación no supervisada y detección de anomalías.

    Cuándo usarla:
      - Reducción de dimensionalidad no lineal (alternativa a PCA/UMAP).
      - Detección de anomalías (alto error de reconstrucción = outlier).
      - Pre-entrenamiento no supervisado antes de fine-tuning supervisado.
      - Generación de features para modelos downstream.

    Arquitectura:
      Encoder: input_dim → 256 → 128 → 64 → 32 (latent)
      Decoder: 32 → 64 → 128 → 256 → input_dim

    Parameters
    ----------
    input_dim  : número de features de entrada (= salida de reconstrucción)
    output_dim : se ignora (la salida es la reconstrucción = input_dim)
    latent_dim : dimensión del espacio latente
    """
    def __init__(self, input_dim, output_dim=None, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        """Devuelve la representación latente."""
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Fábrica de modelos — selección por copier
# ---------------------------------------------------------------------------
def _build_model(input_dim: int, output_dim: int) -> nn.Module:
    """
    Devuelve la arquitectura seleccionada en nn_model.

    MLP          → datos tabulares sin estructura temporal.
    CNN1D        → patrones locales entre features (señales, sensores).
    LSTM         → dependencias temporales largas.
    GRU          → como LSTM pero más ligero.
    Transformer  → relaciones globales, alta dimensionalidad.
    ResNet       → redes profundas con conexiones residuales.
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
{% elif nn_model == "ResNet" %}
    return ResNet(input_dim=input_dim, output_dim=output_dim,
                  hidden_dim=128, num_blocks=6, dropout=0.2)
{% endif %}


MODEL_NAME = "{{ nn_model }}"


def build_model(input_dim: int, output_dim: int) -> "nn.Module":
    """
    API pública para construir el modelo. Delega en _build_model.
    Necesaria para reconstruir la arquitectura en test_model() de predict_model.py.
    """
    return _build_model(input_dim=input_dim, output_dim=output_dim)


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
    val_split: float = 0.1,
    patience: int = 0,
) -> dict:
    """
    Entrena la red neuronal seleccionada ({{ nn_model }}) con PyTorch.

    Características:
    - CUDA / MPS automático si está disponible
    - TensorBoard: Loss/train y Loss/val por época en runs/
    - Checkpoints periódicos en models/checkpoint-{epoch}.pt
    - Guardado final de pesos en models/{{ nn_model }}.pt
    - Early stopping opcional (patience > 0)

    Parameters
    ----------
    input_dim        : número de features de entrada
    output_dim       : número de clases de salida
    epochs           : épocas máximas de entrenamiento
    batch_size       : tamaño de mini-lote
    lr               : learning rate para AdamW
    checkpoint_every : cada cuántas épocas guardar checkpoint (0 = desactivado)
    val_split        : fracción de X_train reservada para validación (default 0.1)
    patience         : paradas anticipadas si val_loss no mejora N épocas
                       consecutivas. 0 = desactivado.

    Returns
    -------
    dict : {'{{ nn_model }}': modelo_entrenado}
    """
    print(f"--> Entrenando red neuronal: {{ nn_model }}...")

    X_arr = X_train.values if hasattr(X_train, "values") else X_train
    y_arr = y_train.values if hasattr(y_train, "values") else y_train

    # Validation split
    n_val   = max(1, int(len(X_arr) * val_split))
    X_tr, X_val = X_arr[:-n_val], X_arr[-n_val:]
    y_tr, y_val = y_arr[:-n_val], y_arr[-n_val:]

    X_t    = torch.tensor(X_tr, dtype=torch.float32)
    X_v    = torch.tensor(X_val, dtype=torch.float32)
{% if task_type == 'regresion' %}
    y_t    = torch.tensor(y_tr, dtype=torch.float32)
    y_v    = torch.tensor(y_val, dtype=torch.float32)
{% else %}
    y_t    = torch.tensor(y_tr, dtype=torch.long)
    y_v    = torch.tensor(y_val, dtype=torch.long)
{% endif %}
    loader     = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=batch_size, shuffle=False)

    model     = _build_model(input_dim=input_dim, output_dim=output_dim).to(device)
    # model   = torch.compile(model)   # PyTorch >= 2.0: descomentar para mayor velocidad
    print(f"    Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizador configurable ─────────────────────────────────────────
{% set _opt = optimizer_type if optimizer_type is defined else 'AdamW' %}
{% if _opt == 'AdamW' %}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
{% elif _opt == 'Adam' %}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
{% elif _opt == 'SGD' %}
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
{% elif _opt == 'RMSProp' %}
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
{% elif _opt == 'Adagrad' %}
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
{% else %}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
{% endif %}
    print(f"    Optimizador: {{ optimizer_type if optimizer_type is defined else 'AdamW' }}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Función de pérdida configurable ─────────────────────────────────
{% set _loss_raw = nn_loss_fn if nn_loss_fn is defined else 'Auto' %}
{% if _loss_raw == 'Auto' %}
{%   if task_type == 'regresion' %}
    criterion = nn.MSELoss()
    print("    Loss: MSELoss (regresión — Auto)")
{%   else %}
    criterion = nn.CrossEntropyLoss()
    print("    Loss: CrossEntropyLoss (clasificación — Auto)")
{%   endif %}
{% elif _loss_raw == 'MSELoss' %}
    criterion = nn.MSELoss()
    print("    Loss: MSELoss")
{% elif _loss_raw == 'L1Loss' %}
    criterion = nn.L1Loss()
    print("    Loss: L1Loss")
{% elif _loss_raw == 'BCEWithLogitsLoss' %}
    criterion = nn.BCEWithLogitsLoss()
    print("    Loss: BCEWithLogitsLoss")
{% else %}
    criterion = nn.CrossEntropyLoss()
    print("    Loss: CrossEntropyLoss")
{% endif %}

    # ── TorchMetrics ─────────────────────────────────────────────────────
    try:
        from torchmetrics import MetricCollection
{% if task_type == 'clasificacion' %}
        from torchmetrics.classification import (
            MulticlassAccuracy, MulticlassF1Score,
            MulticlassPrecision, MulticlassRecall,
        )
        _metrics_train = MetricCollection({
            "acc":       MulticlassAccuracy(num_classes=output_dim),
            "f1":        MulticlassF1Score(num_classes=output_dim, average="macro"),
            "precision": MulticlassPrecision(num_classes=output_dim, average="macro"),
            "recall":    MulticlassRecall(num_classes=output_dim, average="macro"),
        }).to(device)
        _metrics_val = _metrics_train.clone().to(device)
{% else %}
        from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
        _metrics_train = MetricCollection({
            "mae":  MeanAbsoluteError(),
            "rmse": MeanSquaredError(squared=False),
            "r2":   R2Score(),
        }).to(device)
        _metrics_val = _metrics_train.clone().to(device)
{% endif %}
        _use_metrics = True
        print("    TorchMetrics: activo")
    except ImportError:
        _use_metrics = False
        print("    TorchMetrics: no disponible (pip install torchmetrics)")

    tb        = SummaryWriter(log_dir=str(RUNS_DIR))
    print(f"    TensorBoard: tensorboard --logdir {RUNS_DIR}")
    if patience > 0:
        print(f"    Early stopping activo (patience={patience})")

    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # ── Entrenamiento ────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        if _use_metrics:
            _metrics_train.reset()
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds_b = model(X_b)
{% if task_type == 'regresion' %}
            loss = criterion(preds_b.squeeze(-1), y_b)
{% else %}
            loss = criterion(preds_b, y_b)
{% endif %}
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            if _use_metrics:
{% if task_type == 'clasificacion' %}
                _metrics_train.update(preds_b.argmax(dim=-1), y_b)
{% else %}
                _metrics_train.update(preds_b.squeeze(), y_b.float())
{% endif %}

        scheduler.step()
        avg_loss = total_loss / len(loader)

        # ── Validación ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        if _use_metrics:
            _metrics_val.reset()
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds_b = model(X_b)
{% if task_type == 'regresion' %}
                val_loss += criterion(preds_b.squeeze(-1), y_b).item()
{% else %}
                val_loss += criterion(preds_b, y_b).item()
{% endif %}
                if _use_metrics:
{% if task_type == 'clasificacion' %}
                    _metrics_val.update(preds_b.argmax(dim=-1), y_b)
{% else %}
                    _metrics_val.update(preds_b.squeeze(), y_b.float())
{% endif %}
        avg_val_loss = val_loss / len(val_loader)

        tb.add_scalar("Loss/train", avg_loss,     epoch)
        tb.add_scalar("Loss/val",   avg_val_loss, epoch)
        tb.add_scalar("LR",         scheduler.get_last_lr()[0], epoch)
        if _use_metrics:
            for k, v in _metrics_train.compute().items():
                tb.add_scalar(f"Train/{k}", v.item(), epoch)
            for k, v in _metrics_val.compute().items():
                tb.add_scalar(f"Val/{k}", v.item(), epoch)

        if (epoch + 1) % 10 == 0:
            metrics_str = ""
            if _use_metrics:
                m = _metrics_val.compute()
                metrics_str = "  ".join(f"{k}={v.item():.3f}" for k, v in m.items())
            print(f"    Epoch {epoch+1}/{epochs} — "
                  f"train: {avg_loss:.4f}  val: {avg_val_loss:.4f}  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}"
                  + (f"  [{metrics_str}]" if metrics_str else ""))

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss":                 avg_loss,
            }, MODELS_DIR / f"checkpoint-{epoch+1}.pt")

        # ── Early stopping ───────────────────────────────────────────────
        if patience > 0:
            if avg_val_loss < best_val_loss:
                best_val_loss    = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), MODELS_DIR / f"{MODEL_NAME}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping en epoch {epoch+1} "
                          f"(val_loss no mejora desde hace {patience} epochs)")
                    # Restaurar los mejores pesos
                    model.load_state_dict(torch.load(
                        MODELS_DIR / f"{MODEL_NAME}_best.pt",
                        map_location=device, weights_only=True,
                    ))
                    break

    tb.close()
    out_path = MODELS_DIR / f"{MODEL_NAME}_final.pt"
    torch.save(model.state_dict(), out_path)
    print(f"    Guardado: {out_path}")

    # Guardar output_dim para que la API sepa cuántas clases/dimensiones usar al cargar
    joblib.dump(output_dim, ARTIFACTS_DIR / "output_dim.joblib")

    return {MODEL_NAME: model}


def load_model(input_dim: int, output_dim: int, weights_path: str = None):
    """Carga pesos finales y devuelve el modelo en modo eval."""
    if weights_path is None:
        weights_path = f"{MODEL_NAME}_final.pt"
    path  = MODELS_DIR / weights_path if not str(weights_path).startswith("/") else weights_path
    model = _build_model(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    print(f"    Modelo cargado desde {path}")
    return model


def load_checkpoint(input_dim: int, output_dim: int, checkpoint_path: str):
    """Carga un checkpoint para continuar el entrenamiento."""
    # Necesario weights_only=False: el checkpoint incluye optimizer_state_dict
    # y metadatos (epoch, loss) que no se pueden cargar con weights_only=True
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model      = _build_model(input_dim=input_dim, output_dim=output_dim).to(device)
{% set _opt2 = optimizer_type if optimizer_type is defined else 'AdamW' %}
{% if _opt2 == 'Adam' %}
    optimizer  = torch.optim.Adam(model.parameters())
{% elif _opt2 == 'SGD' %}
    optimizer  = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
{% elif _opt2 == 'RMSProp' %}
    optimizer  = torch.optim.RMSprop(model.parameters())
{% elif _opt2 == 'Adagrad' %}
    optimizer  = torch.optim.Adagrad(model.parameters())
{% else %}
    optimizer  = torch.optim.AdamW(model.parameters())
{% endif %}
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_inicio = checkpoint["epoch"]
    print(f"    Checkpoint cargado: epoch {epoch_inicio}")
    return model, optimizer, epoch_inicio


{% elif ml_type == "hibrido" %}
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
{% if task_type == "regresion" %}
from sklearn.ensemble import RandomForestRegressor
{% endif %}
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
{% if task_type == "regresion" %}
from sklearn.neighbors import KNeighborsRegressor
{% endif %}
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

{% if use_xgboost or model_type == "XGBoost" %}
from xgboost import XGBClassifier
{% endif %}
{% if use_lightgbm or model_type == "LightGBM" %}
from lightgbm import LGBMClassifier
{% endif %}
{% if use_catboost or model_type == "CatBoost" %}
{% if task_type == "clasificacion" %}
from catboost import CatBoostClassifier
{% else %}
from catboost import CatBoostRegressor
{% endif %}
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
{% if task_type == "clasificacion" %}
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
{% else %}
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            max_features="sqrt",
            max_samples=0.8,
            random_state=42,
            n_jobs=-1,
        ),
        "KNN": KNeighborsRegressor(n_neighbors=7, weights="distance"),
{% endif %}
    }

    if strategy in ("pca_clf", "umap_clf"):
{% if task_type == "clasificacion" %}
        base["SVM_RBF"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", probability=True, random_state=42,
            )),
        ])
{% else %}
        from sklearn.svm import SVR
        base["SVR_RBF"] = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR(kernel="rbf", C=1.0, gamma="scale")),
        ])
{% endif %}

    if strategy in ("kmeans_features", "iso_feature"):
{% if task_type == "clasificacion" %}
        base["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
{% else %}
        from sklearn.ensemble import GradientBoostingRegressor
        base["GradientBoosting"] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
{% endif %}

{% if use_xgboost or model_type == "XGBoost" %}
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
        random_state=42,
        n_jobs=-1,
    )
{% endif %}

{% if use_lightgbm or model_type == "LightGBM" %}
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

{% if use_catboost or model_type == "CatBoost" %}
    # CatBoost: manejo nativo de categoricas, no requiere encoders externos.
{% if task_type == "clasificacion" %}
    base["CatBoost"] = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        border_count=254,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=0,
    )
{% else %}
    base["CatBoost"] = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        border_count=254,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
    )
{% endif %}
{% endif %}

    return base


{% if model_type == "todos" or model_type == "KNN" %}
def _find_best_k(X_train, y_train, k_range=range(1, 21)) -> int:
    """
    Busca el mejor k para KNN por validación cruzada (5-fold).
    Devuelve el k con mayor métrica media, priorizando k más alto en empates.
    """
    print("    Buscando mejor k para KNN...")
    best_k, best_score = 1, -float("inf")
    for k in k_range:
{% if task_type == "clasificacion" %}
        knn   = KNeighborsClassifier(n_neighbors=k, weights="distance")
        score = cross_val_score(knn, X_train, y_train, cv=5, scoring="f1_weighted").mean()
{% else %}
        knn   = KNeighborsRegressor(n_neighbors=k, weights="distance")
        score = cross_val_score(knn, X_train, y_train, cv=5,
                                scoring="neg_root_mean_squared_error").mean()
{% endif %}
        if score >= best_score:
            best_k, best_score = k, score
{% if task_type == "clasificacion" %}
    print(f"    Mejor k = {best_k}  (F1_weighted CV = {best_score:.3f})")
{% else %}
    print(f"    Mejor k = {best_k}  (RMSE CV = {-best_score:.4f})")
{% endif %}
    return best_k

{% endif %}

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

{% if model_type == "todos" or model_type == "KNN" %}
    if tune_knn and "KNN" in models:
        best_k = _find_best_k(X_train, y_train)
{% if task_type == "clasificacion" %}
        models["KNN"] = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
{% else %}
        models["KNN"] = KNeighborsRegressor(n_neighbors=best_k, weights="distance")
{% endif %}

{% endif %}
    trained = {}
    for name, model in models.items():
        print(f"    [{name}] entrenando...")
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

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"      Guardado → {name}.joblib")
        trained[name] = model

    print(f"--> {len(trained)} modelos guardados en {MODELS_DIR}")
    return trained


{{ load_models_macro() }}
{% endif %}