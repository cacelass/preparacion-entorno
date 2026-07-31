{% if ml_type == 'supervisado' or ml_type == 'hibrido' %}
"""
predict_model.py — Evaluación de modelos {{ ml_type }}.
Tarea: {{ task_type }}
"""
from typing import Any

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

{% if task_type == "clasificacion" %}
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)
{% else %}
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
)
{% endif %}

{% if use_mlflow %}
import mlflow
import mlflow.sklearn
{% endif %}

from {{ project_slug }}.utils.paths import FIGURES_DIR, MODELS_DIR, REPORTS_DIR

{% if task_type == "clasificacion" %}
# Umbral de decisión. Bajar (e.g. 0.3) aumenta recall de clase minoritaria.
DECISION_THRESHOLD: float = 0.5
{% endif %}


def evaluate_models(
    models: dict[str, Any],
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
{% if task_type == "clasificacion" %}
    threshold: float = DECISION_THRESHOLD,
{% endif %}
) -> pd.DataFrame:
    """
    Evalúa todos los modelos sobre train y test.

{% if task_type == "clasificacion" %}
    Métricas: Accuracy, F1 weighted, Precision, Recall, ROC-AUC (binario).
    Genera matrices de confusión en figures/.
{% else %}
    Métricas: RMSE, MAE, MAPE, R².
    Genera gráfico real vs predicho y residuos en figures/.
{% endif %}

    Returns
    -------
    pd.DataFrame ordenado por métrica principal.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
{% if task_type == "clasificacion" %}
    print(f"\n{'='*60}\n  Evaluación — clasificacion (umbral={threshold})\n{'='*60}")
{% else %}
    print(f"\n{'='*60}\n  Evaluación — regresion\n{'='*60}")
{% endif %}

{% if use_mlflow %}
    mlflow.set_experiment("{{ project_slug }}")
{% endif %}

    results = []
    for name, model in models.items():
        print(f"\n--- {name} ---")
{% if use_mlflow %}
        mlflow.end_run()  # cerrar run activo antes de abrir uno nuevo
{% endif %}

{% if task_type == "clasificacion" %}
        n_classes = len(np.unique(y_test))
        if threshold != 0.5 and hasattr(model, "predict_proba") and n_classes == 2:
            proba_test   = model.predict_proba(X_test)[:, 1]
            y_pred_test  = (proba_test >= threshold).astype(int)
            proba_train  = model.predict_proba(X_train)[:, 1]
            y_pred_train = (proba_train >= threshold).astype(int)
        else:
            y_pred_test  = model.predict(X_test)
            y_pred_train = model.predict(X_train)

        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test  = accuracy_score(y_test,  y_pred_test)
        f1_train  = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)
        f1_test   = f1_score(y_test,  y_pred_test,  average="weighted", zero_division=0)
        prec_test = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
        rec_test  = recall_score(y_test,  y_pred_test,  average="weighted", zero_division=0)
        roc_auc   = None
        if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        print(f"  Accuracy  → train: {acc_train:.3f} | test: {acc_test:.3f}")
        print(f"  F1 (w)    → train: {f1_train:.3f}  | test: {f1_test:.3f}")
        print(f"  Precision → {prec_test:.3f}  | Recall → {rec_test:.3f}")
        if roc_auc is not None:
            print(f"  ROC-AUC   → {roc_auc:.3f}")
        print()
        print(classification_report(y_test, y_pred_test, zero_division=0))
        _plot_confusion_matrix(y_test, y_pred_test, name)

        row = {
            "Modelo":    name,
            "Acc_train": round(acc_train, 4), "Acc_test":  round(acc_test,  4),
            "F1_train":  round(f1_train,  4), "F1_test":   round(f1_test,   4),
            "Prec_test": round(prec_test, 4), "Rec_test":  round(rec_test,  4),
        }
        if roc_auc is not None:
            row["ROC_AUC"] = round(roc_auc, 4)

{% if use_mlflow %}
        with mlflow.start_run(run_name=f"{name}_eval"):
            mlflow.log_metrics({
                "acc_train": acc_train, "acc_test": acc_test,
                "f1_train":  f1_train,  "f1_test":  f1_test,
                "prec_test": prec_test, "rec_test":  rec_test,
            })
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_artifact(str(FIGURES_DIR / f"cm_{name}.png"))
{% endif %}

{% else %}
        y_pred_test  = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        rmse_train = mean_squared_error(y_train, y_pred_train) ** 0.5
        rmse_test  = mean_squared_error(y_test,  y_pred_test)  ** 0.5
        mae_test   = mean_absolute_error(y_test, y_pred_test)
        mape_test  = mean_absolute_percentage_error(y_test, y_pred_test)
        r2_train   = r2_score(y_train, y_pred_train)
        r2_test    = r2_score(y_test,  y_pred_test)

        print(f"  RMSE → train: {rmse_train:.4f} | test: {rmse_test:.4f}")
        print(f"  MAE  → {mae_test:.4f}")
        print(f"  MAPE → {mape_test:.4f}")
        print(f"  R²   → train: {r2_train:.4f} | test: {r2_test:.4f}")

        _plot_real_vs_pred(y_test, y_pred_test, name)
        _plot_residuals(y_test, y_pred_test, name)

        row = {
            "Modelo":     name,
            "RMSE_train": round(rmse_train, 4), "RMSE_test": round(rmse_test, 4),
            "MAE_test":   round(mae_test,   4), "MAPE_test": round(mape_test, 4),
            "R2_train":   round(r2_train,   4), "R2_test":   round(r2_test,   4),
        }

{% if use_mlflow %}
        with mlflow.start_run(run_name=f"{name}_eval"):
            mlflow.log_metrics({
                "rmse_train": rmse_train, "rmse_test": rmse_test,
                "mae_test":   mae_test,   "mape_test": mape_test,
                "r2_train":   r2_train,   "r2_test":   r2_test,
            })
            mlflow.log_artifact(str(FIGURES_DIR / f"real_vs_pred_{name}.png"))
            mlflow.log_artifact(str(FIGURES_DIR / f"residuals_{name}.png"))
{% endif %}

{% endif %}
        results.append(row)

{% if task_type == "clasificacion" %}
    df_results = pd.DataFrame(results).sort_values("Acc_test", ascending=False)
{% else %}
    df_results = pd.DataFrame(results).sort_values("RMSE_test", ascending=True)
{% endif %}
    out_csv = REPORTS_DIR / "resultados_modelos.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n{'='*60}\n  Resumen:\n{'='*60}")
    print(df_results.to_string(index=False))
    print(f"\n  Guardado → {out_csv}")
    return df_results


{% if task_type == "clasificacion" %}
def _plot_confusion_matrix(y_true: Any, y_pred: Any, model_name: str) -> None:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de confusion — {model_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cm_{model_name}.png", dpi=150)
    plt.close(fig)
    print(f"    cm_{model_name}.png guardado")

{% else %}
def _plot_real_vs_pred(y_true, y_pred, model_name: str) -> None:
    """Scatter real vs predicho con línea de referencia perfecta."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, label="muestras")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="prediccion perfecta")
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicho")
    ax.set_title(f"{model_name} — Real vs Predicho")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"real_vs_pred_{model_name}.png", dpi=150)
    plt.close(fig)
    print(f"    real_vs_pred_{model_name}.png guardado")


def _plot_residuals(y_true, y_pred, model_name: str) -> None:
    """Gráfico de residuos — detecta heterocedasticidad y outliers."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=20)
    axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[0].set_xlabel("Predicho")
    axes[0].set_ylabel("Residuo")
    axes[0].set_title("Residuos vs Predicho")

    axes[1].hist(residuals, bins=30, edgecolor="white")
    axes[1].set_xlabel("Residuo")
    axes[1].set_title("Distribucion de residuos")

    fig.suptitle(f"{model_name} — Analisis de residuos", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"residuals_{model_name}.png", dpi=150)
    plt.close(fig)
    print(f"    residuals_{model_name}.png guardado")
{% endif %}


{% if use_shap %}
import shap

def explain_models(
    models: dict,
    X_train,
    feature_names: list = None,
    max_display: int = 15,
    kernel_background: int = 50,
    kernel_max_samples: int = 100,
) -> None:
    """
    Genera explicaciones SHAP para cada modelo entrenado.

    Por cada modelo produce dos gráficas en reports/figures/:
      - shap_bar_{nombre}.png   → importancia global media (resumen ejecutivo)
      - shap_beeswarm_{nombre}.png → distribución + dirección del impacto

    Selección de explainer por tipo de modelo:
      TreeExplainer   → RandomForest, DecisionTree, XGBoost, LightGBM  (exacto, rápido)
      LinearExplainer → LogisticRegression, Ridge, Lasso                (exacto, rápido)
      KernelExplainer → KNN y otros sin soporte nativo                  (aprox., lento)

    Parameters
    ----------
    models            : dict nombre→modelo (salida de train_models)
    X_train           : datos de entrenamiento ya preprocesados
    feature_names     : lista de nombres de features (opcional)
    max_display       : nº máximo de features a mostrar en las gráficas
    kernel_background : nº de muestras de fondo para KernelExplainer
    kernel_max_samples: nº máximo de filas a explicar con KernelExplainer
    """
    import warnings
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if hasattr(X_train, "values"):
        X_arr = X_train.values
        feat_names = feature_names or list(X_train.columns)
    else:
        X_arr = X_train
        feat_names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]

    print(f"\n{'='*60}\n  SHAP — Explicabilidad de modelos\n{'='*60}")

    for name, model in models.items():
        print(f"\n--- {name} ---")
        try:
            shap_values, X_explain = _compute_shap(
                model, X_arr, feat_names,
                kernel_background=kernel_background,
                kernel_max_samples=kernel_max_samples,
            )
        except Exception as exc:
            print(f"    ⚠ SHAP no disponible para {name}: {exc}")
            continue

        _shap_bar(shap_values, X_explain, feat_names, name, max_display)
        _shap_beeswarm(shap_values, X_explain, feat_names, name, max_display)

    print(f"\n  Gráficas SHAP guardadas en {FIGURES_DIR}")


def _compute_shap(model, X_arr, feat_names, kernel_background, kernel_max_samples):
    """Selecciona el explainer adecuado y devuelve (shap_values, X_explain)."""
    module = type(model).__module__

    is_tree = (
        hasattr(model, "estimators_")       # RandomForest
        or hasattr(model, "tree_")          # DecisionTree
        or "xgboost" in module
        or "lightgbm" in module
    )
    is_linear = hasattr(model, "coef_")     # LogisticRegression, Ridge, Lasso

    if is_tree:
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_arr)
        X_explain  = X_arr

    elif is_linear:
        explainer  = shap.LinearExplainer(model, X_arr)
        shap_vals  = explainer.shap_values(X_arr)
        X_explain  = X_arr

    else:
        # KNN y otros → KernelExplainer (lento)
        n_bg = min(kernel_background, len(X_arr))
        bg   = shap.sample(X_arr, n_bg)
        fn   = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        explainer  = shap.KernelExplainer(fn, bg)
        n_exp      = min(kernel_max_samples, len(X_arr))
        X_explain  = X_arr[:n_exp]
        print(f"    KernelExplainer: {n_bg} muestras fondo, "
              f"{n_exp} muestras a explicar (puede tardar...)")
        shap_vals = explainer.shap_values(X_explain)

    # RandomForest y multiclase devuelven lista — tomamos clase positiva (binario)
    # o la media absoluta entre clases (multiclase)
    if isinstance(shap_vals, list):
        if len(shap_vals) == 2:
            shap_vals = shap_vals[1]            # clase positiva binaria
        else:
            import numpy as _np
            shap_vals = _np.abs(_np.stack(shap_vals)).mean(axis=0)  # media multiclase

    return shap_vals, X_explain


def _shap_bar(shap_values, X_explain, feat_names, model_name, max_display):
    """Barra de importancia global media (|SHAP|)."""
    fig, ax = plt.subplots(figsize=(9, max(4, min(max_display, len(feat_names)) * 0.4 + 1)))
    shap.summary_plot(
        shap_values, X_explain,
        feature_names=feat_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP — Importancia global ({model_name})", fontsize=12, pad=10)
    plt.tight_layout()
    path = FIGURES_DIR / f"shap_bar_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    shap_bar_{model_name}.png guardado")


def _shap_beeswarm(shap_values, X_explain, feat_names, model_name, max_display):
    """Beeswarm: distribución de valores SHAP por feature (dirección + magnitud)."""
    fig, ax = plt.subplots(figsize=(10, max(4, min(max_display, len(feat_names)) * 0.4 + 1)))
    shap.summary_plot(
        shap_values, X_explain,
        feature_names=feat_names,
        plot_type="dot",        # beeswarm
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP — Beeswarm ({model_name})", fontsize=12, pad=10)
    plt.tight_layout()
    path = FIGURES_DIR / f"shap_beeswarm_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    shap_beeswarm_{model_name}.png guardado")

{% endif %}

def predict_new(model_name: str, X_new: Any) -> np.ndarray:
    """Carga un modelo y predice sobre nuevas muestras (ya preprocesadas)."""
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    return np.asarray(joblib.load(path).predict(X_new))


{% if task_type == "clasificacion" %}
def predict_proba_new(model_name: str, X_new: Any) -> np.ndarray:
    """Carga un modelo y devuelve probabilidades de clase."""
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"{model_name} no soporta predict_proba")
    return np.asarray(model.predict_proba(X_new))


# ---------------------------------------------------------------------------
# Búsqueda automática de umbral óptimo por F1
# ---------------------------------------------------------------------------
# ACTIVAR solo en clasificación BINARIA (dos clases).
# Si tu problema es multiclase, mantén esta función comentada y usa
# DECISION_THRESHOLD = 0.5 (o ajústalo manualmente).
#
# Uso típico en train_model.py, tras entrenar el mejor modelo:
#
#   from sklearn.metrics import precision_recall_curve
#   import numpy as np
#   from {{ project_slug }}.models.predict_model import find_best_threshold
#   from {{ project_slug }}.utils.paths import ARTIFACTS_DIR
#   import joblib
#
#   proba_val = best_model.predict_proba(X_val)[:, 1]
#   threshold, f1 = find_best_threshold(y_val, proba_val)
#   print(f"Umbral óptimo: {threshold:.4f}  |  F1: {f1:.4f}")
#   joblib.dump(threshold, ARTIFACTS_DIR / "threshold.joblib")
#
# En main.py, carga el umbral antes de evaluar:
#
#   threshold = joblib.load(ARTIFACTS_DIR / "threshold.joblib")
#   evaluate_models(models, X_train, y_train, X_test, y_test, threshold=threshold)
#
# def find_best_threshold(y_true, y_proba):
#     """
#     Calcula el umbral de decisión que maximiza el F1-score binario.
#
#     Parameters
#     ----------
#     y_true  : array-like con etiquetas reales (0/1)
#     y_proba : array-like con probabilidades de clase positiva
#
#     Returns
#     -------
#     best_threshold : float — umbral que maximiza F1
#     best_f1        : float — F1 en ese umbral
#     """
#     from sklearn.metrics import precision_recall_curve
#     precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
#     # f1_scores tiene longitud N; thresholds tiene longitud N-1 → recortamos
#     f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
#     best_idx       = np.nanargmax(f1_scores[:-1])
#     best_threshold = thresholds[best_idx]
#     best_f1        = f1_scores[best_idx]
#     return best_threshold, best_f1


{% endif %}{# end task_type == clasificacion #}


# ---------------------------------------------------------------------------
# Modo prueba: carga artefactos y evalúa sobre datos nuevos introducidos
# por el usuario en tiempo de ejecución.
# ---------------------------------------------------------------------------
def try_model() -> None:
    """
    Modo interactivo: introduce tus datos y el modelo predice el resultado.

    Flujo:
      1. Lista los modelos disponibles en models/ y pide elegir uno.
      2. Carga el modelo y los artefactos de preprocesado (scaler, PCA,
         encoders, threshold) guardados durante el entrenamiento.
      3. Pide al usuario los valores de cada feature por consola.
      4. Preprocesa la entrada con process_input() de build_features.
      5. Imprime la predicción (y probabilidad si está disponible).

    Requisitos previos:
      - Haber ejecutado run_full_pipeline() al menos una vez para que
        existan los joblibs en artifacts/ y los modelos en models/.
    """
    from {{ project_slug }}.features.build_features import process_input
    from {{ project_slug }}.utils.paths import ARTIFACTS_DIR, PROCESSED_DATA_DIR
    import pandas as pd

    # ── 1. Elegir modelo ────────────────────────────────────────────────
    available = sorted(MODELS_DIR.glob("*.joblib"))
    if not available:
        print("No hay modelos entrenados en models/. Ejecuta primero la opción 0.")
        return

    print("\nModelos disponibles:")
    for i, p in enumerate(available):
        print(f"  [{i}] {p.stem}")
    try:
        idx = int(input("Elige modelo (número): "))
        model = joblib.load(available[idx])
        model_name = available[idx].stem
    except (ValueError, IndexError):
        print("Selección inválida.")
        return

    # ── 2. Cargar nombres de features ──────────────────────────────────
    feat_path = ARTIFACTS_DIR / "feature_names.joblib"
    if feat_path.exists():
        feature_names = joblib.load(feat_path)
    else:
        x_train_path = PROCESSED_DATA_DIR / "X_train.csv"
        if x_train_path.exists():
            feature_names = pd.read_csv(x_train_path).columns.tolist()
        else:
            print("No se encontró feature_names.joblib ni X_train.csv. Ejecuta primero run_full_pipeline().")
            return

    # ── 3. Pedir valores al usuario ────────────────────────────────────
    print(f"\nIntroduce los valores para el modelo '{model_name}':")
    print("  (deja en blanco para usar 0 como valor por defecto)\n")
    row: dict[str, str | float] = {}
    for feat in feature_names:
        raw = input(f"  {feat}: ").strip()
        try:
            row[feat] = float(raw) if raw else 0.0
        except ValueError:
            row[feat] = raw if raw else 0.0

    df_input = pd.DataFrame([row])

    # ── 4. Preprocesar ─────────────────────────────────────────────────
    try:
        X_new = process_input(df_input)
    except Exception as e:
        print(f"\nError en preprocesado: {e}")
        return

    # ── 6. Predecir ────────────────────────────────────────────────────
    print(f"\n{'='*50}")
{% if task_type == "clasificacion" %}
    # ── 5. Cargar umbral (si existe) ───────────────────────────────────
    threshold_path = ARTIFACTS_DIR / "threshold.joblib"
    threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.5

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[0]
        pred  = int(proba[1] >= threshold)
        print(f"  Modelo     : {model_name}")
        print(f"  Umbral     : {threshold:.4f}")
        print(f"  Predicción : {pred}")
        print(f"  Probabilidades: {dict(enumerate(proba.round(4).tolist()))}")
        # Decodificar etiqueta original si existe target_encoder
        te_path = ARTIFACTS_DIR / "target_encoder.joblib"
        if te_path.exists():
            te = joblib.load(te_path)
            print(f"  Clase      : {te.inverse_transform([pred])[0]}")
    else:
        pred = model.predict(X_new)[0]
        print(f"  Modelo     : {model_name}")
        print(f"  Predicción : {int(pred)}")
{% else %}
    pred = model.predict(X_new)[0]
    print(f"  Modelo     : {model_name}")
    print(f"  Valor pred.: {float(pred):.4f}")
{% endif %}
    print(f"{'='*50}\n")


{% elif ml_type == 'no_supervisado' %}
from typing import Any
"""
predict_model.py — Evaluación de modelos no supervisados (clustering).
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

from {{ project_slug }}.utils.paths import MODELS_DIR, FIGURES_DIR, REPORTS_DIR


def evaluate_models(models: dict[str, Any], X: Any) -> pd.DataFrame:
    """
    Evalúa modelos de clustering con métricas internas:
      - Silhouette Score  → [-1, +1]; más cercano a +1 es mejor.
      - Davies-Bouldin    → [0, ∞);  menor es mejor.
      - Calinski-Harabasz → (0, ∞);  mayor es mejor.

    También genera proyecciones PCA-2D por modelo.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n  Evaluación de clustering\n{'='*60}")
    results = []

    for name, model in models.items():
        print(f"\n--- {name} ---")
        labels   = model.labels_ if hasattr(model, "labels_") else model.predict(X)
        n_unique = len(set(labels)) - (1 if -1 in labels else 0)

        if n_unique < 2:
            print(f"  ⚠ Solo {n_unique} cluster(s) — métricas no aplicables.")
            continue

        sil = silhouette_score(X, labels, metric="euclidean")
        db  = davies_bouldin_score(X, labels)
        ch  = calinski_harabasz_score(X, labels)

        print(f"  Clusters        : {n_unique}")
        print(f"  Silhouette      : {sil:.4f}  (mejor → +1)")
        print(f"  Davies-Bouldin  : {db:.4f}   (mejor → 0)")
        print(f"  Calinski-Harabasz: {ch:.1f}  (mejor → mayor)")
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Distribución    : {dict(zip(unique.tolist(), counts.tolist()))}")

        _plot_clusters_pca(X, labels, name)
        results.append({
            "Modelo": name, "N_clusters": n_unique,
            "Silhouette": round(sil, 4),
            "Davies_Bouldin": round(db,  4),
            "Calinski_Harabasz": round(ch, 2),
        })

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results.to_csv(REPORTS_DIR / "resultados_clustering.csv", index=False)
        print(f"\n{'='*60}\n  Resumen:\n{'='*60}")
        print(df_results.to_string(index=False))
    return df_results


def plot_dendrogram(X: Any, method: str = "ward", color_threshold: float | None = None) -> None:
    """
    Dendrograma de clustering jerárquico.
    Cómo leer el corte: busca el tramo vertical más largo sin horizontales
    cruzándolo, traza una horizontal a mitad, cuenta las ramas → ese es k.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    linked = sch.linkage(X, method=method)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_title(f"Dendrograma — linkage='{method}'", fontsize=14)
    ax.set_xlabel("Muestras")
    ax.set_ylabel("Distancia euclidiana")
    sch.dendrogram(linked, ax=ax, no_labels=len(X) > 50, leaf_rotation=90, leaf_font_size=8)
    if color_threshold is not None:
        ax.axhline(y=color_threshold, color="red", lw=2, linestyle="--",
                   label=f"Corte en {color_threshold:.1f}")
        ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dendrogram.png", dpi=150)
    plt.close(fig)
    print(f"    dendrogram.png guardado (method='{method}')")


def _plot_clusters_pca(X: Any, labels: Any, model_name: str) -> None:
    pca  = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=15, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Cluster")
    ax.set_title(
        f"{model_name} — PCA 2D\n"
        f"Varianza: PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%}", fontsize=12,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"clusters_{model_name}_pca.png", dpi=150)
    plt.close(fig)
    print(f"    clusters_{model_name}_pca.png guardado")


def load_models(model_names: list[str] | None = None) -> dict[str, Any]:
    if model_names is None:
        model_names = [p.stem for p in MODELS_DIR.glob("*.joblib")]
    models = {}
    for name in model_names:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
        else:
            print(f"    ⚠ No encontrado: {path}")
    return models


# ---------------------------------------------------------------------------
# Modo prueba: asigna cluster a una muestra nueva introducida por el usuario.
# ---------------------------------------------------------------------------
def try_model() -> None:
    """
    Modo interactivo de prueba para modelos de clustering.

    Flujo:
      1. Lista los modelos disponibles en models/ y pide elegir uno.
      2. Carga el scaler guardado durante el entrenamiento.
      3. Pide al usuario los valores de cada feature por consola.
      4. Escala la entrada y predice el cluster asignado.
      5. Imprime el cluster y, si el modelo es KMeans, las distancias
         a cada centroide para dar contexto de confianza.

    Requisitos previos:
      - Haber ejecutado run_full_pipeline() al menos una vez para que
        existan los joblibs en artifacts/ y los modelos en models/.
    """
    from {{ project_slug }}.utils.paths import ARTIFACTS_DIR
    import pandas as pd

    # ── 1. Elegir modelo ────────────────────────────────────────────────
    available = sorted(MODELS_DIR.glob("*.joblib"))
    if not available:
        print("No hay modelos entrenados en models/. Ejecuta primero la opción 0.")
        return

    print("\nModelos disponibles:")
    for i, p in enumerate(available):
        print(f"  [{i}] {p.stem}")
    try:
        idx = int(input("Elige modelo (número): "))
        model = joblib.load(available[idx])
        model_name = available[idx].stem
    except (ValueError, IndexError):
        print("Selección inválida.")
        return

    # ── 2. Cargar scaler y nombres de features ──────────────────────────
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    if not scaler_path.exists():
        print("No se encontró scaler.joblib. Ejecuta primero run_full_pipeline().")
        return
    scaler   = joblib.load(scaler_path)
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib") if (ARTIFACTS_DIR / "encoders.joblib").exists() else {}

    feat_path = ARTIFACTS_DIR / "feature_names.joblib"
    if feat_path.exists():
        feature_names = joblib.load(feat_path)
    else:
        feature_names = [f"feature_{i}" for i in range(scaler.n_features_in_)]

    # ── 3. Pedir valores al usuario ────────────────────────────────────
    print(f"\nIntroduce los valores para '{model_name}':")
    print("  (deja en blanco para usar 0 como valor por defecto)\n")
    row = {}
    for feat in feature_names:
        raw = input(f"  {feat}: ").strip()
        try:
            row[feat] = float(raw) if raw else 0.0
        except ValueError:
            row[feat] = 0.0

    X_new_df = pd.DataFrame([row])

    # Aplicar encoders del entrenamiento a columnas categóricas
    for col, le_col in encoders.items():
        if col in X_new_df.columns:
            try:
                X_new_df[col] = le_col.transform(X_new_df[col].astype(str))
            except ValueError:
                X_new_df[col] = 0

    # Alinear columnas con feature_names del entrenamiento
    if "feature_names" in locals() and len(feature_names) == scaler.n_features_in_:
        X_new_df = X_new_df[feature_names]

    X_new = scaler.transform(X_new_df)

    # ── 4. Predecir cluster ────────────────────────────────────────────
    if hasattr(model, "predict"):
        cluster = model.predict(X_new)[0]
    else:
        # AgglomerativeClustering y similares no tienen .predict(): solo asignan
        # labels_ a los datos de entrenamiento y no admiten muestras nuevas.
        print(f"'{model_name}' no soporta predicción sobre muestras nuevas. "
              "Usa KMeans para inferencia.")
        return

    print(f"\n{'='*50}")
    print(f"  Modelo  : {model_name}")
    print(f"  Cluster : {cluster}")
    if hasattr(model, "transform"):
        dists = model.transform(X_new)[0]
        print("  Distancias a centroides:")
        for i, d in enumerate(dists):
            marker = " ←" if i == cluster else ""
            print(f"    Cluster {i}: {d:.4f}{marker}")
    print(f"{'='*50}\n")


{% elif ml_type == 'redes_neuronales' %}
from typing import Any
"""
predict_model.py — Evaluación y exportación de predicciones (PyTorch).
Arquitectura : {{ nn_model }}
Tarea        : {{ task_type }}
Optimizador  : {{ optimizer_type if optimizer_type is defined else 'AdamW' }}
Loss         : {{ nn_loss_fn if nn_loss_fn is defined else 'Auto' }}
"""
import os
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

{% if task_type == 'clasificacion' %}
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
{% else %}
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
{% endif %}

from {{ project_slug }}.utils.paths import FIGURES_DIR, MODELS_DIR, REPORTS_DIR

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ── TorchMetrics (opcional) ──────────────────────────────────────────────────
try:
    from torchmetrics import MetricCollection
{% if task_type == 'clasificacion' %}
    from torchmetrics.classification import (
        MulticlassAccuracy, MulticlassF1Score,
        MulticlassPrecision, MulticlassRecall,
    )
    def _build_tm_metrics(num_classes: int):
        return MetricCollection({
            "Accuracy":  MulticlassAccuracy(num_classes=num_classes, average="macro"),
            "F1":        MulticlassF1Score(num_classes=num_classes, average="macro"),
            "Precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
            "Recall":    MulticlassRecall(num_classes=num_classes, average="macro"),
        }).to(device)
{% else %}
    from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
    def _build_tm_metrics(_=None):
        return MetricCollection({
            "MAE":  MeanAbsoluteError(),
            "RMSE": MeanSquaredError(squared=False),
            "R2":   R2Score(),
        }).to(device)
{% endif %}
    _HAS_TM = True
except ImportError:
    _HAS_TM = False


{% if task_type == 'clasificacion' %}
def evaluate_models(models, X_test, y_test, num_classes: int = 2,
                    tb_writer=None) -> pd.DataFrame:
    """
    Evalúa modelos PyTorch (clasificación) sobre el conjunto de test.

    Genera por cada modelo:
      - Matriz de confusión PNG en figures/
      - Distribución de probabilidades PNG (solo binario)
      - CSV con métricas en reports/resultados_{{ nn_model }}.csv
      - CSV con predicciones individuales

    Métricas: Accuracy, F1, Precision, Recall (macro via torchmetrics si disponible,
    sklearn como fallback).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n  Evaluación clasificación — {{ nn_model }}\n{'='*60}")
    results = []

    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.eval()

        X_t = torch.tensor(
            X_test.values if hasattr(X_test, "values") else X_test,
            dtype=torch.float32,
        ).to(device)
        y_true = np.array(y_test.values if hasattr(y_test, "values") else y_test)

        with torch.no_grad():
            logits = model(X_t)
            if num_classes == 1:
                proba     = torch.sigmoid(logits).cpu().numpy().flatten()
                y_pred    = (proba >= 0.5).astype(int)
                proba_out = proba
            else:
                proba_mat = torch.softmax(logits, dim=1).cpu().numpy()
                y_pred    = np.argmax(proba_mat, axis=1)
                proba_out = proba_mat

        # ── Métricas ─────────────────────────────────────────────────────
        if _HAS_TM and num_classes > 1:
            _tm = _build_tm_metrics(num_classes)
            _tm.update(
                torch.tensor(y_pred, dtype=torch.long),
                torch.tensor(y_true, dtype=torch.long),
            )
            tm_res = {k: v.item() for k, v in _tm.compute().items()}
            acc, f1, prec, rec = (
                tm_res["Accuracy"], tm_res["F1"],
                tm_res["Precision"], tm_res["Recall"],
            )
            print("  [torchmetrics — macro]")
        else:
            acc  = accuracy_score(y_true, y_pred)
            f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            print("  [sklearn — weighted]")

        print(f"  Accuracy  : {acc:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        if num_classes > 1:
            print()
            print(classification_report(y_true, y_pred, zero_division=0))

        if tb_writer:
            for tag, val in [("Accuracy", acc), ("F1", f1),
                             ("Precision", prec), ("Recall", rec)]:
                tb_writer.add_scalar(f"Eval/{tag}", val, 0)

        _plot_confusion_matrix(y_true, y_pred, name, tb_writer)
        if num_classes == 2:
            _plot_proba_distribution(proba_out, y_true, name)
        _export_predictions(y_true, y_pred, proba_out, name, num_classes)

        # ── Diagrama de calibración (si hay temperature scaler) ──────────
        _calib_path = MODELS_DIR / "artifacts" / "temperature_scaler.joblib"
        if _calib_path.exists() and num_classes > 1:
            try:
                import joblib as _jl
                from {{ project_slug }}.models.calibrate import TemperatureScaler
                _scaler = TemperatureScaler()
                _scaler.load_state_dict(_jl.load(_calib_path))
                _T = _scaler.get_temperature()
                with torch.no_grad():
                    _proba_cal = torch.softmax(_scaler(logits), dim=1).cpu().numpy()
                _plot_reliability_diagram(
                    _proba_cal, y_true, name, T=_T,
                )
            except Exception as _calib_err:
                print(f"    Calibración no disponible: {_calib_err}")

        results.append({
            "Modelo": name, "Accuracy": round(acc, 4), "F1": round(f1, 4),
            "Precision": round(prec, 4), "Recall": round(rec, 4),
        })

    df_results = pd.DataFrame(results).sort_values("F1", ascending=False)
    out_csv = REPORTS_DIR / "resultados_{{ nn_model }}.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n  Métricas guardadas → {out_csv}")
    return df_results

{% else %}
# ── REGRESIÓN ────────────────────────────────────────────────────────────────
def evaluate_models(models, X_test, y_test, tb_writer=None) -> pd.DataFrame:
    """
    Evalúa modelos PyTorch (regresión) sobre el conjunto de test.

    Genera por cada modelo:
      - Scatter predicho vs real PNG en figures/
      - Distribución de residuos PNG en figures/
      - CSV con métricas (RMSE, MAE, MAPE, R²) en reports/
      - CSV con predicciones individuales

    Métricas: RMSE, MAE, MAPE, R² (torchmetrics si disponible, sklearn como fallback).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n  Evaluación regresión — {{ nn_model }}\n{'='*60}")
    results = []

    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.eval()

        X_t = torch.tensor(
            X_test.values if hasattr(X_test, "values") else X_test,
            dtype=torch.float32,
        ).to(device)
        y_true = np.array(y_test.values if hasattr(y_test, "values") else y_test,
                          dtype=np.float32)

        with torch.no_grad():
            preds_t = model(X_t).squeeze().cpu()

        y_pred = preds_t.numpy().astype(np.float32)

        # ── Métricas ─────────────────────────────────────────────────────
        if _HAS_TM:
            _tm = _build_tm_metrics()
            _tm.update(preds_t, torch.tensor(y_true))
            tm_res  = {k: v.item() for k, v in _tm.compute().items()}
            rmse    = tm_res["RMSE"]
            mae     = tm_res["MAE"]
            r2      = tm_res["R2"]
            print("  [torchmetrics]")
        else:
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae  = float(mean_absolute_error(y_true, y_pred))
            r2   = float(r2_score(y_true, y_pred))
            print("  [sklearn]")

        # MAPE manual (evitar div/0)
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) \
               if mask.any() else float("nan")

        print(f"  RMSE : {rmse:.4f}")
        print(f"  MAE  : {mae:.4f}")
        print(f"  MAPE : {mape:.2f}%")
        print(f"  R²   : {r2:.4f}")

        if tb_writer:
            for tag, val in [("RMSE", rmse), ("MAE", mae), ("R2", r2)]:
                tb_writer.add_scalar(f"Eval/{tag}", val, 0)

        _plot_regression_scatter(y_true, y_pred, name)
        _plot_residuals(y_true, y_pred, name)

        # CSV predicciones individuales
        df_preds = pd.DataFrame({"y_true": y_true, "y_pred": y_pred.round(4),
                                  "residuo": (y_true - y_pred).round(4)})
        out_pred = REPORTS_DIR / f"predicciones_{name}.csv"
        df_preds.to_csv(out_pred, index=False)
        print(f"    predicciones guardadas → {out_pred}")

        results.append({
            "Modelo": name,
            "RMSE": round(rmse, 4), "MAE": round(mae, 4),
            "MAPE_%": round(mape, 2), "R2": round(r2, 4),
        })

    df_results = pd.DataFrame(results).sort_values("RMSE", ascending=True)
    out_csv = REPORTS_DIR / "resultados_{{ nn_model }}.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n  Métricas guardadas → {out_csv}")
    return df_results


def _plot_regression_scatter(y_true, y_pred, model_name: str):
    """Scatter predicho vs real con línea y=x."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="y = ŷ")
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicho")
    ax.set_title(f"Predicho vs Real — {model_name}")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / f"scatter_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    scatter_{model_name}.png guardado")


def _plot_residuals(y_true, y_pred, model_name: str):
    """Distribución de residuos + Q-Q plot."""
    residuos = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(residuos, bins=40, color="steelblue", edgecolor="white")
    axes[0].axvline(0, color="red", lw=1.5, linestyle="--")
    axes[0].set_title(f"Distribución de residuos — {model_name}")
    axes[0].set_xlabel("Residuo")

    import scipy.stats as stats
    stats.probplot(residuos, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q plot (normalidad de residuos)")

    fig.tight_layout()
    path = FIGURES_DIR / f"residuos_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    residuos_{model_name}.png guardado")
{% endif %}


def _export_predictions(y_true, y_pred, proba, model_name: str, num_classes: int):
    """Exporta predicciones individuales a CSV."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if num_classes == 2:
        p = proba if proba.ndim == 1 else proba[:, 1]
        df["proba_pos"] = p.round(4)
    else:
        for i in range(proba.shape[1]):
            df[f"proba_cls{i}"] = proba[:, i].round(4)
    df["correcto"] = (df["y_true"] == df["y_pred"]).astype(int)

    out = REPORTS_DIR / f"predicciones_{model_name}.csv"
    df.to_csv(out, index=False)
    print(f"    predicciones guardadas → {out}")


def _plot_confusion_matrix(y_true, y_pred, model_name, tb_writer=None):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de confusión — {model_name}", fontsize=13)
    fig.tight_layout()
    path = FIGURES_DIR / f"cm_{model_name}.png"
    fig.savefig(path, dpi=150)
    if tb_writer:
        tb_writer.add_figure(f"Eval/CM_{model_name}", fig)
    plt.close(fig)
    print(f"    cm_{model_name}.png guardado")


def _plot_proba_distribution(proba, y_true, model_name):
    if proba.ndim == 2:
        proba = proba[:, 1]
    fig, ax = plt.subplots(figsize=(9, 5))
    for label in np.unique(y_true):
        ax.hist(proba[y_true == label], bins=30, alpha=0.6, label=f"Clase {label}")
    ax.axvline(0.5, color="red", linestyle="--", lw=1.5, label="Umbral 0.5")
    ax.set_xlabel("Probabilidad clase positiva")
    ax.set_ylabel("Muestras")
    ax.set_title(f"{model_name} — Distribución de probabilidades")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"proba_dist_{model_name}.png", dpi=150)
    plt.close(fig)
    print(f"    proba_dist_{model_name}.png guardado")


def predict_new(model, X_new, num_classes=2, threshold=0.5,
                export_csv: bool = False, out_name: str = "predicciones_nuevas") -> np.ndarray:
    """
    Genera predicciones sobre datos nuevos (sin etiquetas).

    Parameters
    ----------
    model       : modelo PyTorch en modo eval
    X_new       : array/DataFrame de entrada
    num_classes : 1 para regresión binaria con sigmoid, >1 para softmax
    threshold   : umbral de clasificación (solo num_classes == 1)
    export_csv  : si True, guarda predicciones en reports/out_name.csv
    out_name    : nombre del archivo CSV de salida (sin extensión)

    Returns
    -------
    np.ndarray con las predicciones (clases)
    """
    model.eval()
    with torch.no_grad():
        if hasattr(X_new, "values"):
            X_new = X_new.values
        X_t    = torch.tensor(X_new, dtype=torch.float32).to(device)
        logits = model(X_t)
        if num_classes == 1:
            proba  = torch.sigmoid(logits).cpu().numpy().flatten()
            preds  = (proba >= threshold).astype(int)
        else:
            proba  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(proba, axis=1)

    if export_csv:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"y_pred": preds})
        if num_classes == 2:
            df["proba_pos"] = (proba if proba.ndim == 1 else proba[:, 1]).round(4)
        out = REPORTS_DIR / f"{out_name}.csv"
        df.to_csv(out, index=False)
        print(f"    Predicciones nuevas guardadas → {out}")

    return preds


# ---------------------------------------------------------------------------
# Búsqueda automática de umbral óptimo por F1 (solo clasificación BINARIA)
# ---------------------------------------------------------------------------
# ACTIVAR únicamente si num_classes == 2 (salida con sigmoid).
# Para multiclase (softmax) mantén comentado y usa threshold=0.5 implícito.
#
# Uso típico tras el entrenamiento (en train_model.py o al final del pipeline):
#
#   from {{ project_slug }}.models.predict_model import find_best_threshold
#   from {{ project_slug }}.utils.paths import ARTIFACTS_DIR
#   import joblib, torch
#
#   model.eval()
#   with torch.no_grad():
#       logits = model(X_val_tensor)
#       proba_val = torch.sigmoid(logits).cpu().numpy().flatten()
#   threshold, f1 = find_best_threshold(y_val, proba_val)
#   print(f"Umbral óptimo: {threshold:.4f}  |  F1: {f1:.4f}")
#   joblib.dump(threshold, ARTIFACTS_DIR / "threshold.joblib")
#
# En evaluate_models se cargará automáticamente si existe threshold.joblib.
#
# def find_best_threshold(y_true, y_proba):
#     """
#     Calcula el umbral de decisión que maximiza el F1-score binario.
#
#     Parameters
#     ----------
#     y_true  : array-like con etiquetas reales (0/1)
#     y_proba : array-like con probabilidades de clase positiva (sigmoid)
#
#     Returns
#     -------
#     best_threshold : float — umbral que maximiza F1
#     best_f1        : float — F1 en ese umbral
#     """
#     from sklearn.metrics import precision_recall_curve
#     precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
#     # f1_scores tiene longitud N; thresholds tiene longitud N-1 → recortamos
#     f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
#     best_idx       = np.nanargmax(f1_scores[:-1])
#     best_threshold = thresholds[best_idx]
#     best_f1        = f1_scores[best_idx]
#     return best_threshold, best_f1


# ---------------------------------------------------------------------------
# Modo prueba: carga el modelo PyTorch y predice sobre entrada del usuario.
# ---------------------------------------------------------------------------
def try_model() -> None:
    """
    Modo interactivo de prueba del modelo de red neuronal entrenado.

    Flujo:
      1. Carga el checkpoint más reciente de models/ (archivo .pt o .pth).
         Si no existe, intenta con los joblibs por compatibilidad.
      2. Carga scaler y PCA (si existe) de artifacts/.
      3. Pide al usuario los valores de cada feature por consola.
      4. Preprocesa con process_input() de build_features.
      5. Pasa por la red y muestra la predicción + probabilidades.

    Requisitos previos:
      - Haber ejecutado run_full_pipeline() al menos una vez.
      - El modelo debe estar guardado como .pt/.pth en models/.
    """
    from {{ project_slug }}.features.build_features import process_input
    from {{ project_slug }}.utils.paths import ARTIFACTS_DIR, PROCESSED_DATA_DIR
    import pandas as pd

    # ── 1. Buscar checkpoint ────────────────────────────────────────────
    checkpoints = sorted(MODELS_DIR.glob("*.pt")) + sorted(MODELS_DIR.glob("*.pth"))
    if not checkpoints:
        print("No se encontraron checkpoints (.pt/.pth) en models/. Ejecuta primero la opción 0.")
        return

    print("\nCheckpoints disponibles:")
    for i, p in enumerate(checkpoints):
        print(f"  [{i}] {p.name}")
    try:
        idx = int(input("Elige checkpoint (número): "))
        ckpt_path = checkpoints[idx]
    except (ValueError, IndexError):
        print("Selección inválida.")
        return

    # Intentar carga segura primero; si el checkpoint incluye optimizador
    # (full checkpoint), reintentar con weights_only=False
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Soporte para dict {'model_state_dict': ...} o state_dict directo
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        print(f"  Epoch guardada: {ckpt.get('epoch', '?')}  |  Loss: {ckpt.get('loss', '?')}")
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # ── 2. Cargar nombres de features ──────────────────────────────────
    x_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    if x_train_path.exists():
        feature_names = pd.read_csv(x_train_path).columns.tolist()
    else:
        scaler_path = ARTIFACTS_DIR / "scaler.joblib"
        if scaler_path.exists():
            import joblib as _jl
            n_feats = _jl.load(scaler_path).n_features_in_
            feature_names = [f"feature_{i}" for i in range(n_feats)]
        else:
            print("No se encontró X_train.csv ni scaler.joblib. Ejecuta primero run_full_pipeline().")
            return

    # ── 3. Pedir valores al usuario ────────────────────────────────────
    print(f"\nIntroduce los valores para '{ckpt_path.stem}':")
    print("  (deja en blanco para usar 0 como valor por defecto)\n")
    row = {}
    for feat in feature_names:
        raw = input(f"  {feat}: ").strip()
        try:
            row[feat] = float(raw) if raw else 0.0
        except ValueError:
            row[feat] = raw if raw else 0.0

    df_input = pd.DataFrame([row])

    # ── 4. Preprocesar ─────────────────────────────────────────────────
    try:
        X_new = process_input(df_input)
    except Exception as e:
        print(f"\nError en preprocesado: {e}")
        return

    # ── 5. Inferencia ──────────────────────────────────────────────────
    # Necesitamos el modelo instanciado — importamos desde train_model
    try:
        from {{ project_slug }}.models.train_model import build_model
        output_dim_path = ARTIFACTS_DIR / "output_dim.joblib"
        if output_dim_path.exists():
            num_classes = joblib.load(output_dim_path)
        else:
            num_classes = len(set(pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").iloc[:, 0]))
        model = build_model(input_dim=X_new.shape[1], output_dim=num_classes).to(device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"\nNo se pudo reconstruir el modelo: {e}")
        print("Asegúrate de que train_model.py expone una función build_model(input_dim, output_dim).")
        return

    threshold_path = ARTIFACTS_DIR / "threshold.joblib"
    import joblib as _jl
    threshold = _jl.load(threshold_path) if threshold_path.exists() else 0.5

    preds = predict_new(model, X_new, num_classes=num_classes, threshold=threshold)

    print(f"\n{'='*50}")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Predicción : {preds[0]}")
    with torch.no_grad():
        X_t = torch.tensor(X_new, dtype=torch.float32).to(device)
        logits = model(X_t)
        if num_classes == 2:
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
            print(f"  Umbral     : {threshold:.4f}")
            print(f"  P(clase 1) : {proba[0]:.4f}")
        else:
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
            print(f"  Probabilidades: {dict(enumerate(proba.round(4).tolist()))}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Diagrama de calibración (reliability)
# ---------------------------------------------------------------------------
def _plot_reliability_diagram(probas, y_true, model_name, T=None, n_bins=10):
    """
    Reliability diagram: confianza media vs accuracy real por bin.

    Parameters
    ----------
    probas     : array (n, n_classes) probabilidades calibradas (si es 1D se
                 interpreta como binario, clase positiva)
    y_true     : array (n,) etiquetas verdaderas
    model_name : nombre del modelo para el título
    T          : temperatura (opcional, para mostrar en el gráfico)
    n_bins     : número de bins de confianza
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if probas.ndim == 2:
        pred_class = np.argmax(probas, axis=1)
        conf = probas.max(axis=1)
    else:
        pred_class = (probas >= 0.5).astype(int)
        conf = probas

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    acc_per_bin = np.zeros(n_bins)
    conf_per_bin = np.zeros(n_bins)
    count_per_bin = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        count_per_bin[i] = mask.sum()
        if count_per_bin[i] > 0:
            conf_per_bin[i] = conf[mask].mean()
            acc_per_bin[i] = (y_true[mask] == pred_class[mask]).mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectamente calibrado")
    ax.plot(conf_per_bin, acc_per_bin, "bo-", lw=2, markersize=6, label="Modelo")
    ax.fill_between(bin_centers, 0, count_per_bin / count_per_bin.max() * 0.1,
                    alpha=0.15, label="Frecuencia (norm.)")
    ax.set_xlabel("Confianza media")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram — {model_name}" + (f" (T={T:.3f})" if T else ""))
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = FIGURES_DIR / f"reliability_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    reliability_{model_name}.png guardado")


# ---------------------------------------------------------------------------
# Explicabilidad con Captum
# ---------------------------------------------------------------------------
def explain_models(
    models: dict,
    X_test,
    feature_names: list = None,
    target: int = None,
    max_display: int = 15,
) -> None:
    """
    Genera explicaciones Captum para cada modelo PyTorch.

    Por modelo produce en reports/figures/:
      - attribution_bar_{name}.png  → importancia global media
      - attribution_hist_{name}.png → distribución de atribuciones

    Parameters
    ----------
    models        : dict nombre → modelo (output de train_models)
    X_test        : datos de test (array-like)
    feature_names : nombres de features
    target        : clase objetivo (None = clase predicha)
    max_display   : máx features a mostrar en barras
    """
    try:
        from {{ project_slug }}.models.explain import explain_model, summarize_attributions
    except ImportError as _e:
        print(f"  Captum no disponible: {_e}")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if hasattr(X_test, "values"):
        X_arr = X_test.values
    else:
        X_arr = X_test

    feat_names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]
    X_t = torch.tensor(X_arr[:32], dtype=torch.float32)  # primeras 32 muestras

    print(f"\n{'='*60}\n  Captum — Explicabilidad de modelos\n{'='*60}")

    for name, model in models.items():
        print(f"\n--- {name} ---")
        try:
            attr = explain_model(model, X_t, target=target)
            imp = summarize_attributions(attr, agg="absmean")

            _plot_attribution_bar(imp, feat_names, name, max_display)
            _plot_attribution_hist(attr, feat_names, name)
        except Exception as exc:
            print(f"    Captum no disponible para {name}: {exc}")


def _plot_attribution_bar(importances, feat_names, model_name, max_display=15):
    """Barra de importancia global (|atribución| media)."""
    n_feat = len(importances)
    top_k = min(max_display, n_feat)
    indices = np.argsort(np.abs(importances))[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(9, max(4, top_k * 0.4)))
    ax.barh(range(top_k), importances[indices][::-1], color="steelblue")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feat_names[i] for i in indices[::-1]], fontsize=9)
    ax.set_xlabel("Atribución media (|Captum IG|)")
    ax.set_title(f"Importancia de features — {model_name}")
    fig.tight_layout()
    path = FIGURES_DIR / f"attribution_bar_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    attribution_bar_{model_name}.png guardado")


def _plot_attribution_hist(attributions, feat_names, model_name):
    """Heatmap de atribuciones por muestra (primeras 16) vs feature."""
    n_samples = min(16, attributions.shape[0])
    n_feat = attributions.shape[1]
    if n_feat > 30:
        return  # demasiadas features para heatmap legible

    fig, ax = plt.subplots(figsize=(10, max(4, n_samples * 0.5)))
    im = ax.imshow(attributions[:n_samples].T, aspect="auto", cmap="RdBu_r")
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([feat_names[i] if i < len(feat_names) else f"F{i}" for i in range(n_feat)], fontsize=8)
    ax.set_xlabel("Muestra")
    ax.set_xticks(range(n_samples))
    ax.set_title(f"Atribuciones por muestra — {model_name}")
    plt.colorbar(im, ax=ax, label="Atribución")
    fig.tight_layout()
    path = FIGURES_DIR / f"attribution_hist_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    attribution_hist_{model_name}.png guardado")
{% endif %}
