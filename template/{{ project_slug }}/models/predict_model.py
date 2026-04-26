{% if ml_type == 'supervisado' or ml_type == 'hibrido' %}
"""
predict_model.py — Evaluación de modelos {{ ml_type }}.
Tarea: {{ task_type }}
"""
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
    models: dict,
    X_train,
    y_train,
    X_test,
    y_test,
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

{% if task_type == "clasificacion" %}
        if threshold != 0.5 and hasattr(model, "predict_proba"):
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
def _plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
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


def predict_new(model_name: str, X_new) -> np.ndarray:
    """Carga un modelo y predice sobre nuevas muestras (ya preprocesadas)."""
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    return joblib.load(path).predict(X_new)


{% if task_type == "clasificacion" %}
def predict_proba_new(model_name: str, X_new) -> np.ndarray:
    """Carga un modelo y devuelve probabilidades de clase."""
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"{model_name} no soporta predict_proba")
    return model.predict_proba(X_new)
{% endif %}


{% elif ml_type == 'no_supervisado' %}
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

from {{ project_slug }}.utils.paths import FIGURES_DIR, MODELS_DIR


def evaluate_models(models: dict, X) -> pd.DataFrame:
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
        df_results.to_csv(FIGURES_DIR / "resultados_clustering.csv", index=False)
        print(f"\n{'='*60}\n  Resumen:\n{'='*60}")
        print(df_results.to_string(index=False))
    return df_results


def plot_dendrogram(X, method: str = "ward", color_threshold: float = None) -> None:
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


def _plot_clusters_pca(X, labels, model_name: str) -> None:
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


def load_models(model_names: list = None) -> dict:
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


{% elif ml_type == 'redes_neuronales' %}
"""
predict_model.py — Evaluación y exportación de predicciones (PyTorch).
Arquitectura activa: {{ nn_model }}
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

from {{ project_slug }}.utils.paths import FIGURES_DIR, MODELS_DIR, REPORTS_DIR

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_models(models, X_test, y_test, num_classes=2, tb_writer=None) -> pd.DataFrame:
    """
    Evalúa modelos PyTorch sobre el conjunto de test.

    Genera por cada modelo:
      - Matriz de confusión PNG en reports/figures/
      - Distribución de probabilidades PNG (solo binario)
      - CSV con métricas en reports/resultados_{{ nn_model }}.csv
      - CSV con predicciones individuales en reports/predicciones_{{ nn_model }}.csv
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n  Evaluación — {{ nn_model }}\n{'='*60}")
    results = []

    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.eval()
        with torch.no_grad():
            X_t    = torch.tensor(
                X_test.values if hasattr(X_test, "values") else X_test,
                dtype=torch.float32,
            ).to(device)
            logits = model(X_t)
            if num_classes == 1:
                proba  = torch.sigmoid(logits).cpu().numpy().flatten()
                y_pred = (proba >= 0.5).astype(int)
                proba_out = proba
            else:
                proba_mat  = torch.softmax(logits, dim=1).cpu().numpy()
                y_pred     = np.argmax(proba_mat, axis=1)
                proba_out  = proba_mat

        y_true = y_test.values if hasattr(y_test, "values") else np.array(y_test)

        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"  Accuracy  : {acc:.4f}")
        print(f"  F1 (w)    : {f1:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print()
        print(classification_report(y_true, y_pred, zero_division=0))

        if tb_writer:
            tb_writer.add_scalar("Eval/Accuracy",  acc, 0)
            tb_writer.add_scalar("Eval/F1",        f1,  0)
            tb_writer.add_scalar("Eval/Precision", prec, 0)
            tb_writer.add_scalar("Eval/Recall",    rec,  0)

        _plot_confusion_matrix(y_true, y_pred, name, tb_writer)

        if num_classes == 2:
            _plot_proba_distribution(proba_out, y_true, name)

        # ── Exportar predicciones individuales a CSV ──────────────────────
        _export_predictions(y_true, y_pred, proba_out, name, num_classes)

        results.append({
            "Modelo":    name,
            "Accuracy":  round(acc,  4),
            "F1":        round(f1,   4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
        })

    df_results = pd.DataFrame(results).sort_values("F1", ascending=False)
    out_csv = REPORTS_DIR / "resultados_{{ nn_model }}.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n  Métricas guardadas → {out_csv}")
    return df_results


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
{% endif %}