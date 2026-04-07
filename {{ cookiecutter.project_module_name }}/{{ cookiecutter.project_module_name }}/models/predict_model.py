{% if cookiecutter.ml_type == 'supervisado' or cookiecutter.ml_type == 'hibrido' %}
"""
predict_model.py — Evaluación de modelos {{ cookiecutter.ml_type }}.
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

from {{ cookiecutter.project_slug }}.utils.paths import FIGURES_DIR, MODELS_DIR

# Umbral de decisión para clasificación binaria.
# Bajar (e.g., 0.3) aumenta recall de clase minoritaria a costa de más FP.
DECISION_THRESHOLD: float = 0.5


def evaluate_models(
    models: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    threshold: float = DECISION_THRESHOLD,
) -> pd.DataFrame:
    """
    Evalúa todos los modelos sobre train y test.

    Métricas: Accuracy, F1 weighted, Precision, Recall, ROC-AUC (binario).
    Guarda matrices de confusión en figures/ y resultados en CSV.

    Returns
    -------
    pd.DataFrame ordenado de mayor a menor Acc_test.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n  Evaluación de modelos (umbral={threshold})\n{'='*60}")

    results = []
    for name, model in models.items():
        print(f"\n--- {name} ---")

        if threshold != 0.5 and hasattr(model, "predict_proba"):
            proba_test  = model.predict_proba(X_test)[:, 1]
            y_pred_test = (proba_test >= threshold).astype(int)
            proba_train = model.predict_proba(X_train)[:, 1]
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

        roc_auc = None
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
        results.append(row)

    df_results = pd.DataFrame(results).sort_values("Acc_test", ascending=False)
    df_results.to_csv(FIGURES_DIR / "resultados_modelos.csv", index=False)

    print(f"\n{'='*60}\n  Resumen (ordenado por Acc_test):\n{'='*60}")
    print(df_results.to_string(index=False))
    return df_results


def _plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de confusión — {model_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cm_{model_name}.png", dpi=150)
    plt.close(fig)
    print(f"    cm_{model_name}.png guardado")


def predict_new(model_name: str, X_new) -> np.ndarray:
    """Carga un modelo y predice sobre nuevas muestras (ya preprocesadas)."""
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    return joblib.load(path).predict(X_new)


def predict_proba_new(model_name: str, X_new) -> np.ndarray:
    """Carga un modelo y devuelve probabilidades de clase."""
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"{model_name} no soporta predict_proba")
    return model.predict_proba(X_new)


{% elif cookiecutter.ml_type == 'no_supervisado' %}
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

from {{ cookiecutter.project_slug }}.utils.paths import FIGURES_DIR, MODELS_DIR


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


{% elif cookiecutter.ml_type == 'redes_neuronales' %}
"""
predict_model.py — Evaluación de redes neuronales PyTorch.
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

from {{ cookiecutter.project_slug }}.utils.paths import FIGURES_DIR, MODELS_DIR

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_models(models, X_test, y_test, num_classes=2, tb_writer=None) -> pd.DataFrame:
    """Evalúa modelos PyTorch sobre el conjunto de test."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n  Evaluación de red neuronal\n{'='*60}")
    results = []

    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.eval()
        with torch.no_grad():
            X_t    = torch.tensor(X_test.values, dtype=torch.float32).to(device)
            logits = model(X_t)
            if num_classes == 1:
                proba  = torch.sigmoid(logits).cpu().numpy().flatten()
                y_pred = (proba >= 0.5).astype(int)
            else:
                proba  = torch.softmax(logits, dim=1).cpu().numpy()
                y_pred = np.argmax(proba, axis=1)

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

        _plot_confusion_matrix(y_true, y_pred, name, tb_writer)

        if num_classes == 2:
            _plot_proba_distribution(proba, y_true, name)

        results.append({
            "Modelo":    name,
            "Accuracy":  round(acc,  4), "F1":        round(f1,   4),
            "Precision": round(prec, 4), "Recall":    round(rec,  4),
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(FIGURES_DIR / "resultados_nn.csv", index=False)
    return df_results


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


def predict_new(model, X_new, num_classes=2, threshold=0.5) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if hasattr(X_new, "values"):
            X_new = X_new.values
        X_t    = torch.tensor(X_new, dtype=torch.float32).to(device)
        logits = model(X_t)
        if num_classes == 1:
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
            return (proba >= threshold).astype(int)
        return torch.argmax(logits, dim=1).cpu().numpy()
{% endif %}