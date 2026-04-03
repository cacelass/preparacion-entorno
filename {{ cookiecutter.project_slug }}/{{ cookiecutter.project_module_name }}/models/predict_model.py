{% if cookiecutter.ml_type == 'supervisado' %}
"""
predict_model.py — Evaluación de modelos supervisados.

Importar desde main.py:
    from {{ cookiecutter.project_module_name }}.models.predict_model import evaluate_models, DECISION_THRESHOLD
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from {{ cookiecutter.project_module_name }}.utils.paths import FIGURES_DIR, MODELS_DIR

# ---------------------------------------------------------------------------
# Umbral de decisión para clasificación binaria
# ---------------------------------------------------------------------------
# En datasets desbalanceados, bajar este umbral (e.g., 0.3) aumenta el recall
# de la clase minoritaria a costa de más falsos positivos.
# En datasets balanceados, 0.5 es el estándar.
DECISION_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Función principal de evaluación
# ---------------------------------------------------------------------------
def evaluate_models(
    models: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    threshold: float = DECISION_THRESHOLD,
) -> pd.DataFrame:
    """
    Evalúa todos los modelos entrenados sobre train y test.

    Métricas calculadas:
      - Accuracy (train y test)
      - F1-score weighted (train y test)
      - Precision weighted (train y test)
      - Recall weighted (train y test)
      - ROC-AUC (si el modelo soporta predict_proba; solo binario)

    También guarda:
      - Matriz de confusión por modelo (figures/cm_{nombre}.png)
      - Report de clasificación en consola

    Parameters
    ----------
    models    : dict devuelto por train_model.train_models()
    X_train   : features de entrenamiento (escaladas)
    y_train   : etiquetas de entrenamiento
    X_test    : features de test (escaladas)
    y_test    : etiquetas de test
    threshold : umbral de decisión para clasificación binaria (default: 0.5)

    Returns
    -------
    pd.DataFrame con una fila por modelo y columnas de métricas,
    ordenado de mayor a menor Acc_test.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Evaluación de modelos (umbral={threshold})")
    print(f"{'='*60}")

    results = []

    for name, model in models.items():
        print(f"\n--- {name} ---")

        # ── Predicciones ────────────────────────────────────────────────
        if threshold != 0.5 and hasattr(model, "predict_proba"):
            # Clasificación binaria con umbral personalizado
            proba = model.predict_proba(X_test)[:, 1]
            y_pred_test = (proba >= threshold).astype(int)
            proba_train = model.predict_proba(X_train)[:, 1]
            y_pred_train = (proba_train >= threshold).astype(int)
        else:
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)

        # ── Métricas ─────────────────────────────────────────────────────
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test  = accuracy_score(y_test,  y_pred_test)
        f1_train  = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)
        f1_test   = f1_score(y_test,  y_pred_test,  average="weighted", zero_division=0)
        prec_test = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
        rec_test  = recall_score(y_test,  y_pred_test,  average="weighted", zero_division=0)

        # ROC-AUC solo para clasificación binaria con predict_proba
        roc_auc = None
        n_classes = len(np.unique(y_test))
        if hasattr(model, "predict_proba") and n_classes == 2:
            proba_test = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, proba_test)

        # ── Impresión ─────────────────────────────────────────────────────
        print(f"  Accuracy  → train: {acc_train:.3f} | test: {acc_test:.3f}")
        print(f"  F1 (w)    → train: {f1_train:.3f}  | test: {f1_test:.3f}")
        print(f"  Precision → {prec_test:.3f}  | Recall → {rec_test:.3f}")
        if roc_auc is not None:
            print(f"  ROC-AUC   → {roc_auc:.3f}")
        print()
        print(classification_report(y_test, y_pred_test, zero_division=0))

        # ── Matriz de confusión ───────────────────────────────────────────
        _plot_confusion_matrix(y_test, y_pred_test, name)

        # ── Acumular resultados ───────────────────────────────────────────
        row = {
            "Modelo":     name,
            "Acc_train":  round(acc_train, 4),
            "Acc_test":   round(acc_test,  4),
            "F1_train":   round(f1_train,  4),
            "F1_test":    round(f1_test,   4),
            "Prec_test":  round(prec_test, 4),
            "Rec_test":   round(rec_test,  4),
        }
        if roc_auc is not None:
            row["ROC_AUC"] = round(roc_auc, 4)
        results.append(row)

    df_results = pd.DataFrame(results).sort_values("Acc_test", ascending=False)
    df_results.to_csv(FIGURES_DIR / "resultados_modelos.csv", index=False)

    print(f"\n{'='*60}")
    print("  Resumen de resultados (ordenado por Acc_test):")
    print(f"{'='*60}")
    print(df_results.to_string(index=False))

    return df_results


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------
def _plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    """Guarda la matriz de confusión como imagen PNG."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de confusión — {model_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cm_{model_name}.png", dpi=150)
    plt.close(fig)
    print(f"    cm_{model_name}.png guardado")


def predict_new(model_name: str, X_new) -> np.ndarray:
    """
    Carga un modelo guardado y predice sobre nuevas muestras.

    Parameters
    ----------
    model_name : nombre del modelo sin extensión (e.g., 'RandomForest')
    X_new      : array o DataFrame con las mismas features que el entrenamiento
                 (ya preprocesado con process_input de build_features.py)

    Returns
    -------
    np.ndarray con las predicciones
    """
    path = MODELS_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    model = joblib.load(path)
    return model.predict(X_new)


def predict_proba_new(model_name: str, X_new) -> np.ndarray:
    """
    Carga un modelo y devuelve probabilidades de clase.

    Returns
    -------
    np.ndarray shape (n_samples, n_classes)
    """
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

Importar desde main.py:
    from {{ cookiecutter.project_module_name }}.models.predict_model import evaluate_models, plot_dendrogram
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA

from {{ cookiecutter.project_module_name }}.utils.paths import FIGURES_DIR, MODELS_DIR


# ---------------------------------------------------------------------------
# Evaluación de modelos de clustering
# ---------------------------------------------------------------------------
def evaluate_models(models: dict, X) -> pd.DataFrame:
    """
    Evalúa todos los modelos de clustering con métricas internas:

      - Silhouette Score  → [-1, +1]; cuanto más cerca de +1, mejor separación.
      - Davies-Bouldin    → [0, ∞);  cuanto menor, mejor compacidad/separación.
      - Calinski-Harabasz → (0, ∞);  cuanto mayor, clusters más compactos y separados.

    También genera visualizaciones PCA-2D de cada clustering.

    Parameters
    ----------
    models : dict {nombre_modelo: modelo_ajustado}
             Devuelto por train_model.train_models().
    X      : array escalado (output de build_features.preprocess_data())

    Returns
    -------
    pd.DataFrame con una fila por modelo y las tres métricas.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("  Evaluación de clustering")
    print(f"{'='*60}")

    results = []

    for name, model in models.items():
        print(f"\n--- {name} ---")

        # Obtener etiquetas de cluster
        if hasattr(model, "labels_"):
            labels = model.labels_          # AgglomerativeClustering, DBSCAN
        else:
            labels = model.predict(X)       # KMeans, MiniBatchKMeans

        n_unique = len(set(labels)) - (1 if -1 in labels else 0)

        if n_unique < 2:
            print(f"  ⚠ Solo {n_unique} cluster(s) encontrado(s) — métricas no aplicables.")
            continue

        # ── Métricas ─────────────────────────────────────────────────────
        sil  = silhouette_score(X, labels, metric="euclidean")
        db   = davies_bouldin_score(X, labels)
        ch   = calinski_harabasz_score(X, labels)

        print(f"  Clusters encontrados : {n_unique}")
        print(f"  Silhouette Score     : {sil:.4f}  (mejor → +1)")
        print(f"  Davies-Bouldin       : {db:.4f}   (mejor → 0)")
        print(f"  Calinski-Harabasz    : {ch:.1f}  (mejor → mayor)")

        # Distribución de muestras por cluster
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"  Distribución clusters: {dist}")

        # Visualización PCA 2D
        _plot_clusters_pca(X, labels, name)

        results.append({
            "Modelo":              name,
            "N_clusters":          n_unique,
            "Silhouette":          round(sil, 4),
            "Davies_Bouldin":      round(db,  4),
            "Calinski_Harabasz":   round(ch,  2),
        })

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results.to_csv(FIGURES_DIR / "resultados_clustering.csv", index=False)
        print(f"\n{'='*60}")
        print("  Resumen:")
        print(f"{'='*60}")
        print(df_results.to_string(index=False))

    return df_results


# ---------------------------------------------------------------------------
# Dendrograma
# ---------------------------------------------------------------------------
def plot_dendrogram(X, method: str = "ward", color_threshold: float = None) -> None:
    """
    Genera y guarda un dendrograma de clustering jerárquico (scipy).

    Cómo leer el dendrograma para elegir k:
      1. Busca el tramo vertical más largo sin líneas horizontales cruzándolo.
      2. Dibuja una línea horizontal imaginaria a mitad de ese tramo.
      3. Cuenta cuántas líneas verticales cruza → ese es el k óptimo.

    Parameters
    ----------
    X               : array escalado (output de preprocess_data)
    method          : criterio de enlace:
                      'ward'     → minimiza varianza intraclúster (mejor en general)
                      'complete' → distancia máxima entre clusters (clusters compactos)
                      'average'  → distancia media (intermedio)
                      'single'   → distancia mínima (propenso a encadenamiento)
    color_threshold : altura del corte (línea roja horizontal).
                      Si None, no se dibuja el corte.
                      Ejemplo: color_threshold=240 para Mall_Customers.

    Salida
    ------
    figures/dendrogram.png
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    linked = sch.linkage(X, method=method)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_title(f"Dendrograma — linkage='{method}'", fontsize=14)
    ax.set_xlabel("Muestras (índice)")
    ax.set_ylabel("Distancia euclidiana")

    sch.dendrogram(
        linked,
        ax=ax,
        no_labels=len(X) > 50,      # oculta etiquetas con muchas muestras
        leaf_rotation=90,
        leaf_font_size=8,
    )

    if color_threshold is not None:
        ax.axhline(
            y=color_threshold,
            color="red",
            lw=2,
            linestyle="--",
            label=f"Corte en {color_threshold:.1f}",
        )
        ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dendrogram.png", dpi=150)
    plt.close(fig)
    print(f"    dendrogram.png guardado (method='{method}')")


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------
def _plot_clusters_pca(X, labels, model_name: str) -> None:
    """Proyección 2D con PCA, coloreada por cluster. Guarda PNG en figures/."""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap="tab10", s=15, alpha=0.8, edgecolors="none"
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(
        f"{model_name} — PCA 2D\n"
        f"Varianza explicada: PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%}",
        fontsize=12,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fname = FIGURES_DIR / f"clusters_{model_name}_pca.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"    {fname.name} guardado")


def load_models(model_names: list = None) -> dict:
    """Carga modelos desde disco. Si model_names es None, carga todos los .joblib."""
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


{% elif cookiecutter.ml_type == 'redes_neuronales' %}
"""
predict_model.py — Evaluación de modelos de redes neuronales (PyTorch).

Importar desde main.py:
    from {{ cookiecutter.project_module_name }}.models.predict_model import evaluate_models
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from {{ cookiecutter.project_module_name }}.utils.paths import FIGURES_DIR, MODELS_DIR

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Evaluación de modelos PyTorch
# ---------------------------------------------------------------------------
def evaluate_models(
    models: dict,
    X_test,
    y_test,
    num_classes: int = 2,
    tb_writer=None,
) -> pd.DataFrame:
    """
    Evalúa modelos PyTorch sobre el conjunto de test.

    Métricas calculadas:
      - Accuracy
      - F1-score weighted
      - Precision weighted
      - Recall weighted

    Guarda:
      - Matriz de confusión (figures/cm_{nombre}.png)
      - Reporte de clasificación en consola
      - Métricas en TensorBoard (si tb_writer es proporcionado)

    Parameters
    ----------
    models      : dict {'MLP': modelo_entrenado}
    X_test      : DataFrame con features de test (escaladas)
    y_test      : Series con etiquetas de test
    num_classes : número de clases (output_dim del modelo)
    tb_writer   : SummaryWriter de TensorBoard (opcional)

    Returns
    -------
    pd.DataFrame con métricas por modelo.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("  Evaluación de red neuronal")
    print(f"{'='*60}")

    results = []

    for name, model in models.items():
        print(f"\n--- {name} ---")

        # ── Inferencia ───────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test.values, dtype=torch.float32).to(device)
            logits = model(X_t)

            if num_classes == 1:
                # Regresión o clasificación binaria con sigmoid
                proba = torch.sigmoid(logits).cpu().numpy().flatten()
                y_pred = (proba >= 0.5).astype(int)
            else:
                # Clasificación multiclase con softmax
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                y_pred = np.argmax(proba, axis=1)

        y_true = y_test.values if hasattr(y_test, "values") else np.array(y_test)

        # ── Métricas ─────────────────────────────────────────────────────
        acc   = accuracy_score(y_true, y_pred)
        f1    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        prec  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec   = recall_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"  Accuracy  : {acc:.4f}")
        print(f"  F1 (w)    : {f1:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print()
        print(classification_report(y_true, y_pred, zero_division=0))

        # ── TensorBoard ───────────────────────────────────────────────────
        if tb_writer is not None:
            tb_writer.add_scalar("Eval/Accuracy",  acc,  0)
            tb_writer.add_scalar("Eval/F1",        f1,   0)
            tb_writer.add_scalar("Eval/Precision", prec, 0)
            tb_writer.add_scalar("Eval/Recall",    rec,  0)

        # ── Matriz de confusión ───────────────────────────────────────────
        _plot_confusion_matrix(y_true, y_pred, name, tb_writer)

        # ── Distribución de probabilidades (solo binario) ─────────────────
        if num_classes == 2:
            _plot_proba_distribution(proba, y_true, name)

        results.append({
            "Modelo":    name,
            "Accuracy":  round(acc,  4),
            "F1":        round(f1,   4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(FIGURES_DIR / "resultados_nn.csv", index=False)
    return df_results


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------
def _plot_confusion_matrix(y_true, y_pred, model_name: str, tb_writer=None) -> None:
    """Guarda la matriz de confusión como PNG y, opcionalmente, en TensorBoard."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de confusión — {model_name}", fontsize=13)
    fig.tight_layout()
    path = FIGURES_DIR / f"cm_{model_name}.png"
    fig.savefig(path, dpi=150)
    if tb_writer is not None:
        tb_writer.add_figure(f"Eval/ConfusionMatrix_{model_name}", fig)
    plt.close(fig)
    print(f"    cm_{model_name}.png guardado")


def _plot_proba_distribution(proba, y_true, model_name: str) -> None:
    """
    Histograma de probabilidades de clase positiva separado por clase real.
    Útil para detectar calibración del modelo y ajustar el umbral de decisión.
    """
    if proba.ndim == 2:
        proba = proba[:, 1]     # columna clase positiva

    fig, ax = plt.subplots(figsize=(9, 5))
    for label in np.unique(y_true):
        mask = y_true == label
        ax.hist(proba[mask], bins=30, alpha=0.6, label=f"Clase {label}", edgecolor="none")
    ax.axvline(0.5, color="red", linestyle="--", lw=1.5, label="Umbral 0.5")
    ax.set_xlabel("Probabilidad clase positiva")
    ax.set_ylabel("Muestras")
    ax.set_title(f"{model_name} — Distribución de probabilidades")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"proba_dist_{model_name}.png", dpi=150)
    plt.close(fig)
    print(f"    proba_dist_{model_name}.png guardado")


def predict_new(model, X_new, num_classes: int = 2, threshold: float = 0.5) -> np.ndarray:
    """
    Inferencia sobre nuevas muestras con un modelo PyTorch ya cargado.

    Parameters
    ----------
    model       : modelo MLP en modo eval (cargar con train_model.load_model)
    X_new       : DataFrame o ndarray preprocesado
    num_classes : número de clases de salida
    threshold   : umbral para clasificación binaria

    Returns
    -------
    np.ndarray con las predicciones de clase
    """
    model.eval()
    with torch.no_grad():
        if hasattr(X_new, "values"):
            X_new = X_new.values
        X_t = torch.tensor(X_new, dtype=torch.float32).to(device)
        logits = model(X_t)
        if num_classes == 1:
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
            return (proba >= threshold).astype(int)
        else:
            return torch.argmax(logits, dim=1).cpu().numpy()

{% endif %}