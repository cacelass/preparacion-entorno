{% if cookiecutter.ml_type == "supervisado" %}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # sin GUI, necesario en entornos headless

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

from {{ cookiecutter.project_module_name }}.utils.paths import FIGURES_DIR


# ---------------------------------------------------------------------------
# Umbral de decisión
# ---------------------------------------------------------------------------
# Por defecto, sklearn usa 0.5. Ajústalo si tus clases están desbalanceadas:
#   - Umbral < 0.5 → detecta más positivos (mayor recall, menor precisión)
#   - Umbral > 0.5 → más conservador (mayor precisión, menor recall)
DECISION_THRESHOLD: float = 0.5


def _predict_with_threshold(model, X_test, threshold: float):
    """Aplica un umbral personalizado sobre predict_proba (si disponible)."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        return (probs >= threshold).astype(int), probs
    else:
        # LinearSVC y modelos sin predict_proba
        return model.predict(X_test), None


def evaluate_models(
    models: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    threshold: float = DECISION_THRESHOLD,
) -> pd.DataFrame:
    """
    Evalúa todos los modelos y genera:
      - Informe de clasificación en consola
      - Matriz de confusión por modelo  → figures/confusion_matrix_{nombre}.png
      - Tabla comparativa               → figures/model_comparison.png
      - Curvas ROC conjuntas            → figures/roc_curves.png

    Parameters
    ----------
    threshold : umbral de probabilidad para decidir clase positiva (default 0.5)

    Returns
    -------
    pd.DataFrame con columnas: Modelo, Acc_train, Acc_test, F1_macro, AUC
    """
    print(f"--> Evaluando modelos (umbral = {threshold})...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # ─── Curvas ROC (todas en el mismo gráfico) ───
    fig_roc, ax_roc = plt.subplots(figsize=(9, 7))
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC = 0.50)")

    for name, model in models.items():
        print(f"\n  [{name}]")

        # Predicciones
        y_pred, y_prob = _predict_with_threshold(model, X_test, threshold)

        # Puntuaciones
        acc_train = model.score(X_train, y_train)
        acc_test = model.score(X_test, y_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        f1_macro = report["macro avg"]["f1-score"]

        print(f"    Acc train: {acc_train:.3f}  |  Acc test: {acc_test:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        # ─── Matriz de confusión ───
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
        ax_cm.set_title(f"Matriz de confusión — {name}")
        fig_cm.tight_layout()
        fig_cm.savefig(FIGURES_DIR / f"confusion_matrix_{name}.png", dpi=150)
        plt.close(fig_cm)

        # ─── ROC ───
        roc_auc = float("nan")
        if y_prob is not None:
            try:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")
            except Exception:
                pass

        results.append({
            "Modelo": name,
            "Acc_train": round(acc_train, 3),
            "Acc_test": round(acc_test, 3),
            "F1_macro": round(f1_macro, 3),
            "AUC": round(roc_auc, 3) if not np.isnan(roc_auc) else "N/A",
        })

    # Guardar curvas ROC
    ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax_roc.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
    ax_roc.set_title("Curvas ROC — comparación de modelos", fontsize=14)
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)
    fig_roc.tight_layout()
    fig_roc.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig_roc)
    print(f"\n    ROC guardado → figures/roc_curves.png")

    # ─── Tabla comparativa ───
    df_results = pd.DataFrame(results)
    _plot_comparison_table(df_results)
    print("\n--> Resumen comparativo:")
    print(df_results.to_string(index=False))

    return df_results


def _plot_comparison_table(df: pd.DataFrame):
    """Genera una tabla visual con los resultados de todos los modelos."""
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 2), 2.5))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Destacar el mejor Acc_test
    try:
        acc_vals = [float(v) for v in df["Acc_test"]]
        best_row = acc_vals.index(max(acc_vals)) + 1  # +1 por la fila de cabecera
        for col in range(len(df.columns)):
            table[best_row, col].set_facecolor("#c6efce")  # verde claro
    except Exception:
        pass

    ax.set_title("Comparación de modelos supervisados", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)
    print("    Tabla comparativa → figures/model_comparison.png")


def predict_new(model, X_new, threshold: float = DECISION_THRESHOLD):
    """
    Genera predicciones sobre datos nuevos.

    Returns
    -------
    predictions : np.ndarray con clases predichas
    probabilities : np.ndarray de probabilidades (None si el modelo no lo soporta)
    """
    preds, probs = _predict_with_threshold(model, X_new, threshold)
    return preds, probs

{% elif cookiecutter.ml_type == "no_supervisado" %}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from {{ cookiecutter.project_module_name }}.utils.paths import FIGURES_DIR


def evaluate_models(models: dict, X) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        labels = model.labels_ if hasattr(model, "labels_") else model.predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        sil = silhouette_score(X, labels) if n_clusters > 1 else float("nan")
        print(f"[{name}] clusters={n_clusters}  silhouette={sil:.3f}")

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Cluster")
        ax.set_title(f"{name} — PCA 2D  (silhouette={sil:.3f})")
        fig.savefig(FIGURES_DIR / f"clusters_{name}.png", dpi=150)
        plt.close(fig)

{% elif cookiecutter.ml_type == "redes_neuronales" %}
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

import torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassROC,
    MulticlassAUROC,
)

from {{ cookiecutter.project_module_name }}.utils.paths import FIGURES_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_models(models: dict, X_test, y_test, num_classes: int, tb_writer=None) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_t = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_test.values, dtype=torch.long).to(device)

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            logits = model(X_t)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

        acc_micro = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
        acc_per_class = MulticlassAccuracy(num_classes=num_classes, average="none").to(device)
        cm_metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
        roc_metric = MulticlassROC(num_classes=num_classes).to(device)
        auroc_metric = MulticlassAUROC(num_classes=num_classes, average="macro").to(device)

        acc_micro.update(preds, y_t)
        acc_per_class.update(preds, y_t)
        cm_metric.update(preds, y_t)
        roc_metric.update(probs, y_t)
        auroc_metric.update(probs, y_t)

        acc_val = acc_micro.compute().item()
        acc_cls = acc_per_class.compute().cpu().numpy()
        cm_val = cm_metric.compute().cpu().numpy()
        auroc_val = auroc_metric.compute().item()
        fpr_list, tpr_list, _ = roc_metric.compute()

        print(f"[{name}] Accuracy: {acc_val:.3f}  AUROC: {auroc_val:.3f}")

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm_val, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {name}")
        fig.savefig(FIGURES_DIR / f"confusion_matrix_{name}.png", dpi=150)
        plt.close(fig)

        # Accuracy per class
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(num_classes), acc_cls)
        ax.set_title(f"Accuracy por clase — {name}")
        ax.set_xlabel("Clase")
        ax.set_ylabel("Accuracy")
        fig.savefig(FIGURES_DIR / f"accuracy_per_class_{name}.png", dpi=150)
        plt.close(fig)

        # ROC
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
            ax.plot(fpr.cpu(), tpr.cpu(), label=f"Clase {i}")
        ax.set_title(f"ROC — {name} (AUROC={auroc_val:.3f})")
        ax.legend()
        fig.savefig(FIGURES_DIR / f"roc_{name}.png", dpi=150)
        plt.close(fig)

        if tb_writer:
            tb_writer.add_scalar(f"Accuracy/{name}", acc_val)
            tb_writer.add_scalar(f"AUROC/{name}", auroc_val)
{% endif %}
