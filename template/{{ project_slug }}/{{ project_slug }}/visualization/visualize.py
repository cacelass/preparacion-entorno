{% if ml_type == 'supervisado' %}
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from {{ project_slug }}.utils.paths import FIGURES_DIR

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 7)


def plot_distributions(df: pd.DataFrame, target_col: str = None) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    if not num_cols:
        return

    fig, axes = plt.subplots(len(num_cols), 2, figsize=(14, 4 * len(num_cols)))
    if len(num_cols) == 1:
        axes = [axes]
    for i, col in enumerate(num_cols):
        if target_col and target_col in df.columns:
            for label, grp in df.groupby(target_col):
                axes[i][0].hist(grp[col].dropna(), bins=30, alpha=0.6, label=str(label))
            axes[i][0].legend(title=target_col)
            sns.boxplot(x=target_col, y=col, data=df, ax=axes[i][1])
        else:
            axes[i][0].hist(df[col].dropna(), bins=30, color="steelblue", alpha=0.7)
            axes[i][1].boxplot(df[col].dropna(), vert=False)
            axes[i][1].set_yticklabels([col])
        axes[i][0].set_title(f"Distribución — {col}")
        axes[i][1].set_title(f"Boxplot — {col}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distributions.png", dpi=150)
    plt.close(fig)
    print("    distributions.png guardado")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.8), max(6, len(corr) * 0.7)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)
    print("    correlation_matrix.png guardado")


def plot_class_balance(df: pd.DataFrame, target_col: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    counts = df[target_col].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title(f"Conteo por clase — {target_col}")
    axes[0].set_ylabel("Muestras")
    axes[0].tick_params(axis="x", rotation=0)
    axes[1].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    axes[1].set_title(f"Proporción de clases — {target_col}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)
    print("    class_balance.png guardado")


def plot_categorical_vs_target(df: pd.DataFrame, target_col: str, max_cols: int = 6) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != target_col][:max_cols]
    if not cat_cols:
        return
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(5 * len(cat_cols), 6))
    if len(cat_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cat_cols):
        sns.countplot(data=df, x=col, hue=target_col, order=df[col].value_counts().index, ax=ax)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Variables categóricas vs {target_col}", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "categorical_vs_target.png", dpi=150)
    plt.close(fig)
    print("    categorical_vs_target.png guardado")


def plot_feature_importance(models: dict, feature_names: list) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    supported = {}
    for name, model in models.items():
        est = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else model
        if hasattr(est, "feature_importances_"):
            supported[name] = (est.feature_importances_, "Importancia (Gini)")
        elif hasattr(est, "coef_"):
            coef = np.abs(est.coef_)
            supported[name] = (coef[0] if coef.ndim > 1 else coef, "Magnitud coeficiente")
    if not supported:
        print("    Ningún modelo soporta importancia de variables")
        return

    fig, axes = plt.subplots(1, len(supported), figsize=(8 * len(supported), 7))
    if len(supported) == 1:
        axes = [axes]
    for ax, (name, (importances, xlabel)) in zip(axes, supported.items()):
        # Ajustar feature_names a la longitud real (puede diferir si hay PCA)
        n_imp = len(importances)
        names = feature_names[:n_imp] if len(feature_names) >= n_imp \
                else [f"feature_{i}" for i in range(n_imp)]
        df_imp = pd.DataFrame({"Feature": names, "Importance": importances}) \
                   .sort_values("Importance", ascending=False)
        sns.barplot(x="Importance", y="Feature", data=df_imp, ax=ax, orient="h")
        ax.set_title(f"Importancia — {name}")
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("    feature_importance.png guardado")


def plot_pca_variance(pca_or_X, n_components: int = None) -> None:
    """
    Curva de varianza explicada acumulada por PCA.

    Acepta dos formas de llamada:
      plot_pca_variance(pca)          # objeto PCA ya ajustado (desde artifacts/)
      plot_pca_variance(X_scaled)     # array: ajusta PCA internamente

    Muestra:
      - Varianza explicada por cada componente (barras)
      - Varianza acumulada (línea)
      - Marca el punto donde se alcanza el 95% y el 99%

    ¿Cuántas componentes usar?
      Regla práctica: conservar las que acumulen ≥ 95% de varianza.
      Puedes ser más agresivo (90%) si quieres mayor compresión.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    from sklearn.decomposition import PCA as _PCA

    if hasattr(pca_or_X, "explained_variance_ratio_"):
        # Ya es un objeto PCA ajustado
        evr    = pca_or_X.explained_variance_ratio_
        cumvar = np.cumsum(evr)
    else:
        # Es un array: ajustar PCA
        pca    = _PCA(n_components=n_components)
        pca.fit(pca_or_X)
        evr    = pca.explained_variance_ratio_
        cumvar = np.cumsum(evr)

    n = len(evr)
    x = np.arange(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Barras de varianza por componente
    ax1.bar(x, evr * 100, color="steelblue", edgecolor="white", alpha=0.8)
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Varianza explicada (%)")
    ax1.set_title("Varianza por componente")
    ax1.set_xticks(x[::max(1, n // 15)])

    # Curva acumulada
    ax2.plot(x, cumvar * 100, "b-o", lw=2, markersize=4)
    ax2.axhline(95, color="red",    linestyle="--", lw=1.5, label="95%")
    ax2.axhline(99, color="orange", linestyle="--", lw=1.5, label="99%")

    idx_95 = int(np.argmax(cumvar >= 0.95))
    idx_99 = int(np.argmax(cumvar >= 0.99))
    ax2.axvline(idx_95 + 1, color="red",    lw=1, linestyle=":")
    ax2.axvline(idx_99 + 1, color="orange", lw=1, linestyle=":")
    ax2.annotate(f"d={idx_95+1}", xy=(idx_95+1, cumvar[idx_95]*100),
                 xytext=(idx_95+2, cumvar[idx_95]*100 - 5), fontsize=9, color="red")
    ax2.set_xlabel("Número de componentes")
    ax2.set_ylabel("Varianza acumulada (%)")
    ax2.set_title("Varianza explicada acumulada")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(f"Análisis PCA ({n} componentes)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_variance.png", dpi=150)
    plt.close(fig)
    print(f"    pca_variance.png guardado  (95% varianza con d={idx_95+1} componentes)")


def plot_pairplot(df: pd.DataFrame, target_col: str, max_features: int = 6) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    cols_to_plot = num_cols[:max_features] + [target_col]
    g = sns.pairplot(df[cols_to_plot], hue=target_col, diag_kind="kde",
                     plot_kws={"alpha": 0.5})
    g.figure.suptitle("Pairplot — separabilidad por clase", y=1.02, fontsize=13)
    g.figure.savefig(FIGURES_DIR / "pairplot.png", dpi=120, bbox_inches="tight")
    plt.close(g.figure)
    print("    pairplot.png guardado")


{% elif ml_type == 'no_supervisado' %}
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from {{ project_slug }}.utils.paths import FIGURES_DIR

plt.style.use("ggplot")


def plot_distributions(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(len(num_cols), 2, figsize=(14, 4 * len(num_cols)))
    if len(num_cols) == 1:
        axes = [axes]
    for i, col in enumerate(num_cols):
        axes[i][0].hist(df[col].dropna(), bins=30, color="steelblue", alpha=0.7)
        axes[i][0].set_title(f"Distribución — {col}")
        axes[i][1].boxplot(df[col].dropna(), vert=False)
        axes[i][1].set_yticklabels([col])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distributions.png", dpi=150)
    plt.close(fig)
    print("    distributions.png guardado")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)
    print("    correlation_matrix.png guardado")


def plot_elbow_and_silhouette(metrics: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    k_range = metrics["k_range"]
    n_plots = 3 if "db_scores" in metrics else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))

    axes[0].plot(k_range, metrics["inertias"], "bx-", lw=2, markersize=8)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inercia (WCSS)")
    axes[0].set_title("Método del codo")
    axes[0].grid(True)

    best_k = k_range[int(np.argmax(metrics["silhouettes"]))]
    axes[1].plot(k_range, metrics["silhouettes"], "go-", lw=2, markersize=8)
    axes[1].axvline(x=best_k, color="red", lw=1.5, linestyle="--", label=f"Mejor k={best_k}")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score")
    axes[1].legend()
    axes[1].grid(True)

    if "db_scores" in metrics:
        best_k_db = k_range[int(np.argmin(metrics["db_scores"]))]
        axes[2].plot(k_range, metrics["db_scores"], "rs-", lw=2, markersize=8)
        axes[2].axvline(x=best_k_db, color="blue", lw=1.5, linestyle="--", label=f"Mejor k={best_k_db}")
        axes[2].set_xlabel("k")
        axes[2].set_ylabel("Davies-Bouldin")
        axes[2].set_title("Davies-Bouldin (menor = mejor)")
        axes[2].legend()
        axes[2].grid(True)

    fig.suptitle("Selección del número óptimo de clusters", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "elbow_silhouette.png", dpi=150)
    plt.close(fig)
    print(f"    elbow_silhouette.png guardado  (mejor k silhouette: {best_k})")


def plot_dendrogram(X, method: str = "ward", color_threshold: float = None) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_title(f"Dendrograma — linkage='{method}'", fontsize=14)
    ax.set_xlabel("Muestras")
    ax.set_ylabel("Distancia euclidiana")
    sch.dendrogram(sch.linkage(X, method=method), ax=ax, no_labels=len(X) > 50)
    if color_threshold is not None:
        ax.axhline(y=color_threshold, color="red", lw=2, linestyle="--",
                   label=f"Corte en {color_threshold:.1f}")
        ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dendrogram.png", dpi=150)
    plt.close(fig)
    print("    dendrogram.png guardado")


def plot_pca_variance(X, n_components: int = None) -> None:
    """
    Curva de varianza explicada acumulada por PCA.
    Acepta un array escalado o un objeto PCA ya ajustado.

    Útil tanto en no_supervisado (antes de clustering) como en supervisado
    (para decidir cuántas componentes conservar).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    from sklearn.decomposition import PCA as _PCA

    if hasattr(X, "explained_variance_ratio_"):
        evr    = X.explained_variance_ratio_
        cumvar = np.cumsum(evr)
    else:
        pca    = _PCA(n_components=n_components)
        pca.fit(X)
        evr    = pca.explained_variance_ratio_
        cumvar = np.cumsum(evr)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, "bo-", lw=2, markersize=5)
    ax.axhline(0.95, color="red",    linestyle="--", lw=1.5, label="95% varianza")
    ax.axhline(0.99, color="orange", linestyle="--", lw=1.5, label="99% varianza")
    idx_95 = int(np.argmax(cumvar >= 0.95))
    ax.axvline(idx_95 + 1, color="red", lw=1, linestyle=":")
    ax.annotate(f"d={idx_95+1}", xy=(idx_95+1, cumvar[idx_95]),
                xytext=(idx_95+3, cumvar[idx_95] - 0.05), fontsize=10)
    ax.set_xlabel("Número de componentes")
    ax.set_ylabel("Varianza explicada acumulada")
    ax.set_title("PCA — varianza explicada")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_variance.png", dpi=150)
    plt.close(fig)
    print(f"    pca_variance.png guardado  (d para 95%: {idx_95+1})")


def plot_umap(X, labels=None, model_name: str = "UMAP") -> None:
    """
    Proyección UMAP 2D del espacio de features.

    UMAP (Uniform Manifold Approximation and Projection) es una alternativa
    no lineal a PCA que preserva mejor la estructura local del espacio.
    Muy útil para visualizar clusters antes de aplicar el algoritmo.

    Parameters
    ----------
    X      : array escalado (output de preprocess_data)
    labels : etiquetas de cluster o clase para colorear. Si None, sin color.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import umap
    except ImportError:
        print("    ⚠ umap-learn no instalado. Ejecuta: uv add umap-learn")
        return

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_2d    = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels if labels is not None else "steelblue",
        cmap="tab10" if labels is not None else None,
        s=10, alpha=0.7, edgecolors="none",
    )
    if labels is not None:
        plt.colorbar(scatter, ax=ax, label="Cluster / Clase")
    ax.set_title(f"{model_name} — Proyección 2D", fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"umap_{model_name}.png", dpi=150)
    plt.close(fig)
    print(f"    umap_{model_name}.png guardado")


def plot_clusters_pca(X, labels, model_name: str = "Clustering") -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pca  = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=15, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(
        f"{model_name} — PCA 2D\n"
        f"Varianza: PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%}", fontsize=12,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"clusters_{model_name}_pca.png", dpi=150)
    plt.close(fig)


def plot_cluster_profiles(X, labels, feature_names: list = None) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n_features   = X.shape[1]
    feature_names = feature_names or [f"Feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["Cluster"] = labels

    n_cols = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).ravel()

    for i, feat in enumerate(feature_names):
        sns.boxplot(x="Cluster", y=feat, data=df, ax=axes[i], palette="tab10")
        axes[i].set_title(f"{feat} por cluster")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Perfil de variables por cluster", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_profiles.png", dpi=150)
    plt.close(fig)
    print("    cluster_profiles.png guardado")


{% elif ml_type == 'redes_neuronales' %}
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from {{ project_slug }}.utils.paths import FIGURES_DIR

plt.style.use("ggplot")


def plot_distributions(df, target_col=None) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    if not num_cols:
        return
    n_cols = min(3, len(num_cols))
    n_rows = int(np.ceil(len(num_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).ravel()
    for i, col in enumerate(num_cols):
        if target_col and target_col in df.columns:
            for label, grp in df.groupby(target_col):
                axes[i].hist(grp[col].dropna(), bins=25, alpha=0.6, label=str(label))
            axes[i].legend(fontsize=8)
        else:
            axes[i].hist(df[col].dropna(), bins=25, alpha=0.7)
        axes[i].set_title(col, fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distribución de variables", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distributions.png", dpi=150)
    plt.close(fig)


def plot_correlation_matrix(df) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.7), max(6, len(corr) * 0.6)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def plot_pca_variance(pca_or_X, n_components: int = None) -> None:
    """Curva de varianza explicada por PCA. Acepta objeto PCA o array escalado."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    from sklearn.decomposition import PCA as _PCA
    if hasattr(pca_or_X, "explained_variance_ratio_"):
        evr = pca_or_X.explained_variance_ratio_
    else:
        pca = _PCA(n_components=n_components)
        pca.fit(pca_or_X)
        evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar * 100, "b-o", lw=2, markersize=4)
    ax.axhline(95, color="red",    linestyle="--", lw=1.5, label="95%")
    ax.axhline(99, color="orange", linestyle="--", lw=1.5, label="99%")
    idx_95 = int(np.argmax(cumvar >= 0.95))
    ax.axvline(idx_95 + 1, color="red", lw=1, linestyle=":")
    ax.set_xlabel("Componentes")
    ax.set_ylabel("Varianza acumulada (%)")
    ax.set_title("PCA — varianza explicada")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_variance.png", dpi=150)
    plt.close(fig)
    print(f"    pca_variance.png guardado  (95% con d={idx_95+1})")


def plot_training_history(train_losses, val_losses=None, train_accs=None,
                          val_accs=None, tb_writer=None) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n_plots = 2 if train_accs is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(9 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, "b-o", lw=2, markersize=3, label="Train loss")
    if val_losses:
        axes[0].plot(epochs, val_losses, "r-o", lw=2, markersize=3, label="Val loss")
    axes[0].set_title("Curva de loss")
    axes[0].set_xlabel("Época")
    axes[0].legend()
    axes[0].grid(True)
    if train_accs is not None:
        axes[1].plot(epochs, train_accs, "b-o", lw=2, markersize=3, label="Train acc")
        if val_accs:
            axes[1].plot(epochs, val_accs, "r-o", lw=2, markersize=3, label="Val acc")
        axes[1].set_title("Curva de accuracy")
        axes[1].set_xlabel("Época")
        axes[1].legend()
        axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_history.png", dpi=150)
    if tb_writer:
        tb_writer.add_figure("Training/History", fig)
    plt.close(fig)
    print("    training_history.png guardado")


def plot_confusion_matrix(y_true, y_pred, model_name: str = "MLP", tb_writer=None) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"{model_name} — Conteos")
    ConfusionMatrixDisplay(confusion_matrix=np.round(cm_norm, 2)).plot(
        ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title(f"{model_name} — Proporción")
    fig.tight_layout()
    path = FIGURES_DIR / f"cm_{model_name}.png"
    fig.savefig(path, dpi=150)
    if tb_writer:
        tb_writer.add_figure(f"Eval/CM_{model_name}", fig)
    plt.close(fig)
    print(f"    cm_{model_name}.png guardado")


def plot_class_balance(df, target_col: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    counts = df[target_col].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title(f"Conteo por clase — {target_col}")
    axes[1].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    axes[1].set_title(f"Proporción — {target_col}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)


{% elif ml_type == 'hibrido' %}
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from {{ project_slug }}.utils.paths import FIGURES_DIR

plt.style.use("ggplot")


def plot_distributions(df: pd.DataFrame, target_col: str = None) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    if not num_cols:
        return
    n_cols = min(3, len(num_cols))
    n_rows = int(np.ceil(len(num_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).ravel()
    for i, col in enumerate(num_cols):
        if target_col and target_col in df.columns:
            for label, grp in df.groupby(target_col):
                axes[i].hist(grp[col].dropna(), bins=25, alpha=0.6, label=str(label))
            axes[i].legend(fontsize=7)
        else:
            axes[i].hist(df[col].dropna(), bins=25, alpha=0.7)
        axes[i].set_title(col, fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distribución de variables", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distributions.png", dpi=150)
    plt.close(fig)


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.8), max(6, len(corr) * 0.7)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def plot_class_balance(df: pd.DataFrame, target_col: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    counts = df[target_col].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title(f"Conteo por clase — {target_col}")
    axes[0].tick_params(axis="x", rotation=0)
    axes[1].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    axes[1].set_title(f"Proporción de clases — {target_col}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)


def plot_pca_variance(pca_or_X, n_components: int = None) -> None:
    """
    Curva de varianza explicada acumulada por PCA.
    Acepta objeto PCA ya ajustado (desde artifacts/pca.joblib) o array escalado.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    from sklearn.decomposition import PCA as _PCA
    if hasattr(pca_or_X, "explained_variance_ratio_"):
        evr = pca_or_X.explained_variance_ratio_
    else:
        pca = _PCA(n_components=n_components)
        pca.fit(pca_or_X)
        evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    n = len(evr)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(range(1, n + 1), evr * 100, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Componente")
    ax1.set_ylabel("Varianza (%)")
    ax1.set_title("Varianza por componente")
    ax2.plot(range(1, n + 1), cumvar * 100, "b-o", lw=2, markersize=4)
    ax2.axhline(95, color="red",    linestyle="--", lw=1.5, label="95%")
    ax2.axhline(99, color="orange", linestyle="--", lw=1.5, label="99%")
    idx_95 = int(np.argmax(cumvar >= 0.95))
    ax2.axvline(idx_95 + 1, color="red", lw=1, linestyle=":")
    ax2.annotate(f"d={idx_95+1}", xy=(idx_95+1, cumvar[idx_95]*100),
                 xytext=(idx_95+2, cumvar[idx_95]*100 - 5), fontsize=9, color="red")
    ax2.set_xlabel("Componentes")
    ax2.set_ylabel("Varianza acumulada (%)")
    ax2.set_title("Varianza acumulada")
    ax2.legend()
    ax2.grid(True)
    fig.suptitle("PCA — análisis de varianza", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_variance.png", dpi=150)
    plt.close(fig)
    print(f"    pca_variance.png guardado  (95% con d={idx_95+1})")


def plot_umap(X, labels=None, model_name: str = "UMAP") -> None:
    """
    Proyección UMAP 2D. Si se pasan labels, colorea por clase/cluster.
    Ideal para visualizar el espacio reducido antes del clasificador.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import umap
    except ImportError:
        print("    ⚠ umap-learn no instalado. Ejecuta: uv add umap-learn")
        return
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_2d    = reducer.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels if labels is not None else "steelblue",
        cmap="tab10" if labels is not None else None,
        s=10, alpha=0.7,
    )
    if labels is not None:
        plt.colorbar(sc, ax=ax, label="Clase / Cluster")
    ax.set_title(f"{model_name} — Proyección 2D no lineal", fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"umap_{model_name}.png", dpi=150)
    plt.close(fig)
    print(f"    umap_{model_name}.png guardado")


def plot_feature_importance(models: dict, feature_names: list) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    supported = {}
    for name, model in models.items():
        est = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else model
        if hasattr(est, "feature_importances_"):
            supported[name] = (est.feature_importances_, "Importancia (Gini)")
        elif hasattr(est, "coef_"):
            coef = np.abs(est.coef_)
            supported[name] = (coef[0] if coef.ndim > 1 else coef, "Magnitud coeficiente")
    if not supported:
        return
    fig, axes = plt.subplots(1, len(supported), figsize=(8 * len(supported), 7))
    if len(supported) == 1:
        axes = [axes]
    for ax, (name, (importances, xlabel)) in zip(axes, supported.items()):
        n_imp = len(importances)
        names = feature_names[:n_imp] if len(feature_names) >= n_imp \
                else [f"feature_{i}" for i in range(n_imp)]
        df_imp = pd.DataFrame({"Feature": names, "Importance": importances}) \
                   .sort_values("Importance", ascending=False)
        sns.barplot(x="Importance", y="Feature", data=df_imp, ax=ax, orient="h")
        ax.set_title(f"Importancia — {name}")
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("    feature_importance.png guardado")
{% endif %}