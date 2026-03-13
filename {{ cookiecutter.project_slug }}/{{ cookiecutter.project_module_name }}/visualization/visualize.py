{% if cookiecutter.ml_type == 'supervisado' %}
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from {{ cookiecutter.project_module_name }}.utils.paths import FIGURES_DIR

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 7)


def plot_distributions(df: pd.DataFrame, target_col: str = None) -> None:
    """Distribución de cada variable numérica (histograma + boxplot). Colorea por clase si se indica target_col."""
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
    """Mapa de calor de la matriz de correlación."""
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
    """Balance de clases: barplot + pie. Detecta datasets desbalanceados."""
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
    """Countplot de variables categóricas coloreado por target."""
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
    """
    Importancia de variables:
      - RandomForest / GradientBoosting → feature_importances_ (Gini)
      - LogisticRegression              → |coef_| (magnitud de coeficientes)
    """
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
        df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)
        sns.barplot(x="Importance", y="Feature", data=df_imp, ax=ax, orient="h")
        ax.set_title(f"Importancia — {name}")
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("    feature_importance.png guardado")

{% elif cookiecutter.ml_type == 'no_supervisado' %}
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from {{ cookiecutter.project_module_name }}.utils.paths import FIGURES_DIR

plt.style.use("ggplot")


def plot_distributions(df: pd.DataFrame) -> None:
    """Histograma y boxplot de cada variable numérica."""
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
    """
    Dibuja el método del codo (inercia) y el Silhouette Score en el mismo gráfico.

    Parameters
    ----------
    metrics : dict devuelto por train_model.find_optimal_k()
              con claves: 'k_range', 'inertias', 'silhouettes'

    Interpretación:
      Elbow    → el K óptimo está en el "codo" donde la inercia deja de bajar bruscamente.
      Silhouette → el K óptimo es el que maximiza el score (más cercano a +1).
      Usa ambos y elige el K donde ambos métodos coincidan o comprometes.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    k_range = metrics["k_range"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow
    ax1.plot(k_range, metrics["inertias"], "bx-", lw=2, markersize=8)
    ax1.set_xlabel("Número de clusters (k)")
    ax1.set_ylabel("Inercia (WCSS)")
    ax1.set_title("Método del codo")
    ax1.grid(True)

    # Silhouette
    ax2.plot(k_range, metrics["silhouettes"], "go-", lw=2, markersize=8)
    best_k = k_range[int(np.argmax(metrics["silhouettes"]))]
    ax2.axvline(x=best_k, color="red", lw=1.5, linestyle="--", label=f"Mejor k={best_k}")
    ax2.set_xlabel("Número de clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score por k")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle("Selección del número óptimo de clusters", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "elbow_silhouette.png", dpi=150)
    plt.close(fig)
    print(f"    elbow_silhouette.png guardado  (mejor k por silhouette: {best_k})")


def plot_dendrogram(X, method: str = "ward", color_threshold: float = None) -> None:
    """
    Dendrograma de clustering jerárquico (scipy).

    Cómo elegir k:
      1. Busca la línea vertical más larga sin cruzar ninguna horizontal extendida.
      2. Traza una horizontal en mitad de ese tramo.
      3. Cuenta cuántas ramas verticales cruza → ese es el k óptimo.

    Parameters
    ----------
    method          : 'ward' (minimiza varianza, mejor general) | 'complete' | 'average' | 'single'
    color_threshold : altura del corte horizontal (línea roja). Si None, no se dibuja.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_title(f"Dendrograma — linkage='{method}'", fontsize=14)
    ax.set_xlabel("Muestras")
    ax.set_ylabel("Distancia euclidiana")

    sch.dendrogram(
        sch.linkage(X, method=method),
        ax=ax,
        no_labels=len(X) > 50,  # oculta etiquetas si hay muchas muestras
    )

    if color_threshold is not None:
        ax.axhline(y=color_threshold, color="red", lw=2, linestyle="--",
                   label=f"Corte en {color_threshold:.1f}")
        ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dendrogram.png", dpi=150)
    plt.close(fig)
    print("    dendrogram.png guardado")


def plot_pca_variance(X, max_components: int = None) -> None:
    """
    Curva de varianza explicada acumulada por PCA.

    Permite elegir cuántas componentes conservar para explicar un % mínimo
    de la varianza (recomendado: >= 95%).
    Útil también como paso previo a clustering en espacios de alta dimensión.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    from sklearn.decomposition import PCA as _PCA
    pca = _PCA(n_components=max_components)
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, "bo-", lw=2, markersize=5)
    ax.axhline(0.95, color="red", linestyle="--", lw=1.5, label="95% varianza")
    ax.axhline(0.99, color="orange", linestyle="--", lw=1.5, label="99% varianza")

    # Marcar el punto donde se alcanza el 95%
    idx_95 = int(np.argmax(cumvar >= 0.95))
    ax.axvline(idx_95 + 1, color="red", lw=1, linestyle=":")
    ax.annotate(f"d={idx_95+1}", xy=(idx_95 + 1, cumvar[idx_95]),
                xytext=(idx_95 + 3, cumvar[idx_95] - 0.05), fontsize=10)

    ax.set_xlabel("Número de componentes principales")
    ax.set_ylabel("Varianza explicada acumulada")
    ax.set_title("PCA — varianza explicada")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_variance.png", dpi=150)
    plt.close(fig)
    print(f"    pca_variance.png guardado  (d para 95%: {idx_95+1})")


def plot_clusters_pca(X, labels, model_name: str = "Clustering") -> None:
    """Proyección PCA 2D con colores por cluster. Generado también por evaluate_models()."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=15, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"{model_name} — PCA 2D")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"clusters_{model_name}_pca.png", dpi=150)
    plt.close(fig)

{% elif cookiecutter.ml_type == 'redes_neuronales' %}
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from {{ cookiecutter.project_module_name }}.utils.paths import FIGURES_DIR

plt.style.use("ggplot")


def plot_distributions(df, target_col=None) -> None:
    """Histograma de cada variable numérica."""
    import pandas as pd
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[col].dropna(), bins=30, alpha=0.7)
        ax.set_title(f"Distribución — {col}")
        fig.savefig(FIGURES_DIR / f"dist_{col}.png", dpi=120)
        plt.close(fig)
    print(f"    {len(num_cols)} histogramas guardados en figures/")


def plot_correlation_matrix(df) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)
    print("    correlation_matrix.png guardado")


def plot_training_history(train_losses, val_losses=None, tb_writer=None) -> None:
    """Curva de loss durante el entrenamiento."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_losses, label="Train loss", lw=2)
    if val_losses:
        ax.plot(val_losses, label="Val loss", lw=2)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Curva de entrenamiento")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_history.png", dpi=150)
    plt.close(fig)
    if tb_writer:
        tb_writer.add_figure("Training/History", fig)
    print("    training_history.png guardado")
{% endif %}
