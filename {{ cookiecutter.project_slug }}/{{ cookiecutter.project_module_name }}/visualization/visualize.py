{% if cookiecutter.ml_type == "supervisado" %}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import seaborn as sns

from {{ cookiecutter.project_module_name }}.utils.paths import FIGURES_DIR

# Estilo global
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 7)


def plot_distributions(df: pd.DataFrame, target_col: str = None) -> None:
    """
    Distribución de cada variable numérica.
    Si se indica target_col, colorea por clase.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    n = len(num_cols)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    for i, col in enumerate(num_cols):
        # Histograma
        if target_col and target_col in df.columns:
            for label, grp in df.groupby(target_col):
                axes[i][0].hist(grp[col].dropna(), bins=30, alpha=0.6, label=str(label))
            axes[i][0].legend(title=target_col)
        else:
            axes[i][0].hist(df[col].dropna(), bins=30, color="steelblue", alpha=0.7)
        axes[i][0].set_title(f"Distribución — {col}")
        axes[i][0].set_xlabel(col)

        # Boxplot
        if target_col and target_col in df.columns:
            sns.boxplot(x=target_col, y=col, data=df, ax=axes[i][1])
        else:
            axes[i][1].boxplot(df[col].dropna(), vert=False)
            axes[i][1].set_yticklabels([col])
        axes[i][1].set_title(f"Boxplot — {col}")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distributions.png", dpi=150)
    plt.close(fig)
    print("    distributions.png guardado")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Mapa de calor con la matriz de correlación de variables numéricas."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    corr = df.select_dtypes(include=[np.number]).corr()

    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.8), max(6, len(corr) * 0.7)))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, ax=ax, annot_kws={"size": 9},
    )
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)
    print("    correlation_matrix.png guardado")


def plot_class_balance(df: pd.DataFrame, target_col: str) -> None:
    """
    Muestra el balance de clases del target.
    Útil para detectar datasets desbalanceados y decidir si usar class_weight='balanced'.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    counts = df[target_col].value_counts()
    pcts = df[target_col].value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title(f"Conteo por clase — {target_col}")
    axes[0].set_ylabel("Número de muestras")
    axes[0].tick_params(axis="x", rotation=0)

    axes[1].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    axes[1].set_title(f"Proporción de clases — {target_col}")

    fig.suptitle(f"Balance de clases: {pcts.to_dict()}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)
    print("    class_balance.png guardado")


def plot_categorical_vs_target(df: pd.DataFrame, target_col: str, max_cols: int = 6) -> None:
    """
    Para cada variable categórica, muestra un countplot coloreado por el target.
    Ayuda a identificar variables con poder predictivo.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    cat_cols = cat_cols[:max_cols]

    if not cat_cols:
        return

    n = len(cat_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cat_cols):
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, hue=target_col, order=order, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(f"Variables categóricas vs {target_col}", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "categorical_vs_target.png", dpi=150)
    plt.close(fig)
    print("    categorical_vs_target.png guardado")


def plot_feature_importance(models: dict, feature_names: list) -> None:
    """
    Importancia de variables para modelos que lo soportan:
      - RandomForest / GradientBoosting → feature_importances_ (Gini)
      - LogisticRegression              → |coef_| (magnitud de coeficientes)

    Se genera un gráfico de barras horizontal por cada modelo.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    supported = {}
    for name, model in models.items():
        # Extraer importancias del pipeline si es necesario
        estimator = model
        if hasattr(model, "named_steps"):
            # Pipeline: buscar el último estimador
            steps = list(model.named_steps.values())
            estimator = steps[-1]

        if hasattr(estimator, "feature_importances_"):
            supported[name] = (estimator.feature_importances_, "Importancia (Gini)")
        elif hasattr(estimator, "coef_"):
            coef = np.abs(estimator.coef_)
            importances = coef[0] if coef.ndim > 1 else coef
            supported[name] = (importances, "Magnitud del coeficiente")

    if not supported:
        print("    Ningún modelo soporta importancia de variables")
        return

    fig, axes = plt.subplots(1, len(supported), figsize=(8 * len(supported), 7))
    if len(supported) == 1:
        axes = [axes]

    for ax, (name, (importances, xlabel)) in zip(axes, supported.items()):
        df_imp = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
        )
        sns.barplot(x="Importance", y="Feature", data=df_imp, ax=ax, orient="h")
        ax.set_title(f"Importancia de variables — {name}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("    feature_importance.png guardado")

{% elif cookiecutter.ml_type == "no_supervisado" %}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import seaborn as sns

from sklearn.cluster import KMeans
from {{ cookiecutter.project_module_name }}.utils.paths import FIGURES_DIR

plt.style.use("ggplot")


def plot_distributions(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(len(num_cols), 1, figsize=(10, 3 * len(num_cols)))
    if len(num_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, num_cols):
        ax.hist(df[col].dropna(), bins=30, color="steelblue", alpha=0.7)
        ax.set_title(f"Distribución — {col}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distributions.png", dpi=150)
    plt.close(fig)


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def plot_elbow(X, max_k: int = 10) -> None:
    """Método del codo para seleccionar el número óptimo de clusters."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    inertias = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), inertias, marker="o", linewidth=2)
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inercia (WCSS)")
    ax.set_title("Método del codo — selección de k óptimo")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "elbow.png", dpi=150)
    plt.close(fig)
    print("    elbow.png guardado")

{% elif cookiecutter.ml_type == "redes_neuronales" %}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import seaborn as sns

from {{ cookiecutter.project_module_name }}.utils.paths import FIGURES_DIR

plt.style.use("ggplot")


def plot_distributions(df, target_col=None):
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


def plot_correlation_matrix(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def plot_training_history(train_losses, val_losses=None, tb_writer=None):
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
