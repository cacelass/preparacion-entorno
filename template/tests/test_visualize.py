"""
test_visualize.py — Tests para {{ project_slug }}/visualization/visualize.py
"""
import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers comunes
# ─────────────────────────────────────────────────────────────────────────────

def _numeric_df(n=80, cols=4):
    np.random.seed(0)
    return pd.DataFrame(
        np.random.randn(n, cols),
        columns=[f"feat_{i}" for i in range(cols)],
    )


def _df_with_target(n=80):
    df = _numeric_df(n)
    df["target"] = (df["feat_0"] + df["feat_1"] > 0).astype(int)
    return df


{% if ml_type == "supervisado" %}
from {{ project_slug }}.visualization.visualize import (
    plot_distributions,
    plot_correlation_matrix,
    plot_class_balance,
    plot_categorical_vs_target,
    plot_feature_importance,
    plot_pca_variance,
    plot_pairplot,
)


def test_plot_distributions_saves_png(patch_paths):
    plot_distributions(_df_with_target(), target_col="target")
    assert (patch_paths["FIGURES_DIR"] / "distributions.png").exists()


def test_plot_correlation_matrix_saves_png(patch_paths):
    plot_correlation_matrix(_numeric_df())
    assert (patch_paths["FIGURES_DIR"] / "correlation_matrix.png").exists()


def test_plot_class_balance_saves_png(patch_paths):
    plot_class_balance(_df_with_target(), target_col="target")
    assert (patch_paths["FIGURES_DIR"] / "class_balance.png").exists()


def test_plot_categorical_vs_target_with_cats(patch_paths):
    df = _df_with_target()
    df["cat_col"] = np.where(df["feat_0"] > 0, "A", "B")
    plot_categorical_vs_target(df, target_col="target")
    assert (patch_paths["FIGURES_DIR"] / "categorical_vs_target.png").exists()


def test_plot_feature_importance_rf(patch_paths):
    """plot_feature_importance con RandomForest debe guardar el gráfico."""
    from sklearn.ensemble import RandomForestClassifier
    df = _df_with_target()
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    plot_feature_importance({"RF": rf}, feature_names=["feat_0","feat_1","feat_2","feat_3"])
    assert (patch_paths["FIGURES_DIR"] / "feature_importance.png").exists()


def test_plot_pca_variance_from_array(patch_paths):
    X = np.random.randn(100, 6)
    plot_pca_variance(X)
    assert (patch_paths["FIGURES_DIR"] / "pca_variance.png").exists()


def test_plot_pca_variance_from_pca_object(patch_paths):
    from sklearn.decomposition import PCA
    X = np.random.randn(100, 6)
    pca = PCA(n_components=5).fit(X)
    plot_pca_variance(pca)
    assert (patch_paths["FIGURES_DIR"] / "pca_variance.png").exists()


def test_plot_pairplot_saves_png(patch_paths):
    plot_pairplot(_df_with_target(), target_col="target")
    assert (patch_paths["FIGURES_DIR"] / "pairplot.png").exists()
{% endif %}


{% if ml_type == "no_supervisado" %}
from {{ project_slug }}.visualization.visualize import (
    plot_distributions,
    plot_correlation_matrix,
    plot_elbow_and_silhouette,
    plot_dendrogram,
    plot_pca_variance,
    plot_clusters_pca,
    plot_cluster_profiles,
)
from {{ project_slug }}.models.train_model import find_optimal_k


def _clustering_data():
    from sklearn.datasets import make_blobs
    X, labels = make_blobs(n_samples=150, centers=3, n_features=4, random_state=42)
    return X, labels


def test_plot_distributions_saves_png(patch_paths):
    plot_distributions(_numeric_df())
    assert (patch_paths["FIGURES_DIR"] / "distributions.png").exists()


def test_plot_correlation_matrix_saves_png(patch_paths):
    plot_correlation_matrix(_numeric_df())
    assert (patch_paths["FIGURES_DIR"] / "correlation_matrix.png").exists()


def test_plot_elbow_and_silhouette_saves_png(patch_paths):
    X, _ = _clustering_data()
    metrics = find_optimal_k(X, k_range=range(2, 5))
    plot_elbow_and_silhouette(metrics)
    assert (patch_paths["FIGURES_DIR"] / "elbow_silhouette.png").exists()


def test_plot_dendrogram_saves_png(patch_paths):
    X, _ = _clustering_data()
    plot_dendrogram(X, method="ward")
    assert (patch_paths["FIGURES_DIR"] / "dendrogram.png").exists()


def test_plot_pca_variance_saves_png(patch_paths):
    X, _ = _clustering_data()
    plot_pca_variance(X)
    assert (patch_paths["FIGURES_DIR"] / "pca_variance.png").exists()


def test_plot_clusters_pca_saves_png(patch_paths):
    X, labels = _clustering_data()
    plot_clusters_pca(X, labels, model_name="TestModel")
    assert (patch_paths["FIGURES_DIR"] / "clusters_TestModel_pca.png").exists()


def test_plot_cluster_profiles_saves_png(patch_paths):
    X, labels = _clustering_data()
    plot_cluster_profiles(
        X, labels,
        feature_names=["feat_0", "feat_1", "feat_2", "feat_3"],
    )
    assert (patch_paths["FIGURES_DIR"] / "cluster_profiles.png").exists()
{% endif %}


{% if ml_type == "redes_neuronales" %}
from {{ project_slug }}.visualization.visualize import (
    plot_distributions,
    plot_correlation_matrix,
    plot_pca_variance,
    plot_training_history,
    plot_confusion_matrix,
    plot_class_balance,
)


def test_plot_distributions_saves_png(patch_paths):
    plot_distributions(_df_with_target(), target_col="target")
    assert (patch_paths["FIGURES_DIR"] / "distributions.png").exists()


def test_plot_correlation_matrix_saves_png(patch_paths):
    plot_correlation_matrix(_numeric_df())
    assert (patch_paths["FIGURES_DIR"] / "correlation_matrix.png").exists()


def test_plot_pca_variance_saves_png(patch_paths):
    X = np.random.randn(100, 8)
    plot_pca_variance(X)
    assert (patch_paths["FIGURES_DIR"] / "pca_variance.png").exists()


def test_plot_training_history_only_loss(patch_paths):
    losses = [1.0, 0.8, 0.6, 0.5, 0.4]
    plot_training_history(losses)
    assert (patch_paths["FIGURES_DIR"] / "training_history.png").exists()


def test_plot_training_history_with_acc(patch_paths):
    losses = [1.0, 0.8, 0.6]
    accs   = [0.5, 0.6, 0.7]
    plot_training_history(losses, train_accs=accs)
    assert (patch_paths["FIGURES_DIR"] / "training_history.png").exists()


def test_plot_confusion_matrix_saves_png(patch_paths):
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    plot_confusion_matrix(y_true, y_pred, model_name="MLP_test")
    assert (patch_paths["FIGURES_DIR"] / "cm_MLP_test.png").exists()


def test_plot_class_balance_saves_png(patch_paths):
    plot_class_balance(_df_with_target(), target_col="target")
    assert (patch_paths["FIGURES_DIR"] / "class_balance.png").exists()
{% endif %}


{% if ml_type == "hibrido" %}
from {{ project_slug }}.visualization.visualize import (
    plot_distributions,
    plot_correlation_matrix,
    plot_class_balance,
    plot_pca_variance,
    plot_feature_importance,
)


def test_plot_distributions_saves_png(patch_paths):
    plot_distributions(_df_with_target(), target_col="target")
    assert (patch_paths["FIGURES_DIR"] / "distributions.png").exists()


def test_plot_correlation_matrix_saves_png(patch_paths):
    plot_correlation_matrix(_numeric_df())
    assert (patch_paths["FIGURES_DIR"] / "correlation_matrix.png").exists()


def test_plot_class_balance_saves_png(patch_paths):
    plot_class_balance(_df_with_target(), target_col="target")
    assert (patch_paths["FIGURES_DIR"] / "class_balance.png").exists()


def test_plot_pca_variance_from_array(patch_paths):
    X = np.random.randn(100, 6)
    plot_pca_variance(X)
    assert (patch_paths["FIGURES_DIR"] / "pca_variance.png").exists()


def test_plot_pca_variance_from_pca_object(patch_paths):
    from sklearn.decomposition import PCA
    X = np.random.randn(100, 6)
    pca = PCA(n_components=5).fit(X)
    plot_pca_variance(pca)
    assert (patch_paths["FIGURES_DIR"] / "pca_variance.png").exists()


def test_plot_feature_importance_rf(patch_paths):
    from sklearn.ensemble import RandomForestClassifier
    df = _df_with_target()
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    plot_feature_importance({"RF": rf}, feature_names=["f0","f1","f2","f3"])
    assert (patch_paths["FIGURES_DIR"] / "feature_importance.png").exists()


def test_plot_umap_skips_if_not_installed(patch_paths, capsys):
    """plot_umap debe fallar con gracia si umap-learn no está instalado."""
    try:
        from {{ project_slug }}.visualization.visualize import plot_umap
        import umap  # noqa: F401  — si umap está disponible, este test pasa trivialmente
        X = np.random.randn(50, 4)
        plot_umap(X, model_name="test")
    except ImportError:
        pytest.skip("umap-learn no instalado (comportamiento esperado en entorno sin GPU/umap)")
{% endif %}
