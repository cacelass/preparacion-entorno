"""
test_predict_model.py — Tests para {{ project_slug }}/models/predict_model.py
"""
import numpy as np
import pandas as pd
import pytest


{% if ml_type in ["supervisado", "hibrido"] %}
from {{ project_slug }}.models.predict_model import (
    evaluate_models,
    predict_new,
{% if task_type == "clasificacion" %}
    predict_proba_new,
    _plot_confusion_matrix,
{% endif %}
)
from {{ project_slug }}.models.train_model import train_models


def _make_data():
{% if task_type == "clasificacion" %}
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = make_classification(
        n_samples=160, n_features=4, n_classes=2, random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.25, random_state=42)
{% else %}
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = make_regression(
        n_samples=160, n_features=4, noise=0.1, random_state=42
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.25, random_state=42)
{% endif %}


def test_evaluate_models_returns_dataframe(patch_paths):
    X_train, X_test, y_train, y_test = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    df_res = evaluate_models(trained, X_train, y_train, X_test, y_test)
    assert isinstance(df_res, pd.DataFrame)
    assert len(df_res) == len(trained)


def test_evaluate_models_columns(patch_paths):
    """El resultado debe contener las columnas de métricas esperadas."""
    X_train, X_test, y_train, y_test = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    df_res = evaluate_models(trained, X_train, y_train, X_test, y_test)
{% if task_type == "clasificacion" %}
    for col in ["Modelo", "Acc_train", "Acc_test", "F1_train", "F1_test"]:
        assert col in df_res.columns, f"Falta columna: {col}"
{% else %}
    for col in ["Modelo", "RMSE_train", "RMSE_test", "MAE_test", "R2_train", "R2_test"]:
        assert col in df_res.columns, f"Falta columna: {col}"
{% endif %}


{% if task_type == "clasificacion" %}
def test_evaluate_models_accuracy_in_range(patch_paths):
    """Accuracy debe estar entre 0 y 1."""
    X_train, X_test, y_train, y_test = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    df_res = evaluate_models(trained, X_train, y_train, X_test, y_test)
    assert (df_res["Acc_test"].between(0, 1)).all()
    assert (df_res["Acc_train"].between(0, 1)).all()


def test_evaluate_models_saves_confusion_matrices(patch_paths):
    """Debe guardar una imagen de matriz de confusión por modelo."""
    X_train, X_test, y_train, y_test = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    evaluate_models(trained, X_train, y_train, X_test, y_test)
    pngs = list(patch_paths["FIGURES_DIR"].glob("cm_*.png"))
    assert len(pngs) == len(trained)
{% else %}
def test_evaluate_models_rmse_positive(patch_paths):
    """RMSE debe ser >= 0."""
    X_train, X_test, y_train, y_test = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    df_res = evaluate_models(trained, X_train, y_train, X_test, y_test)
    assert (df_res["RMSE_test"] >= 0).all()
    assert (df_res["RMSE_train"] >= 0).all()


def test_evaluate_models_saves_regression_plots(patch_paths):
    """Debe guardar real_vs_pred y residuals por modelo."""
    X_train, X_test, y_train, y_test = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    evaluate_models(trained, X_train, y_train, X_test, y_test)
    rvp = list(patch_paths["FIGURES_DIR"].glob("real_vs_pred_*.png"))
    res = list(patch_paths["FIGURES_DIR"].glob("residuals_*.png"))
    assert len(rvp) == len(trained), "Faltan gráficos real_vs_pred"
    assert len(res) == len(trained), "Faltan gráficos residuals"
{% endif %}


def test_evaluate_models_saves_csv(patch_paths):
    """Debe guardar resultados_modelos.csv en REPORTS_DIR."""
    X_train, X_test, y_train, y_test = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    evaluate_models(trained, X_train, y_train, X_test, y_test)
    assert (patch_paths["REPORTS_DIR"] / "resultados_modelos.csv").exists()


def test_predict_new_after_train(patch_paths):
    """predict_new debe cargar el modelo y predecir correctamente."""
    X_train, X_test, y_train, _ = _make_data()
    train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    preds = predict_new("RandomForest", X_test)
    assert len(preds) == len(X_test)
{% if task_type == "clasificacion" %}
    assert set(preds).issubset({0, 1})
{% else %}
    assert np.issubdtype(preds.dtype, np.floating)
{% endif %}


def test_predict_new_raises_if_missing(patch_paths):
    """predict_new debe lanzar FileNotFoundError si el modelo no existe."""
    with pytest.raises(FileNotFoundError):
        predict_new("ModeloInexistente", np.zeros((5, 4)))


{% if task_type == "clasificacion" %}
def test_predict_proba_new_shape(patch_paths):
    """predict_proba_new debe devolver array (n_samples, n_classes)."""
    X_train, X_test, y_train, _ = _make_data()
    train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    proba = predict_proba_new("RandomForest", X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_predict_proba_new_raises_for_no_proba(patch_paths):
    """predict_proba_new debe fallar si el modelo no existe."""
    with pytest.raises(Exception):
        predict_proba_new("ModeloSinProba", np.zeros((5, 4)))


def test_custom_threshold(patch_paths):
    """Con threshold distinto de 0.5 debe aplicarse sobre predict_proba."""
    X_train, X_test, y_train, y_test = _make_data()
    trained = train_models(X_train, y_train, tune_knn=False, cv_evaluate=False)
    df_low  = evaluate_models(trained, X_train, y_train, X_test, y_test, threshold=0.3)
    df_high = evaluate_models(trained, X_train, y_train, X_test, y_test, threshold=0.7)
    assert isinstance(df_low,  pd.DataFrame)
    assert isinstance(df_high, pd.DataFrame)
{% endif %}
{% endif %}


{% if ml_type == "no_supervisado" %}
from {{ project_slug }}.models.predict_model import evaluate_models, plot_dendrogram
from {{ project_slug }}.models.train_model import train_models


def _make_X():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=150, centers=3, n_features=4, random_state=42)
    return X


def test_evaluate_models_returns_dataframe(patch_paths):
    X = _make_X()
    fitted = train_models(X, n_clusters=3)
    df_res = evaluate_models(fitted, X)
    assert isinstance(df_res, pd.DataFrame)


def test_evaluate_models_clustering_columns(patch_paths):
    X = _make_X()
    fitted = train_models(X, n_clusters=3)
    df_res = evaluate_models(fitted, X)
    for col in ["Modelo", "Silhouette", "Davies_Bouldin"]:
        assert col in df_res.columns, f"Falta columna: {col}"


def test_evaluate_models_silhouette_range(patch_paths):
    X = _make_X()
    fitted = train_models(X, n_clusters=3)
    df_res = evaluate_models(fitted, X)
    assert (df_res["Silhouette"].between(-1, 1)).all()


def test_evaluate_models_saves_pca_plots(patch_paths):
    """Debe guardar proyecciones PCA 2D por modelo."""
    X = _make_X()
    fitted = train_models(X, n_clusters=3)
    evaluate_models(fitted, X)
    pngs = list(patch_paths["FIGURES_DIR"].glob("clusters_*_pca.png"))
    assert len(pngs) == len(fitted)


def test_evaluate_models_saves_csv(patch_paths):
    X = _make_X()
    fitted = train_models(X, n_clusters=3)
    evaluate_models(fitted, X)
    assert (patch_paths["FIGURES_DIR"] / "resultados_clustering.csv").exists()


def test_plot_dendrogram_saves_png(patch_paths):
    X = _make_X()
    plot_dendrogram(X, method="ward")
    assert (patch_paths["FIGURES_DIR"] / "dendrogram.png").exists()
{% endif %}


{% if ml_type == "redes_neuronales" %}
torch = pytest.importorskip("torch")
from {{ project_slug }}.models.predict_model import evaluate_models, predict_new
from {{ project_slug }}.models.train_model import MLP, train_models


def _make_splits(n=200, n_feat=4):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(n, n_feat),
                     columns=[f"feat_{i}" for i in range(n_feat)])
    y = pd.Series((X["feat_0"] + X["feat_1"] > 0).astype(int), name="target")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    return X_train_s, X_test_s, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def test_evaluate_models_returns_dataframe(patch_paths):
    X_train, X_test, y_train, y_test = _make_splits()
    trained = train_models(
        X_train, y_train,
        input_dim=X_train.shape[1], output_dim=y_train.nunique(),
        epochs=2, batch_size=32, checkpoint_every=0,
    )
    df_res = evaluate_models(trained, X_test, y_test, num_classes=y_train.nunique())
    assert isinstance(df_res, pd.DataFrame)
    assert "MLP" in df_res["Modelo"].values


def test_evaluate_models_columns(patch_paths):
    X_train, X_test, y_train, y_test = _make_splits()
    trained = train_models(
        X_train, y_train,
        input_dim=X_train.shape[1], output_dim=y_train.nunique(),
        epochs=2, batch_size=32, checkpoint_every=0,
    )
    df_res = evaluate_models(trained, X_test, y_test, num_classes=y_train.nunique())
    for col in ["Modelo", "Accuracy", "F1", "Precision", "Recall"]:
        assert col in df_res.columns


def test_evaluate_models_accuracy_range(patch_paths):
    X_train, X_test, y_train, y_test = _make_splits()
    trained = train_models(
        X_train, y_train,
        input_dim=X_train.shape[1], output_dim=y_train.nunique(),
        epochs=2, batch_size=32, checkpoint_every=0,
    )
    df_res = evaluate_models(trained, X_test, y_test, num_classes=y_train.nunique())
    assert (df_res["Accuracy"].between(0, 1)).all()


def test_evaluate_models_saves_confusion_matrix_png(patch_paths):
    X_train, X_test, y_train, y_test = _make_splits()
    trained = train_models(
        X_train, y_train,
        input_dim=X_train.shape[1], output_dim=y_train.nunique(),
        epochs=2, batch_size=32, checkpoint_every=0,
    )
    evaluate_models(trained, X_test, y_test, num_classes=y_train.nunique())
    assert list(patch_paths["FIGURES_DIR"].glob("cm_MLP.png"))


def test_predict_new_returns_predictions(patch_paths):
    X_train, X_test, y_train, y_test = _make_splits()
    trained = train_models(
        X_train, y_train,
        input_dim=X_train.shape[1], output_dim=y_train.nunique(),
        epochs=2, batch_size=32, checkpoint_every=0,
    )
    model = trained["MLP"]
    preds = predict_new(model, X_test, num_classes=y_train.nunique())
    assert len(preds) == len(X_test)
    assert set(preds).issubset(set(y_train.unique()))
{% endif %}
