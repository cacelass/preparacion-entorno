"""
test_train_model.py — Tests para {{ project_slug }}/models/train_model.py
"""
import numpy as np
import pandas as pd
import pytest


{% if ml_type == "supervisado" %}
from {{ project_slug }}.models.train_model import (
    _build_models,
    _find_best_k,
    train_models,
    load_models,
)


def _make_Xy():
    """Datos sintéticos pequeños para entrenamiento rápido en tests."""
{% if task_type == "clasificacion" %}
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=120, n_features=4, n_classes=2, random_state=42
    )
{% else %}
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=120, n_features=4, noise=0.1, random_state=42
    )
{% endif %}
    return X, y


def test_build_models_returns_dict():
    models = _build_models()
    assert isinstance(models, dict)
    assert len(models) > 0


def test_build_models_expected_keys():
    models = _build_models()
{% if task_type == "clasificacion" %}
{% if model_type == "todos" %}
    for key in ["KNN", "RandomForest", "LogisticRegression"]:
        assert key in models, f"Falta modelo: {key}"
{% else %}
    assert "{{ model_type }}" in models
{% endif %}
{% else %}
{% if model_type == "todos" %}
    for key in ["LinearRegression", "Ridge", "Lasso", "KNN", "RandomForest"]:
        assert key in models, f"Falta modelo: {key}"
{% else %}
    assert "{{ model_type }}" in models
{% endif %}
{% endif %}


{% if model_type == "todos" or model_type == "KNN" %}
def test_find_best_k_returns_int_in_range():
    X, y = _make_Xy()
    best_k = _find_best_k(X, y, k_range=range(1, 6))
    assert isinstance(best_k, int)
    assert 1 <= best_k <= 5

{% endif %}
def test_train_models_returns_trained_dict(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, tune_knn=False, cv_evaluate=False)
    assert isinstance(trained, dict)
    assert len(trained) > 0
    for model in trained.values():
        assert hasattr(model, "predict")


def test_train_models_saves_joblib_files(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, tune_knn=False, cv_evaluate=False)
    saved = list(patch_paths["MODELS_DIR"].glob("*.joblib"))
    assert len(saved) == len(trained)
    for name in trained:
        assert (patch_paths["MODELS_DIR"] / f"{name}.joblib").exists()


def test_load_models_loads_saved(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, tune_knn=False, cv_evaluate=False)
    loaded  = load_models()
    assert set(loaded.keys()) == set(trained.keys())
    for model in loaded.values():
        assert hasattr(model, "predict")


{% if model_type == "todos" or model_type == "RandomForest" %}
def test_load_models_specific_names(patch_paths):
    X, y = _make_Xy()
    train_models(X, y, tune_knn=False, cv_evaluate=False)
    loaded = load_models(["RandomForest"])
    assert "RandomForest" in loaded

{% endif %}
def test_load_models_missing_returns_empty(patch_paths):
    """Si no hay modelos guardados, load_models() debe devolver dict vacío."""
    loaded = load_models(["ModeloInexistente"])
    assert loaded == {}


def test_models_can_predict(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, tune_knn=False, cv_evaluate=False)
    for name, model in trained.items():
        preds = model.predict(X)
        assert len(preds) == len(y), f"{name}: longitud incorrecta"
{% if task_type == "clasificacion" %}
        assert set(preds).issubset({0, 1}), f"{name}: predicciones fuera de clases"
{% else %}
        assert np.issubdtype(preds.dtype, np.floating), f"{name}: se esperaban floats"
{% endif %}
{% endif %}


{% if ml_type == "no_supervisado" %}
from {{ project_slug }}.models.train_model import (
    _build_models,
    find_optimal_k,
    train_models,
    train_kmeans_pipeline,
    load_models,
)


def _make_X():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=150, centers=3, n_features=4, random_state=42)
    return X


def test_build_models_returns_dict():
    models = _build_models(n_clusters=3)
    assert isinstance(models, dict)
    assert "KMeans" in models
    assert "AgglomerativeClustering" in models


def test_find_optimal_k_returns_expected_keys():
    X = _make_X()
    result = find_optimal_k(X, k_range=range(2, 5))
    for key in ["k_range", "inertias", "silhouettes", "db_scores", "ch_scores"]:
        assert key in result, f"Falta clave: {key}"


def test_find_optimal_k_lengths_match():
    X = _make_X()
    result = find_optimal_k(X, k_range=range(2, 5))
    n = len(result["k_range"])
    for key in ["inertias", "silhouettes", "db_scores", "ch_scores"]:
        assert len(result[key]) == n, f"{key}: longitud incorrecta"


def test_find_optimal_k_silhouette_range():
    """Silhouette score debe estar en [-1, 1]."""
    X = _make_X()
    result = find_optimal_k(X, k_range=range(2, 4))
    for s in result["silhouettes"]:
        assert -1.0 <= s <= 1.0


def test_train_models_fits_and_saves(patch_paths):
    X = _make_X()
    fitted = train_models(X, n_clusters=3)
    assert len(fitted) > 0
    for name in fitted:
        assert (patch_paths["MODELS_DIR"] / f"{name}.joblib").exists()


def test_train_models_labels_attribute(patch_paths):
    """Los modelos ajustados deben tener el atributo labels_."""
    X = _make_X()
    fitted = train_models(X, n_clusters=3)
    for name, model in fitted.items():
        assert hasattr(model, "labels_"), f"{name} debe tener labels_"
        assert len(model.labels_) == len(X)


{% if cluster_model == "todos" or cluster_model == "KMeans" %}
def test_train_kmeans_pipeline(patch_paths):
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=120, n_features=4, n_classes=2, random_state=42
    )
    pipeline = train_kmeans_pipeline(X, y, n_clusters=5)
    assert hasattr(pipeline, "predict")
    assert (patch_paths["MODELS_DIR"] / "KMeansPipeline.joblib").exists()

{% endif %}
def test_load_models_after_train(patch_paths):
    X = _make_X()
    fitted  = train_models(X, n_clusters=3)
    loaded  = load_models()
    assert set(loaded.keys()) == set(fitted.keys())
{% endif %}


{% if ml_type == "redes_neuronales" %}
torch = pytest.importorskip("torch")
from {{ project_slug }}.models.train_model import (
    MLP, CNN1D, LSTMClassifier, GRUClassifier, TransformerClassifier,
    _build_model, train_models, load_model, MODEL_NAME,
)

INPUT_DIM  = 8
OUTPUT_DIM = 3
BATCH      = 10


# ─── Forward pass por arquitectura ─────────────────────────────────────────

def test_mlp_forward():
    model = MLP(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dims=[16, 8])
    out   = model(torch.randn(BATCH, INPUT_DIM))
    assert out.shape == (BATCH, OUTPUT_DIM)


def test_cnn1d_forward():
    model = CNN1D(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    out   = model(torch.randn(BATCH, INPUT_DIM))
    assert out.shape == (BATCH, OUTPUT_DIM)
{% endif %}


{% if ml_type == "hibrido" %}
from {{ project_slug }}.models.train_model import (
    _build_models,
    _find_best_k,
    train_models,
    load_models,
)


def _make_Xy():
    """Datos sintéticos pequeños para entrenamiento rápido en tests."""
{% if task_type == "clasificacion" %}
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=120, n_features=4, n_classes=2, random_state=42
    )
{% else %}
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=120, n_features=4, noise=0.1, random_state=42
    )
{% endif %}
    return X, y


def test_build_models_returns_dict():
    models = _build_models()
    assert isinstance(models, dict)
    assert len(models) > 0


def test_train_models_returns_trained_dict(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, tune_knn=False, cv_evaluate=False)
    assert isinstance(trained, dict)
    assert len(trained) > 0
    for model in trained.values():
        assert hasattr(model, "predict")


def test_train_models_saves_joblib_files(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, tune_knn=False, cv_evaluate=False)
    for name in trained:
        assert (patch_paths["MODELS_DIR"] / f"{name}.joblib").exists()


def test_load_models_loads_saved(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, tune_knn=False, cv_evaluate=False)
    loaded  = load_models()
    assert set(loaded.keys()) == set(trained.keys())


def test_models_can_predict(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, tune_knn=False, cv_evaluate=False)
    for name, model in trained.items():
        preds = model.predict(X)
        assert len(preds) == len(y), f"{name}: longitud incorrecta"
{% if task_type == "clasificacion" %}
        assert set(preds).issubset({0, 1}), f"{name}: predicciones fuera de clases"
{% else %}
        assert np.issubdtype(preds.dtype, np.floating), f"{name}: se esperaban floats"
{% endif %}
{% endif %}