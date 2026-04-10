"""
test_train_model.py — Tests para {{ cookiecutter.project_slug }}/models/train_model.py
"""
import numpy as np
import pandas as pd
import pytest


{% if cookiecutter.ml_type == "supervisado" %}
from {{ cookiecutter.project_slug }}.models.train_model import (
    _build_models,
    _find_best_k,
    train_models,
    load_models,
)


def _make_Xy():
    """Datos sintéticos pequeños para entrenamiento rápido en tests."""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=120, n_features=4, n_classes=2, random_state=42
    )
    return X, y


def test_build_models_returns_dict():
    models = _build_models()
    assert isinstance(models, dict)
    assert len(models) > 0


def test_build_models_expected_keys():
    models = _build_models()
    for key in ["KNN", "RandomForest", "LogisticRegression"]:
        assert key in models, f"Falta modelo: {key}"


def test_find_best_k_returns_int_in_range():
    X, y = _make_Xy()
    best_k = _find_best_k(X, y, k_range=range(1, 6))
    assert isinstance(best_k, int)
    assert 1 <= best_k <= 5


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


def test_load_models_specific_names(patch_paths):
    X, y = _make_Xy()
    train_models(X, y, tune_knn=False, cv_evaluate=False)
    loaded = load_models(["RandomForest"])
    assert "RandomForest" in loaded


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
        assert set(preds).issubset({0, 1}), f"{name}: predicciones fuera de clases"
{% endif %}


{% if cookiecutter.ml_type == "no_supervisado" %}
from {{ cookiecutter.project_slug }}.models.train_model import (
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


def test_train_kmeans_pipeline(patch_paths):
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=120, n_features=4, n_classes=2, random_state=42
    )
    pipeline = train_kmeans_pipeline(X, y, n_clusters=5)
    assert hasattr(pipeline, "predict")
    assert (patch_paths["MODELS_DIR"] / "KMeansPipeline.joblib").exists()


def test_load_models_after_train(patch_paths):
    X = _make_X()
    fitted  = train_models(X, n_clusters=3)
    loaded  = load_models()
    assert set(loaded.keys()) == set(fitted.keys())
{% endif %}


{% if cookiecutter.ml_type == "redes_neuronales" %}
torch = pytest.importorskip("torch")
from {{ cookiecutter.project_slug }}.models.train_model import MLP, train_models, load_model


def test_mlp_forward_pass():
    """MLP debe procesar un batch sin errores."""
    model = MLP(input_dim=8, output_dim=3, hidden_dims=[16, 8])
    x = torch.randn(10, 8)
    out = model(x)
    assert out.shape == (10, 3)


def test_mlp_hidden_dims_custom():
    """MLP con hidden_dims personalizados debe construirse correctamente."""
    model = MLP(input_dim=4, output_dim=2, hidden_dims=[32, 16, 8])
    x = torch.randn(5, 4)
    out = model(x)
    assert out.shape == (5, 2)


def test_mlp_default_hidden_dims():
    """MLP sin hidden_dims debe usar [128, 64]."""
    model = MLP(input_dim=4, output_dim=2)
    x = torch.randn(3, 4)
    out = model(x)
    assert out.shape == (3, 2)


def test_train_models_saves_weights(df_with_target, patch_paths):
    """train_models debe guardar MLP.pt en MODELS_DIR."""
    X_train = df_with_target.drop(columns=["target"])
    y_train = df_with_target["target"]
    input_dim  = X_train.shape[1]
    output_dim = y_train.nunique()

    train_models(
        X_train, y_train,
        input_dim=input_dim, output_dim=output_dim,
        epochs=2, batch_size=32, checkpoint_every=0,
    )
    assert (patch_paths["MODELS_DIR"] / "MLP.pt").exists()


def test_train_models_returns_mlp_key(df_with_target, patch_paths):
    """train_models debe devolver dict con clave 'MLP'."""
    X_train = df_with_target.drop(columns=["target"])
    y_train = df_with_target["target"]
    result = train_models(
        X_train, y_train,
        input_dim=X_train.shape[1], output_dim=y_train.nunique(),
        epochs=2, batch_size=32, checkpoint_every=0,
    )
    assert "MLP" in result


def test_load_model_after_training(df_with_target, patch_paths):
    """load_model debe cargar los pesos guardados por train_models."""
    X_train = df_with_target.drop(columns=["target"])
    y_train = df_with_target["target"]
    input_dim  = X_train.shape[1]
    output_dim = y_train.nunique()

    train_models(
        X_train, y_train,
        input_dim=input_dim, output_dim=output_dim,
        epochs=2, batch_size=32, checkpoint_every=0,
    )
    model = load_model(input_dim, output_dim, weights_path="MLP.pt")
    assert isinstance(model, MLP)
    # Debe estar en modo eval
    assert not model.training


def test_checkpoint_saved_every_n_epochs(df_with_target, patch_paths):
    """Si checkpoint_every > 0, debe guardar archivos checkpoint-N.pt."""
    X_train = df_with_target.drop(columns=["target"])
    y_train = df_with_target["target"]
    train_models(
        X_train, y_train,
        input_dim=X_train.shape[1], output_dim=y_train.nunique(),
        epochs=4, batch_size=32, checkpoint_every=2,
    )
    checkpoints = list(patch_paths["MODELS_DIR"].glob("checkpoint-*.pt"))
    assert len(checkpoints) >= 1
{% endif %}


{% if cookiecutter.ml_type == "hibrido" %}
from {{ cookiecutter.project_slug }}.models.train_model import (
    _build_models,
    _find_best_k,
    train_models,
    load_models,
)


def _make_Xy():
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=120, n_features=4, n_classes=2, random_state=42
    )
    return X, y


def test_build_models_pca_clf():
    """Estrategia pca_clf debe incluir SVM_RBF además de los modelos base."""
    models = _build_models(strategy="pca_clf")
    assert "SVM_RBF" in models
    for key in ["LogisticRegression", "RandomForest", "KNN"]:
        assert key in models


def test_build_models_kmeans_features():
    """Estrategia kmeans_features debe incluir GradientBoosting."""
    models = _build_models(strategy="kmeans_features")
    assert "GradientBoosting" in models


def test_build_models_base_strategy():
    """Estrategia semi_supervisado solo debe tener los modelos base."""
    models = _build_models(strategy="semi_supervisado")
    assert "SVM_RBF"          not in models
    assert "GradientBoosting" not in models
    assert "LogisticRegression" in models


def test_find_best_k_returns_valid_int():
    X, y = _make_Xy()
    best_k = _find_best_k(X, y, k_range=range(1, 6))
    assert isinstance(best_k, int)
    assert 1 <= best_k <= 5


def test_train_models_pca_clf_trains(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, strategy="pca_clf",
                           tune_knn=False, cv_evaluate=False)
    assert len(trained) > 0
    for name in trained:
        assert (patch_paths["MODELS_DIR"] / f"{name}.joblib").exists()


def test_train_models_kmeans_features_trains(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, strategy="kmeans_features",
                           tune_knn=False, cv_evaluate=False)
    assert "GradientBoosting" in trained


def test_load_models_after_train(patch_paths):
    X, y = _make_Xy()
    trained = train_models(X, y, strategy="pca_clf",
                           tune_knn=False, cv_evaluate=False)
    loaded = load_models()
    assert set(loaded.keys()) == set(trained.keys())
{% endif %}
