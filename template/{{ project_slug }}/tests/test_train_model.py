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


def test_lstm_forward_tabular():
    """LSTM con datos tabulares (sin dimensión temporal explícita)."""
    model = LSTMClassifier(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)
    out   = model(torch.randn(BATCH, INPUT_DIM))
    assert out.shape == (BATCH, OUTPUT_DIM)


def test_lstm_forward_sequential():
    """LSTM con datos secuenciales (batch, seq_len, features)."""
    model = LSTMClassifier(input_dim=4, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)
    out   = model(torch.randn(BATCH, 5, 4))   # seq_len=5
    assert out.shape == (BATCH, OUTPUT_DIM)


def test_gru_forward():
    model = GRUClassifier(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dim=16, num_layers=1)
    out   = model(torch.randn(BATCH, INPUT_DIM))
    assert out.shape == (BATCH, OUTPUT_DIM)


def test_transformer_forward():
    model = TransformerClassifier(
        input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
        d_model=16, nhead=2, num_layers=1, dim_ff=32,
    )
    out = model(torch.randn(BATCH, INPUT_DIM))
    assert out.shape == (BATCH, OUTPUT_DIM)


def test_transformer_nhead_must_divide_d_model():
    """d_model no divisible por nhead debe lanzar AssertionError."""
    with pytest.raises(AssertionError):
        TransformerClassifier(input_dim=INPUT_DIM, output_dim=2, d_model=10, nhead=3)


# ─── Fábrica _build_model ──────────────────────────────────────────────────

def test_build_model_returns_correct_class():
    """_build_model() debe devolver la clase correspondiente a MODEL_NAME."""
    model = _build_model(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    class_map = {
        "MLP":         MLP,
        "CNN1D":       CNN1D,
        "LSTM":        LSTMClassifier,
        "GRU":         GRUClassifier,
        "Transformer": TransformerClassifier,
    }
    expected = class_map[MODEL_NAME]
    assert isinstance(model, expected), (
        f"Se esperaba {expected.__name__} para nn_model='{MODEL_NAME}', "
        f"pero se obtuvo {type(model).__name__}"
    )


# ─── Entrenamiento y persistencia ──────────────────────────────────────────

@pytest.mark.smoke
def test_train_saves_weights(df_with_target, patch_paths):
    """train_models debe guardar {MODEL_NAME}.pt en MODELS_DIR."""
    X = df_with_target.drop(columns=["target"])
    y = df_with_target["target"]
    train_models(X, y, input_dim=X.shape[1], output_dim=y.nunique(),
                 epochs=2, batch_size=16, checkpoint_every=0)
    assert (patch_paths["MODELS_DIR"] / f"{MODEL_NAME}.pt").exists()


@pytest.mark.smoke
def test_train_returns_model_name_key(df_with_target, patch_paths):
    """train_models debe devolver dict con clave MODEL_NAME."""
    X = df_with_target.drop(columns=["target"])
    y = df_with_target["target"]
    result = train_models(X, y, input_dim=X.shape[1], output_dim=y.nunique(),
                          epochs=2, batch_size=16, checkpoint_every=0)
    assert MODEL_NAME in result


@pytest.mark.smoke
def test_load_model_eval_mode(df_with_target, patch_paths):
    """load_model debe cargar pesos y dejar el modelo en modo eval."""
    X = df_with_target.drop(columns=["target"])
    y = df_with_target["target"]
    input_dim, output_dim = X.shape[1], y.nunique()
    train_models(X, y, input_dim=input_dim, output_dim=output_dim,
                 epochs=2, batch_size=16, checkpoint_every=0)
    model = load_model(input_dim, output_dim, weights_path=f"{MODEL_NAME}.pt")
    assert not model.training


@pytest.mark.smoke
def test_checkpoint_created(df_with_target, patch_paths):
    """checkpoint_every=2 debe crear al menos un checkpoint-*.pt."""
    X = df_with_target.drop(columns=["target"])
    y = df_with_target["target"]
    train_models(X, y, input_dim=X.shape[1], output_dim=y.nunique(),
                 epochs=4, batch_size=16, checkpoint_every=2)
    checkpoints = list(patch_paths["MODELS_DIR"].glob("checkpoint-*.pt"))
    assert len(checkpoints) >= 1


def test_model_output_has_no_nan(df_with_target, patch_paths):
    """El forward pass no debe producir NaN en la salida."""
    X = df_with_target.drop(columns=["target"])
    y = df_with_target["target"]
    result = train_models(X, y, input_dim=X.shape[1], output_dim=y.nunique(),
                          epochs=1, batch_size=16, checkpoint_every=0)
    model = result[MODEL_NAME]
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X.values, dtype=torch.float32)
        out = model(x_t)
    assert not torch.isnan(out).any(), "Forward pass produjo NaN"
{% endif %}


{% if ml_type == "hibrido" %}
from {{ project_slug }}.models.train_model import (
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
