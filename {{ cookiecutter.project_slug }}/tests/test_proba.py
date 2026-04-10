"""
test_proba.py — Smoke tests: verifican que todos los módulos son importables
y que las funciones principales existen con las firmas esperadas.
"""
import pytest
import inspect


def test_import_data_module():
    from {{ cookiecutter.project_slug }}.data import make_dataset
    assert hasattr(make_dataset, "load_data")


def test_import_features_module():
    from {{ cookiecutter.project_slug }}.features import build_features
    assert hasattr(build_features, "preprocess_data")


def test_import_models_train():
    from {{ cookiecutter.project_slug }}.models import train_model
    assert hasattr(train_model, "train_models")


def test_import_models_predict():
    from {{ cookiecutter.project_slug }}.models import predict_model
    assert hasattr(predict_model, "evaluate_models")


def test_import_visualization():
    from {{ cookiecutter.project_slug }}.visualization import visualize
    assert hasattr(visualize, "plot_distributions")
    assert hasattr(visualize, "plot_correlation_matrix")


def test_import_utils_paths():
    from {{ cookiecutter.project_slug }}.utils import paths
    assert hasattr(paths, "MODELS_DIR")
    assert hasattr(paths, "RAW_DATA_DIR")
    assert hasattr(paths, "FIGURES_DIR")


def test_load_data_signature():
    """load_data debe aceptar un argumento 'filename'."""
    from {{ cookiecutter.project_slug }}.data.make_dataset import load_data
    sig = inspect.signature(load_data)
    assert "filename" in sig.parameters


def test_preprocess_data_signature():
    """preprocess_data debe tener los parámetros mínimos esperados."""
    from {{ cookiecutter.project_slug }}.features.build_features import preprocess_data
    sig = inspect.signature(preprocess_data)
    assert "df" in sig.parameters


{% if cookiecutter.ml_type == "supervisado" %}
def test_train_models_signature():
    from {{ cookiecutter.project_slug }}.models.train_model import train_models
    sig = inspect.signature(train_models)
    assert "X_train" in sig.parameters
    assert "y_train" in sig.parameters


def test_evaluate_models_signature():
    from {{ cookiecutter.project_slug }}.models.predict_model import evaluate_models
    sig = inspect.signature(evaluate_models)
    assert "models" in sig.parameters
{% endif %}


{% if cookiecutter.ml_type == "no_supervisado" %}
def test_find_optimal_k_signature():
    from {{ cookiecutter.project_slug }}.models.train_model import find_optimal_k
    sig = inspect.signature(find_optimal_k)
    assert "X" in sig.parameters
    assert "k_range" in sig.parameters


def test_plot_elbow_signature():
    from {{ cookiecutter.project_slug }}.visualization.visualize import plot_elbow_and_silhouette
    sig = inspect.signature(plot_elbow_and_silhouette)
    assert "metrics" in sig.parameters
{% endif %}


{% if cookiecutter.ml_type == "redes_neuronales" %}
def test_mlp_class_exists():
    from {{ cookiecutter.project_slug }}.models.train_model import MLP
    sig = inspect.signature(MLP.__init__)
    assert "input_dim"  in sig.parameters
    assert "output_dim" in sig.parameters
{% endif %}


{% if cookiecutter.ml_type == "hibrido" %}
def test_train_models_has_strategy_param():
    from {{ cookiecutter.project_slug }}.models.train_model import train_models
    sig = inspect.signature(train_models)
    assert "strategy" in sig.parameters


def test_preprocess_data_has_strategy_param():
    from {{ cookiecutter.project_slug }}.features.build_features import preprocess_data
    sig = inspect.signature(preprocess_data)
    assert "strategy" in sig.parameters
{% endif %}