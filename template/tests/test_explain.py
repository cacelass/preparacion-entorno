{% if ml_type == "redes_neuronales" %}
"""
test_explain.py — Tests para Captum explainability.
"""
import numpy as np
import pytest
torch = pytest.importorskip("torch")
captum = pytest.importorskip("captum")

from {{ project_slug }}.models.explain import explain_model, summarize_attributions
from {{ project_slug }}.models.train_model import (
    MLP, CNN1D, LSTMClassifier, GRUClassifier, TransformerClassifier,
)

INPUT_DIM  = 8
OUTPUT_DIM = 3
BATCH      = 4


def _make_model(arch_class, **kwargs):
    return arch_class(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, **kwargs)


def _make_input():
    return torch.randn(BATCH, INPUT_DIM)


def _make_seq_input():
    return torch.randn(BATCH, 5, INPUT_DIM)  # seq_len=5


@pytest.mark.parametrize("arch,name", [
    (MLP, "MLP"),
    (CNN1D, "CNN1D"),
    (LSTMClassifier, "LSTM"),
    (GRUClassifier, "GRU"),
    (TransformerClassifier, "Transformer"),
])
def test_explain_model_returns_correct_shape(arch, name):
    model = _make_model(arch)
    model.eval()
    X = _make_input()
    if name in ("LSTM", "GRU", "Transformer"):
        X = _make_seq_input()
    attr = explain_model(model, X, target=0)
    assert attr.shape == X.shape, f"{name}: esperado {X.shape}, got {attr.shape}"


@pytest.mark.parametrize("arch,name", [
    (MLP, "MLP"),
    (CNN1D, "CNN1D"),
    (LSTMClassifier, "LSTM"),
])
def test_summarize_attributions_shape(arch, name):
    model = _make_model(arch)
    model.eval()
    X = _make_input()
    if name == "LSTM":
        X = _make_seq_input()
    attr = explain_model(model, X, target=0)
    imp = summarize_attributions(attr, agg="absmean")
    assert imp.ndim == 1, f"{name}: esperado 1D, got {imp.ndim}"
    assert imp.shape[0] == INPUT_DIM, f"{name}: esperado {INPUT_DIM}, got {imp.shape[0]}"


def test_explain_model_no_target_auto_selects():
    model = _make_model(MLP)
    model.eval()
    X = _make_input()
    attr = explain_model(model, X, target=None)
    assert attr.shape == X.shape


def test_summarize_attributions_mean_vs_absmean():
    model = _make_model(MLP)
    model.eval()
    X = _make_input()
    attr = explain_model(model, X, target=0)
    mean = summarize_attributions(attr, agg="mean")
    absmean = summarize_attributions(attr, agg="absmean")
    assert mean.shape == absmean.shape
    assert np.all(absmean >= np.abs(mean))  # absmean >= |mean|
{% endif %}
