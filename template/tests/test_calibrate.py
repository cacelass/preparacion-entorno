{% if ml_type == "redes_neuronales" and use_calibration %}
"""
test_calibrate.py — Tests para calibración de confianza.
"""
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from {{ project_slug }}.models.calibrate import TemperatureScaler, calibrate_model


def test_temperature_scaler_init():
    scaler = TemperatureScaler(init_temp=1.5)
    assert scaler.get_temperature() == 1.5


def test_temperature_scaler_forward():
    scaler = TemperatureScaler(init_temp=2.0)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    out = scaler(logits)
    expected = logits / 2.0
    assert torch.allclose(out, expected, atol=1e-6)


def test_temperature_scaler_forward_clamps_negative():
    """Temperatura negativa se clamp a 1e-6."""
    scaler = TemperatureScaler(init_temp=-1.0)
    logits = torch.tensor([[1.0, 2.0]])
    out = scaler(logits)
    assert torch.isfinite(out).all()


def test_temperature_scaler_fit_reduces_nll():
    """T optimizado debe reducir (o al menos no empeorar) NLL."""
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 3)
    X = torch.randn(50, 4)
    y = torch.randint(0, 3, (50,))
    loader = torch.utils.data.DataLoader(
        list(zip(X, y)), batch_size=16, shuffle=True
    )
    scaler = TemperatureScaler(init_temp=1.0)
    nll_before = torch.nn.functional.cross_entropy(
        scaler(model(X)), y
    ).item()
    scaler.fit(model, loader, device="cpu", max_iter=30)
    nll_after = torch.nn.functional.cross_entropy(
        scaler(model(X)), y
    ).item()
    assert nll_after <= nll_before + 0.01  # tolerancia por stochastic


def test_calibrate_model_returns_scaler():
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 2)
    X = torch.randn(30, 4)
    y = torch.randint(0, 2, (30,))
    loader = torch.utils.data.DataLoader(
        list(zip(X, y)), batch_size=10, shuffle=True
    )
    scaler = calibrate_model(model, loader, device="cpu", max_iter=10)
    assert isinstance(scaler, TemperatureScaler)
    assert scaler.get_temperature() > 0
{% endif %}
