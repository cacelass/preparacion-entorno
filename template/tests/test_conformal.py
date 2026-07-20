{% if use_conformal %}
"""
test_conformal.py — Tests para Conformal Prediction (split-conformal).
"""
import numpy as np
import pytest

{% if task_type == "clasificacion" %}
from {{project_slug}}.models.conformal import (
    ConformalClassifier,
    conformalize_classifier,
)

N_CLASSES = 4
RNG = np.random.default_rng(42)


def _make_calibration_data(n=400, n_classes=N_CLASSES):
    logits = RNG.normal(size=(n, n_classes))
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y_true = np.array([RNG.choice(n_classes, p=p) for p in proba])
    return proba, y_true


@pytest.mark.parametrize("method", ["lac", "aps"])
def test_predict_sets_shape(method):
    proba_cal, y_cal = _make_calibration_data()
    proba_eval, _ = _make_calibration_data(n=100)
    conformal = conformalize_classifier(proba_cal, y_cal, alpha=0.1, method=method)
    sets = conformal.predict_sets(proba_eval)
    assert sets.shape == proba_eval.shape
    assert sets.dtype == bool


@pytest.mark.parametrize("method", ["lac", "aps"])
def test_empirical_coverage_meets_nominal(method):
    """Con datos suficientes, la cobertura empírica debe rondar 1 - alpha (con margen)."""
    proba_cal, y_cal = _make_calibration_data(n=1000)
    proba_eval, y_eval = _make_calibration_data(n=1000)
    alpha = 0.1
    conformal = conformalize_classifier(proba_cal, y_cal, alpha=alpha, method=method)
    coverage = conformal.empirical_coverage(proba_eval, y_eval)
    # Margen amplio: split conformal garantiza cobertura marginal, no exacta.
    assert coverage >= (1 - alpha) - 0.07


def test_smaller_alpha_yields_larger_or_equal_sets():
    """Más cobertura exigida (alpha menor) -> sets iguales o más grandes en promedio."""
    proba_cal, y_cal = _make_calibration_data()
    proba_eval, _ = _make_calibration_data(n=100)
    strict = conformalize_classifier(proba_cal, y_cal, alpha=0.05, method="lac")
    loose = conformalize_classifier(proba_cal, y_cal, alpha=0.3, method="lac")
    assert strict.avg_set_size(proba_eval) >= loose.avg_set_size(proba_eval)


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        ConformalClassifier(method="not_a_method")


def test_predict_sets_before_fit_raises():
    conformal = ConformalClassifier()
    with pytest.raises(RuntimeError):
        conformal.predict_sets(np.ones((5, N_CLASSES)) / N_CLASSES)


def test_aps_set_always_includes_top_class():
    """APS debe incluir siempre la clase de mayor probabilidad en el set."""
    proba_cal, y_cal = _make_calibration_data()
    proba_eval, _ = _make_calibration_data(n=50)
    conformal = conformalize_classifier(proba_cal, y_cal, alpha=0.1, method="aps")
    sets = conformal.predict_sets(proba_eval)
    top_class = np.argmax(proba_eval, axis=1)
    assert sets[np.arange(len(proba_eval)), top_class].all()

{% else %}
from {{project_slug}}.models.conformal import ConformalRegressor, conformalize_regressor

RNG = np.random.default_rng(42)


def _make_calibration_data(n=500, noise_std=1.0):
    y_true = RNG.normal(size=n)
    y_pred = y_true + RNG.normal(scale=noise_std, size=n)
    return y_pred, y_true


def test_predict_interval_shape_and_ordering():
    pred_cal, y_cal = _make_calibration_data()
    pred_eval, _ = _make_calibration_data(n=50)
    conformal = conformalize_regressor(pred_cal, y_cal, alpha=0.1)
    lower, upper = conformal.predict_interval(pred_eval)
    assert lower.shape == pred_eval.shape == upper.shape
    assert np.all(upper > lower)


def test_empirical_coverage_meets_nominal():
    pred_cal, y_cal = _make_calibration_data(n=1000)
    pred_eval, y_eval = _make_calibration_data(n=1000)
    alpha = 0.1
    conformal = conformalize_regressor(pred_cal, y_cal, alpha=alpha)
    coverage = conformal.empirical_coverage(pred_eval, y_eval)
    assert coverage >= (1 - alpha) - 0.07


def test_smaller_alpha_yields_wider_or_equal_interval():
    pred_cal, y_cal = _make_calibration_data()
    strict = conformalize_regressor(pred_cal, y_cal, alpha=0.05)
    loose = conformalize_regressor(pred_cal, y_cal, alpha=0.3)
    assert strict.qhat_ >= loose.qhat_


def test_predict_interval_before_fit_raises():
    conformal = ConformalRegressor()
    with pytest.raises(RuntimeError):
        conformal.predict_interval(np.array([1.0, 2.0]))


def test_qhat_is_positive():
    pred_cal, y_cal = _make_calibration_data()
    conformal = conformalize_regressor(pred_cal, y_cal, alpha=0.1)
    assert conformal.qhat_ > 0
{% endif %}
{% endif %}
