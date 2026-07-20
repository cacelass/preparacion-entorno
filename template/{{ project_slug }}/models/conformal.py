{% if use_conformal %}
"""
models/conformal.py — Conformal Prediction (split-conformal, distribution-free).

{% if task_type == "clasificacion" %}
Produce SETS de predicción (no una única clase) con garantía de cobertura
marginal finita: P(y_true en el set) >= 1 - alpha, sin asumir nada sobre el
modelo ni la distribución de los datos (solo exchangeability).

Métodos de score de no-conformidad:
  lac → 1 - proba de la clase verdadera (Least Ambiguous set-valued Classifier).
        Sets más pequeños en promedio, pero menos adaptativos por instancia.
  aps → suma acumulada de probas ordenadas hasta incluir la clase verdadera
        (Adaptive Prediction Sets, Romano et al. 2020). Sets que se adaptan
        mejor a la incertidumbre de cada muestra.

Referencia: Vovk, Gammerman & Shafer (2005) — Algorithmic Learning in a
Random World. Romano, Sesia & Candès (2020) — Classification with Valid and
Adaptive Coverage.

Uso:
    from {{ project_slug }}.models.conformal import conformalize_classifier

    # X_cal/y_cal: conjunto de CALIBRACIÓN, separado de train y del eval final
    # (p.ej. divide X_test 50/50 en X_cal/X_eval antes de llamar a esto).
    proba_cal = model.predict_proba(X_cal)
    conformal = conformalize_classifier(proba_cal, y_cal, alpha=0.1, method="lac")

    proba_eval = model.predict_proba(X_eval)
    pred_sets  = conformal.predict_sets(proba_eval)         # (n, n_clases) bool
    coverage   = conformal.empirical_coverage(proba_eval, y_eval)
    avg_size   = conformal.avg_set_size(proba_eval)
{% else %}
Produce INTERVALOS de predicción [lower, upper] con garantía de cobertura
marginal finita: P(y_true en [lower, upper]) >= 1 - alpha, sin asumir nada
sobre el modelo ni la distribución de los residuos (solo exchangeability).

Score de no-conformidad: residuo absoluto |y_true - y_pred| sobre un
conjunto de calibración separado del de entrenamiento y del eval final.

Referencia: Vovk, Gammerman & Shafer (2005) — Algorithmic Learning in a
Random World. Lei et al. (2018) — Distribution-Free Predictive Inference
for Regression.

Uso:
    from {{ project_slug }}.models.conformal import conformalize_regressor

    # X_cal/y_cal: conjunto de CALIBRACIÓN, separado de train y del eval final
    # (p.ej. divide X_test 50/50 en X_cal/X_eval antes de llamar a esto).
    pred_cal  = model.predict(X_cal)
    conformal = conformalize_regressor(pred_cal, y_cal, alpha=0.1)

    pred_eval        = model.predict(X_eval)
    lower, upper     = conformal.predict_interval(pred_eval)
    coverage         = conformal.empirical_coverage(pred_eval, y_eval)
{% endif %}
"""
from __future__ import annotations

import numpy as np


def _finite_sample_qhat(scores: np.ndarray, alpha: float) -> float:
    """
    Cuantil empírico de los scores de calibración, con la corrección de
    tamaño finito de split conformal: q_level = ceil((n+1)(1-alpha)) / n.

    Si q_level >= 1 (alpha muy pequeño para n muestras), se satura a 1.0
    → qhat es el score máximo observado (cobertura garantizada al 100%,
    pero el resultado será muy conservador; conviene más datos de calibración).
    """
    n = len(scores)
    if n == 0:
        raise ValueError("El conjunto de calibración no puede estar vacío.")
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


{% if task_type == "clasificacion" %}
class ConformalClassifier:
    """
    Conformal Prediction para clasificación — genera sets de predicción con
    cobertura marginal garantizada (1 - alpha).

    Parameters
    ----------
    alpha  : nivel de significancia. Cobertura objetivo = 1 - alpha (default 0.1 → 90%).
    method : "lac" (Least Ambiguous set-valued Classifier) | "aps" (Adaptive Prediction Sets)
    """

    def __init__(self, alpha: float = 0.1, method: str = "lac"):
        if method not in ("lac", "aps"):
            raise ValueError(f"method debe ser 'lac' o 'aps', recibido: {method!r}")
        self.alpha = alpha
        self.method = method
        self.qhat_: float | None = None

    def fit(self, y_proba_cal: np.ndarray, y_true_cal: np.ndarray) -> "ConformalClassifier":
        """
        Calibra el umbral qhat_ sobre un conjunto de calibración (NO usado en
        entrenamiento ni en la evaluación final).

        Parameters
        ----------
        y_proba_cal : array (n_cal, n_clases) — probabilidades del modelo ya entrenado
        y_true_cal  : array (n_cal,) — etiquetas verdaderas (enteros 0..n_clases-1)
        """
        y_proba_cal = np.asarray(y_proba_cal)
        y_true_cal = np.asarray(y_true_cal)
        n = len(y_true_cal)

        if self.method == "lac":
            scores = 1.0 - y_proba_cal[np.arange(n), y_true_cal]
        else:  # aps
            scores = self._aps_scores(y_proba_cal, y_true_cal)

        self.qhat_ = _finite_sample_qhat(scores, self.alpha)
        return self

    def predict_sets(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Devuelve una máscara booleana (n, n_clases): True si la clase está
        incluida en el set de predicción de esa muestra.
        """
        if self.qhat_ is None:
            raise RuntimeError("Llama a fit() antes de predict_sets().")
        y_proba = np.asarray(y_proba)

        if self.method == "lac":
            return y_proba >= (1.0 - self.qhat_)

        # aps: incluir clases en orden descendente de proba hasta que la
        # suma acumulada supere qhat_ (mismo criterio que en calibración).
        order = np.argsort(-y_proba, axis=1)
        sorted_proba = np.take_along_axis(y_proba, order, axis=1)
        cumsum = np.cumsum(sorted_proba, axis=1)
        sorted_mask = cumsum <= self.qhat_
        # Incluir siempre al menos la clase de mayor probabilidad, y la
        # primera clase que hace que cumsum supere qhat_ (criterio estándar APS).
        sorted_mask[:, 0] = True
        first_exceed = np.argmax(cumsum > self.qhat_, axis=1)
        rows = np.arange(len(y_proba))
        sorted_mask[rows, first_exceed] = True

        mask = np.zeros_like(sorted_mask)
        np.put_along_axis(mask, order, sorted_mask, axis=1)
        return mask

    def empirical_coverage(self, y_proba: np.ndarray, y_true: np.ndarray) -> float:
        """Fracción de muestras cuya etiqueta verdadera cae dentro del set predicho."""
        y_true = np.asarray(y_true)
        sets = self.predict_sets(y_proba)
        return float(sets[np.arange(len(y_true)), y_true].mean())

    def avg_set_size(self, y_proba: np.ndarray) -> float:
        """Tamaño medio de los sets de predicción — mide qué tan informativos son."""
        return float(self.predict_sets(y_proba).sum(axis=1).mean())

    @staticmethod
    def _aps_scores(y_proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Score APS: suma acumulada de probas (orden descendente) hasta la clase verdadera."""
        n = len(y_true)
        order = np.argsort(-y_proba, axis=1)
        sorted_proba = np.take_along_axis(y_proba, order, axis=1)
        cumsum = np.cumsum(sorted_proba, axis=1)
        true_rank = np.where(order == y_true[:, None])[1]
        return cumsum[np.arange(n), true_rank]


def conformalize_classifier(
    y_proba_cal: np.ndarray,
    y_true_cal: np.ndarray,
    alpha: float = 0.1,
    method: str = "lac",
) -> ConformalClassifier:
    """Crea y calibra un ConformalClassifier sobre (y_proba_cal, y_true_cal)."""
    return ConformalClassifier(alpha=alpha, method=method).fit(y_proba_cal, y_true_cal)

{% else %}
class ConformalRegressor:
    """
    Conformal Prediction para regresión — genera intervalos de predicción con
    cobertura marginal garantizada (1 - alpha).

    Parameters
    ----------
    alpha : nivel de significancia. Cobertura objetivo = 1 - alpha (default 0.1 → 90%).
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.qhat_: float | None = None

    def fit(self, y_pred_cal: np.ndarray, y_true_cal: np.ndarray) -> "ConformalRegressor":
        """
        Calibra el margen qhat_ sobre un conjunto de calibración (NO usado en
        entrenamiento ni en la evaluación final).

        Parameters
        ----------
        y_pred_cal : array (n_cal,) — predicciones del modelo ya entrenado
        y_true_cal : array (n_cal,) — valores verdaderos
        """
        y_pred_cal = np.asarray(y_pred_cal)
        y_true_cal = np.asarray(y_true_cal)
        scores = np.abs(y_true_cal - y_pred_cal)
        self.qhat_ = _finite_sample_qhat(scores, self.alpha)
        return self

    def predict_interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (lower, upper) = y_pred -/+ qhat_."""
        if self.qhat_ is None:
            raise RuntimeError("Llama a fit() antes de predict_interval().")
        y_pred = np.asarray(y_pred)
        return y_pred - self.qhat_, y_pred + self.qhat_

    def empirical_coverage(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Fracción de muestras cuyo valor verdadero cae dentro del intervalo predicho."""
        y_true = np.asarray(y_true)
        lower, upper = self.predict_interval(y_pred)
        return float(((y_true >= lower) & (y_true <= upper)).mean())


def conformalize_regressor(
    y_pred_cal: np.ndarray,
    y_true_cal: np.ndarray,
    alpha: float = 0.1,
) -> ConformalRegressor:
    """Crea y calibra un ConformalRegressor sobre (y_pred_cal, y_true_cal)."""
    return ConformalRegressor(alpha=alpha).fit(y_pred_cal, y_true_cal)

{% endif %}
{% endif %}
