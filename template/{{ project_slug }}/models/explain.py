{% if ml_type == "redes_neuronales" %}
"""
models/explain.py — Explicabilidad de redes neuronales con Captum.

Proporciona wrappers sobre Captum para atribuciones de features
compatibles con todas las arquitecturas del template (MLP, CNN1D,
LSTM, GRU, Transformer, ResNet).

Uso:
    from {{ project_slug }}.models.explain import explain_model

    attributions = explain_model(model, X_tensor, target=0)
    plot_attributions(attributions, feature_names, "LSTM")
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def explain_model(
    model: nn.Module,
    X: torch.Tensor,
    target: int | None = None,
    method: str = "ig",
    n_steps: int = 50,
) -> np.ndarray:
    """
    Calcula atribuciones (importancia de features) para una entrada.

    Parameters
    ----------
    model  : modelo PyTorch en eval mode
    X      : tensor de entrada (batch, input_dim) o (batch, seq_len, input_dim)
    target : clase objetivo (None = clase predicha)
    method : método de atribución ('ig' | 'dl' | 'gs')
    n_steps: pasos de integración para IG

    Returns
    -------
    np.ndarray : atribuciones con misma shape que X
    """
    import captum.attr as attr

    model.eval()
    if target is None:
        with torch.no_grad():
            target = model(X[:1]).argmax(dim=-1).item()

    # Detectar arquitectura para elegir explainer
    has_lstm = hasattr(model, "lstm")
    has_gru = hasattr(model, "gru")
    has_embedding = hasattr(model, "embedding")

    # LayerIntegratedGradients para modelos recurrentes (atribuye a input)
    if has_lstm:
        explainer = attr.LayerIntegratedGradients(model, model.lstm)
    elif has_gru:
        explainer = attr.LayerIntegratedGradients(model, model.gru)
    elif has_embedding and hasattr(model, "pos_enc"):
        # Transformer: atribuir sobre el embedding lineal
        explainer = attr.LayerIntegratedGradients(model, model.embedding)
    else:
        # MLP, CNN1D, ResNet: IntegratedGradients directo
        explainer = attr.IntegratedGradients(model)

    attributions, _ = explainer.attribute(
        X, target=target, n_steps=n_steps, return_convergence_delta=True,
    )
    return attributions.detach().cpu().numpy()


def summarize_attributions(
    attributions: np.ndarray,
    agg: str = "mean",
) -> np.ndarray:
    """
    Agrega atribuciones a una sola importancia por feature.

    Para LSTM/Transformer con shape (batch, seq_len, input_dim):
      - 'mean': media sobre pasos temporales y batch
      - 'absmean': media del valor absoluto sobre batch
    Para MLP/ResNet con shape (batch, input_dim):
      - 'mean': media sobre batch
      - 'absmean': media del valor absoluto

    Parameters
    ----------
    attributions : array de atribuciones del explainer
    agg          : 'mean' | 'absmean'

    Returns
    -------
    np.ndarray : (input_dim,) importancia por feature
    """
    if attributions.ndim == 3:
        if agg == "absmean":
            return np.abs(attributions).mean(axis=(0, 1))
        return attributions.mean(axis=(0, 1))
    if agg == "absmean":
        return np.abs(attributions).mean(axis=0)
    return attributions.mean(axis=0)
{% endif %}
