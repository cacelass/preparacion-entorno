{% if ml_type == "redes_neuronales" %}
"""
models/calibrate.py — Calibración de confianza para redes neuronales.

Temperature Scaling: post-hoc calibration con un solo parámetro T.
Escala los logits antes del softmax: softmax(logits / T).

Uso:
    from {{ project_slug }}.models.calibrate import TemperatureScaler, calibrate_model

    scaler = TemperatureScaler()
    scaler.fit(model, val_loader, device)
    logits_calib = scaler(logits)
    probs_calib = torch.softmax(logits_calib, dim=-1)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """
    Temperature Scaling — calibra softmax con un solo escalar T > 0.

    Parameters
    ----------
    init_temp : float
        Temperatura inicial (default 1.0 = sin calibración).
    """
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-6)

    def fit(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_iter: int = 100,
        lr: float = 1e-2,
    ) -> float:
        """
        Ajusta T minimizando NLL sobre el conjunto de validación.

        Parameters
        ----------
        model      : modelo ya entrenado (en eval mode)
        val_loader : DataLoader con (X, y) de validación
        device     : dispositivo donde está el modelo
        max_iter   : épocas de optimización (converge rápido, <50)
        lr         : learning rate para optimizar T

        Returns
        -------
        float : NLL final después de calibración
        """
        model.eval()
        self.to(device)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        logits_list, labels_list = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device)
                logits_list.append(model(X_b).detach().cpu())
                labels_list.append(y_b.cpu())

        logits_all = torch.cat(logits_list)
        labels_all = torch.cat(labels_list)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self(logits_all), labels_all)
            loss.backward()
            return loss

        optimizer.step(closure)
        final_nll = F.cross_entropy(self(logits_all), labels_all).item()
        return final_nll

    def get_temperature(self) -> float:
        return self.temperature.item()


def calibrate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    init_temp: float = 1.0,
    max_iter: int = 100,
) -> TemperatureScaler:
    """
    Crea, entrena y devuelve un TemperatureScaler.

    Parameters
    ----------
    model      : modelo entrenado (eval mode)
    val_loader : datos de validación
    device     : torch.device
    init_temp  : temperatura inicial
    max_iter   : iteraciones máximas de optimización

    Returns
    -------
    TemperatureScaler calibrado
    """
    scaler = TemperatureScaler(init_temp=init_temp)
    scaler.fit(model, val_loader, device, max_iter=max_iter)
    return scaler
{% endif %}
