{% if use_api %}
"""
schemas.py — Modelos Pydantic para la API REST de {{ project_name }}.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Cuerpo de la petición POST /predict."""

    features: dict[str, Any] = Field(
        ...,
        description="Diccionario feature_name → valor. Debe incluir todas las columnas de entrada.",
        json_schema_extra={"example": {"feat_0": 1.2, "feat_1": -0.5, "feat_2": 3.1}},
    )


{% if ml_type == "supervisado" or ml_type == "hibrido" %}
class PredictResponse(BaseModel):
    """Respuesta del endpoint POST /predict."""

    prediction: int | float = Field(..., description="Predicción del modelo")
{% if task_type == "clasificacion" %}
    probability: float | None = Field(
        None,
        description="Probabilidad de la clase positiva (si el modelo lo soporta)",
    )
    label: str | None = Field(
        None,
        description="Etiqueta original de la clase (si se usó LabelEncoder)",
    )
{% endif %}
    model_name: str = Field(..., description="Nombre del modelo usado")

{% elif ml_type == "no_supervisado" %}
class PredictResponse(BaseModel):
    """Respuesta del endpoint POST /predict."""

    cluster: int = Field(..., description="Cluster asignado al input")
    model_name: str = Field(..., description="Nombre del modelo de clustering usado")

{% elif ml_type == "redes_neuronales" %}
class PredictResponse(BaseModel):
    """Respuesta del endpoint POST /predict."""

    prediction: int | float = Field(..., description="Predicción del modelo")
{% if task_type == "clasificacion" %}
    probability: float | None = Field(
        None,
        description="Probabilidad de la clase predicha (softmax)",
    )
{% endif %}
    model_name: str = Field(..., description="Arquitectura de red usada")
{% endif %}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    project: str


class InfoResponse(BaseModel):
    project: str
    ml_type: str
    model_name: str
    feature_names: list[str]
{% if ml_type == "supervisado" or ml_type == "hibrido" %}
{% if task_type == "clasificacion" %}
    classes: list[str] | None = None
{% endif %}
{% endif %}
{% endif %}