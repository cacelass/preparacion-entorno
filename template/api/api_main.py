{% if use_api %}
"""
api/main.py — API REST de {{ project_name }}.

Arranca con:
    make serve
o directamente:
    uv run uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health    → estado del servicio
    GET  /info      → metadata del modelo y features
    POST /predict   → predicción sobre nuevos datos
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger

from {{ project_slug }}.utils.paths import MODELS_DIR, ARTIFACTS_DIR
from api.schemas import HealthResponse, InfoResponse, PredictRequest, PredictResponse

{% if ml_type == "redes_neuronales" %}
import torch
from {{ project_slug }}.models.train_model import build_model
{% endif %}

# ---------------------------------------------------------------------------
# Estado global del servicio
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "models":        {},
    "scaler":        None,
    "encoders":      {},
    "feature_names": [],
    "target_encoder": None,
    "model_loaded":  False,
}

_PROJECT   = "{{ project_name }}"
_ML_TYPE   = "{{ ml_type }}"
{% if ml_type == "redes_neuronales" %}
_MODEL_NAME = "{{ nn_model }}"
{% else %}
_MODEL_NAME = "{{ model_type }}"
{% endif %}


# ---------------------------------------------------------------------------
# Carga de artefactos al arrancar
# ---------------------------------------------------------------------------
def _load_artifacts() -> None:
    """Carga modelos y artefactos de preprocesado desde models/artifacts/."""

    # Feature names
    fn_path = ARTIFACTS_DIR / "feature_names.joblib"
    if fn_path.exists():
        _state["feature_names"] = joblib.load(fn_path)
        logger.info(f"feature_names cargado: {len(_state['feature_names'])} features")
    else:
        logger.warning("feature_names.joblib no encontrado — usa nombres genéricos")

    # Scaler
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    if scaler_path.exists():
        _state["scaler"] = joblib.load(scaler_path)
        logger.info("scaler.joblib cargado")

    # Encoders de features categoricas
    enc_path = ARTIFACTS_DIR / "encoders.joblib"
    if enc_path.exists():
        _state["encoders"] = joblib.load(enc_path)
        logger.info(f"encoders.joblib cargado: {list(_state['encoders'].keys())}")

    # Target encoder (si existe)
    te_path = ARTIFACTS_DIR / "target_encoder.joblib"
    if te_path.exists():
        _state["target_encoder"] = joblib.load(te_path)
        logger.info("target_encoder.joblib cargado")

{% if ml_type == "supervisado" or ml_type == "hibrido" %}
    # Modelos joblib
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        if path.stem.startswith("scaler") or path.stem.startswith("encoders"):
            continue
        try:
            _state["models"][path.stem] = joblib.load(path)
            logger.info(f"Modelo cargado: {path.stem}")
        except Exception as exc:
            logger.warning(f"No se pudo cargar {path.name}: {exc}")

{% elif ml_type == "no_supervisado" %}
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        if path.stem in ("scaler", "encoders", "pca"):
            continue
        try:
            _state["models"][path.stem] = joblib.load(path)
            logger.info(f"Modelo clustering cargado: {path.stem}")
        except Exception as exc:
            logger.warning(f"No se pudo cargar {path.name}: {exc}")

{% elif ml_type == "redes_neuronales" %}
    pt_path = MODELS_DIR / f"{{ nn_model }}_final.pt"
    best_path = MODELS_DIR / f"{{ nn_model }}_best.pt"
    load_path = best_path if best_path.exists() else pt_path
    if load_path.exists():
        input_dim  = len(_state["feature_names"]) if _state["feature_names"] else 1
        output_dim_path = ARTIFACTS_DIR / "output_dim.joblib"
        output_dim = int(joblib.load(output_dim_path)) if output_dim_path.exists() else 2
        model = build_model(input_dim=input_dim, output_dim=output_dim)
        model.load_state_dict(
            torch.load(load_path, map_location="cpu", weights_only=True)
        )
        model.eval()
        _state["models"]["{{ nn_model }}"] = model
        logger.info(f"Red neuronal cargada desde {load_path.name}")
    else:
        logger.warning(f"No se encontró modelo .pt en {MODELS_DIR}")
{% endif %}

    _state["model_loaded"] = bool(_state["models"])
    logger.info(f"Modelos disponibles: {list(_state['models'].keys())}")


# ---------------------------------------------------------------------------
# Preprocesado del input
# ---------------------------------------------------------------------------
def _preprocess_input(features: dict[str, Any]) -> np.ndarray:
    """Convierte el dict de features a un array escalado listo para predecir."""
    if _state["feature_names"]:
        missing = [f for f in _state["feature_names"] if f not in features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Faltan features: {missing}",
            )
        df = pd.DataFrame([{f: features[f] for f in _state["feature_names"]}])
    else:
        df = pd.DataFrame([features])

    # Encoders de features categoricas
    for col, enc in _state["encoders"].items():
        if col == "__target__":
            continue
        if col in df.columns:
            try:
                df[col] = enc.transform(df[col].astype(str))
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"Valor desconocido en feature '{col}': {df[col].iloc[0]}",
                )

    # Scaler
    if _state["scaler"] is not None:
        X = _state["scaler"].transform(df)
    else:
        X = df.values

    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Ciclo de vida de la app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Arrancando API de {{ project_name }}...")
    _load_artifacts()
    if not _state["model_loaded"]:
        logger.warning(
            "No se encontró ningún modelo. "
            "Ejecuta 'make train' antes de lanzar la API."
        )
    yield
    logger.info("Apagando API.")


app = FastAPI(
    title="{{ project_name }} API",
    description="API REST generada por dskit para el modelo de ML de {{ project_name }}.",
    version="{{ project_version }}",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Servicio"])
def health() -> HealthResponse:
    """Comprueba que la API está activa y el modelo cargado."""
    return HealthResponse(
        status="ok",
        model_loaded=_state["model_loaded"],
        project=_PROJECT,
    )


@app.get("/info", response_model=InfoResponse, tags=["Servicio"])
def info() -> InfoResponse:
    """Devuelve metadata del proyecto y del modelo."""
    return InfoResponse(
        project=_PROJECT,
        ml_type=_ML_TYPE,
        model_name=_MODEL_NAME,
        feature_names=_state["feature_names"],
{% if (ml_type == "supervisado" or ml_type == "hibrido") and task_type == "clasificacion" %}
        classes=(
            list(_state["target_encoder"].classes_)
            if _state["target_encoder"] is not None
            else None
        ),
{% endif %}
    )


@app.post("/predict", response_model=PredictResponse, tags=["Predicción"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Realiza una predicción sobre un nuevo ejemplo.

    Envía un JSON con el campo `features`: un diccionario con los nombres
    y valores de cada feature del modelo.

    Ejemplo:
    ```json
    {
      "features": {"feat_0": 1.2, "feat_1": -0.5, "feat_2": 3.1}
    }
    ```
    """
    if not _state["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta 'make train' primero.",
        )

    X = _preprocess_input(request.features)

{% if ml_type == "supervisado" or ml_type == "hibrido" %}
    # Usamos el primer modelo disponible (o el único si model_type != "todos")
    model_name = list(_state["models"].keys())[0]
    model      = _state["models"][model_name]
    pred       = int(model.predict(X)[0])

{% if task_type == "clasificacion" %}
    prob: float | None = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        prob  = float(proba.max())

    # Decodificar etiqueta original si existe target_encoder
    label: str | None = None
    if _state["target_encoder"] is not None:
        try:
            label = str(_state["target_encoder"].inverse_transform([pred])[0])
        except Exception:
            label = str(pred)

    return PredictResponse(
        prediction=pred,
        probability=prob,
        label=label,
        model_name=model_name,
    )
{% else %}
    return PredictResponse(
        prediction=float(model.predict(X)[0]),
        model_name=model_name,
    )
{% endif %}

{% elif ml_type == "no_supervisado" %}
    model_name = list(_state["models"].keys())[0]
    model      = _state["models"][model_name]
    cluster    = int(model.predict(X)[0])
    return PredictResponse(cluster=cluster, model_name=model_name)

{% elif ml_type == "redes_neuronales" %}
    model_name = "{{ nn_model }}"
    model      = _state["models"].get(model_name)
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(tensor)
{% if task_type == "clasificacion" %}
        probs  = torch.softmax(logits, dim=-1)
        pred   = int(probs.argmax(dim=-1).item())
        prob   = float(probs.max().item())
    return PredictResponse(prediction=pred, probability=prob, model_name=model_name)
{% else %}
        pred = float(logits.squeeze().item())
    return PredictResponse(prediction=pred, model_name=model_name)
{% endif %}
{% endif %}
{% endif %}
