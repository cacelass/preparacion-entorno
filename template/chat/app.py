"""
chat/app.py — Interfaz web de chat para {{ project_name }}.

Servidor FastAPI + WebSocket que expone un chat interactivo
para interactuar con los modelos entrenados del proyecto.

Se inicia automaticamente via:
    docker compose up -d
o directamente:
    python -m uvicorn chat.app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

{% if ml_type == 'redes_neuronales' %}
try:
    import torch
    from {{ project_slug }}.models.train_model import build_model
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
{% endif %}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR    = PROJECT_DIR / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
CHAT_DIR    = PROJECT_DIR / "chat"
STATIC_DIR    = CHAT_DIR / "static"

# ---------------------------------------------------------------------------
# Estado global del servicio
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "models":         {},
    "scaler":         None,
    "encoders":       {},
    "feature_names":  [],
    "target_encoder": None,
    "model_loaded":   False,
}

# Constantes del proyecto (fijadas en la generacion del template)
_PROJECT     = "{{ project_name }}"
_ML_TYPE     = "{{ ml_type }}"
_TASK_TYPE   = "{{ task_type if ml_type != 'no_supervisado' else 'clustering' }}"
_VERSION     = "{{ project_version }}"


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------
def load_models() -> None:
    """Carga modelos y artefactos de preprocesado desde models/."""

    # Feature names
    fn_path = ARTIFACTS_DIR / "feature_names.joblib"
    if fn_path.exists():
        _state["feature_names"] = joblib.load(fn_path)

    # Scaler
    sc_path = ARTIFACTS_DIR / "scaler.joblib"
    if sc_path.exists():
        _state["scaler"] = joblib.load(sc_path)

    # Encoders de features categoricas
    enc_path = ARTIFACTS_DIR / "encoders.joblib"
    if enc_path.exists():
        _state["encoders"] = joblib.load(enc_path)

    # Target encoder
    te_path = ARTIFACTS_DIR / "target_encoder.joblib"
    if te_path.exists():
        _state["target_encoder"] = joblib.load(te_path)

{% if ml_type == 'supervisado' or ml_type == 'hibrido' %}
    _skip = {"scaler", "encoders", "pca", "threshold", "feature_names",
             "target_encoder", "output_dim"}
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        if path.stem in _skip or path.stem.startswith("best_params_"):
            continue
        try:
            _state["models"][path.stem] = joblib.load(path)
        except Exception as exc:
            print(f"[chat/app] No se pudo cargar {path.name}: {exc}", file=sys.stderr)

{% elif ml_type == 'no_supervisado' %}
    _skip = {"scaler", "encoders", "pca", "feature_names", "KMeansPipeline"}
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        if path.stem in _skip or path.stem.startswith("best_params_"):
            continue
        try:
            _state["models"][path.stem] = joblib.load(path)
        except Exception as exc:
            print(f"[chat/app] No se pudo cargar {path.name}: {exc}", file=sys.stderr)

{% elif ml_type == 'redes_neuronales' %}
    if _TORCH_AVAILABLE:
        pt_path   = MODELS_DIR / "{{ nn_model }}_final.pt"
        best_path = MODELS_DIR / "{{ nn_model }}_best.pt"
        load_path = best_path if best_path.exists() else pt_path
        if load_path.exists():
            try:
                fn        = _state["feature_names"]
                input_dim = len(fn) if fn else 1
                out_path  = ARTIFACTS_DIR / "output_dim.joblib"
                if out_path.exists():
                    output_dim = int(joblib.load(out_path))
                else:
                    # Inferir output_dim desde el ultimo bias del state_dict
                    sd = torch.load(load_path, map_location="cpu", weights_only=True)
                    last_bias = [v for k, v in sd.items() if k.endswith(".bias")]
                    output_dim = last_bias[-1].shape[0] if last_bias else 2
                model = build_model(input_dim=input_dim, output_dim=output_dim)
                model.load_state_dict(
                    torch.load(load_path, map_location="cpu", weights_only=True)
                )
                model.eval()
                _state["models"]["{{ nn_model }}"] = model
            except Exception as exc:
                print(f"[chat/app] Error cargando modelo torch: {exc}", file=sys.stderr)
    else:
        print("[chat/app] torch no disponible — modelo PyTorch no cargado.", file=sys.stderr)
{% endif %}

    _state["model_loaded"] = bool(_state["models"])
    print(f"[chat/app] Modelos cargados: {list(_state['models'].keys())}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Prediccion generica
# ---------------------------------------------------------------------------
def _preprocess(features: dict[str, Any]) -> np.ndarray:
    """Convierte un dict de features a ndarray listo para predecir."""
    if _state["feature_names"]:
        missing = [f for f in _state["feature_names"] if f not in features]
        if missing:
            raise ValueError(f"Faltan features: {missing}")
        df = pd.DataFrame([{f: features[f] for f in _state["feature_names"]}])
    else:
        df = pd.DataFrame([features])

    for col, enc in _state["encoders"].items():
        if col == "__target__" or col not in df.columns:
            continue
        try:
            df[col] = enc.transform(df[col].astype(str))
        except ValueError as exc:
            raise ValueError(f"Valor desconocido en '{col}': {exc}") from exc

    if _state["scaler"] is not None:
        X = _state["scaler"].transform(df)
    else:
        try:
            X = df.values.astype(np.float64)  # solo funciona si todas las cols son numericas
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "No hay scaler cargado y el dataframe contiene columnas no numericas. "
                "Vuelve a entrenar el modelo con `train`."
            ) from exc
    return X.astype(np.float32)


def predict_one(features: dict[str, Any]) -> dict[str, Any]:
    """Genera una prediccion a partir de un dict de features."""
    if not _state["model_loaded"]:
        return {"error": "No hay modelos cargados. Entrena primero con `train`."}

    try:
        X = _preprocess(features)
    except ValueError as exc:
        return {"error": str(exc)}

{% if ml_type == 'supervisado' or ml_type == 'hibrido' %}
    model_name = list(_state["models"].keys())[0]
    model      = _state["models"][model_name]
    pred       = model.predict(X)[0]

{% if task_type == 'clasificacion' %}
    prob: float | None = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0].max())
    label: str | None = None
    if _state["target_encoder"] is not None:
        try:
            label = str(_state["target_encoder"].inverse_transform([int(pred)])[0])
        except Exception:
            label = str(pred)
    return {"prediction": int(pred), "probability": prob, "label": label, "model": model_name}
{% else %}
    return {"prediction": float(pred), "model": model_name}
{% endif %}

{% elif ml_type == 'no_supervisado' %}
    model_name = list(_state["models"].keys())[0]
    model      = _state["models"][model_name]
    # AgglomerativeClustering y DBSCAN no tienen .predict(); usamos fit_predict
    if hasattr(model, "predict"):
        cluster = int(model.predict(X)[0])
    elif hasattr(model, "fit_predict"):
        cluster = int(model.fit_predict(X)[0])
    else:
        return {"error": f"El modelo '{model_name}' no soporta prediccion en nuevos datos."}
    return {"cluster": cluster, "model": model_name}

{% elif ml_type == 'redes_neuronales' %}
    if not _TORCH_AVAILABLE:
        return {"error": "torch no disponible en este entorno."}

    model_name = "{{ nn_model }}"
    model      = _state["models"].get(model_name)
    if model is None:
        return {"error": "Modelo PyTorch no encontrado."}

    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(tensor)
{% if task_type == 'clasificacion' %}
        probs  = torch.softmax(logits, dim=-1)
        pred   = int(probs.argmax(dim=-1).item())
        prob   = float(probs.max().item())
    return {"prediction": pred, "probability": prob, "model": model_name}
{% else %}
        pred = float(logits.squeeze().item())
    return {"prediction": pred, "model": model_name}
{% endif %}
{% endif %}


# ---------------------------------------------------------------------------
# Mensajes del bot
# ---------------------------------------------------------------------------
def _welcome_message() -> str:
    status_line = (
        f" ✔ **{len(_state['models'])} modelo(s) listo(s):** "
        f"`{'`, `'.join(_state['models'].keys())}`"
        if _state["model_loaded"]
        else "   Sin modelos entrenados — escribe `train` para entrenar."
    )
    return (
        f"#    Bienvenido a **{{ project_name }}**\n\n"
        f"| Campo | Valor |\n"
        f"|---|---|\n"
        f"| **Tipo ML** | `{{ ml_type }}` |\n"
        f"| **Tarea** | `{_TASK_TYPE}` |\n"
        f"| **Versión** | `{{ project_version }}` |\n\n"
        f"{status_line}\n\n"
        f"---\n\n"
        f"**Comandos disponibles:**\n"
        f"- `status` — estado del sistema\n"
        f"- `info` — detalles del modelo y features\n"
        f"- `predict` — hacer una prediccion paso a paso\n"
        f"- `train` — lanzar el entrenamiento\n"
        f"- `reload` — recargar modelos del disco\n"
        f"- `help` — mostrar este mensaje"
    )


def _status_message() -> str:
    feat_count = len(_state["feature_names"])
    if _state["model_loaded"]:
        models_list = "\n".join(f"  - `{m}`" for m in _state["models"].keys())
        return (
            f"**Estado del sistema**  ✔\n\n"
            f"**Proyecto:** {{ project_name }}\n"
            f"**ML Type:** `{{ ml_type }}`\n"
            f"**Tarea:** `{_TASK_TYPE}`\n"
            f"**Features detectadas:** {feat_count}\n\n"
            f"**Modelos disponibles:**\n{models_list}"
        )
    return (
        f"**Estado del sistema**   \n\n"
        f"No hay modelos entrenados todavia.\n"
        f"Escribe `train` para lanzar el entrenamiento."
    )


def _info_message() -> str:
    if not _state["model_loaded"]:
        return "   No hay modelos cargados. Entrena primero con `train`."
    features = _state["feature_names"]
    feat_str = ", ".join(f"`{f}`" for f in features[:8])
    if len(features) > 8:
        feat_str += f" ... *y {len(features) - 8} mas*"
    classes_info = ""
    if _state["target_encoder"] is not None:
        classes = list(_state["target_encoder"].classes_)
        classes_info = f"\n**Clases:** {', '.join(str(c) for c in classes)}"
    return (
        f"**Informacion del modelo**\n\n"
        f"**Modelos:** {', '.join(_state['models'].keys())}\n"
        f"**Features ({len(features)}):** {feat_str}{classes_info}"
    )


def _start_prediction(session: dict) -> str:
    if not _state["model_loaded"]:
        return "   No hay modelos cargados. Entrena primero con `train`."
    features = _state["feature_names"]
    if not features:
        return "   No se detectaron nombres de features. Vuelve a entrenar el modelo."
    # Snapshot de features en la sesion — inmune a reload() durante la prediccion
    session.update({"state": "collecting", "features": {}, "idx": 0,
                    "snapshot_features": list(features)})
    return (
        f"   **Modo prediccion** — introduce el valor de cada feature.\n\n"
        f"**{len(features)} features** en total. Escribe `cancelar` para salir.\n\n"
        f"---\n\n"
        f"**[1/{len(features)}]** `{features[0]}`"
    )


def _handle_feature(msg: str, session: dict) -> str:
    if msg.lower() in ("cancel", "cancelar", "salir", "exit"):
        session["state"] = "idle"
        return " ✕ Prediccion cancelada."

    # Usar snapshot de la sesion (inmune a reload durante la prediccion)
    features = session.get("snapshot_features") or _state["feature_names"]
    idx      = session["idx"]
    if idx >= len(features):          # proteccion ante desfase por reload
        session["state"] = "idle"
        return "   Sesion de prediccion desfasada. Escribe `predict` para empezar de nuevo."
    name     = features[idx]

    try:
        value = float(msg.replace(",", "."))
    except ValueError:
        value = msg  # Categorica — mantener como string

    session["features"][name] = value
    session["idx"] += 1

    if session["idx"] >= len(features):
        session["state"] = "idle"
        result = predict_one(session["features"])

        if "error" in result:
            return f" ✕ Error: {result['error']}"

{% if ml_type == 'no_supervisado' %}
        cluster    = result.get("cluster", "?")
        model_name = result.get("model", "?")
        return (
            f" ✔ **Resultado**\n\n"
            f"**Cluster asignado:** `{cluster}`\n"
            f"**Modelo:** `{model_name}`\n\n"
            f"¿Otra prediccion? Escribe `predict`."
        )
{% else %}
        pred       = result.get("prediction", "?")
        prob       = result.get("probability")
        label      = result.get("label")
        model_name = result.get("model", "?")
        prob_str   = f"\n**Confianza:** `{prob:.1%}`" if prob is not None else ""
        label_str  = f"\n**Clase:** `{label}`" if label else ""
        return (
            f" ✔ **Resultado de la prediccion**\n\n"
            f"**Prediccion:** `{pred}`{label_str}{prob_str}\n"
            f"**Modelo:** `{model_name}`\n\n"
            f"¿Otra prediccion? Escribe `predict`."
        )
{% endif %}

    next_name = features[session["idx"]]  # features ya es el snapshot
    return (
        f"✓ `{name}` = `{value}`\n\n"
        f"**[{session['idx'] + 1}/{len(features)}]** `{next_name}`"
    )


# ---------------------------------------------------------------------------
# Maquina de estados del chat
# ---------------------------------------------------------------------------
async def process_message(msg: str, session: dict) -> str:
    """Enruta cada mensaje al handler correcto segun el estado de la sesion."""
    low = msg.lower().strip()

    if session["state"] == "collecting":
        return _handle_feature(msg, session)

    if low in ("help", "ayuda", "?", ""):
        return _welcome_message()
    if low in ("status", "estado"):
        return _status_message()
    if low in ("info"):
        return _info_message()
    if low in ("reload", "recargar"):
        _state.update({
            "models": {}, "model_loaded": False,
            "scaler": None, "encoders": {},
            "feature_names": [], "target_encoder": None,
        })
        load_models()
        if _state["model_loaded"]:
            return f" ✔ Modelos recargados: **{', '.join(_state['models'].keys())}**"
        return "   No se encontraron modelos en `models/`."
    if low in ("train", "entrenar", "training"):
        # Buscar dataset en ubicaciones posibles
        candidates = [
            PROJECT_DIR / "dataset.csv",
            PROJECT_DIR / "data" / "raw",
        ]
        dataset_found = candidates[0].exists()
        if not dataset_found:
            raw_dir = candidates[1]
            if raw_dir.is_dir():
                dataset_found = any(raw_dir.glob("*.csv"))
        if not dataset_found:
            return (
                "   No se encontro `dataset.csv`.\n\n"
                "Coloca el dataset en la raiz del proyecto o en `data/raw/` y escribe `train` de nuevo."
            )
        try:
            proc = subprocess.Popen(
                [sys.executable, "main.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,   # evita deadlock por pipe lleno
                stderr=subprocess.DEVNULL,
                cwd=str(PROJECT_DIR),
                text=True,
            )
            # Enviar "0" al prompt interactivo y desacoplar (background)
            proc.stdin.write("0\n")
            proc.stdin.flush()
            proc.stdin.close()
            # No esperamos (proc.wait()) — corre en background
            return (
                "  **Entrenamiento iniciado** en segundo plano.\n\n"
                "El proceso puede tardar varios minutos dependiendo del dataset.\n"
                "Cuando termine, escribe `reload` para cargar los modelos."
            )
        except Exception as exc:
            return f" ✕ Error al iniciar entrenamiento: {exc}"
    if low in ("predict", "predecir", "prediccion"):
        return _start_prediction(session)
    if low in ("cancel", "cancelar"):
        return "ℹ No hay ninguna operacion activa."

    return (
        f"❓ No reconozco `{msg}`.\n\n"
        f"Escribe `help` para ver los comandos disponibles."
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="{{ project_name }} — Chat",
    description="Interfaz web de chat para {{ project_name }}. Generado por dskit.",
    version="{{ project_version }}",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Cargar modelos al arrancar (errores no fatales: app arranca igual)
try:
    load_models()
except Exception as _startup_exc:
    print(f"[chat/app] Aviso al cargar modelos al inicio: {_startup_exc}", file=sys.stderr)


@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.get("/api/status")
async def api_status():
    return {
        "project":      _PROJECT,
        "ml_type":      _ML_TYPE,
        "task_type":    _TASK_TYPE,
        "version":      _VERSION,
        "model_loaded": _state["model_loaded"],
        "models":       list(_state["models"].keys()),
        "feature_count": len(_state["feature_names"]),
        "features":     _state["feature_names"],
    }


@app.get("/api/reload")
async def api_reload():
    _state.update({
        "models": {}, "model_loaded": False,
        "scaler": None, "encoders": {},
        "feature_names": [], "target_encoder": None,
    })
    load_models()
    return {
        "model_loaded": _state["model_loaded"],
        "models": list(_state["models"].keys()),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "bot", "text": _welcome_message()})

    session: dict[str, Any] = {"state": "idle", "features": {}, "idx": 0}

    try:
        while True:
            try:
                data = await ws.receive_json()
            except WebSocketDisconnect:
                raise   # propagar para que el outer except lo capture
            except Exception:
                # Mensaje malformado (no JSON) — ignorar y seguir
                try:
                    await ws.send_json({"type": "bot", "text": " ✕ Mensaje no válido. Usa texto plano."})
                except Exception:
                    pass
                continue
            msg   = data.get("text", "").strip()
            reply = await process_message(msg, session)
            await ws.send_json({"type": "bot", "text": reply})
    except WebSocketDisconnect:
        pass