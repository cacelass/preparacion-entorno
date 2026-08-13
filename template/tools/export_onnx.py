"""
export_onnx.py — exporta los modelos entrenados a ONNX para la demo web.

Replica EXACTAMENTE el pipeline de inferencia de process_input() + predict():
  features crudas → (encode categóricas en el navegador con meta.json)
  → scaler → PCA (si existe) → modelo.

El ONNX embebe scaler (+PCA) + modelo como un único grafo; las categóricas se
codifican en el cliente (label → índice) porque onnxruntime-web maneja mucho
mejor tensores float que string. El resultado es idéntico al de make predict.

Uso:
    uv run python tools/export_onnx.py            # exporta todos los modelos
    uv run python tools/export_onnx.py --dry-run  # lista qué se exportará
    uv run python tools/export_onnx.py --model RandomForest   # solo uno

Limitaciones (honestas):
  - skl2onnx no soporta CatBoost: se omite con un aviso.
  - El demo asume preprocesado por defecto. Si LOGCOLS, ORDINAL_MAPPINGS,
    COLS_TO_DROP o _feature_engineering están personalizados, la demo NO
    replica esos pasos y la predicción puede divergir de make predict.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from {{ project_slug }}.utils.paths import ARTIFACTS_DIR, MODELS_DIR, PROJECT_DIR

ML_TYPE = "{{ ml_type }}"
TASK_TYPE = "{{ task_type }}"
NN_MODEL = "{{ nn_model }}"
PROJECT_NAME = "{{ project_name }}"

DEMO_DIR = PROJECT_DIR / "demo"
DEMO_MODELS_DIR = DEMO_DIR / "models"

TARGET_OPSET = 15  # estable en onnxruntime-web (WASM)


def _load_artifacts() -> dict:
    """Carga los artefactos de preprocesado guardados por build_features.py."""
    a = {}
    a["feature_names"] = joblib.load(ARTIFACTS_DIR / "feature_names.joblib")
    a["scaler"] = joblib.load(ARTIFACTS_DIR / "scaler.joblib")

    enc_path = ARTIFACTS_DIR / "encoders.joblib"
    a["encoders"] = joblib.load(enc_path) if enc_path.exists() else {}

    pca_path = ARTIFACTS_DIR / "pca.joblib"
    a["pca"] = joblib.load(pca_path) if pca_path.exists() else None

    te_path = ARTIFACTS_DIR / "target_encoder.joblib"
    a["target_encoder"] = joblib.load(te_path) if te_path.exists() else None

    th_path = ARTIFACTS_DIR / "threshold.joblib"
    a["threshold"] = joblib.load(th_path) if th_path.exists() else None

    od_path = ARTIFACTS_DIR / "output_dim.joblib"
    a["output_dim"] = joblib.load(od_path) if od_path.exists() else None
    return a


def _check_custom_preprocessing() -> None:
    """Avisa si el preprocesado se desvía del default que la demo puede replicar."""
    try:
        from {{ project_slug }}.features.build_features import (
            COLS_TO_DROP,
            LOGCOLS,
            ORDINAL_MAPPINGS,
        )
    except ImportError:
        return
    custom = [k for k, v in
              {"LOGCOLS": LOGCOLS, "ORDINAL_MAPPINGS": ORDINAL_MAPPINGS,
               "COLS_TO_DROP": COLS_TO_DROP}.items()
              if v]
    if custom:
        print(f"  ⚠ AVISO: {', '.join(custom)} no vacíos en build_features.py. "
              f"La demo NO replica esos pasos — la predicción puede divergir de make predict.")


def _numeric_refs(scaler) -> tuple:
    """
    (ref, lo, hi) por feature para rellenar el form con valores plausibles.
    StandardScaler → ref=media, rango media±2·scale. MinMaxScaler → rango real.
    """
    import numpy as np
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = np.asarray(scaler.mean_, dtype=float)
        scale = np.asarray(scaler.scale_, dtype=float)
        return mean.tolist(), (mean - 2 * scale).tolist(), (mean + 2 * scale).tolist()
    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        lo = np.asarray(scaler.data_min_, dtype=float).tolist()
        hi = np.asarray(scaler.data_max_, dtype=float).tolist()
        return [0.5 * (a + b) for a, b in zip(lo, hi)], lo, hi
    return None, None, None


def _feature_specs(art: dict) -> list[dict]:
    """Descripción de cada feature para meta.json (nombre, tipo, clases si categórica)."""
    ref, lo, hi = _numeric_refs(art["scaler"])
    specs = []
    for i, name in enumerate(art["feature_names"]):
        if name in art["encoders"]:
            le = art["encoders"][name]
            specs.append({
                "name": name,
                "type": "categorical",
                "classes": [str(c) for c in le.classes_.tolist()],
            })
        else:
            spec = {"name": name, "type": "numeric"}
            if ref is not None:
                spec["ref"] = round(ref[i], 6)
                spec["min"] = round(lo[i], 6)
                spec["max"] = round(hi[i], 6)
            specs.append(spec)
    return specs


def _export_sklearn(name: str, model, art: dict) -> bool:
    """Convierte un modelo sklearn (scikit-learn) a ONNX. True si se exportó."""
    from sklearn.pipeline import Pipeline
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    n_features = len(art["feature_names"])
    steps = [("scaler", art["scaler"])]
    if art["pca"] is not None:
        steps.append(("pca", art["pca"]))
    steps.append(("model", model))
    pipe = Pipeline(steps)

    options = {}
    if hasattr(model, "predict_proba"):
        # zipmap=False → output_label + output_probability como tensores,
        # no como Map (onnxruntime-web no parsea Maps cómodamente).
        options[id(model)] = {"zipmap": False}

    onx = to_onnx(
        pipe,
        [("float_input", FloatTensorType([None, n_features]))],
        target_opset=TARGET_OPSET,
        options=options,
    )
    out_path = DEMO_MODELS_DIR / f"{name}.onnx"
    import onnx
    onnx.save(onx, str(out_path))
    print(f"    {name}.joblib → {out_path.name} ({out_path.stat().st_size/1024:.0f} KB)")
    return True


def _affine_from_scaler(scaler) -> tuple:
    """
    Coeficientes a·x+b que replica el scaler como capa lineal torch.
    Soporta StandardScaler y MinMaxScaler (los dos del template).
    """
    import numpy as np
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        mean = np.asarray(scaler.mean_, dtype=np.float64)
        return 1.0 / scale, -mean / scale
    if hasattr(scaler, "data_min_") and hasattr(scaler, "scale_"):
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        data_min = np.asarray(scaler.data_min_, dtype=np.float64)
        rmin = scaler.feature_range[0]
        return scale, rmin - data_min * scale
    raise NotImplementedError("Scaler no soportado para export ONNX")


def _export_nn(art: dict) -> bool:
    """Exporta la red neuronal ({{ nn_model }}) a ONNX embebiendo scaler+PCA."""
    import numpy as np
    import torch
    from {{ project_slug }}.models.train_model import _build_model

    output_dim = art["output_dim"]
    if output_dim is None:
        raise SystemExit("  ✗ output_dim.joblib no encontrado. Ejecuta make train.")

    input_dim = art["pca"].n_components_ if art["pca"] is not None else len(art["feature_names"])
    net = _build_model(input_dim=input_dim, output_dim=int(output_dim))
    weights_path = MODELS_DIR / f"{NN_MODEL}_final.pt"
    if not weights_path.exists():
        raise SystemExit(f"  ✗ {weights_path.name} no encontrado. Ejecuta make train.")
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net.eval()

    factor, offset = _affine_from_scaler(art["scaler"])
    factor_t = torch.tensor(factor, dtype=torch.float32)
    offset_t = torch.tensor(offset, dtype=torch.float32)

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.factor = factor_t
            self.offset = offset_t
            self.has_pca = art["pca"] is not None
            if self.has_pca:
                pca = art["pca"]
                w = torch.tensor(np.asarray(pca.components_, dtype=np.float64),
                                 dtype=torch.float32)
                b = torch.tensor(-np.asarray(pca.mean_, dtype=np.float64) @ np.asarray(
                    pca.components_, dtype=np.float64).T, dtype=torch.float32)
                self.register_buffer("pca_w", w)
                self.register_buffer("pca_b", b)

        def forward(self, x):
            x = x * self.factor + self.offset
            if self.has_pca:
                x = x @ self.pca_w.T + self.pca_b
            return self.net(x)

    wrapper = Wrapper().eval()
    dummy = torch.randn(1, len(art["feature_names"]), dtype=torch.float32)
    out_path = DEMO_MODELS_DIR / f"{NN_MODEL}.onnx"
    torch.onnx.export(
        wrapper, dummy, str(out_path),
        input_names=["float_input"], output_names=["output"],
        dynamic_axes={"float_input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=TARGET_OPSET,
    )
    print(f"    {NN_MODEL}_final.pt → {out_path.name} ({out_path.stat().st_size/1024:.0f} KB)")
    return True


def _output_kind(model) -> str:
    """classification | regression | clustering según el modelo."""
    if hasattr(model, "predict_proba"):
        return "classification"
    if TASK_TYPE == "regresion":
        return "regression"
    return "clustering"


def _classes(art: dict, model) -> list[str] | None:
    if not hasattr(model, "predict_proba"):
        return None
    if art["target_encoder"] is not None:
        return [str(c) for c in art["target_encoder"].classes_.tolist()]
    if hasattr(model, "classes_"):
        return [str(c) for c in model.classes_.tolist()]
    return None


def _build_meta(art: dict, exported: list[dict]) -> dict:
    return {
        "project": {
            "name": "{{ project_name }}",
            "description": "{{ project_description }}",
            "ml_type": ML_TYPE,
            "task_type": TASK_TYPE,
        },
        "features": _feature_specs(art),
        "models": exported,
        "threshold": art["threshold"],
        "target_encoder": bool(art["target_encoder"]),
    }


_NAV = """<nav class="topnav">
  <a class="brand" href="index.html">{{ project_name }}</a>
  <div class="links">
    <a href="index.html">Home</a>
    <a href="docs.html">Docs</a>
    <a href="try.html" class="cta">Try model</a>
{% if use_mcp %}    <a href="mcp.html">MCP</a>
{% endif %}  </div>
</nav>"""


def _render_docs() -> None:
    """Genera demo/docs.html desde README.md en Python (sin JS de terceros)."""
    try:
        import markdown
    except ImportError:
        print("    ⚠ markdown no instalado (extra onnx) — docs.html no se actualizó")
        return
    readme = PROJECT_DIR / "README.md"
    if not readme.exists():
        print("    ⚠ README.md no encontrado — docs.html no se actualizó")
        return
    body = markdown.markdown(readme.read_text(), extensions=["tables", "fenced_code"])
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Documentación — {PROJECT_NAME}</title>
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
{_NAV}
<div class="container doc-body">
  <h1>Documentación</h1>

  <div class="notice hint">Documentación generada desde el <code>README.md</code> por
    <code>make demo-export</code> (renderizada en Python, sin JavaScript). Para la
    documentación completa de la API del paquete, corré <code>make docs</code> (Sphinx)
    y mirá <code>docs/build/html/</code>.</div>

  <div class="readme-content">{body}</div>
</div>

<script src="assets/app.js"></script>
</body>
</html>
"""
    (DEMO_DIR / "docs.html").write_text(html)
    print("    README.md → demo/docs.html (renderizado en Python)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="lista modelos sin exportar")
    parser.add_argument("--model", help="exporta solo un modelo (por nombre, sin extensión)")
    args = parser.parse_args(argv)

    if not MODELS_DIR.exists():
        raise SystemExit("  ✗ models/ no existe. Ejecuta make train primero.")

    DEMO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    art = _load_artifacts()
    _check_custom_preprocessing()

    if ML_TYPE == "redes_neuronales":
        if args.dry_run:
            print(f"  → {NN_MODEL}_final.pt  se exportará como {NN_MODEL}.onnx")
            return
        _export_nn(art)
        kind = "classification" if TASK_TYPE == "clasificacion" else "regression"
        classes = None
        if art["target_encoder"] is not None:
            classes = [str(c) for c in art["target_encoder"].classes_.tolist()]
        _write_meta(_build_meta(art, [{"name": NN_MODEL, "onnx": f"{NN_MODEL}.onnx",
                                       "kind": kind, "classes": classes}]))
        _render_docs()
        return

    candidates = sorted(MODELS_DIR.glob("*.joblib"))
    if not candidates:
        raise SystemExit("  ✗ No hay modelos en models/*.joblib. Ejecuta make train.")

    skip_names = {"target_encoder", "encoders", "scaler", "pca"}
    exported: list[dict] = []
    for path in candidates:
        name = path.stem
        if name in skip_names:
            continue
        if args.model and name != args.model:
            continue
        model = joblib.load(path)

        if args.dry_run:
            print(f"  → {name}.joblib  ({type(model).__name__})")
            continue

        try:
            _export_sklearn(name, model, art)
        except Exception as exc:  # noqa: BLE001 — un modelo no debe tumbar al resto
            print(f"  ⚠ {name}.joblib no exportado ({type(model).__name__}): "
                  f"{type(exc).__name__}: {exc}")
            continue

        exported.append({
            "name": name,
            "onnx": f"{name}.onnx",
            "kind": _output_kind(model),
            "classes": _classes(art, model),
        })

    if not args.dry_run:
        if not exported:
            raise SystemExit("  ✗ Ningún modelo se pudo exportar a ONNX.")
        _write_meta(_build_meta(art, exported))
        _render_docs()
        print(f"--> {len(exported)} modelos exportados a {DEMO_MODELS_DIR}")
        print(f"    meta.json actualizado. Sirve la demo:  python -m http.server -d demo")


def _write_meta(meta: dict) -> None:
    meta_path = DEMO_MODELS_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"    meta.json guardado ({len(meta['features'])} features, "
          f"{len(meta['models'])} modelos)")


if __name__ == "__main__":
    main()
