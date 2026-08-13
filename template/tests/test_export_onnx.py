"""
test_export_onnx.py — Tests para tools/export_onnx.py (demo web).
"""
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

{% if ml_type == "supervisado" or ml_type == "hibrido" %}

def _load_export_module():
    """Carga tools/export_onnx.py como módulo (tools/ no es paquete)."""
    path = Path(__file__).resolve().parents[1] / "tools" / "export_onnx.py"
    spec = importlib.util.spec_from_file_location("export_onnx", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_onnx"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def onnx_env(tmp_path, monkeypatch):
    """Prepara models/ y artifacts/ con un modelo real y rutas en tmp_path."""
    mod = _load_export_module()

    models_dir = tmp_path / "models"
    art_dir = models_dir / "artifacts"
    demo_dir = tmp_path / "demo"
    demo_models = demo_dir / "models"
    for d in [models_dir, art_dir, demo_models]:
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(100, 3), columns=["a", "b", "c"])
    y = (X["a"] + X["b"] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(model, models_dir / "RandomForest.joblib")
    joblib.dump(scaler, art_dir / "scaler.joblib")
    joblib.dump(list(X.columns), art_dir / "feature_names.joblib")
    joblib.dump({}, art_dir / "encoders.joblib")

    monkeypatch.setattr(mod, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mod, "ARTIFACTS_DIR", art_dir)
    monkeypatch.setattr(mod, "DEMO_DIR", demo_dir)
    monkeypatch.setattr(mod, "DEMO_MODELS_DIR", demo_models)
    monkeypatch.setattr(mod, "PROJECT_DIR", tmp_path)
    return dict(mod=mod, models=models_dir, art=art_dir, demo_models=demo_models)


def test_export_creates_onnx_and_meta(onnx_env, capsys):
    """Sin flag exporta .onnx + meta.json + docs.html."""
    onnx_env["mod"].main([])
    out = capsys.readouterr().out
    assert "RandomForest.onnx" in out
    assert (onnx_env["demo_models"] / "RandomForest.onnx").exists()
    meta = json.loads((onnx_env["demo_models"] / "meta.json").read_text())
    assert meta["features"][0]["name"] == "a"
    assert meta["features"][0]["type"] == "numeric"
    assert "ref" in meta["features"][0]
    assert "min" in meta["features"][0]
    assert "max" in meta["features"][0]
    assert meta["models"][0]["name"] == "RandomForest"
    assert meta["models"][0]["kind"] == "classification"
    assert meta["features"][0]["min"] < meta["features"][0]["ref"] < meta["features"][0]["max"]


def test_export_generates_docs_html(onnx_env, capsys):
    """demo/docs.html se genera desde README.md en Python (sin JS de terceros)."""
    readme = onnx_env["mod"].PROJECT_DIR / "README.md"
    readme.write_text("# Mi Proyecto\n\nTabla de contenido **en negrita**.\n")
    onnx_env["mod"].main([])
    docs = onnx_env["mod"].DEMO_DIR / "docs.html"
    assert docs.exists()
    html = docs.read_text()
    assert "<h1>Mi Proyecto</h1>" in html
    assert "<strong>en negrita</strong>" in html
    assert "marked" not in html
    assert "cdn.jsdelivr.net/npm/marked" not in html


def test_export_onnx_run_matches_sklearn(onnx_env):
    """La predicción ONNX (features crudas) debe coincidir con la del pipeline sklearn."""
    import onnxruntime as ort

    onnx_env["mod"].main([])
    model = joblib.load(onnx_env["models"] / "RandomForest.joblib")
    X = pd.DataFrame(
        [[0.5, -1.2, 2.1], [-0.3, 0.8, -1.5], [1.1, 1.1, 0.0]],
        columns=["a", "b", "c"],
    )
    scaler = joblib.load(onnx_env["art"] / "scaler.joblib")
    expected = model.predict(scaler.transform(X))

    sess = ort.InferenceSession(
        str(onnx_env["demo_models"] / "RandomForest.onnx"),
        providers=["CPUExecutionProvider"],
    )
    # El ONNX embebe el scaler: la entrada son las features CRUDAS, sin escalar.
    out = sess.run(None, {"float_input": X.to_numpy().astype(np.float32)})
    label = out[0]
    assert label.ravel().tolist() == expected.tolist()


def test_export_dry_run_writes_nothing(onnx_env, capsys):
    """--dry-run no debe escribir ningún archivo."""
    onnx_env["mod"].main(["--dry-run"])
    assert not list(onnx_env["demo_models"].glob("*.onnx"))
    assert "RandomForest" in capsys.readouterr().out
{% endif %}
