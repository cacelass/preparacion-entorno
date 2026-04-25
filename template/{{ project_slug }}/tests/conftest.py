"""
conftest.py — Fixtures compartidas para todos los tests.
Los fixtures se adaptan automáticamente al ml_type elegido en 
"""
import importlib
import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures de datos sintéticos
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame genérico con 8 columnas numéricas (200 filas)."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(200, 8),
        columns=[f"feat_{i}" for i in range(8)],
    )


{% if ml_type in ["supervisado", "hibrido"] %}
@pytest.fixture
def df_with_target(sample_df):
    """DataFrame con features numéricas + columna target binaria."""
    df = sample_df.copy()
    np.random.seed(42)
    df["target"] = (df["feat_0"] + df["feat_1"] > 0).astype(int)
    return df
{% endif %}


{% if ml_type == "redes_neuronales" %}
@pytest.fixture
def df_with_target(sample_df):
    """
    DataFrame con features numéricas + target multiclase (3 clases).

    Notas:
    - 200 filas: suficiente para que BatchNorm1d no se queje con batch_size=16
    - 8 features: cubre todos los modelos (MLP, CNN1D, LSTM, GRU, Transformer)
    - 3 clases: verifica que output_dim > 2 funciona en todas las arquitecturas
    - target balanceado: evita que los tests fallen por class imbalance severo
    """
    df = sample_df.copy()
    np.random.seed(42)
    score = df["feat_0"] + df["feat_1"] - df["feat_2"]
    df["target"] = pd.cut(score, bins=3, labels=[0, 1, 2]).astype(int)
    return df
{% endif %}


{% if ml_type == "no_supervisado" %}
@pytest.fixture
def df_clustering():
    """DataFrame con 3 clusters naturales (200 muestras, 8 features)."""
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=200, centers=3, n_features=8,
                      cluster_std=0.8, random_state=42)
    return pd.DataFrame(X, columns=[f"feat_{i}" for i in range(8)])
{% endif %}


# ─────────────────────────────────────────────────────────────────────────────
# Fixture de parcheo de rutas (aislamiento del filesystem)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_paths(monkeypatch, tmp_path):
    """
    Redirige todas las constantes de ruta del proyecto a tmp_path.
    Se aplica automáticamente a cada test para evitar escrituras en disco real.
    """
    dirs = {
        "MODELS_DIR":         tmp_path / "models",
        "ARTIFACTS_DIR":      tmp_path / "models" / "artifacts",
        "FIGURES_DIR":        tmp_path / "reports" / "figures",
        "REPORTS_DIR":        tmp_path / "reports",
        "PROCESSED_DATA_DIR": tmp_path / "data" / "processed",
        "RAW_DATA_DIR":       tmp_path / "data" / "raw",
        "RUNS_DIR":           tmp_path / "runs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    slug = "{{ project_slug }}"
    candidate_modules = [
        f"{slug}.utils.paths",
        f"{slug}.models.train_model",
        f"{slug}.models.predict_model",
        f"{slug}.features.build_features",
        f"{slug}.visualization.visualize",
        f"{slug}.data.make_dataset",
    ]
    for mod_path in candidate_modules:
        try:
            mod = importlib.import_module(mod_path)
            for attr, val in dirs.items():
                if hasattr(mod, attr):
                    monkeypatch.setattr(mod, attr, val)
        except (ImportError, ModuleNotFoundError):
            pass

    return dirs
