"""
test_build_features.py — Tests para {{ cookiecutter.project_slug }}/features/build_features.py
"""
import numpy as np
import pandas as pd
import pytest


{% if cookiecutter.ml_type == "supervisado" %}
from {{ cookiecutter.project_slug }}.features.build_features import (
    preprocess_data,
    _feature_engineering,
    process_input,
)


def test_preprocess_data_returns_four_splits(df_with_target, patch_paths):
    """preprocess_data debe devolver (X_train, X_test, y_train, y_test)."""
    result = preprocess_data(df_with_target, target_col="target")
    assert len(result) == 4
    X_train, X_test, y_train, y_test = result
    assert X_train.shape[0] > X_test.shape[0]
    assert len(y_train) == X_train.shape[0]
    assert len(y_test)  == X_test.shape[0]


def test_preprocess_data_creates_scaler_artifact(df_with_target, patch_paths):
    """Debe guardar scaler.joblib en ARTIFACTS_DIR."""
    preprocess_data(df_with_target, target_col="target")
    assert (patch_paths["ARTIFACTS_DIR"] / "scaler.joblib").exists()


def test_preprocess_data_with_pca(df_with_target, patch_paths):
    """Con use_pca=0.95 debe reducir dimensionalidad y guardar pca.joblib."""
    X_train, X_test, _, _ = preprocess_data(
        df_with_target, target_col="target", use_pca=0.95
    )
    assert (patch_paths["ARTIFACTS_DIR"] / "pca.joblib").exists()
    # La dimensión reducida debe ser <= la original
    assert X_train.shape[1] <= 4


def test_preprocess_data_saves_processed_csvs(df_with_target, patch_paths):
    """Debe guardar X_train.csv, X_test.csv, y_train.csv, y_test.csv."""
    preprocess_data(df_with_target, target_col="target")
    proc = patch_paths["PROCESSED_DATA_DIR"]
    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        assert (proc / fname).exists(), f"Falta {fname}"


def test_preprocess_data_test_size(df_with_target, patch_paths):
    """test_size=0.3 debe producir ~30% en test."""
    X_train, X_test, _, _ = preprocess_data(
        df_with_target, target_col="target", test_size=0.3
    )
    total = X_train.shape[0] + X_test.shape[0]
    ratio = X_test.shape[0] / total
    assert 0.25 <= ratio <= 0.35


def test_preprocess_data_removes_duplicates(patch_paths):
    """Debe eliminar filas duplicadas antes de procesar."""
    np.random.seed(0)
    df = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
    df["target"] = (df["a"] > 0).astype(int)
    df = pd.concat([df, df.iloc[:10]])  # 10 duplicados

    X_train, X_test, y_train, y_test = preprocess_data(df, target_col="target")
    assert X_train.shape[0] + X_test.shape[0] <= 50


def test_feature_engineering_returns_dataframe(sample_df):
    """_feature_engineering debe devolver un DataFrame."""
    result = _feature_engineering(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == sample_df.shape


def test_process_input_requires_scaler(df_with_target, patch_paths):
    """process_input debe fallar si no existe el scaler (no entrenado aún)."""
    with pytest.raises(Exception):
        process_input(df_with_target.drop(columns=["target"]))


def test_process_input_after_preprocess(df_with_target, patch_paths):
    """Después de preprocess_data, process_input debe transformar datos nuevos."""
    preprocess_data(df_with_target, target_col="target")
    X_new = df_with_target.drop(columns=["target"]).head(5)
    result = process_input(X_new)
    assert result.shape[0] == 5
{% endif %}


{% if cookiecutter.ml_type == "no_supervisado" %}
from {{ cookiecutter.project_slug }}.features.build_features import preprocess_data


def test_preprocess_data_returns_ndarray(df_clustering, patch_paths):
    """preprocess_data debe devolver un numpy array escalado."""
    X = preprocess_data(df_clustering)
    assert isinstance(X, np.ndarray)
    assert X.shape == df_clustering.shape


def test_preprocess_data_creates_scaler_artifact(df_clustering, patch_paths):
    """Debe guardar scaler.joblib en ARTIFACTS_DIR."""
    preprocess_data(df_clustering)
    assert (patch_paths["ARTIFACTS_DIR"] / "scaler.joblib").exists()


def test_preprocess_data_zero_mean_unit_std(df_clustering, patch_paths):
    """StandardScaler debe producir media ~0 y std ~1 por columna."""
    X = preprocess_data(df_clustering)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X.std(axis=0),  1, atol=1e-2)


def test_preprocess_data_drops_high_null_columns(patch_paths):
    """Columnas con >50% de nulos deben eliminarse."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, np.nan, np.nan, np.nan] * 10,
        "b": np.random.randn(60),
    })
    X = preprocess_data(df)
    # La columna 'a' tiene 67% nulos → debe eliminarse → solo 1 columna
    assert X.shape[1] == 1


def test_preprocess_data_saves_processed_csv(df_clustering, patch_paths):
    """Debe guardar X_processed.csv en PROCESSED_DATA_DIR."""
    preprocess_data(df_clustering)
    assert (patch_paths["PROCESSED_DATA_DIR"] / "X_processed.csv").exists()
{% endif %}


{% if cookiecutter.ml_type == "redes_neuronales" %}
from {{ cookiecutter.project_slug }}.features.build_features import preprocess_data, process_input


def test_preprocess_data_returns_dataframes(df_with_target, patch_paths):
    """Para redes neuronales, preprocess_data debe devolver DataFrames."""
    X_train, X_test, y_train, y_test = preprocess_data(
        df_with_target, target_col="target"
    )
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test,  pd.DataFrame)
    assert X_train.shape[0] + X_test.shape[0] == len(df_with_target)


def test_preprocess_data_creates_scaler_artifact(df_with_target, patch_paths):
    """Debe guardar scaler.joblib."""
    preprocess_data(df_with_target, target_col="target")
    assert (patch_paths["ARTIFACTS_DIR"] / "scaler.joblib").exists()


def test_preprocess_data_with_pca(df_with_target, patch_paths):
    """Con use_pca=0.95 debe guardar pca.joblib y reducir dims."""
    X_train, X_test, _, _ = preprocess_data(
        df_with_target, target_col="target", use_pca=0.95
    )
    assert (patch_paths["ARTIFACTS_DIR"] / "pca.joblib").exists()
    assert X_train.shape[1] <= 4


def test_preprocess_data_y_reset_index(df_with_target, patch_paths):
    """y_train e y_test deben tener índice reseteado (0, 1, 2, ...)."""
    _, _, y_train, y_test = preprocess_data(df_with_target, target_col="target")
    assert list(y_train.index) == list(range(len(y_train)))
    assert list(y_test.index)  == list(range(len(y_test)))
{% endif %}


{% if cookiecutter.ml_type == "hibrido" %}
from {{ cookiecutter.project_slug }}.features.build_features import (
    preprocess_data,
    _feature_engineering,
    _strategy_pca,
    _strategy_kmeans_features,
    _strategy_iso_feature,
)


def test_preprocess_data_pca_clf(df_with_target, patch_paths):
    """Estrategia pca_clf: debe guardar scaler.joblib y pca.joblib."""
    preprocess_data(df_with_target, target_col="target", strategy="pca_clf")
    assert (patch_paths["ARTIFACTS_DIR"] / "scaler.joblib").exists()
    assert (patch_paths["ARTIFACTS_DIR"] / "pca.joblib").exists()


def test_preprocess_data_kmeans_features(df_with_target, patch_paths):
    """Estrategia kmeans_features: debe aumentar el nº de columnas."""
    n_clusters = 3
    X_train, X_test, _, _ = preprocess_data(
        df_with_target, target_col="target",
        strategy="kmeans_features", n_clusters=n_clusters,
    )
    # original: 4 features + n_clusters distancias + 1 label = 4 + 3 + 1 = 8
    assert X_train.shape[1] == 4 + n_clusters + 1


def test_preprocess_data_iso_feature(df_with_target, patch_paths):
    """Estrategia iso_feature: debe añadir 1 columna extra (anomaly score)."""
    X_train, X_test, _, _ = preprocess_data(
        df_with_target, target_col="target", strategy="iso_feature"
    )
    assert X_train.shape[1] == 5  # 4 features + 1 score


def test_preprocess_data_invalid_strategy(df_with_target, patch_paths):
    """Una estrategia desconocida debe lanzar ValueError."""
    with pytest.raises(ValueError, match="Estrategia desconocida"):
        preprocess_data(
            df_with_target, target_col="target", strategy="no_existe"
        )


def test_preprocess_data_semi_supervisado(df_with_target, patch_paths):
    """Estrategia semi_supervisado: debe propagar etiquetas."""
    X_train, X_test, y_train, y_test = preprocess_data(
        df_with_target, target_col="target", strategy="semi_supervisado"
    )
    # No debe haber -1 en y_train ni y_test (etiquetas propagadas)
    assert -1 not in set(y_train)


def test_strategy_pca_reduces_dimensions(patch_paths):
    """_strategy_pca debe reducir dimensiones manteniendo >= varianza deseada."""
    X = np.random.randn(200, 10)
    y = np.random.randint(0, 2, 200)
    X_pca, y_out = _strategy_pca(X, y, pca_variance=0.95)
    assert X_pca.shape[0] == 200
    assert X_pca.shape[1] < 10
    assert (patch_paths["ARTIFACTS_DIR"] / "pca.joblib").exists()


def test_strategy_kmeans_features_shape(patch_paths):
    """_strategy_kmeans_features debe añadir n_clusters + 1 columnas extra."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    X_new, _ = _strategy_kmeans_features(X, y, n_clusters=4, random_state=42)
    assert X_new.shape == (100, 5 + 4 + 1)
{% endif %}
