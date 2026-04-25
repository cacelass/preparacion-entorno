"""
test_make_dataset.py — Tests para {{ project_slug }}/data/make_dataset.py
"""
import pandas as pd
import numpy as np
import pytest


def test_load_data_reads_csv(patch_paths):
    """load_data debe leer un CSV válido y devolver un DataFrame."""
    from {{ project_slug }}.data.make_dataset import load_data

    # Crear CSV temporal en RAW_DATA_DIR (ya parcheado)
    sample = pd.DataFrame(
        np.random.randn(50, 3),
        columns=["a", "b", "c"],
    )
    csv_path = patch_paths["RAW_DATA_DIR"] / "test.csv"
    sample.to_csv(csv_path, index=False)

    df = load_data("test.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (50, 3)
    assert list(df.columns) == ["a", "b", "c"]


def test_load_data_raises_on_missing_file(patch_paths):
    """load_data debe lanzar una excepción si el archivo no existe."""
    from {{ project_slug }}.data.make_dataset import load_data
    with pytest.raises(Exception):
        load_data("no_existe.csv")


{% if ml_type == "redes_neuronales" %}
def test_load_data_polars_reads_csv(patch_paths):
    """load_data_polars debe cargar un CSV y devolver un DataFrame de Polars."""
    polars = pytest.importorskip("polars")
    from {{ project_slug }}.data.make_dataset import load_data_polars, polars_to_pandas

    sample = pd.DataFrame(
        np.random.randn(40, 3),
        columns=["x", "y", "z"],
    )
    csv_path = patch_paths["RAW_DATA_DIR"] / "polars_test.csv"
    sample.to_csv(csv_path, index=False)

    df_pl = load_data_polars("polars_test.csv")
    assert df_pl.shape == (40, 3)


def test_polars_to_pandas_conversion(patch_paths):
    """polars_to_pandas debe retornar un DataFrame de pandas."""
    polars = pytest.importorskip("polars")
    from {{ project_slug }}.data.make_dataset import load_data_polars, polars_to_pandas

    sample = pd.DataFrame(
        np.random.randn(30, 2),
        columns=["p", "q"],
    )
    csv_path = patch_paths["RAW_DATA_DIR"] / "conv_test.csv"
    sample.to_csv(csv_path, index=False)

    df_pl = load_data_polars("conv_test.csv")
    df_pd = polars_to_pandas(df_pl)
    assert isinstance(df_pd, pd.DataFrame)
    assert df_pd.shape == (30, 2)
{% endif %}
