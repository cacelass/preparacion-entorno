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
    """load_data debe lanzar FileNotFoundError si el archivo no existe."""
    from {{ project_slug }}.data.make_dataset import load_data
    with pytest.raises(FileNotFoundError):
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

{% if use_duckdb %}
# ---------------------------------------------------------------------------
# Tests DuckDB
# ---------------------------------------------------------------------------
def test_load_data_duckdb_csv(patch_paths):
    """load_data_duckdb carga un CSV y devuelve DataFrame correcto."""
    duckdb = pytest.importorskip("duckdb")
    from {{ project_slug }}.data.make_dataset import load_data_duckdb

    sample = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})
    csv_path = patch_paths["RAW_DATA_DIR"] / "duck_test.csv"
    sample.to_csv(csv_path, index=False)

    df = load_data_duckdb("duck_test.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["a", "b", "c"]


def test_load_data_duckdb_con_query(patch_paths):
    """load_data_duckdb filtra con SQL y devuelve solo filas seleccionadas."""
    pytest.importorskip("duckdb")
    from {{ project_slug }}.data.make_dataset import load_data_duckdb

    sample = pd.DataFrame({"valor": [10, 20, 30, 40], "cat": ["A", "B", "A", "B"]})
    csv_path = patch_paths["RAW_DATA_DIR"] / "duck_query.csv"
    sample.to_csv(csv_path, index=False)

    df = load_data_duckdb("duck_query.csv", query="SELECT * FROM datos WHERE valor > 15")
    assert len(df) == 3
    assert all(df["valor"] > 15)


def test_load_data_duckdb_sample_n(patch_paths):
    """load_data_duckdb con sample_n devuelve como mucho N filas."""
    pytest.importorskip("duckdb")
    from {{ project_slug }}.data.make_dataset import load_data_duckdb

    sample = pd.DataFrame({"x": range(100)})
    csv_path = patch_paths["RAW_DATA_DIR"] / "duck_sample.csv"
    sample.to_csv(csv_path, index=False)

    df = load_data_duckdb("duck_sample.csv", sample_n=10)
    assert len(df) <= 10


def test_load_data_duckdb_parquet(patch_paths):
    """load_data_duckdb carga un Parquet correctamente."""
    pytest.importorskip("duckdb")
    from {{ project_slug }}.data.make_dataset import load_data_duckdb

    sample = pd.DataFrame({"p": [1, 2, 3], "q": [7, 8, 9]})
    pq_path = patch_paths["RAW_DATA_DIR"] / "duck_test.parquet"
    sample.to_parquet(pq_path, index=False)

    df = load_data_duckdb("duck_test.parquet")
    assert df.shape == (3, 2)


def test_load_data_duckdb_archivo_no_encontrado(patch_paths):
    """load_data_duckdb lanza FileNotFoundError si el archivo no existe."""
    pytest.importorskip("duckdb")
    from {{ project_slug }}.data.make_dataset import load_data_duckdb
    import pytest as _pytest

    with _pytest.raises(FileNotFoundError):
        load_data_duckdb("no_existe.csv")


def test_query_duckdb_sql_directo(patch_paths):
    """query_duckdb ejecuta SQL con alias 'datos' sobre un CSV."""
    pytest.importorskip("duckdb")
    from {{ project_slug }}.data.make_dataset import query_duckdb

    sample = pd.DataFrame({"n": [1, 2, 3, 4, 5]})
    csv_path = patch_paths["RAW_DATA_DIR"] / "duck_direct.csv"
    sample.to_csv(csv_path, index=False)

    df = query_duckdb("SELECT COUNT(*) AS total FROM datos", filename="duck_direct.csv")
    assert df["total"].iloc[0] == 5
{% endif %}