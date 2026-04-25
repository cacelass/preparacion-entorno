import pandas as pd
from {{ cookiecutter.project_slug }}.utils.paths import RAW_DATA_DIR

{% if cookiecutter.ml_type == 'redes_neuronales' %}
# ---------------------------------------------------------------------------
# Carga con Polars (más eficiente que pandas para datasets grandes)
# ---------------------------------------------------------------------------
# Polars permite especificar tipos de columna explícitamente al leer, lo que
# reduce el uso de memoria significativamente (Float32 en lugar de Float64,
# Categorical en lugar de String, etc.).
#
# Ejemplo de uso con tipos explícitos:
#
#   import polars as pl
#   df_pl = load_data_polars(
#       "mi_dataset.csv",
#       col_types={
#           "columna_numerica": pl.Float32,
#           "columna_categorica": pl.Categorical,
#           "columna_entero": pl.Int16,
#       }
#   )
#   # Convertir a pandas si el resto del pipeline lo necesita:
#   df = df_pl.to_pandas()
#
# Si el dataset tiene cabecera, omite `has_header=False` y `new_columns`.
# Si no tiene cabecera, usa `has_header=False` y `new_columns=[...]`.
# ---------------------------------------------------------------------------

def load_data_polars(
    filename: str,
    col_types: dict = None,
    has_header: bool = True,
    new_columns: list = None,
):
    """
    Carga un CSV con Polars especificando tipos de columna para minimizar memoria.

    Parameters
    ----------
    filename    : nombre del CSV en data/raw/
    col_types   : dict {nombre_columna: pl.Type} con tipos explícitos.
                  Si None, Polars infiere los tipos automáticamente.
    has_header  : True si el CSV tiene fila de cabecera
    new_columns : lista de nombres de columnas (solo si has_header=False)

    Returns
    -------
    polars.DataFrame

    Ejemplo de ahorro de memoria:
        Sin tipos explícitos (Float64, String) : 100%
        Con Float32 + Categorical              :  ~25-40%
    """
    import polars as pl

    file_path = RAW_DATA_DIR / filename
    print(f"--> Cargando datos con Polars desde {file_path}...")

    kwargs = {"has_header": has_header}
    if col_types:
        kwargs["dtypes"] = col_types
    if new_columns:
        kwargs["new_columns"] = new_columns

    df = pl.read_csv(file_path, **kwargs).drop_nulls()
    size_mb = df.estimated_size() / 1024 / 1024
    print(f"    Shape: {df.shape}  |  Memoria estimada: {size_mb:.2f} MB")
    return df


def polars_to_pandas(df_polars) -> pd.DataFrame:
    """
    Convierte un DataFrame de Polars a pandas.
    Las columnas Categorical se convierten a object.
    """
    return df_polars.to_pandas()

{% endif %}

def load_data(filename: str = "<nombre>.csv") -> pd.DataFrame:
    """
    Carga el dataset desde data/raw/ con pandas.

    Parameters
    ----------
    filename : nombre del CSV en data/raw/  (solo el nombre, sin ruta)

    Returns
    -------
    pd.DataFrame
    """
    file_path = RAW_DATA_DIR / filename
    print(f"--> Cargando datos desde {file_path}...")
    if not file_path.exists():
        raise FileNotFoundError(
            f"\n  Archivo no encontrado: {file_path}\n"
            f"  Coloca tu dataset en: data/raw/{filename}\n"
            f"  O cambia DATA_FILE en main.py con el nombre correcto."
        )
    df = pd.read_csv(file_path)
    print(f"    Shape: {df.shape}")
    print(f"    Tipos:\n{df.dtypes.to_string()}")
    return df