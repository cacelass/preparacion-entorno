import pandas as pd
from {{ project_slug }}.utils.paths import RAW_DATA_DIR
import os
import requests

{% if use_duckdb %}
# ---------------------------------------------------------------------------
# Carga con DuckDB — SQL directo sobre archivos CSV, Parquet y JSON
# ---------------------------------------------------------------------------
# DuckDB ejecuta queries SQL sobre archivos sin cargarlos en memoria completa.
# Es columnar, vectorizado y multicore — ideal para datasets grandes (>1 GB).
#
# Ventajas frente a pandas.read_csv:
#   - Selección de columnas en el query → menos memoria desde el inicio
#   - Filtrado, agregaciones y joins directamente sobre el archivo
#   - Soporta CSV, Parquet, JSON y JSON-Lines
#   - Sin servidor ni configuración
#
# Ejemplo de uso:
#
#   df = load_data_duckdb(
#       "ventas.csv",
#       query="SELECT region, SUM(importe) AS total FROM datos GROUP BY region"
#   )
#
#   # Cargar solo ciertas columnas:
#   df = load_data_duckdb("clientes.csv", query="SELECT id, edad, churn FROM datos")
#
#   # Desde Parquet:
#   df = load_data_duckdb("features.parquet")
#
#   # Exploración interactiva desde Python:
#   import duckdb
#   rel = duckdb.read_csv(str(RAW_DATA_DIR / "dataset.csv"))
#   print(rel.describe())
# ---------------------------------------------------------------------------

import duckdb as _duckdb


def load_data_duckdb(
    filename: str,
    query: str = None,
    sample_n: int = None,
) -> pd.DataFrame:
    """
    Carga datos con DuckDB: soporta CSV, Parquet y JSON.

    Parameters
    ----------
    filename : nombre del archivo en data/raw/ (incluye extensión)
    query    : SQL opcional sobre la tabla virtual 'datos'.
               Si None, ejecuta SELECT * FROM datos.
               Ejemplo: "SELECT col1, col2 FROM datos WHERE col3 > 0"
    sample_n : si se indica, añade USING SAMPLE N ROWS al query para
               cargar solo una muestra aleatoria (útil para exploración).

    Returns
    -------
    pd.DataFrame

    Formatos soportados automáticamente:
        .csv            → read_csv
        .parquet / .pq  → read_parquet
        .json / .jsonl  → read_json
    """
    file_path = RAW_DATA_DIR / filename
    print(f"--> Cargando datos con DuckDB desde {file_path}...")

    if not file_path.exists():
        raise FileNotFoundError(
            f"\n  Archivo no encontrado: {file_path}\n"
            f"  Coloca tu dataset en: data/raw/{filename}"
        )

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        source = f"read_csv('{file_path}', auto_detect=true)"
    elif suffix in (".parquet", ".pq"):
        source = f"read_parquet('{file_path}')"
    elif suffix in (".json", ".jsonl"):
        source = f"read_json('{file_path}', auto_detect=true)"
    else:
        raise ValueError(f"Formato no soportado: {suffix}. Usa CSV, Parquet o JSON.")

    base_query = query.replace("datos", source) if query else f"SELECT * FROM {source}"
    if sample_n:
        base_query += f" USING SAMPLE {sample_n} ROWS"

    df = _duckdb.query(base_query).df()
    size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"    Shape: {df.shape}  |  Memoria: {size_mb:.2f} MB")
    if sample_n:
        print(f"    (muestra aleatoria de {sample_n} filas)")
    return df


def query_duckdb(sql: str, filename: str = None) -> pd.DataFrame:
    """
    Ejecuta un query SQL arbitrario con DuckDB.

    Si se pasa `filename`, la tabla virtual se llama 'datos'.
    Sin filename, el SQL puede referenciar rutas directamente.

    Ejemplos:
        # Contar nulos por columna
        query_duckdb(
            "SELECT COUNT(*) - COUNT(edad) AS nulos_edad FROM datos",
            filename="clientes.csv"
        )

        # Join entre dos archivos Parquet
        query_duckdb(
            \"\"\"
            SELECT a.id, a.feature, b.label
            FROM read_parquet('data/raw/features.parquet') a
            JOIN read_parquet('data/raw/labels.parquet') b USING (id)
            \"\"\"
        )
    """
    if filename:
        file_path = RAW_DATA_DIR / filename
        suffix    = file_path.suffix.lower()
        if suffix == ".csv":
            source = f"read_csv('{file_path}', auto_detect=true)"
        elif suffix in (".parquet", ".pq"):
            source = f"read_parquet('{file_path}')"
        elif suffix in (".json", ".jsonl"):
            source = f"read_json('{file_path}', auto_detect=true)"
        else:
            raise ValueError(f"Formato no soportado: {suffix}")
        sql = sql.replace("datos", source)

    return _duckdb.query(sql).df()

{% endif %}

{% if ml_type == 'redes_neuronales' %}
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

def download_data(url: str, filename: str, params: dict | None = None, api_key_env: str | None = None) -> pd.DataFrame:
    """
    Descarga datos desde una API externa y los guarda en data/raw/.

    Parameters
    ----------
    url : endpoint de la API
    filename : nombre con el que se guardará el CSV en data/raw/
    params : query params para la petición (opcional)
    api_key_env : nombre de la variable de entorno con la API key (opcional)
    """
    headers = {}
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise EnvironmentError(f"Falta la variable de entorno {api_key_env}")
        headers["Authorization"] = f"Bearer {key}"

    print(f"--> Descargando datos desde {url}...")
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()

    file_path = RAW_DATA_DIR / filename
    content_type = resp.headers.get("Content-Type", "")
    if "json" in content_type:
        df = pd.DataFrame(resp.json())
    else:
        import io
        df = pd.read_csv(io.StringIO(resp.text))
    df.to_csv(file_path, index=False)
    print(f"    Guardado en {file_path} — Shape: {df.shape}")
    return df

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
