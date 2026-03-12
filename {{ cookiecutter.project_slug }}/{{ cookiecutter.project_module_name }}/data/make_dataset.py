import pandas as pd
from {{ cookiecutter.project_module_name }}.utils.paths import RAW_DATA_DIR


def load_data(filename: str = "<nombre>.csv") -> pd.DataFrame:
    """
    Carga el dataset desde data/raw/.

    Parameters
    ----------
    filename : str
        Nombre del archivo CSV dentro de data/raw/.

    Returns
    -------
    pd.DataFrame
    """
    file_path = RAW_DATA_DIR / filename
    print(f"--> Cargando datos desde {file_path}...")
    df = pd.read_csv(file_path)
    print(f"    Shape: {df.shape}")
    return df
