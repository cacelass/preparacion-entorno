import pandas as pd
from {{ cookiecutter.project_module_name }}.utils.paths import RAW_DATA_DIR

def load_data(filename="<csv_name>.csv"):
    """
    Carga el dataset desde la carpeta data/raw.
    """
    file_path = RAW_DATA_DIR / filename
    print(f"--> Cargando datos desde {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"    Datos cargados. Dimensiones: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {filename} en {RAW_DATA_DIR}")
        raise