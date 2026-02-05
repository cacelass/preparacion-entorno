# credito/utils/paths.py
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Nueva ruta para guardar los transformadores (encoders/scaler)
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

def make_dirs():
    for dir_path in [MODELS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

make_dirs()