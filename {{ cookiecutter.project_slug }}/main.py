import sys
import os
import joblib
import pandas as pd

# Importamos las rutas y funciones de tu proyecto
from {{ cookiecutter.project_module_name }}.utils.paths import MODELS_DIR, ARTIFACTS_DIR
from {{ cookiecutter.project_module_name }}.data.make_dataset import load_data
from {{ cookiecutter.project_module_name }}.features.build_features import preprocess_data, process_input
from {{ cookiecutter.project_module_name }}.models.train_model import train_models
from {{ cookiecutter.project_module_name }}.models.predict_model import evaluate_models
