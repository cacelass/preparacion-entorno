import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from {{ cookiecutter.project_module_name }}.utils.paths import ARTIFACTS_DIR