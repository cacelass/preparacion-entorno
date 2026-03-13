{% if cookiecutter.ml_type == 'supervisado' %}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import joblib
from {{ cookiecutter.project_module_name }}.utils.paths import PROCESSED_DATA_DIR, ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# Configuración de codificación ordinal para variables con orden lógico
# ---------------------------------------------------------------------------
# Añade aquí las variables que tengan un orden natural (ej: nivel educativo)
# Si no tienes ninguna, deja el diccionario vacío: {}
ORDINAL_MAPPINGS: dict = {
    # Ejemplo:
    # "education": {
    #     "illiterate": 1, "basic.4y": 2, "basic.6y": 3, "basic.9y": 4,
    #     "high.school": 5, "professional.course": 6, "university.degree": 7,
    #     "unknown": 8,
    # },
}

# Columnas a eliminar antes de modelar (alta correlación, fuga de datos, etc.)
COLS_TO_DROP: list = [
    # Ejemplos comunes:
    # "duration",     # fuga de datos: si es 0, el target siempre es 'no'
    # "nr_employed",  # alta correlación con euribor3m
    # "emp_var_rate", # alta correlación con euribor3m
    # "day_of_week",  # sin poder predictivo
]


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    scaler_type: str = "standard",  # "standard" | "minmax"
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Pipeline completo de preprocesado para aprendizaje supervisado.

    Pasos:
      1. Elimina duplicados
      2. Feature engineering personalizable (editar _feature_engineering)
      3. Aplica codificación ordinal (ORDINAL_MAPPINGS)
      4. Elimina columnas no deseadas (COLS_TO_DROP)
      5. Rellena nulos numéricos con la media, categóricos con la moda
      6. Codifica variables categóricas restantes con LabelEncoder
      7. Train/test split con stratify para mantener proporción de clases
      8. Escala features (StandardScaler o MinMaxScaler)
      9. Guarda scaler y encoder en artifacts/

    Parameters
    ----------
    scaler_type : "standard" (media=0, var=1) | "minmax" (rango [0,1], mejor con outliers)

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    print(f"--> Preprocesando datos (target='{target_col}', scaler='{scaler_type}')...")

    df = df.copy()

    # 1. Eliminar duplicados
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"    Duplicados eliminados: {n_dropped}")

    # 2. Feature engineering (editar la función de abajo según el problema)
    df = _feature_engineering(df)

    # 3. Codificación ordinal
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            print(f"    Codificación ordinal: {col}")

    # 4. Eliminar columnas no deseadas
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    if cols_present:
        df.drop(columns=cols_present, inplace=True)
        print(f"    Columnas eliminadas: {cols_present}")

    # 5. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 6. Nulos numéricos → media; categóricos → moda
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # 7. LabelEncoder para categóricas restantes
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # Si el target es categórico, codificarlo también
    if y.dtype == object or str(y.dtype) == "category":
        y = le.fit_transform(y.astype(str))
        joblib.dump(le, ARTIFACTS_DIR / "target_encoder.joblib")
        print("    Target codificado → target_encoder.joblib")

    # 8. Split estratificado (mantiene proporción de clases)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # ← clave para datasets desbalanceados
    )

    # 9. Escalado
    if scaler_type == "minmax":
        scaler = MinMaxScaler()   # mejor cuando hay outliers fuertes
    else:
        scaler = StandardScaler() # por defecto: media=0, varianza=1

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    print(f"    Scaler guardado → scaler.joblib")
    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"    Proporción clases (train): {pd.Series(y_train).value_counts(normalize=True).to_dict()}")

    # Guardar conjuntos procesados
    pd.DataFrame(X_train).to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    pd.Series(y_train).to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    pd.Series(y_test).to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    return X_train, X_test, y_train, y_test


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformaciones y nuevas variables antes del modelado.
    Edita esta función según las necesidades del problema.

    Ejemplos comunes:
      - Convertir valores centinela en binario:
            df['was_contacted'] = df['pdays'].apply(lambda x: 0 if x == 999 else 1)
            df.drop('pdays', axis=1, inplace=True)
      - Combinar columnas:
            df['total_loans'] = df['housing'] + df['loan']
    """
    # --- Añade tus transformaciones aquí ---
    return df


def process_input(df_new: pd.DataFrame) -> np.ndarray:
    """
    Preprocesa nuevos datos para inferencia usando los artefactos guardados.

    Parameters
    ----------
    df_new : DataFrame con las mismas columnas que los datos de entrenamiento
             (sin la columna target).

    Returns
    -------
    np.ndarray listo para model.predict()
    """
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")

    df_new = df_new.copy()
    df_new = _feature_engineering(df_new)

    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df_new.columns:
            df_new[col] = df_new[col].map(mapping)

    cols_present = [c for c in COLS_TO_DROP if c in df_new.columns]
    if cols_present:
        df_new.drop(columns=cols_present, inplace=True)

    # Codificar categóricas
    cat_cols = df_new.select_dtypes(exclude=[np.number]).columns
    le = LabelEncoder()
    for col in cat_cols:
        df_new[col] = le.fit_transform(df_new[col].astype(str))

    # Nulos
    num_cols = df_new.select_dtypes(include=[np.number]).columns
    df_new[num_cols] = df_new[num_cols].fillna(df_new[num_cols].mean())

    return scaler.transform(df_new)

{% elif cookiecutter.ml_type == 'no_supervisado' %}
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from {{ cookiecutter.project_module_name }}.utils.paths import PROCESSED_DATA_DIR, ARTIFACTS_DIR


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """
    Pipeline de preprocesado para clustering (aprendizaje no supervisado).

    Pasos:
      1. Elimina columnas con >50% de nulos
      2. Rellena nulos restantes con la media/moda
      3. Codifica variables categóricas con LabelEncoder
      4. Escala con StandardScaler
      5. Guarda el scaler en artifacts/

    Returns
    -------
    np.ndarray escalado listo para clustering
    """
    print("--> Preprocesando datos (no supervisado)...")
    df = df.copy()

    # Eliminar columnas con demasiados nulos
    threshold = 0.5
    null_pct = df.isnull().mean()
    cols_to_drop = null_pct[null_pct > threshold].index.tolist()
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"    Columnas eliminadas (>{threshold*100:.0f}% nulos): {cols_to_drop}")

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    print(f"    Shape final: {X_scaled.shape}")

    pd.DataFrame(X_scaled, columns=df.columns).to_csv(
        PROCESSED_DATA_DIR / "X_processed.csv", index=False
    )
    return X_scaled

{% elif cookiecutter.ml_type == 'redes_neuronales' %}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from {{ cookiecutter.project_module_name }}.utils.paths import PROCESSED_DATA_DIR, ARTIFACTS_DIR


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Pipeline de preprocesado para redes neuronales con PyTorch.

    Pasos:
      1. Rellena nulos
      2. Codifica categóricas con LabelEncoder
      3. Split estratificado
      4. Escala con StandardScaler
      5. Guarda artefactos

    Returns
    -------
    X_train, X_test, y_train, y_test (DataFrames/Series)
    """
    print(f"--> Preprocesando datos para red neuronal (target='{target_col}')...")
    df = df.copy()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == object or str(y.dtype) == "category":
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)
        joblib.dump(le, ARTIFACTS_DIR / "target_encoder.joblib")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    print(f"    Train: {X_train_s.shape} | Test: {X_test_s.shape}")

    return X_train_s, X_test_s, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def process_input(df_new: pd.DataFrame) -> "np.ndarray":
    import numpy as np
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    df_new = df_new.copy()
    cat_cols = df_new.select_dtypes(exclude=[np.number]).columns
    le = LabelEncoder()
    for col in cat_cols:
        df_new[col] = le.fit_transform(df_new[col].astype(str))
    num_cols = df_new.select_dtypes(include=[np.number]).columns
    df_new[num_cols] = df_new[num_cols].fillna(df_new[num_cols].mean())
    return scaler.transform(df_new)
{% endif %}
