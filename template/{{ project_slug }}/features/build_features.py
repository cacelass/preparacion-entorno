{% if ml_type == 'supervisado' %}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib
from loguru import logger
from {{ project_slug }}.utils.paths import PROCESSED_DATA_DIR, ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# Configuración de codificación ordinal
# ---------------------------------------------------------------------------
ORDINAL_MAPPINGS: dict = {
    # Ejemplo:
    # "education": {
    #     "illiterate": 1, "basic.4y": 2, "basic.6y": 3, "basic.9y": 4,
    #     "high.school": 5, "professional.course": 6, "university.degree": 7,
    #     "unknown": 8,
    # },
}

COLS_TO_DROP: list = [
    # "duration",     # fuga de datos
    # "nr_employed",  # alta correlación con euribor3m
]

# Columnas a las que aplicar transformación logarítmica (np.log1p).
# Útil para features con distribución muy sesgada (skewness > 1).
# Ejemplo: ["amount", "salary", "tenure_days"]
LOGCOLS: list = []


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    scaler_type: str = "standard",
    test_size: float = 0.2,
    random_state: int = 42,
    use_pca=None,
):
    """
    Pipeline completo de preprocesado para aprendizaje supervisado.

    Pasos:
      1. Elimina duplicados
      2. Feature engineering personalizable (_feature_engineering)
      3. Codificación ordinal (ORDINAL_MAPPINGS)
      4. Elimina columnas no deseadas (COLS_TO_DROP)
      5. Rellena nulos (media/moda)
      6. LabelEncoder para categóricas
      7. Train/test split estratificado
      8. Escalado (StandardScaler o MinMaxScaler)
      9. PCA opcional (use_pca)
      10. Guarda artefactos en artifacts/

    Parameters
    ----------
    scaler_type : "standard" | "minmax"
    use_pca     : None → sin PCA
                  float (0 < n < 1) → nº componentes por varianza explicada, e.g. 0.95
                  int  → nº fijo de componentes, e.g. 10

    Returns
    -------
    X_train, X_test, y_train, y_test  (arrays numpy)
    """
    print(f"--> Preprocesando datos (target='{target_col}', scaler='{scaler_type}', PCA={use_pca})...")

    df = df.copy()

    # 1. Duplicados
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    if n_before - len(df):
        print(f"    Duplicados eliminados: {n_before - len(df)}")

    # 2. Feature engineering
    df = _feature_engineering(df)

    # 2.5 Transformación logarítmica
    df = _apply_logcols(df, LOGCOLS)

    # 3. Codificación ordinal
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            print(f"    Codificación ordinal: {col}")

    # 4. Eliminar columnas
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    if cols_present:
        df.drop(columns=cols_present, inplace=True)
        print(f"    Columnas eliminadas: {cols_present}")

    # 5. X / y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 6. Nulos
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # 7. LabelEncoder
    encoders = {}  # guardamos un encoder por columna categórica para reproducibilidad
    le = LabelEncoder()
    for col in cat_cols:
        le_col = LabelEncoder()
        X[col] = le_col.fit_transform(X[col].astype(str))
        encoders[col] = le_col

    if y.dtype == object or str(y.dtype) == "category":
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))
        encoders["__target__"] = le_target
        joblib.dump(le_target, ARTIFACTS_DIR / "target_encoder.joblib")
        print("    Target codificado → target_encoder.joblib")

    # Guardar todos los encoders en un único joblib (reproducibilidad inferencia)
    joblib.dump(encoders, ARTIFACTS_DIR / "encoders.joblib")
    if encoders:
        cols_encoded = [c for c in encoders if c != "__target__"]
        print(f"    Encoders guardados → encoders.joblib  ({cols_encoded})")

    # 8. Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # Guardar nombres de features originales (antes de PCA) para test_model()
    joblib.dump(list(X.columns), ARTIFACTS_DIR / "feature_names.joblib")
    print(f"    feature_names.joblib guardado ({len(X.columns)} features)")

    # 9. Escalado
    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    print(f"    Scaler guardado → scaler.joblib")

    # threshold.joblib se genera DESPUÉS del entrenamiento, no aquí.
    # Descomenta find_best_threshold en predict_model.py (solo binaria) y
    # guárdalo al final de train_model.py:
    #   joblib.dump(best_threshold, ARTIFACTS_DIR / "threshold.joblib")

    # 10. PCA opcional
    if use_pca is not None:
        X_train, X_test = _apply_pca(X_train, X_test, use_pca)

    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"    Proporción clases (train): {pd.Series(y_train).value_counts(normalize=True).to_dict()}")

    # Guardar conjuntos procesados
    pd.DataFrame(X_train).to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(PROCESSED_DATA_DIR  / "X_test.csv",  index=False)
    pd.Series(y_train).to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    pd.Series(y_test).to_csv(PROCESSED_DATA_DIR  / "y_test.csv",  index=False)

    return X_train, X_test, y_train, y_test


def _apply_pca(X_train, X_test, n_components):
    """
    Aplica PCA a train/test y guarda el objeto PCA en artifacts/.

    Parameters
    ----------
    n_components : float (varianza) | int (componentes fijos)
                   Ejemplos: 0.95 → 95% varianza | 10 → 10 componentes

    ¿Cuándo usar PCA antes del clasificador?
      - Muchas features correladas (|r| > 0.8 en varios pares)
      - Alta dimensionalidad (>50 features) → riesgo de maldición dimensional
      - Modelos lentos en alta dimensión (SVM, KNN)
      - Datos con ruido: PCA elimina las componentes de menor varianza

    ¿Cuándo NO usar PCA?
      - Cuando la interpretabilidad de features es crítica
      - Árboles y ensembles (RandomForest, XGBoost): ya gestionan la
        dimensionalidad internamente; PCA no suele mejorar resultados
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)
    joblib.dump(pca, ARTIFACTS_DIR / "pca.joblib")

    n_comp = pca.n_components_
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"    PCA: {X_train.shape[1]} → {n_comp} componentes "
          f"({var_exp:.1%} varianza explicada) → pca.joblib")
    return X_train_pca, X_test_pca


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformaciones y nuevas variables antes del modelado.
    Edita esta función según las necesidades del problema.

    Ejemplos comunes:
      df['was_contacted'] = df['pdays'].apply(lambda x: 0 if x == 999 else 1)
      df['total_loans']   = df['housing'] + df['loan']
    """
    # --- Añade tus transformaciones aquí ---
    return df


def _apply_logcols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Aplica transformación logarítmica np.log1p() a las columnas indicadas.

    Úsala con features numéricas de distribución muy sesgada (skewness > 1)
    para acercarlas a una distribución normal antes del escalado.

    Parameters
    ----------
    df   : DataFrame con las columnas a transformar.
    cols : Lista de nombres de columna. Las columnas que no existan en df
           se ignoran con un aviso. Ejemplo: LOGCOLS = ["amount", "tenure_days"]

    Notes
    -----
    - np.log1p(x) = log(1 + x) → evita log(0) cuando hay ceros.
    - Para valores negativos, aplica primero un offset: x - x.min() + 1.
    - Configura LOGCOLS en la sección de constantes de este fichero.
    """
    if not cols:
        return df

    df = df.copy()
    applied, skipped = [], []

    for col in cols:
        if col not in df.columns:
            skipped.append(col)
            continue
        if df[col].min() < 0:
            offset = -df[col].min() + 1
            df[col] = np.log1p(df[col] + offset)
            logger.warning(f"logcols | '{col}' tiene valores negativos → offset {offset:.4f} aplicado antes de log1p")
        else:
            df[col] = np.log1p(df[col])
        applied.append(col)

    if applied:
        logger.info(f"logcols | log1p aplicado → {applied}")
    if skipped:
        logger.warning(f"logcols | columnas no encontradas (ignoradas) → {skipped}")

    return df


def process_input(df_new: pd.DataFrame) -> np.ndarray:
    """
    Preprocesa nuevos datos para inferencia usando los artefactos guardados.
    Aplica: feature_engineering → ordinal → drop → encode (encoders.joblib)
            → scaler → PCA (si existe).

    Los encoders.joblib garantizan que el mapping de categorías sea idéntico
    al del entrenamiento, evitando silenciosos errores de codificación.
    """
    import os
    scaler   = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib") if (ARTIFACTS_DIR / "encoders.joblib").exists() else {}

    df_new = df_new.copy()
    df_new = _feature_engineering(df_new)
    df_new = _apply_logcols(df_new, LOGCOLS)

    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df_new.columns:
            df_new[col] = df_new[col].map(mapping)

    cols_present = [c for c in COLS_TO_DROP if c in df_new.columns]
    if cols_present:
        df_new.drop(columns=cols_present, inplace=True)

    cat_cols = df_new.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if col in encoders:
            # Usar el mismo encoder del entrenamiento → mismo mapping de clases
            le = encoders[col]
            df_new[col] = le.transform(df_new[col].astype(str))
        else:
            # Fallback: re-fit (puede diferir del entrenamiento si hay categorías nuevas)
            le = LabelEncoder()
            df_new[col] = le.fit_transform(df_new[col].astype(str))

    num_cols = df_new.select_dtypes(include=[np.number]).columns
    df_new[num_cols] = df_new[num_cols].fillna(df_new[num_cols].mean())

    X = scaler.transform(df_new)

    pca_path = ARTIFACTS_DIR / "pca.joblib"
    if pca_path.exists():
        pca = joblib.load(pca_path)
        X = pca.transform(X)

    return X


{% elif ml_type == 'no_supervisado' %}
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from loguru import logger
from {{ project_slug }}.utils.paths import PROCESSED_DATA_DIR, ARTIFACTS_DIR

# Columnas a las que aplicar transformación logarítmica (np.log1p).
LOGCOLS: list = []


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
    np.ndarray escalado listo para clustering / PCA / UMAP
    """
    print("--> Preprocesando datos (no supervisado)...")
    df = df.copy()

    threshold = 0.5
    null_pct   = df.isnull().mean()
    cols_to_drop = null_pct[null_pct > threshold].index.tolist()
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"    Columnas eliminadas (>{threshold*100:.0f}% nulos): {cols_to_drop}")

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Transformación logarítmica
    df = _apply_logcols(df, LOGCOLS)

    le = LabelEncoder()
    encoders = {}
    for col in cat_cols:
        le_col = LabelEncoder()
        df[col] = le_col.fit_transform(df[col].astype(str))
        encoders[col] = le_col

    # Guardar nombres de features y encoders para test_model()
    joblib.dump(list(df.columns), ARTIFACTS_DIR / "feature_names.joblib")
    joblib.dump(encoders, ARTIFACTS_DIR / "encoders.joblib")
    print(f"    feature_names.joblib guardado ({len(df.columns)} features)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    print(f"    Shape final: {X_scaled.shape}")

    pd.DataFrame(X_scaled, columns=df.columns).to_csv(
        PROCESSED_DATA_DIR / "X_processed.csv", index=False
    )
    return X_scaled


def _apply_logcols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Aplica transformación logarítmica np.log1p() a las columnas indicadas.
    Configura LOGCOLS en la sección de constantes de este fichero.
    """
    if not cols:
        return df
    df = df.copy()
    applied, skipped = [], []
    for col in cols:
        if col not in df.columns:
            skipped.append(col)
            continue
        if df[col].min() < 0:
            offset = -df[col].min() + 1
            df[col] = np.log1p(df[col] + offset)
            logger.warning(f"logcols | '{col}' tiene valores negativos → offset {offset:.4f} aplicado")
        else:
            df[col] = np.log1p(df[col])
        applied.append(col)
    if applied:
        logger.info(f"logcols | log1p aplicado → {applied}")
    if skipped:
        logger.warning(f"logcols | columnas no encontradas (ignoradas) → {skipped}")
    return df


{% elif ml_type == 'redes_neuronales' %}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib
from loguru import logger
from {{ project_slug }}.utils.paths import PROCESSED_DATA_DIR, ARTIFACTS_DIR

# Columnas a las que aplicar transformación logarítmica (np.log1p).
LOGCOLS: list = []


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    use_pca=None,
):
    """
    Pipeline de preprocesado para redes neuronales con PyTorch.

    Parameters
    ----------
    use_pca : None | float (varianza) | int (componentes)
              PCA antes de la red reduce ruido y dimensionalidad.
              Útil si hay muchas features muy correladas.

    Returns
    -------
    X_train, X_test, y_train, y_test (DataFrames/Series)
    """
    print(f"--> Preprocesando datos para red neuronal (target='{target_col}', PCA={use_pca})...")
    df = df.copy()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # Transformación logarítmica
    X = _apply_logcols(X, LOGCOLS)

    encoders = {}
    for col in cat_cols:
        le_col = LabelEncoder()
        X[col] = le_col.fit_transform(X[col].astype(str))
        encoders[col] = le_col

    if y.dtype == object or str(y.dtype) == "category":
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y.astype(str)), name=target_col)
        encoders["__target__"] = le_target
        joblib.dump(le_target, ARTIFACTS_DIR / "target_encoder.joblib")

    joblib.dump(encoders, ARTIFACTS_DIR / "encoders.joblib")
    if encoders:
        cols_encoded = [c for c in encoders if c != "__target__"]
        print(f"    Encoders guardados → encoders.joblib  ({cols_encoded})")

    # Guardar nombres de features originales (antes de PCA) para test_model()
    joblib.dump(list(X.columns), ARTIFACTS_DIR / "feature_names.joblib")
    print(f"    feature_names.joblib guardado ({len(X.columns)} features)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")

    # PCA opcional
    if use_pca is not None:
        pca = PCA(n_components=use_pca, random_state=42)
        X_tr_arr = pca.fit_transform(X_train_s.values)
        X_te_arr = pca.transform(X_test_s.values)
        joblib.dump(pca, ARTIFACTS_DIR / "pca.joblib")
        n_comp  = pca.n_components_
        var_exp = pca.explained_variance_ratio_.sum()
        print(f"    PCA: {X_train_s.shape[1]} → {n_comp} componentes ({var_exp:.1%} varianza)")
        X_train_s = pd.DataFrame(X_tr_arr, columns=[f"PC{i+1}" for i in range(n_comp)])
        X_test_s  = pd.DataFrame(X_te_arr, columns=[f"PC{i+1}" for i in range(n_comp)])

    print(f"    Train: {X_train_s.shape} | Test: {X_test_s.shape}")
    return (
        X_train_s,
        X_test_s,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def process_input(df_new: pd.DataFrame) -> "np.ndarray":
    """
    Preprocesa nuevos datos para inferencia usando los artefactos guardados.
    Aplica: logcols → encode (encoders.joblib) → scaler → PCA (si existe).
    """
    scaler   = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib") if (ARTIFACTS_DIR / "encoders.joblib").exists() else {}

    df_new = df_new.copy()
    df_new = _apply_logcols(df_new, LOGCOLS)

    cat_cols = df_new.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if col in encoders:
            df_new[col] = encoders[col].transform(df_new[col].astype(str))
        else:
            le = LabelEncoder()
            df_new[col] = le.fit_transform(df_new[col].astype(str))

    num_cols = df_new.select_dtypes(include=[np.number]).columns
    df_new[num_cols] = df_new[num_cols].fillna(df_new[num_cols].mean())
    X = scaler.transform(df_new)
    pca_path = ARTIFACTS_DIR / "pca.joblib"
    if pca_path.exists():
        X = joblib.load(pca_path).transform(X)
    return X


def _apply_logcols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Aplica transformación logarítmica np.log1p() a las columnas indicadas.
    Configura LOGCOLS en la sección de constantes de este fichero.
    """
    if not cols:
        return df
    df = df.copy()
    applied, skipped = [], []
    for col in cols:
        if col not in df.columns:
            skipped.append(col)
            continue
        if df[col].min() < 0:
            offset = -df[col].min() + 1
            df[col] = np.log1p(df[col] + offset)
            logger.warning(f"logcols | '{col}' tiene valores negativos → offset {offset:.4f} aplicado")
        else:
            df[col] = np.log1p(df[col])
        applied.append(col)
    if applied:
        logger.info(f"logcols | log1p aplicado → {applied}")
    if skipped:
        logger.warning(f"logcols | columnas no encontradas (ignoradas) → {skipped}")
    return df


{% elif ml_type == 'hibrido' %}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.semi_supervised import LabelSpreading
import joblib
from loguru import logger
from {{ project_slug }}.utils.paths import PROCESSED_DATA_DIR, ARTIFACTS_DIR


COLS_TO_DROP: list = []
ORDINAL_MAPPINGS: dict = {}
LOGCOLS: list = []


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    strategy: str = "pca_clf",
    n_clusters: int = 5,
    labeled_fraction: float = 0.3,
    pca_variance: float = 0.95,
):
    """
    Pipeline híbrido: preprocesado base + transformación según estrategia.

    Estrategias
    -----------
    'pca_clf'         → Escala → PCA → split → listo para clasificador.
                        Conserva `pca_variance` de la varianza (default 95%).

    'umap_clf'        → Escala → UMAP 2D → split → listo para clasificador.
                        Reducción no lineal; captura estructuras complejas.
                        Requiere: pip install umap-learn

    'kmeans_features' → Escala → KMeans(n_clusters) → añade como nuevas features:
                        · distancia a cada centroide (n_clusters columnas)
                        · etiqueta del cluster asignado (1 columna)
                        → split → listo para clasificador.
                        Útil: el clustering captura grupos latentes del dataset.

    'iso_feature'     → Escala → IsolationForest → añade anomaly_score como
                        feature extra → split → listo para clasificador.
                        Útil: el score de anomalía tiene poder predictivo.

    'semi_supervisado'→ Escala → LabelSpreading sobre fracción sin etiqueta
                        → etiquetas propagadas → split → entrena clasificador.
                        Útil: solo `labeled_fraction` del dataset tiene labels;
                        el resto se etiqueta automáticamente.

    Returns
    -------
    X_train, X_test, y_train, y_test (arrays numpy)
    """
    print(f"--> Preprocesando datos (estrategia='{strategy}')...")
    df = df.copy()

    # ── Preprocesado base ────────────────────────────────────────────────
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    if n_before - len(df):
        print(f"    Duplicados eliminados: {n_before - len(df)}")

    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    if cols_present:
        df.drop(columns=cols_present, inplace=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # Transformación logarítmica
    X = _apply_logcols(X, LOGCOLS)

    encoders = {}
    for col in cat_cols:
        le_col = LabelEncoder()
        X[col] = le_col.fit_transform(X[col].astype(str))
        encoders[col] = le_col

    if y.dtype == object or str(y.dtype) == "category":
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y.astype(str)), name=target_col)
        encoders["__target__"] = le_target
        joblib.dump(le_target, ARTIFACTS_DIR / "target_encoder.joblib")

    joblib.dump(encoders, ARTIFACTS_DIR / "encoders.joblib")
    if encoders:
        cols_encoded = [c for c in encoders if c != "__target__"]
        print(f"    Encoders guardados → encoders.joblib  ({cols_encoded})")

    # Escalar siempre antes de cualquier transformación
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    print(f"    Scaler guardado → scaler.joblib")

    # ── Transformación según estrategia ──────────────────────────────────
    if strategy == "pca_clf":
        X_final, y_final = _strategy_pca(X_scaled, y.values, pca_variance)

    elif strategy == "umap_clf":
        X_final, y_final = _strategy_umap(X_scaled, y.values)

    elif strategy == "kmeans_features":
        X_final, y_final = _strategy_kmeans_features(X_scaled, y.values, n_clusters, random_state)

    elif strategy == "iso_feature":
        X_final, y_final = _strategy_iso_feature(X_scaled, y.values)

    elif strategy == "semi_supervisado":
        X_final, y_final = _strategy_semi_supervised(X_scaled, y.values, labeled_fraction, random_state)

    else:
        raise ValueError(f"Estrategia desconocida: '{strategy}'. "
                         f"Opciones: pca_clf | umap_clf | kmeans_features | iso_feature | semi_supervisado")

    # ── Split estratificado ───────────────────────────────────────────────
    # Guardar nombres de features originales (antes de PCA/UMAP) para test_model()
    joblib.dump(list(X.columns), ARTIFACTS_DIR / "feature_names.joblib")
    print(f"    feature_names.joblib guardado ({len(X.columns)} features)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final, test_size=test_size, random_state=random_state,
        stratify=y_final,
    )

    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
    pd.DataFrame(X_train).to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(PROCESSED_DATA_DIR  / "X_test.csv",  index=False)
    pd.Series(y_train).to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    pd.Series(y_test).to_csv(PROCESSED_DATA_DIR  / "y_test.csv",  index=False)

    return X_train, X_test, y_train, y_test


# ── Estrategias privadas ─────────────────────────────────────────────────────

def _strategy_pca(X_scaled, y, pca_variance):
    """PCA → conserva `pca_variance` de la varianza total."""
    pca = PCA(n_components=pca_variance, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, ARTIFACTS_DIR / "pca.joblib")
    n_comp  = pca.n_components_
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"    PCA: {X_scaled.shape[1]} → {n_comp} componentes ({var_exp:.1%} varianza) → pca.joblib")
    return X_pca, y


def _strategy_umap(X_scaled, y, n_components=2):
    """
    UMAP: reducción no lineal a `n_components` dimensiones.

    Ventajas sobre PCA:
      - Captura estructuras no lineales (manifolds)
      - Mejor separación visual de clusters
      - Más útil cuando PCA pierde información relevante

    Parámetros clave de UMAP:
      n_neighbors (15): más alto → estructura global; más bajo → local
      min_dist (0.1): más alto → puntos más dispersos en el embedding
    """
    try:
        import umap
    except ImportError:
        raise ImportError("Instala umap-learn: uv add umap-learn")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        verbose=True,
    )
    X_umap = reducer.fit_transform(X_scaled)
    joblib.dump(reducer, ARTIFACTS_DIR / "umap.joblib")
    print(f"    UMAP: {X_scaled.shape[1]} → {n_components}D → umap.joblib")
    return X_umap, y


def _strategy_kmeans_features(X_scaled, y, n_clusters, random_state):
    """
    KMeans como extractor de features.

    Añade al dataset original:
      · cluster_0 ... cluster_{k-1}: distancia euclidiana a cada centroide
      · cluster_label: índice del cluster asignado (como feature numérica)

    ¿Por qué funciona?
      El clustering captura la estructura global del espacio de features.
      La distancia a cada centroide codifica "qué tan cerca está cada punto
      de cada grupo", lo que puede ser muy informativo para el clasificador.
    """
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)
    km.fit(X_scaled)
    joblib.dump(km, ARTIFACTS_DIR / "kmeans_feature.joblib")

    dists  = km.transform(X_scaled)                        # (n, k)
    labels = km.labels_.reshape(-1, 1).astype(float)       # (n, 1)
    X_new  = np.hstack([X_scaled, dists, labels])

    print(f"    KMeans({n_clusters}) features: {X_scaled.shape[1]} → {X_new.shape[1]} columnas")
    print(f"    Distribución clusters: {dict(zip(*np.unique(km.labels_, return_counts=True)))}")
    return X_new, y


def _strategy_iso_feature(X_scaled, y, contamination=0.05):
    """
    IsolationForest → añade anomaly_score como feature extra.

    El anomaly score es negativo para outliers (más negativo = más raro).
    Si los outliers tienen un comportamiento diferente para el target,
    este score puede mejorar la predicción del clasificador.
    """
    iso = IsolationForest(
        contamination=contamination, n_estimators=200, random_state=42, n_jobs=-1
    )
    iso.fit(X_scaled)
    joblib.dump(iso, ARTIFACTS_DIR / "isolation_forest.joblib")

    scores = iso.decision_function(X_scaled).reshape(-1, 1)  # (n, 1)
    X_new  = np.hstack([X_scaled, scores])

    n_out  = (iso.predict(X_scaled) == -1).sum()
    print(f"    IsolationForest: {n_out} outliers ({n_out/len(X_scaled):.1%}) | "
          f"score añadido como feature → {X_new.shape[1]} columnas")
    return X_new, y


def _strategy_semi_supervised(X_scaled, y, labeled_fraction, random_state):
    """
    LabelSpreading: propaga etiquetas desde datos etiquetados al resto.

    Simula un escenario real donde solo tienes labels para una fracción
    del dataset. El algoritmo propaga las etiquetas por el grafo de similitud.

    ⚠ Esta estrategia modifica y, ya que el y resultante incluye las
      etiquetas propagadas para las muestras que originalmente eran -1.
    """
    rng = np.random.default_rng(random_state)
    y_semi = y.copy().astype(int)

    # Marcar la fracción no etiquetada con -1 (convención de scikit-learn)
    n_unlabeled = int(len(y) * (1 - labeled_fraction))
    unlabeled_idx = rng.choice(len(y), size=n_unlabeled, replace=False)
    y_semi[unlabeled_idx] = -1

    n_labeled = (y_semi != -1).sum()
    print(f"    Semi-supervisado: {n_labeled} etiquetados, {n_unlabeled} sin etiquetar")

    ls = LabelSpreading(kernel="rbf", alpha=0.2, max_iter=100, n_jobs=-1)
    ls.fit(X_scaled, y_semi)
    y_propagated = ls.transduction_.astype(int)
    joblib.dump(ls, ARTIFACTS_DIR / "label_spreading.joblib")

    # Cuántas etiquetas se han propagado con éxito
    changed = (y_propagated != y).sum()
    print(f"    LabelSpreading: {changed} etiquetas propagadas/corregidas")
    return X_scaled, y_propagated


def _apply_logcols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Aplica transformación logarítmica np.log1p() a las columnas indicadas.
    Configura LOGCOLS en la sección de constantes de este fichero.
    """
    if not cols:
        return df
    df = df.copy()
    applied, skipped = [], []
    for col in cols:
        if col not in df.columns:
            skipped.append(col)
            continue
        if df[col].min() < 0:
            offset = -df[col].min() + 1
            df[col] = np.log1p(df[col] + offset)
            logger.warning(f"logcols | '{col}' tiene valores negativos → offset {offset:.4f} aplicado")
        else:
            df[col] = np.log1p(df[col])
        applied.append(col)
    if applied:
        logger.info(f"logcols | log1p aplicado → {applied}")
    if skipped:
        logger.warning(f"logcols | columnas no encontradas (ignoradas) → {skipped}")
    return df
{% endif %}