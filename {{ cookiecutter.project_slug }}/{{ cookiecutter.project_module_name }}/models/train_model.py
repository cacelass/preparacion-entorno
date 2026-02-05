import joblib
from tasa_churn.utils.paths import MODELS_DIR
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):
    """
    Entrena los modelos y guarda los artefactos.
     - Random Forest con parámetros específicos para evitar overfitting.
     - Guarda cada modelo entrenado en la carpeta models.
     - Devuelve un diccionario con los modelos entrenados.
     - X_train: DataFrame con las características de entrenamiento.
     - y_train: Serie con las etiquetas de entrenamiento.
     - return: dict con modelos entrenados, e.g. {'RandomForest': model_object}
    """
    print("--> Entrenando modelos...")
    models = {}

    print("    Entrenando el random forest...")
    # dt = RandomForestClassifier(
    #     n_estimators=100,
    #     class_weight='balanced',  
    #     max_depth=10,
    #     random_state=42
    # )
    
    dt.fit(X_train, y_train)
    models['<nombre_modelo>'] = dt

    # Guardar modelos
    for name, model in models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    
    return models