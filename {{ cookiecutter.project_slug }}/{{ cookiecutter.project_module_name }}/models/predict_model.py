from sklearn.metrics import classification_report, confusion_matrix

def evaluate_models(models, X_test, y_test):
    """
    Evalúa los modelos entrenados y muestra métricas.
    """
    print("--> Evaluando modelos...")
    
    for name, model in models.items():
        print(f"\n{'='*10} Reporte para: {name} {'='*10}")
        predictions = model.predict(X_test)
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))