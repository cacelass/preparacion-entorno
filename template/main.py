"""
Punto de entrada principal del proyecto.
Ejecutar: python main.py
"""
{% if ml_type == 'supervisado' %}
from {{ project_slug }}.data.make_dataset import load_data
from {{ project_slug }}.features.build_features import preprocess_data
from {{ project_slug }}.models.train_model import train_models
from {{ project_slug }}.models.predict_model import evaluate_models, DECISION_THRESHOLD
{% if task_type == "clasificacion" %}
from {{ project_slug }}.models.predict_model import test_model
{% endif %}
from {{ project_slug }}.visualization.visualize import (
    plot_distributions,
    plot_correlation_matrix,
    plot_class_balance,
    plot_categorical_vs_target,
    plot_feature_importance,
    plot_pca_variance,
)

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
DATA_FILE    = 'dataset.csv'
TARGET_COL   = 'target'
SCALER_TYPE  = 'standard'   # 'standard' | 'minmax'
TEST_SIZE    = 0.2
THRESHOLD    = DECISION_THRESHOLD

# PCA opcional: reducción de dimensionalidad antes del modelado.
# None → sin PCA | 0.95 → conservar 95% varianza | int → nº componentes fijo
USE_PCA      = None   # ← ajusta: None | 0.95 | 10


def run_full_pipeline() -> None:
    print('=' * 60)
    print('1. Cargando datos...')
    df = load_data(DATA_FILE)
    print(f'   Shape: {df.shape}')

    print('\n2. EDA visual...')
    plot_distributions(df, target_col=TARGET_COL)
    plot_correlation_matrix(df)
    plot_class_balance(df, target_col=TARGET_COL)
    plot_categorical_vs_target(df, target_col=TARGET_COL)

    print('\n3. Preprocesando...')
    X_train, X_test, y_train, y_test = preprocess_data(
        df, target_col=TARGET_COL, scaler_type=SCALER_TYPE,
        test_size=TEST_SIZE, use_pca=USE_PCA,
    )

    print('\n4. Entrenando modelos...')
    models = train_models(X_train, y_train, tune_knn=True, cv_evaluate=True)

    print('\n5. Evaluando...')
    df_results = evaluate_models(
        models, X_train, y_train, X_test, y_test, threshold=THRESHOLD
    )

    from {{ project_slug }}.utils.paths import PROCESSED_DATA_DIR
    import pandas as pd
    try:
        feature_names = pd.read_csv(PROCESSED_DATA_DIR / 'X_train.csv').columns.tolist()
    except FileNotFoundError:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

{% if use_shap %}
    print('\n6. SHAP — explicabilidad de modelos...')
    from {{ project_slug }}.models.predict_model import explain_models
    explain_models(models, X_train, feature_names=feature_names)

    print('\n7. Importancia de variables...')
{% else %}
    print('\n6. Importancia de variables...')
{% endif %}
    plot_feature_importance(models, feature_names)

    if USE_PCA is not None:
{% if use_shap %}
        print('\n8. Varianza explicada por PCA...')
{% else %}
        print('\n7. Varianza explicada por PCA...')
{% endif %}
        import joblib
        from {{ project_slug }}.utils.paths import ARTIFACTS_DIR
        try:
            pca = joblib.load(ARTIFACTS_DIR / 'pca.joblib')
            plot_pca_variance(pca)
        except FileNotFoundError:
            pass

    print('\n' + '=' * 60)
    print('Pipeline completado.')
    best = df_results.sort_values('Acc_test', ascending=False).iloc[0]
    print(f'Mejor modelo: {best.to_dict()}')


def main():
    print('=' * 60)
    accion = input('Ejecutar pipeline completo (0) o probar el modelo (1)? (0/1): ').strip()
    if accion == '0':
        run_full_pipeline()
    elif accion == '1':
        test_model()
    else:
        print('Opción no válida. Ejecutando pipeline completo por defecto.')
        run_full_pipeline()


if __name__ == '__main__':
    main()

{% elif ml_type == 'no_supervisado' %}
from {{ project_slug }}.data.make_dataset import load_data
from {{ project_slug }}.features.build_features import preprocess_data
from {{ project_slug }}.models.train_model import train_models, find_optimal_k
from {{ project_slug }}.models.predict_model import evaluate_models, plot_dendrogram, test_model
from {{ project_slug }}.visualization.visualize import (
    plot_distributions,
    plot_correlation_matrix,
    plot_elbow_and_silhouette,
    plot_dendrogram as viz_dendrogram,
    plot_pca_variance,
    plot_umap,
)

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
DATA_FILE  = 'dataset.csv'
N_CLUSTERS = 3   # ajustar tras analizar el codo y el dendrograma


def run_full_pipeline() -> None:
    print('=' * 60)
    print('1. Cargando datos...')
    df = load_data(DATA_FILE)

    print('\n2. EDA visual...')
    plot_distributions(df)
    plot_correlation_matrix(df)

    print('\n3. Preprocesando...')
    X = preprocess_data(df)

    print('\n4. PCA — varianza explicada...')
    plot_pca_variance(X)

    print('\n5. UMAP — proyección 2D no lineal...')
    plot_umap(X)

    print('\n6. Dendrograma (clustering jerarquico)...')
    viz_dendrogram(X, method='ward')
    # Indica el umbral de corte tras ver el dendrograma:
    # viz_dendrogram(X, method='ward', color_threshold=50)

    print('\n7. Metricas para seleccion de k (Elbow + Silhouette + DB)...')
    metrics = find_optimal_k(X, k_range=range(2, 11))
    plot_elbow_and_silhouette(metrics)

    print(f'\n8. Entrenando modelos (k={N_CLUSTERS})...')
    models = train_models(X, n_clusters=N_CLUSTERS)

    print('\n9. Evaluando...')
    evaluate_models(models, X)

    print('\nPipeline completado.')


def main():
    print('=' * 60)
    accion = input('Ejecutar pipeline completo (0) o probar el modelo (1)? (0/1): ').strip()
    if accion == '0':
        run_full_pipeline()
    elif accion == '1':
        test_model()
    else:
        print('Opción no válida. Ejecutando pipeline completo por defecto.')
        run_full_pipeline()


if __name__ == '__main__':
    main()

{% elif ml_type == 'redes_neuronales' %}
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from {{ project_slug }}.data.make_dataset import load_data
from {{ project_slug }}.features.build_features import preprocess_data
from {{ project_slug }}.models.train_model import train_models, MODEL_NAME
from {{ project_slug }}.models.predict_model import evaluate_models, test_model
from {{ project_slug }}.visualization.visualize import (
    plot_distributions,
    plot_correlation_matrix,
    plot_pca_variance,
)
from {{ project_slug }}.utils.paths import RUNS_DIR

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
DATA_FILE   = 'dataset.csv'
TARGET_COL  = 'target'
EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 1e-3
CHECKPOINT  = 10

# PCA opcional antes de la red (útil si hay muchas features correladas)
USE_PCA     = None   # None | 0.95 | int

# Arquitectura seleccionada en copier: {{ nn_model }}
# Cambiar aquí sólo si se modifica manualmente (no recomendado —
# regenerar el proyecto con copier para otro modelo).
_MODEL_INFO = {
    "MLP":         "Perceptrón Multicapa — datos tabulares generales.",
    "CNN1D":       "Red Convolucional 1-D — patrones locales entre features.",
    "LSTM":        "LSTM — dependencias temporales largas.",
    "GRU":         "GRU — como LSTM, más ligero y rápido.",
    "Transformer": "Transformer Encoder — relaciones globales, alta dimensionalidad.",
}


def run_full_pipeline() -> None:
    print('=' * 60)
    print(f'Red neuronal: {MODEL_NAME}')
    print(f'  {_MODEL_INFO.get(MODEL_NAME, "")}')
    print('=' * 60)

    tb = SummaryWriter(log_dir=str(RUNS_DIR))
    print(f'TensorBoard: tensorboard --logdir {RUNS_DIR}')

    print('\n1. Cargando datos...')
    df = load_data(DATA_FILE)
    print(f'   Shape: {df.shape}')

    print('\n2. EDA visual...')
    plot_distributions(df, target_col=TARGET_COL)
    plot_correlation_matrix(df)

    print('\n3. Preprocesando...')
    X_train, X_test, y_train, y_test = preprocess_data(
        df, target_col=TARGET_COL, use_pca=USE_PCA,
    )

    if USE_PCA is not None:
        import joblib
        from {{ project_slug }}.utils.paths import ARTIFACTS_DIR
        try:
            pca = joblib.load(ARTIFACTS_DIR / 'pca.joblib')
            plot_pca_variance(pca)
        except FileNotFoundError:
            pass

    input_dim  = X_train.shape[1]
    output_dim = len(y_train.unique())
    print(f'   input_dim={input_dim}  output_dim={output_dim}')

    print(f'\n4. Entrenando {MODEL_NAME}...')
    models = train_models(
        X_train, y_train,
        input_dim=input_dim, output_dim=output_dim,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        lr=LR, checkpoint_every=CHECKPOINT,
    )

    print('\n5. Evaluando...')
    df_results = evaluate_models(
        models, X_test, y_test, num_classes=output_dim, tb_writer=tb,
    )

    tb.close()
    print('\n' + '=' * 60)
    print('Pipeline completado.')
    if not df_results.empty:
        best = df_results.iloc[0]
        print(f'Resultado: Accuracy={best["Accuracy"]:.4f}  F1={best["F1"]:.4f}')


def main():
    print('=' * 60)
    accion = input('Ejecutar pipeline completo (0) o probar el modelo (1)? (0/1): ').strip()
    if accion == '0':
        run_full_pipeline()
    elif accion == '1':
        test_model()
    else:
        print('Opción no válida. Ejecutando pipeline completo por defecto.')
        run_full_pipeline()


if __name__ == '__main__':
    main()

{% elif ml_type == 'hibrido' %}
"""
Pipeline híbrido: combina técnicas no supervisadas y supervisadas.

Estrategias disponibles (configura STRATEGY):

  'pca_clf'        → PCA → clasificador supervisado
                     Reduce dimensionalidad antes del modelado.
                     Útil cuando hay muchas features correladas.

  'umap_clf'       → UMAP → clasificador supervisado
                     Reducción no lineal; mejor que PCA cuando las
                     relaciones entre variables son complejas.

  'kmeans_features'→ KMeans → distancias a centroides como nuevas
                     features → clasificador supervisado.
                     El clustering actúa como extractor de features.

  'iso_feature'    → IsolationForest → anomaly score como feature extra
                     → clasificador supervisado.
                     Útil cuando la anomalía es predictiva del target.

  'semi_supervisado'→ LabelSpreading sobre datos parcialmente etiquetados
                     → propaga etiquetas → entrena clasificador.
                     Útil cuando solo una parte del dataset tiene labels.
"""
from {{ project_slug }}.data.make_dataset import load_data
from {{ project_slug }}.features.build_features import preprocess_data
from {{ project_slug }}.models.train_model import train_models
from {{ project_slug }}.models.predict_model import evaluate_models, DECISION_THRESHOLD, test_model
from {{ project_slug }}.visualization.visualize import (
    plot_distributions,
    plot_correlation_matrix,
    plot_class_balance,
    plot_pca_variance,
    plot_umap,
    plot_feature_importance,
)

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
DATA_FILE  = 'dataset.csv'
TARGET_COL = 'target'
TEST_SIZE  = 0.2
THRESHOLD  = DECISION_THRESHOLD

# Estrategia híbrida (ver docstring del módulo)
STRATEGY   = 'pca_clf'          # ← ajusta
N_CLUSTERS = 5                   # para kmeans_features
LABELED_FRACTION = 0.3           # para semi_supervisado


def run_full_pipeline() -> None:
    print('=' * 60)
    print(f'Pipeline híbrido — estrategia: {STRATEGY}')
    print('=' * 60)

    print('\n1. Cargando datos...')
    df = load_data(DATA_FILE)
    print(f'   Shape: {df.shape}')

    print('\n2. EDA visual...')
    plot_distributions(df, target_col=TARGET_COL)
    plot_correlation_matrix(df)
    plot_class_balance(df, target_col=TARGET_COL)

    print('\n3. Preprocesando...')
    X_train, X_test, y_train, y_test = preprocess_data(
        df, target_col=TARGET_COL, test_size=TEST_SIZE,
        strategy=STRATEGY,
        n_clusters=N_CLUSTERS,
        labeled_fraction=LABELED_FRACTION,
    )

    print('\n4. Visualizando espacio reducido...')
    if STRATEGY in ('pca_clf',):
        import joblib
        from {{ project_slug }}.utils.paths import ARTIFACTS_DIR
        try:
            pca = joblib.load(ARTIFACTS_DIR / 'pca.joblib')
            plot_pca_variance(pca)
        except FileNotFoundError:
            pass
    if STRATEGY in ('umap_clf',):
        plot_umap(X_train, labels=y_train.values)

    print('\n5. Entrenando modelos...')
    models = train_models(X_train, y_train, strategy=STRATEGY, tune_knn=True)

    print('\n6. Evaluando...')
    df_results = evaluate_models(
        models, X_train, y_train, X_test, y_test, threshold=THRESHOLD
    )

    import pandas as pd
    from {{ project_slug }}.utils.paths import PROCESSED_DATA_DIR
    try:
        feature_names = pd.read_csv(PROCESSED_DATA_DIR / 'X_train.csv').columns.tolist()
    except FileNotFoundError:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

{% if use_shap %}
    print('\n7. SHAP — explicabilidad de modelos...')
    from {{ project_slug }}.models.predict_model import explain_models
    explain_models(models, X_train, feature_names=feature_names)

    print('\n8. Importancia de variables...')
{% else %}
    print('\n7. Importancia de variables...')
{% endif %}
    plot_feature_importance(models, feature_names)

    print('\n' + '=' * 60)
    print('Pipeline completado.')
    best = df_results.sort_values('Acc_test', ascending=False).iloc[0]
    print(f'Mejor modelo: {best.to_dict()}')


def main():
    print('=' * 60)
    accion = input('Ejecutar pipeline completo (0) o probar el modelo (1)? (0/1): ').strip()
    if accion == '0':
        run_full_pipeline()
    elif accion == '1':
        test_model()
    else:
        print('Opción no válida. Ejecutando pipeline completo por defecto.')
        run_full_pipeline()
    best = df_results.sort_values('Acc_test', ascending=False).iloc[0]
    print(f'Mejor modelo: {best.to_dict()}')


if __name__ == '__main__':
    main()
{% endif %}