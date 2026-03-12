"""
Punto de entrada principal del proyecto.
Ejecutar: python main.py
"""
{% if cookiecutter.ml_type == "supervisado" %}
from {{ cookiecutter.project_module_name }}.data.make_dataset import load_data
from {{ cookiecutter.project_module_name }}.features.build_features import preprocess_data
from {{ cookiecutter.project_module_name }}.models.train_model import train_models
from {{ cookiecutter.project_module_name }}.models.predict_model import evaluate_models, DECISION_THRESHOLD
from {{ cookiecutter.project_module_name }}.visualization.visualize import (
    plot_distributions,
    plot_correlation_matrix,
    plot_class_balance,
    plot_categorical_vs_target,
    plot_feature_importance,
)

# ---------------------------------------------------------------------------
# ⚙ Configuración — editar según el problema
# ---------------------------------------------------------------------------
DATA_FILE    = "data/raw/dataset.csv"   # ruta al fichero de datos
TARGET_COL   = "target"                 # columna objetivo
SCALER_TYPE  = "standard"               # "standard" | "minmax"
TEST_SIZE    = 0.2                      # fracción de datos para test
THRESHOLD    = DECISION_THRESHOLD       # umbral de probabilidad (0.5 por defecto)

# ---------------------------------------------------------------------------

def main():
    # 1. Carga
    print("=" * 60)
    print("1. Cargando datos...")
    df = load_data(DATA_FILE)
    print(f"   Shape: {df.shape}")
    print(f"   Nulos:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # 2. Análisis visual exploratorio
    print("\n2. EDA visual...")
    plot_distributions(df, target_col=TARGET_COL)
    plot_correlation_matrix(df)
    plot_class_balance(df, target_col=TARGET_COL)
    plot_categorical_vs_target(df, target_col=TARGET_COL)

    # 3. Preprocesado
    print("\n3. Preprocesando...")
    X_train, X_test, y_train, y_test = preprocess_data(
        df,
        target_col=TARGET_COL,
        scaler_type=SCALER_TYPE,
        test_size=TEST_SIZE,
    )

    # 4. Entrenamiento
    print("\n4. Entrenando modelos...")
    models = train_models(X_train, y_train, tune_knn=True, cv_evaluate=True)

    # 5. Evaluación
    print("\n5. Evaluando...")
    df_results = evaluate_models(
        models, X_train, y_train, X_test, y_test, threshold=THRESHOLD
    )

    # 6. Importancia de variables
    print("\n6. Importancia de variables...")
    # Recuperar nombres de columnas desde el CSV procesado si están disponibles
    from {{ cookiecutter.project_module_name }}.utils.paths import PROCESSED_DATA_DIR
    import pandas as pd
    try:
        feature_names = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv").columns.tolist()
    except FileNotFoundError:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    plot_feature_importance(models, feature_names)

    print("\n" + "=" * 60)
    print("Pipeline completado.")
    print(f"Mejor modelo por Acc_test:\n{df_results.sort_values('Acc_test', ascending=False).iloc[0].to_dict()}")
    print("Figuras generadas en reports/figures/")

if __name__ == "__main__":
    main()

{% elif cookiecutter.ml_type == "no_supervisado" %}
from {{ cookiecutter.project_module_name }}.data.make_dataset import load_data
from {{ cookiecutter.project_module_name }}.features.build_features import preprocess_data
from {{ cookiecutter.project_module_name }}.models.train_model import train_models
from {{ cookiecutter.project_module_name }}.models.predict_model import evaluate_models
from {{ cookiecutter.project_module_name }}.visualization.visualize import (
    plot_distributions,
    plot_correlation_matrix,
    plot_elbow,
)

DATA_FILE  = "data/raw/dataset.csv"
N_CLUSTERS = 3


def main():
    print("1. Cargando datos...")
    df = load_data(DATA_FILE)

    print("2. EDA visual...")
    plot_distributions(df)
    plot_correlation_matrix(df)

    print("3. Preprocesando...")
    X = preprocess_data(df)

    print("4. Método del codo (selección de k)...")
    plot_elbow(X, max_k=10)

    print("5. Entrenando modelos...")
    models = train_models(X, n_clusters=N_CLUSTERS)

    print("6. Evaluando...")
    evaluate_models(models, X)

    print("Pipeline completado.")

if __name__ == "__main__":
    main()

{% elif cookiecutter.ml_type == "redes_neuronales" %}
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from {{ cookiecutter.project_module_name }}.data.make_dataset import load_data
from {{ cookiecutter.project_module_name }}.features.build_features import preprocess_data
from {{ cookiecutter.project_module_name }}.models.train_model import train_models
from {{ cookiecutter.project_module_name }}.models.predict_model import evaluate_models
from {{ cookiecutter.project_module_name }}.visualization.visualize import (
    plot_distributions,
    plot_correlation_matrix,
)
from {{ cookiecutter.project_module_name }}.utils.paths import RUNS_DIR

DATA_FILE   = "data/raw/dataset.csv"
TARGET_COL  = "target"
EPOCHS      = 50
BATCH_SIZE  = 32
CHECKPOINT  = 10


def main():
    tb = SummaryWriter(log_dir=str(RUNS_DIR))
    print(f"TensorBoard: tensorboard --logdir {RUNS_DIR}")

    df = load_data(DATA_FILE)

    plot_distributions(df, target_col=TARGET_COL)
    plot_correlation_matrix(df)

    X_train, X_test, y_train, y_test = preprocess_data(df, target_col=TARGET_COL)
    input_dim  = X_train.shape[1]
    output_dim = len(y_train.unique())

    models = train_models(
        X_train, y_train,
        input_dim=input_dim,
        output_dim=output_dim,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        checkpoint_every=CHECKPOINT,
    )

    evaluate_models(models, X_test, y_test, num_classes=output_dim, tb_writer=tb)

    tb.close()
    print("Pipeline completado.")

if __name__ == "__main__":
    main()
{% endif %}
