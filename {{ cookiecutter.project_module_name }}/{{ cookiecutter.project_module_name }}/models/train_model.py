{% if cookiecutter.ml_type == "supervisado" %}
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

from {{ cookiecutter.project_module_name }}.utils.paths import MODELS_DIR


# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------

def _build_models() -> dict:
    """
    Define los modelos a entrenar.

    KNN            → lazy learner, sin suposiciones sobre los datos.
                     Requiere features escaladas. Sensible a k y a dimensiones altas.

    LogisticReg    → modelo base en clasificación binaria. Rápido, interpretable
                     y genera probabilidades calibradas.

    DecisionTree   → caja blanca, fácil de interpretar. Propenso a overfitting
                     → regularizar con max_depth, min_samples_leaf.

    RandomForest   → ensemble de árboles. Robusto y buen rendimiento por defecto.
                     Permite calcular importancia de variables (feature_importances_).

    GradBoost      → mayor precisión que RF en muchos casos, pero más lento
                     y más sensible a hiperparámetros.

    SVM (RBF)      → potente en espacios de alta dimensión. Lento en datasets grandes.
                     El pipeline incluye StandardScaler propio.

    Pipelines con PCA integrado:
      PCA_LogReg   → PCA(95%) → LogReg. Útil con muchas features correladas.
      PCA_SVM      → PCA(95%) → SVM RBF. Acelera SVM enormemente en alta dim.

    ¿Cuándo incluir los pipelines PCA?
      - Muchas features muy correladas entre sí (|r| > 0.8)
      - Alta dimensionalidad (>50 features)
      - SVM o KNN muy lentos sin reducción previa
      - Quieres comparar directamente si PCA mejora o no el rendimiento
    """
    return {
        # --- KNN: el valor de n_neighbors se optimiza automáticamente en train_models() ---
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),

        # --- Regresión Logística: modelo base rápido e interpretable ---
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),

        # --- Árbol de Decisión: ajustar max_depth para evitar overfitting ---
        "DecisionTree": DecisionTreeClassifier(
            max_depth=7,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),

        # --- Random Forest: robusto, con importancia de variables ---
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            max_features="sqrt",   # sqrt(n_features) por árbol
            max_samples=0.8,       # bootstrap sample del 80%
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        # --- Gradient Boosting: alta precisión, mayor coste computacional ---
        # "GradientBoosting": GradientBoostingClassifier(
        #     n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
        # ),

        # --- SVM RBF: muy eficaz en dimensiones altas ---
        # El pipeline escala internamente; no necesita X ya escalado.
        # "SVM": Pipeline([
        #     ("scaler", StandardScaler()),
        #     ("clf", SVC(
        #         kernel="rbf", C=1.0, gamma="scale",
        #         class_weight="balanced", probability=True, random_state=42,
        #     )),
        # ]),

        # --- Pipeline PCA + Regresión Logística ---
        # "PCA_LogReg": Pipeline([
        #     ("pca", PCA(n_components=0.95, random_state=42)),
        #     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        # ]),

        # --- Pipeline PCA + SVM RBF ---
        # SVM es O(n²~n³); PCA lo acelera mucho. probability=True para ROC-AUC.
        # "PCA_SVM": Pipeline([
        #     ("pca", PCA(n_components=0.95, random_state=42)),
        #     ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
        #                 class_weight="balanced", probability=True, random_state=42)),
        # ]),
    }


def _find_best_k(X_train, y_train, k_range=range(1, 21)) -> int:
    """
    Busca el mejor k para KNN por validación cruzada (5-fold, métrica F1_weighted).
    Devuelve el k con mayor F1 medio, priorizando k más alto en empates.
    """
    print("    Buscando mejor k para KNN...")
    best_k, best_score = 1, 0.0
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        score = cross_val_score(knn, X_train, y_train, cv=5, scoring="f1_weighted").mean()
        if score >= best_score:   # >= → preferimos k más alto en empates
            best_k, best_score = k, score
    print(f"    Mejor k = {best_k}  (F1_weighted CV = {best_score:.3f})")
    return best_k


def train_models(
    X_train,
    y_train,
    tune_knn: bool = True,
    cv_evaluate: bool = True,
) -> dict:
    """
    Entrena todos los modelos definidos en _build_models() y los guarda en models/.

    Parameters
    ----------
    X_train      : features de entrenamiento (array-like)
    y_train      : etiquetas de entrenamiento (array-like)
    tune_knn     : si True, optimiza k de KNN por cross-validation antes de entrenar.
    cv_evaluate  : si True, muestra F1_weighted (5-fold CV) de cada modelo.

    Returns
    -------
    dict : {nombre_modelo: modelo_entrenado}
    """
    print("--> Entrenando modelos supervisados...")
    models = _build_models()

    # Optimización automática de k
    if tune_knn and "KNN" in models:
        best_k = _find_best_k(X_train, y_train)
        models["KNN"] = KNeighborsClassifier(n_neighbors=best_k, weights="distance")

    trained = {}
    for name, model in models.items():
        print(f"    [{name}] entrenando...")
        model.fit(X_train, y_train)

        if cv_evaluate:
            cv_score = cross_val_score(
                model, X_train, y_train, cv=5, scoring="f1_weighted"
            ).mean()
            print(f"      F1_weighted 5-fold CV: {cv_score:.3f}")

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"      Guardado → {name}.joblib")
        trained[name] = model

    print(f"--> {len(trained)} modelos guardados en {MODELS_DIR}")
    return trained


def load_models(model_names: list = None) -> dict:
    """
    Carga modelos desde disco.

    Parameters
    ----------
    model_names : lista de nombres sin extensión, e.g. ["RandomForest", "KNN"].
                  Si None, carga todos los .joblib disponibles en models/.

    Returns
    -------
    dict : {nombre_modelo: modelo_cargado}
    """
    if model_names is None:
        model_names = [p.stem for p in MODELS_DIR.glob("*.joblib")]

    models = {}
    for name in model_names:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
            print(f"    Cargado: {name}")
        else:
            print(f"    ⚠ No encontrado: {path}")
    return models


{% elif cookiecutter.ml_type == "no_supervisado" %}
import joblib
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from {{ cookiecutter.project_module_name }}.utils.paths import MODELS_DIR


# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------

def _build_models(n_clusters: int = 3) -> dict:
    """
    Define los modelos de clustering a ajustar.

    KMeans            → rápido y escalable. Asume clusters esféricos.
                        Inicialización k-means++ reduce el riesgo de mínimos locales.

    AgglomerativeClustering → clustering jerárquico aglomerativo (bottom-up).
                               No requiere reinicializaciones. Permite usar un dendrograma
                               para elegir k antes de ajustar.
                               linkage: 'ward' (minimiza varianza intraclúster, mejor general),
                               'complete', 'average', 'single'.

    MiniBatchKMeans   → versión acelerada de KMeans para datasets grandes.
                        Usa mini-lotes; ligeramente peor calidad, mucho más rápido.

    DBSCAN            → basado en densidad; detecta clusters de cualquier forma
                        y es robusto a outliers. No necesita especificar k,
                        pero requiere ajustar eps y min_samples.
    """
    return {
        "KMeans": KMeans(
            n_clusters=n_clusters,
            init="k-means++",   # mejor inicialización que random
            n_init=10,
            max_iter=300,
            random_state=42,
        ),

        "AgglomerativeClustering": AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",     # 'ward' | 'complete' | 'average' | 'single'
        ),

        # "MiniBatchKMeans": MiniBatchKMeans(
        #     n_clusters=n_clusters, n_init=10, random_state=42
        # ),

        # "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    }


def find_optimal_k(X, k_range=range(2, 11)) -> dict:
    """
    Calcula el método del codo (inercia), el Silhouette Score,
    el Davies-Bouldin Score y el Calinski-Harabasz Score para cada k.

    Devuelve un diccionario con:
      - 'k_range'    : lista de k probados
      - 'inertias'   : inercia (WCSS) por k — buscar el codo
      - 'silhouettes': Silhouette Score por k — mayor es mejor (+1 máximo)
      - 'db_scores'  : Davies-Bouldin por k   — menor es mejor (0 mínimo)
      - 'ch_scores'  : Calinski-Harabasz por k — mayor es mejor

    Uso típico:
      metrics = find_optimal_k(X)
      plot_elbow_and_silhouette(metrics)   # en visualize.py
    """
    print("--> Calculando métricas para selección de k...")
    inertias, silhouettes, db_scores, ch_scores = [], [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(X)
        labels = km.labels_

        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels, metric="euclidean")
        db  = davies_bouldin_score(X, labels)
        ch  = calinski_harabasz_score(X, labels)

        silhouettes.append(sil)
        db_scores.append(db)
        ch_scores.append(ch)

        print(f"    k={k}  inercia={km.inertia_:.1f}  silhouette={sil:.3f}  "
              f"davies-bouldin={db:.3f}  calinski-harabasz={ch:.1f}")

    return {
        "k_range":     list(k_range),
        "inertias":    inertias,
        "silhouettes": silhouettes,
        "db_scores":   db_scores,
        "ch_scores":   ch_scores,
    }


def train_models(X, n_clusters: int = 3) -> dict:
    """
    Ajusta todos los modelos definidos en _build_models() y los guarda en models/.

    ⚠ AgglomerativeClustering no tiene método .predict() — usa .labels_ para
    asignar clusters a los datos de entrenamiento.

    Parameters
    ----------
    X          : datos de entrenamiento (array-like)
    n_clusters : número de clusters (ajústalo tras analizar el codo y silhouette)

    Returns
    -------
    dict : {nombre_modelo: modelo_ajustado}
    """
    print(f"--> Ajustando modelos de clustering (k={n_clusters})...")
    models = _build_models(n_clusters)
    fitted = {}

    for name, model in models.items():
        print(f"    [{name}] ajustando...")
        model.fit(X)

        # Silhouette Score (no aplicable a DBSCAN con un solo cluster)
        labels   = model.labels_ if hasattr(model, "labels_") else model.predict(X)
        n_unique = len(set(labels)) - (1 if -1 in labels else 0)
        if n_unique > 1:
            sil = silhouette_score(X, labels)
            db  = davies_bouldin_score(X, labels)
            print(f"      Silhouette Score  : {sil:.3f}  (mejor → +1)")
            print(f"      Davies-Bouldin    : {db:.3f}   (mejor → 0)")

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"      Guardado → {name}.joblib")
        fitted[name] = model

    return fitted


def train_kmeans_pipeline(X_train, y_train, n_clusters: int = 50):
    """
    Pipeline KMeans → LogisticRegression.
    Usa el clustering como reducción de dimensionalidad antes de un clasificador.

    Útil cuando se dispone de etiquetas (semisupervisado):
      la distancia a cada centroide se usa como features para el clasificador.

    Parameters
    ----------
    n_clusters : número de centroides KMeans (actúan como nuevas features)

    Returns
    -------
    pipeline entrenado
    """
    print(f"--> Entrenando pipeline KMeans({n_clusters}) + LogisticRegression...")
    pipeline = Pipeline([
        ("kmeans",  KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)),
        ("log_reg", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODELS_DIR / "KMeansPipeline.joblib")
    print("    Guardado → KMeansPipeline.joblib")
    return pipeline


def load_models(model_names: list = None) -> dict:
    """
    Carga modelos desde disco.

    Parameters
    ----------
    model_names : lista de nombres sin extensión.
                  Si None, carga todos los .joblib disponibles en models/.

    Returns
    -------
    dict : {nombre_modelo: modelo_cargado}
    """
    if model_names is None:
        model_names = [p.stem for p in MODELS_DIR.glob("*.joblib")]

    models = {}
    for name in model_names:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
            print(f"    Cargado: {name}")
        else:
            print(f"    ⚠ No encontrado: {path}")
    return models


{% elif cookiecutter.ml_type == "redes_neuronales" %}
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from {{ cookiecutter.project_module_name }}.utils.paths import MODELS_DIR, RUNS_DIR


# ---------------------------------------------------------------------------
# Detección de dispositivo (CPU / CUDA)
# ---------------------------------------------------------------------------
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class MLP(nn.Module):
    """
    Red neuronal densa (MLP) con capas ocultas configurables.

    Parameters
    ----------
    input_dim   : número de features de entrada
    output_dim  : número de clases (clasificación) o 1 (regresión)
    hidden_dims : lista con el tamaño de cada capa oculta, e.g. [128, 64]
    dropout     : tasa de dropout aplicada tras cada capa oculta (regularización)
    """
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_models(
    X_train,
    y_train,
    input_dim,
    output_dim,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    checkpoint_every: int = 10,
) -> dict:
    """
    Entrena una MLP con PyTorch.

    Características:
    - CUDA automático si está disponible
    - TensorBoard: loss por época en runs/
    - Checkpoints periódicos en models/checkpoint-{epoch}.pt
    - Guardado final de pesos en models/MLP.pt

    Parameters
    ----------
    input_dim        : número de features de entrada
    output_dim       : número de clases de salida
    epochs           : épocas de entrenamiento
    batch_size       : tamaño de mini-lote
    lr               : learning rate para Adam
    checkpoint_every : cada cuántas épocas guardar checkpoint (0 = desactivado)

    Returns
    -------
    dict : {'MLP': modelo_entrenado}
    """
    print("--> Entrenando red neuronal...")

    X_t    = torch.tensor(X_train.values, dtype=torch.float32)
    y_t    = torch.tensor(y_train.values, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model     = MLP(input_dim=input_dim, output_dim=output_dim).to(device)
    # model   = torch.compile(model)   # PyTorch ≥ 2.0: descomentar para mayor velocidad

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # cambiar a MSELoss para regresión
    tb        = SummaryWriter(log_dir=str(RUNS_DIR))
    print(f"    TensorBoard → tensorboard --logdir {RUNS_DIR}")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        tb.add_scalar("Loss/train", avg_loss, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss":                 avg_loss,
            }, MODELS_DIR / f"checkpoint-{epoch+1}.pt")

    tb.close()
    torch.save(model.state_dict(), MODELS_DIR / "MLP.pt")
    print("    Guardado: MLP.pt")
    return {"MLP": model}


def load_model(input_dim, output_dim, weights_path="MLP.pt"):
    """Carga pesos finales y devuelve el modelo en modo eval."""
    path  = MODELS_DIR / weights_path if not str(weights_path).startswith("/") else weights_path
    model = MLP(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"    Modelo cargado desde {path}")
    return model


def load_checkpoint(input_dim, output_dim, checkpoint_path):
    """Carga un checkpoint para continuar el entrenamiento."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model      = MLP(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer  = torch.optim.Adam(model.parameters())
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_inicio = checkpoint["epoch"]
    print(f"    Checkpoint cargado: epoch {epoch_inicio}")
    return model, optimizer, epoch_inicio


{% elif cookiecutter.ml_type == "hibrido" %}
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from {{ cookiecutter.project_module_name }}.utils.paths import MODELS_DIR


# ---------------------------------------------------------------------------
# Configuración de modelos
# ---------------------------------------------------------------------------

def _build_models(strategy: str) -> dict:
    """
    Clasificadores adaptados a la estrategia híbrida elegida.

    'pca_clf' / 'umap_clf'
        Espacio reducido (pocas dimensiones). SVM y LogReg son muy competitivos.
        RandomForest pierde algo de ventaja cuando hay pocas features.

    'kmeans_features' / 'iso_feature'
        Espacio original + features extra. GradientBoosting aprovecha bien
        las distancias a centroides o el anomaly score.

    'semi_supervisado'
        Espacio original con etiquetas propagadas. Todos los modelos válidos.
    """
    base = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            max_features="sqrt",
            max_samples=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    }

    if strategy in ("pca_clf", "umap_clf"):
        # SVM con escalado propio: muy eficaz en espacios reducidos
        base["SVM_RBF"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", probability=True, random_state=42,
            )),
        ])

    if strategy in ("kmeans_features", "iso_feature"):
        # GradBoost aprovecha bien distancias a centroides o anomaly scores
        base["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    return base


def _find_best_k(X_train, y_train, k_range=range(1, 21)) -> int:
    """
    Busca el mejor k para KNN por validación cruzada (5-fold, métrica F1_weighted).
    Devuelve el k con mayor F1 medio, priorizando k más alto en empates.
    """
    print("    Buscando mejor k para KNN...")
    best_k, best_score = 1, 0.0
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        score = cross_val_score(knn, X_train, y_train, cv=5, scoring="f1_weighted").mean()
        if score >= best_score:   # >= → preferimos k más alto en empates
            best_k, best_score = k, score
    print(f"    Mejor k = {best_k}  (F1_weighted CV = {best_score:.3f})")
    return best_k


def train_models(
    X_train,
    y_train,
    strategy: str = "pca_clf",
    tune_knn: bool = True,
    cv_evaluate: bool = True,
) -> dict:
    """
    Entrena clasificadores sobre el espacio transformado por la estrategia híbrida.

    Parameters
    ----------
    X_train      : features de entrenamiento ya transformadas (array-like)
    y_train      : etiquetas de entrenamiento (array-like)
    strategy     : estrategia híbrida usada para generar X_train.
                   Valores: 'pca_clf' | 'umap_clf' | 'kmeans_features' |
                             'iso_feature' | 'semi_supervisado'
    tune_knn     : si True, optimiza k de KNN por cross-validation antes de entrenar.
    cv_evaluate  : si True, muestra F1_weighted (5-fold CV) de cada modelo.

    Returns
    -------
    dict : {nombre_modelo: modelo_entrenado}
    """
    print(f"--> Entrenando modelos híbridos (estrategia='{strategy}')...")
    models = _build_models(strategy)

    if tune_knn and "KNN" in models:
        best_k = _find_best_k(X_train, y_train)
        models["KNN"] = KNeighborsClassifier(n_neighbors=best_k, weights="distance")

    trained = {}
    for name, model in models.items():
        print(f"    [{name}] entrenando...")
        model.fit(X_train, y_train)

        if cv_evaluate:
            cv_score = cross_val_score(
                model, X_train, y_train, cv=5, scoring="f1_weighted"
            ).mean()
            print(f"      F1_weighted 5-fold CV: {cv_score:.3f}")

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"      Guardado → {name}.joblib")
        trained[name] = model

    print(f"--> {len(trained)} modelos guardados en {MODELS_DIR}")
    return trained


def load_models(model_names: list = None) -> dict:
    """
    Carga modelos desde disco.

    Parameters
    ----------
    model_names : lista de nombres sin extensión, e.g. ["RandomForest", "KNN"].
                  Si None, carga todos los .joblib disponibles en models/.

    Returns
    -------
    dict : {nombre_modelo: modelo_cargado}
    """
    if model_names is None:
        model_names = [p.stem for p in MODELS_DIR.glob("*.joblib")]

    models = {}
    for name in model_names:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
            print(f"    Cargado: {name}")
        else:
            print(f"    ⚠ No encontrado: {path}")
    return models
{% endif %}