"""
pre_gen_project.py
------------------
Hook que se ejecuta ANTES de generar el proyecto con cookiecutter.
Valida que la combinación de variables elegidas sea coherente.
Si detecta un error, aborta con un mensaje claro — no se genera nada.
"""
import sys

ML_TYPE    = "{{ cookiecutter.ml_type }}"
NN_MODEL   = "{{ cookiecutter.nn_model }}"
MODEL_TYPE = "{{ cookiecutter.model_type }}"
USE_XGB    = "{{ cookiecutter.use_xgboost }}"
USE_LGBM   = "{{ cookiecutter.use_lightgbm }}"
PY_VER     = "{{ cookiecutter.python_version }}"
EMAIL      = "{{ cookiecutter.project_author_email }}"
SLUG       = "{{ cookiecutter.project_slug }}"


VALID_ML_TYPES    = {"supervisado", "no_supervisado", "redes_neuronales", "hibrido"}
VALID_NN_MODELS   = {"MLP", "CNN1D", "LSTM", "GRU", "Transformer"}
VALID_MODEL_TYPES = {"todos", "RandomForest", "XGBoost", "LightGBM",
                     "LogisticRegression", "KNN", "DecisionTree"}
VALID_PY_VERS     = {"3.10", "3.11", "3.12", "3.13"}


def error(msg: str) -> None:
    """Imprime el error en rojo y sale con código 1."""
    RED   = "\033[91m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"
    print(f"\n{RED}{BOLD}ERROR — dskit pre-generación{RESET}")
    print(f"{RED}{msg}{RESET}\n")
    sys.exit(1)


def warn(msg: str) -> None:
    YELLOW = "\033[93m"
    RESET  = "\033[0m"
    print(f"{YELLOW}⚠  ADVERTENCIA: {msg}{RESET}")


# ── 1. ml_type válido ────────────────────────────────────────────────────────
if ML_TYPE not in VALID_ML_TYPES:
    error(
        f"ml_type='{ML_TYPE}' no es válido.\n"
        f"  Opciones: {sorted(VALID_ML_TYPES)}"
    )

# ── 2. nn_model solo tiene sentido en redes_neuronales ───────────────────────
if ML_TYPE != "redes_neuronales" and NN_MODEL != "MLP":
    # El usuario cambió nn_model pero eligió otro ml_type — adviértele
    warn(
        f"nn_model='{NN_MODEL}' se ignora porque ml_type='{ML_TYPE}'.\n"
        f"  nn_model solo se usa cuando ml_type='redes_neuronales'."
    )

if ML_TYPE == "redes_neuronales" and NN_MODEL not in VALID_NN_MODELS:
    error(
        f"nn_model='{NN_MODEL}' no es válido para redes_neuronales.\n"
        f"  Opciones: {sorted(VALID_NN_MODELS)}"
    )

# ── 3. XGBoost/LightGBM no tienen efecto en redes_neuronales ────────────────
if ML_TYPE == "redes_neuronales":
    if USE_XGB == "si":
        warn(
            "use_xgboost='si' no tiene efecto cuando ml_type='redes_neuronales'.\n"
            "  XGBoost solo se añade en ml_type='supervisado' o 'hibrido'."
        )
    if USE_LGBM == "si":
        warn(
            "use_lightgbm='si' no tiene efecto cuando ml_type='redes_neuronales'.\n"
            "  LightGBM solo se añade en ml_type='supervisado' o 'hibrido'."
        )

# ── 4. XGBoost/LightGBM no tienen efecto en no_supervisado ──────────────────
if ML_TYPE == "no_supervisado":
    if USE_XGB == "si":
        warn(
            "use_xgboost='si' no tiene efecto cuando ml_type='no_supervisado'.\n"
            "  Los modelos de clustering no usan XGBoost."
        )
    if USE_LGBM == "si":
        warn(
            "use_lightgbm='si' no tiene efecto cuando ml_type='no_supervisado'.\n"
            "  Los modelos de clustering no usan LightGBM."
        )

# ── 4b. model_type validación ───────────────────────────────────────────────
if MODEL_TYPE not in VALID_MODEL_TYPES:
    error(
        f"model_type='{MODEL_TYPE}' no es válido.\n"
        f"  Opciones: {sorted(VALID_MODEL_TYPES)}"
    )

if ML_TYPE not in {"supervisado", "hibrido"} and MODEL_TYPE != "todos":
    warn(
        f"model_type='{MODEL_TYPE}' se ignora porque ml_type='{ML_TYPE}'.\n"
        f"  model_type solo tiene efecto en ml_type='supervisado' o 'hibrido'."
    )

if MODEL_TYPE == "XGBoost" and USE_XGB != "si":
    error(
        "model_type='XGBoost' requiere use_xgboost='si'.\n"
        "  Activa use_xgboost='si' o elige otro model_type."
    )
if MODEL_TYPE == "LightGBM" and USE_LGBM != "si":
    error(
        "model_type='LightGBM' requiere use_lightgbm='si'.\n"
        "  Activa use_lightgbm='si' o elige otro model_type."
    )

# ── 5. Python version válida ─────────────────────────────────────────────────
if PY_VER not in VALID_PY_VERS:
    error(
        f"python_version='{PY_VER}' no está soportada.\n"
        f"  Versiones soportadas: {sorted(VALID_PY_VERS)}"
    )

# ── 6. Email con formato mínimo ──────────────────────────────────────────────
if "@" not in EMAIL or "." not in EMAIL.split("@")[-1]:
    error(
        f"project_author_email='{EMAIL}' no parece un email válido.\n"
        f"  Ejemplo: nombre@dominio.com"
    )

# ── 7. project_slug sin caracteres problemáticos ────────────────────────────
import re
if not re.match(r"^[a-z][a-z0-9_]*$", SLUG):
    error(
        f"project_slug='{SLUG}' contiene caracteres no válidos.\n"
        f"  Debe empezar por letra minúscula y solo contener [a-z0-9_].\n"
        f"  Revisa project_name — los espacios y guiones se convierten a '_'."
    )

print(f"\n  ✓  Validación pre-generación OK  (ml_type={ML_TYPE}"
      + (f", nn_model={NN_MODEL}" if ML_TYPE == "redes_neuronales" else "")
      + ")\n")
