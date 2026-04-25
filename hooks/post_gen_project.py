"""
post_gen_project.py
-------------------
Hook que se ejecuta automáticamente tras generar el proyecto con cookiecutter.

Hace:
  1. Crea el entorno virtual con `uv sync --extra dev --extra <ml_type>`
  2. Imprime instrucciones claras para activar el entorno y arrancar
"""
import subprocess
import sys
from pathlib import Path

ML_TYPE       = "{{ cookiecutter.ml_type }}"
PROJECT_NAME  = "{{ cookiecutter.project_name }}"
PROJECT_SLUG  = "{{ cookiecutter.project_slug }}"

# Extras adicionales según flags opcionales
EXTRA_FLAGS = []
{% if cookiecutter.use_xgboost == "si" %}
# xgboost ya está incluido en el extra del ml_type, no hace falta extra separado
{% endif %}
{% if cookiecutter.use_lightgbm == "si" %}
# lightgbm ya está incluido en el extra del ml_type, no hace falta extra separado
{% endif %}


def run(cmd: list[str], description: str) -> bool:
    """Ejecuta un comando y devuelve True si tuvo éxito."""
    print(f"\n  ▶  {description}")
    print(f"     {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n  ✗  Error en: {' '.join(cmd)}")
        print(f"     Puedes ejecutarlo manualmente dentro de '{PROJECT_SLUG}/'")
        return False
    return True


def main():
    print()
    print("━" * 60)
    print(f"  Proyecto generado: {PROJECT_NAME}")
    print(f"  Tipo ML          : {ML_TYPE}")
    print("━" * 60)

    # ── 1. Verificar que uv está disponible ──────────────────────────────
    uv_check = subprocess.run(["uv", "--version"], capture_output=True)
    if uv_check.returncode != 0:
        print("\n  ⚠  'uv' no encontrado en el PATH.")
        print("     Instálalo con: pip install uv  o  curl -Ls https://astral.sh/uv | sh")
        print("     Luego ejecuta manualmente desde el directorio del proyecto:")
        print(f"       uv sync --extra dev --extra {ML_TYPE}")
        print(f"       source .venv/bin/activate")
        _print_next_steps()
        return

    # ── 2. Instalar dependencias ─────────────────────────────────────────
    sync_cmd = ["uv", "sync", "--extra", "dev", "--extra", ML_TYPE]
    for flag in EXTRA_FLAGS:
        sync_cmd += ["--extra", flag]

    ok = run(sync_cmd, f"Instalando dependencias (dev + {ML_TYPE})...")
    if not ok:
        _print_next_steps()
        return

    # ── 3. Verificar instalación ─────────────────────────────────────────
    run(
        ["uv", "run", "python", "-c", "import sklearn; print('  scikit-learn OK')"],
        "Verificando entorno...",
    )

    {% if cookiecutter.ml_type == "redes_neuronales" %}
    run(
        ["uv", "run", "python", "-c", "import torch; print(f'  torch {torch.__version__} OK — CUDA: {torch.cuda.is_available()}')"],
        "Verificando PyTorch...",
    )
    {% endif %}
    {% if cookiecutter.use_xgboost == "si" %}
    run(
        ["uv", "run", "python", "-c", "import xgboost; print(f'  xgboost {xgboost.__version__} OK')"],
        "Verificando XGBoost...",
    )
    {% endif %}
    {% if cookiecutter.use_lightgbm == "si" %}
    run(
        ["uv", "run", "python", "-c", "import lightgbm; print(f'  lightgbm {lightgbm.__version__} OK')"],
        "Verificando LightGBM...",
    )
    {% endif %}

    print()
    print("━" * 60)
    print("  ✓  Entorno listo")
    print("━" * 60)
    _print_next_steps()


def _print_next_steps():
    print()
    print("  Próximos pasos:")
    print(f"    cd {PROJECT_SLUG}")
    print("    source .venv/bin/activate")
    print("    make run          # ejecuta main.py")
    print("    make test         # pytest completo")
    print("    make smoke        # test de humo rápido")
    {% if cookiecutter.ml_type == "redes_neuronales" %}
    print("    make tb           # TensorBoard en localhost:6006")
    {% endif %}
    print()


if __name__ == "__main__":
    main()
