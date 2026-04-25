"""
post_gen_project.py
-------------------
Tarea post-generación ejecutada por copier (_tasks en copier.yml).
Se ejecuta desde el directorio del proyecto ya generado.
"""
import subprocess
import sys
import os

ML_TYPE      = "{{ ml_type }}"
PROJECT_NAME = "{{ project_name }}"
PROJECT_SLUG = "{{ project_slug }}"


def run(cmd, description):
    print(f"\n  ▶  {description}")
    print(f"     {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  ✗  Error en: {' '.join(cmd)}")
        print(f"     Ejecútalo manualmente en el directorio del proyecto")
        return False
    return True


def main():
    print()
    print("━" * 60)
    print(f"  Proyecto generado: {PROJECT_NAME}")
    print(f"  Tipo ML          : {ML_TYPE}")
    print("━" * 60)

    uv_check = subprocess.run(["uv", "--version"], capture_output=True)
    if uv_check.returncode != 0:
        print("\n  ⚠  'uv' no encontrado. Instálalo con: pip install uv")
        print(f"     Luego ejecuta: uv sync --extra dev --extra {ML_TYPE}")
        _print_next_steps()
        return

    sync_cmd = ["uv", "sync", "--extra", "dev", "--extra", ML_TYPE]
    ok = run(sync_cmd, f"Instalando dependencias (dev + {ML_TYPE})...")
    if not ok:
        _print_next_steps()
        return

    run(
        ["uv", "run", "python", "-c", "import sklearn; print('  scikit-learn OK')"],
        "Verificando entorno...",
    )

    {% if ml_type == "redes_neuronales" %}
    run(
        ["uv", "run", "python", "-c",
         "import torch; print(f'  torch {torch.__version__} — CUDA: {torch.cuda.is_available()}')"],
        "Verificando PyTorch...",
    )
    {% endif %}
    {% if use_xgboost %}
    run(
        ["uv", "run", "python", "-c",
         "import xgboost; print(f'  xgboost {xgboost.__version__} OK')"],
        "Verificando XGBoost...",
    )
    {% endif %}
    {% if use_lightgbm %}
    run(
        ["uv", "run", "python", "-c",
         "import lightgbm; print(f'  lightgbm {lightgbm.__version__} OK')"],
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
    print("    make run")
    print("    make smoke")
    {% if ml_type == "redes_neuronales" %}
    print("    make tb   # TensorBoard en localhost:6006")
    {% endif %}
    print()


if __name__ == "__main__":
    main()
