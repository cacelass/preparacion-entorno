#!/usr/bin/env python3
"""
Ejecuta el CI del proyecto generado, sobre un clon limpio.

Por qué existe
--------------
`validate_template.py` demuestra que la plantilla *renderiza*. `smoke` demuestra
que el proyecto generado *arranca*. Nadie demostraba que el
`.github/workflows/ci.yml` que le entregamos al usuario **pase**, así que podía
estar roto durante versiones enteras sin que nada se pusiera rojo. Ocurrió: el
workflow hacía `uv sync --frozen` mientras el `.gitignore` ignoraba `uv.lock`,
de modo que el primer push de cualquier proyecto generado fallaba con
«Unable to find lockfile».

Cómo lo comprueba
-----------------
1. Toma el proyecto ya generado por copier.
2. Hace `git init && git add -A && git commit` — que respeta su `.gitignore`.
3. Clona ese repositorio a un directorio nuevo.

El paso 3 es el que importa: el clon contiene **exactamente** lo que ve GitHub
Actions al hacer checkout. Si un fichero que el CI necesita está gitignorado, se
cae aquí y no en el repositorio de un usuario.

4. Lee el workflow generado y ejecuta sus pasos `run:` en orden, en ese clon.

Se leen del fichero de verdad en vez de replicarlos aquí: una copia se
desincroniza, y entonces estaríamos probando nuestra idea del CI en lugar del
CI. Los pasos `uses:` (checkout, setup-uv, codecov) se saltan — son acciones de
GitHub, no shell.

Uso
---
    python .github/scripts/run_generated_ci.py <dir-del-proyecto-generado>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

WORKFLOW = Path(".github/workflows/ci.yml")


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def clonar_limpio(proyecto: Path, destino: Path) -> Path:
    """
    Commitea el proyecto y lo clona. Lo que sobrevive al clon es lo que tendrá
    CI: si el `.gitignore` se traga algo necesario, aquí desaparece.
    """
    for cmd in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "ci@dskit.local"],
        ["git", "config", "user.name", "dskit CI"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "chore: proyecto generado"],
    ):
        res = _run(cmd, proyecto)
        if res.returncode != 0:
            print(f"✗ falló {' '.join(cmd)}\n{res.stderr}", file=sys.stderr)
            sys.exit(1)

    clon = destino / "clon"
    res = _run(["git", "clone", "-q", str(proyecto), str(clon)], destino)
    if res.returncode != 0:
        print(f"✗ falló el clon\n{res.stderr}", file=sys.stderr)
        sys.exit(1)
    return clon


def pasos_ejecutables(workflow: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """(nombre, comando, tolera_fallo) de cada paso `run:` del workflow."""
    pasos = []
    for nombre_job, job in (workflow.get("jobs") or {}).items():
        for i, paso in enumerate(job.get("steps") or []):
            comando = paso.get("run")
            if not comando:
                continue  # `uses:` — es una acción de GitHub, no shell
            etiqueta = paso.get("name") or f"{nombre_job}[{i}]"
            pasos.append((etiqueta, comando, bool(paso.get("continue-on-error"))))
    return pasos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proyecto", type=Path, help="Directorio del proyecto generado")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Ejecuta todos los pasos aunque alguno falle (para ver el cuadro completo)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Dónde clonar. Por defecto, junto al proyecto: el clon instala un "
        "entorno completo y /tmp suele quedarse corto en máquinas locales.",
    )
    args = parser.parse_args()

    proyecto = args.proyecto.resolve()
    if not (proyecto / WORKFLOW).exists():
        print(f"✗ no hay {WORKFLOW} en {proyecto}", file=sys.stderr)
        return 1

    base = str(args.workdir) if args.workdir else str(proyecto.parent)
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        clon = clonar_limpio(proyecto, Path(tmp))

        if not (clon / "uv.lock").exists():
            print(
                "✗ el clon no tiene uv.lock — el workflow usa `uv sync --frozen` y "
                "fallará con «Unable to find lockfile». Está en el .gitignore?",
                file=sys.stderr,
            )
            return 1

        workflow = yaml.safe_load((clon / WORKFLOW).read_text(encoding="utf-8"))
        pasos = pasos_ejecutables(workflow)
        print(f"▶  {len(pasos)} paso(s) ejecutables en {WORKFLOW}\n")

        fallos = []
        for etiqueta, comando, tolera in pasos:
            print(f"── {etiqueta}")
            res = subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", comando], cwd=clon, text=True
            )
            if res.returncode == 0:
                print(f"   ✓ {etiqueta}\n")
                continue
            if tolera:
                print(f"   ~ {etiqueta} falló, pero es continue-on-error\n")
                continue
            print(f"   ✗ {etiqueta} (exit {res.returncode})\n")
            fallos.append(etiqueta)
            if not args.keep_going:
                break

        if fallos:
            print(f"✗ CI del proyecto generado en rojo: {', '.join(fallos)}")
            return 1
        print("✓ El CI del proyecto generado pasa sobre un clon limpio")
        return 0


if __name__ == "__main__":
    sys.exit(main())
