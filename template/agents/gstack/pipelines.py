"""
gstack.pipelines — Pipelines predefinidos para flujos autónomos comunes.

Cada pipeline es una función que construye y ejecuta una GStack.

Los commits automáticos entre pasos están **desautorizados por defecto**: sin
`confirm=True` (o `--yes` en la CLI) el pipeline hace su trabajo y deja los
cambios en el árbol para que los revises, anotando en el log de eventos cada
commit que se saltó. Escribir en el historial de git es una decisión del
humano, no un efecto secundario de un pipeline.

Puedes ejecutarlos desde CLI:
    python -m agents pipeline develop
    python -m agents pipeline develop --yes      # autoriza los auto-commits
    python -m agents pipeline release --version 1.2.0
"""

from __future__ import annotations

from agents.gstack.stack import GStack, StackResult


def auto_analyze(*, auto_commit: bool = True, confirm: bool = False) -> StackResult:
    """
    Analiza el proyecto actual sin modificarlo: revisa código, datos,
    dependencias y entorno. Ideal como diagnóstico rápido.
    """
    stack = GStack(auto_commit=auto_commit, commit_on_error=False, confirm=confirm)
    stack.push("env", "info")
    stack.push("env", "check_python_version")
    stack.push("git", "analyze_diff")
    stack.push("review", "review_package", path=".", result_key="review")
    stack.push("env", "check_lock_sync")
    return stack.run()


def auto_develop(*, auto_commit: bool = True, confirm: bool = False) -> StackResult:
    """
    Pipeline de desarrollo autónomo: revisa el código, ejecuta tests,
    y si todo pasa, hace commit. Solo ejecuta test si review encuentra
    problemas no críticos — si review no encuentra nada, test se omite.
    """
    stack = GStack(auto_commit=auto_commit, commit_on_error=True, confirm=confirm)
    stack.push("git", "analyze_diff")
    stack.push("review", "review_package", path=".", result_key="review")
    stack.push(
        "test", "run_tests",
        run_if=lambda r, m: r[-1].success if r else True,
    )
    stack.push("git", "commit_with_changelog", message="chore: auto-commit tras develop pipeline")
    return stack.run()


def auto_release(version: str, *, auto_commit: bool = True, confirm: bool = False) -> StackResult:
    """
    Pipeline de release autónomo: ejecuta tests, genera release.
    """
    stack = GStack(auto_commit=auto_commit, commit_on_error=False, confirm=confirm)
    stack.push("test", "run_tests")
    stack.push("git", "tag_release", version=version)
    return stack.run()


def auto_fix(*, auto_commit: bool = True, confirm: bool = False) -> StackResult:
    """
    Pipeline de corrección autónoma: test → review → fix → commit.
    """
    stack = GStack(auto_commit=auto_commit, commit_on_error=True, confirm=confirm)
    stack.push("test", "run_tests", result_key="baseline")
    stack.push("review", "review_package", path=".", result_key="review")
    stack.push("refactor", "fix_bare_excepts")
    stack.push("refactor", "add_type_hints")
    stack.push("test", "run_tests")
    stack.push("git", "commit_with_changelog", message="fix: correcciones automáticas")
    return stack.run()


def auto_commit_cycle(*, phases: int = 3, auto_commit: bool = True, confirm: bool = False) -> StackResult:
    """Ciclo iterativo: revisa → test → repite."""
    stack = GStack(auto_commit=auto_commit, commit_on_error=True, confirm=confirm)
    for i in range(phases):
        stack.push("review", "review_package", path=".", result_key=f"review_{i}")
        stack.push("test", "run_tests")
    stack.push("git", "commit_with_changelog", message="chore: fin del ciclo de desarrollo autónomo")
    return stack.run()


def auto_data_pipeline(*, filename: str, target_col: str | None = None, auto_commit: bool = True, confirm: bool = False) -> StackResult:
    """
    Pipeline completo de datos: EDA → leakage → skewness → commit.
    """
    stack = GStack(auto_commit=auto_commit, commit_on_error=False, confirm=confirm)
    stack.push("data", "eda_report", filename=filename, target_col=target_col, result_key="eda")
    stack.push("data", "detect_skewness", filename=filename, result_key="skew")
    if target_col:
        stack.push("data", "detect_leakage", filename=filename, target_col=target_col)
    stack.push("git", "commit_with_changelog", message="feat: análisis de datos completo")
    return stack.run()


_AUTO_PIPELINES = {
    "analyze": auto_analyze,
    "develop": auto_develop,
    "release": auto_release,
    "fix": auto_fix,
    "cycle": auto_commit_cycle,
    "data": auto_data_pipeline,
}


def run_pipeline(name: str, **kwargs) -> StackResult:
    """Ejecuta un pipeline por nombre. Usado por la CLI."""
    if name not in _AUTO_PIPELINES:
        available = ", ".join(sorted(_AUTO_PIPELINES))
        raise ValueError(f"Pipeline '{name}' no encontrado. Disponibles: {available}")
    return _AUTO_PIPELINES[name](**kwargs)
