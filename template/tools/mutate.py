#!/usr/bin/env python3
"""
tools/mutate.py — Mutation testing sin dependencias (flujo Robert C. Martin).

Muta pequeñas partes del código de producción (operadores de comparación,
operadores booleanos y literales True/False) y ejecuta la suite de tests por
cada mutante. Si un test falla, el mutante está "killed" — los tests muerden.
Si los tests siguen pasando, el mutante "survives": hay un hueco en la suite
que la cobertura por líneas no detecta.

Uso:
    python tools/mutate.py <archivo.py> [--tests tests/] [--timeout 60]

El archivo objetivo se muta EN SITIO con backup y se restaura siempre en un
`finally`: si el proceso muere a mitad, el código original queda intacto.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import time
from pathlib import Path


#: Cada regla describe una mutación textual: (tipo de nodo, atributo, valor
#: nuevo) para AST, mapeado a texto por `_mutated_source`.
COMPARISON_FLIPS = {
    "Lt": "GtE",
    "GtE": "Lt",
    "Gt": "LtE",
    "LtE": "Gt",
    "Eq": "NotEq",
    "NotEq": "Eq",
}

BOOLEAN_FLIPS = {"And": "Or", "Or": "And"}

CONSTANT_FLIPS = {"True": "False", "False": "True"}


class _OperatorFlip(ast.NodeTransformer):
    """Aplica la mutación del operador `flip` a la PRIMERA aparición real.

    Cada llamada muta exactamente un operador objetivo (p. ej. "Lt"), no
    "cualquier operador que se parezca". El flag `_applied` corta tras la
    primera aparición para que un `mutate_file` sobre la misma fuente genere
    un mutante por sitio real, no N mutantes del mismo sitio.
    """

    def __init__(self, flip: str):
        self._flip = flip
        self._applied = False

    def _swap(self, node, flips: dict) -> bool:
        if self._applied:
            return False
        if type(node).__name__ == self._flip:
            self._applied = True
            return True
        return False

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if self._applied:
            return node
        new_ops = []
        for op in node.ops:
            if not self._applied and self._swap(op, COMPARISON_FLIPS):
                new_ops.append(_build_op(COMPARISON_FLIPS[self._flip]))
            else:
                new_ops.append(op)
        node.ops = new_ops
        return self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        if self._applied:
            return node
        if self._swap(node.op, BOOLEAN_FLIPS):
            node.op = _build_op(BOOLEAN_FLIPS[self._flip])
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self._applied:
            return node
        if self._flip == "True" and node.value is True:
            self._applied = True
            node.value = False
        elif self._flip == "False" and node.value is False:
            self._applied = True
            node.value = True
        return node


def _build_op(name: str) -> ast.operator:
    cls = getattr(ast, name)
    return cls()


def _mutated_source(source: str, flip: str) -> str | None:
    """Aplica UNA mutación a `source`. Devuelve None si ya no hay sitios que mutar."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    transformer = _OperatorFlip(flip)
    transformer.visit(tree)
    if not transformer._applied:
        return None
    return ast.unparse(tree)


def _mutation_sites(source: str) -> list[str]:
    """Los distintos operadores mutables presentes en el código, en orden."""
    sites = []
    for name in ("Lt", "GtE", "Gt", "LtE", "Eq", "NotEq", "And", "Or", "True", "False"):
        if _mutated_source(source, name) is not None:
            sites.append(name)
    return sites


def _run_tests(root: Path, tests_dir: Path, timeout: int) -> str:
    """Ejecuta la suite. Devuelve 'killed' | 'survived' | 'timeout'."""
    args = [sys.executable, "-m", "pytest", str(tests_dir), "-q", "--no-header"]
    try:
        start = time.monotonic()
        proc = subprocess.run(
            args, cwd=str(root), capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.monotonic() - start
        if proc.returncode != 0:
            return f"killed ({elapsed:.1f}s)"
        return f"survived ({elapsed:.1f}s)"
    except subprocess.TimeoutExpired:
        return "timeout"


def mutate_file(path: Path, tests_dir: Path, timeout: int) -> dict:
    """Muta `path` en sitios reales y mide si los tests los detectan."""
    if not path.exists():
        return {"error": f"No existe el archivo objetivo: {path}"}
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {"error": f"No se pudo leer {path}: {exc}"}

    original = path.read_bytes()
    root = Path.cwd()
    sites = _mutation_sites(source)
    results = {
        "killed": 0,
        "survived": 0,
        "timeout": 0,
        "total": len(sites),
        "detail": [],
    }

    try:
        for site in sites:
            mutated = _mutated_source(source, site)
            if mutated is None:
                continue
            try:
                path.write_text(mutated, encoding="utf-8")
            except OSError as exc:
                results["detail"].append(
                    {"site": site, "status": "error", "error": str(exc)}
                )
                continue
            status = _run_tests(root, tests_dir, timeout)
            kind = status.split(" ")[0]
            results[kind] = results.get(kind, 0) + 1
            results["detail"].append({"site": site, "status": kind})
    finally:
        path.write_bytes(original)

    killed = results["killed"]
    total_mutants = killed + results["survived"]
    results["score"] = (
        round(killed / total_mutants * 100, 1) if total_mutants else 100.0
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Mutation testing sin dependencias.")
    parser.add_argument("target", help="Archivo Python a mutar (ruta al módulo).")
    parser.add_argument(
        "--tests", default="tests/", help="Directorio de tests (por defecto tests/)."
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="Timeout por mutante en segundos."
    )
    args = parser.parse_args()

    root = Path.cwd()
    target = Path(args.target).resolve()
    if not str(target).startswith(str(root.resolve())):
        print("El archivo objetivo debe estar dentro del proyecto.", file=sys.stderr)
        return 1
    tests_dir = Path(args.tests).resolve()

    report = mutate_file(target, tests_dir, args.timeout)
    if "error" in report:
        print(report["error"], file=sys.stderr)
        return 1

    print(f"Mutación de {target.name}:")
    print(
        f"  {report['total']} sitio(s) · killed {report['killed']} · "
        f"survived {report['survived']} · timeout {report['timeout']}"
    )
    print(f"  Score de mutación: {report['score']}%")
    for item in report["detail"]:
        mark = "✔ killed" if item["status"] == "killed" else "✘ survived"
        print(f"    {mark:<12} {item['site']}")

    if report["survived"]:
        print("\nSurvivientes: hay código de producción que los tests no protegen.")
        print(
            "Considera añadir tests para esos casos (mira los sitios marcados con ✘)."
        )

    return 0 if report["survived"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
