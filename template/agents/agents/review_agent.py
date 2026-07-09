"""
agents.agents.review_agent — Revisión de código Python del proyecto.

Complementa a `ruff` (`make lint`), no lo sustituye: `ruff` cubre estilo y
errores sintácticos; este agente busca señales que necesitan contexto de
varias funciones o archivos a la vez (funciones largas, duplicación
estructural entre archivos, mutables por defecto, TODO/FIXME, etc.).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.code_analysis_tool import CodeAnalysisTool


@register_agent
class ReviewAgent(BaseAgent):
    name = "review"
    description = (
        "Revisa código Python: funciones largas, demasiados argumentos, except "
        "desnudos, duplicación, TODO/FIXME, mutables por defecto, type hints faltantes."
    )
    capabilities = [
        "revisar", "review", "code smell", "duplicacion",
        "calidad de codigo", "bug", "todo", "fixme",
    ]

    def actions(self) -> dict:
        return {
            "review_package": self.review_package,
            "review_file": self.review_file,
        }

    def action_aliases(self) -> dict:
        # Sin esto, "revisa la calidad de codigo del paquete" rutea bien al
        # agente (por 'calidad de codigo') pero best_action no adivina la
        # acción: los nombres son ingleses y la consulta española no comparte
        # ninguna palabra con ellos (caso real encontrado probando PlanAgent).
        return {
            "review_package": ["revisa", "revisar", "paquete", "codigo", "calidad", "proyecto"],
            "review_file": ["archivo", "fichero", "modulo"],
        }

    def _deep_scan_file(self, path: Path) -> list[dict]:
        """Escanea un archivo .py buscando señales adicionales."""
        findings: list[dict] = []
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            return findings

        rel = str(path.relative_to(self.ctx.root))

        # TODO/FIXME en comentarios
        for i, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                for tag in ("TODO", "FIXME", "HACK", "XXX", "BUG"):
                    if tag in stripped.upper():
                        findings.append({
                            "file": rel, "line": i, "kind": "todo_comment",
                            "message": f"'{tag}' encontrado: {stripped.strip('# ')[:80]}",
                        })
                        break

        # Mutables como argumento por defecto
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                offset = len(node.args.args) - len(node.args.defaults)
                for i, default in enumerate(node.args.defaults):
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        arg_idx = offset + i
                        arg_name = node.args.args[arg_idx].arg if 0 <= arg_idx < len(node.args.args) else "?"
                        findings.append({
                            "file": rel, "line": default.lineno, "kind": "mutable_default",
                            "message": f"'{node.name}': argumento '{arg_name}' usa {type(default).__name__} como default",
                        })

            # Type hints faltantes en funciones públicas
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_") and not node.name.startswith("__"):
                    continue  # solo funciones públicas (no privadas ni dunder)
                if node.returns is None:
                    findings.append({
                        "file": rel, "line": node.lineno, "kind": "missing_return_type",
                        "message": f"'{node.name}' no tiene tipo de retorno",
                    })

            # Complexidad: if/for/while anidados
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                branches = sum(1 for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While, ast.AsyncFor, ast.Try)))
                if branches > 10:
                    findings.append({
                        "file": rel, "line": node.lineno, "kind": "high_complexity",
                        "message": f"'{node.name}' tiene {branches} ramas (if/for/while/try) — considera simplificar",
                    })

            # weights_only=False
            if isinstance(node, ast.Call):
                func_name = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
                if "torch.load" in func_name or "torch" in func_name:
                    for kw in node.keywords:
                        if kw.arg == "weights_only" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                            findings.append({
                                "file": rel, "line": node.lineno, "kind": "weights_only_false",
                                "message": "torch.load(weights_only=False): riesgo de pickle arbitrario. Usa True con fallback.",
                            })

        return findings

    def review_file(self, *, relative_path: str) -> AgentResult:
        path = self.ctx.root / relative_path
        if not path.exists() or path.suffix != ".py":
            return AgentResult(False, self.name, "review_file", f"'{relative_path}' no existe o no es un archivo .py.")

        smells, _functions = CodeAnalysisTool.analyze_file(path)
        deep = self._deep_scan_file(path)
        all_findings = [s.__dict__ for s in smells] + deep
        return AgentResult(
            True, self.name, "review_file", f"{len(all_findings)} hallazgo(s) en '{relative_path}'.",
            data=all_findings,
        )

    def review_package(self, *, within: str | None = None) -> AgentResult:
        """
        Revisa `{{ project_slug }}/` por defecto (el paquete principal del
        proyecto). Pasa `within` para revisar otra carpeta, p. ej. "tests".
        """
        target = within or self.ctx.config.project_slug
        if not target:
            return AgentResult(
                False, self.name, "review_package",
                "No se pudo determinar qué carpeta revisar: project_slug está vacío "
                "y no se pasó 'within' explícitamente. Pasa within='mi_paquete' o "
                "revisa .copier-answers.yml.",
            )
        base = self.ctx.root / target
        if not base.exists():
            return AgentResult(False, self.name, "review_package", f"La carpeta '{base}' no existe.")

        py_files = [
            p for p in base.rglob("*.py")
            if "__pycache__" not in p.parts
        ]
        if not py_files:
            return AgentResult(True, self.name, "review_package", f"No hay archivos .py en '{base}'.", data=[])

        all_smells = []
        all_functions = []
        all_deep: list[dict] = []
        for path in py_files:
            smells, functions = CodeAnalysisTool.analyze_file(path)
            all_smells.extend(smells)
            all_functions.extend(functions)
            all_deep.extend(self._deep_scan_file(path))

        duplicate_groups = CodeAnalysisTool.find_duplicates(all_functions)

        deep_by_kind: dict[str, list[dict]] = {}
        for f in all_deep:
            deep_by_kind.setdefault(f["kind"], []).append(f)

        report = {
            "n_files_analyzed": len(py_files),
            "smells": [s.__dict__ for s in all_smells],
            "deep_scan": all_deep,
            "deep_summary": {k: len(v) for k, v in deep_by_kind.items()},
            "duplicate_function_groups": [
                [{"file": f.file, "name": f.name, "line": f.line} for f in group]
                for group in duplicate_groups
            ],
        }
        warnings = []
        if duplicate_groups:
            warnings.append(
                f"{len(duplicate_groups)} grupo(s) de funciones con estructura AST idéntica "
                f"— posible duplicación (revisa manualmente, un falso positivo es posible)."
            )
        for kind, items in deep_by_kind.items():
            warnings.append(f"{len(items)} caso(s) de '{kind}'.")

        total = len(all_smells) + len(all_deep)
        return AgentResult(
            True, self.name, "review_package",
            f"{len(py_files)} archivo(s) analizados, {total} hallazgo(s) "
            f"({len(all_smells)} estructurales + {len(all_deep)} de escaneo profundo), "
            f"{len(duplicate_groups)} grupo(s) de posible duplicación.",
            data=report, warnings=warnings,
        )
