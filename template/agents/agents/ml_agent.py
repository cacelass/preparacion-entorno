"""
agents.agents.ml_agent — Análisis de modelos entrenados para este template.

Conoce que los modelos se guardan en `models/*.joblib` (ver
`{{ project_slug }}/models/train_model.py`) y los encoders/scalers en
`models/artifacts/`. No entrena modelos nuevos — eso es responsabilidad de
`make train`; este agente analiza lo que ya existe en disco.

Es consciente del `ml_type` del proyecto: ajusta su comportamiento según si
el proyecto es `supervisado`, `no_supervisado`, `redes_neuronales` o `hibrido`
(ver `self.ctx.config.ml_type`).
"""

from __future__ import annotations

from pathlib import Path

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.sklearn_tool import SklearnTool


@register_agent
class MLAgent(BaseAgent):
    name = "ml"
    description = (
        "Analiza modelos entrenados (.joblib): overfitting/underfitting a partir de "
        "métricas dadas, importancia de variables, inspección de hiperparámetros. "
        "Es consciente del tipo de ML (supervisado/no_supervisado/redes/híbrido)."
    )
    capabilities = [
        "modelo", "overfitting", "underfitting", "hiperparametros", "importancia",
        "metricas", "algoritmo", "entrenamiento", "optuna", "comparar modelos",
    ]

    def action_aliases(self) -> dict:
        return {
            "analyze_optuna": ["optuna", "hiperparametros", "estudio", "trial"],
            "model_comparison": ["comparar", "comparativa", "mejor modelo", "ranking"],
        }

    def actions(self) -> dict:
        base = {
            "list_models": self.list_models,
            "inspect_model": self.inspect_model,
            "feature_importance": self.feature_importance,
            "check_overfitting": self.check_overfitting,
        }
        if self.ctx.config.use_optuna:
            base["analyze_optuna"] = self.analyze_optuna
        if self.ctx.config.ml_type in ("supervisado", "hibrido"):
            base["model_comparison"] = self.model_comparison
        return base

    def inspect_model(self, *, model_name: str) -> AgentResult:
        path = self._resolve_model_path(model_name)
        if path is None:
            return AgentResult(False, self.name, "inspect_model", f"No se encontró el modelo '{model_name}' en models/.")
        try:
            estimator = SklearnTool.load(path)
        except Exception as exc:  # noqa: BLE001 — cualquier fallo de deserialización es un error de datos, no de lógica
            return AgentResult(False, self.name, "inspect_model", f"No se pudo cargar '{model_name}': {exc}")

        info = SklearnTool.inspect(estimator)
        return AgentResult(True, self.name, "inspect_model", f"'{model_name}' inspeccionado.", data=info)

    def feature_importance(self, *, model_name: str, feature_names: list[str] | None = None) -> AgentResult:
        path = self._resolve_model_path(model_name)
        if path is None:
            return AgentResult(False, self.name, "feature_importance", f"No se encontró el modelo '{model_name}' en models/.")
        estimator = SklearnTool.load(path)
        importances = SklearnTool.feature_importances(estimator, feature_names)
        if importances is None:
            return AgentResult(
                True, self.name, "feature_importance",
                f"'{model_name}' no expone feature_importances_ ni coef_ (¿es KNN u otro modelo sin esa propiedad?).",
                data=None,
            )
        top = dict(list(importances.items())[:15])
        return AgentResult(
            True, self.name, "feature_importance",
            f"Top {len(top)} variables por importancia en '{model_name}'.", data=top,
        )

    def check_overfitting(self, *, train_score: float, test_score: float, gap_threshold: float = 0.1) -> AgentResult:
        verdict = SklearnTool.detect_overfitting(train_score, test_score, gap_threshold=gap_threshold)
        warnings = [] if verdict["verdict"] == "ok" else [verdict["note"]]
        return AgentResult(
            True, self.name, "check_overfitting", verdict["note"], data=verdict, warnings=warnings,
        )

    def list_models(self) -> AgentResult:
        """Lista modelos según el ml_type del proyecto."""
        models = SklearnTool.list_models(self.ctx.models_dir)
        if self.ctx.config.ml_type == "no_supervisado":
            models = [p for p in models if "KMeans" in p.name or "Agglomerative" in p.name or "Pipeline" in p.name]
        elif self.ctx.config.ml_type == "redes_neuronales":
            models = [p for p in models if p.suffix in (".pt", ".pth")]
        else:
            models = [p for p in models if p.suffix == ".joblib"]

        if not models:
            return AgentResult(
                True, self.name, "list_models",
                "No hay modelos en models/ todavía (ejecuta 'make train' primero).",
                data=[],
            )
        return AgentResult(
            True, self.name, "list_models", f"{len(models)} modelo(s) encontrado(s).",
            data=[p.name for p in models],
        )

    def analyze_optuna(self) -> AgentResult:
        """Analiza estudios de Optuna en busca de mejores parámetros."""
        try:
            import optuna
        except ImportError:
            return AgentResult(False, self.name, "analyze_optuna", "Optuna no está instalado.")

        study_dir = self.ctx.tools_dir
        if not study_dir.exists():
            return AgentResult(False, self.name, "analyze_optuna", "El directorio 'tools/' no existe.")

        dbs = list(study_dir.glob("*.db"))
        if not dbs:
            return AgentResult(True, self.name, "analyze_optuna", "No se encontraron estudios de Optuna.", data=[])

        results = []
        for db_path in dbs:
            try:
                storage = optuna.storages.RDBStorage(f"sqlite:///{db_path}")
                study_names = [s.study_name for s in storage.get_all_studies()]
                for name in study_names:
                    study = optuna.load_study(storage=storage, study_name=name)
                    if study.best_trial:
                        results.append({
                            "study_name": name,
                            "best_value": study.best_value,
                            "best_params": study.best_params,
                            "n_trials": len(study.trials),
                        })
            except Exception as exc:
                results.append({"file": db_path.name, "error": str(exc)})

        return AgentResult(
            True, self.name, "analyze_optuna",
            f"{len(results)} estudio(s) analizado(s).",
            data=results,
        )

    def model_comparison(self) -> AgentResult:
        """
        Compara todos los modelos supervisados entrenados: métricas, tamaño y tipo.
        Útil para elegir el mejor modelo para producción.
        """
        models = SklearnTool.list_models(self.ctx.models_dir)
        joblib_models = [p for p in models if p.suffix == ".joblib"]
        if not joblib_models:
            return AgentResult(False, self.name, "model_comparison", "No hay modelos .joblib para comparar.")

        comparison = []
        for path in joblib_models:
            try:
                estimator = SklearnTool.load(path)
                info = SklearnTool.inspect(estimator)
                size_kb = round(path.stat().st_size / 1024, 1)
                comparison.append({
                    "name": path.stem,
                    "type": info.get("estimator_type", "?"),
                    "params": info.get("n_params", "?"),
                    "features": info.get("n_features", "?"),
                    "size_kb": size_kb,
                })
            except Exception:
                comparison.append({"name": path.stem, "error": "no se pudo cargar"})

        comparison.sort(key=lambda x: x.get("size_kb", 0))
        return AgentResult(
            True, self.name, "model_comparison",
            f"{len(comparison)} modelo(s) comparados. El más ligero: {comparison[0]['name']} ({comparison[0].get('size_kb', '?')} KB).",
            data=comparison,
        )

    def _resolve_model_path(self, model_name: str):
        candidate = self.ctx.models_dir / model_name
        if not candidate.suffix:
            candidate = candidate.with_suffix(".joblib")
        return candidate if candidate.exists() else None
