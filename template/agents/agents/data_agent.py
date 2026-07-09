"""
agents.agents.data_agent — Análisis de calidad de datos para este template.

Conoce dónde vive cada etapa del pipeline de datos:
`data/raw/` (crudo) -> `data/processed/` (`{{ project_slug }}/data/make_dataset.py`)
-> `data/interim/` (`{{ project_slug }}/features/build_features.py`). Por
defecto analiza `data/raw/`, que es donde tiene más sentido detectar
problemas antes de que se propaguen al resto del pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.data_io_tool import DataIOTool
from agents.tools.dataframe_analysis_tool import DataFrameAnalysisTool
from agents.tools.validate_tool import ValidateTool
from agents.tools.stats_tool import StatsTool


def _has_plt() -> bool:
    """Verifica si matplotlib está disponible."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        return True
    except ImportError:
        return False


@register_agent
class DataAgent(BaseAgent):
    name = "data"
    description = (
        "Analiza datasets: columnas constantes, cardinalidad, outliers, fuga de "
        "información, correlaciones, sesgo (skewness) para LOGCOLS, y genera "
        "informes de perfilado / sugerencias de imputación."
    )
    capabilities = [
        "dataset", "datos", "eda", "outlier", "outliers", "cardinalidad",
        "fuga de informacion", "leakage", "correlacion", "csv", "parquet",
        "limpieza", "features", "skewness", "sesgo", "imputacion", "profiling",
    ]

    def action_aliases(self) -> dict:
        return {
            "profiling_report": ["perfil", "profiling", "informe completo"],
            "suggest_imputation": ["imputar", "imputacion", "rellenar nulos", "missing values"],
            "detect_skewness": ["sesgo", "skewness", "logcols", "transformacion log"],
            "generate_plots": ["graficos", "pairplot", "correlacion matriz", "heatmap", "distribuciones"],
            "quality_check": ["calidad", "data quality", "profile", "perfil rapido"],
            "statistical_summary": ["estadistico", "normalidad", "correlacion", "test estadistico"],
        }

    def actions(self) -> dict:
        return {
            "list_datasets": self.list_datasets,
            "eda_report": self.eda_report,
            "detect_leakage": self.detect_leakage,
            "profiling_report": self.profiling_report,
            "suggest_imputation": self.suggest_imputation,
            "detect_skewness": self.detect_skewness,
            "generate_plots": self.generate_plots,
            "quality_check": self.quality_check,
            "statistical_summary": self.statistical_summary,
        }

    # -------------------------------------------------------------------------
    def _load(self, path: Path) -> pd.DataFrame:
        reader = DataIOTool.infer_reader(path)
        result = reader(path)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"'{path.name}' no se pudo cargar como tabla (¿es un .json con estructura no tabular?)."
            )
        return result

    def list_datasets(self) -> AgentResult:
        candidates = []
        for stage_dir in (self.ctx.raw_data_dir, self.ctx.interim_data_dir, self.ctx.processed_data_dir):
            if stage_dir.exists():
                candidates.extend(
                    p for p in stage_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in (".csv", ".parquet", ".json")
                )
        return AgentResult(
            True, self.name, "list_datasets",
            f"{len(candidates)} archivo(s) de datos encontrado(s).",
            data=[str(p.relative_to(self.ctx.root)) for p in sorted(candidates)],
        )

    def eda_report(self, *, filename: str, target_col: str | None = None) -> AgentResult:
        """
        Genera un informe EDA sobre `data/raw/<filename>` (o una ruta relativa
        a la raíz del proyecto si `filename` no está en `data/raw/`).
        """
        path = self.ctx.raw_data_dir / filename
        if not path.exists():
            path = self.ctx.root / filename
        if not path.exists():
            return AgentResult(False, self.name, "eda_report", f"No se encontró el archivo '{filename}'.")

        try:
            df = self._load(path)
        except (ValueError, TypeError) as exc:
            return AgentResult(False, self.name, "eda_report", str(exc))

        tool = DataFrameAnalysisTool
        report = {
            "summary": tool.summary(df),
            "constant_columns": [f.__dict__ for f in tool.constant_columns(df)],
            "high_cardinality_columns": [f.__dict__ for f in tool.high_cardinality_columns(df)],
            "high_missing_columns": [f.__dict__ for f in tool.high_missing_columns(df)],
            "outliers": [f.__dict__ for f in tool.outliers_iqr(df)],
            "highly_correlated_pairs": tool.highly_correlated_pairs(df),
        }
        warnings = []
        if target_col:
            leakage = tool.leakage_suspects(df, target_col)
            report["leakage_suspects"] = [f.__dict__ for f in leakage]
            if leakage:
                warnings.append(f"{len(leakage)} columna(s) sospechosa(s) de fuga de información con '{target_col}'.")

        n_issues = sum(
            len(report[k]) for k in
            ("constant_columns", "high_cardinality_columns", "high_missing_columns", "outliers")
        )
        return AgentResult(
            True, self.name, "eda_report",
            f"EDA de '{filename}' completo: {df.shape[0]} filas × {df.shape[1]} columnas, {n_issues} hallazgo(s).",
            data=report, warnings=warnings,
        )

    def detect_leakage(self, *, filename: str, target_col: str, correlation_threshold: float = 0.95) -> AgentResult:
        path = self.ctx.raw_data_dir / filename
        if not path.exists():
            path = self.ctx.root / filename
        if not path.exists():
            return AgentResult(False, self.name, "detect_leakage", f"No se encontró el archivo '{filename}'.")

        try:
            df = self._load(path)
        except (ValueError, TypeError) as exc:
            return AgentResult(False, self.name, "detect_leakage", str(exc))

        suspects = DataFrameAnalysisTool.leakage_suspects(df, target_col, correlation_threshold=correlation_threshold)
        return AgentResult(
            True, self.name, "detect_leakage",
            f"{len(suspects)} columna(s) sospechosa(s) de fuga de información.",
            data=[f.__dict__ for f in suspects],
            warnings=["Correlación alta no es prueba de fuga — es una señal para revisar manualmente."] if suspects else [],
        )

    def profiling_report(self, *, filename: str, output: str | None = None) -> AgentResult:
        """
        Genera un informe HTML de perfilado con ydata-profiling.

        Si ydata-profiling no está instalado, informa del error sin
        bloquear — no es una dependencia obligatoria del template.
        """
        path = self.ctx.raw_data_dir / filename
        if not path.exists():
            path = self.ctx.root / filename
        if not path.exists():
            return AgentResult(False, self.name, "profiling_report", f"No se encontró el archivo '{filename}'.")

        try:
            import ydata_profiling  # noqa: F401
            from ydata_profiling import ProfileReport
        except ImportError:
            try:
                import pandas_profiling  # noqa: F401
                from pandas_profiling import ProfileReport
            except ImportError:
                return AgentResult(
                    False, self.name, "profiling_report",
                    "ydata-profiling no está instalado. Ejecuta: uv add ydata-profiling",
                )

        try:
            df = self._load(path)
        except (ValueError, TypeError) as exc:
            return AgentResult(False, self.name, "profiling_report", str(exc))

        output_path = Path(output) if output else self.ctx.agent_workspace("data") / f"{path.stem}_profile.html"
        profile = ProfileReport(df, title=f"Perfil de {path.name}", minimal=True)
        profile.to_file(str(output_path))

        return AgentResult(
            True, self.name, "profiling_report",
            f"Informe de perfilado generado: {output_path}",
            data={"path": str(output_path), "n_rows": df.shape[0], "n_cols": df.shape[1]},
        )

    def suggest_imputation(self, *, filename: str) -> AgentResult:
        """
        Analiza valores nulos y sugiere estrategias de imputación.
        """
        path = self.ctx.raw_data_dir / filename
        if not path.exists():
            path = self.ctx.root / filename
        if not path.exists():
            return AgentResult(False, self.name, "suggest_imputation", f"No se encontró el archivo '{filename}'.")

        try:
            df = self._load(path)
        except (ValueError, TypeError) as exc:
            return AgentResult(False, self.name, "suggest_imputation", str(exc))

        null_pct = df.isnull().mean()
        cols_with_nulls = null_pct[null_pct > 0].to_dict()

        suggestions = []
        for col, pct in cols_with_nulls.items():
            if df[col].dtype.kind in ("i", "f"):
                if pct < 0.05:
                    suggestions.append({"column": col, "null_pct": round(pct, 3), "suggestion": f"fillna(mean) — {pct:.1%} nulos, pocos"})
                elif pct < 0.3:
                    suggestions.append({"column": col, "null_pct": round(pct, 3), "suggestion": "fillna(median) — distribución robusta a sesgo"})
                else:
                    suggestions.append({"column": col, "null_pct": round(pct, 3), "suggestion": "considera si la columna es útil; >30% nulos"})
            else:
                if pct < 0.1:
                    suggestions.append({"column": col, "null_pct": round(pct, 3), "suggestion": "fillna(mode) — categórica con pocos nulos"})
                else:
                    suggestions.append({"column": col, "null_pct": round(pct, 3), "suggestion": "considera agrupar nulos como categoría 'Unknown'"})

        return AgentResult(
            True, self.name, "suggest_imputation",
            f"{len(suggestions)} columna(s) con nulos analizadas.",
            data={"suggestions": suggestions, "n_rows": df.shape[0]},
            warnings=[] if len(suggestions) < 5 else [f"{len(suggestions)} columnas con nulos — revisa si todas son relevantes."],
        )

    def detect_skewness(self, *, filename: str, threshold: float = 1.0) -> AgentResult:
        """
        Detecta columnas numéricas con distribución sesgada (skewness > threshold)
        candidatas para transformación logarítmica (LOGCOLS).
        """
        path = self.ctx.raw_data_dir / filename
        if not path.exists():
            path = self.ctx.root / filename
        if not path.exists():
            return AgentResult(False, self.name, "detect_skewness", f"No se encontró el archivo '{filename}'.")

        try:
            df = self._load(path)
        except (ValueError, TypeError) as exc:
            return AgentResult(False, self.name, "detect_skewness", str(exc))

        num_cols = df.select_dtypes(include=[np.number]).columns
        skew_values = df[num_cols].skew().dropna().to_dict()

        high_skew = {k: round(v, 3) for k, v in sorted(skew_values.items(), key=lambda x: -abs(x[1])) if abs(v) > threshold}
        low_skew = {k: round(v, 3) for k, v in sorted(skew_values.items(), key=lambda x: -abs(x[1])) if abs(v) <= threshold}

        return AgentResult(
            True, self.name, "detect_skewness",
            f"{len(high_skew)} columna(s) con |skew| > {threshold} candidatas para LOGCOLS.",
            data={"high_skew": high_skew, "low_skew": low_skew, "threshold": threshold},
            warnings=(
                [f"Añade a LOGCOLS: {list(high_skew.keys())[:10]}" + ("..." if len(high_skew) > 10 else "")]
                if high_skew else []
            ),
        )

    def quality_check(self, *, filename: str) -> AgentResult:
        """Evalúa la calidad del dataset: nulos, constantes, outliers, perfil."""
        path = self.ctx.raw_data_dir / filename
        if not path.exists():
            path = self.ctx.root / filename
        if not path.exists():
            return AgentResult(False, self.name, "quality_check", f"No se encontró el archivo '{filename}'.")

        try:
            df = self._load(path)
        except (ValueError, TypeError) as exc:
            return AgentResult(False, self.name, "quality_check", str(exc))

        quality = ValidateTool.check_data_quality(df)
        profile = ValidateTool.profile(df)
        constant_cols = [c for c, info in quality["columns"].items() if info.get("constant")]
        high_null = [c for c, info in quality["columns"].items() if info["null_pct"] > 0.1]
        high_card = [c for c, info in quality["columns"].items() if info["cardinality_pct"] > 0.95]

        warnings = []
        if constant_cols:
            warnings.append(f"{len(constant_cols)} columna(s) constantes: {constant_cols[:5]}{'...' if len(constant_cols) > 5 else ''}")
        if high_null:
            warnings.append(f"{len(high_null)} columna(s) con >10% nulos: {high_null[:5]}{'...' if len(high_null) > 5 else ''}")
        if high_card:
            warnings.append(f"{len(high_card)} columna(s) con >95% cardinalidad (posibles IDs): {high_card[:5]}{'...' if len(high_card) > 5 else ''}")
        if quality["duplicate_rows_pct"] > 0.01:
            warnings.append(f"{quality['duplicate_rows_pct']:.1%} filas duplicadas.")

        return AgentResult(
            True, self.name, "quality_check",
            f"Calidad de '{filename}': {profile['rows']} filas × {profile['cols']} columnas, {len(warnings)} advertencia(s).",
            data={"profile": profile, "quality": quality},
            warnings=warnings,
        )

    def statistical_summary(self, *, filename: str, target_col: str | None = None) -> AgentResult:
        """Tests estadísticos sobre columnas numéricas: normalidad y correlaciones."""
        path = self.ctx.raw_data_dir / filename
        if not path.exists():
            path = self.ctx.root / filename
        if not path.exists():
            return AgentResult(False, self.name, "statistical_summary", f"No se encontró el archivo '{filename}'.")

        try:
            df = self._load(path)
        except (ValueError, TypeError) as exc:
            return AgentResult(False, self.name, "statistical_summary", str(exc))

        num_cols = df.select_dtypes(include=[np.number]).columns
        normality_results = {}
        for col in num_cols[:20]:
            clean = df[col].dropna()
            if len(clean) >= 8:
                normality_results[col] = StatsTool.normal_test(clean.values)

        non_normal = [c for c, r in normality_results.items() if not r["normal"]]

        correlation_results = {}
        if len(num_cols) >= 2:
            for i, c1 in enumerate(num_cols[:10]):
                for c2 in num_cols[i + 1:10]:
                    pair = (c1, c2)
                    clean = df[[c1, c2]].dropna()
                    if len(clean) >= 10:
                        corr = StatsTool.correlation(clean[c1].values, clean[c2].values)
                        if abs(corr["statistic"]) > 0.7:
                            correlation_results[f"{c1} vs {c2}"] = corr

        target_stats = None
        if target_col and target_col in df.columns:
            if df[target_col].dtype.kind in "ifc":
                target_stats = {"mean": float(df[target_col].mean()), "std": float(df[target_col].std()),
                                "min": float(df[target_col].min()), "max": float(df[target_col].max())}
            else:
                target_stats = {"value_counts": df[target_col].value_counts().head(10).to_dict()}

        warnings = []
        if non_normal:
            warnings.append(f"{len(non_normal)} columna(s) no normales (n={len(non_normal[:10])} mostradas): {non_normal[:10]}")
        if correlation_results:
            warnings.append(f"{len(correlation_results)} par(es) con |r| > 0.7.")

        return AgentResult(
            True, self.name, "statistical_summary",
            f"Análisis estadístico de '{filename}': {len(num_cols)} numéricas, {len(non_normal)} no normales, {len(correlation_results)} correlaciones fuertes.",
            data={
                "normality": normality_results,
                "high_correlations": correlation_results,
                "target_stats": target_stats,
                "n_numeric": len(num_cols),
            },
            warnings=warnings,
        )

    def generate_plots(self, *, filename: str, output_dir: str | None = None) -> AgentResult:
        """
        Genera un conjunto de gráficos exploratorios: pairplot, matriz de
        correlación (heatmap), histogramas y boxplots numéricos.

        Requiere matplotlib y seaborn. Si no están instalados, lo indica
        sin bloquear.
        """
        if not _has_plt():
            return AgentResult(
                False, self.name, "generate_plots",
                "matplotlib no está instalado. Ejecuta: uv add matplotlib seaborn",
            )

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            return AgentResult(
                False, self.name, "generate_plots",
                "seaborn no está instalado. Ejecuta: uv add seaborn",
            )

        path = self.ctx.raw_data_dir / filename
        if not path.exists():
            path = self.ctx.root / filename
        if not path.exists():
            return AgentResult(False, self.name, "generate_plots", f"No se encontró el archivo '{filename}'.")

        try:
            df = self._load(path)
        except (ValueError, TypeError) as exc:
            return AgentResult(False, self.name, "generate_plots", str(exc))

        out_dir = Path(output_dir) if output_dir else self.ctx.agent_workspace("data") / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        num_cols = df.select_dtypes(include=[np.number]).columns[:10]
        cat_cols = df.select_dtypes(exclude=[np.number]).columns[:5]
        sns.set_theme(style="whitegrid")
        generated = []

        if len(num_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[num_cols].corr(), annot=len(num_cols) <= 6, cmap="RdBu_r", center=0, ax=ax, fmt=".2f")
            ax.set_title("Matriz de Correlación")
            heat_path = out_dir / f"{path.stem}_correlation.png"
            fig.savefig(heat_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            generated.append(str(heat_path.relative_to(self.ctx.root)))

        if len(num_cols) >= 2 and len(num_cols) <= 6:
            fig = sns.pairplot(df[num_cols], diag_kind="kde", corner=True)
            pair_path = out_dir / f"{path.stem}_pairplot.png"
            fig.savefig(pair_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            generated.append(str(pair_path.relative_to(self.ctx.root)))

        if len(num_cols) > 0:
            n_plots = min(len(num_cols), 9)
            fig, axes = plt.subplots(n_plots, 2, figsize=(14, 3 * n_plots))
            if n_plots == 1:
                axes = [axes]
            for i, col in enumerate(num_cols[:n_plots]):
                ax_hist = axes[i][0] if n_plots > 1 else axes[0]
                ax_box  = axes[i][1] if n_plots > 1 else axes[1]
                sns.histplot(df[col].dropna(), kde=True, ax=ax_hist)
                ax_hist.set_title(f"{col} — Histograma")
                sns.boxplot(y=df[col].dropna(), ax=ax_box)
                ax_box.set_title(f"{col} — Boxplot")
            for j in range(i + 1, n_plots):
                fig.delaxes(axes[j][0] if n_plots > 1 else axes[j])
                fig.delaxes(axes[j][1] if n_plots > 1 else axes[j])
            fig.tight_layout()
            dist_path = out_dir / f"{path.stem}_distributions.png"
            fig.savefig(dist_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            generated.append(str(dist_path.relative_to(self.ctx.root)))

        if len(cat_cols) > 1:
            fig, axes = plt.subplots(1, min(len(cat_cols), 4), figsize=(16, 4))
            if len(cat_cols) == 1:
                axes = [axes]
            for i, col in enumerate(cat_cols[:4]):
                df[col].value_counts().head(15).plot(kind="bar", ax=axes[i])
                axes[i].set_title(f"{col} — Top 15 categorías")
                axes[i].tick_params(axis="x", rotation=45)
            fig.tight_layout()
            cat_path = out_dir / f"{path.stem}_categorical.png"
            fig.savefig(cat_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            generated.append(str(cat_path.relative_to(self.ctx.root)))

        return AgentResult(
            True, self.name, "generate_plots",
            f"{len(generated)} gráfico(s) generado(s) en {out_dir}",
            data={"generated": generated, "output_dir": str(out_dir)},
        )
