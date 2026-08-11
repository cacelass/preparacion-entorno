# Corpus de conocimiento profundo

Este directorio es el **conocimiento estable del proyecto**: matemáticas,
estadística, probabilidad, matrices, algoritmos y su aplicación, más la
ingeniería que los rodea. No es una colección de "qué es X": es teoría
profunda con fórmulas, derivaciones y el "cómo se aplica y cómo se rompe" de
cada concepto, para que el `lider` aconseje desde el conocimiento y no desde
el resumen genérico que el modelo ya sabe.

Se consulta en lenguaje natural y está indexado por el RAG (`use_rag`), así
que es buscable sin releerlo:

```bash
uv run python -m agents --json run rag search --query "regularización L2 vs L1" --file_type knowledge
uv run python -m agents --json run rag search --query "cómo elegir k en clustering" --file_type knowledge
```

Antes de aconsejar sobre un tema cubierto aquí (métricas, validación,
regularización, arquitectura de red, serving, exprimir el modelo...), el
`lider` lo consulta — el razonamiento básico no se improvisa.

## Matemáticas

| Fichero | Qué resuelve |
|---------|--------------|
| `matematicas/algebra-lineal.md` | Normas, autovalores, SVD, pseudoinversa, condicionamiento, Cholesky/QR |
| `matematicas/calculo-optimizacion.md` | Gradiente, convexidad, SGD y sus variantes, Adam, KKT, regularización |
| `matematicas/estadistica.md` | Inferencia, CLT, tests de hipótesis, bootstrap, múltiples comparaciones, OLS |
| `matematicas/probabilidad.md` | Distribuciones, Bayes, MLE/MAP, priors conjugados |
| `matematicas/matrices-app.md` | Cómo se aplica el álgebra lineal en ML: PCA, low-rank, geometría de datos |
| `matematicas/causalidad.md` | Correlación ≠ causa, DAGs, do-calculus, contrafactuales, Simpson |
| `matematicas/teoria-informacion.md` | Entropía, KL, información mutua, cross-entropy y sus usos en ML |

## Machine learning

| Fichero | Cuándo aplica |
|---------|---------------|
| `ml/supervisado.md` | `ml_type` es `supervisado` o `hibrido` — lineales, árboles, boosting, SVM, KNN |
| `ml/no-supervisado.md` | `ml_type` es `no_supervisado` — clustering y su matemática |
| `ml/redes-neuronales.md` | `ml_type` es `redes_neuronales` o `hibrido` — backprop, optimizadores, arquitecturas |
| `ml/clasificacion.md` | `task_type` es `clasificacion` |
| `ml/regresion.md` | `task_type` es `regresion` |
| `ml/metricas-y-evaluacion.md` | Siempre — métricas por tarea y comparación honesta de modelos |
| `ml/validacion.md` | Siempre — CV, series temporales, nested CV y la taxonomía de fugas |
| `ml/formas-de-aplicar.md` | Siempre — orden del pipeline, escalado, imputación, diagnóstico |
| `ml/exprime-el-modelo.md` | Siempre — palancas de datos y del modelo para sacar el máximo |
| `ml/optimizacion-hiperparametros.md` | Siempre (con el bloque `use_optuna` para `make tune`) |
| `ml/ingenieria-features.md` | Siempre — encodings, transformaciones, selección de features |
| `ml/interpretabilidad.md` | Siempre (bloque `use_shap` para la integración SHAP) |
| `ml/deuda-tecnica.md` | Siempre — la deuda oculta de un sistema ML |
| `ml/fairness-y-seguridad.md` | Siempre — sesgo, adversariales, privacidad, inyección de prompts, regulación |
| `ml/privacidad-y-fuga-datos.md` | Siempre — datos de usuario, PII, redacción, cifrado, GDPR |
| `ml/gestion-riesgo.md` | Siempre — identificar/medir/mitigar los riesgos del proyecto |
| `ml/ciclo-vida-mlops.md` | Siempre — reentrenar, drift, shadow/canary, reproducibilidad (bloque `use_monitoring`) |
| `ml/testing-ml.md` | Siempre — tests de datos, de modelo, por slices, invariantes |
| `ml/gestion-incertidumbre.md` | Siempre — calibración vs cobertura, conformal, incertidumbre a la decisión |
| `ml/llms-aplicados.md` | Siempre — LLMs: prompting, evaluación, RAG, fine-tune vs prompt, coste |
| `ml/series-temporales.md` | Siempre — forecasting, ARIMA/ETS/ML, backtest y sus fugas |
| `ml/diseno-experimentos.md` | Siempre — A/B: power, tamaño de muestra, peeking, CUPED |
| `ml/uplift-causal.md` | Siempre — efectos heterogéneos del tratamiento y targeting |
| `ml/recomendacion.md` | Siempre — CF, factorización, two-tower, cold start, evaluación |
| `ml/modelos-bayesianos.md` | Siempre — modelado bayesiano aplicado: MCMC/HMC, GLMs, jerárquicos |
| `ml/deteccion-anomalias.md` | Siempre — outlier vs novelty, Isolation Forest, LOF, evaluación |
| `ml/compresion-modelos.md` | Siempre — cuantización, destilación, pruning (bloque `redes_neuronales`) |
| `ml/reinforcement-learning.md` | Problema secuencial con recompensa — MDP, DQN/PPO/SAC, reward design, evaluación |
| `ml/metaheuristica.md` | Optimización no diferenciable o combinatoria — GA, recocido simulado, búsqueda |
| `ml/modelos-fundacionales.md` | Siempre (bloque LLM/FM) — pre-training, adaptación (LoRA/QLoRA/aLoRA), evaluación, coste |
| `ml/guardarraíles.md` | FM/LLM expuestos — capas de contención: entrada, filtros, acciones, red teaming |
| `ml/evals-de-sistemas.md` | Sistemas con LLM — golden sets, evals-as-code, evaluar trayectorias de agentes, cuándo el eval miente |
| `ml/contexto-y-memoria.md` | Agentes — la ventana como recurso, memoria externa, handoff sin heredar, compresión |
| `ml/neurodifuso.md` | "Un modelo diferente" — ANFIS: el nicho y la explosión combinatoria de reglas |

## Ingeniería

| Fichero | Qué resuelve |
|---------|--------------|
| `ingenieria/eficiencia.md` | Complejidad, vectorización, memoria, paralelismo, GPU |
| `ingenieria/calidad-codigo.md` | Tipado, linting, tests, refactoring, revisión |
| `ingenieria/estructuras-codigo.md` | Layout de proyecto, fronteras de módulos, pipelines, reproducibilidad |
| `ingenieria/reglas-codigo.md` | Checklist de reglas duras para el reviewer y el `lider` |
| `ingenieria/patrones-diseño.md` | Patrones con relevancia DS y cuándo NO usarlos |
| `ingenieria/git.md` | Modelo de objetos, flujos, rebase, bisect, reglas del arnés |
| `ingenieria/linux.md` | Shell, procesos, texto, GPU, seguridad de la máquina |
| `ingenieria/ci-cd.md` | CI de código vs CI de ML, el workflow que trae el proyecto, puertas |

## Datos, backend y frontend

| Fichero | Cuándo aplica |
|---------|---------------|
| `data/calidad-datos.md` | Siempre — contratos de datos, validación, versionado, drift |
| `data/ingenieria-datos.md` | Siempre — SQL, pandas idiomático, DuckDB (bloque `use_duckdb`), escala |
| `data/visualizacion.md` | Siempre — figuras honestas, percepción, storytelling |
| `backend/servir-modelos.md` | Siempre — latencia, batching, versionado, monitoreo en producción |
| `backend/api.md` | `use_api` — FastAPI, async, pydantic, producción |
| `backend/docker.md` | `use_docker` — imagen reproducible y segura |
| `backend/mlflow.md` | `use_mlflow` — tracking, registry, flavors |
| `frontend/chat.md` | `use_docker` — la interfaz web de chat |

## Mantenimiento: el corpus se actualiza solo

Las fuentes y su vigencia viven en `sources.json` (máquina) y `sources.md`
(humano). El agente `rag` las mantiene:

```bash
uv run python -m agents --json run rag refresh --dry-run   # ¿qué hay nuevo? ¿qué se superó?
uv run python -m agents --json run rag refresh             # descarga los nuevos y reindexa
```

La feature `KNOW-001` del backlog lo ejecuta periódicamente.
