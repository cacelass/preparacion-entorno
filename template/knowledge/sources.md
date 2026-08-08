# Fuentes del corpus de conocimiento

Registro humano de las fuentes que respaldan `knowledge/`. La versión
**máquina** es `sources.json` — es la que lee `rag refresh` para verificar
vigencia y detectar papers nuevos. Este fichero es la lectura cómoda.

Cada fuente se autorizó a fondo al escribir el corpus (los PDFs se descargaron
y se convirtieron con `markitdown` para extraer fórmulas exactas). La fecha de
última verificación vive en `sources.json`.

## Fuentes activas verificadas (arXiv)

| Tema | Fuente | arXiv | Versión | Estado |
|------|--------|-------|---------|--------|
| PCA / SVD | A Tutorial on Principal Component Analysis (Shlens) | [1404.1100](https://arxiv.org/abs/1404.1100) | v1 | activa |
| SVD aleatorio / low-rank | Finding Structure with Randomness (Halko, Martinsson, Tropp) | [0909.4061](https://arxiv.org/abs/0909.4061) | v2 | activa |
| Optimizadores | Adam: A Method for Stochastic Optimization (Kingma, Ba) | [1412.6980](https://arxiv.org/abs/1412.6980) | v9 | activa |
| Transformers | Attention Is All You Need (Vaswani et al.) | [1706.03762](https://arxiv.org/abs/1706.03762) | v7 | activa |
| Redes residuales | Deep Residual Learning for Image Recognition (He et al.) | [1512.03385](https://arxiv.org/abs/1512.03385) | v1 | activa |
| Boosting | XGBoost: A Scalable Tree Boosting System (Chen, Guestrin) | [1603.02754](https://arxiv.org/abs/1603.02754) | v3 | activa |
| Boosting | CatBoost: Unbiased Boosting with Categorical Features | [1706.09516](https://arxiv.org/abs/1706.09516) | v5 | activa |
| Clustering | A Tutorial on Spectral Clustering (von Luxburg) | [0711.0189](https://arxiv.org/abs/0711.0189) | v1 | activa |
| Álgebra lineal | The Matrix Cookbook (Petersen, Pedersen) | [web](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) | — | activa |

## Temas cubiertos por queries (sin fuente fija)

Estadística, probabilidad, causalidad, hiperparámetros, validación, métricas,
interpretabilidad, features, fairness, seguridad, serving, deuda técnica y
calidad de datos se mantienen consultando arXiv con las queries de
`sources.json` — la literatura cambia y es mejor verificar por tema que
clavar una fuente.

## Cómo mantenerlo

```bash
uv run python -m agents --json run rag refresh --dry-run
# revisa el informe: papers nuevos relevantes + fuentes superadas
uv run python -m agents --json run rag refresh
# descarga los nuevos a knowledge/papers/<tema>/<id>.md, actualiza
# sources.json y reindexa el corpus
```

Sin red, `refresh` falla de forma controlada y no toca nada. Un paper nuevo
solo entra si la búsqueda del tema lo devuelve — la decisión de añadirlo al
corpus la toma el `lider` con el informe delante (o la feature `KNOW-001`).
