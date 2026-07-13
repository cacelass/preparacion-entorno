# Guía de modelos — {{ project_name }}

> Cuándo usar cada modelo, trade-offs, y referencias.

## Clasificación / Regresión

| Modelo | Lineal | No lineal | Interpretable | Outliers | Gran dim. | Grandes datos | Paper |
|--------|--------|-----------|---------------|----------|-----------|---------------|-------|
| LinearRegression | ✅ | ❌ | ✅✅ | ❌ | ❌ | ✅ | |
| Ridge | ✅ | ❌ | ✅✅ | ⚠️ | ✅ | ✅ | |
| Lasso | ✅ | ❌ | ✅✅ | ⚠️ | ✅✅ | ✅ | |
| LogisticRegression | ✅ | ❌ | ✅✅ | ⚠️ | ✅ | ✅ | |
| KNN | ❌ | ✅ | ❌ | ⚠️ | ❌ | ❌ | |
| DecisionTree | ❌ | ✅ | ✅✅ | ✅ | ❌ | ✅ | |
| SVM (RBF) | ❌ | ✅ | ❌ | ⚠️ | ✅ | ❌ | Cortes & Vapnik 1995 |
| RandomForest | ❌ | ✅ | ⚠️ | ✅ | ✅ | ✅ | Breiman 2001 |
| ExtraTrees | ❌ | ✅ | ⚠️ | ✅ | ✅ | ✅ | Geurts et al. 2006 |
| GradientBoosting | ❌ | ✅ | ❌ | ⚠️ | ✅ | ⚠️ | Friedman 2001 |
| AdaBoost | ❌ | ✅ | ⚠️ | ⚠️ | ✅ | ⚠️ | Freund & Schapire 1997 |
| XGBoost | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | Chen & Guestrin 2016 |
| LightGBM | ❌ | ✅ | ❌ | ✅ | ✅ | ✅✅ | Ke et al. 2017 |
| CatBoost | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | Prokhorenkova et al. 2018 |

**Legend:** ✅=bueno / ⚠️=moderado / ❌=malo

## Clustering

| Modelo | Forma clusters | Escalable | n_clusters | Outliers | Paper |
|--------|---------------|-----------|------------|----------|-------|
| KMeans | Esféricos | ✅ | Requerido | ❌ | Lloyd 1982 |
| Agglomerative | Cualquiera (con linkage) | ⚠️ | Requerido | ❌ | |
| DBSCAN | Arbitraria | ⚠️ | No requerido | ✅ | Ester et al. 1996 |
| GaussianMixture | Elipsoidal | ⚠️ | Requerido | ⚠️ | |
| SpectralClustering | No convexa | ❌ | Requerido | ❌ | |
| Birch | Esféricos | ✅ | Opcional | ⚠️ | Zhang et al. 1996 |

## Redes neuronales

| Modelo | Tabular | Secuencias | Parámetros | Velocidad | Paper |
|--------|---------|------------|------------|-----------|-------|
| MLP | ✅✅ | ❌ | Altos | ⚠️ | |
| CNN1D | ⚠️ | ✅(locales) | Medios | ✅ | |
| LSTM | ❌ | ✅✅(largas) | Altos | ❌ | Hochreiter & Schmidhuber 1997 |
| GRU | ❌ | ✅(largas) | Medios | ⚠️ | Cho et al. 2014 |
| Transformer | ⚠️ | ✅✅(globales) | Muy altos | ❌ | Vaswani et al. 2017 |
| ResNet | ✅ | ❌ | Altos | ⚠️ | He et al. 2016 |

## Reglas prácticas

1. **Siempre empezar con:** LogisticRegression o RandomForest como baseline
2. **Datos < 10K filas:** SVM o GradientBoosting
3. **Datos > 100K filas:** LightGBM o XGBoost
4. **Datos con categóricas:** CatBoost (manejo nativo) o LightGBM
5. **Interpretabilidad necesaria:** LogisticRegression, DecisionTree, o Lasso
6. **Outliers frecuentes:** RandomForest, XGBoost, o IsolationForest en preprocesado
7. **Clusters no esféricos:** DBSCAN o SpectralClustering
8. **Detección de anomalías:** DBSCAN (etiqueta -1) o Autoencoder
