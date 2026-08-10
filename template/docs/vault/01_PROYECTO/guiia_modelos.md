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
| SVM (RBF) | ❌ | ✅ | ❌ | ⚠️ | ✅ | ❌ | [[07_REFERENCIAS/svm\|Cortes & Vapnik 1995]] |
| RandomForest | ❌ | ✅ | ⚠️ | ✅ | ✅ | ✅ | [[07_REFERENCIAS/random_forest\|Breiman 2001]] |
| ExtraTrees | ❌ | ✅ | ⚠️ | ✅ | ✅ | ✅ | [[07_REFERENCIAS/extra_trees\|Geurts et al. 2006]] |
| GradientBoosting | ❌ | ✅ | ❌ | ⚠️ | ✅ | ⚠️ | [[07_REFERENCIAS/gradient_boosting\|Friedman 2001]] |
| AdaBoost | ❌ | ✅ | ⚠️ | ⚠️ | ✅ | ⚠️ | [[07_REFERENCIAS/adaboost\|Freund & Schapire 1997]] |
| XGBoost | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | [[07_REFERENCIAS/xgboost\|Chen & Guestrin 2016]] |
| LightGBM | ❌ | ✅ | ❌ | ✅ | ✅ | ✅✅ | [[07_REFERENCIAS/lightgbm\|Ke et al. 2017]] |
| CatBoost | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | [[07_REFERENCIAS/catboost\|Prokhorenkova et al. 2018]] |

**Legend:** ✅=bueno / ⚠️=moderado / ❌=malo

## Clustering

| Modelo | Forma clusters | Escalable | n_clusters | Outliers | Paper |
|--------|---------------|-----------|------------|----------|-------|
| KMeans | Esféricos | ✅ | Requerido | ❌ | [[07_REFERENCIAS/kmeans\|Lloyd 1982]] |
| Agglomerative | Cualquiera (con linkage) | ⚠️ | Requerido | ❌ | |
| DBSCAN | Arbitraria | ⚠️ | No requerido | ✅ | [[07_REFERENCIAS/dbscan\|Ester et al. 1996]] |
| GaussianMixture | Elipsoidal | ⚠️ | Requerido | ⚠️ | |
| SpectralClustering | No convexa | ❌ | Requerido | ❌ | [[07_REFERENCIAS/spectral_clustering\|Ng et al. 2001]] |
| Birch | Esféricos | ✅ | Opcional | ⚠️ | [[07_REFERENCIAS/birch\|Zhang et al. 1996]] |

## Redes neuronales

| Modelo | Tabular | Secuencias | Parámetros | Velocidad | Paper |
|--------|---------|------------|------------|-----------|-------|
| MLP | ✅✅ | ❌ | Altos | ⚠️ | |
| CNN1D | ⚠️ | ✅(locales) | Medios | ✅ | |
| LSTM | ❌ | ✅✅(largas) | Altos | ❌ | [[07_REFERENCIAS/lstm\|Hochreiter & Schmidhuber 1997]] |
| GRU | ❌ | ✅(largas) | Medios | ⚠️ | [[07_REFERENCIAS/gru\|Cho et al. 2014]] |
| Transformer | ⚠️ | ✅✅(globales) | Muy altos | ❌ | [[07_REFERENCIAS/transformer\|Vaswani et al. 2017]] |
| ResNet | ✅ | ❌ | Altos | ⚠️ | [[07_REFERENCIAS/resnet\|He et al. 2016]] |

## Reglas prácticas

1. **Siempre empezar con:** LogisticRegression o RandomForest como baseline
2. **Datos < 10K filas:** SVM o GradientBoosting
3. **Datos > 100K filas:** LightGBM o XGBoost
4. **Datos con categóricas:** CatBoost (manejo nativo) o LightGBM
5. **Interpretabilidad necesaria:** LogisticRegression, DecisionTree, o Lasso
6. **Outliers frecuentes:** RandomForest, XGBoost, o IsolationForest en preprocesado
7. **Clusters no esféricos:** DBSCAN o SpectralClustering
8. **Detección de anomalías:** DBSCAN (etiqueta -1) o Autoencoder
