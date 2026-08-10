# Ingeniería de features

## Valores ausentes

### Deletión

- **Eliminar filas (complete-case)**: válida solo si la ausencia es MCAR
  (Missing Completely At Random). Si la ausencia correlaciona con el target o
  con otras variables, borrar filas sesga la muestra y pierde potencia. Coste:
  descartar observaciones que sí aportan señal.
- **Eliminar la columna**: razonable con tasas de ausencia altísimas (> 90 %)
  y redundancia clara. Antes de borrar, comprueba si la ausencia misma es
  informativa.

### Imputación

Media/mediana es barata y engaña de dos formas:

- **Comprime la varianza**: si solo $n_{obs}$ de $n$ valores se observan y el
  resto se imputan con la media, la varianza muestral se encoge en un factor
  $(n_{obs}-1)/(n-1)$; las correlaciones con esa columna se degradan y los
  errores estándar subestiman la incertidumbre real.
- **Rompe distribuciones**: imputar la media en colas largas atrae outliers
  hacia el centro y distorsiona umbrales (la mediana es más robusta).

Alternativas según el patrón:

| Método | Cuándo | Riesgo |
|--------|--------|--------|
| Media / mediana | tasa baja, MCAR, para modelos lineales | comprime varianza |
| MICE (ecuaciones encadenadas) | patrón arbitrario, MCAR/MAR | caro, respetar estructura |
| KNN (vecinos) | numéricas correlacionadas | escala mal con $p$ grande |
| Model-based | otras columnas predicen el hueco | leakage si usas el target |

Regla: **imputa con el mecanismo, no con la media**. Si $X_2$ predice $X_1$
faltante, un modelo (o KNN/MICE) aprovecha esa estructura; la media la ignora.

### Missing Not At Random (MNAR)

Cuando la ausencia depende del valor ausente (un sensor que no reporta justo
cuando la lectura se dispara), ninguna imputación basada en lo observado es
correcta: el dato no está, y el hecho de que no esté es informativo. Se usa el
**indicador de ausencia** (columna binaria `is_missing`) además del valor
imputado.

{% if use_xgboost or use_lightgbm or use_catboost %}
### Con árboles en este proyecto

XGBoost, LightGBM y CatBoost aprenden de forma nativa el mejor split para NaN:
no necesitan imputar antes. Dejar el hueco como NaN suele ganar a imputar con
media/mediana, porque el split puede separar "faltante" como rama propia.
Regla: con gradient boosting, **prueba primero sin imputar** (NaN nativo +
indicador de ausencia); imputa solo si el modelo lineal de la competencia lo
exige o el NaN nativo degrada la convergencia.
{% endif %}

Para modelos lineales (regresión, SVM, redes) el NaN nativo no existe: imputa
sí o sí, y decide entre media y mediana en validación, no por intuición.

## Codificación de categóricas

### One-hot y la maldición de la cardinalidad

One-hot convierte $k$ categorías en $k$ columnas binarias. Bien con $k$
pequeño; con $k$ alto (miles de códigos, IDs) explota: memoria, esparsidad y
columnas con pocos positivos que son puro ruido. Límite práctico: si
$k > n/10$, sospecha.

Reglas:

- **Ordinal si existe orden**: nivel educativo, severidad, rango. Si el orden
  es monótono con el target, una columna entera captura la tendencia con 1
  dimensión en vez de $k-1$. Si el orden es dudoso, trata como nominal.
- **Nominal y $k$ bajo**: one-hot / dummies.
- **Nominal y $k$ alto**: mira las alternativas de abajo.

### Target / mean encoding

Sustituye la categoría por la media (o tasa) del target dentro de ella. Muy
potente y muy traicionera:

- **Trampa de leakage**: calcular la media con las mismas filas que luego se
  entrenan infla la métrica de forma brutal. La media se computa **solo con el
  train** (fit de un transformador sobre el split) y se aplica a validación y
  test con los valores aprendidos.
- **Sobreajuste en categorías raras**: una categoría con 2 muestras tiene una
  media con varianza altísima. Se corrige con suavizado hacia la media global:

$$\hat{m}(c) = \frac{n_c \cdot \bar{y}_c + \lambda \cdot \bar{y}}{n_c + \lambda},$$

- con $\lambda$ el peso del prior global. Alternativa robusta: **out-of-fold**
  (calcular $\hat{m}(c)$ con CV interna del train, nunca con la fila actual) o
  leave-one-out con ruido. Nunca uses el valor de la propia fila.

### Otras codificaciones

- **Frequency encoding**: sustituye por la frecuencia de la categoría. Captura
  "lo común vs lo raro" sin mirar el target; sin leakage directo, pero pierde
  la relación con el target.
- **Hashing**: mapea la categoría a un vector hash de longitud fija. Acota la
  dimensionalidad sin tabla de mapeo; útil con cardinalidades enormes, y las
  colisiones son aceptables si el vector es lo bastante largo.
- **Embeddings**: representaciones densas aprendidas (o pre-entrenadas) para
  cardinalidades altas. Caro; solo cuando la señal de la categórica es grande
  y $n$ lo permite.
- **Native categorical**: CatBoost (nativo) y LightGBM/XGBoost con
  `enable_categorical`/hist codifican con target stats internas y control de
  overfit. Si ya usas árboles, prueba esto antes que target encoding manual.

## Transformaciones numéricas

### Escalado

| Escalado | Definición | Cuándo |
|----------|-----------|--------|
| Standard | $(x - \mu)/\sigma$ | basados en distancia o gradiente: SVM, KNN, redes, regresión |
| Robust   | $(x - \mathrm{med})/\mathrm{IQR}$ | colas u outliers |
| MinMax   | $(x - x_{\min})/(x_{\max} - x_{\min})$ | rangos acotados, salida en $[0,1]$ |

- **Árboles (XGBoost, LightGBM, CatBoost, RF) son invariantes a la escala**:
  un split umbral no cambia al reescalar. Escalarles es coste sin beneficio.
- **SVM, KNN, PCA, regresión lineal y redes NO son invariantes**: sin
  escalado, las columnas de mayor rango dominan la distancia o el gradiente.
  Standard (o Robust con outliers) **ajustado sobre el train y aplicado a
  validación/test**, dentro del pipeline, nunca sobre el split completo.

### Asimetría (skew) y outliers

- **log**: $x \to \log(x + c)$ para magnitudes positivas con cola larga
  (ingresos, conteos). Comprime la cola y hace efectos multiplicativos casi
  aditivos.
- **Box-Cox**:

$$y^{(\lambda)} = \frac{y^\lambda - 1}{\lambda}, \quad \lambda \neq 0, \qquad
y^{(0)} = \log y,$$

  con $\lambda$ estimado por máxima verosimilitud sobre el train
  (`PowerTransformer(method="box-cox")`); requiere positividad estricta.
  Yeo-Johnson si hay ceros o negativos.
- **Winsorizing/clipping**: recortar valores fuera de un cuantil (p. ej.
  1 %–99 %) en vez de borrar la fila. Protege a los modelos sensibles a
  outliers sin perder observaciones.
- **Binarización**: 0/1 por umbral cuando importa la presencia y no la
  magnitud (¿compró antes?, ¿accedió?).

El error común: transformar y escalar **antes** del split, filtrando
información del test hacia el train. Igual que en imputación y target
encoding, todo transformador se ajusta en el train.

## Construcción de features

### Interacciones y polinomios

$x_i \cdot x_j$ y potencias $x_i^2$ capturan no linealidades que un modelo
lineal no ve. Coste: **explosión combinatoria** — con $p$ features hay
$p(p-1)/2$ interacciones de grado 2. Los árboles capturan interacciones
nativas (splits anidados); para lineales, sé selectivo: interacciones guiadas
por dominio, no el producto cartesiano.

### Features de dominio y de ratio

- **Ratios**: $a/b$ suele resumir mejor que dos columnas (velocidad =
  dist/tiempo, densidad = recuento/área). Cuidado con $b = 0$: define el ratio
  con NaN o un sentinel explícito.
- **Temporales**: recencia (días desde el último evento), frecuencia (cuántas
  veces en la ventana), monotónicas (días desde el inicio, edad). Suelen
  aportar más que el timestamp crudo.

### Agregados de ventana

Rolling mean/std codifica tendencia y volatilidad:

```python
df["sales_7d_mean"] = df.groupby("store")["sales"].transform(
    lambda s: s.rolling(7, min_periods=1).mean()
)
```

**Trampa en series temporales**: si la ventana incluye el valor futuro o el
presente (sin `shift`), el modelo "ve" el futuro: en validación temporal
aparece como métrica excelente que se derrumba en producción. En datos
temporales, los agregados se calculan **solo con pasado**, con `shift`, y la
validación es walk-forward.

### Texto

- **TF-IDF**: frecuencia de término por inversa de frecuencia documental;
  representación esparsa y barata, sigue funcionando bien.
- **n-grams**: pares/ternas de tokens capturan negaciones ("no me gusta").
  Aumentan la dimensionalidad; filtra por frecuencia mínima.
- **Embeddings**: densos y semánticos, pero costosos; sin señal de texto
  relevante aportan poco frente a TF-IDF. Para texto corto, un embedding de
  frase + promedio suele bastar.

### Joins externos

Enriquecer con datos externos (clima, censo, calendario) suele ser lo que más
mejora un modelo real. Dos riesgos:

- **Leakage**: si el join trae información del futuro (clima del día de
  predicción, dato publicado después de la fecha de corte), el modelo engaña.
- **Drift**: la tabla externa cambia entre entrenamiento y producción. Fija la
  fecha de corte, documenta fuente y versión, y monitoriza el drift.

## Selección de features

| Familia | Cómo | Ejemplos | Riesgo |
|---------|------|----------|--------|
| Filter  | métrica de cada feature por separado | varianza, mutual information | ignora interacciones |
| Wrapper | evalúa subconjuntos con el modelo | RFE, selección hacia atrás | caro, overfit al espacio |
| Embedded| selección dentro del entrenamiento | L1, gain, permutation importance | el más práctico |

- **L1**: empuja pesos a cero; las features con peso 0 se descartan. Barato y
  dentro del modelo.
- **Gain de árboles**: suma de la reducción de impureza por feature; rápido,
  pero sesgado hacia cardinalidad alta y splits repetidos.
- **Permutation importance**: caída de la métrica al permutar la feature; la
  más honesta de las embedded, y solo es válida calculada sobre validación.

**Leakage de selección**: elegir features con todo el dataset y luego dividir
es selección-antes-de-split — el mismo error que imputar o escalar antes del
split. La selección se hace con validación interna (o con el train del fold
exterior en nested CV), nunca sobre el conjunto completo.

**Por qué menos features gana**: cada feature extra añade ruido de estimación;
en árboles, un split basado en una feature débil o duplicada puede empeorar el
modelo por debajo de no dividir (sesgo del split). Quitar features
correlacionadas y ruidosas suele mejorar generalización y tiempo.

## Bucle práctico

Iteración corta y medible:

1. **Hipótesis**: "la recencia de la última compra predice el churn".
2. **Construir**: la feature como código reutilizable del pipeline, no en un
   notebook suelto.
3. **Validar con CV**: métrica honesta, misma partición que el resto del
   pipeline, sin tocar el test.
4. **Conservar solo si mejora**: la feature queda si supera un umbral de
   mejora (p. ej. +0.001 de AUC) de forma estable en los folds; si no, se
   descarta sin nostalgia.

Reglas de seguridad:

- No añadas decenas de features de golpe: si son correlacionadas, la ganancia
  es ilusoria (multicolinealidad, splits redundantes) y la causa no se
  atribuye. Una feature a la vez, o un grupo por experimento.
- Mide la correlación entre features nuevas y existentes; una "nueva" que
  duplica una vieja no es una feature, es ruido de entrenamiento.
- El bucle entero —construcción, imputación, escalado, selección— va dentro de
  un Pipeline validado por CV; fuera de él, cada paso es un punto de leakage.

{% if use_xgboost or use_lightgbm or use_catboost %}
**Con árboles en este proyecto**: no escales features (innecesario), deja los
NaN nativos, prefiere native categorical antes que target encoding manual, y
usa gain/permutation importance del modelo final para podar features antes de
reentrenar. La regla de "menos features gana" aplica igual.
{% endif %}

## Fuentes

- Chen, T. y Guestrin, C., *XGBoost: A Scalable Tree Boosting System*, KDD
  2016. arXiv:1603.02754. https://arxiv.org/abs/1603.02754
- Ke, G. et al., *LightGBM: A Highly Efficient Gradient Boosting Decision
  Tree*, NeurIPS 2017. arXiv:1703.01952. https://arxiv.org/abs/1703.01952
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. y Gulin, A.,
  *CatBoost: Unbiased Boosting with Categorical Features*, NeurIPS 2018.
  arXiv:1706.09516. https://arxiv.org/abs/1706.09516
- Altmann, A., Toloşi, L., Sander, O. y Lengauer, T., *Permutation Importance:
  A Corrected Feature Importance Measure*, Bioinformatics 2010. arXiv:1805.01455.
  https://arxiv.org/abs/1805.01455
