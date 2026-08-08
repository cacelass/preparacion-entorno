# Sistemas de recomendación

## El problema: ranking sobre un catálogo

El problema es ordenar un catálogo para un usuario: devolver la lista de
k items más relevantes. No es clasificación binaria ("¿le gusta o no?"): es
**ranking**. La métrica que importa es la utilidad del orden — si los items
relevantes quedan arriba —, no cuántos pares usuario-item se etiquetan bien.

- **Sesgo de popularidad**: el catálogo real tiene cola larga (pocos items
  concentran la interacción). Un modelo que solo aprende "lo que gusta a la
  media" replica el ranking de popularidad y no personaliza; el ranking por
  popularidad es el **baseline** contra el que se mide todo lo demás.
- **La métrica no es "acierto"**: en un ranking de 10 slots, clasificar
  correctamente pero en el puesto 9 es casi inútil. NDCG y precision@k pesan
  por posición; el acierto plano no distingue un ranking útil de uno inútil.
- Tradeoff central: **precisión (relevancia) vs diversidad/novedad**. Un
  sistema que solo recomienda lo que el usuario ya sabe que le gusta es
  preciso y aburrido; la métrica de negocio (engagement a plazo) decide el
  punto de equilibrio.

## Collaborative filtering: la matriz usuario×item

La matriz $R \in \mathbb{R}^{U \times I}$ con $r_{ui}$ = interacción (rating,
clic, vista). El juego es predecir $r_{ui}$ para las entradas no observadas a
partir de las observadas, asumiendo que usuarios parecidos valoran items
parecidos.

### Factorización: SVD / ALS

$$ \hat{r}_{ui} = b_u + b_i + p_u^\top q_i $$

- $p_u, q_i \in \mathbb{R}^d$: embeddings latentes de usuario e item; $d$
  (típicamente 8-200) es el número de factores.
- $b_u, b_i$: sesgos (usuario generoso, item popular). Absorben la
  popularidad y dejan que el producto $p_u^\top q_i$ capture la *preferencia
  específica*.
- **Objetivo**: minimizar el error en los pares observados

$$ \min \sum_{(u,i)\in\Omega} (r_{ui} - \hat{r}_{ui})^2 + \lambda(\|p_u\|^2 + \|q_i\|^2), $$

  con $\Omega$ el conjunto de entradas vistas.
- **ALS**: alterna fijar $Q$ y resolver $P$ (mínimos cuadrados por fila) y
  viceversa. Cada paso es un sistema lineal con forma cerrada; converge a un
  óptimo local y es paralelizable por filas — el algoritmo clásico en Spark.

### Diferencia con el SVD de álgebra lineal

El SVD de álgebra lineal factoriza **toda** la matriz $R = U\Sigma V^\top$.
Aquí la matriz es **incompleta**: la mayoría de las entradas no se observaron
y, peor, no son cero ni ruido — son "no visto", que no es "no gusta". Por eso
no se aplica SVD directo: imputar 0 a lo no visto destruye el problema
(convierte el ranking en una predicción de popularidad). En CF se factoriza
minimizando el error solo en $\Omega$, nunca rellenando la matriz.

## Feedback implícito

En la mayoría de productos no hay ratings: hay **clics, vistas, tiempo de
reproducción**. La señal es un proxy ruidoso (ver ≠ gustar; no ver ≠ no
gustar) pero abundante, y no se puede tratar como ratings de 1 a 5.

### Weighted matrix factorization

La interacción se codifica con $p_{ui} = 1$ si observada, $0$ si no, y una
**confianza** $c_{ui} = 1 + \alpha\, r_{ui}$ que crece con la intensidad
(repeticiones, tiempo). El objetivo:

$$ \min \sum_{u,i} c_{ui}(p_{ui} - p_u^\top q_i)^2 + \lambda(\|p_u\|^2 + \|q_i\|^2) $$

La suma corre sobre **todas** las entradas (incluidas las no vistas, con
confianza baja), lo que permite factorizar por ALS sobre matrices implícitas:
el costo es $O((U+I)d^2)$ por iteración, no $O(UI)$.

### BPR (Bayesian Personalized Ranking)

BPR (Rendle et al., 2009) reencuadra el problema como **ranking por pares**
en vez de regresión punto a punto: para el usuario $u$ con item positivo $i$
e item no visto $j$, se maximiza la probabilidad $P(i >_u j)$:

$$ \sum_{(u,i,j)} \ln \sigma(\hat{x}_{uij}) - \lambda\|\Theta\|^2, \qquad
   \hat{x}_{uij} = \hat{r}_{ui} - \hat{r}_{uj} $$

- Solo compara un par (positivo, negativo) a la vez; el negativo se muestrea
  (típicamente de los items no observados).
- **Por qué ranking loss**: el objetivo de negocio es el orden, no el valor
  exacto del score. Optimizar error cuadrático sobre valores puntuales no
  penaliza que un item relevante quede en la posición 50; la loss por pares
  sí.
- La muestra del negativo domina el resultado: muestrear de la cola larga o
  ponderar por popularidad cambia el modelo más que la arquitectura.

## Recuperación a dos torres (two-tower / DNN retrieval)

Cuando el catálogo es grande (≫ 10⁶ items), puntuar cada item por usuario en
serve es caro. La arquitectura de dos torres separa la tarea:

- **Torre de usuario** $f(u)$ y **torre de item** $g(i)$: dos MLPs que
  embeben a un espacio común de dimensión $d$.
- Score: producto escalar $\langle f(u), g(i)\rangle$; el producto escalar en
  un espacio latente permite **ANN** (annoy, faiss, HNSW) para recuperar los
  top-k candidatos en ms.
- **Retrieval (candidatos) + ranking (reordenar)**: la primera capa recupera
  ~10²-10³ candidatos baratos y aproximados; la segunda, un modelo más rico
  (features, contexto) reordena esos pocos.
- **Tradeoff retrieval vs precisión**: un retriever más grande (más
  candidatos, embeddings de más dimensión) sube recall a costa de latencia y
  costo. La decisión se mide por separado: **recall@k del retriever** (¿el
  item relevante está entre los candidatos?) vs ganancia del ranking.
- Entrenar con softmax muestreado (sampled softmax) sobre items negativos: no
  se puntúa todo el catálogo, solo un mini-batch de negativos.

## Cold start

Usuarios o items **sin señales** no tienen embeddings significativos. La
factorización falla en el borde del catálogo:

- **Content-based**: usar features del item (categoría, texto, embeddings de
  contenido) para inicializar $q_i$; para usuarios, features demográficas o
  de onboarding. Da cobertura a items nuevos sin interacciones.
- **Popularidad temporal**: en frío total, recomendar lo más popular del
  momento con un sesgo de recencia es el mejor baseline; la exploración lo
  corrige.
- **Exploración**: si siempre se recomienda lo conocido, los items nuevos
  jamás reciben señal (ver sesgos). Reservar un % de slots o de tráfico a
  items/usuarios nuevos; los bandits contextuales formalizan el
  exploit/explore.
- Un modelo sin cold start es un modelo que solo recomienda lo viejo: la
  cobertura de items nuevos es una métrica de producto, no un extra.

## Evaluación offline y online

Offline se particiona el histórico (con **separación temporal**, ver trampas)
y se miden:

| Métrica | Qué responde |
|---------|--------------|
| precision@k | ¿Cuánto de lo recomendado es relevante? |
| recall@k | ¿Cuánto de lo relevante está en el top-k? |
| NDCG@k | ¿Está lo relevante arriba? (penaliza posición) |
| MAP@k | Promedio de precision por usuario, ponderado por posición |

- **El sesgo de evaluar solo lo observado**: lo no visto no es un negativo —
  es incierto (no gusta o no se mostró). Tratar "todo lo no clicado" como
  negativo castiga al sistema por recomendar items que nunca se mostraron.
  Usar datos de *exposición* (qué se mostró) y métricas *as-interactions*
  reduce el sesgo.
- La evaluación offline solo ordena candidatos: no mide engagement real. La
  **evaluación online** (A/B) es la que decide — diseño y análisis en
  [diseno-experimentos.md](diseno-experimentos.md). El paso intermedio: un
  *shadow* (el modelo puntúa en paralelo sin servir) para medir cuánto habría
  cambiado el ranking servido.

## Sesgos y efectos

- **Filtro burbuja**: el modelo refuerza lo que el usuario ya consume; NDCG
  sube mientras la diversidad cae. Medir novedad (¿items nuevos en el top-k?)
  y diversidad (intra-list distance) además de la relevancia.
- **Feedback loops**: el sistema recomienda X → el usuario ve X → el log gana
  masa en X → el siguiente modelo aprende X. El sesgo de popularidad se
  auto-amplifica y la evaluación offline lo ratifica: se evalúa sobre el mismo
  sesgo que se quiere arreglar.
- **Contrarrestos**: debiasing del log (reponderar por exposición o
  propagación); penalizar el score por popularidad; forzar diversidad/novedad
  en el ranking final; exploración explícita.
- Los datos de entrenamiento ya son la salida de un sistema anterior: un
  modelo entrenado sobre clicks pasados aprende a replicar las decisiones del
  recomendador previo, no la preferencia del usuario.

## Serving

En producción el ranking útil se arma en dos etapas:

```python
# serving: retrieval → ranking sobre el catálogo en memoria
from {{ project_slug }}.recommend.serving import build_serving_graph

@build_serving_graph
def recommend(user_id: str, k: int = 10):
    cands = retriever.top_k(user_id, n=200)    # ANN, ~ms
    return reranker.rank(user_id, cands)[:k]    # modelo rico, ~200 pts
```

- **Catálogo en memoria**: los embeddings de items viven en RAM (o índice ANN
  en disco mmap), no en una BD por query. Latencia objetivo: retrieval en
  unidades de ms, ranking en decenas de ms.
- **Cuando un baseline gana**: con datos escasos, popularidad, co-ocurrencia
  (items frecuentemente vistos juntos) o *related items* empatan o superan al
  modelo de CF. El modelo caro solo se justifica cuando el baseline ya no da
  señal: medirlo siempre.
- {% if use_api %}La capa de API expone el servicio de recomendación como un
  endpoint versionado (ver [api.md](../backend/api.md)); {% endif %}el modelo
  servido se versiona junto a los embeddings y el snapshot del catálogo.
- Latencia total = retrieval + ranking + serialización; cachear los top-k por
  usuario popular amortiza el costo cuando el tráfico repite consultas.

## Trampas

- **Usar ratings como confianza**: tratar un clic (o un skip) como un juicio
  de preferencia. El clic dice exposición + interés débil; el skip no es un
  negativo. Modelar la intensidad con confianza (implícito), no como rating.
- **Ignorar el tiempo en los datos**: particionar train/test al azar filtra
  el futuro dentro del entrenamiento (leakage temporal) y mide memoria en vez
  de generalización. Split **por tiempo**: entrenar con lo anterior a T,
  evaluar con lo posterior.
- **Evaluar sobre el mismo sesgo de popularidad que se quiere arreglar**: si
  el histórico está dominado por 20 items populares y se evalúa recall sobre
  esa misma distribución, un modelo que solo recomienda populares "gana". La
  evaluación debe cubrir la cola larga (estratificar los items por
  popularidad) y usar métricas de novedad/diversidad.
- **Confundir el modelo con el sistema**: el ranking servido es modelo +
  filtros + negocio; un fallo de negocio (un filtro que elimina una categoría)
  no es un fallo del modelo, y viceversa.

## Fuentes

- **BPR: Bayesian Personalized Ranking from Implicit Feedback** — S. Rendle,
  C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme (2009). arXiv:1205.2618 —
  https://arxiv.org/abs/1205.2618
- **Matrix Factorization Techniques for Recommender Systems** — Y. Koren,
  R. Bell, C. Volinsky (2009). DOI 10.1109/MC.2009.263 —
  https://doi.org/10.1109/MC.2009.263
- **Collaborative Filtering for Implicit Feedback Datasets** — Y. Hu,
  Y. Koren, C. Volinsky (2008). DOI 10.1109/ICDM.2008.22 —
  https://doi.org/10.1109/ICDM.2008.22
- **Deep Neural Networks for YouTube Recommendations** — P. Covington,
  J. Adams, E. Sargin (2016). arXiv:1606.07792 —
  https://arxiv.org/abs/1606.07792
