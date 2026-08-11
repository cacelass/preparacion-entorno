# Modelos fundacionales (FMs)

Modelos preentrenados a escala —LLM, visión, multimodales— que se adaptan a una
tarea sin entrenar desde cero. Este fichero cubre el ciclo completo de un FM en
un proyecto: qué es el pre-training, cómo se adapta (prompting, RAG,
fine-tuning, LoRA), cómo se evalúa de verdad y cuánto cuesta. Complementa a
`llms-aplicados.md` (uso y prompting) y a `compresion-modelos.md` (cuantización/
destilación); aquí el foco es la **economía y la decisión** de usar o adaptar un
FM, y sus modos de fallo.

## El paradigma: pre-training + adaptación

Un FM se entrena **una vez** (caro, semanas de GPU) sobre corpus enormes con un
objetivo de auto-supervisión (predecir el siguiente token, enmascarar, contraste
entre modalidades). El resultado es un modelo de propósito general. Luego se
**adapta** a una tarea concreta pagando mucho menos:

| Estrategia | Qué cambia | Coste | Cuándo |
|------------|-----------|-------|--------|
| Prompting / ICL | Nada (solo el prompt) | ~0 | Tareas simples, pocos ejemplos |
| RAG | Nada (contexto externo) | ~0 | Tareas factuales, conocimiento que cambia |
| Fine-tuning parcial | Últimas capas | Bajo | Estilo/formato, dominio |
| LoRA / adapters | Low-rank updates | Bajo | Tarea con datos moderados |
| Full fine-tuning | Todo | Alto | Dominio muy distinto, muchos datos |

La decisión clave no es "¿cuál es el mejor modelo?" sino **"¿cuánto necesito
adaptarme y con qué presupuesto?"**. En orden ascendente de coste de adaptación:
prompting → RAG → fine-tuning parcial → LoRA → full fine-tuning. La regla
práctica: **empieza con la adaptación más barata que resuelva la tarea**; solo
sube de nivel cuando hay evidencia de que la actual no alcanza.

## Qué significa "modelo abierto" y por qué importa

- **Open weights** (pesos abiertos): puedes descargarlos, servirlos tú,
  auditar; la licencia decide uso comercial y reentrenamiento.
- **Closed API**: conveniencia y calidad, pero dependes del proveedor (precio,
  disponibilidad, cambios de comportamiento), y los datos que envías salen de tu
  control (ver `privacidad-y-fuga-datos.md`).
- **True open-source** (pesos + datos + código): raro en FMs de frontera; casi
  todo lo "open" es open-weights.

Esta elección condiciona el proyecto entero: coste por inferencia, latencia,
cumplimiento y portabilidad. Si los datos son sensibles, una API cerrada puede
estar descartada desde el día uno por política.

## Anatomía de un LLM (lo mínimo para hablar el idioma)

- **Tokenización**: el texto se parte en tokens (subpalabras, BPE/WordPiece).
  "1 token ≈ 0.75 palabras en inglés ≈ 0.4-0.5 en español/otros". El límite de
  contexto se cuenta en tokens, no en palabras.
- **Context window**: cuánto contexto cabe (2k → 200k+ según el modelo). El
  contexto no es memoria: el modelo atiende a todo, pero "lo relevante" se
  diluye con la distancia y el ruido (lost-in-the-middle).
- **Arquitectura**: transformer con attention (ver `redes-neuronales.md` y la
  fuente de `sources.json`); los FMs modernos usan MoE (mixture-of-experts) y
  atención local/esparsa para escalar con coste fijo por token.
- **Reasoning**: los modelos de "pensamiento" (chain-of-thought explícito,
  hidden reasoning) gastan tokens de razonamiento antes de responder; ganan en
  tareas de lógica y pierden en latencia y coste.

## Pre-training a grandes rasgos

No se hace en un proyecto típico (salvo que el proyecto *sea* eso). Pero
conviene conocer las tres fases porque el vocabulario aparece en todo:

1. **Pre-training**: objetivo auto-supervisado sobre corpus masivo; aprende
   representaciones generales. Escala de $10^{12}$–$10^{14}$ tokens.
2. **SFT (supervised fine-tuning)**: alinear con ejemplos de instrucción/QA
   escritos por humanos; convierte el modelo "next-token" en un asistente.
3. **RLHF / DPO**: optimizar preferencias humanas (o un reward model) para que
   las respuestas sean preferidas, útiles y menos dañinas. DPO evita el reward
   model entrenando directamente con pares preferido/no-preferido.

Los "capability jumps" entre versiones suelen venir del **esfuerzo de
post-entrenamiento** (SFT+RLHF con calidad de datos), no solo de más parámetros.

### Scaling laws (la economía del tamaño)

Los resultados empíricos (Kaplan, Chinchilla) dicen que el error de pre-training
decae como una ley de potencia en el número de parámetros $N$ y de tokens $D$:

$$\text{loss}(N, D) \approx A N^{-\alpha} + B D^{-\beta} + E.$$

- Doblar los parámetros o los tokens mejora, pero con rendimientos decrecientes
  ($\alpha, \beta \approx 0.1$-$0.4$).
- **Chinchilla** muestra que, para un presupuesto fijo de cómputo, hay un punto
  óptimo de *balance* entre $N$ y $D$ (≈ 20 tokens por parámetro), no "más
  grande siempre".
- Consecuencia práctica: el modelo "mejor para tu tarea" no es el más grande —
  un modelo más pequeño con buena adaptación y RAG puede superar a uno gigante
  con coste y latencia mucho menores.

## Adaptación: LoRA y el coste marginal

**LoRA** congela el modelo y entrena una descomposición de bajo rango del
delta de pesos:

$$W' = W_0 + \frac{\alpha}{r} BA, \qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k).$$

Solo se actualizan $A$ y $B$:
- Parámetros entrenables típicamente $< 1\%$ del modelo.
- Se puede servir el modelo base + el adapter con un coste de memoria marginal
  (los adapters pesan MB frente a GB del base).
- Hiperparámetros que importan: el rango $r$ (capacidad del adapter; demasiado
  pequeño no aprende, demasiado grande sobreajusta), el escalado
  $\alpha \approx r$ como punto de partida, el learning rate del adapter, y
  **qué capas** se adaptan (suele bastar con attention).

**Full fine-tuning** actualiza todo: mayor capacidad de adaptación pero mucho
más coste y riesgo de **catastrophic forgetting** (olvida lo que ya sabía).
Si solo tienes cientos de ejemplos, fine-tuning completo de un modelo grande
suele sobreajustar; LoRA o RAG/prompting rinden más.

### RAG vs fine-tuning: el marco de decisión

| Pregunta | RAG | Fine-tuning |
|----------|-----|-------------|
| ¿El conocimiento es factual y cambia? | **Sí** (retrieval al día) | No (queda congelado) |
| ¿Necesitas citar fuentes? | **Sí** | No |
| ¿La tarea es de estilo/formato? | No | **Sí** |
| ¿Hay que adaptar a vocabulario de dominio? | Parcial (contexto) | **Sí** |
| Coste de mantenimiento | Bajo | Alto (re-entrenar) |
| Latencia | +retrieval | ~igual |

La combinación es la normal en producción: **RAG para conocimiento + LoRA para
comportamiento** (formato, tono, esquema). No es "o uno o el otro".

### QLoRA: LoRA sobre un modelo cuantizado

**QLoRA** (Dettmers et al., 2023) afina el mismo LoRA pero sobre un base
cuantizado a **4 bits**: el adapter vive en bf16 y solo se actualiza él; los
pesos del base se descuantizan on-the-fly al hacer el forward. Un modelo de
65B cabe en una sola GPU (~24-48GB) y el coste de memoria baja a ~3-4 bits por
parámetro frente a los 16 de LoRA, con calidad cercana al fine-tune completo en
los benchmarks del paper. Tres piezas:

- **NF4 (Normal Float 4)**: cuantización por **bloques de 64 pesos** con un
  factor de escala por bloque. Aprovecha que los pesos entrenados se
  distribuyen ~$\mathcal{N}(0, \sigma)$: asigna los 4 bits para maximizar
  precisión donde hay más densidad de masa, no de forma uniforme como INT4.
  El escalado por bloque (en vez de por tensor) reduce el error cuando una
  columna tiene outliers.
- **Double quantization**: los factores de escala NF4 (FP32) se cuantizan a su
  vez a FP8, ahorrando ~0.37 bits/param adicionales. Pequeño, pero de gratis.
- **Paged optimizers**: el estado del optimizador (Adam) se pagina a la CPU
  cuando la VRAM se llena, como hace un sistema operativo con la RAM —
  permite batch sizes que de otro modo harían OOM.

**Cómo se rompe**:

- **No es para modelos pequeños**: la pérdida de la cuantización puede pesar
  más que el ahorro; el beneficio escala con el tamaño del base.
- **No acelera, solo ahorra memoria**: la descuantización on-the-fly añade
  cómputo. Si la VRAM sobra, LoRA en bf16 directo puede ser más rápido.
- **Medir en la tarea, no en perplexity**: NF4 degrada distinto según capas y
  pesos grandes; la perplexity del corpus no dice cómo responde a tu tarea.
- La cuantización para **inferencia** (AWQ/GPTQ, PTQ/QAT) es otro mundo y vive
  en `compresion-modelos.md`; QLoRA es cuantización para **entrenamiento**.
- La matemática de rango bajo común a LoRA/QLoRA está en
  `matematicas/matrices-app.md`.

### aLoRA: activación por tokens de invocación

LoRA aplica el adapter a **todos** los tokens. Eso significa que al cambiar de
adapter hay que rehacer el prefill del contexto entero — con un RAG de 50k
tokens, cada cambio de especialista recuenta el coste. **aLoRA** (Activated
LoRA, Greenewald et al., IBM, 2025) modifica el marco para que el adapter solo
se aplique a los tokens **en y después** de una *secuencia de invocación*
(fuera de los límites del modelo), dejando intactos los Q, K, V del contexto
anterior:

$$Q = \begin{bmatrix} X_{1:t_{inv}-1} W_Q \\ X_{t_{inv}:t}\,(W_Q + \Delta Q) \end{bmatrix},
\quad K = \begin{bmatrix} X_{1:t_{inv}-1} W_K \\ X_{t_{inv}:t}\,(W_K + \Delta K) \end{bmatrix},
\quad V = \begin{bmatrix} X_{1:t_{inv}-1} W_V \\ X_{t_{inv}:t}\,(W_V + \Delta V) \end{bmatrix}$$

Como antes de la invocación los pesos son los del base, las keys y values del
contexto previo son idénticas entre base y adapter ($K^{adapter}_{1:t_{inv}-1} =
K^{base}_{1:t_{inv}-1}$ y lo mismo para $V$, proposición formal del paper). El
adapter **acepta el KV cache del base** para todo lo anterior a la
invocación: cambiar de especialista no re-prefillea el contexto, solo el
fragmento nuevo. Esto habilita los *intrinsics*: especialistas invocados bajo
demanda para una operación bien definida sobre un trozo del hilo
(verificar un formato, puntuar confianza, detectar alucinación), mientras el
resto del hilo lo genera el base.

- **Implementación**: está en HF PEFT (`alora_invocation_tokens`); la
  reutilización real del prefijo en un servidor de inferencia exige alinear el
  *prefix caching* entre base y adapters (vLLM, arXiv:2512.17910) — no sale
  gratis de activar el flag.
- **Tokenización**: la secuencia de invocación debe ser delimitada por tokens
  especiales, o la tokenización la "absorbe" dentro de un token mayor y la
  invocación falla silenciosamente.
- **Costo en calidad**: aLoRA suele necesitar rango $r$ mayor que LoRA (hasta
  32) para rendir igual; en los benchmarks del paper la precisión es
  estadísticamente equivalente a LoRA.

**Cómo se rompe** (y cuándo NO hace falta):

- **No es "infinitos adapters gratis"**: requiere entrenar cada adapter con su
  secuencia de invocación, respetar las condiciones de cache (la invocación
  presente en todo input que lo use) y un servidor que soporte el prefix
  reuse entre modelos. Sin eso, es un LoRA normal con más pasos.
- **Para un sistema sencillo** (chatbot + RAG + un solo especialista) LoRA o
  QLoRA bastan; aLoRA paga cuando hay **varios adapters alternándose sobre el
  mismo contexto** (agentes, multi-especialidad).
- La reutilización del KV cache **solo vale antes de la invocación**: después
  de activarse, el adapter tiene su propio cache, como cualquier modelo.

## Evaluación: el problema sin resolver

La evaluación de FMs es donde más se miente y más cuesta ser honesto:

- **Benchmarks estáticos se saturan**: una vez que los ítems entran en el
  entrenamiento (contaminación), la puntuación deja de medir habilidad.
- **Métrica ≠ tarea**: exact match, F1, BLEU/ROUGE correlacionan mal con
  utilidad real. Para tareas abiertas se necesita **LLM-as-judge** (con sus
  sesgos conocidos: prefiere longitud, posición, su propio estilo) o evaluación
  humana.
- **Lo que importa en producción**: adherencia a la tarea, formato, robustez a
  variaciones del prompt, latencia y coste. Una demo brillante en 5 casos no
  mide eso.
- **Golden set propio**: dskit lo aplica al RAG (`rag_golden.json`); para un FM
  el equivalente es un set curado de la tarea real, con casos adversariales
  incluidos. Sin él, cualquier cambio de modelo/prompt es fe.
- **Evaluar el sistema, no el modelo**: RAG + prompt + post-procesado +
  guardarraíles forman el sistema; el modelo solo es un componente. Evalúa el
  sistema completo con la métrica de la tarea, no el modelo con benchmarks.

## Coste y latencia: la decisión económica

- **Coste por token** (API) o **coste por inferencia** (self-hosted: GPU/hora ×
  tiempo por request). Un modelo 10× más barato que rinde 3% peor en tu tarea
  suele ser la decisión correcta.
- **Latencia**: modelos grandes = más tiempo a primer token (prefill) y por
  token (decode). Para aplicaciones interactivas o de alta concurrencia, modelos
  pequeños con RAG/evocación superan a un FM gigante "de memoria".
- **Caché y batching**: reutilizar respuestas idénticas (caché de prompts),
  agrupar requests (batching), y truncar/segmentar contexto cambia el coste real
  más que elegir el modelo.
- **Modelos pequeños para tareas repetitivas**: routing por intención — un
  modelo barato para el grueso del tráfico y uno grande solo para los casos
  difíciles — es la palanca de coste más infrautilizada.

## Despliegue de un FM propio (self-hosting)

- **Servir**: vLLM / llama.cpp / TGI. La cuantización (ver
  `compresion-modelos.md`) reduce memoria 2-4× con pérdida pequeña si se hace
  bien (AWQ/GPTQ frente a naive).
- **GPU**: el cuello es VRAM (pesos + KV-cache + overhead). Un modelo de 7B en
  FP16 ≈ 14GB de pesos; el KV-cache crece con la concurrencia y el contexto.
- **Especulación**: decodificación especulativa (un modelo pequeño propone,
  el grande verifica) acelera el decode manteniendo la calidad.
- **Monitoreo**: latencia, throughput, y drift del comportamiento (ver
  `ciclo-vida-mlops.md`); un FM self-hosted también puede degradarse con el
  tiempo de uso.

## Cómo se rompe (checklist para el `lider`)

- **Contaminación de benchmarks**: el número de la ficha no es tu métrica.
- **Alucinación no detectable sin evidencia**: en tareas factuales, exigir
  citas/retrieval (RAG) o rechazar responder si no hay fuente — el modelo no
  sabe lo que no sabe.
- **Deriva del proveedor**: una API cerrada cambia de comportamiento sin aviso;
  si es crítico, versionar el prompt/modelo y monitorizar.
- **Prompt fragility**: pequeñas variaciones del prompt cambian la salida;
  evaluar con múltiples redacciones del mismo prompt.
- **Forgetting tras fine-tuning**: verificar que la tarea base no se degradó.
- **Sesgo heredado**: el FM trae los sesgos del pre-training; ver
  `fairness-y-seguridad.md` y `guardarraíles.md`.
- **Mal uso / jailbreak**: si el FM está expuesto, asume entradas hostiles; ver
  `guardarraíles.md`.
- **Contexto mal usado**: meter todo el documento en el prompt "por si acaso"
  degrada la respuesta (lost-in-the-middle) y dispara el coste; el retrieval
  selectivo gana.
- **Bajo presupuesto de evaluación**: cambiar de modelo/prompt sin golden set es
  apostar; el foso está en la evaluación, no en el modelo.

## Dónde encaja en dskit

dskit no entrena FMs por defecto; los usa vía API o self-hosted. Este fichero da
al `lider` el criterio para decidir adaptación (prompt → RAG → LoRA → full FT,
con QLoRA para ajustar con poca VRAM y aLoRA para especialistas activables bajo
demanda), evaluación honesta y coste, y se mantiene con `rag refresh` (topics de
`sources.json`). Cruza con `llms-aplicados.md`, `backend/servir-modelos.md`,
`fairness-y-seguridad.md` y `guardarraíles.md`.
