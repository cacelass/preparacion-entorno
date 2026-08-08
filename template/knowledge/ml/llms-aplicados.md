# LLMs aplicados: cuándo, cómo y por qué

## LLM vs ML clásico: la decisión de arquitectura

La primera pregunta no es "qué modelo", es "¿un LLM es la herramienta
correcta?". La respuesta depende de tres ejes: la forma de los datos, la
semántica de la salida y la operación del sistema.

| Eje | Favor del ML clásico | Favor del LLM |
|-----|----------------------|---------------|
| Datos | Tabular, numérico, con estructura fija | Lenguaje, texto libre, multimodal |
| Salida | Un número, una clase, un score calibrado | Texto, resumen, generación, razonamiento |
| Coste por inferencia | Céntimos por millón, corre en CPU | Dólares por millón de tokens |
| Latencia | Milisegundos estables | 0.5–30 s, crece con la longitud de salida |
| Determinismo | Idéntico para el mismo input | Estocástico; exige seed y temperatura baja |
| Privacidad | Puede vivir 100 % on-prem | Los datos salen al proveedor |
| Mantenimiento | Reentrenar con cada cambio de distribución | Actualizar prompt/RAG sin reentrenar |

Regla de oro: **si la tarea admite reglas, una tabla o una expresión, hazla
determinista.** Un LLM no es un if/else caro ni un reemplazo de un árbol de
decisión: es una máquina de distribución sobre lenguaje. Usarlo para
clasificar tres categorías con 10 M de filas diarias es pagar latencia y coste
para empeorar la calibración.

La decisión correcta es frecuentemente **híbrida**: el LLM para las partes que
exigen comprensión y generación, y código clásico para lo lógico o
determinista. Un clasificador de fraude decide con un gradient boosting; un
LLM redacta la explicación para el cliente. Cada pieza en su capa.

El coste total no es el precio por token: es precio por token × tokens
generados × volumen × latencia asumible. Estimar ese producto antes de
arquitectar evita descubrir el problema al facturar la primera semana.

## Fundamentos operativos

### Tokenización y ventana de contexto

Un LLM no lee caracteres: lee **tokens**, subpalabras producidas por
tokenizadores basados en BPE (byte-pair encoding). El tokenizador parte de un
vocabulario fijo y fusiona iterativamente los pares de bytes más frecuentes,
produciendo tokens de longitud variable.

| Idioma/texto | Tokens aproximados por 1 000 chars |
|--------------|------------------------------------|
| Español/inglés general | 250–350 |
| Código (Python, SQL) | 300–400 (los símbolos se comen tokens) |
| JSON denso | 400–600 (cada clave/llave es un token) |
| CJK | 600–1 500 (un char ≈ 1–2 tokens) |

La **ventana de contexto** es el total de tokens que el modelo puede
considerar a la vez (prompt + completion). Se llena rápido: 8 000 tokens ≈
25 páginas; 128 000 tokens ≈ 400 páginas. Con RAG, cada documento
recuperado, cada histórico de chat y cada instrucción larga compiten por el
mismo presupuesto. Regla operativa: **mide el prompt en tokens reales, no en
"páginas"**, porque el coste y la ventana viven en tokens.

Dos consecuencias prácticas:
- El **prompt largo degrada la calidad** incluso dentro de la ventana
  ("Lost in the Middle", ver Prompting). Contexto no es lo mismo que calidad.
- La **completion come presupuesto de contexto y de coste**: `max_tokens`
  limita la salida, pero el modelo paga por lo que genera. Recortar
  respuestas verbosas es dinero y latencia.

### Sampling: temperatura, top-k, top-p, max_tokens

En cada paso el modelo produce una distribución sobre el vocabulario. El
decoding convierte esa distribución en el siguiente token:

- **Temperatura** $T$: divide los logits por $T$ antes del softmax. $T \to 0$
  concentra la masa en el máximo (casi determinista); $T$ alta suaviza y da
  diversidad. Para repetibilidad real usa greedy y guarda la seed.
- **top-k**: restringe el muestreo a los $k$ tokens más probables.
- **top-p** (nucleus): restringe al conjunto más pequeño cuya probabilidad
  acumulada supera $p$. Más suave que top-k.
- **max_tokens**: techo duro de la completion; si la respuesta lo corta,
  devuelves un fragmento truncado sin señal de que lo esté.

```python
import math, random

def sample(logits, temperature=1.0, top_p=1.0, top_k=50):
    logits = [l / temperature for l in logits]
    m = max(logits)
    probs = [math.exp(l - m) for l in logits]        # softmax estable
    z = sum(probs)
    probs = [p / z for p in probs]
    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    ranked = ranked[:top_k]
    cum, cut = 0.0, len(ranked)
    for idx, i in enumerate(ranked):                 # truncar por top-p
        cum += probs[i]
        if cum > top_p:
            cut = idx + 1
            break
    ranked = ranked[:cut]
    weights = [probs[i] for i in ranked]
    return random.choices(ranked, weights=weights)[0]
```

Para producción: **temperatura baja (0–0.3)** si la tarea es precisa
(extracción, clasificación, código); alta solo para ideación o parafraseo.
Guarda la seed y registra los parámetros de sampling junto a cada respuesta:
sin eso, un fallo intermitente es irreproducible.

### El coste por token: prompt vs completion

Las tarifas distinguen dos flujos: los tokens de entrada (prompt) y los de
salida (completion), que cuestan 3–10× más en la mayoría de proveedores. El
presupuesto se optimiza en el prompt (menos tokens de entrada, caché) y en la
salida (menos verbosidad, `max_tokens` ajustado).

| Factor | Efecto en factura |
|--------|-------------------|
| Prompt con toda la historia del chat | Crece por cada turno; se repaga en cada llamada |
| RAG sin recortar | Recupera 10 documentos, usa 3: pagas 10 de entrada cada vez |
| `max_tokens` generoso | Paga por salida que nadie lee |
| Retries sobre error transitorio | Duplica el coste sin necesidad (ver Producción) |

El prompt fijo (instrucciones, few-shot) no se repite si el proveedor soporta
**caché de prompt** (los tokens cacheados cuestan ~10× menos). Estructurar el
prompt en bloques estables y variables facilita esa caché.

## Prompting

### Estructura system / user / assistant

| Rol | Función |
|-----|---------|
| system | Instrucciones estables: rol, reglas, formato de salida, restricciones |
| user | La consulta concreta, los datos del turno |
| assistant | Respuestas previas del modelo; en few-shot, los ejemplos |

El `system` es para lo que no cambia entre llamadas (política, formato,
tono); el `user` para lo que cambia (consulta, contexto). Meter contexto
volátil en system rompe la caché de prompt y confunde la jerarquía. Las
respuestas `assistant` en few-shot muestran el formato de salida esperado; no
se usan para conversar.

### Few-shot: cuántos ejemplos y cuáles

Los ejemplos en contexto (in-context learning) enseñan el formato, el tono y
el estilo de razonamiento sin tocar los pesos:

- **Cuántos**: típicamente 3–5. Pasar de 0 a 3 ejemplos da la mayor mejora;
  más allá de ~8 el rendimiento se estabiliza o cae (los ejemplos compiten
  con la ventana y el modelo "encaja" el ruido).
- **Cuáles**: representativos del caso real, con la salida que *quieres*, no
  la que un ejemplo mal escrito produciría. Un ejemplo ambiguo fija el estilo
  equivocado durante todo el prompt.
- **Variedad > cantidad**: cubrir los casos límite (borde, negativo, formato
  raro) enseña más que cinco variaciones del caso típico.
- Si los ejemplos fallan, selecciónalos dinámicamente: con embeddings, elige
  los k ejemplos más cercanos a la consulta (kNN sobre el set de ejemplos).

```text
system:  Convierte fechas de texto a ISO-8601. Solo responde la fecha.
user:    El pedido llega el 3 de enero del 24.
assistant: 2024-01-03
user:    Mañana a las diez de la noche
assistant: <fecha ISO>
```

### Chain-of-thought: cuándo ayuda y cuándo no

Pedirle al modelo que razone paso a paso antes de responder mejora
sistemáticamente las tareas que exigen varias operaciones (aritmética,
multi-saltos lógicos, extracción condicional). Es barato de activar ("razona
paso a paso") pero costoso de ejecutar: cada paso son tokens de salida
pagados.

- **Ayuda**: razonamiento multi-paso, comprensión, decisiones con condiciones
  encadenadas, tareas donde el error está en la combinación de pasos.
- **No ayuda**: tareas de una sola mirada (clasificación simple, extracción
  directa, formateo), donde añade tokens sin aportar precisión — y en
  latencia sensible, duplica o triplica el tiempo de respuesta.
- **Variante**: self-consistency — muestrear varias cadenas ($T$ moderada) y
  votar la respuesta mayoritaria. Mejora robustez a coste multiplicativo;
  útil para decisiones caras, no para cada llamada.
- **Riesgo**: el razonamiento puede ser una *racionalización* (el modelo
  argumenta bien una respuesta errónea). No se valida leyendo el CoT; se
  valida contra la salida final sobre un golden.

### El orden importa: "Lost in the Middle"

Con contextos largos, el modelo atiende **desproporcionadamente al principio
y al final** del prompt y pierde lo del medio (Liu et al., 2023).
Implicaciones prácticas:

- Lo que no debe perderse (la instrucción central, el formato de salida, la
  pregunta) va al **principio** o al final del prompt; nunca enterrado en el
  medio.
- Los documentos recuperados por RAG se ordenan por relevancia, no por fecha:
  el top-1 arriba, los accesorios abajo. Reordenar por utilidad es una de las
  mejoras más baratas de un pipeline RAG.
- Si un fragmento es crítico y la ventana lo permite, repetirlo en dos
  posiciones ayuda: el coste de duplicarlo es menor que el riesgo de perderlo.

### Instrucciones negativas

Las órdenes directas ("escribe una tabla markdown") funcionan mejor que las
prohibiciones ("no escribas prosa"), porque la negación obliga al modelo a
inferir el comportamiento complementario. Pero la negación sí vale como
**límite** ("no inventes citas", "no inventes datos"). Estructura: primero la
instrucción positiva del formato deseado, después el límite negativo. Las
listas largas de prohibiciones se diluyen y compiten con las positivas.

## Evaluación

Evaluar un LLM con "se ve bien" no es evaluación. La regla: **un golden fijo,
métricas con referencia o sin ella, y un número reproducible por commit**.
Como en cualquier sistema ML, el modelo y el prompt son hipótesis sobre un set
de evaluación.

### Con referencia

Cuando existe una salida esperada (golden):

| Métrica | Mide | Límites |
|---------|------|---------|
| BLEU | Solapamiento de n-gramas con referencia | Castiga sinónimos válidos; no entiende semántica |
| ROUGE | Recall de n-gramas/LCS con referencia | Favorece respuestas largas; malo en resúmenes divergentes |
| Embeddings | Coseno entre vectores de respuesta y ref. | Capta parafraseo; opaco al razonamiento |

BLEU y ROUGE son útiles para generación cercana a la referencia
(transcripción, extracción, resumen extractivo) y engañosas para generación
abierta: dos respuestas correctas pueden no compartir n-gramas. La similitud
por embeddings cubre el parafraseo pero tiene techo: no distingue entre
"correcto pero reformulado" e "incorrecto pero elocuente".

### Sin referencia: LLM-as-judge

Cuando no hay golden (resumen abierto, respuesta a una pregunta, crítica), el
juez es otro LLM que puntúa con una rúbrica. Procedimiento:

1. **Rúbrica explícita**: criterios y escala (1–5) escritos en el prompt del
   juez, no "¿es buena respuesta?".
2. **Salida estructurada**: el juez emite JSON con puntuación por criterio y
   una justificación breve.
3. **Una dimensión por criterio**, no una nota global.

Sesgos documentados del juez LLM (Zheng et al., 2023):

- **Autocomplacencia** (self-enhancement): puntúa mejor al modelo del que
  procede (si juez y candidato son el mismo modelo).
- **Sensibilidad al orden**: el candidato que aparece primero recibe
  ventaja; barajar posiciones y promediar dos pasadas.
- **Verbosidad** (length bias): las respuestas largas puntúan mejor aunque
  sean peores; controlar por longitud en el análisis.
- **Sobre-referencias**: el juez premia citar "investigación" inventada.

Mitigaciones baratas: juez de un proveedor distinto al modelo evaluado,
presentar los candidatos en orden aleatorio (2 pasadas) y exigir la rúbrica en
el prompt del juez. Un juez sesgado pero *consistente* sirve para detectar
regresiones entre commits; no confíes en la nota absoluta.

### Evaluar sobre un golden fijo

El golden vive en `tests/eval/` (o `data/eval/`): pares consulta→respuesta
esperada, o criterios de aceptación anotados. El pipeline de evaluación corre
en CI y falla si la métrica baja de umbral:

```python
def evaluate(golden, model_fn, metric):
    scores = []
    for item in golden:
        pred = model_fn(item["prompt"])
        scores.append(metric(pred, item["expected"]))
    return {"mean": sum(scores) / len(scores), "p95": sorted(scores)[-1]}
```

Reglas: **el golden no se cambia a ojo para que el modelo apruebe** (si
cambia, se documenta y se re-benchmarka todo), y el set de evaluación se
separa de cualquier set usado para few-shot, para que los ejemplos no
"entrenen" la evaluación.

## RAG bien construido

RAG (Retrieval-Augmented Generation) inyecta conocimiento externo en el
contexto para reducir alucinaciones y actualizar conocimiento sin reentrenar.
Calidad del sistema = calidad de recuperación × calidad de generación; un
recuperador mal construido hunde al mejor generador.

### Chunking que respete la semántica

El grano del texto indexado determina lo que el modelo puede "ver" y citar:

- **Fragmentos cortos** (100–300 tokens): precisos para recuperar el dato,
  pero pierden contexto global y el modelo responde sin el marco.
- **Fragmentos largos** (500–1 000 tokens): más contexto, recuperación menos
  precisa (se cuela el ruido del párrafo), más coste por token de entrada.
- Regla: **partir por límites semánticos** (párrafos, secciones, bloques de
  código) y no por longitud a ciegas; solapar ventanas (~15–20 %) para no
  cortar la idea por la mitad.

### Embeddings + re-ranking (cross-encoder)

La recuperación en dos etapas es el estándar:

1. **Embeddings bi-encoder**: codifica consulta y documentos en vectores y
   busca por coseno (ANN). Rápido, pero la similitud de vectores no entiende
   intención: dos documentos pueden ser semánticamente cercanos y responder
   cosas distintas.
2. **Re-ranking con cross-encoder**: el top-N recuperado por vectores (p. ej.
   50) se re-escora con un modelo que codifica (consulta, documento) juntos,
   capturando la interacción. Se quedan los top-k finales.

```text
consulta → embeddings (ANN) → top-50 → cross-encoder → top-3 → LLM
```

La mejora del cross-encoder en los primeros puestos es consistente y barata:
solo re-escora decenas de candidatos, no el corpus entero.

### Recuperar pocos fragmentos buenos

- **top-k pequeño** (2–5): más tokens no es más información; a partir de un
  punto, los fragmentos marginales añaden ruido que el modelo mezcla con el
  bueno ("Lost in the Middle" + confusión).
- **Poner los más relevantes primero** (ver el orden) y, si el proveedor lo
  permite, marcarlos en bloques delimitados.
- **Citas**: cada afirmación de la respuesta debe poder trazarse al fragmento
  recuperado. Exigir en el prompt el número de fragmento y, en salida
  estructurada, un objeto con la respuesta y los ids de fuente. Sin citas no
  hay forma de auditar si la respuesta salió del contexto o de la memoria del
  modelo.

### Evaluación del RAG: dos ejes

El RAG tiene dos fuentes de error independientes y se evalúan por separado:

| Eje | Pregunta | Métrica |
|-----|----------|---------|
| Relevancia del contexto | ¿El fragmento recuperado responde a la consulta? | hit_rate, recall@k, MRR |
| Fidelidad de la respuesta | ¿La respuesta se sostiene en el contexto? | faithfulness (no alucina) |

Un sistema puede fallar en ambos o en solo uno: recupera perfecto y alucina,
o recupera basura y responde coherente pero inventado. La fidelidad se mide
con LLM-as-judge: un juez comprueba si cada afirmación está respaldada por los
fragmentos citados.

{% if use_rag %}
### Este proyecto: el corpus es exactamente esto

El RAG de este proyecto indexa el código, los prompts de los agentes,
`docs/`, `vault/` y este corpus (`knowledge/`), y el `lider` lo consulta con
`rag search`. Aquí opera la ingeniería descrita arriba: chunking por semántica
al indexar, embeddings (all-MiniLM-L6-v2, ONNX) para recuperar y
`hit_rate`/`recall@k`/MRR contra `agents/evals/rag_golden.json` para medir la
búsqueda. Mantén ese golden: es la evidencia de que recuperar conocimiento
sigue funcionando entre cambios.

```bash
uv run python -m agents --json run rag search --query "..." --file_type knowledge
make index-rag
make eval-rag
```
{% endif %}

## Fine-tuning vs prompting vs RAG

No son alternativas excluyentes; son palancas para problemas distintos.

| Problema | Palanca | Por qué |
|----------|---------|---------|
| No sabe un conocimiento específico | **RAG** | Se inyecta en contexto; se actualiza sin reentrenar |
| El formato/tono/estructura no encaja | **Fine-tune** | El modelo aprende el estilo de salida en los pesos |
| El razonamiento falla en casos sencillos | **Prompting** | Mejorar prompt/ejemplos es barato y reversible |
| Coste/latencia de producción | **Distilación** | Entrenar un modelo pequeño que imite al grande |

**Cuándo NO fine-tunear**: si el problema es que el modelo no sabe algo que
está en documentos → RAG. El fine-tune es caro de producir (dataset curado,
entrenamiento, evaluación), caro de mantener (se desactualiza con cada cambio
de datos y hay que repetirlo) y opaco (los fallos no se arreglan editando un
prompt). Su ventaja real es **eficiencia y control de formato**: una salida
estructurada o un tono consistentes con un modelo pequeño cuestan menos por
token que el prompt de un modelo gigante.

**Riesgo de memorización**: el fine-tune puede memorizar el dataset de
entrenamiento en vez de generalizar (sobreajuste a los ejemplos, alucinación
de las salidas del set de entrenamiento, olvido catastrófico del conocimiento
anterior). Se mitiga con un set de evaluación separado del de entrenamiento,
mezclar datos genéricos y validar la calibración antes y después. Si el
comportamiento buscado cabe en un prompt de 2 000 tokens con 5 ejemplos, no
hace falta tocar los pesos.

## Producción

### Salida estructurada: JSON mode y function calling

Nunca parsees la prosa del modelo como si fuera datos. Dos mecanismos:

- **JSON mode**: el proveedor garantiza que la salida sea JSON válido. El
  esquema se pide en el prompt o se declara al API.
- **Function calling / tools**: el modelo elige entre funciones declaradas y
  argumentos JSON válidos; para orquestar acciones, no solo texto.

```json
{"tipo": "consulta", "entidad": "cliente_42", "accion": "obtener_saldo"}
```

Trata la salida como no confiable aun con JSON mode: **valida contra esquema**
(pydantic) y rechaza lo que no cumpla. El JSON mode garantiza sintaxis, no
semántica.

### Retries y validación de salida

La inferencia es estocástica y el API falla: el código debe asumir ambos.

```python
def call_with_retry(prompt, schema, retries=3):
    for attempt in range(retries):
        try:
            text = provider.complete(prompt, response_format="json")
            return schema.model_validate_json(text)
        except (ProviderTimeout, ValidationError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
```

- **Retry solo sobre errores transitorios** (timeout, 429, 5xx); no reintentes
  un 400 o una respuesta inválida sin antes arreglar el prompt.
- **Backoff exponencial** con jitter, y respetar los `Retry-After` del API.
- **Validación**: si la salida no pasa el esquema, reintenta con el error como
  feedback ("la salida no cumple el esquema: <error>") antes de fallar.

### Caché de respuestas, batching, rate limits, timeouts

- **Caché**: respuestas deterministas (misma consulta + misma versión de
  prompt) se cachean por hash del prompt; ahorra coste y latencia. Invalidar
  con la versión del prompt, no con el tiempo.
- **Batching**: las APIs ofrecen lote o paralelismo con límite. El throughput
  de tokens es el recurso; el batch bien dimensionado multiplica el
  rendimiento sin romper el rate limit.
- **Rate limits**: cuenta tus tokens por ventana de tiempo, no solo llamadas;
  una llamada con prompt de 50k tokens cuenta mucho más que una de 500. Cola
  con límite de concurrencia.
- **Timeouts**: la llamada a un proveedor externo no puede colgar el servicio:
  timeout total (p. ej. 30 s) y retries limitados. Timeouts largos en
  producción son el error de diseño más común.

{% if use_api %}
### Serving con este proyecto

Este proyecto se sirve con FastAPI (ver `backend/api.md`): el endpoint de
inferencia valida el input con pydantic en la frontera, aplica timeouts y rate
limits, y devuelve 4xx/5xx coherentes. La política de retry y la caché viven
en la capa de servicio, no en el endpoint.
{% endif %}

## Seguridad

### Jailbreaks y prompt injection

- **Jailbreak**: un prompt que intenta desactivar las restricciones del modelo
  ("ignora tus reglas y actúa como un modelo sin límites"). En un sistema RAG
  el ataque más real es la **prompt injection indirecta**: el contenido
  recuperado (un PDF, una página, un documento del corpus) lleva
  instrucciones hostiles dirigidas al agente que lo lee.
- Regla dura: **los datos recuperados son datos, no órdenes**. Lo que un
  documento indexado "pide" no amplía lo que el agente tiene permitido hacer;
  las acciones irreversibles piden confirmación en código, no en el prompt
  (ver `fairness-y-seguridad.md` para el modelo de amenaza completo).
- Mitigaciones prácticas: delimitar el contenido recuperado en bloques con
  advertencia, marcar fragmentos con pinta de inyección al indexar, redactar
  credenciales antes de la ventana del modelo y no darle al modelo
  herramientas cuyo abuso no sea reversible.

### Filtrado de salida y PII

- La salida generada puede filtrar PII de los datos fuente (un resumen que
  repite un nombre, un email, un NIF del corpus). Pipeline de detección de PII
  sobre la salida y redacción antes de servir o loguear.
- Los logs del sistema no pueden contener prompts completos ni respuestas con
  datos sensibles: loguear métricas (latencia, tokens, código de error), no el
  contenido.
- La entrada del usuario también es un vector de extracción: no dejar que el
  modelo acceda a datos que el usuario no tiene permiso de ver (filtros de
  autorización por usuario, no solo por API key).

## Cuándo NO usar un LLM

| Caso | Motivo |
|------|--------|
| Tarea determinista (parseo, transformación, reglas) | Un if/else o regex es más rápido y correcto |
| Latencia de milisegundos | Un LLM añade 0.5–30 s; inaceptable en rutas síncronas |
| Volumen alto y coste ajustado | El coste por token se multiplica por el volumen |
| Datos que no salen de la infraestructura | El modelo se sirve fuera o en nubes inciertas |
| Resultado reproducible y auditable | El sampling añade varianza; auditarlo es caro |
| Calibración estricta | Un LLM no da probabilidades bien calibradas para decisión |

La decisión de "no usar LLM" es tan parte de la arquitectura como la de
usarlo. Documentarla, y por qué, es el ADR que evita que el siguiente
ingeniero reabra el debate sin los números.

## Fuentes

- **Attention Is All You Need** — A. Vaswani et al. (2017).
  arXiv:1706.03762 — https://arxiv.org/abs/1706.03762
- **Lost in the Middle: How Language Models Use Long Contexts** — N. F. Liu et
  al. (2023). arXiv:2307.03172 — https://arxiv.org/abs/2307.03172
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** —
  P. Lewis et al. (2020). arXiv:2005.11401 — https://arxiv.org/abs/2005.11401
- **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** —
  J. Wei et al. (2022). arXiv:2201.11903 — https://arxiv.org/abs/2201.11903
- **Self-Consistency Improves Chain of Thought Reasoning** — X. Wang et al.
  (2022). arXiv:2203.11171 — https://arxiv.org/abs/2203.11171
- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — L. Zheng et
  al. (2023). arXiv:2306.05685 — https://arxiv.org/abs/2306.05685
- **Searching for Best Practices in Retrieval-Augmented Generation** — G. Gao
  et al. (2024). arXiv:2404.06981 — https://arxiv.org/abs/2404.06981
- **LoRA: Low-Rank Adaptation of Large Language Models** — E. Hu et al.
  (2021). arXiv:2106.09685 — https://arxiv.org/abs/2106.09685
- **Universal and Transferable Adversarial Attacks on Aligned Language
  Models** — A. Zou et al. (2023). arXiv:2307.15043 — https://arxiv.org/abs/2307.15043
- **Not what you've signed up for: Compromising Real-World LLM-Integrated
  Applications with Indirect Prompt Injection** — K. Greshake et al. (2023).
  arXiv:2302.12173 — https://arxiv.org/abs/2302.12173
- **Prompt engineering (OpenAI)** — https://platform.openai.com/docs/guides/prompt-engineering
- **Prompt engineering (Anthropic)** —
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
