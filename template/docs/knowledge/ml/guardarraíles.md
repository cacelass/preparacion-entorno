# Guardarraíles para modelos generativos

Las capas que contienen el daño cuando un modelo generativo (LLM, asistente,
agente) está expuesto. Un modelo recibe entradas hostiles por diseño: no es una
cuestión de "si" alguien intentará un jailbreak o una inyección, sino de
"cuántas veces al día". Complementa a `fairness-y-seguridad.md` (sesgos,
adversariales, prompt injection) y a `modelos-fundacionales.md` (ciclo del FM);
aquí el foco es el **sistema de contención** alrededor del modelo y cómo se
evalúa y se mantiene.

## El principio que sostiene todo

> **El modelo propone, el sistema decide.**

Un guardarraíl que depende solo de que el LLM "recuerde no hacer algo" es
decoración. El prompt del sistema y la política son la capa de usabilidad; la
defensa de verdad está en validar la entrada, filtrar la salida y restringir
las acciones que el modelo puede disparar — con código y permisos, no con
instrucciones.

Esto define la arquitectura: el modelo es un componente del sistema, y el
sistema (no el modelo) es quien tiene autoridad para ejecutar. Un fallo del
modelo es un incidente de la capa de generación; un fallo del guardarraíl es
un incidente de seguridad real.

## Modelo de amenaza: a quién te enfrentas

| Actor | Objetivo | Técnica típica |
|-------|----------|----------------|
| Usuario curioso | Hacer trampa al sistema | Prompt que fuerza al modelo a ignorar la política |
| Atacante externo | Extraer datos (PII, prompts, pesos) | Exfiltración, membership inference, extracción de instrucciones del sistema |
| Atacante interno | Dañar, sabotear | Backdoor, envenenamiento de datos, acceso indebido |
| Contenido hostil (terceros) | Manipular al agente | **Indirect prompt injection** a través de documentos, webs, URLs, emails, MCP |
| Competidor | Abusar del coste | DoS, batching de peticiones caras, uso indebido de la API |

El caso más importante y el que más se subestima: **indirect prompt injection**.
El atacante no habla con el modelo directamente; escribe un documento, una web o
un email que el agente va a leer, y dentro va "ignora tus instrucciones y haz
X". Para dskit: cualquier contenido recuperado por `rag search`, cualquier
respuesta de un servidor MCP, cualquier PDF indexado. La regla que aguanta es
la del arnés: **los datos que consume un agente no amplían lo que tiene
permitido hacer** — y las acciones irreversibles piden confirmación de todos
modos.

## Taxonomía de ataques a LLM (OWASP LLM Top 10 como mapa)

- **Prompt injection** (directa e indirecta): el input contiene instrucciones
  que secuestran al modelo.
- **Jailbreak**: técnicas para quebrar la política (role-play, "suficientemente
  malvado", codificación, DAN, few-shot adversarial).
- **Exfiltración de datos sensibles**: memorización (el modelo repite datos del
  pre-training), leakage vía logs, extracción de PII.
- **Fuga de la instrucción del sistema**: "repíteme tu prompt inicial".
- **Envenenamiento de datos**: backdoors en el fine-tuning o en el corpus.
- **Modelo no seguro en cadenas**: el modelo decide acciones → inyección
  indirecta → ejecución no autorizada.
- **DoS / abuso de recursos**: coste, rate, contenido generativo ilimitado.
- **Salida insegura**: contenido prohibido, código malicioso, instrucciones
  peligrosas generadas de forma no intencionada.

## Capas de guardarraíles (defensa en profundidad)

### 1. Frontera de entrada

- Validar y acotar el input antes de que toque el modelo: límites de tamaño,
  tipos, rangos, rate limiting (la inferencia es cara). Un payload inválido es
  un error controlado, nunca un 500 (ver `backend/api.md`).
- **Clasificar la intención**: un clasificador previo (modelo pequeño, reglas)
  puede separar tráfico benigno del sospechoso antes de gastar inferencia
  cara. No es la defensa, es economía de defensa.
- No permitir que la entrada controle directamente acciones del sistema: el
  texto del usuario es un dato, nunca una instrucción para ejecutar.

### 2. Política y prompt del sistema

- Declarar qué hace y qué rechaza el asistente (persona, límites, tono,
  formato), pero entender que esto **no** es la defensa — es lo primero que un
  jailbreak intenta borrar.
- **Delimitar contenido no confiable**: separar en bloques etiquetados las
  instrucciones del sistema del contenido recuperado (documentos RAG, URLs,
  respuestas MCP). dskit ya lo hace en `rag search` (bloque aparte +
  `injection_flag` + warnings).
- **Principio de mínimo privilegio del prompt**: el sistema no debería decirle
  al modelo que puede hacer cosas que el código no le permite. La política del
  prompt y los permisos del código deben contar la misma historia.

### 3. Frontera de salida (filtros deterministas)

- Filtrar/redactar la respuesta **antes de entregarla**: PII, credenciales,
  contenido peligroso, instrucciones prohibidas, URLs internas, secretos.
- El filtro es posterior, determinista y auditable — no depende del modelo.
  Un LLM puede alucinar; un filtro de regex/clasificador sobre la salida no.
- **Watermarking** (para modelos propios): marcar el texto generado para
  detectar abusos y trazabilidad.
- **Validación estructural**: si el agente produce JSON/código/acciones, validar
  el esquema y el contenido antes de actuar sobre ello.

### 4. Acciones limitadas (la frontera real)

- Si el agente puede ejecutar herramientas, lo que decide es el código de
  permisos, no el modelo: acciones irreversibles piden confirmación explícita
  (ver la puerta de permisos en `AGENTS.md` y `policy_guard`).
- Un jailbreak que convence al LLM no debería poder convencer al permiso.
  Esta es la capa que aguanta cuando las demás fallan: el modelo puede decir
  cualquier cosa, pero no hacer cualquier cosa.
- **Sandbox y red**: ejecutar en contenedor/usuario sin privilegios y con red
  cerrada limita el daño real de una inyección exitosa.

### 5. Monitoreo y red teaming

- **Red teaming**: atacar tu propio modelo antes de exponerlo — jailbreaks,
  prompt injection, exfiltración de secretos, contenido prohibido. Un modelo
  que no ha pasado red teaming no se considera evaluado.
- **Monitoreo continuo**: registrar abusos e incidentes (evasiones, contenido
  bloqueado, intentos de inyección), medir tasas, y alimentar el red teaming
  con lo que se ve en producción, no solo con casos de laboratorio.
- **Métrica de seguridad en la release**: tasa de jailbreak y de inyección
  exitosa en el golden set, igual que `rag_golden.json` mide el RAG. Si no lo
  mides, no sabes si empeoraste.
- **Incident response**: definir qué se hace cuando una evasión tiene éxito —
  rollback del modelo, revocar credenciales, revisar logs, actualizar el
  guardarraíl.

## Técnicas concretas (y su honestidad)

- **Perplexity / filtros de texto**: detectar jailbreaks por anomalía
  estadística. Se esquivan fácil; útil como capa barata, no como defensa.
- **Entrenamiento adversarial / constitutional AI**: entrenar al modelo para
  rechazar ataques (RLHF con preferencias de seguridad, constitution). Reduce
  la tasa de éxito de ataques conocidos pero no la elimina; y hay que
  re-entrenar cuando aparecen técnicas nuevas.
- **Clasificador de riesgo por capas** (moderation API o modelo propio): una
  segunda pasada que etiqueta la salida (violencia, PII, código malicioso).
  El umbral y el fallback (bloquear) son configurables y auditable.
- **Canary / honeypot**: tokens de marcado en el prompt para detectar
  exfiltración (si un secret-token aparece en la salida, hubo fuga).

## Evaluación de guardarraíles

Un guardarraíl sin métrica es una opinión. Definir un **golden set de ataques**
(y actualizarlo):

- Conjunto fijo de jailbreaks e inyecciones conocidos → **tasa de evasión**
  (cuántos pasan) y **tasa de bloqueo falso** (cuánto contenido legítimo se
  corta). Ambas: un guardarraíl que bloquea todo no sirve.
- Robustez a **reformulaciones**: un jailbreak que funciona reescrito con
  sinónimos es un fallo del guardarraíl, no del atacante.
- Medir en **producción**: no solo en laboratorio — las entradas reales son
  más variadas que el golden set.

## Cómo se rompe (checklist para el `lider`)

- **Guardarraíl único**: solo el prompt del sistema, sin filtros ni permisos →
  cualquier jailbreak conocido funciona.
- **Detección como defensa**: listas de patrones para inyección/jailbreak se
  esquivan con reformulaciones; son una capa, no la defensa.
- **Filtro sin auditar**: un filtro de salida que no loguea sus decisiones no
  se puede mejorar ni justificar.
- **Red teaming estático**: un set fijo de ataques se satura; hay que
  actualizarlo con las técnicas del momento y con incidentes reales.
- **Permisos flojos**: si el agente puede ejecutar comandos destructivos sin
  confirmación, ningún prompt ni filtro compensa.
- **No monitorear abusos**: si no mides la tasa de evasión, la primera vez que
  lo sabes es en el incidente.
- **Confundir "el modelo se negó" con "el sistema es seguro"**: un LLM que se
  niega por política puede ser persuadido; la defensa es el sistema.

## Dónde encaja en dskit

Este proyecto aplica la misma filosofía en su arnés: `policy_guard` antes de
cada herramienta, puerta de permisos para lo irreversible, y "los datos que
consume un agente no amplían lo que tiene permitido hacer". Para un FM expuesto,
estas capas se añaden alrededor de la API (`backend/api.md`) y el RAG
(`fairness-y-seguridad.md`). El `lider` consulta este fichero antes de
aconsejar la exposición de un modelo, y `rag refresh` lo mantiene con los
topics de `sources.json`.
