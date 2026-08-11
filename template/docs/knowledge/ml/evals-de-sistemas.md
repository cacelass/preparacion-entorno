# Evaluación de sistemas con LLM: evals, no vibes

Evaluar **modelos** es compararlos; evaluar **sistemas** es definir qué es
"bueno" para tu caso y medir si lo cumples. Son dos cosas distintas, y
confundirlas es la forma más común de construir un sistema de IA que "parece
que funciona". Un benchmark de modelo (MMLU, HumanEval) no te dice nada sobre
tu pipeline de RAG, tus agentes o tu prompt; un *system eval* sí.

La regla que sostiene este fichero (y la práctica del proyecto):

> **Producción de IA empieza por los evals.** Si no puedes medir que el
> sistema hace lo que debe, no sabes si un cambio lo mejora o lo rompe —
> solo sabes que "parece que va bien", que es exactamente como se rompe en
> producción.

Este fichero complementa a `metricas-y-evaluacion.md` (métricas de modelo,
comparación honesta) y a `testing-ml.md` (tests de datos y de modelo). Aquí el
foco es el **sistema**: el prompt, el flujo de recuperación, los agentes y sus
trayectorias.

## Model evals vs system evals

| | Model eval | System eval |
|---|------------|-------------|
| Qué mide | La capacidad de un modelo en aislamiento | Si el sistema sirve bien una consulta |
| Ejemplo | "¿Este LLM razona mejor que el otro?" | "¿Mi RAG devuelve la fuente correcta para esta pregunta?" |
| Referencia | Benchmarks públicos | Tus casos reales |
| Se rompe | Cuando usas el benchmark como si fuera tu caso | Cuando los casos no reflejan a tus usuarios |

Garry Tan: "Don't rawdog your prompts! Write evals!". swyx: "Production AI
Engineering starts with Evals". El meme es de risa, la lección no: un prompt
que nadie ha puesto a prueba es código sin tests.

## Vibe evals: el punto de partida honesto

Empieza por aquí, funciona mejor de lo que crees, pero se agota rápido:

- **Prueba manual por muestreo**: ejecutas 20 consultas, miras las respuestas
  a ojo, y apruebas. Es "LGTM@K" — un humano que echa un vistazo.
- **Vale** para validar hipótesis en desarrollo, cuando aún no sabes ni qué
  quieres medir.
- **No vale** para regresión: cuando cambias el prompt o el modelo, no puedes
  re-ejecutar "la sensación" de hace un mes. No hay número que comparar.

El salto de madurez es convertir la sensación en un **golden set** versionado
que se re-ejecuta en cada cambio — igual que un test unitario.

## El golden set: el test de regresión del sistema

Un golden set es un fichero versionado con casos `(entrada, salida esperada)`
de tu dominio real. Es la columna vertebral de todo system eval.

Qué lo hace bueno:

- **Casos reales, no inventados.** Las preguntas que tus usuarios hacen de
  verdad, y las que *deberían* funcionar pero fallan. Cada respuesta que
  devuelve basura se convierte en un caso.
- **Cubre la diversidad.** Varios caminos al mismo resultado, casos límite,
  entradas hostiles (si tu sistema lee contenido externo: inyecciones).
- **Es versionado.** El golden set vive en el repo, se revisa en PR, cambia
  con el código. Si cambia el producto, cambian los casos.
- **Se ejecuta en CI.** Cada push re-ejecuta la suite contra el sistema real
  y compara con el commit anterior. Eso es lo que convierte "parece que
  mejoró" en "la MRR subió de 0.31 a 0.55".

En este proyecto, el patrón ya existe: `agents/evals/rag_golden.json` + el
runner (`python -m agents.evals.runner --rag`) mide `hit_rate`, `recall@k`,
`MRR` y `lexical_share` contra casos reales. La regla del `lider` es: toda
pregunta que devuelva basura entra al golden set.

## Tipos de verificación

| Tipo | Qué comprueba | Cuándo usarlo |
|------|---------------|---------------|
| Coincidencia exacta | La salida es un string concreto | Salida estructurada (JSON, enum, ruta) |
| Coincidencia parcial / substring | Contiene algo esperado | Respuestas libres, presencia de keywords |
| Property-based | Una propiedad se cumple para muchas entradas | Invariantes: "el JSON parsea", "no contiene secretos", "la ruta existe" |
| Heurística determinista | Regla de negocio en código | "Todas las fuentes citadas existen" |
| LLM-as-judge | Un segundo LLM puntúa | Respuestas abiertas sin referencia objetiva |

### Property-based tests: el menos usado y el más barato

Para un sistema de agentes, la mayoría de las propiedades son invariantes que
se pueden comprobar en código sin ningún LLM de por medio:

- **Idempotencia**: ejecutar la misma acción dos veces da el mismo estado.
- **No-fuga**: ninguna salida contiene un secreto (se puede verificar con
  `redaction` / un regex).
- **Esquema**: toda respuesta estructurada parsea contra su schema.
- **Invariante de dominio**: "un feature en `done` tiene evidencia no vacía".

Estos son tests *normales*, escritos con pytest, que corren en milisegundos y
no cuestan un token. Son la primera línea del system eval — antes de gastar
en LLM-as-judge.

## LLM-as-judge: potente y traicionero

Cuando la salida es abierta y no hay referencia objetiva, un LLM puede puntuar
("¿esta respuesta responde a la pregunta?"). Es la herramienta correcta para
lo que no se puede automatizar — pero tiene sesgos sistemáticos conocidos:

- **Favorece la verbosidad**: respuestas largas puntúan más alto aunque sean
  peores.
- **Favorece su propia voz**: el juez puntúa mejor lo que se parece a cómo él
  escribiría.
- **Orden de presentación**: la posición de la respuesta cambia el veredicto.
- **Deriva del juez**: cambiar de modelo-juez cambia la puntuación — y puede
  no ser por mérito.

Cómo usar bien un juez LLM:

1. **Rubrica explícita y acotada.** "Puntúa 1-5 si la respuesta cita la fuente
   correcta y no alucina números". Sin rubrica, el juez inventa los criterios.
2. **Juez determinista cuando puedas.** Un checklist en código sobre la salida
   (¿contiene la fuente? ¿coincide el número?) gana a un LLM que "opina".
3. **Calibra el juez** contra un puñado de casos etiquetados por humanos: mide
   con qué frecuencia coincide, igual que medirías cualquier clasificador.
4. **Reporta el juez como otra métrica**, no como la verdad.

## Evaluar agentes: la trayectoria, no solo la respuesta

Un agente no devuelve solo una respuesta — ejecuta **acciones en secuencia**
(leer, buscar, escribir, ejecutar). Evaluar solo la respuesta final mide la
punta del iceberg; el valor del agente está en la *trayectoria*.

Qué se evalúa de un agente:

| Aspecto | Pregunta | Cómo |
|---------|----------|------|
| Resultado final | ¿Terminó la tarea? | El criterio de aceptación de la feature |
| Acciones | ¿Usó las herramientas correctas en el orden correcto? | Registro de llamadas vs trayectoria esperada |
| Restricciones | ¿Respetó los límites? | Ninguna acción destructiva sin confirmación |
| Eficiencia | ¿Cuántas llamadas / tokens gastó para el resultado? | Auditoría: `audit.jsonl` |
| Fallo elegante | ¿Qué hizo cuando una herramienta falló? | Escenario de error inyectado |

La lección práctica: **los agentes se evalúan por lo que ejecutan, no por lo
que dicen**. En este proyecto la base ya existe: cada ejecución de cada agente
queda en el log de auditoría (`agents/workspace/audit/audit.jsonl`), y el
runner de evals mide smoke/routing/contracts/harness/rag. El siguiente paso
natural es un *eval de trayectoria*: una feature de prueba donde el agente
debe llegar al criterio de aceptación, y el eval verifica que las acciones
registradas lo demuestran.

Para evaluar *sistemas* de agentes (orquestación, handoff, subagentes), el
patrón que de verdad funciona es el que ya usa este arnés:

- **Restricciones duras en código** (la puerta `init.sh`, la evidencia, la
  certeza `μ.cert`), no instrucciones que el modelo puede ignorar.
- **Cada subagente registra su informe** (`harness record`) — la trayectoria
  queda fuera de la ventana, en un fichero, y es lo que el siguiente agente lee.
- **El reviewer ejecuta, no cree** (`.opencode/agents/reviewer.md`): corre la
  puerta él mismo y verifica criterios con comandos, no con el informe ajeno.

## Cuándo el eval miente (y cómo se nota)

Un eval no es una campana que suena sola: se corrompe, y saber cuándo es la
mitad del oficio.

- **El eval enseña a pasar el eval.** Si la métrica es "presencia de la
  fuente", el sistema "aprende" a meter la fuente en todas las respuestas
  aunque no responda. El overfitting al eval es la forma moderna del Goodhart
  (ver `gestion-riesgo.md`).
- **Los casos dejan de representar a los usuarios.** El golden set se quedó
  en las preguntas de hace 6 meses; el producto cambió. Se detecta cuando la
  suite da 100% pero producción se queja — entonces los casos están rotos, no
  el sistema.
- **El juez es el sistema.** Un LLM-as-judge que puntúa a otro LLM de la misma
  familia comparte los mismos sesgos; es una sola opinión con dos cabezas.
- **"No medido" se confunde con "mide cero".** Un sistema que no se pudo
  evaluar (sin datos, sin índice, sin casos) no falla — simplemente no se sabe.
  Reportarlo como cero es la forma más fácil de volver inútil la métrica (es
  la decisión explícita del runner de evals de este proyecto: `available=False`
  ≠ fallo).

Señal de que el eval está sano: **un cambio que empeora el sistema baja la
métrica sin que tengas que leer una respuesta a mano.** Si un cambio rompe y
la suite sigue verde, el eval no está mirando donde duele.

## El flujo completo (eval-driven development)

```
1. Define "bueno" como casos concretos → el golden set
2. Escribe el eval en código (property-based primero, heurística después, juez al final)
3. Ejecuta contra el sistema actual → línea base
4. Cambia prompt/modelo/flujo → re-ejecuta → compara con la línea base
5. Un cambio que baja la métrica se rechaza o se arregla, igual que un test rojo
6. Cada respuesta mala real → caso nuevo al golden set → vuelta al paso 3
```

Es el ciclo del `lider` de este proyecto aplicado a la ingeniería de prompts:
primero al backlog (el caso es un "feature"), después se implementa (el eval),
después evidencia (la métrica) antes de cerrar.

## Referencias cruzadas

- `metricas-y-evaluacion.md` — las métricas de modelo por tarea y la comparación honesta
- `testing-ml.md` — tests de datos y de modelo, invariantes, property-based
- `llms-aplicados.md` — evaluación de LLM con y sin referencia, golden fijo
- `guardarraíles.md` — el sistema de contención que también hay que evaluar
- `gestion-riesgo.md` — Goodhart, sesgos de medición, el eval que miente
- `ciclo-vida-mlops.md` — observabilidad: los evals y las métricas en producción
