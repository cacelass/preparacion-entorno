# Contexto y memoria de agentes: la ventana como recurso finito

La ventana de contexto es el recurso más caro y más malentendido de un sistema
con LLM. No es "memoria": es un buffer de tokens que el modelo lee en cada
paso, que se degrada a medida que se llena y que se paga (en tiempo y en
dinero) cada vez que se reenvía. Casi todo lo que separa un agente que
funciona de uno que "se le olvida" es gestión de contexto, no capacidad del
modelo.

Dos ideas que sostienen este fichero:

1. **El contexto no se rellena: se compila.** Todo lo que entra en la ventana
   ocupa espacio en cada llamada. Un subagente que "hereda" el contexto del
   padre paga por él sin haberlo pedido, y rinde peor: el ruido tapa la señal.
2. **La memoria no es un requisito del modelo, es un diseño del sistema.**
   Un agente que olvida no es un modelo débil — es un sistema sin memoria
   externa. El modelo solo razona sobre lo que hay en la ventana; lo que
   necesita "recordar" más allá de la ventana tiene que vivir en ficheros,
   no en la conversación.

Complementa a `llms-aplicados.md` (tokenización, ventana, coste por token) y
a `ingenieria/estructuras-codigo.md` (fronteras de módulos). Aquí el foco es
cómo un sistema de agentes gestiona su contexto y su memoria de trabajo.

## Por qué la ventana se degrada antes de llenarse

El modelo no presta la misma atención a todo el contexto. La evidencia
empírica es clara y se resume en "Lost in the Middle" (Liu et al., 2023): el
modelo retiene mejor el principio y el final de la ventana, y se le caen las
cosas del medio. Las implicaciones prácticas:

- **Lo más importante no va en el medio.** El objetivo, los criterios y las
  restricciones al principio; la instrucción de la tarea, al final.
- **El contexto crece y empuja lo viejo al medio.** Un agente que va
  acumulando pasos relee cada vez una ventana más grande donde su propia
  historia lo ahoga.
- **No es memoria.** El modelo no "recuerda" el principio de la conversación:
  lo re-lee (o lo pierde) en cada llamada.

Consecuencia de diseño: **el estado que importa se escribe fuera de la
ventana.** Cada agente termina registrando su resultado en un fichero; el
siguiente agente lee el fichero, no la conversación. La ventana queda para
razonar, no para almacenar.

## Las tres capas de memoria de un sistema de agentes

| Capa | Qué es | Plazo | Dónde vive |
|------|--------|-------|------------|
| **Memoria de trabajo** | La feature en curso, el estado vivo del trabajo | La duración de la tarea | `harness/progress/current.md` |
| **Memoria de ejecución** | Trayectorias de los agentes: qué hicieron, cuánto tardaron, qué falló | Sesiones | Log de auditoría (`audit.jsonl`) |
| **Memoria estable** | Conocimiento duradero del dominio: decisiones, hallazgos, porqués | El proyecto | El vault / corpus (`docs/vault/`, `docs/knowledge/`) |

La regla que evita pisarse: **cada memoria tiene un dueño y un plazo.** Un
hallazgo duradero ("por qué elegimos este modelo") no vive en el progreso de
la feature — vive en el vault, que es de donde no se borra. El progreso es
memoria de trabajo: se vacía al cerrar la feature.

## El patrón de handoff: no heredar, apuntar

El error más caro en sistemas de agentes es el **handoff con contexto
heredado**: el líder lanza un subagente y le pasa "toda la conversación" para
que "sepa de qué va". Resultado: el subagente paga tokens por contexto que no
le sirve, se distrae con ruido, y al devolver el control el líder tiene aún
más contexto que gestionar.

El patrón correcto es el que ya usa este arnés:

```
lanzar(subagente) → darle solo: ID de la tarea, criterios, rutas a leer
subagente → hace su trabajo → escribe su informe en un fichero → devuelve el control
siguiente subagente → lee la RUTA del informe anterior, no su contenido
```

- **No heredes contexto.** Un subagente con el contexto lleno rinde peor que
  uno que arranca limpio: el "saber de qué va" se compra con tokens y se paga
  en atención.
- **No repitas lo que ya está en un fichero.** Pasa la ruta, no el contenido.
  Si el siguiente agente necesita el detalle, lo lee; si no, no paga por él.
- **Registra antes de devolver.** Todo lo que un agente solo dice en su
  respuesta se pierde. Lo que escribe en un fichero permanece.

### Formato compacto del handoff (protocolo §1)

En este proyecto, el informe de un subagente se factoriza a un **packet
compacto** (ejes E/S/R/Δ/μ — entidades, estado, relaciones, cambios y
certeza) que el siguiente agente lee en una línea en vez de releer la prosa.
Es el mismo principio que la teoría de compresión de contexto de trasgo:
factorizar la información a sus dimensiones mínimas, en lugar de pasarla en
prosa redundante. Ver `prompts/harness_workflow.md` — "Protocolo §1".

- El packet resume **qué cambió** (`Δ`) y **con qué certeza** (`μ.cert`).
- La prosa/evidencia sigue existiendo debajo, para quien la necesite.
- El handoff paga los pocos tokens del resumen, no los cientos de la prosa.

## Compresión y eviction: cuándo y cómo compactar

Cuando una conversación o un fichero de progreso crece, llega un punto donde
"que todo esté" cuesta más de lo que aporta. Opciones, en orden de coste:

1. **Resumir a lo esencial** (compresión): quedarse con decisiones y
   resultados, no con los pasos. Es la práctica de `harness/progress/`: "si un
   fichero pasa de ~100 líneas, resume".
2. **Eviction por relevancia**: sacar lo que ya no hace falta para la tarea en
   curso. Un log de auditoría no se relee entero: se consulta por agente, por
   fallo, por rango de fechas.
3. **Indexar en vez de retener**: no releer el histórico, buscarlo. Eso es lo
   que aporta el RAG: el histórico entra en el índice semántico y se consulta
   en lenguaje natural ("¿por qué elegimos este modelo?") sin pagar por
   releerlo entero.

La regla económica: **el contexto se compra por lo que aporta a la tarea
actual.** Un fragmento del histórico que no responde a la pregunta actual es
ruido con precio. No se relee el histórico — se le pregunta al índice.

## El coste por token del contexto

Cada token que viaja en la ventana se paga **en cada llamada**: una ventana de
10k tokens con 20 pasos de agente son 200k tokens de entrada solo en contexto.
Las palancas para reducir el coste de contexto son las mismas que para reducir
el ruido:

- **Corto en la prosa, estructurado en el dato.** Un `message` que repite el
  `data` paga dos veces por lo mismo. La regla de este proyecto: en `--json`,
  si el resultado está en `data`, el `message` no viaja.
- **Mínimo contexto por subagente.** Menos tokens por paso = menos coste total
  del flujo, y mejor atención por token.
- **Comprimir el handoff.** El packet §1 (Δ + μ.cert) en vez de la prosa.
- **Caché de contexto** (si el proveedor lo soporta): las partes del prompt
  que no cambian (system, herramientas) se cachean y no se re-pagan.

## Referencias cruzadas

- `llms-aplicados.md` — tokenización, ventana, "Lost in the Middle", coste por token
- `evals-de-sistemas.md` — evaluar trayectorias: lo que el agente *hizo*, no solo lo que dijo
- `ciclo-vida-mlops.md` — observabilidad y logs como memoria de ejecución
- `guardarraíles.md` — contenido externo en contexto: inyección indirecta
- `ingenieria/estructuras-codigo.md` — fronteras de módulos y reproducibilidad
- `matematicas/teoria-informacion.md` — entropía y compresión: el contexto como canal
