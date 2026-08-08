# Deuda técnica en sistemas de ML

## La tesis: el modelo es la parte pequeña

En un sistema de ML en producción, el código del modelo (entrenar + inferir)
es una fracción mínima del sistema total. El resto — ingestión, limpieza,
ingeniería de features, etiquetado, validación, serving, monitorización,
configuración, orquestación, despliegue — concentra la complejidad y, con
ella, la deuda. El modelo no existe sin ese sistema; y es en el sistema donde
se acumula el costo diferido que luego se paga con intereses.

Sculley et al. llaman a esto deuda técnica **oculta**: la que no se ve en el
código del modelo sino en dependencias frágiles, contratos tácitos y
comportamiento que nadie declaró. El riesgo no es que el modelo deje de
aprender; es que el sistema que lo sostiene se rompa por una arista que nadie
mapeó.

## Bucles de retroalimentación ocultos

Un modelo modifica los datos sobre los que se reentrena, y eso cambia el
modelo siguiente: deuda que se autocultiva.

| Tipo | Cómo se cierra el bucle | Costo de corregirlo |
|------|-------------------------|---------------------|
| Directo | El modelo cambia decisiones, que generan el siguiente dato | Guardar predicción vs. outcome |
| Semi-supervisado | El modelo etiqueta datos que alimentan el retrain | Los errores se vuelven ground truth |
| Externo | El modelo afecta al mundo, que genera el dato | Instrumentación externa |

La defensa es registrar **lo que el modelo predijo y cuál fue el resultado**
(log de decisiones con ground truth diferido): el bucle solo se audita si las
predicciones se retienen como dato.

## Enredo (entanglement)

Features y representaciones compartidas por varios modelos acoplan sus ciclos:
mejorar la feature de un modelo degrada a los demás. El cambio de una entrada
compartida es un experimento sobre todos los consumidores a la vez, y las
correlaciones hacen que "funciona" en un modelo no signifique nada en otro.

- Versionar features y tratarlas como contrato entre modelos.
- Aislar las representaciones compartidas detrás de una interfaz estable.
- Tratar el conjunto de modelos como un sistema acoplado: un cambio de input
  exige tests sobre todos sus consumidores.

## Cascadas de corrección

Cuando A alimenta a B, y se detecta que A produce datos mal, la tentación es
corregir en B (un parche en el consumidor) en lugar de arreglar A (la
fuente). Cada parche es deuda que se apila: B acaba con capas de correcciones
cuyo efecto conjunto nadie entiende. Arreglar la fuente puede romper a los
consumidores que ya se adaptaron al dato corrupto — por eso la cascada
persiste. La salida es la misma que en ingeniería: corregir en la fuente, con
test de regresión, y versionar el cambio de contrato de datos.

## Consumidores no declarados

Alguien consume la salida del modelo sin que el productor lo sepa: un
dashboard, un informe, otro equipo. Cuando la salida cambia, el consumidor no
declarado se rompe en silencio. La defensa no es adivinar quién usa qué, sino
declararlo:

- Contrato de salida (schema y semántica) versionado y visible.
- Registro de consumidores conocido, en el repo y no en la cabeza.
- Cambios de salida señalizados y migrados explícitamente, nunca "de paso".

## Dependencias de datos

El código se testea; los datos, casi nunca. Una dependencia de datos es tan
real como una de código, pero el compilador no la detecta: un schema cambia y
el pipeline no falla, solo produce una salida distinta. Deuda clásica: "el
pipeline funciona" hasta que el dataset cambia de formato o de significado.

- Checksum y versión del dataset como parte del run.
- Tests de datos (schema, rangos, invariantes, tipos) en CI, no solo tests de
  código.
- Distinguir features passthrough de features "debatidas": las segundas
  cambian de significado y hay que justificarlas.

## Deuda de configuración

Configuración es todo valor que cambia el comportamiento sin tocar código:
umbrales, paths, hiperparámetros, flags. La entropía crece cuando se esparcen
valores mágicos, defaults silenciosos y banderas activadas a mano en
producción. Configurar con el mismo rigor que el código: versionado, review,
validación y tests. En este proyecto, la configuración del arnés vive en
`harness/featureslist.json` y se valida en `./init.sh`: un backlog mal formado
bloquea el trabajo en lugar de corromperlo en silencio.

## Glue code y pipeline jungles

La mayor parte del código real de un sistema de ML no es ML: es pegado de
datos entre sistemas (glue code). Cada librería y cada fuente añade una capa
de pegamento que nadie quiere mantener. Los pasos de preparación encadenados
en scripts sin estructura forman "pipeline jungles", donde cada "y aquí añado
otro paso" incrementa la deuda. Tratar la preparación de datos como un
programa de software (módulos, tests, versionado), no como una receta que se
ejecuta a mano.

## Código experimental muerto

El código de experimentos que se queda en el repo — ramas fusionadas, scripts
de prueba, variantes sin ganador — corrompe la pregunta "¿qué es lo que corre
en producción?". El modelo activo no se identifica por el código más reciente
sino por el registry: lo que está en `Production` es la fuente de verdad; todo
lo demás es experimento y puede borrarse sin miedo.

## Deuda del sistema de serving: train/serve skew

La forma clásica: el modelo se entrena sobre datos procesados de una manera y
se sirve sobre datos procesados de otra. Tres fuentes:

| Tipo | Causa | Defensa |
|------|-------|---------|
| Skew de código | Transformación distinta en train y serve | Pipeline compartido; tests de la misma ruta |
| Skew de entorno | Versiones de librerías distintas | Entorno congelado (uv.lock / imagen) |
| Skew de datos | Distribución distinta en serve | Monitorización y comparación train vs. serve |

**Shadow mode**: desplegar el modelo nuevo junto al viejo, en paralelo, sin
actuar sobre sus salidas, comparando sus predicciones con las del modelo en
servicio. Detecta el skew y el drift de comportamiento antes de arriesgar
producción, sin código de rollback especial. Un solo `predict` que aplica el
preprocesado distinto al del entrenamiento es deuda silenciosa con cara de
bug de datos.

## Umbrales fijos

Un umbral codificado (`score >= 0.7`) fijado una vez y nunca recalibrado es
deuda: la distribución cambia, el umbral queda mal, y nadie lo toca porque
"funcionó". El umbral debe ser configuración versionada, no una constante en
el código, y debe recalibrarse contra ground truth (ver
`backend/servir-modelos.md`).

## Deuda de monitorización

Monitorear es una decisión de producto, no decoración. Dashboards que nadie
mira, métricas sin dueño y alertas sin runbook acumulan la peor deuda:
enseñan a la organización a ignorar las señales. Cada alerta debe tener acción
y responsable; cada métrica, un umbral calibrado y revisado.

## El arnés disciplinado contra la deuda

La tesis del paper tiene una respuesta de ingeniería: convertir las
dependencias frágiles en contratos verificados. Este proyecto implementa esa
respuesta en el arnés:

- **Pipeline determinista**: misma entrada → misma salida. Semillas, orden de
  operaciones y versiones fijas; un run que cambia sin cambiar nada es un bug
  que se persigue.
- **Tests sobre datos y código**: unitarios de transformaciones y tests de
  datos (schema, invariantes) en CI; la suite es la red que detecta el cambio
  de contrato.
- **Datos y modelos versionados**: checksum de dataset y registry de modelos;
  el run loguea las cuatro coordenadas (entorno, código, datos, parámetros).
- **La puerta `./init.sh`**: nada entra si el entorno, la estructura, el
  backlog y los tests no están verdes. La deuda no se cuela "de pasada".
- **Decisiones registradas**: `harness/progress/` y `history.md` guardan el
  porqué; una decisión sin registro se reinventa o se contradice.
- **Registry como fuente de verdad**: qué se sirve es la versión en
  `Production`, no un fichero suelto (ver `backend/mlflow.md`).
{% if use_sdd %}
- **Tests que muerden**: la mutación (`run_mutation_testing`) y el CRAP
  (`crap_report`) verifican que los tests detectan lógica rota, no solo que
  cubren líneas. Un `survived` es deuda que se paga antes de cerrar la
  feature.
{% endif %}

## Reconocer la deuda en un proyecto

Señales operativas, no opiniones:

- "Ayer funcionaba": un cambio cuyo efecto nadie explica.
- Nadie sabe por qué existe una columna o una transformación.
- Reentrenamiento manual: el notebook se ejecuta a mano, se copia-pega, se
  despliega sin registro.
- Frente a un fallo se reintenta en vez de investigar.
- Onboarding de un nuevo miembro: semanas en producir el primer cambio.
- La documentación y el código dicen cosas distintas.

Cualquiera de estas es deuda con intereses: se nota cuando hay que cambiarlo.

## Pagarla incrementalmente

La deuda se paga con refactorizaciones pequeñas y con red de seguridad, no
con una "semana de limpieza":

1. Escribir el test que fija el comportamiento actual (rojo antes del cambio).
2. Refactor pequeño y verificado: extraer módulo, renombrar, quitar parámetro.
3. Convertir scripts sueltos en funciones testeables dentro del paquete.
4. Borrar código muerto y experimentos sin ganador.
5. Registrar la deuda detectada en el backlog (`harness/featureslist.json`),
   con su criterio de aceptación, como cualquier feature.

Cada cambio paga interés y deja la siguiente refactorización más barata.

## Cuándo el ROI es real

La deuda se paga cuando el costo de mantenerla supera al de pagarla — y la
señal es operacional, no estética: tiempo de implementar un cambio, frecuencia
de incidentes, velocidad de onboarding, fricción en cada release. Refactorizar
código que funciona **sin** tests cambia comportamiento en silencio: esa
refactorización no paga deuda, la crea. Y hay deuda que no se paga sino que
se borra: el código muerto no se refactoriza, se elimina.

## Fuentes

- **Hidden Technical Debt in Machine Learning Systems** — D. Sculley, G. Holt,
  D. Golovin, E. Davydov, T. Phillips, D. Ebner, V. Chaudhary, M. Young,
  J.-F. Crespo, D. Dennison (2015). NIPS 2015 (no está en arXiv).
- **Characterizing Technical Debt and Antipatterns in AI-Based Systems** —
  V. Lenarduzzi, F. Lomio, H. Huttunen, D. Taibi (2021).
  arXiv:2103.09783 — https://arxiv.org/abs/2103.09783
- **Continuous Integration of Machine Learning Models with ease.ml/ci** —
  D. Cheng, J. Li, et al. (2019). arXiv:1903.00278 — https://arxiv.org/abs/1903.00278
