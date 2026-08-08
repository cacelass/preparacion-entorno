# Ciclo de vida de un modelo: MLOps de principio a fin

## El ciclo completo y sus puertas

El modelo vive en un bucle: entrenar → validar → desplegar → monitorizar →
reentrenar → retirar. No es un diagrama decorativo: cada fase tiene una
**puerta de entrada**, un criterio verificable que decide si se pasa. Sin
puertas, las fases se solapan, "está en producción" deja de significar nada y
el sistema degenera en deuda técnica (ver `ml/deuda-tecnica.md`).

| Fase | Puerta de entrada | Puerta de salida | Fracaso típico |
|------|-------------------|------------------|----------------|
| Entrenar | métrica con umbral; datos con contrato | run versionado | entrenar sin objetivo |
| Validar | candidato con split honesto + baseline | métricas offline estables | validar solo en train |
| Desplegar | validación verde + rollout gradual | tráfico real sano y verificable | deploy total de golpe |
| Monitorizar | umbrales, dueños, runbook | señales conectadas a acción | dashboard sin lector |
| Reentrenar | drift o calendario + re-evaluación | candidato validado y mejor | retrain a ciegas |
| Retirar | sustituido o ROI negativo | tráfico 0; versión archivada | modelo muerto sirviendo |

La puerta de entrada de una fase es la puerta de salida de la anterior. Si el
criterio no está escrito, la fase no tiene defensa contra el "por si acaso".

## Cuándo reentrenar

Tres disparadores legítimos. No son excluyentes, pero deben ser **explícitos**:
un modelo sin disparador definido se reentrena por inercia o no se reentrena
nunca, y los dos extremos son caros.

| Disparador | Qué lo activa | Riesgo |
|------------|---------------|--------|
| Calendario | cada N días/semanas, fijo | reentrenar por rutina aunque nada cambió |
| Drift | señal de monitorización que supera el umbral | reaccionar a ruido si el umbral está mal |
| Demanda | métrica de negocio cae bajo el umbral de `SCOPE-001` | detectar tarde si el decay no se mide |

El calendario es el peor justificado de los tres si no va acompañado de
evidencia: "reentrenar cada mes" asume que el mundo cambia cada mes, cuando el
modelo puede estar sano o ya muerto mucho antes. El drift y la demanda
disparan sobre señales reales; el calendario solo sobre el paso del tiempo.
En la práctica se combinan: calendario como red de seguridad para lo que no
se mide, drift y demanda como disparadores principales.

### El coste oculto del reentrenamiento

Reentrenar no es "ejecutar el notebook otra vez". Cada ciclo paga:

- **Data pipeline**: reprocesar los datos nuevos con la misma transformación;
  un cambio de fuente o de schema rompe el supuesto de que el retrain usa la
  misma definición de features que el modelo en servicio.
- **Re-evaluación**: validar el candidato contra la misma definición de
  métrica y el mismo split que el baseline; evaluar contra datos viejos no
  mide el presente.
- **Validación**: volver a pasar las puertas (tests de datos, tests de
  pipeline, checaje de drift train/serve); un retrain que salta validación
  introduce regresión en producción.
- **Riesgo de regresión**: el modelo nuevo puede estar mejor en la métrica
  offline y peor en el negocio. Un retrain que gana 0.001 de AUC pero cambia
  la distribución de decisiones es un cambio de producto, no una mejora.

La regla: **reentrenar sin vigilar es deuda.** El ciclo solo se justifica si
la re-evaluación es honesta y el resultado del retrain se compara con el
modelo activo sobre las mismas reglas — y si se puede revertir. Si no puedes
decir "este retrain fue peor y volvemos al anterior", el retrain es una
apuesta, no un proceso.

## Respuesta al drift

El drift se responde en tres tiempos: detectar, diagnosticar, decidir.
Saltarse el diagnóstico y pasar directo a decidir (típicamente "reentrenar")
es la causa más frecuente de empeorar lo que se intentaba arreglar.

### Detectar

| Tipo | Qué mide | Instrumento |
|------|----------|-------------|
| Drift en el input | la distribución de features actual vs. referencia | PSI, KS, chi², Wasserstein/MMD |
| Drift en la predicción | cambio en la distribución de la salida | KS sobre `y_pred`, ratio de clases |
| Decay de la métrica | métrica de negocio vs. ground truth diferido | evolución de etiquetas retrasadas |

El PSI se calcula sobre buckets: `PSI = Σ (p_i − q_i) · ln(p_i / q_i)`, con
regla práctica < 0.1 estable, 0.1–0.25 moderado, > 0.25 significativo. KS y
chi² son las alternativas paramétricas para numéricas y categóricas (detalle
en `data/calidad-datos.md`).

{% if use_monitoring %}
Este proyecto trae el hook de detección en `monitoring/monitor.py`: drift
KS/chi² entre la referencia (X_train) y los datos actuales, y degradación de
métricas frente al baseline, vía `make monitor`. Genera
`reports/monitoring/drift_report.csv` y `drift_report.html`. La salida es la
señal para decidir reentrenar, no la decisión: detectar es solo el paso uno.
{% endif %}

### Diagnosticar

Ante una señal, tres preguntas en orden, con evidencia antes de cada
conclusión:

1. **¿Cambió el dato?** La población o la fuente cambió (covariate shift):
   features nuevas o extremos, schema roto, otra segmentación. Se ve en el
   drift del input y se arregla reetiquetando o reentrenando sobre lo actual.
2. **¿Cambió el modelo?** La relación input→target es la misma, pero el
   preprocesado en serve divergió del de train (train/serve skew), o la
   versión desplegada no es la validada. Se ve en predicciones raras con
   input sano; se arregla corrigiendo el pipeline, no reentrenando.
3. **¿Cambió el mundo?** La relación input→target ya no vale (concept drift):
   el mismo input ya no predice lo mismo. Reentrenar puede no bastar — a veces
   hay que redefinir el problema, el target o las features.

### Decidir

| Diagnóstico | Acción | Nota |
|-------------|--------|------|
| Covariate shift | reentrenar sobre datos actuales | revalidar contra ground truth nuevo |
| El problema gana features | añadir features | no reentrenar por inercia |
| El modelo activo está roto | rollback a tripleta conocida | si el activo empeoró, no "mejorarlo" |
| Ruido / umbral mal calibrado | nada | recalibrar el umbral, documentar el falso positivo |

La tabla no es jerárquica: "nada" es una decisión legítima y documentable.
Reentrenar cuando el diagnóstico es "el mundo cambió" sin redefinir el target
es reentrenar un modelo que ya no responde a la pregunta.

## Despliegue controlado

El despliegue nunca es "subir el modelo nuevo y listo". La comparación offline
no garantiza nada sobre producción; el tráfico real es el único juez, y se
expone de forma gradual:

- **Shadow mode**: el candidato corre en paralelo al activo, sus predicciones
  se guardan pero no actúan. Detecta skew y drift de comportamiento con riesgo
  cero. Es la primera prueba real del pipeline de servir, no del modelo.
- **Canary**: un porcentaje pequeño del tráfico real (1%, luego 5%, 10%)
  va al candidato, comparando métricas de negocio contra el resto. Cada
  subida es una decisión, no un cron.
- **Blue-green**: dos entornos completos (azul y verde); el switch entre
  versiones es instantáneo y el rollback es volver a cambiar el flujo. El
  costo es el doble de infraestructura y sirve mejor cuando el rollback rápido
  importa más que el riesgo incremental.
- **A/B de modelos**: comparar dos versiones en producción sobre tráfico real.
  La decisión la toman **métricas de negocio**, no la AUC offline: conversión,
  ingresos, retención, con la hipótesis, el tamaño de muestra y la regla de
  parada fijados antes de empezar. Sin regla de parada a priori, se compara
  hasta que gana el que se quiere que gane.

{% if use_api %}
En este proyecto la versión nueva se sirve con FastAPI (`api/main.py`,
endpoint `POST /predict`, `make serve`): el servicio carga el artefacto de
`models/` y el rollout consiste en cambiar qué artefacto se sirve, sin tocar
el código. La guía de serving está en `backend/api.md`.
{% endif %}

## Reproducibilidad a tres bandas

Un run reproducible registra tres coordenadas que definen el modelo:

| Banda | Qué graba | Cómo |
|-------|-----------|------|
| Código | commit / sha del repo que produjo el run | tag o sha en el registro del run |
| Datos | versión / hash del dataset (raw y features) | checksum de `data/processed/` |
| Modelo | artefacto + parámetros + pipeline de preprocesado | artefacto versionado en el registry |

Las tres son necesarias: código sin datos no se reproduce, datos sin código no
se explican, y el artefacto sin sus dos bandas no se audita. El **rollback es
volver a una tripleta conocida**: "servir la tripleta (sha, dataset, versión)
que se sabe buena" es un comando; "recuperar el modelo que funcionaba" es una
pesquisa.

{% if use_mlflow %}
El Model Registry es la fuente de verdad de la tripleta: cada versión de
modelo cuelga de su run, que guarda git sha, hash de dataset y parámetros
(ver `backend/mlflow.md`). El despliegue se define como mover el stage a
`Production`; el rollback, como volver a una versión anterior. La API o el
servicio cargan desde `models:/<modelo>/Production` en el arranque — nunca un
fichero suelto que alguien sobrescribió.
{% endif %}

## Monitorización de KPIs

Monitorear es una decisión de producto con cuatro capas, cada una con su
umbral explícito y su dueño que responde cuando salta:

| Capa | Qué medir | Umbral explícito | Quién responde |
|------|-----------|------------------|----------------|
| Métrica de negocio | KPI del problema, ground truth diferido | caída > X% semanal | negocio + ML |
| Distribución de features | drift del input vs. referencia | PSI 0.1 / 0.25 | ingeniero de datos |
| Distribución de predicciones | cambio en la salida del modelo | ratio de clases fuera de banda | ML |
| Latencia / disponibilidad | p99, errores 4xx/5xx, uptime | p99 y error rate > umbral | plataforma |

Cada alerta sin dueño ni runbook es deuda: enseña a la organización a ignorar
las señales (ver `ml/deuda-tecnica.md`). El umbral se calibra con histórico y
se revisa — un umbral que no salta nunca o que salta todo el día es inútil.

## Observabilidad: logs, métricas y trazas

La monitorización vigila el modelo; la observabilidad vigila el sistema. Sin
los tres pilares, un incidente de modelo es "el modelo va mal" sin forma de
saber dónde:

- **Logs**: estructurados (JSON), con `request_id` para correlacionar, y solo
  lo que hace falta: predicción, score, features relevantes, latencia, código
  de error. NUNCA payloads completos ni PII — la redacción y los campos
  prohibidos están en `ml/privacidad-y-fuga-datos.md`.
- **Métricas**: contadores e histogramas de cosas que tienen distribución
  (error rate, p50/p99 de latencia, ratio de clases predichas, drift por
  feature); la tendencia importa más que el valor absoluto.
- **Trazas**: cuando una petición cruza API → modelo → base de datos, un trace
  de punta a punta dice dónde se perdieron los milisegundos; en un solo
  servicio, los logs con request_id bastan.

La regla que lo une todo: **lo que no se puede medir no se puede diagnosticar,
y lo que no se registra no se puede auditar**. Un panel sin runbook, o un log
sin quien lo lea, es ruido. Empieza mínimo (métricas del modelo + logs de la
predicción) y crece cuando un incidente demuestre que falta algo — añadir
observabilidad después de un incidente es la señal de que la primera versión
no preguntaba lo correcto.

## Ownership y runbooks

El ciclo necesita un propietario explícito, no "el equipo de ML":

- **Quién decide reentrenar**: una persona nombrada, con autoridad y
  responsabilidad; la decisión no sale de un default en un cron.
- **Qué mirar cada semana**: la revisión semanal es una lista fija —
  drift del input, drift de predicciones, métrica de negocio, incidentes de
  la semana, decisiones de retrain pendientes. Si algo se mira cada semana y
  nadie lo registra, no se está mirando.
- **Cómo se documenta la decisión**: el resultado de cada revisión y cada
  retrain deja rastro escrito, con el disparador, el diagnóstico y la
  decisión. En este proyecto el rastro vive en `harness/progress/` (y el
  histórico en `harness/progress/history.md`): un retrain sin registro es un
  experimento que nadie puede repetir ni auditar. La regla de `ml/deuda-tecnica.md`
  aplica aquí: una decisión sin registro se reinventa o se contradice.

El runbook de cada alerta dice qué hacer, quién lo hace y en cuánto tiempo.
Un runbook que dice "investigar" no es un runbook.

## Práctica

### Playbook de respuesta a incidente de modelo

Frente a una señal de que el modelo en producción se degrada, cuatro pasos en
orden — y el orden importa:

1. **Detectar**: la alerta con umbral calibrado dispara el runbook. Registrar
   hora, señal y evidencia; no empezar a tocar código.
2. **Congelar**: parar el bleeding. Si el modelo activo está empeorando, no
   reentrenar a ciegas encima: si hay una tripleta conocida buena, rollback a
   ella; si no, dejar de cambiar cosas y acotar el alcance del daño.
3. **Diagnosticar**: aplicar las tres preguntas (dato, modelo, mundo) con
   evidencia. El diagnóstico decide la acción; saltárselo es arreglar el
   síntoma equivocado.
4. **Decidir**: retrain, añadir features, rollback o nada — y **documentarlo**
   en `harness/progress/` con el porqué. El incidente cierra con la decisión
   registrada, no con la alerta apagada.

### Cuándo el ciclo no compensa

El ciclo completo (pipeline de datos, re-evaluación, validación, monitoreo,
rollback) tiene coste fijo por iteración. Para modelos de **bajo riesgo** —
batch interno, sin decisión de dinero ni de personas, error tolerable, pocos
consumidores — ese coste puede superar al beneficio de reentrenar:

- Un modelo interno que se reentrena por calendario "por si acaso" gasta
  cómputo y atención sin retorno.
- Un retrain que no va a alterar ninguna decisión material es una
  re-validación con riesgo de regresión y cero ganancia.

La señal honesta es el ROI: si el coste del ciclo (datos + validación +
monitoreo + riesgo de rollback) supera al valor de una decisión correcta, el
modelo correcto es el más simple que cubre el riesgo — o un modelo que no se
reentrena, solo se vigila. La deuda no es reentrenar poco; es mantener el
ciclo completo donde el ciclo no aporta.

## Fuentes

- **Hidden Technical Debt in Machine Learning Systems** — D. Sculley, G. Holt,
  D. Golovin, E. Davydov, T. Phillips, D. Ebner, V. Chaudhary, M. Young,
  J.-F. Crespo, D. Dennison (2015). NIPS 2015 — el paper no está en arXiv.
- **Towards ML Engineering: A Brief History Of TensorFlow Extended (TFX)** —
  R. Breck et al. (2021). arXiv:2010.02013 — https://arxiv.org/abs/2010.02013
- **What's your ML test score? A rubric for ML production systems** — E. Breck
  et al. (2017). NIPS 2017, Reliable ML in the Wild workshop.
- **Continuous Training for Production ML** — D. Baylor et al. (2017).
  arXiv:1706.00331 — https://arxiv.org/abs/1706.00331
