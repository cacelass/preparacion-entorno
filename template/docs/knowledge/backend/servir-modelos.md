# Servir modelos en producción

## El problema del serving

Servir convierte el modelo de artefacto de entrenamiento a pieza de software
con SLA. El balance es triple:

| Recurso | Qué se paga | Presión |
|---------|-------------|---------|
| Latencia | ms por predicción (cola + cómputo + red) | usuarios síncronos |
| Throughput | predicciones/segundo | volumen de peticiones |
| Costo | cómputo ocioso + infra no aprovechada | presupuesto |

Online y batch resuelven problemas distintos:

- **Online**: una petición → una respuesta en ms. Cada request es
  independiente; el SLA es per-petición (p99); la carga varía y el costo se
  paga sobre cómputo ocioso en valles.
- **Batch (offline)**: predicciones sobre un volumen fijo a intervalos. La
  métrica es throughput total y coste por fila; la latencia individual no
  importa. Ideal para scoring nocturno, recomendaciones precomputadas o
  aguas abajo de un pipeline.

Regla: si el consumidor espera síncrono y el modelo está en el camino crítico
de una experiencia, sirve online; si la decisión admite segundos o minutos, el
batch es más barato y más fácil de operar.

**Fallo en producción**: servir todo online "porque es lo moderno" multiplica
el costo sin ganar nada; servir batch algo que el usuario espera síncrono
destruye la UX. El SLA se define antes del despliegue, no después.

## Serialización y portabilidad

El artefacto que entrenas debe reproducir la misma matemática en servicio.
Tres capas de riesgo: formato del modelo, versión de la librería y
preprocesado.

- **joblib/pickle**: formato natural de sklearn. Frágil: serializa código
  Python arbitrario (la clase del estimador) además de los pesos. Cargar con
  otra versión de sklearn rompe o, peor, cambia la salida en silencio. No
  cargues pickles de fuentes no confiables (ejecutan código). Aceptable para
  servir si congelas el entorno exacto.
- **ONNX**: grafo neutro, portable entre frameworks y a runtimes optimizados
  (ONNX Runtime). No cubre preprocesado ni operadores no estándar; valida con
  `onnx.checker` y compara salidas tras la conversión (tolerancia ~1e-4).
- **TorchScript**: grafo serializable de PyTorch, ejecutable sin Python
  (libtorch). Igual que ONNX: exporta y valida contra datos de test.

La consistencia feature/entorno importa tanto como el formato:

```python
# Antes de servir, comprueba que el artefacto emite lo mismo que en training
from joblib import load
model = load("models/model.joblib")
assert max(abs(model.predict(X_check) - y_check)) < 1e-6
```

**Fallo en producción**: modelo correcto con scaler mal aplicado (entrenado
normalizando, servido sin normalizar) produce predicciones válidas pero falsas.
Scaler, encoders y feature_names son parte del artefacto: viajan con el modelo
o con el run que lo generó.

## Batching y dynamic batching

Para modelos pequeños, el overhead por petición (HTTP, parseo, serialización,
buffers, copia GPU) domina sobre el cómputo. Predecir de uno en uno
desaprovecha la paralelización interna de los kernels.

- **Batching estático**: el cliente agrupa N filas y manda una petición con N
  ejemplos. Simple, pero traslada la cola al cliente.
- **Dynamic batching**: el servidor acumula peticiones durante una ventana o
  hasta un tamaño máximo y las procesa juntas. Compromiso entre latencia
  (esperar más llena el batch) y throughput.

Modelo de cola M/M/1 (llegadas Poisson, servicio exponencial):

```
ρ = λ / μ          # utilización; estable si ρ < 1
W = 1 / (μ − λ)    # tiempo medio en el sistema
L = λ · W          # Little's law: L = λ·W
```

Little's law es la que se recuerda: en un sistema estable, la concurrencia
media es el producto del rate de llegadas por el tiempo de residencia. A ρ → 1
la cola crece sin límite; operar cerca de la saturación convierte el p99 en
segundos.

**Fallo en producción**: dynamic batching con ventana fija bajo picos acumula
peticiones y dispara el p99; con solo tamaño máximo y sin timeout, un modelo
lento bajo carga normal nunca completa el batch. La ventana necesita timeout
máximo y el batch máximo se calibra con benchmarks, no a ojo.

## Latencia, percentiles y cola tail

Un SLA de "latencia media < 100 ms" no sirve: la media esconde que el 1% de
los usuarios espera 2 s. Se mide con percentiles:

- **p50**: experiencia típica (mediana).
- **p99**: lo que siente el peor 1%. Domina la percepción de calidad en
  productos interactivos.

El tail viene de: picos de carga, contention del threadpool/GIL, red entre
servicios, cold starts y reintentos de clientes que amplifican la carga en
cascada.

Caché de predicciones: guardar el resultado por key de features. Riesgo con
datos dinámicos: servir una predicción obsoleta. Solo es correcta si la key
captura lo que cambia la salida (features + versión de modelo) y la TTL es
menor que la tasa de cambio del dato. Caché sin invalidación por versión sirve
el modelo viejo tras un deploy.

**Fallo en producción**: medir solo la media y descubrir el tail en el
incidente. Y el "thundering herd": el fallo hace que todos los clientes
reintenten a la vez y la caché fría amplifica el pico hasta tumbar el servicio.

## Optimizaciones de serving y su coste de precisión

| Técnica | Ganancia | Coste de precisión | Cuándo |
|---------|----------|--------------------|--------|
| Cuantización int8 | 2-4x velocidad y RAM | 0-2% calibrado; rompe outliers | modelos grandes validados |
| float16 | mitad de memoria, kernels rápidos | menor rango dinámico | NN con gradiente estable |
| Destilación | estudiante más pequeño y rápido | depende de la brecha teacher→estudiante | teacher ya bueno |
| Pruning | pesos esparsos, menos cómputo | degrada; re-entrenar tras podar | convolucionales, transformers |

Reglas: cuantizar con datos de calibración representativos (no la media),
validar el modelo cuantizado contra el original con el mismo test y un umbral
de degradación acordado de antemano, y guardar ambas versiones en el registry
para decidir en producción.

**Fallo en producción**: cuantizar int8 sin recalibrar satura el 0.1% de
features extremas (las importantes en anomalías) y el modelo falla justo en
los casos raros que importan.

**Multi-adapter y KV-cache**: si sirves varios adapters LoRA alternándose sobre
el mismo contexto (agentes, multi-especialidad), cada switch re-prefillea el
historial. Con **aLoRA** (adapters activados por tokens de invocación) el
prefijo anterior al trigger es reutilizable entre base y adapters, pero la
reutilización real exige alinear el prefix caching del servidor con esa
semántica (vLLM lo soporta). Detalle en `modelos-fundacionales.md`.

## Versionado y rollout

El model registry es la fuente de verdad: cada versión con parámetros, métricas
y artefacto. Desplegar una versión nueva es transaccional:

- **A/B**: dos versiones conviven, el tráfico se reparte controladamente y se
  comparan métricas online. Requiere telemetría por grupo.
- **Canary**: la versión nueva recibe un % pequeño (5-10%), se observa y se
  sube gradualmente. El rollout más seguro para cambios graduales.
- **Blue-green**: dos entornos completos y conmutación de tráfico de golpe.
  Rollback trivial, pero duplica infraestructura.

Rollback: instantáneo y probado. Conserva la versión anterior desplegada y
ensaya el rollback en un simulacro, no en el incidente.

**Fallo en producción**: A/B sin métrica de negocio acordada (comparar solo
latencia, no calidad); canary aprobado "porque no hay errores" sin comparar la
métrica de calidad. La transición de stage del registry a tráfico real debe
ser automatizada y auditable.

## Health checks y degradación

- **Liveness**: ¿está vivo el proceso? Si falla, el orquestador lo reinicia.
  No debe depender de dependencias externas.
- **Readiness**: ¿puede recibir tráfico? Modelo cargado y dependencias listas.
  Si falla, se retira del balanceo.

Cuando el modelo no está cargado (pesos ausentes, carga fallida):

- **503 Service Unavailable** si el servicio no puede cumplir su función.
- **Fallback degradado** (heurística, respuesta por defecto) solo si la
  experiencia puede degradar sin daño y el fallback está probado.

Timeouts y reintentos: un cliente con timeout menor que el p99 genera
reintentos que multiplican la carga. Reintentar solo errores idempotentes (una
predicción pura lo es: mismo input → mismo output) y con jitter + backoff
exponencial.

**Fallo en producción**: liveness que depende del modelo (reinicios en bucle
cuando el modelo tarda en cargar) y el sin-timeout: un modelo degradado que
cuelga la petición deja colgados a todos los clientes.

## Monitorización

- **Prediction drift**: cambios en la distribución de predicciones (p.ej. ratio
  de clases) sin que cambien las features.
- **Data drift**: cambios en la distribución de las features (KS para
  numéricas, chi² para categóricas).
- **Performance decay**: degradación de la métrica objetivo cuando hay ground
  truth, aunque sea retrasado.

{% if use_monitoring %}
Este proyecto incluye `tools/monitor.py`: drift KS/chi² y rendimiento
frente a baseline, vía `make monitor`, con informes en `reports/monitoring/`
(`drift_report.csv`, `drift_report.html`, `performance.csv`). Es la fuente de
salud del modelo desplegado; se ejecuta sobre los mismos datos que sirve la API.
{% endif %}

Alertar sobre lo que tiene acción: "subió el p99" tiene acción (escalar,
optimizar); "hay drift en feature_X" solo si se puede retraer el modelo o
reentrenar. Cada alerta lleva su runbook.

Feedback loops: las predicciones vuelven a ser datos (servir → guardar →
etiquetar → reentrenar). El buen bucle cierra con retención de logs de
predicción + ground truth.

**Fallo en producción**: alertas sin umbral calibrado que se ignoran por ruido,
o drift detectado semanas después sin conexión a ninguna acción.

## Costo

- **Autoscaling**: escala horizontal por cola o CPU. El modelo en memoria hace
  que escalar a 0 no sea gratis (cold start de recarga).
- **GPU vs CPU**: la GPU gana en batch y modelos grandes; en modelos pequeños
  con latencia baja, el overhead de la GPU (transferencia, sync) puede perder
  contra una CPU bien vectorizada. Medir, no asumir.
- **Cold starts**: arrancar un pod con un modelo de GBs tarda segundos. Warm
  pool, precarga en init container, o asumir la latencia en la primera
  petición.

**Fallo en producción**: pagar GPUs ociosas en valle porque el autoscaler solo
mira CPU, o matar nodos por ahorrar y pagarlo en latencia de cold start.

{% if use_api %}
Este proyecto sirve el modelo entrenado vía FastAPI: `api/main.py`, endpoint
`POST /predict`, puerto 8000 con `make serve`. La API arranca aunque no haya
modelo y `/predict` devuelve 503 hasta que `make train` produzca pesos en
`models/`. La guía de la API está en `api.md` de esta carpeta.
{% endif %}

## Fuentes

- Documentación de FastAPI: deployment y deployment en producción (uvicorn/gunicorn).
- Documentación de ONNX Runtime: cuantización y optimización de inferencia.
- Little, J.D.C., "A Proof for the Queuing Formula: L = λW".
- Stanford CS329 (ML Systems), notas de serving: batching, percentiles, tail.
- Documentación de MLflow: Model Registry y versionado de modelos.
