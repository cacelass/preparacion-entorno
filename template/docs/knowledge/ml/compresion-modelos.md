# Compresión de modelos

## Por qué comprimir

El serving paga en tres monedas: latencia, memoria y coste. La compresión las
reduce todas a la vez: menos parámetros → menos memoria por instancia → menos
bandwidth en los kernels → menos latencia y menos cómputo contratado. El coste
se paga en un cuarto término, la precisión, y es el único que hay que vigilar.
Ver [servir-modelos.md](../backend/servir-modelos.md).

| Dimensión | Qué cambia con la compresión | Dónde duele |
|---|---|---|
| Tamaño | memoria residente, footprint del artefacto | deploy, cold start, red |
| Velocidad | latencia por predicción, throughput | usuarios síncronos, batch |
| Coste | infraestructura, GPU/CPU contratadas | valles de carga |
| Precisión | métrica del modelo, calibración | siempre que se reduce |

El triángulo precisión/velocidad/tamaño: puedes ganar dos vértices solo si
aceptas pagar el tercero. Comprimir bien es elegir qué vértice se sacrifica y
cuánto, con un umbral acordado antes de empezar — no descubrir la pérdida al
final.

Cuándo la compresión es casi gratis: modelos sobreparametrizados con capacidad
sobrante (convnets, transformers, MLPs grandes), pesos con poca entropía, y
tareas donde el margen sobre el umbral es amplio. Cuándo duele: modelos
pequeños que ya operan en el borde de su capacidad, tareas con clases raras o
outliers decisivos, y despliegues donde "cada punto de accuracy" se traduce
directamente en coste o riesgo. En un modelo pequeño, comprimir es devolver
capacidad que no sobra.

{% if ml_type == 'redes_neuronales' %}
## Cuantización

Reducir la precisión numérica de los tensores. Es la palanca más barata: no
toca la arquitectura, cambia solo la representación de los números.

### De FP32 a FP16/BF16

FP16 y BF16 ocupan la mitad que FP32 y duplican el throughput de los kernels
de GPU en cómputo denso. Cuándo basta FP16/BF16:

- **Solo almacenamiento**: los pesos se guardan en 16 bits y se suben a FP32
  en el forward. Casi gratis si los pesos están dentro del rango de FP16
  (~±65504); la pérdida relativa (~1e-3) es despreciable frente a la
  cuantización de activaciones.
- **Inferencia**: mantiene el rango dinámico de las activaciones y acelera
  los kernels mixtos.

Diferencia entre ambos:

| Tipo | Exponente | Mantisa | Rango | Problema típico |
|---|---|---|---|---|
| FP16 | 5 bits | 10 bits | estrecho | overflow en activaciones grandes |
| BF16 | 8 bits | 7 bits | = FP32 | precisión relativa menor |

BF16 hereda el rango de FP32 (no desborda) pero pierde mantisa; FP16 gana
precisión relativa pero desborda. Para pesos dentro de rango, FP16 suele ser
mejor; para activaciones con picos grandes, BF16 no desborda. FP16/BF16 no
necesitan calibración: es un cambio de formato sin parámetros que ajustar.

### INT8: escala y zero-point

INT8 mapea el rango de cada tensor a 256 niveles. La relación entre el valor
real $r$ y el cuantizado $q$ es afín:

$$ r = s\,(q - z), \qquad q = \mathrm{round}\!\left(\frac{r}{s}\right) + z $$

con escala $s = (r_{\max} - r_{\min})/(q_{\max} - q_{\min})$ y zero-point
$z = q_{\min} - r_{\min}/s$. Dos variantes:

- **Simétrica** ($z = 0$): $s = \max(|r|)/127$. Sin offset; la multiplicación
  de matrices queda en enteros puros, más rápida en hardware.
- **Asimétrica** ($z \ne 0$): captura tensores cuyo rango no es simétrico
  (activaciones ReLU, que viven en $r \ge 0$). La variante simétrica de un
  tensor con rango $[0, a]$ desperdicia la mitad de los niveles INT8; el
  zero-point los aprovecha.

El error de cuantización es $\Delta = s/2$ en el peor caso por valor, y crece
con el rango del tensor: un tensor con un par de outliers grandes estira $s$ y
aplasta la precisión de todo lo demás. La elección de $s$ es el corazón del
problema.

### Por tensor vs por canal

| Granularidad | Qué escala comparte | Coste | Cuándo |
|---|---|---|---|
| Por tensor | una $s, z$ para toda la matriz | mínimo | activaciones (cómputo denso) |
| Por canal | una $s, z$ por columna/fila de pesos | más parámetros | pesos de lineales/convolucionales |

Los pesos por canal capturan distribuciones distintas por filtro; las
activaciones suelen cuantizarse por tensor (o por canal en CNNs) porque
per-canal en activaciones rompe el cómputo matricial. El salto de per-tensor a
per-channel en pesos es la corrección de menor esfuerzo cuando la cuantización
degrada.

### PTQ vs QAT

**Post-Training Quantization (PTQ).** Cuantiza un modelo ya entrenado. Los
pesos se mapean directamente; las activaciones necesitan un **dataset de
calibración**: un subconjunto representativo (centenares de muestras) que se
pasa por el modelo en FP32 solo para recoger estadísticas de activaciones
(min/max, percentiles, o mínima divergencia KL entre la distribución FP32 y la
INT8) y fijar $s, z$ por capa. GPTQ (arXiv:2210.17323) mejora la cuantización
de pesos con un ajuste de segundo orden por bloques.

**Quantization-Aware Training (QAT).** Entrena con operaciones simuladas: en
el forward los tensores se cuantizan y descuantizan (round + scale) para que el
optimizador aprenda sobre el error real; el backward usa el **straight-through
estimator** (el gradiente atraviesa el redondeo como identidad). El modelo
"aprende a ser cuantizado": absorbe la pérdida en los pesos.

| | PTQ | QAT |
|---|---|---|
| Coste | minutos (una pasada de calibración) | un entrenamiento completo |
| Datos | dataset de calibración | el dataset de entrenamiento |
| Pérdida típica INT8 | 0-2%, más si hay outliers | ~0% recuperable |
| Cuándo | primer intento, modelos grandes | PTQ no llega al umbral |

Nota: QLoRA es el pariente de QAT para fine-tuning — cuantizar el base a 4 bits
(NF4) y entrenar solo los adapters LoRA encima, en vez de cuantizar un modelo ya
entrenado. Es cuantización para **entrenamiento**, no para inferencia; detalle
en `modelos-fundacionales.md`.

### El patrón de degradación

La degradación no es uniforme ni en magnitud ni en dónde aparece:

- **Colas pesadas**: activaciones con outliers grandes estiran $s$ y aplastan
  el grueso de los valores; la pérdida se concentra en la precisión de los
  valores típicos, no de los outliers (que ya eran extremos). Clipping del
  percentil 99-99.9 en calibración recorre ese trade-off.
- **Capas sensibles**: las primeras capas (multiplican el input crudo) y las
  últimas (salen a la pérdida) degradan más que las intermedias. Cuantizar
  solo esas en FP16 y el resto en INT8 (cuantización mixta) recupera casi
  todo a costa de complejidad de infraestructura.
- **Modelos pequeños**: sin redundancia que absorba el error, la cuantización
  rompe antes. Si el modelo está en el filo, la pérdida de INT8 no es un
  0.1%: es un salto de régimen.

## Destilación

Entrenar un modelo pequeño (student) contra las salidas de uno grande
(teacher). El objetivo no es imitar la clase, sino la **distribución de
probabilidades** que el teacher emite.

### Soft targets con temperatura

El teacher produce logits $z_i$; los soft targets se suavizan con la
temperatura $T$:

$$ p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} $$

Con $T \to 1$ es el softmax normal; con $T$ alto la distribución se aplana y
expone la "dark knowledge": las probabilidades relativas entre las clases
erróneas (un "gato" se confunde más con un "perro" que con un "camión"). El
student optimiza la pérdida dura con las etiquetas más la pérdida blanda contra
los soft targets:

$$ \mathcal{L} = \alpha\,\mathrm{CE}(y, p_s) + (1-\alpha)\,T^2\,
   \mathrm{KL}\!\left(p_t(T) \,\|\, p_s(T)\right) $$

El factor $T^2$ es necesario: el gradiente del término KL respecto a cada
logit del student escala con $1/T^2$, y el factor lo compensa para que
comparar temperaturas sea justo. Bajar $T$ acerca el problema al de etiquetas
duras; subirlo enfatiza la estructura entre clases. La ganancia clave: el
student aprende *por qué* el teacher duda, no solo *qué* decidiría — la
información está en los pesos de las alternativas.

### Cuándo funciona

- **Capacity gap moderado**: el student debe poder representar la función del
  teacher. Con brecha enorme (teacher de miles de millones, student de un
  millón), destilar transfiere menos que entrenar el student directo.
- **Datos suficientes**: el student no inventa la estructura que no ve; con
  pocos datos, los soft targets son lo único rico que tiene.
- **Teacher ya bueno**: destilar de un teacher mediocre hereda sus errores;
  el estudiante copia la distribución, incluida la mala.

### Auto-destilación y coste

La auto-destilación usa el mismo modelo como teacher (un checkpoint tardío de
su propio entrenamiento, la EMA de sus pesos, o una versión ancha del mismo
student) — evita entrenar dos arquitecturas. El coste es real: **se entrenan
dos modelos** (teacher completo y student), y el beneficio solo se materializa
si el student se despliega en el camino crítico. La destilación no reduce el
coste de entrenamiento; traslada la compresión a inferencia.

## Pruning

Eliminar parámetros. Poda por magnitud: los pesos con $|w|$ más pequeños se
ponen a cero, asumiendo que contribuyen poco a la salida. La suposición es
parcialmente falsa (un peso pequeño puede ser esencial), por eso la poda se
acompaña de **re-entrenamiento**: podar, retrain breve, repetir (iterative
pruning) en lugar de podar todo de golpe.

### Estructurada vs no estructurada

- **No estructurada**: pesos individuales a cero sin patrón. La sparsity
  resultante (p.ej. 90% de ceros) solo ahorra memoria si el formato la sabe
  explotar; el cómputo no se acelera porque el hardware multiplica los ceros
  igual que los valores.
- **Estructurada**: se eliminan unidades enteras — canales de convolución,
  neuronas, filas de la matriz de pesos, cabezas de atención. Cambia la forma
  de los tensores y por tanto acelera de verdad: menos multiplicaciones, menos
  memoria, kernels más densos.

| | No estructurada | Estructurada |
|---|---|---|
| Sparsity alcanzable | 90-95% | 30-70% |
| Ahorro real | memoria (si hay formato sparse) | memoria + latencia |
| Riesgo de degradación | menor (poda fina) | mayor (elimina unidades completas) |

{% if nn_model in ['CNN1D', 'ResNet'] %}
En convolucionales la poda estructurada natural es por filtro/canal: se elimina
un mapa de activación entero y se comprime el siguiente kernel. La métrica de
importancia es típicamente la norma $\ell_2$ del filtro o su impacto medido
sobre la salida.
{% endif %}

### Sparsity N:M

El hardware moderno (GPUs Ampere+) acelera un patrón estructurado específico:
**sparsity 2:4** — de cada 4 pesos consecutivos, exactamente 2 son cero. Los
kernels especializados ignoran los ceros y multiplican el throughput. Es un
punto intermedio: tan regular como para acelerar, tan denso como para no
perder demasiada capacidad. La poda N:M se hace normalmente por magnitud
dentro de cada grupo de N.

### Lottery ticket hypothesis

La hipótesis (Frankle & Carbin) afirma que dentro de una red sobredimensionada
existe una subred (el "ticket ganador") que, entrenada desde su inicialización
original, alcanza la precisión de la red completa en no más iteraciones.
Implica que podar no es solo quitar capacidad sobrante, sino **encontrar
arquitectura**: la máscara importa tanto como los pesos. En la práctica es una
búsqueda costosa y su generalización es debatida; su legado operativo es que la
poda temprana puede ser una meta-búsqueda de arquitectura, no solo un
postproceso.

### Entrenamiento con máscaras

La forma estándar de podar y seguir entrenando: se mantiene la máscara binaria
$M$ (1 si el peso sobrevive), el forward y el backward se ejecutan sobre la red
completa pero el gradiente solo fluye por los pesos con $M=1$:

```python
# forward y backward sobre pesos completos; la máscara decide qué se actualiza
w.data.mul_(mask)          # aplicar sparsity en cada paso
loss.backward()
w.grad.mul_(mask)          # el gradiente no toca pesos podados
optimizer.step()
```

El peso podado no se "desconecta": el gradiente se anula, el optimizador no lo
mueve y el valor queda congelado en cero. Reintroducir pesos requiere el
esquema contrario (dense-sparse-dense) y es un hiperparámetro más que rara vez
paga.

## Combinación: prune → quantize → distill

Las técnicas se componen. El orden típico, y por qué:

1. **Prune** primero: elimina estructura redundante y retrain. Cuantizar antes
   de podar desperdicia precisión en parámetros que luego se borran.
2. **Quantize** después: el modelo ya podado tiene menos capacidad sobrante y
   la cuantización se valida sobre la arquitectura final.
3. **Distill** al final, o al principio si cambias de arquitectura: si el
   student es otra arquitectura, se destila desde el modelo original en
   precisión plena *antes* de comprimirlo; si solo se comprime la misma
   arquitectura, el teacher útil es el modelo no comprimido.

Este es el pipeline de *Deep Compression* (Han et al.): podar → cuantizar →
codificación de Huffman de los pesos, con re-entrenamiento tras cada etapa. La
destilación es ortogonal: decide *qué arquitectura* se despliega; la poda y la
cuantización deciden *cómo* se comprime esa arquitectura.

### Medición en cada paso

Cada etapa es un experimento, no un trámite. Registro obligatorio en cada una:

| Etapa | Métricas | Criterio |
|---|---|---|
| Antes | baseline: accuracy + latencia + tamaño | referencia |
| Tras poda | sparsity %, accuracy, latencia | degradación < umbral |
| Tras cuantización | tamaño MB, accuracy, latencia | degradación < umbral |
| Tras destilación | tamaño, accuracy, calibración | student ≥ umbral |

Además de la accuracy: **la calibración cambia** — comprimir modifica la
distribución de las probabilidades predichas (sobreconfianza o subconfianza
que no aparece en accuracy) y eso se ve en el diagrama de confianza, no en la
curva de error. Ver [gestion-incertidumbre.md](gestion-incertidumbre.md).
{% endif %}

## Evaluación post-compresión

Un modelo comprimido no es "el mismo modelo más pequeño": es un modelo distinto
que solo se parece al original en promedio. Evaluarlo solo por la accuracy
global es evaluarlo en su mejor luz.

### No solo accuracy

- **Calibración**: la probabilidad predicha ya no es la del original. Mide
  ECE/Brier antes y después; la compresión puede empujar las confianzas sin
  mover el accuracy (ver [gestion-incertidumbre.md](gestion-incertidumbre.md)).
- **Latencia real**: mide en el hardware de destino con el runtime de
  producción (ONNX Runtime, TensorRT, TorchScript), no en el ordenador de
  desarrollo. Una cuantización que no acelera el kernel concreto no vale nada.
- **Tamaño del artefacto**: bytes serializados + runtime necesario, no solo
  parámetros.

### Comportamiento en slices raros

Los errores de compresión no son uniformes: se concentran donde la capacidad
sobrante del modelo ya era escasa. Las zonas de alto riesgo:

- Outliers de features y colas de distribución (los outliers estiran la escala
  de cuantización).
- Clases raras y slices poco poblados — la pérdida relativa es mayor.
- Entradas OOD, donde el modelo ya confiaba mal y la compresión añade ruido.

Mide por slices el *cambio* frente al original: una caída del 1% global puede
ser una caída del 10% en el slice de la clase minoritaria.

### El test de regresión

La compresión entra como cualquier otro cambio de código: con un test de
regresión. Un set golden de casos con predicción esperada (o tolerancia de
diferencia frente al original) y umbral acordado:

- **Test de paridad**: para un set fijo de entradas, la predicción del
  comprimido difiere del original menos que un umbral ($\|p_c - p_o\|_\infty
  < \epsilon$).
- **Test de calidad**: la métrica de negocio sobre el test de evaluación se
  mantiene por encima del umbral acordado antes de comprimir.
- **Test de calibración**: ECE dentro de banda, sobre el set de validación.

Estos tres se ejecutan en CI/agenda cada vez que se regenera el artefacto
comprimido, y son la puerta de entrada al serving (ver
[servir-modelos.md](../backend/servir-modelos.md)).

## Práctica: la escalera de compresión

{% if ml_type == 'redes_neuronales' %}
1. **PTQ INT8** con calibración representativa. Coste: minutos. Si la
   degradación está bajo el umbral acordado, termina aquí.
2. **QAT** si PTQ no llega: un entrenamiento, recupera casi toda la pérdida.
3. **Cuantización mixta o FP16/BF16** en las capas sensibles si el problema es
   localizado (primeras/últimas capas).
4. **Destilación o poda** si el problema es de arquitectura (tamaño/latencia):
   destilar a otra arquitectura o podar la actual y retrain.

Cada escalón solo se sube si el anterior no alcanzó el umbral, y cada escalón
re-ejecuta la evaluación post-compresión completa.
{% endif %}

## Cuándo NO comprimir

- **Modelos pequeños**: ya viven en el borde de su capacidad; la compresión
  devuelve precisión que no sobra. Un MLP de 10k parámetros no se comprime.
- **Tareas donde cada punto cuenta**: diagnóstico, scoring de riesgo, decisión
  sobre colas de distribución — el margen de error no admite el 0.5-2% que la
  compresión cuesta.
- **El serving no es el cuello**: si la latencia la domina la red, la
  serialización o el preprocesado, comprimir el modelo no mueve el SLA.
- **Coste de oportunidad**: la complejidad de cuantizar (calibración, QAT,
  tooling) tiene un coste de mantenimiento; comprimir un modelo que se sirve
  en batch nocturno sobre CPU barata puede no recuperar ese coste.

{% if use_api %}
## Servir el artefacto comprimido

El artefacto comprimido se sirve igual que el original, con tres añadidos:

1. **Registro doble**: guarda en el registry la versión comprimida y la
   original con sus métricas de evaluación; en producción se sirve la
   comprimida y la original queda como fallback y para el test de paridad.
2. **Preprocesado idéntico**: la compresión no toca scalers ni encoders; el
   pipeline de features del artefacto original se reutiliza tal cual.
3. **Validación en el endpoint**: el `/predict` del proyecto sirve el
   artefacto que encuentra en `models/`; comprueba que la versión comprimida
   está presente y con su test de paridad aprobado antes de desplegarla como
   predeterminada. La guía del endpoint está en
   [api.md](../backend/api.md) de este corpus.

Si el runtime de inferencia es distinto (ONNX Runtime/TensorRT para INT8), el
artefacto comprimido se serializa en ese formato y el endpoint lo carga con ese
runtime; la doble validación (paridad + calidad) es la que evita servir un
modelo que corre rápido pero predice mal.
{% endif %}

## Fuentes

- Hinton, G., Vinyals, O., Dean, J., "Distilling the Knowledge in a Neural
  Network" — arXiv:1503.02531 — https://arxiv.org/abs/1503.02531
- Jacob, B., et al., "Quantization and Training of Neural Networks for
  Efficient Integer-Arithmetic-Only Inference" — arXiv:1712.05877 —
  https://arxiv.org/abs/1712.05877
- Han, S., Mao, H., Dally, W. J., "Deep Compression: Compressing Deep Neural
  Networks with Pruning, Trained Quantization and Huffman Coding" —
  arXiv:1510.00149 — https://arxiv.org/abs/1510.00149
- Frantar, E., et al., "GPTQ: Accurate Post-Training Quantization for
  Generative Pre-trained Transformers" — arXiv:2210.17323 —
  https://arxiv.org/abs/2210.17323
- Frankle, J., Carbin, M., "The Lottery Ticket Hypothesis: Finding Sparse,
  Trainable Neural Networks" — arXiv:1803.03635 —
  https://arxiv.org/abs/1803.03635
