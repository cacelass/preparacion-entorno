# Metaheurísticas: algoritmos genéticos, recocido simulado y búsqueda

Optimizar cuando no hay gradiente, no hay derivada, o el objetivo es una caja
negra (simulación, función no diferenciable, combinatoria). La familia de los
algoritmos evolutivos, el recocido simulado y las búsquedas local/global, con
cuándo usarlos frente a la optimización basada en gradiente
(`matematicas/calculo-optimizacion.md`) y a la optimización de hiperparámetros
(`optimizacion-hiperparametros.md`).

## Cuándo estas técnicas y cuándo no

Son la herramienta correcta cuando:

- **No hay gradiente** disponible (función evaluable pero no diferenciable).
- **El espacio es combinatorio** o mixto (hiperparámetros discretos + continuos,
  orden de features, rutas, asignaciones, grafos).
- **La evaluación es cara y sin ruido informativo** (una simulación, un
  experimento, un cálculo costoso).
- **El objetivo es multimodal**: hay muchos óptimos locales y quieres una
  búsqueda global.

No son la herramienta correcta cuando: tienes gradiente (usa descenso/SGD,
mucho más eficiente), o el problema es convexo (hay métodos exactos), o la
evaluación es barata y puedes probar todo (grid search). Usar un GA para
optimizar una función diferenciable suave es pagar lo caro y lento.

**La regla honesta**: para optimización continua diferenciable, gradiente;
para hiperparámetros con pocas dimensiones, random search o Bayesian
optimization; las metaheurísticas brillan en combinatoria y caja negra, y como
baseline de comparación.

## El marco formal: optimización combinatoria

Minimizar $f: S \to \mathbb{R}$ sobre un espacio discreto $S$ (permutaciones,
subconjuntos, asignaciones) con restricciones. Muchos de estos problemas son
**NP-duros** (TSP, bin packing, job scheduling, feature selection): no existe
algoritmo polinómico conocido y, en la práctica, se resuelven con heurísticas.
El "óptimo" que busca una metaheurística es una **aproximación** con garantías
probabilísticas, no una solución exacta.

La **evaluación** (cuántas llamadas a $f$) es la moneda real: el problema no es
encontrar el mejor punto, es encontrarlo **con el presupuesto de evaluaciones
que tienes**. Cualquier comparación de algoritmos que no iguale el presupuesto
es injusta.

## Búsqueda local: la base de todo

Toda metaheurística parte de la **búsqueda local**: mantener un candidato $x$,
probar vecinos $x' \in N(x)$, quedarse con el mejor.

```
x ← inicial aleatoria
repetir:
  x' ← el mejor vecino de x
  si f(x') >= f(x): devolver x   (máximo local)
  x ← x'
```

- La **definición de vecindad** es la decisión de diseño más importante: define
  qué movimientos son posibles (swap, insert, 2-opt para rutas, flip de bits).
  Una vecindad pobre convierte el espacio en inaccesible.
- **First-improvement vs best-improvement**: aceptar el primer vecino que
  mejora es más barato y a menudo igual de bueno que escanear todos.
- Problema estructural: converge al **primer máximo local**. Toda la familia de
  metaheurísticas existe para escapar de ahí — con aleatoriedad controlada
  (recocido), con memoria (tabú), con población (GA/PSO), o con reinicios
  (multistart, ILS).

## Recocido simulado (simulated annealing)

Inspirado en el recocido de metales: aceptar **a veces** movimientos que
empeoran, con probabilidad que decrece con una "temperatura":

$$P(\text{aceptar } x') = \begin{cases} 1 & f(x') \ge f(x) \\ e^{(f(x') - f(x))/T} & f(x') < f(x)\end{cases}$$

- **Temperatura alta** → acepta casi todo (exploración amplia).
- **Temperatura baja** → solo acepta mejoras (converge).
- El esquema de enfriamiento importa más que la vecindad: un enfriamiento
  geométrico $T_{k+1} = \alpha T_k$ con $\alpha \approx 0.9$ es el punto de
  partida sano. El **recalentamiento** (reinicios de temperatura) ayuda a
  escapar de valles profundos.

Garantía teórica (annealing inhomogéneo): si el enfriamiento es lo bastante
lento, converge al óptimo global **en probabilidad**. En la práctica el
enfriamiento necesario es demasiado lento; se usa como heurística robusta.
Es la elección natural cuando tienes **un solo candidato** y una vecindad bien
definida (rutas, asignaciones, permutaciones).

## Algoritmos genéticos (GA)

Inspirados en la selección natural. Mantienen una **población** de candidatos
(genes), evaluados por una función de fitness $f$.

### Operadores

1. **Selección**: los mejores pasan a reproducirse. Variantes: roulette
   (proporcional a fitness), **torneo** (muestrear k, quedarse con el mejor —
   robusto a escalas de fitness), ranking, elitismo (conservar los mejores de
   la generación anterior para no perder lo ya encontrado).
2. **Cruce** (crossover): combinar dos padres → hijos.
   - *Punto de corte*: partir cada padre en 2 y mezclar (espacios binarios).
   - *Uniforme*: por gen, con probabilidad.
   - *Permutaciones*: order crossover (OX), PMX — **que respetan que no haya
     genes repetidos**. Un cruce naíf sobre una permutación genera hijos
     inválidos (repite ciudades en una ruta) y el GA busca en un espacio que
     no existe.
3. **Mutación**: alterar aleatoriamente con probabilidad baja (mantiene
   diversidad y evita convergencia prematura). Para permutaciones: swap o
   inversión de un segmento.
4. **Reemplazo**: la nueva generación sustituye (elitismo: conservar el mejor).

### Teorema del esquema (Holland)

Los bloques de genes cortos, de bajo orden y de fitness superior al promedio se
propagan exponencialmente en la siguiente generación:

$$m(H, t+1) \ge m(H, t) \cdot \frac{\text{fitness}(H)}{\bar{f}} \cdot (1 - \text{perturbación}).$$

Es la justificación clásica de por qué el cruce funciona. En la práctica, los
GA funcionan razonablemente en espacios combinatorios y mal especificados, y son
fáciles de paralelizar (cada individuo se evalúa solo — hasta una simulación
por núcleo).

### Parámetros que de verdad importan

- Tamaño de población y número de generaciones (presupuesto de evaluaciones).
- Probabilidad de mutación (baja, ~1/gen); si la población converge y el
  fitness no mejora, subir la mutación.
- Presión selectiva: demasiada → convergencia prematura; poca → ruido.
- **Tamaño de población pequeño con muchas generaciones suele ser mejor que
  población grande con pocas generaciones** para presupuestos fijos.

## Otras metaheurísticas que conviene conocer

- **Búsqueda tabú**: búsqueda local + memoria de los últimos movimientos
  (tabú) para no volver a estados recientes; buena en combinatoria dura
  (scheduling, asignación). La lista tabú y el criterio de aspiración (permitir
  un movimiento tabú si produce el mejor global) son las piezas finas.
- **ILS (iterated local search)**: repetir búsqueda local desde una
  **perturbación** del mejor hallado. Muy competitiva en optimización
  combinatoria con poco código.
- **Particle Swarm Optimization (PSO)**: población de partículas con velocidad;
  cada una sigue su mejor personal y el mejor global:

  $$v_{t+1} = w v_t + c_1 r_1 (p_{best} - x_t) + c_2 r_2 (g_{best} - x_t), \qquad x_{t+1} = x_t + v_{t+1}.$$

  Simple, pocos parámetros ($w, c_1, c_2$), buena en espacios continuos de
  baja/media dimensión.
- **Differential Evolution (DE)**: mutación por diferencia de vectores de la
  población; excelente para optimización continua de caja negra, robusta y sin
  gradiente. La alternativa moderna al GA en espacios continuos.
- **Ant Colony Optimization (ACO)**: para rutas/grafos; las "hormigas" dejan
  feromona proporcional a la calidad de las soluciones.

## Búsqueda en espacio de hiperparámetros y configuraciones

Para **hiperparámetros**, la metaheurística adecuada suele ser mejor:
- **Random search** (Bergstra & Bengio) gana a grid search en la práctica
  porque la función de rendimiento es de baja dimensión efectiva: cada variable
  importante tiene muchas más chances de probarse con valores distintos.
- **Bayesian optimization** (GP/surrogate) explota la estructura y necesita
  pocas evaluaciones, ideal cuando cada evaluación es cara. Ver
  `optimizacion-hiperparametros.md` (dskit lo integra con Optuna).
- Un GA tiene sentido cuando el espacio es **mixto y estructurado** (arquitectura
  de red + tasa de aprendizaje + decisiones discretas interdependientes), no para
  el caso típico de Optuna.

## Evaluación honesta

- El presupuesto de **evaluaciones** (nº de llamadas a $f$) es la moneda real;
  reportar curva de mejor fitness vs evaluaciones, no un número final.
- Un GA con 10× evaluaciones casi siempre "gana" — el arte es comparar con
  presupuesto igualado contra baselines (random search, búsqueda local con
  reinicios).
- **Seeds**: reportar media y dispersión sobre varias semillas; la aleatoriedad
  de la inicialización y de la mutación domina la varianza.
- **Comparar contra lo trivial**: si random search con el mismo presupuesto
  llega cerca, la metaheurística no está aportando — y el problema quizá no lo
  necesitaba.

## Cómo se rompe (checklist para el `lider`)

- **Convergencia prematura**: la población colapsa a un óptimo local y la
  mutación no reinyecta diversidad. Síntoma: fitness plano durante generaciones.
- **Representación rota**: cruce/mutación que generan candidatos inválidos
  (permutaciones con repetidos, constraints violadas). Si el operador no
  respeta las constraints, el GA busca en un espacio que no existe.
- **Evaluación con ruido**: si $f$ tiene ruido, la selección por fitness ordena
  ruido; se necesita reevaluación o tolerancia.
- **Escala**: cada evaluación cuesta una simulación de horas → presupuesto
  miserable; considerar surrogate (Bayesian) en vez de GA.
- **Fitness mal especificado**: el GA optimiza la métrica que le das, no la que
  quieres (mismo reward hacking que en RL).
- **Mala vecindad**: búsqueda local sobre una vecindad que no conecta el
  espacio = el algoritmo se queda donde empezó.
- **Parámetros sin calibrar**: un GA con población/mutación mal elegidos
  rinde peor que random search; calibrar con un problema pequeño antes.

## Dónde encaja en dskit

Este fichero cubre la teoría; dskit ya integra Optuna para hiperparámetros y el
agente `tuning`. Las metaheurísticas entran cuando el problema del proyecto es
combinatorio o de caja negra — el `lider` consulta aquí y `rag refresh` lo
mantiene con los topics de `sources.json`.
