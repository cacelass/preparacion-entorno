# Aprendizaje por refuerzo (RL)

Aprender **qué hacer** a partir de consecuencias, no de etiquetas. El marco del
MDP, cómo se resuelve exactamente (value/policy iteration), cómo se aproxima a
escala (DQN, actor-critic, PPO/SAC), y los fallos reales: recompensa
mal especificada, inestabilidad del entrenamiento, no estacionariedad y
evaluación que no mide lo que importa. Complementa a `supervisado.md` (aquí no
hay etiqueta de "respuesta correcta", hay consecuencias diferidas) y a
`redes-neuronales.md` (el optimizador y las arquitecturas).

## Cuándo es el marco correcto (y cuándo no)

RL resuelve problemas con tres ingredientes:

1. **Decisiones secuenciales**: lo que hago ahora afecta a lo que puedo hacer
   después (no es una clasificación one-shot).
2. **Consecuencia diferida**: el efecto de una acción puede no verse hasta
   muchos pasos después (crédito diluido en el tiempo).
3. **Exploración**: necesito probar acciones subóptimas para descubrir las
   mejores.

No es RL si: hay etiquetas y puedes predecir en un solo paso (es supervisado);
o no hay agente que actúe (es forecasting). Forzar RL a un problema supervisado
es pagar toda su complejidad sin ganar nada.

## El MDP: la notación que hay que dominar

Un proceso de decisión de Markov es la tupla $(S, A, P, R, \gamma)$:

- $S$: conjunto de estados.
- $A$: conjunto de acciones.
- $P(s' | s, a)$: dinámica (probabilidad de pasar a $s'$ tras actuar $a$ en $s$).
- $R(s, a, s')$: recompensa inmediata.
- $\gamma \in [0, 1)$: factor de descuento.

La **propiedad de Markov**: $P(s' | s, a) = P(s' | s, a, \text{historia})$ — el
estado condensa todo lo relevante del pasado. Si tu estado no es suficiente
(p.ej. no ves la velocidad en un problema de control), el MDP está mal
especificado y el RL aprenderá mal, por mucho algoritmo que uses.

Una **política** $\pi(a | s)$ es la regla de decisión. El objetivo es maximizar
el retorno esperado descontado:

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}, \qquad J(\pi) = \mathbb{E}_{s_0 \sim d_0, a_t \sim \pi}\left[G_0\right].$$

### Value functions

- **Función de valor** $V^\pi(s)$: retorno esperado desde $s$ siguiendo $\pi$.
- **Función de acción-valor** $Q^\pi(s, a)$: retorno esperado tomando $a$ en $s$
  y luego siguiendo $\pi$.

Ambas cumplen la **ecuación de Bellman**:

$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V^\pi(s')\right].$$

Es un **punto fijo**: la solución es la función que satisface la igualdad, no
un gradiente. De ahí salen los métodos tabulares exactos.

## Métodos tabulares (el terreno donde todo es exacto y comprobable)

Con $S$ y $A$ pequeños se puede resolver **exactamente** — es la base para
entender qué aproxima todo lo demás.

### Value iteration

Actualiza $V$ hasta el punto fijo de la **ecuación de optimalidad de Bellman**:

$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V_k(s')\right].$$

Converge a $V^*$ (la función de valor óptima) de forma lineal en $\gamma$. La
política greedy sobre $V^*$ es óptima.

### Policy iteration

Alterna dos pasos:

1. **Evaluación de política**: resolver $V^\pi$ (iterar Bellman hasta converger).
2. **Mejora**: $\pi'(s) = \arg\max_a Q^\pi(s, a)$.

Cada mejora produce una política estrictamente mejor (o igual). Converge en un
número finito de pasos (hay finitas políticas), normalmente mucho antes que
value iteration. En la práctica, value iteration suele ser más simple de
implementar; policy iteration converge en menos iteraciones pero cada una es
más cara.

### Temporal Difference (TD) y SARSA

Cuando $P$ y $R$ se desconocen (lo habitual), se aprende de **experiencia**:

$$V(s_t) \leftarrow V(s_t) + \alpha\left[\underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{target}} - V(s_t)\right].$$

El término entre corchetes es el **error TD**: lo que acabamos de observar
contra lo que predecíamos. **SARSA** usa la misma idea con $Q$ y la acción que
realmente se tomó (on-policy); **Q-learning** usa la acción greedy (off-policy).
La diferencia no es cosmética:

| | Q-learning (off-policy) | SARSA (on-policy) |
|---|---|---|
| Target | greedy sobre $Q$ | la acción que se tomó |
| Aprende | la política óptima | la política que sigue |
| Reutiliza datos viejos | Sí | No |
| Con recompensas peligrosas | Puede aprender a "caerse" | Aprende a rodear el peligro |

- **Q-learning** aprende la política óptima aunque explore con otra (off-policy);
  puede usar experiencias viejas y de otros agentes.
- **SARSA** aprende la política que realmente sigue; si la recompensa es
  peligrosa (un precipicio), SARSA aprende a rodearlo, Q-learning a veces
  aprende a caerse porque su target es greedy.

### El dilema exploración-explotación

$\epsilon$-greedy: con probabilidad $\epsilon$ tomar una acción aleatoria, con
$1 - \epsilon$ la greedy. No es un detalle: sin exploración el agente confirma
lo que ya cree. La teoría del bandido (UCB, Thompson) da reglas con garantías
de regret; en RL práctico $\epsilon$-greedy con decaimiento sigue siendo la
opción por defecto.

**Multi-armed bandit** es el RL de un solo paso y es el lugar donde probar la
exploración de forma limpia: UCB elige $a_t = \arg\max_a \left(\hat{\mu}_a +
\sqrt{\frac{2\ln t}{n_a}}\right)$ — explota lo conocido, explora lo poco
probado, con cota de regret $O(\sqrt{KT})$. Muchos "problemas de RL" se
resuelven mejor como bandits si la decisión no es secuencial.

## Aproximación a escala: del tabular al gradiente

Con estados continuos o enormes, $Q$ o $\pi$ se representan con una red. Tres
familias:

### Value-based (DQN)

Aproxima $Q^*$ con una red $Q_\theta(s, a)$ minimizando el error TD:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\right)^2\right].$$

Tres trucos sin los cuales DQN no converge:

- **Red target** $\theta^-$: se congela y se actualiza cada N pasos. Sin esto,
  el target cambia con los parámetros que estás entrenando — persigues un
  objetivo en movimiento.
- **Experience replay**: guardas transiciones y muestreas de un buffer. Rompe
  la correlación entre transiciones consecutivas (que rompen el supuesto del
  gradiente estocástico) y reutiliza datos.
- **Double Q**: usar $\arg\max_a Q_\theta(s', a)$ para elegir pero $Q_{\theta^-}$
  para valorar, y así no sobreestimar sistemáticamente el valor del mejor par.

Variantes (Rainbow) apilan mejoras: dueling (separar ventaja y valor),
prioritized replay, distributional Q. Para espacios de acción discretos y
medianos, value-based sigue siendo competitivo y barato.

### Policy-gradient

Optimiza directamente $J(\pi_\theta)$ con el **teorema del gradiente de la
política**: para una política diferenciable $\pi_\theta$,

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s)\, Q^{\pi_\theta}(s, a)\right].$$

El truco de la **ventaja** resta una línea base (normalmente $V$) para reducir
varianza sin sesgo:

$$\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\, A(s_t, a_t)\right],$$

donde $A(s, a) = Q(s,a) - V(s)$ es la **ventaja** (cómo de mejor que la media es
esa acción). **REINFORCE** es el caso con $A = G_t$ (retorno completo); funciona
pero su varianza es enorme. Funciona con acciones continuas (que value-based no
toca bien), pero sufre de **alta varianza**: la estimación del gradiente
depende de la fortuna de las trayectorias muestreadas.

### Actor-critic (A2C/A3C, PPO, SAC)

Combina ambos: el **actor** $\pi_\theta$ y el **crítico** $V_\phi$ que estima la
ventaja. El crítico reduce la varianza del gradiente del actor.

- **GAE** (Generalized Advantage Estimation): ventaja como media ponderada de
  errores TD a varios horizontes $\lambda$ — el $\lambda$ controla el sesgo vs
  varianza (λ→1: retorno completo, alto sesgo bajo; λ→0: TD puro, alta varianza).
- **PPO** corta el ratio de importancia $\rho_t = \frac{\pi_\theta(a_t|s_t)}
  {\pi_{\theta_{old}}(a_t|s_t)}$ con un clip:

  $$\mathcal{L}^{clip} = \mathbb{E}\left[\min(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t)\right].$$

  Si la nueva política se aleja demasiado de la vieja, el gradiente se trunca.
  Es la elección por defecto más robusta: razonable en casi todo.
- **SAC** añade una entropía explícita al objetivo (exploración por diseño) y
  entrena $Q$ con el target suavizado; suele ser el mejor para control continuo
  con datos eficientes, a costa de más hiperparámetros.

## Reward shaping: donde se gana o se pierde el problema

La recompensa es la especificación del problema. Los fallos clásicos:

- **Reward hacking**: el agente encuentra el atajo que maximiza la métrica sin
  resolver la tarea (se queda girando en círculo, repite la misma acción, se
  hace trampa a sí mismo). Si no lo has visto, no has entrenado suficiente.
- **Reward misspecification**: la métrica que puedes medir no es la que quieres
  (minimizar quejas ≠ maximizar satisfacción). El agente optimiza lo medible.
- **Sparse reward**: la recompensa solo llega al final (ganar/perder). El
  agente no tiene señal para explorar. Soluciones: reward shaping, curriculum,
  Hindsight Experience Replay (relabeling: tratar el estado alcanzado como
  objetivo), o imitación para arrancar.

Regla práctica: **la recompensa se diseña para que la política que quieres sea
también la más fácil de encontrar**, no para que sea la única óptima.

## Sample efficiency y datos

- **On-policy** (PPO/A2C) descarta los datos tras cada update: seguro pero
  hambriento de interacciones. **Off-policy** (DQN/SAC) reutiliza el buffer.
- **Offline RL**: aprender de un dataset fijo sin interacción (como RLHF usa
  datos de preferencias humanas). Es el puente entre RL y los datos que ya
  tienes. El reto: la política óptima puede no estar bien cubierta en el
  dataset, y extrapolar mal (overestimation del valor de acciones raras).
- En proyectos reales, la interacción es cara o peligrosa (un robot, un
  paciente): se entrena en **simulación** y se transfiere. El **sim-to-real gap**
  (la sim no es el mundo) se ataca con domain randomization: variar los
  parámetros de la sim para que la política aprenda a ser robusta al rango de
  variación del mundo real.

## Evaluación: el problema que todo el mundo resuelve mal

Un RL no se evalúa con una métrica de test estática:

- **No estacionariedad**: la distribución cambia porque el agente cambia. Un
  modelo que se evaluó contra un oponente fijo puede colapsar ante uno nuevo.
- **Varianza de semilla**: dos entrenamientos con semillas distintas dan
  políticas distintas. La evaluación honesta reporta media y dispersión sobre
  varias semillas, no un número.
- **Curvas de entrenamiento**, no puntos: importa cuántos datos/tiempo costó
  llegar ahí (sample efficiency). Un algoritmo que llega más lejos con 10× más
  datos no es mejor para tu presupuesto.
- **Off-policy evaluation**: evaluar una política con datos de otra es
  estadísticamente delicado (importance sampling con varianza explosiva); una
  estimación OPE sin intervalos de confianza es una afirmación sin evidencia.

## Hiperparámetros que importan (tabla de arranque)

| Algoritmo | Críticos | Frágil en |
|-----------|----------|-----------|
| DQN | learning rate, tamaño buffer, red target period | acciones continuas |
| PPO | clip $\epsilon$, learning rate, número de epochs | entropía colapsada, divergencia |
| SAC | alpha (entropía), learning rates, gradiente del crítico | hiperparámetros de entropía mal puestos |
| Todos | normalización de observaciones y de ventajas | escalas distintas entre estados |

Regla: el RL es mucho más frágil que el supervisado; un baseline bien tuneado
(DQN) supera a un PPO mal tuneado. Antes de culpar al algoritmo, revisa
normalización, seed y escala de recompensa.

## Cómo se rompe (checklist para el `lider`)

- Estado insuficiente (viola Markov) → el RL aprende a "recordar" o falla; hay
  que enriquecer el estado (frames apilados, memoria recurrente).
- Recompensa mal diseñada → reward hacking; audita qué está optimizando de
  verdad muestreando trayectorias.
- Inestabilidad del entrenamiento → bajar el learning rate, revisar el buffer,
  normalizar observaciones y ventajas.
- Hiperparámetros sensibles: el RL es mucho más frágil que el supervisado; un
  baseline (DQN con un buen tuning) supera a un PPO mal tuneado.
- Sobreentrenamiento del entorno: funciona en simulación y muere en el mundo
  real (sim-to-real gap). Domain randomization y evaluar en el objetivo real.
- Exploración que muere: entropía colapsada → la política se queda en un modo;
  vigilar la entropía media del actor.
- Evaluación contaminada por la no estacionariedad: comparar contra el agente
  de ayer, no contra un baseline fijo que ya cambió.

## Dónde encaja en dskit

El template no entrena RL por defecto; este fichero es para cuando el proyecto
sí lo necesita (control, optimización de políticas, recomendación secuencial).
El `lider` consulta aquí antes de aconsejar un algoritmo, y `rag refresh` lo
mantiene al día con los topics de `sources.json`. Un proyecto con `ml_type`
supervisado no necesita esto — ver `index.md`.
