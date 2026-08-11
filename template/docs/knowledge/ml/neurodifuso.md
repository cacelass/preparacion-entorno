# Neurodifuso (ANFIS): cuándo sí, y por qué casi siempre no

ANFIS (Adaptive Neuro-Fuzzy Inference System) combina lógica difusa con redes
neuronales: un sistema de reglas difusas (Takagi-Sugeno) cuyas funciones de
pertenencia y consecuentes se *aprenden* con retropropagación. Suena
inteligente y es la respuesta más frecuente cuando alguien pide "un modelo
diferente, algo más moderno". Este fichero existe para que el `lider` diga la
verdad con la matemática detrás, no para vender ANFIS.

La conclusión en una línea:

> **ANFIS es un candidato de nicho, no un default.** Si no tienes una razón
> concreta (interpretabilidad con reglas de experto + pocas variables), un
> Gradient Boosting o una red bien calibrada le gana en casi todos los
> benchmarks — y el "modelo híbrido" que tanto ilusiona se rompe por la
> explosión combinatoria de reglas.

Complementa a `supervisado.md` (lineales, árboles, boosting) y a
`redes-neuronales.md` (backprop, arquitecturas). Aquí el foco es decidir:
¿cuándo ANFIS es la herramienta correcta, y cuándo es la elección de menos
valor disfrazada de sofisticación?

## Qué es ANFIS, en concreto

Un sistema de inferencia difusa Takagi-Sugeno de primer orden. Para `n`
entradas y `m` funciones de pertenencia por entrada, las reglas son de la forma:

```
SI x1 es A1_i Y x2 es A2_j ENTONCES y = p_ij·x1 + q_ij·x2 + r_ij
```

La salida se calcula por capas:

1. **Fuzzificación**: cada entrada `xk` se mapea a grados de pertenencia con
   funciones `µ(xk)` (campana, gaussiana, triangular) — *aprendibles*.
2. **Fuerza de regla**: producto (o mínimo) de las pertenencias de cada regla,
   `w_r = ∏ µ_ir(xi)`.
3. **Normalización**: `w̄_r = w_r / Σ_s w_s`.
4. **Consecuente**: cada regla aporta `w̄_r · f_r(x)`, con `f_r` lineal en las
   entradas.
5. **Suma**: la salida final es `Σ_r w̄_r · f_r(x)`.

Aprendizaje híbrido: los consecuentes (`p, q, r`) se ajustan por mínimos
cuadrados y las premisas (funciones de pertenencia) por retropropagación. Eso
es todo: un modelo no-lineal, diferenciable, con una salida interpretable
como "mezcla de reglas locales lineales".

## Dónde brilla de verdad (el nicho)

- **Pocas variables y reglas de experto expresables.** Control de procesos,
  diagnóstico con un especialista que ya sabe las reglas ("si la temperatura
  es alta y el flujo bajo, deratear"). ANFIS afina esas reglas, no las inventa.
- **Interpretabilidad por construcción.** La salida se explica como "regla 3
  pesa el 60% y su consecuente es...". Para dominios regulados o con auditoría
  humana, eso vale mucho.
- **Datos escasos** donde una red profunda sobreajusta y un árbol no tiene
  suficiente señal. Las reglas difusas imponen estructura previa.

Si tu problema es clasificación/regresión sobre tablas densas con miles de
filas y decenas de columnas — el caso típico de este proyecto — no estás en
el nicho.

## Cómo se rompe (lo que de verdad enseña)

**1. Explosión combinatoria de reglas.** Con `n` entradas y `m` funciones de
pertenencia por entrada, el número de reglas es `m^n`. Tres entradas con 3
funciones cada una → 27 reglas. Diez entradas con 3 funciones → **59.049**
reglas. Las premisas y los consecuentes que hay que ajustar crecen
exponencialmente, y con ellos el sobreajuste y el coste. Es la maldición de la
dimensionalidad (ver `matematicas/probabilidad.md`) aplicada a un sistema de
reglas: *cada variable que añades multiplica el modelo en vez de sumarlo.*

**2. Difícil de entrenar.** La topología (cuántas funciones por entrada, qué
tipo) se elige a mano o a fuerza de ensayo; el aprendizaje híbrido es sensible
a inicialización y puede converger a óptimos locales malos. No hay un ANFIS
"autoconfigurable" estándar que compita con Optuna sobre XGBoost.

**3. La literatura reciente es clara.** En benchmarks tabulares con tamaño de
datos decente, Gradient Boosting (XGBoost/CatBoost/LightGBM) y las redes
bien reguladas superan a ANFIS en casi todas las configuraciones. ANFIS ganó
en los 90-2000; los árboles con regularización le comieron el terreno por
menos esfuerzo.

**4. "Híbrido" no es una ventaja automática.** Un neurodifuso es un sistema de
reglas con parámetros aprendidos; un bosque aleatorio también es un sistema de
reglas con parámetros aprendidos. La diferencia está en la interpretabilidad y
en la poda — no en ser "más inteligente por mezclar dos familias".

## La decisión honesta

| Situación | Qué elegir |
|-----------|------------|
| Tablas densas, miles de filas, métrica de calidad | Boosting (XGBoost/LightGBM/CatBoost) o red calibrada — ver `supervisado.md`, `redes-neuronales.md` |
| Control de procesos, pocas variables, reglas de experto, auditoría humana | ANFIS puede ser la respuesta correcta — y hay que poder decir *por qué*, no "porque es moderno" |
| Interpretabilidad obligatoria sobre un modelo global | Antes que ANFIS: `interpretabilidad.md` — SHAP sobre un boosting explica sin el coste de las reglas difusas |
| "Quiero probar algo diferente" | No es un criterio. Es una feature para el backlog con criterios de aceptación, no una arquitectura |

Regla del `lider`: si el usuario pide neurodifuso, primero se consulta esta
ficha (y `supervisado.md`), se le pregunta *qué problema concreto resuelve que
un boosting no*, y si la respuesta es "nada / suena bien", se aconseja contra
él con los números. Si el problema es del nicho (control, interpretabilidad,
datos escasos), se estudia en serio — no se descarta por esnobismo.

## Referencias cruzadas

- `supervisado.md` — el default que casi siempre gana: lineales, árboles, boosting
- `redes-neuronales.md` — backprop, la mitad "neuro" de ANFIS
- `interpretabilidad.md` — interpretabilidad sin el coste de las reglas difusas
- `matematicas/probabilidad.md` — maldición de la dimensionalidad, priors
- `optimizacion-hiperparametros.md` — porque el problema de ANFIS no es el algoritmo, es configurarlo
- `metricas-y-evaluacion.md` — comparar honradamente antes de decidir
