# Gestión de riesgo en proyectos de ML

## Qué es gestionar el riesgo

Gestionar el riesgo no es eliminarlo — es imposible — sino hacerlo explícito,
medible y decidible: cada riesgo se declara, se cuantifica y se decide cuánto
se acepta, cuánto se mitiga y quién responde si se materializa. Un proyecto
de ML sin gestión de riesgo no es un proyecto sin riesgos: es un proyecto
cuya exposición se decide por accidente, y esa es la peor forma de decidir.

**Riesgo = probabilidad × impacto.** Las dos variables se estiman con la
información disponible; cuando no hay base estadística, la estimación se
declara como supuesto y se revisa. Un riesgo no cuantificado no es "bajo" por
defecto: es **no medido**, una categoría distinta y peor documentada.

La gestión de riesgo no es un artefacto de documentación: es la disciplina
que une al resto del corpus. Los riesgos de datos, modelo, seguridad,
operación y cumplimiento no se tratan aislados; se inventarían, se priorizan
y se mitigan como un solo sistema. Este fichero es el paraguas de
`ml/deuda-tecnica.md`, `ml/ciclo-vida-mlops.md`, `ml/fairness-y-seguridad.md`
y `ml/privacidad-y-fuga-datos.md`: cada uno detalla un tipo de riesgo, este
los ordena.

## El ciclo de gestión de riesgo

El ciclo es el de ISO 31000 adaptado: identificar → evaluar → mitigar →
monitorizar → revisar. Cada fase responde una pregunta y produce una salida
verificable; si la fase no tiene salida, no se ejecutó.

| Fase | Pregunta que responde | Salida |
|------|-----------------------|--------|
| Identificar | ¿qué puede salir mal, y dónde? | lista de riesgos con causa y efecto |
| Evaluar | ¿cuán probable y cuán grave? | score P×I y prioridad |
| Mitigar | ¿qué capas reducen la exposición? | mitigación con dueño y fecha |
| Monitorizar | ¿están cambiando los riesgos? | señales, umbrales, alertas |
| Revisar | ¿siguen siendo los riesgos correctos? | register actualizado y ADR |

### Matriz 5×5: probabilidad × impacto

| Nivel | Probabilidad | Impacto |
|-------|--------------|---------|
| 1 | Rara: solo en condiciones excepcionales | Insignificante: sin efecto en el negocio |
| 2 | Improbable: no se espera, pero es posible | Menor: efecto pequeño y reversible |
| 3 | Posible: puede ocurrir en algún momento | Moderado: efecto sensible, recuperable |
| 4 | Probable: se espera en la mayoría de los casos | Mayor: efecto grave, difícil de revertir |
| 5 | Casi seguro: ocurrirá | Catastrófico: crítico o irreversible |

Score = P × I (1–25); la matriz clasifica:

```
          Impacto
           1   2   3   4   5
P=1        1   2   3   4   5
P=2        2   4   6   8  10
P=3        3   6   9  12  15
P=4        4   8  12  16  20
P=5        5  10  15  20  25
```

| Banda | Score | Trato |
|-------|-------|-------|
| Bajo | 1–4 | aceptar y registrar |
| Moderado | 5–9 | mitigar en plazo normal |
| Alto | 10–16 | mitigar, escalar y vigilar |
| Crítico | 17–25 | no aceptar: parar y rediseñar |

La matriz clasifica, no decide: un riesgo de impacto 5 se escala **aunque la
probabilidad sea 1**, porque una sola materialización es ya catastrófica (ver
escalado más abajo). El riesgo **inherente** es el que existe sin
mitigaciones; el **residual**, el que queda tras aplicarlas. Se gestiona el
residual, pero se registra el inherente para que el efecto de las capas sea
auditable.

## Riesgos propios de un proyecto de ML

| Categoría | Qué puede salir mal |
|-----------|---------------------|
| Datos | calidad, deriva, contrato que se rompe en el tiempo |
| Modelo | bias, sobreajuste, colapso, fuga en la validación |
| Seguridad y privacidad | adversariales, inversión, fuga de datos de personas |
| Operacionales | drift, serving, dependencias que se rompen |
| Cumplimiento | AI Act / GDPR mal aplicados, decisiones no auditables |
| Proyecto | no responde la pregunta, los datos no sirven, el coste se dispara |

Cada categoría tiene su tratamiento profundo en el corpus:
`data/calidad-datos.md` para datos; `ml/validacion.md` y
`ml/metricas-y-evaluacion.md` para modelo; `ml/fairness-y-seguridad.md` y
`ml/privacidad-y-fuga-datos.md` para seguridad, privacidad y cumplimiento;
`ml/ciclo-vida-mlops.md` y `ml/deuda-tecnica.md` para lo operacional.

El último renglón es el más ignorado y el que más proyectos mata. El riesgo
de **datos que no sirven** se detecta antes de modelar (EDA-001); el de
**modelo que no responde la pregunta**, contra el criterio de `SCOPE-001`; el
de **coste que se dispara**, comparando el valor de una decisión correcta con
el coste del ciclo completo (ver `ml/ciclo-vida-mlops.md`, "cuándo el ciclo
no compensa"). Un modelo técnicamente perfecto que responde a la pregunta
equivocada es el fracaso más caro y el que menos se reporta como riesgo.

## Mitigación por capas: defensa en profundidad

Ninguna medida única controla un riesgo: cada capa cubre los fallos de la
anterior, y es la suma de capas independientes lo que reduce la exposición.
Una mitigación de una sola línea que "resuelve" un riesgo es, en la práctica,
una confianza mal ubicada.

| Capa | Qué fallo cubre | Dónde está en este proyecto |
|------|-----------------|-----------------------------|
| Puerta del pipeline | trabajar sobre un entorno o backlog roto | `./init.sh` |
| Tests | que un cambio rompa un contrato sin avisar | suite pytest + tests de datos en CI |
| Monitoreo | que el drift o la degradación pasen en silencio | umbrales con dueño y runbook |
| Fallback / rollback | que una versión nueva empeore en producción | tripleta (sha, dataset, modelo) |
| Supervisión humana | que el sistema ejecute lo irreversible solo | puerta de permisos y aprobación |

La lectura de abajo a arriba: si la puerta no está o se salta, lo cubre la
suite; si la suite no muerde, lo cubre el monitor; si el monitor no se mira,
lo cubre el rollback; si nada responde, lo cubre la supervisión humana — la
única capa que no se automatiza y la última que se debe quitar.

{% if use_monitoring %}
La capa de monitoreo está implementada en `monitoring/monitor.py`: drift
KS/chi² entre la referencia (`X_train`) y los datos actuales, y degradación
de métricas frente al baseline, vía `make monitor` (detalle en
`ml/ciclo-vida-mlops.md`). El monitor no decide nada: produce la señal que
dispara el playbook de riesgos.
{% endif %}

## El arnés de este proyecto como sistema de riesgo

El arnés no es solo un gestor de tareas: es el sistema de gestión de riesgo
del proyecto, con cada pieza cubriendo un modo de fallo específico.

- **`./init.sh`, la puerta**: nada entra si el entorno, la estructura, el
  backlog y los tests no están en verde. El riesgo de trabajar sobre un
  proyecto roto se paga en la puerta, no en producción.
- **`SCOPE-001`, el criterio de parada**: el riesgo de proyecto — hacer algo
  que a nadie importa — se controla **antes** de modelar, con la pregunta, la
  métrica de éxito con umbral numérico y el criterio de parada en
  `references/00-objetivo.md`. Sin umbral no hay riesgo medible: "funciona"
  no es un criterio.
- **Evidencia obligatoria**: cada criterio se cierra con la salida real del
  comando que lo prueba. El riesgo de cerrar features por declaración queda
  fuera por construcción: "los tests pasan" sin la salida de `pytest` es
  rechazo.
- **Registro de decisiones (ADR)**: cada decisión relevante — elección de
  modelo, umbral, mitigación aceptada — deja rastro con su porqué en
  `harness/progress/` y `history.md` (append-only). El riesgo de que una
  decisión se reinvente o se contradiga se mitiga documentando, no
  recordando.
{% if use_sdd %}
- **Mutación y CRAP**: el riesgo de que los tests no muerdan — código que los
  tests no protegen aunque la cobertura diga lo contrario — se mide con
  `run_mutation_testing` y `crap_report` (umbral 30) antes de cerrar la
  feature. Un `survived` es una capa de mitigación que no cubre la suya.

```bash
uv run python -m agents --json run mutation run_mutation_testing --target {{ project_slug }}/utils.py
uv run python -m agents --json run mutation crap_report --target {{ project_slug }}/utils.py
```
{% endif %}

{% if use_rag %}
Las decisiones y el histórico (`harness/progress/`) y este corpus entran en
el índice RAG: `rag search --query "riesgos abiertos de cumplimiento"`
devuelve los ADR y las reglas de este fichero sin releerlos. `rag status`
avisa si el índice está desfasado — buscar sobre uno viejo devuelve la
respuesta de ayer sin ningún error.
{% endif %}

## Umbrales, apetito de riesgo y el risk register

Aceptar un riesgo no es ignorarlo: es **aceptar el riesgo residual de forma
explícita**. El apetito de riesgo es el máximo residual que el proyecto
tolera por categoría (ej.: "no se acepta riesgo de cumplimiento; se tolera
un score alto en datos si tiene mitigación"). Sin apetito declarado, cada
decisión se re-negocia por defecto, que es la forma cara de decidir.

Reglas de escalado:

- Impacto **≥ 4 → escalar siempre**, aunque la probabilidad sea baja: una
  sola materialización ya es grave.
- Score **≥ 17 (crítico) → bloquea el cierre de la feature**: no hay
  evidencia que lo compense sin rediseño.
- Un riesgo que **cambia de banda** entre revisiones se re-evalúa con su
  dueño y el cambio se registra.

El **risk register** es el documento vivo del proyecto, con su ubicación,
dueño y cadencia fijados, no implícitos:

| Pregunta | Respuesta en este proyecto |
|----------|----------------------------|
| Dónde vive | `references/riesgos.md`, versionado como el código |
| Quién lo actualiza | el dueño de cada riesgo; el `lider` al cerrar cada feature |
| Cuándo se revisa | al cerrar cada feature, tras cada incidente y en cada revisión de retrain |
| Qué se registra | causa, probabilidad, impacto, mitigación, dueño, estado |

## El riesgo de no desplegar

La gestión de riesgo suele mirar solo el despliegue y olvida el riesgo
opuesto: **nunca desplegar**. Un modelo que responde a una pregunta con valor
medible y se queda en experimento es un riesgo de coste de oportunidad — el
valor de una decisión correcta que no se toma. No gestionarlo es tan caro
como desplegar a ciegas; la diferencia es que nadie lo reporta como
incidente.

La decisión de desplegar (o de no hacerlo) es **explícita y documentada**:

1. Comparar el valor esperado de una decisión correcta contra el coste del
   ciclo completo (datos + validación + monitoreo + riesgo de rollback).
2. Fijar el riesgo residual aceptado y el plan de reversión antes del rollout.
3. Registrar la decisión como ADR, con el análisis que la sostiene.

Desplegar a ciegas y no desplegar nunca son los dos extremos de la misma
cobardía; la decisión con análisis, registro y capas de reversión es el punto
que se busca.

## Práctica

### Plantilla de risk register

| ID | Categoría | Prob. | Impacto | Score | Mitigación | Dueño | Estado |
|----|-----------|-------|---------|-------|------------|-------|--------|
| R-01 | proyecto | 3 | 5 | 15 | criterio de `SCOPE-001` verificado en EDA-001 | DS lead | abierto |
| R-02 | datos | 2 | 4 | 8 | tests de datos en CI + checksum del dataset | ing. datos | mitigado |
| R-03 | cumplimiento | 2 | 5 | 10 | tier del AI Act + supervisión humana | PM | abierto |

Estados: `abierto` / `mitigado` / `aceptado` / `materializado` (con el
incidente asociado). Un riesgo que se cierra sin mitigación registrada es un
riesgo aceptado — y se deja como tal, no como "resuelto".

### Playbook de riesgos

Cuando un riesgo se materializa o una señal lo sugiere, cinco pasos en orden
— el mismo patrón que el playbook de incidente de `ml/ciclo-vida-mlops.md`,
generalizado a cualquier fase del ciclo de vida:

1. **Detectar**: la señal con umbral calibrado dispara el runbook. Registrar
   hora, señal y evidencia; no empezar a tocar código.
2. **Congelar**: parar el sangrado. Si hay una tripleta conocida buena,
   rollback a ella; si no, dejar de cambiar cosas y acotar el alcance.
3. **Diagnosticar**: dato, modelo, mundo o proceso — ¿cuál cambió? Con
   evidencia antes de cada conclusión.
4. **Decidir**: mitigar, aceptar, revertir o rediseñar, con el dueño y el
   escalado si el impacto lo exige.
5. **Revisar**: actualizar el risk register y el ADR. El incidente cierra con
   el registro, no con la alerta apagada.

### Los 5 riesgos que un data scientist subestima

| # | Riesgo | Por qué se subestima | Antídoto |
|---|--------|----------------------|----------|
| 1 | La pregunta no importa | la técnica enmascara el propósito | `SCOPE-001` antes de modelar |
| 2 | Los datos no responden la pregunta | se asume que los datos valen | EDA-001 con respuesta explícita |
| 3 | El coste supera el valor | el ROI nunca se cuantifica | coste del ciclo vs. valor de la decisión |
| 4 | Los tests no muerden | la cobertura da falsa confianza | mutación y CRAP antes del cierre |
| 5 | El drift/skew degrada en silencio | el modelo "funciona" en offline | monitoreo con umbral y dueño |

## Fuentes

- **ISO 31000:2018 — Gestión del riesgo**. Principios y directrices; el
  estándar base del ciclo identificar-evaluar-mitigar-monitorizar-revisar.
  https://www.iso.org/standard/65694.html
- **NIST AI Risk Management Framework (AI RMF 1.0)** — National Institute of
  Standards and Technology (2023). Estructura las categorías govern, map,
  measure y manage. https://www.nist.gov/itl/ai-risk-management-framework
- **NIST Risk Management Framework (SP 800-37r2)** — integra la gestión de
  riesgo en el ciclo de vida de los sistemas de información.
  https://csrc.nist.gov/projects/risk-management/about-rmf
- **OWASP Machine Learning Security Top 10** — catálogo de los riesgos de
  seguridad de los sistemas de ML y su mitigación.
  https://owasp.org/www-project-machine-learning-security-top-10/
