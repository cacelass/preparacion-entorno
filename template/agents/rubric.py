"""
agents.rubric — La rúbrica del arnés: qué significa que una feature está bien cerrada.

Sigue la filosofía del `lider` (`.opencode/agents/lider.md`) y de `AGENTS.md`:

- **Evidencia, no afirmaciones.** Cada criterio es binario (cumplido / no
  cumplido) y se cierra con la salida real de un comando, no con una frase.
- **La regla que no se salta.** Los criterios de la PUERTA son código en
  `harness.finish`: si fallan, el `done` no existe, por muy convincente que
  sea quien lo pide. No es una instrucción en un prompt.
- **Contexto mínimo.** El reviewer evalúa desde criterios + diff + evidencia
  reproducible, no desde la narrativa de quien hizo el trabajo — la
  justificación es el vehículo que transmite el punto ciego de quien la
  escribe.
- **El arnés se automejora (patrón ttsr).** Una regla solo entra aquí si nació
  de un fallo real y habría disparado en ese fallo. Un checklist que crece a
  capricho es ruido que se aprende a ignorar.

Por qué rúbricas binarias, criterio por criterio: la investigación sobre jueces
LLM (Panickssery et al., NeurIPS 2024; Wataoka et al.) muestra que un juez
puntúa más alto su propio texto por mecanismo de fluidez, y que una rúbrica de
checks binarios con criterios explícitos suprime ese sesgo que una puntuación
holística no. Y un estudio de gates de producción (2026) encontró que «una
rúbrica desconectada del gate es un sistema de alertas llamado gobernanza»:
por eso los criterios de la puerta se ENFORZAN en `harness.finish` y no se
dejan a la honestidad del reviewer.

Los umbrales son política fijada de antemano por una persona, no algo que el
sistema se autoconceda: si el sistema decidiera cuándo fiarse de sí mismo, el
problema solo subiría un nivel.
"""

from __future__ import annotations

#: Umbral de certeza (`μ.cert`) para cerrar una feature, fijado por el humano.
#: Un `done` con certeza baja es una ronda que iba a fallar — quien la cierra
#: debería saber por qué duda, no colarla por el hueco del `success`.
UMBRAL_CERTEZA = 0.6

#: Criterios de la PUERTA: los que `harness.finish` aplica en código. Si uno
#: falla, la feature NO se cierra. Cada entrada es (id, pregunta binaria);
#: el id es lo que queda en la traza de auditoría del cierre.
CRITERIOS_PUERTA: tuple[tuple[str, str], ...] = (
    ("GATE-1", "¿init.sh pasa en verde? (la puerta del proyecto)"),
    ("GATE-2", "¿Hay evidencia real de verificación, no una afirmación?"),
    ("GATE-3", "¿El reviewer no ha rechazado la feature?"),
    ("GATE-4", "¿La certeza (μ.cert) es suficiente? (≥ umbral)"),
)

#: Criterios de REVISIÓN: checklist binaria que el reviewer evalúa uno a uno,
#: cada uno con su evidencia. No son automatizables del todo (necesitan juicio
#: sobre arquitectura y alcance), así que son del reviewer, no de la puerta.
#: Crecen por ttsr: solo entran reglas nacidas de un fallo real.
CRITERIOS_REVISION: tuple[tuple[str, str], ...] = (
    ("R-1", "¿Se cumple cada criterio de aceptación? (evidencia por criterio)"),
    ("R-2", "¿Hay un test que falla si se revierte el cambio?"),
    ("R-3", "¿Respeta la arquitectura? (un dueño por recurso, el código donde le toca)"),
    ("R-4", "¿El diff no toca nada fuera del alcance de la feature?"),
    ("R-5", "¿No hay abstracción anticipada? (una sola implementación real no es interfaz)"),
    ("R-6", "¿No hay secretos ni rutas absolutas? (agents run secrets scan)"),
)

#: Categorías de decisión de criterio. No son irreversibles — configuran el
#: proyecto (qué librería, cómo se diseña, qué enfoque) —, así que la regla no
#: es bloquear sino REGISTRAR: al cerrar, `harness finish --decisions` las
#: declara y quedan en `harness/progress/history.md` para que un humano las
#: audite a posteriori sin depender del recuerdo de quien las tomó.
CATEGORIAS_DECISION: tuple[str, ...] = (
    "librería o dependencia nueva",
    "arquitectura o estructura del código",
    "enfoque o algoritmo elegido",
    "cambio de alcance respecto al plan",
)
