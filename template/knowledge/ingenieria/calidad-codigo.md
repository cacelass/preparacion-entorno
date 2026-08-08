# Calidad de código en proyectos de datos

Referencia para el agente lider y para el reviewer: qué es calidad, cómo
hacerla objetiva con herramientas, y qué falla en proyectos reales de DS.

## Qué es calidad (y cómo hacerla objetiva)

**Principio.** Calidad es legibilidad (lo entiende quien lo hereda),
corrección (hace lo que dice) y mantenibilidad (se puede cambiar sin romper).
No es gusto: es un conjunto de propiedades comprobables por máquina.

**Práctica.** Todo criterio de calidad que no se puede verificar con una
herramienta en CI es opinión. La lista mínima objetiva:

| Propiedad | Herramienta que la mide |
|-----------|-------------------------|
| Formato consistente | Black, isort |
| Lint (errores, smells) | Ruff (E/F/W/I), Pylint |
| Tipos | mypy `--strict` |
| Complejidad | radon (cc ≤ 10) |
| Cobertura | pytest-cov (≥ 80%) |
| Seguridad | pip-audit, Bandit |
| Código muerto | Ruff (F401, F841), vulture |

**Cómo falla.** Debates de "esto se lee mejor así" que bloquean la review y no
pueden ganarse; o un proyecto donde la calidad depende de quién revisa y en
qué humor está. Con herramientas en CI, la discusión pasa de gusto a
evidencia.

## Formato y lint

**Principio.** La consistencia vale más que la preferencia personal: un
formato automático elimina el diff de estilo y deja las reviews para la
lógica.

**Práctica.**

- **Black**: formato canónico sin opciones que discutir.
- **isort**: orden de imports (`stdlib` → terceros → locales).
- **Ruff**: linter por categorías — `E` (errores de estilo), `F`
  (bugs: imports sin usar, variables sin asignar), `W` (warnings de
  sintaxis), `I` (orden de imports). Muchas reglas E/F son auto-fixables.

```bash
ruff check . --fix      # autofix de lo seguro
black .                 # formatea
```

**Cómo falla.** Proyectos sin formateador donde cada commit trae reordenación
de imports que contamina el diff; o lint configurado "relajado" que permite
`import *` y `except Exception` silenciosos porque "a nadie le molesta".

## Tipos

**Principio.** Los type hints son documentación ejecutable que el compilador
verifica; convierten errores de llamada en fallos en CI en lugar de fallos en
runtime.

**Práctica.**

- Hints en toda la API pública (firmas de funciones y clases exportadas).
- `mypy --strict`: `disallow_untyped_defs`, `warn_return_any`,
  `disallow_any_generics`. Si `--strict` rompe un módulo heredado, márcalo
  con `# type: ignore` justificado, no con un perfile global.
- **Tipado gradual**: se puede migrar módulo a módulo, empezando por la capa
  de datos (donde los errores de tipo son caros).
- **`Any` aceptable** en fronteras con librerías sin tipos, en datos JSON
  arbitrarios y en reflection; nunca en la interfaz pública si se puede
  declarar un `Protocol`.
- **Genéricos**: `list[float]`, `dict[str, int]`, `Sequence[T]`, `TypeVar`
  para estructuras que conservan el tipo interno.

**Cómo falla.** Una firma `def fit(X, y)` sin tipos que cambia el orden o el
formato de `X` y nadie lo nota hasta producción; o `Any` en todos lados, que
es escribir tipos para apagar mypy y fingir que hay tipado.

## Testing

**Principio.** Los tests son el registro de comportamiento del proyecto: pasan
cuando el código es correcto hoy y detectan cuando deja de serlo.

**Práctica.**

- **Unitarios** aíslan una unidad (función, transform); **integración**
  combinan módulos; **E2E** corren el pipeline/CLI completo sobre un dataset
  controlado.
- **TDD como herramienta de pensamiento**: escribir el test antes fuerza a
  decidir el contrato (qué entra, qué sale, qué es éxito) antes de codear.
  El test fracasa primero por la razón correcta.
- **Cobertura es una métrica, no un objetivo**: ≥80% por módulo es un mínimo,
  pero no prueba que los tests "muerdan". La mutación (ver `use_sdd`) mide
  que un test detecte lógica rota, no solo líneas tocadas.
- **Property-based testing (Hypothesis)**: genera entradas automáticamente y
  busca invariantes. Útil para parsers, serialización, agregados, aritmética.
- **Fixtures fijas para pipelines de datos**: un mini-dataset versionado en
  `tests/data/` garantiza determinismo y velocidad, frente a leer datos
  reales que cambian.

**Anti-patrones de mocking.**

- Mockear todo: el test no prueba la integración real y acopla el test a la
  implementación.
- Mockear la librería que se está probando (p. ej. mockear `pd.merge`).
- Mocks que reproducen la firma "esperada" pero no el comportamiento: el test
  pasa mientras producción falla.

**Cómo falla.** Un pipeline de datos sin tests que "funciona en el notebook"
y revienta la semana siguiente al cambiar un schema; o 100% de cobertura con
`assert x is not None` y mocks que nunca ejecutan la lógica real.

## Docstrings y comentarios

**Principio.** El docstring documenta la API (qué hace, qué recibe, qué
devuelve, qué lanza); el comentario explica el *porqué*, nunca el *qué* (eso
lo dice el código).

**Práctica.** Estilo Google, tres secciones donde aplique:

```python
def scale(values: Sequence[float], lo: float, hi: float) -> list[float]:
    """Escala un rango [min, max] a [lo, hi].

    Args:
        values: Secuencia de valores numéricos.
        lo: Límite inferior del rango destino.
        hi: Límite superior del rango destino.

    Returns:
        Lista con los valores escalados.

    Raises:
        ZeroDivisionError: Si todos los valores son iguales.
    """
```

Comentarios legítimos: por qué se eligió este algoritmo, por qué este
número mágico, por qué el orden de las operaciones importa. Comentarios
basura: los que repiten el nombre de la variable o documentan lo obvio.

**Cómo falla.** Docstrings que dicen "hace lo que dice" sin Args/Returns, o
comentarios que explican *qué* hace el código (se pudren cuando el código
cambia) mientras que el *porqué* (el motivo real) desaparece con la persona
que lo escribió.

## Refactorización

**Principio.** Refactorizar es cambiar la estructura sin cambiar el
comportamiento; se hace cuando el costo de mantener supera al de cambiar.

**Práctica.**

1. **Tests primero**: red verde sobre el comportamiento actual.
2. Cambios pequeños y verificables, un movimiento a la vez (renombrar,
   extraer función, mover módulo).
3. **El nombre es la herramienta principal**: un buen nombre elimina la
   necesidad de comentarios. Si el nombre exige comentario, el nombre está
   mal.
4. **Funciones pequeñas, una responsabilidad**: fácil de testear, de
   reutilizar, de razonar.
5. **Eliminar código muerto**: funciones sin llamadores, parámetros sin uso,
   imports huérfanos. Cada línea muerta es ruido para el lector.

**Cómo falla.** Refactorizar sin tests (cambia comportamiento "un poquito" y
nadie lo detecta), o refactorizar código que no está roto para "mejorar el
estilo", quemando horas y rompiendo lo que funcionaba.

## Métricas de complejidad

**Principio.** La complejidad ciclomática mide cuántos caminos de decisión
tiene una función; por encima de un umbral, es más barato dividir que
entender.

**Práctica.**

- **radon**: `radon cc -a -s <módulo>` — umbral ~10 por función.
- **Carga cognitiva**: cuánto anidamiento y estado hay que retener; menos
  formal, se aproxima con funciones que hacen una sola cosa y sin anidar
  `if/for` más de dos niveles.
- Regla práctica: si una función necesita un test por cada rama para sentirse
  cubierta, está haciendo demasiado.

**Cómo falla.** Un `clean_text()` de 80 líneas con 14 `if` anidados y 3
`elif` por caso borde que nadie puede auditar; la métrica diría 20+ y el
fix correcto es dividir, no añadir otro parámetro booleano.

## Code review

**Principio.** El revisor verifica evidencia, no afirmaciones: lee el diff,
ejecuta los comandos que el autor afirma que pasan y valida contra los
criterios de aceptación.

**Práctica.** Qué comprueba un revisor:

- ¿El cambio hace lo que los criterios dicen?
- ¿Los tests existen, cubren el cambio y fallan si el cambio es incorrecto?
- ¿El diff es mínimo y legible? ¿Toca cosas fuera del alcance?
- ¿Los tipos, el lint y las métricas pasan en CI?
- ¿Hay secretos, credenciales o datos en el commit?
- ¿La lógica se entiende sin explicación verbal?

**Autoridad del revisor.** La autoridad no es jerárquica: es la del evidence
gate. Un revisor bloquea si la evidencia no existe (falta la salida del
comando, falta el test), no por preferencia estética.

**Cómo falla.** Reviews que aprueban "porque lo escribió un senior", o el
patrón inverso: aprobar la funcionalidad sin correr nada, leyendo solo el
diff de texto.

## Higiene de seguridad

**Principio.** Los secretos y la entrada no validada son el vector de fallo
más barato de explotar; la prevención es un checklist, no una actitud.

**Práctica.**

- **Nunca logues secretos**: tokens, claves, URLs con credenciales o dataset
  de passwords. Los logs son texto plano y viajan (Vault, CI, terceros).
- **Valida entradas** en la frontera: tipos, rangos, esquemas (pydantic),
  antes de que lleguen al cómputo.
- **Pin de dependencias**: `uv.lock` (o `requirements.lock`) para reproducir;
  rangos amplios solo en la spec de librería, no en la app.
- **`pip-audit`** en CI: falla si una dependencia del lock tiene CVE conocida.

**Cómo falla.** Un `print(df)` de diagnóstico que arrastra una columna de
API keys al log del pipeline; o un `pip install` flotante que en la próxima
release trae una versión vulnerable sin que nadie se entere.

## Checklist mínimo de calidad (machine-enforceable)

**Principio.** Todo lo de abajo lo corre la CI; si no está en CI, no es una
regla.

```text
[ ] black --check .
[ ] isort --check-only .
[ ] ruff check .          # E/F/W/I
[ ] mypy --strict {{ project_slug }}
[ ] radon cc -a -s -e tests/ | complejidad ≤ 10
[ ] pytest --cov={{ project_slug }} --cov-fail-under=80
[ ] pip-audit (deps del lock)
[ ] sin secretos en el diff (scan)
```

**Cómo falla.** El checklist "de memoria" que el lider o el reviewer aplican
a mano cuando se acuerdan: las reglas que no son código se violan siempre en
la semana 6.

## Fuentes

- PEP 8 — Style Guide for Python Code.
- PEP 484 — Type Hints; PEP 526 — Variable Annotations.
- Documentación de Black, isort y Ruff (reglas E/F/W/I).
- Documentación de mypy (`--strict` y banderas por archivo).
- Documentación de pytest, pytest-cov, Hypothesis.
- Radon (métrica ciclomática; umbral tradicional 10, McCabe).
- Robert C. Martin, "Clean Code: A Handbook of Agile Software Craftsmanship"
  (Prentice Hall, 2008).
- Steve McConnell, "Code Complete" (Microsoft Press, 2004).
- Documentación de Bandit y pip-audit.
