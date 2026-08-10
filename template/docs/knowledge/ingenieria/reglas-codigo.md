# Reglas de código: checklist de revisión

Hoja de reglas duras. Referencia para el lider y para el reviewer: cada regla
es comprobable (por máquina, por evidencia o por inspección) y bloquea la
aprobación si se incumple. No es un ensayo: es un checklist.

## Seguridad

| # | Regla | Cómo se comprueba |
|---|-------|-------------------|
| R1 | No secretos en código ni en logs | `grep -rE "api_key\|password\|token\|secret"`; revisar diffs |
| R2 | `.env` no entra en git | `.gitignore` tiene `.env`; `git ls-files \| grep .env` |
| R3 | Validar entradas en la frontera | Revisión: toda entrada externa se valida |
| R4 | Dependencias fijadas en `uv.lock`; `pip-audit` en verde | `pip-audit` en CI |

## Tipos y firmas

| # | Regla | Cómo se comprueba |
|---|-------|-------------------|
| R5 | Type hints en toda función pública | `mypy --strict` en verde |
| R6 | Sin kwargs sin anotar cuando se puede evitar | `mypy --disallow-untyped-defs`; revisión de firmas |
| R7 | `Any` solo en fronteras justificadas | Revisión de firmas y `# type: ignore` |

## Errores

| # | Regla | Cómo se comprueba |
|---|-------|-------------------|
| R8 | No `except:` a secas; capturar excepciones específicas | `ruff` (E722); inspección |
| R9 | Fail fast: validar arriba, no diferir el error | Revisión: errores aparecen en el punto de origen |
| R10 | No tragar excepciones: loguear o propagar | Inspección: todo `except` hace algo o relanza |

## Limpieza

| # | Regla | Cómo se comprueba |
|---|-------|-------------------|
| R11 | Sin código muerto, imports sin usar, sin comentado | `ruff` (F401, F841); diff limpio |
| R12 | Nombres por significado; una cosa por función | Revisión: el nombre no necesita comentario |
| R13 | No mutar argumentos como canal de salida | Revisión de firmas; tests de inmutabilidad donde aplique |
| R14 | Sin estado global mutable en código de librería | `grep` de `global` / módulo-mutables; revisión |

## Determinismo

| # | Regla | Cómo se comprueba |
|---|-------|-------------------|
| R15 | Sin aleatoriedad ni tiempo sin semilla fija | `random`/`np.random`/`time.now` usan seed registrada |
| R16 | Pipeline determinista: mismo input → mismo output | Re-ejecutar una etapa y diff de salida |

## Control de versiones

| # | Regla | Cómo se comprueba |
|---|-------|-------------------|
| R17 | No commitear artefactos generados ni binarios | `.gitignore`; `git ls-files` (modelos, `*.parquet`) |
| R18 | No `force-push` | Regla de equipo; verificar con `git push` normal |
| R19 | Commits convencionales (`feat:`, `fix:`, `chore:`, `docs:`) | `git log --oneline` |

{% if use_sdd %}
## Spec-driven

| # | Regla | Cómo se comprueba |
|---|-------|-------------------|
| R20 | Backlog y `features/` solo se editan vía `harness` | Ningún commit toca `harness/` ni `features/` |
| R21 | Cerrar feature exige escenarios aprobados y evidencia | `harness finish`: evidencia + gate verde |
{% endif %}

## Tests y evidencia

| # | Regla | Cómo se comprueba |
|---|-------|-------------------|
| R22 | Un bug report trae un test que falla | Revisión del PR: el fix trae su test rojo→verde |
| R23 | "Funciona" solo con salida de comando | El reviewer pega la salida de `pytest`/`make test` |
| R24 | Cobertura ≥ 80% por módulo; CI lo hace fallar si baja | `pytest --cov-fail-under=80` |

## Revisión

| # | Regla | Cómo se comprueba |
|---|-------|-------------------|
| R25 | Sin auto-aprobación: el autor no revisa su propio cambio | El PR lo cierra un revisor distinto |
| R26 | Evidencia sobre afirmaciones | Aprobar requiere salida de comando; "parece bien" no |
| R27 | Diff mínimo: no toca código ajeno | Revisión: solo líneas necesarias para la feature |

## Checklist del reviewer (orden de aplicación)

```text
1.  ¿El diff toca solo lo necesario?                     (R27)
2.  ¿Hay secretos o .env en el commit?                   (R1, R2)
3.  ¿Lint, formato y tipos pasan?                        (R5, R6, R11)
4.  ¿Los `except` capturan específico y no tragan?       (R8, R10)
5.  ¿El pipeline es determinista y sin estado global?    (R14, R15, R16)
6.  ¿La feature trae su test, y el test falla sin el fix? (R22)
7.  ¿La evidencia de CI (pytest, cov) está pegada?       (R23, R24)
8.  ¿Cumple las reglas de spec-driven del proyecto?      (R20, R21 si aplica)
9.  Decisión: APPROVE o REJECT con el nº de regla incumplida.
```

## Fuentes

- PEP 8, PEP 484, PEP 20 (zen de Python: "errors should never pass silently").
- Documentación de Ruff (E722), mypy (`--strict`), Black, isort.
- Robert C. Martin, "Clean Code" — nombres, funciones pequeñas, una
  responsabilidad (Prentice Hall, 2008).
- Documentación de Conventional Commits.
- Documentación de pytest y pytest-cov.
- Documentación de Bandit y pip-audit (higiene de seguridad).
