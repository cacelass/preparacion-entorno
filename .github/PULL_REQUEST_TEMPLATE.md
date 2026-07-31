# Qué cambia

<!-- Y por qué. Si arregla un issue: «Closes #N». -->

## Verificación

<!-- Pega la salida real, no digas que pasa. Es lo mismo que el arnés le exige
     a los agentes: evidencia, no afirmaciones. -->

- [ ] `uv run python .github/scripts/validate_template.py` → 0 bugs
- [ ] `uv run python .github/scripts/check_copier.py`
- [ ] Toqué `template/` y he generado un proyecto real para probarlo
- [ ] `python .github/scripts/run_generated_ci.py <proyecto>` en verde

```
(salida aquí)
```

## Lista de comprobación

- [ ] Si añadí una opción a `copier.yml`: está en `_exclude` si condiciona
      ficheros, cubierta por la matriz smoke y documentada en el README
- [ ] Si toqué código de `template/`: no introduje `{{` ni `{%` accidentales
      (Jinja se aplica a **todos** los ficheros)
- [ ] Si un cambio rompe proyectos existentes: hay migración o está anotado
      en el CHANGELOG bajo un aviso claro
- [ ] CHANGELOG actualizado, explicando el porqué del cambio
