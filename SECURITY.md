# Política de seguridad

## Versiones con soporte

Solo la última versión publicada. dskit es una plantilla: los proyectos ya
generados no reciben parches automáticos, hay que traerlos con
`copier update` (ver la sección «Actualizar un proyecto existente» del README).

| Versión | Soporte |
|---|---|
| 1.13.x | ✅ |
| < 1.13 | ❌ — actualiza con `copier update` |

## Reportar una vulnerabilidad

**No abras un issue público.** Usa
[Security Advisories](https://github.com/cacelass/dskit/security/advisories/new)
de GitHub, que es privado.

Incluye si puedes: la versión de dskit, las opciones de generación (tu
`.copier-answers.yml`), qué se puede conseguir explotándolo y cómo reproducirlo.

Respuesta en un plazo razonable; si es explotable, se publica el arreglo y un
advisory con crédito, salvo que prefieras el anonimato.

## Qué cuenta como vulnerabilidad aquí

Esta es una plantilla, así que el modelo de amenaza es algo distinto al de una
aplicación:

**Sí cuenta**
- Código generado que expone credenciales, o que las escribe en un fichero que
  el `.gitignore` no cubre.
- Inyección a través de las respuestas de copier: un `project_name` malicioso
  que acabe ejecutándose en `_tasks`, en el `Makefile` o en un workflow.
- Endpoints inseguros por defecto en el `api/` o el `chat/` generados.
- Un workflow de GitHub Actions generado con permisos excesivos o que ejecute
  entrada no confiable.
- Dependencias fijadas a versiones con CVE conocido.

**No cuenta**
- Avisos de `bandit` sobre código de ejemplo en `notebooks/`.
- Que el `chat/` generado no tenga autenticación: es una interfaz de desarrollo
  y está documentado como tal. Si crees que debería avisarlo más fuerte, abre un
  issue normal.
- Vulnerabilidades de dependencias de terceros sin relación con cómo dskit las
  usa: repórtalas aguas arriba (aunque avisar aquí también se agradece).

## Lo que ya hace el proyecto

- `bandit` y `pip-audit` sobre cada proyecto generado en CI.
- `bandit` como hook de `pre-commit` en el stage `push`.
- Dependabot semanal para pip y para GitHub Actions.
- Sin secretos en la plantilla: las claves se leen de `.env`, que está
  gitignorado en el proyecto generado.
