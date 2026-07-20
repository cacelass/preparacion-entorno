# Dev Workflow — Ciclo de desarrollo

## Pipeline
```
review → test → fix → commit → release
```

## Paso a paso

| Paso | Comando | Agente | Verificación |
|------|---------|--------|-------------|
| Code review | `run review review_package` | `review` | busca funciones largas, except desnudos, duplicación |
| Tests | `run test run_tests` | `test` | pytest pasa, cobertura ≥ umbral |
| Fix | `pipeline fix` | `refactor` + `test` | arregla y verifica |
| Commit | `run git commit_with_changelog` | `git` + `documentation` | Conventional Commit + CHANGELOG |
| Release | `run git tag_release --version X.Y.Z` | `git` | tag + changelog + bump |

## Pipelines GStack (multi-paso, auto-commit)

| Pipeline | Flujo | Cuándo usarlo |
|----------|-------|---------------|
| `pipeline develop` | review → test → commit | Desarrollo diario |
| `pipeline fix` | test → review → fix → test → commit | Tests fallando |
| `pipeline release --version X.Y.Z` | test → tag_release | Sacar versión |
| `pipeline cycle --phases N` | (review → test) × N → commit | Ciclo iterativo |
| `pipeline analyze` | env → git diff → review → lock | Diagnóstico sin modificar |

## Agentes clave

| Agente | Acción principal |
|--------|-----------------|
| `git` | `suggest_commit_message`, `commit_with_changelog`, `tag_release`, `analyze_diff` |
| `test` | `run_tests`, `coverage_summary`, `list_untested_modules` |
| `review` | `review_package` — funciones largas, except, duplicación |
| `refactor` | `fix_bare_excepts`, `add_type_hints`, `fix_mutable_defaults` |
| `documentation` | `update_changelog`, `bump_version`, `sync_readme` |

## Convenciones
- Commits: Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`)
- CHANGELOG: Keep a Changelog format
- Versionado: SemVer
