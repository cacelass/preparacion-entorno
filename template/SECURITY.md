# Security Policy

## Reporting a Vulnerability

Si encuentras una vulnerabilidad de seguridad en {{ project_name }}, por favor
notifícala a través del canal de seguridad del repositorio.

**No abras issues públicos para vulnerabilidades de seguridad.**

## Scope

Se consideran vulnerabilidades:
- Exposición de datos sensibles (API keys, tokens, credenciales)
- Ejecución remota de código no intencionada
- Inyección de comandos a través de la API o el pipeline
- Dependencias con CVEs conocidos de alta gravedad

## Uso del agente de secretos

Este proyecto incluye un agente `secrets` que escanea el código en busca de
credenciales hardcodeadas. Ejecútalo regularmente:

```bash
uv run python -m agents run secrets scan
```
