---
tags:
  - agente
  - entorno
---
# Dependency Agent

> Vigilante de dependencias: obsolescencia y vulnerabilidades contra PyPI/OSV. Solo lee.

## Contrato

- **Rol:** Dependency watchdog — vulnerabilidades
- **Capacidades:** detectar paquetes desactualizados y vulnerabilidades conocidas (necesita internet)
- **Límites:** no actualiza ni instala nada (→ env)
- **Colabora con:** env

## Responsabilidades

1. Escanear dependencias contra PyPI (versiones recientes) y OSV (vulnerabilidades)
2. Reportar paquetes desactualizados
3. Reportar vulnerabilidades conocidas
