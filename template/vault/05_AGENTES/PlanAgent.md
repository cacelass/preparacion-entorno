---
tags:
  - agente
  - coordinacion
---
# Plan Agent

> Jefe de proyecto: convierte un encargo humano en una orden de trabajo, pregunta lo que falte y delega.

## Contrato

Extraído de `agents/contracts.py`:

- **Rol:** Jefe de proyecto — descomposición y delegación
- **Capacidades:** descomponer encargos en pasos y asignarlos; detectar información faltante; ejecutar orden de trabajo via GStack
- **Límites:** no ejecuta acciones de dominio él mismo; no inventa argumentos; no ejecuta sin respuestas
- **Necesita:** el encargo en lenguaje natural; respuestas a preguntas que genere
- **Colabora con:** todos — es el punto de entrada

## Responsabilidades

1. Recibir el encargo del humano
2. Descomponerlo en pasos atómicos
3. Asignar cada paso al agente responsable
4. Generar preguntas si falta información
5. Ejecutar la orden de trabajo aprobada

## Dependencias

- Todos los agentes del proyecto (delega según el dominio)
