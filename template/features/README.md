# features/ — contratos Gherkin del proyecto

Cada fichero `.feature` es el contrato de una feature del arnés, escrito por
`harness write_feature` y aprobado por un humano (`harness approve`) **antes**
de escribir código de producción. La ambigüedad se resuelve en el punto de
máximo apalancamiento: cuando aún no hay código que corregir.

Nadie edita estos ficheros a mano: su dueño es el agente `harness`. Si un
escenario no captura el comportamiento, se reescribe el contrato y se vuelve
a aprobar — es un fallo de la spec, no del código.

Flujo completo en `AGENTS.md` (sección *Spec-driven*) y en `skill mutation_agent`.
