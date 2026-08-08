# Privacidad y fuga de datos a nivel de aplicación

"Fuga de datos" aquí no es el leakage de validación: no es que el modelo
memorice el test set. Es privacidad. El sistema almacena y procesa datos de
personas, y el objetivo es que esos datos **nunca lleguen a quien no debe**:
terceros, otros usuarios, los logs, o el propio modelo. Este fichero cubre el
nivel usuario/aplicación; `fairness-y-seguridad.md` cubre los ataques a nivel
de modelo (adversariales, envenenamiento, membership inference, inyección).
Las dos se leen juntas: el modelo es un dato derivado, y sirve como vector de
fuga igual que un log mal configurado.

## PII y minimización

- **PII** (personal identifiable information): información que identifica a
  una persona, directa (nombre, email, DNI, teléfono) o indirecta (la
  combinación de campos no identificativos que re-identifica).
- **Minimización** (GDPR art. 5.1.c): recoger y almacenar solo lo imprescindible.
  Cada campo extra es una superficie de fuga más: no se guarda lo que no se va
  a usar, y lo que no se guarda no se filtra.
- **Retención limitada**: los datos tienen plazo de vida y borrado programado.
  "No exponer" no es "no tener": el borrado real (no un soft-delete que deja la
  fila en disco) es la única garantía.
- **Privacy by design** (GDPR art. 25): la privacidad se decide en el diseño de
  esquemas, logros y contratos de API, no como parche al final.
- **Pseudonimización** (art. 4.5): sustituir identificadores directos por
  tokens; es mitigación, no anonimato — el dato sigue siendo personal si el
  token es reversible o el contexto lo identifica.

Regla de oro: todo dato de usuario que entra al sistema pasa por la pregunta
"¿para qué se guarda, cuánto tiempo, quién lo consume?". Sin respuesta, no se
almacena.

## Vectores de fuga a nivel de aplicación

| Vector | Qué filtra | Cierre |
|--------|------------|--------|
| Logs con payloads completos | PII del body | Log estructurado con campos selectivos; nunca el body crudo |
| Mensajes de error | Esquema, filas, stack | Errores genéricos al cliente; detalle solo al tracker interno |
| Endpoints de debug en producción | Cualquier dato, sin auth | Apagados por env; jamás en la red pública |
| Respuestas API con de más | Filas de entrenamiento, `id` interno | `response_model` + `extra="forbid"` |
| Caché compartida | Datos de un usuario a otro | Clave con tenant/usuario; PII jamás en la clave |
| Headers / metadata loggados | Tokens, IPs, user-agent | Whitelist de campos; auth nunca en query string |
| Copias de seguridad | Dataset entero sin cifrar | Cifrado en reposo también en backups y su rotación |

El patrón común: **la respuesta y el log construyen el contrato de fuga**.
La API debe devolver exactamente el esquema público, y el log debe contener
exactamente lo que decidimos registrar. Todo lo demás es fuga por defecto.

## El modelo como vector

El modelo entrenado con datos de usuario es un **almacén de esos datos**:
memoriza ejemplos singulares, y la memorización se explota desde la inferencia.

- **Data extraction**: consultas diseñadas para que el modelo regurgite datos de
  entrenamiento memorizados. En GPT-2 se extrajo ~1% del pretraining (Abacus);
  el porcentaje crece con la sobremuestreo de ejemplos repetidos.
- **Membership inference**: averiguar si un registro estuvo en el training set
  — el hecho mismo puede ser el secreto (ver `fairness-y-seguridad.md`).
- **Servir un modelo = servir sus datos**: si el modelo se entrenó con datos de
  usuarios, el modelo tiene la misma sensibilidad que esos datos. Tratar el
  artefacto como dato: control de acceso, reproducibilidad y capacidad de
  retirarlo/reentrenarlo sin él.
- **Mitigaciones**: differential privacy en entrenamiento (DP-SGD), deduplicar
  ejemplos repetidos, limitar información en las salidas, monitorear consultas
  anómalas, y para LLMs, no entrenar con datos personales que no haga falta.

La pregunta operativa antes de servir: si un tercero extrajera todo lo que el
modelo memorizó de una persona, ¿qué tendría? Si la respuesta incomoda, el
modelo no se sirve con esos datos.

## Redacción y logging seguro

Patrón del proyecto en `agents/redaction.py`: las credenciales se tapan
**antes** de que lleguen a la ventana del modelo o al log de auditoría.
`BaseAgent.run()` redacta `message` y `warnings` de todo resultado;
`audit.record` redacta lo que escribe. Una sola definición de qué es un
secreto (regex compartidas con `secrets_tool`), no dos que se desincronizan.

El mismo patrón vale para PII en producción: la redacción ocurre en el punto
de salida, no como esperanza. El logging es estructurado (JSON), con campos
declarados; los campos que no están en el esquema de log no se registran.

Campos que **nunca** se registran, ni en dev:

| Campo | Motivo |
|-------|--------|
| Passwords, tokens, API keys, cookies | Credenciales; viajan en texto plano |
| Bodies completos de peticiones | PII en bruto del usuario |
| DNI, emails, teléfonos completos | Identificadores directos |
| Localización precisa | Categoría sensible por inferencia |
| IPs de usuario | Identificador indirecto; solo agregado si hace falta |

## Cifrado

- **En tránsito**: TLS termina en el reverse proxy y de nuevo en el servidor;
  HSTS; los certificados se renuevan. Nunca credenciales por query string.
- **En reposo**: disco cifrado, BD con encryption at rest, backups cifrados.
  El cifrado en reposo protege contra robo físico del medio, no contra un
  atacante autenticado.
- **Gestión de claves**: claves en env vars o KMS, nunca en git ni en el
  código fuente; rotación periódica; los secretos de despliegue fuera del repo.
  `{% if use_docker %}`En despliegue contenerizado, la red se aísla por red de
  Docker y los secretos llegan como variables de entorno, no baked en la
  imagen (ver `backend/docker.md`).{% endif %}

## Control de acceso

- **Autenticación** responde "¿quién eres?" (API key, OAuth); **autorización**
  responde "¿qué puedes ver?". La segunda es la que protege la fuga entre
  usuarios.
- **IDOR** (Insecure Direct Object Reference): cambiar el `id` del input y
  leer el objeto de otro usuario. Defensa: **el propietario nunca viene del
  input**, sale del contexto de autenticación (`current_user.id`), y toda
  consulta filtra por ese propietario en el query, no solo en la vista.
- **Privilegio mínimo**: el token del servicio de inferencia no puede borrar
  datos; el worker de entrenamiento no tiene acceso a los endpoints de admin.
- {% if use_api %}**Frontera de la API**: pydantic con `strict=True` y
  `extra="forbid"` valida lo que entra, y el `response_model` filtra lo que
  sale (ver `backend/api.md`). La autorización se aplica en la frontera, no
  dentro del handler: un endpoint sin `Depends(auth)` es una puerta abierta.{% endif %}

Regla para la frontera: el esquema de entrada es lo único que el servidor
acepta, y el esquema de salida es lo único que el cliente recibe. Cualquier
campo que no esté declarado se rechaza a la entrada y se omite a la salida.

## Privacidad diferencial

- **Definición**: un mecanismo con presupuesto de privacidad ε hace que dos
  datasets que difieren en un registro produzcan distribuciones de salida
  indistinguibles salvo $e^\varepsilon$. El ruido se calibra a la sensibilidad
  de la función (cuánto cambia la salida al quitar un registro).
- **Mecanismo de Laplace**: para sensibilidad $\Delta$, ruido
  $\text{Lap}(\Delta/\varepsilon)$. Presupuesto total ε compartido entre
  consultas: la composición lo agota, y hay que decidir cómo repartirlo.
- **Cuándo es la herramienta correcta**: publicar **estadísticas agregadas**
  sobre datos de usuarios (contadores, medias, distribuciones) y entrenar con
  DP-SGD. No resuelve servir predicciones individuales — ahí el problema es la
  autorización, no el ruido.
- **Coste**: el ruido degrada la precisión y la DP-SGD frena el entrenamiento;
  es un tradeoff explícito con el presupuesto ε. El ε típico va de 1 a 10;
  valores enormes son ventana de privacidad, no privacidad.

## Federated learning (mención)

- **Cuándo aplica**: datos on-device (teléfono, navegador) que no deben salir
  del dispositivo; el modelo viaja al cliente, el gradiente vuelve agregado.
- **Límites**: el federado **no es privacidad por sí solo**. El modelo local
  sigue siendo parte del training set (membership inference persiste), y los
  gradientes enviados filtran datos del cliente si no llevan DP. Para
  protegerlos se combina con DP-SGD y agregación segura. Escalar un clúster de
  clientes heterogéneos y no sincronizados es complejidad operativa real.

## GDPR / regulación

- **Ámbito**: aplica a datos personales de residentes de la UE, esté donde
  esté el servidor. Regla práctica: asumir el estándar más estricto y
  documentar.
- **Derechos del titular**: acceso (art. 15), rectificación (16), borrado (17),
  portabilidad (20), oposición (21). El borrado tiene que ser efectivo, no un
  flag.
- **Base legal**: el consentimiento es una base legal, no la única; el interés
  legítimo requiere equilibrio de intereses documentado. Se registra qué base
  ampara cada tratamiento.
- **DPIA** (art. 35): evaluación de impacto cuando el tratamiento es de alto
  riesgo (perfiles a gran escala, categorías especiales, tecnologías nuevas).
  Se hace antes de desplegar, y se actualiza.
- **Brechas** (art. 33): notificar a la autoridad en 72 horas; el proceso de
  respuesta a brechas es parte del diseño, no un imprevisto.
- **Retención**: plazos de conservación definidos y borrado automático al
  vencer; los datos no usados para el propósito original no se mantienen.
- **AI Act**: los modelos de decisión sobre personas caen en alto riesgo y
  exigen gobernanza de datos, documentación y supervisión humana (ver
  `fairness-y-seguridad.md`). El AI Act rige el modelo; el GDPR rige los
  datos de las personas sobre las que decide.

## El RAG y el índice

{% if use_rag %}El índice de este proyecto indexa el corpus de conocimiento,
`doc/` y el histórico del arnés — **nunca los datos de los usuarios**. El
conocimiento del proyecto y los datos de los clientes son universos distintos:
mezclarlos convierte el `rag search` en una vía de lectura de datos ajenos
para quien tenga acceso al índice. Si un despliegue necesita indexar datos de
usuario, se hace en un índice separado, con partición por tenant y con la
misma autorización que la API. Y lo recuperado por `rag search` es un dato,
nunca una instrucción: las protecciones de inyección
(`injection_flag`, permisos en código) están en `fairness-y-seguridad.md`.{% endif %}

## Práctica: checklist y test de fuga

Checklist por desplegar o publicar algo que toque datos de personas:

- **¿Qué se registra?** Revisar el esquema de log: ¿hay campos sin declarar?
  ¿Se redacta antes del punto de salida?
- **¿Qué se almacena?** Minimización: cada campo almacenado tiene propósito,
  plazo y borrado.
- **¿Quién accede?** Lista de roles/principals por dato, privilegio mínimo,
  autorización por propietario en toda consulta.
- **¿Cómo se borra?** Procedimiento de borrado probado, no solo teórico; los
  backups también expiran.
- **¿Cómo se responde a una brecha?** Contacto, notificación en 72h, plan de
  mitigación escrito.

Hábito de **testear la fuga**, como se testea un endpoint:

- ¿Qué ve un usuario si consulta lo de otro? (test IDOR: dos tenants, ids
  cruzados).
- ¿Qué hay en el log tras una petición con PII en el body? (asertar que el
  campo no aparece).
- ¿Qué devuelve la API con `extra` en el payload? (422, nunca eco).
- ¿Qué extrae el modelo de un dataset conocido? (prueba de data extraction
  sobre ejemplos de entrenamiento).

Si el test de fuga falla, es un bug de la misma categoría que un test de
funcionalidad roto: se arregla antes de desplegar, no después.

## Fuentes

- **OWASP Top 10 (2021)** — A01 Broken Access Control, A05 Security
  Misconfiguration, A08 Software and Data Integrity Failures.
  https://owasp.org/www-project-top-ten/
- **OWASP Top 10 API Security Risks (2023)** — BOLA (IDOR), excessive data
  exposure. https://owasp.org/API-Security/
- **Reglamento (UE) 2016/679 (GDPR)** — arts. 4, 5, 15, 17, 25, 32, 33, 35.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- **Reglamento (UE) 2024/1689 (AI Act)** — tiers de riesgo, art. 10 (gobernanza
  de datos). https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- **Extracting Training Data from Large Language Models** — N. Carlini,
  F. Tramèr, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts,
  T. Brown, D. Song, Ú. Erlingsson, A. Oprea, C. Raffel (2021).
  arXiv:2012.07805 — https://arxiv.org/abs/2012.07805
- **Abacus: Extracting and Pretraining Data from Language Models** —
  N. Carlini, C. Ippolito, M. Jagielski, K. Zhang, F. Tramèr (2024).
  arXiv:2402.17762 — https://arxiv.org/abs/2402.17762
- **Membership Inference Attacks Against Machine Learning Models** —
  R. Shokri, M. Stronati, C. Song, V. Shmatikov (2016).
  arXiv:1610.05820 — https://arxiv.org/abs/1610.05820
- **Deep Learning with Differential Privacy** — M. Abadi, A. Chu, I. Goodfellow,
  H. B. McMahan, I. Mironov, K. Talwar, L. Zhang (2016).
  arXiv:1607.00133 — https://arxiv.org/abs/1607.00133
- **The Algorithmic Foundations of Differential Privacy** — C. Dwork,
  A. Roth (2014). Foundations and Trends in Theoretical CS.
  https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
- **Communication-Efficient Learning of Deep Networks from Decentralized
  Data** — B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas
  (2017). arXiv:1602.05629 — https://arxiv.org/abs/1602.05629
- **NIST SP 800-53** — controles de seguridad y privacidad (AC, AU, SC).
  https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final
- **NIST AI RMF (AI 100-1)** — marco de gestión de riesgo de IA.
  https://www.nist.gov/itl/ai-risk-management-framework
