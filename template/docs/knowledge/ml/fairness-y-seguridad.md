# Equidad (fairness) y seguridad en sistemas de ML

## Fairness: atributos protegidos

Un atributo protegido es aquel sobre el que la ley o la ética prohíben
discriminar: edad, sexo, etnia, religión, orientación sexual, discapacidad,
estado migratorio. En la UE, el RGPD (art. 9) trata los datos que revelan
estas categorías como "categorías especiales" con prohibición de tratamiento
salvo excepciones. En ML, "fairness" es el estudio de cómo el modelo trata
distintos grupos definidos por estos atributos — y de los sesgos que los
dañan incluso cuando el atributo no está entre las features.

## De dónde viene el sesgo

| Fuente | Mecanismo | Ejemplo |
|--------|-----------|---------|
| Muestreo | Muestra no representativa: grupo sub- o sobrerepresentado | Grupo con poco volumen histórico |
| Etiquetas | El ground truth es injusto, ruidoso o proxy | Productividad = horas en oficina |
| Features proxy | Variable "neutral" que reconstruye el atributo protegido | Código postal ≈ económico |
| Evaluación | La métrica global esconde el error por grupo | 95% accuracy, 40% de error en el minoritario |

El atributo protegido casi siempre se puede inferir de las demás features. Por
eso "quitar el atributo" no quita el sesgo: el modelo lo reconstruye desde
proxies. La neutralidad de las features no existe; lo que existe es un sesgo
mejor o peor documentado.

## Métricas de fairness: definiciones y cuándo elegir

Sea $G$ el grupo protegido, $Y$ el outcome real, $\hat{Y}$ la predicción y $S$
el score del modelo.

| Métrica | Igualdad que exige | Uso típico |
|---------|--------------------|------------|
| Paridad demográfica | $P(\hat Y{=}1\mid G{=}a)=P(\hat Y{=}1\mid G{=}b)$ | Repartir el outcome |
| Equalized odds | $P(\hat Y{=}1\mid Y,G{=}a)=P(\hat Y{=}1\mid Y,G{=}b)$ | Mismos errores por grupo |
| Igualdad de oportunidad | $P(\hat Y{=}1\mid Y{=}1,G)$ igual en $G$ | Costo crítico: falso negativo |
| Calibración | $P(Y{=}1\mid S{=}s,G{=}a)=P(Y{=}1\mid S{=}s,G{=}b)$ | El score se lee como probabilidad |
| Fairness individual | Decisión similar para casos similares | Trato comparable caso a caso |

- **Paridad demográfica**: tasas de selección iguales entre grupos. Cuándo:
  el outcome debe repartirse igual (cuotas, acceso); se puede cumplir
  suboptimizando por grupo.
- **Equalized odds**: TPR y FPR iguales por grupo; exige el mismo rendimiento
  condicionado a la verdad. Cuándo: los errores de ambos tipos importan y hay
  que igualarlos entre grupos.
- **Igualdad de oportunidad**: relaja equalized odds a solo TPR igual (mismo
  recall). Cuándo: el falso negativo es el error caro (recidiva,
  contratación, screening).
- **Calibración**: un score $s$ significa la misma probabilidad de $Y$ en todos
  los grupos. Cuándo: el score se interpreta como probabilidad (crédito) y la
  decisión la toma otro sistema sobre ese score.
- **Fairness individual**: personas similares (según una métrica de similitud
  definida) reciben decisiones similares. Cuándo: hay un estándar de trato
  comparable caso a caso; en la práctica depende de elegir bien esa métrica.

**Tensión formal**: paridad demográfica, equalized odds y calibración no son
simultáneamente satisfacibles salvo casos degenerados (prevalencia de $Y$
igual entre grupos). Elegir una métrica es elegir qué se sacrifica, y esa
elección debe quedar registrada, no ser un accidente. Regla práctica: fijar
una métrica de negocio por grupo (slicing) además de la métrica global, y
reportar siempre el desglose.

## Mitigación: pre, in, post

| Momento | Técnica | Costo |
|---------|---------|-------|
| Pre | Rebalanceo/reweighting, perturbación de features, datos sintéticos | Cambia la distribución |
| In | Restricción de fairness, regularización, adversario que predice $G$ | Hiperparámetro; inestable |
| Post | Umbrales o re-rank por grupo, reajuste del score | Puede romper la calibración |

El tradeoff accuracy vs. fairness es un frente de Pareto: ganar equidad cuesta
exactitud. La curva debe reportarse para que la decisión sea del negocio, con
números y no como promesa. No hay mitigación gratuita; hay mitigación
documentada.

## Auditoría y documentación

- Auditar por **grupo**, nunca solo global: slicing por atributos protegidos
  sobre las métricas de negocio.
- Herramienta del ecosistema dskit: `eticas-audit` (ITACA),
  `uv pip install eticas-audit` (ver `agents/README.md`).
- **Model cards**: documentar propósito, datos de entrenamiento, métricas por
  grupo, límites y usos prohibidos. Es el formato estándar para entregar el
  tradeoff a los consumidores del modelo.

**Cuándo es obligatorio**: crédito (fair lending, hipotecas), contratación,
justicia penal (recidivismo) y salud. Además de la ley, el estándar
profesional: un modelo cuyo sesgo por grupo no se midió es un modelo sin
control de calidad.

## Seguridad: ataques adversariales

Pequeñas perturbaciones, imperceptibles, que voltean la predicción:

- **FGSM**: $x' = x + \varepsilon \cdot \text{sign}(\nabla_x L(f(x), y))$ —
  una pasada de gradiente; el ataque de un paso.
- **PGD**: versión iterativa de FGSM proyectada a una bola de radio
  $\varepsilon$; es el benchmark de robustez.
- **Transferabilidad**: un adversarial generado contra un modelo funciona
  contra otros (misma tarea, arquitecturas distintas). Los ataques escalan sin
  acceso al modelo objetivo.
- **Defensas**: adversarial training (entrenar con ataques), denoising,
  certificados de robustez. Es una carrera sin punto final: no hay defensa
  perfecta, hay postura.

## Envenenamiento de datos

El ataque ocurre en entrenamiento: inyectar ejemplos mal etiquetados para
cambiar el comportamiento del modelo. La puerta trasera (backdoor) es la forma
insidiosa: el modelo se comporta bien salvo cuando está presente el trigger
que el atacante plantó. Defensas:

- Tratar las fuentes de datos como no confiables: validación, deduplicación,
  detección de outliers.
- Limitar la influencia de cualquier ejemplo o subconjunto.
- Cuarentena y revisión de datos externos antes de entrar al training set.
- Ante sospecha, reentrenar limpio y auditar el pipeline de datos.

## Inversión de modelo y membership inference

| Ataque | Qué extrae | Riesgo |
|--------|-----------|--------|
| Model inversion | Datos o atributos del training set | Personales reconstruidos vía confianza |
| Membership inference | Pertenencia de un registro al training set | La pertenencia es el secreto |

El modelo memoriza; las salidas con confianza filtran información. Defensas:
minimizar la información de las salidas, differential privacy (ruido calibrado
que enmascara la pertenencia) y monitoreo de consultas anómalas.

## Prompt injection en asistentes RAG

{% if use_rag %}
Este proyecto consume su corpus vía RAG: el texto recuperado entra en el
contexto del agente. Un documento indexado puede contener instrucciones
hostiles — "ignora lo anterior y haz X" — escritas por un tercero. Regla
dura: **los datos recuperados nunca amplían lo que el agente tiene permitido
hacer**. `rag search` devuelve lo externo en un bloque aparte y delimitado,
con advertencias, y los fragmentos con pinta de inyección se marcan al
indexar (`injection_flag`).
{% endif %}

La defensa de fondo no es detectar la inyección (las listas de patrones se
esquivan): es que las acciones irreversibles pidan confirmación de todos
modos, con la puerta de permisos en código (ver `AGENTS.md`).

## Guardarraíles para modelos generativos

Las capas que contienen el daño cuando un modelo generativo está expuesto
(entrada hostil, filtros de salida, acciones limitadas, red teaming y
monitoreo) están en `guardarraíles.md`. El principio que las sostiene: **el
modelo propone, el sistema decide** — la defensa es la puerta de permisos en
código, no un prompt que "lo hace seguro".

## Secretos

Nunca loguear secretos: los logs son texto plano y viajan. No commitear
`.env` ni claves. Redactar credenciales antes de que lleguen a la ventana del
modelo o al log de auditoría (este proyecto lo hace en `agents/redaction.py`).

## Validación en la frontera de serving

{% if use_api %}
La API de este proyecto valida todo el input con pydantic en la frontera:
`strict=True` y `extra="forbid"` (ver `backend/api.md`). Un payload inválido
es un 422, nunca un 500; el modelo jamás recibe input sin validar.
{% endif %}

La entrada no validada es el vector más barato de explotar: límites de tamaño,
tipos y rangos en la frontera, rate limiting en los endpoints de inferencia
(caros) y auth por cabecera, nunca por query string.

## Escaneo de dependencias

`pip-audit` en CI: falla si una dependencia del lock tiene una CVE conocida.
Las dependencias se pinchan (`uv.lock`); un `pip install` flotante puede
traer una versión vulnerable sin que nadie se entere.

## Modelo de amenaza de un proyecto ML

| Etapa | Activos | Amenazas | Defensa mínima |
|-------|---------|----------|----------------|
| Datos | Dataset, etiquetas | Envenenamiento, fuga, personales | Checksum, acceso, minimización |
| Entrenamiento | Pesos, código, deps | Código malicioso, deps con CVE | CI, pip-audit, pin de deps |
| Serving | API, modelo, secretos | Adversariales, inversión, DoS | Validación pydantic, rate limit, vault |
| Consumidores | Outputs, dashboards | Mal uso, fuga por logs, pertenencia | Logs sin sensibles, monitoreo |

## Tensiones accuracy / fairness / seguridad

Los tres objetivos se pelean: el adversarial training baja la accuracy limpia;
la fairness cuesta exactitud; el ruido de la differential privacy degrada
ambas. No hay equilibrio mágico — hay **tradeoff explícito y auditable**:

- Reportar las tres métricas (accuracy, fairness por grupo, robustez) para las
  opciones candidatas, en una tabla.
- Registrar la decisión y el porqué (ADR / model card), con su umbral.
- Re-evaluar cuando cambia el modelo, los datos o la regulación.

Superficiar el tradeoff es la responsabilidad profesional; esconderlo es la
deuda que se descubre en una auditoría.

## Regulación (AI Act y compañía)

Cuando el proyecto toca personas (crédito, salud, contratación, educación),
la ética deja de ser opcional y pasa a ser ley. Lo que importa a nivel
operativo:

- **Tiers de riesgo del AI Act (UE)**: mínimo/limitado/alto/inaceptable. La
  mayoría de modelos de decisión sobre personas caen en **alto riesgo**: se
  exigen sistema de gestión de riesgos, gobernanza de datos, documentación
  técnica, registro de decisiones, supervisión humana y logging de eventos.
- **La respuesta documental es la que de verdad se audita**: model cards,
  ADRs con el umbral elegido, registro de datos (origen, licencia, sesgos
  conocidos) y de métricas por grupo. Lo que no está escrito no existe.
- **GDPR y privacidad de datos de usuario**: ver `privacidad-y-fuga-datos.md`
  — derecho de acceso/borrado, minimización, retención, y la fuga de datos de
  personas a través de logs, respuestas o el propio modelo.
- **Aplicabilidad**: el AI Act se aplica por el mercado (quien vende/sirve en
  la UE), no por dónde está el servidor; GDPR aplica a datos de residentes en
  la UE. La regla práctica: asume el estándar más estricto y documenta.
- **Práctica**: antes de desplegar un modelo que decide sobre personas —
  (1) identifica el tier de riesgo, (2) asegúrate de que las decisiones son
  explicables y reversibles (supervisión humana), (3) deja el rastro en el
  vault/registry, (4) re-evalúa cuando cambian el modelo, los datos o la ley.

## Fuentes

- **Fairness Definitions Explained** — S. Verma, J. Rubin (2018).
  arXiv:1807.09910 — https://arxiv.org/abs/1807.09910
- **Equality of Opportunity in Supervised Learning** — M. Hardt, E. Price,
  N. Srebro (2016). arXiv:1610.02413 — https://arxiv.org/abs/1610.02413
- **Inherent Trade-Offs in the Fair Determination of Risk Scores** —
  J. Kleinberg, S. Mullainathan, M. Raghavan (2016).
  arXiv:1609.05807 — https://arxiv.org/abs/1609.05807
- **Fairness Through Awareness** — C. Dwork, M. Hardt, T. Pitassi,
  O. Reingold, R. Zemel (2011). arXiv:1104.3913 — https://arxiv.org/abs/1104.3913
- **Explaining and Harnessing Adversarial Examples** — I. Goodfellow,
  J. Shlens, C. Szegedy (2014). arXiv:1412.6572 — https://arxiv.org/abs/1412.6572
- **Towards Deep Learning Models Resistant to Adversarial Attacks** —
  A. Madry, A. Makelov, L. Schmidt, D. Tsipras, A. Vladu (2017).
  arXiv:1706.06083 — https://arxiv.org/abs/1706.06083
- **Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning** —
  B. Biggio, F. Roli (2018). arXiv:1612.03156 — https://arxiv.org/abs/1612.03156
- **Privacy in Pharmacogenetics: An End-to-End Case Study of Personalized
  Warfarin Dosing** — M. Fredrikson et al. (2014).
  arXiv:1405.2476 — https://arxiv.org/abs/1405.2476
- **Membership Inference Attacks Against Machine Learning Models** —
  R. Shokri, M. Stronati, C. Song, V. Shmatikov (2016).
  arXiv:1610.05820 — https://arxiv.org/abs/1610.05820
- **Not what you've signed up for: Compromising Real-World LLM-Integrated
  Applications with Indirect Prompt Injection** — K. Greshake et al. (2023).
  arXiv:2302.12173 — https://arxiv.org/abs/2302.12173
