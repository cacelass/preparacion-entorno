# Chat web: interfaz de predicción

## Arquitectura mínima

Una página única (HTML+JS) que llama al backend de predicción. En este
proyecto, el backend es `chat/app.py` (FastAPI, puerto 8080) con los modelos
cargados en memoria; el chat expone `POST /api/predict`, `GET /api/status`,
`GET /api/reload` y `WS /ws`. La página (`chat/static/index.html`) no tiene
dependencias de build: JS vanilla servido por el propio backend.

El flujo base con `fetch`:

```js
async function predict(features) {
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
```

Regla de arquitectura: la UI es tonta. Toda la lógica (validación del
modelo, preprocesado, selección de modelo) vive en el backend. Si la UI
duplica la lógica de negocio, aparecen dos fuentes de verdad que divergen.

**Fallo en producción**: lógica de validación de features duplicada en JS y
en Python → el cliente acepta valores que el servidor rechaza (422) o, peor,
el servidor devuelve algo que la UI no sabe mostrar.

## Ciclo de vida de una petición

1. **Serializar input**: construir el JSON con las features exactas que espera
   el backend (nombres y tipos). `JSON.stringify` no convierte tipos por ti:
   un `input` de HTML es string; hay que `parseFloat`/`Number` donde el modelo
   espera número.
2. **fetch async**: nunca bloquear el hilo; `await` o `.then`.
3. **Estado de carga**: deshabilitar el botón de enviar, mostrar un indicador.
   Sin estado de carga, el usuario envía dos veces y duplica trabajo.
4. **Manejo de errores**, cada código con mensaje propio:
   - **Timeout** (la petición se cuelga): `AbortController` con un límite; el
     usuario debe poder reintentar.
   - **503** (modelo no disponible): "El modelo no está cargado. Ejecuta
     `make train` o `reload`." El backend avisa de esto en `/api/status`.
   - **422** (input inválido): mostrar el error del servidor junto al campo
     problemático, no un genérico "algo falló".
   - **Red caída**: mensaje de conexión perdida + reintento.

```js
const controller = new AbortController();
const timer = setTimeout(() => controller.abort(), 10000);
try {
  const res = await fetch('/api/predict', { signal: controller.signal, ... });
} catch (err) {
  if (err.name === 'AbortError') showError('La petición tardó demasiado.');
  else showError('No se pudo conectar con el servidor.');
} finally { clearTimeout(timer); }
```

**Fallo en producción**: tratar todos los errores igual ("Error de red") hace
imposible al usuario distinguir "el modelo no está" de "tu input está mal" de
"se cayó el servidor". Cada estado necesita su mensaje y su acción.

## UX básica

- **Validación client-side**: validar antes de enviar (campo vacío, tipo, rango)
  para evitar un round-trip; pero la validación de verdad es la del servidor —
  la del cliente es cortesía, no seguridad.
- **Enter para enviar**: `keydown` en el input con `e.key === 'Enter'`; sin
  ello, en móvil el usuario no puede enviar.
- **Historial en la página**: los mensajes se apilan en el DOM. Limitar el
  crecimiento (máx. N mensajes, hacer scroll, borrar el exceso) para que la
  página no se degrade en sesiones largas.
- **Debouncing**: agrupar eventos rápidos (tecleo para búsquedas, clics
  repetidos). En predicción, evitar el doble envío con el botón deshabilitado
  mientras la petición está en vuelo.

**Fallo en producción**: el doble clic de un usuario ansioso manda 2
predicciones (y si el endpoint es caro, 2× cómputo); el historial sin límite
congela el navegador tras cientos de mensajes.

## Streaming: SSE o WebSocket

Para respuestas largas (generación, token a token), una respuesta JSON única
obliga al usuario a esperar en blanco. Opciones:

- **SSE** (`EventSource`): el servidor empuja trozos de texto por HTTP. Simple,
  funciona con proxies HTTP normales, y el cliente no necesita librería. Ideal
  para streaming unidireccional de tokens.
- **WebSocket**: bidireccional, necesario si el cliente también envía a mitad
  de flujo (cancelar, peticiones concurrentes). Más complejo: reconexión,
  heartbeat, orden de mensajes.

Cuándo merece la pena: si la respuesta tarda más de ~1 s y se puede empezar a
mostrar antes de que termine, el streaming mejora la percepción de latencia
enormemente; para predicciones de una sola decisión en ms, añadir WebSocket es
complejidad sin retorno.

**Fallo en producción**: stream sin cancelación → el usuario cierra la página
y el servidor sigue computando; reconexión que duplica el stream (falta de
idempotencia en el protocolo del chat).

## Seguridad

- **Nunca claves en JS**: todo el JS viaja al navegador y es legible. API
  keys, tokens o secretos en el frontend son públicos por diseño. La
  autenticación del chat va por cookie/sesión del backend, nunca por clave
  embebida.
- **Sanitizar la salida del modelo antes de renderizar**: el texto que genera
  un modelo (o que devuelve el backend) es dato no confiable. Insertarlo con
  `innerHTML` directamente permite inyección si el modelo emite HTML/script.
  Escapar o construir nodos con `textContent`; si se renderiza markdown,
  hacerlo con un parser que escape antes.
- **CSRF/CORS**: el backend solo debe aceptar llamadas del origen del frontend
  (CORS con `allow_origins` explícito, no `*`); si la sesión usa cookies,
  proteger con tokens anti-CSRF. El `/api/predict` no debe ser invocable desde
  cualquier página web que navegue el usuario.
- **UI tonta**: el frontend no toma decisiones de negocio ni de seguridad;
  solo pinta y envía. Cualquier regla "en el cliente" es eludible.

**Fallo en producción**: `innerHTML` con la respuesta del modelo y una etiqueta
que contiene `<img src=x onerror=...>` → ejecución de script en la sesión del
usuario (stored XSS). Y un CORS `*` que permite a cualquier página a la que el
usuario navegue leer las predicciones.

## Accesibilidad y responsive

- Semántica básica: `button` para el envío (no `div` con onclick), `label`
  ligado al input, `aria-live` en la zona de mensajes para que un lector de
  pantalla anuncie la respuesta del bot.
- Contraste suficiente, foco visible, y `alt` en imágenes.
- Responsive: la caja de chat en móvil ocupa todo el ancho; el input y el botón
  se escalan; `viewport` meta declarado. Probar a 360 px de ancho, no solo en
  desktop.
- El estado de carga se anuncia también a lectores de pantalla (texto, no solo
  un spinner).

## Testing de la UI contra un mock

La UI se testea sin modelo real: un mock del backend que responde los mismos
contratos. Cobertura mínima:

- `/api/predict` 200 → se pinta la predicción y la confianza.
- `/api/predict` 503 → mensaje de modelo no disponible.
- `/api/predict` 422 → mensaje de input inválido con el detalle.
- Timeout (mock que no responde) → mensaje de timeout y botón re-habilitado.
- Reintento tras fallo de red.

```js
// test simple con fetch mockeado
const realFetch = window.fetch;
window.fetch = async () => new Response(
  JSON.stringify({ prediction: 1, probability: 0.9 }), { status: 200 });
await sendForm({ feat_0: 1.0 });          // ahora se renderiza la tarjeta
window.fetch = realFetch;
```

El mock valida el contrato (mismo shape de respuesta) sin depender de tener un
modelo entrenado. Ejecutarlo en CI con un navegador headless o con el
`TestClient` del backend sirviendo la página.

**Fallo en producción**: la UI probada solo contra el happy path con el modelo
cargado; el primer despliegue sin `make train` deja al usuario con un mensaje
de error genérico que nadie probó.

## Fuentes

- MDN Web Docs: `fetch`, `AbortController`, `EventSource`, WebSocket.
- OWASP: DOM-based XSS y sanitización de datos no confiables en el navegador.
- MDN: atributos ARIA para regiones live y accesibilidad de formularios.
- Documentación de FastAPI: `StaticFiles`, `HTMLResponse` y WebSocket.
