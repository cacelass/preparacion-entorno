# API REST con FastAPI

## Endpoints síncronos vs asíncronos

Un endpoint `def` (síncrono) corre en el threadpool de Starlette (máx. 40
threads por defecto); un `async def` corre en el event loop. La distinción
manda en servir modelos:

- **`async def`**: paga cuando el cuerpo hace I/O que bloquea esperando
  (llamadas a otro servicio, base de datos, disco). Mientras espera, el event
  loop sigue atendiendo otras peticiones.
- **`def`**: correcto para cómputo. FastAPI lo manda al threadpool, que
  paraleliza entre cores; el event loop no se bloquea.

El caso prohibido: cómputo CPU-bound pesado (una inferencia de un modelo
grande, transformación de features) dentro de un `async def` bloquea el event
loop completo: ninguna otra petición avanza mientras corre. Reglas:

```python
# MAL: la inferencia bloquea el event loop
@app.post("/predict")
async def predict(req: PredictRequest):
    return model.predict(req.features)  # CPU-bound en async

# BIEN: el threadpool la ejecuta
@app.post("/predict")
def predict(req: PredictRequest):
    return model.predict(req.features)

# BIEN: I/O async (llamar a otro servicio de scoring)
@app.post("/predict")
async def predict(req: PredictRequest):
    return await scoring_service.forward(req.features)
```

**Fallo en producción**: la inferencia en `async def` hace que una sola
petición costosa congele todo el servicio; el p99 salta a los segundos sin que
la CPU esté saturada. El preprocesado (pandas/numpy) también es CPU-bound: si
es caro, va al threadpool o a un worker dedicado.

## Pydantic: validación, 422 vs 500, y respuesta

Los schemas son la frontera del servicio: todo lo que entra se valida contra
ellos, y lo que sale se filtra contra el `response_model`.

- **`422 Unprocessable Entity`**: el payload no pasa la validación (falta un
  campo, tipo incorrecto, valor fuera de rango). Es un error del cliente: el
  servidor nunca debería crashear por un body mal formado.
- **`500 Internal Server Error`**: el código crasheó durante el handler. Es un
  bug del servidor. Un endpoint bien construido solo devuelve 500 por causas
  no anticipadas (bug, infra), nunca por el input.

`model_config` para endurecer el contrato:

```python
from pydantic import BaseModel, ConfigDict, Field

class PredictRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    features: dict[str, float] = Field(..., description="feature → valor")
```

- `strict=True`: sin coerción automática de tipos (p.ej. no convierte "1.5"
  en float). El cliente debe mandar el tipo exacto.
- `extra="forbid"`: rechaza campos desconocidos. Evita que features
  mal escritas se ignoren en silencio.
- `response_model` en el decorador filtra y serializa la respuesta: no se
  filtran campos no declarados, y los valores se validan al salir.

Errores: `HTTPException(status_code, detail)` para errores de negocio (503 sin
modelo, 422 por features desconocidas). Para errores con forma propia se
registran handlers:

```python
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def on_validation_error(_, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "input invalido",
                                                  "details": exc.errors()})
```

**Fallo en producción**: validación laxa (`strict=False`, `extra="ignore"`)
deja pasar features con el tipo equivocado que el modelo interpreta mal; el
400/422 se vuelve un 200 con predicción basura. Y validar a mano dentro del
handler en vez de usar el schema → lógica duplicada y errores inconsistentes.

## Dependencias, lifespan y estado compartido

El modelo se carga una vez y se comparte entre peticiones. Nada de cargarlo
dentro de cada handler.

- `lifespan`: arranque y apagado del servicio. El sitio para cargar el modelo,
  el scaler, los encoders y hacer `model.eval()`.
- `app.state`: el lugar para colgar el estado compartido; es un singleton por
  proceso de uvicorn.
- `Depends`: inyección para cross-cutting (auth, rate limit, acceso al modelo).
  Se puede usar para exponer el modelo sin globals:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()      # una vez por proceso
    yield
    app.state.model = None

app = FastAPI(lifespan=lifespan)

def get_model(request: Request):
    return request.app.state.model

@app.post("/predict")
def predict(req: PredictRequest, model=Depends(get_model)):
    return model.predict(req.features)
```

**Fallo en producción**: cargar el modelo en el arranque de cada worker es
lento (multiplicado por el número de workers); usar un pool de workers con
carga lazy + timeout corto provoca "carga bajo demanda" en el primer pico de
tráfico. El modelo en memoria hace que cada worker duplique la RAM: presupuesta
`workers × tamaño_del_modelo`.

## Routing y versionado

- Prefija la API con versión (`/v1`) desde el primer día: romper el contrato
  sin cambiar de ruta rompe a los clientes. Un `APIRouter(prefix="/v1")` o el
  parámetro `root_path` del app.
- `tags=["Predicción"]` agrupa los endpoints en OpenAPI.
- OpenAPI es gratis en FastAPI: `/docs` (Swagger UI) y `/openapi.json` son el
  contrato consumible por el cliente. Documenta los esquemas con descripciones
  y `json_schema_extra` para ejemplos.

**Fallo en producción**: no versionar y luego querer cambiar el schema del
`/predict` (rompe clientes desplegados); o versionar la ruta pero no el schema,
que es lo que de verdad cambia el contrato.

## Serving ML: schemas, incertidumbre, batch y streaming

- **Request/response tipados**: el request lleva las features; la respuesta
  lleva predicción y, cuando se puede, confianza. Si el modelo expone
  probabilidades o intervalos (calibración, conformal), devolverlos como
  campos opcionales.
- **Batch**: el endpoint acepta una lista y devuelve una lista. Mejora el
  throughput y evita N peticiones HTTP; un batch parcialmente fallido debe
  fallar como 422 con el índice del primer ejemplo inválido, no devolver
  predicciones mezcladas.
- **Streaming**: para respuestas largas (generación, scores por lote) usar
  `StreamingResponse` o WebSocket; el cliente debe poder abortar.

**Fallo en producción**: devolver solo el índice de la clase sin la
probabilidad hace al cliente incapaz de umbralizar; el batch sin límite de
tamaño permite a un cliente mandar 100k filas y tumbar la RAM del worker.

## Producción: límites, auth, CORS, workers

- **Rate limiting**: detrás de un proxy o middleware. Los endpoints de
  inferencia son caros: sin límite, un cliente que buclea agota el threadpool
  para todos.
- **Auth**: API keys (simples, para M2M) o bearer JWT (si hay usuarios). Las
  keys se validan por cabecera, nunca por query string (quedan en logs y
  cachés). Verificar en un `Depends`.
- **CORS**: `allow_origins` limitado a los orígenes reales del frontend. No
  usar `"*"` si se envían cookies o credenciales.
- **Workers**: `uvicorn` solo (1 proceso) no escala de verdad. En producción,
  `gunicorn -k uvicorn.workers.UvicornWorker -w N`. El número de workers escala
  CPU-bound; los threads (dentro de uvicorn) escalan I/O-bound. Para un modelo
  en memoria, `N` está limitado por la RAM total.
- **Reverse proxy**: nginx/Traefik delante para TLS, cabeceras, límite de body
  y rate limit a nivel de red.

**Fallo en producción**: `--reload` en producción (reinicia por cambios),
timeout del proxy menor que el p99 del modelo (peticiones válidas cortadas), o
el threadpool saturado por peticiones de validación baratas que no dejan pasar
a la inferencia.

## Testing con TestClient

```python
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_schema_ok():
    resp = client.post("/v1/predict", json={"features": {"feat_0": 1.0}})
    assert resp.status_code == 200
    assert resp.json()["prediction"] is not None

def test_predict_input_invalido_es_422():
    resp = client.post("/v1/predict", json={"features": "no soy dict"})
    assert resp.status_code == 422

def test_predict_sin_modelo_es_503():
    # con el modelo ausente o mockeado a no cargado
    resp = client.post("/v1/predict", json={"features": {"feat_0": 1.0}})
    assert resp.status_code == 503
```

Cubre los tres caminos que fallan en producción: validación (422), ausencia de
pesos (503), y el happy path con un modelo pequeño de prueba. El caso "sin
modelo" se testea mockeando el estado o con `models/` vacío, no esperando a
que ocurra en el despliegue.

**Fallo en producción**: testear solo el 200; el 503 llega en producción con
el primer deploy antes de `make train` y el cliente no lo maneja.

## Fuentes

- Documentación de FastAPI: first steps, dependencies, lifespan, testing,
  deployment en producción (gunicorn/uvicorn).
- Documentación de Starlette: threadpool de `run_in_threadpool`, middleware.
- Documentación de Pydantic v2: `model_config`, strict mode, validación.
- FastAPI issues conocidos sobre endpoints síncronos vs asíncronos y el event loop.
