# Docker en producción

## Dockerfile: buenas prácticas

- **Imagen base fijada**: `python:3.12-slim` siempre con digest o tag exacto.
  `FROM python` (latest) o `python:latest` es un bug: el build de mañana puede
  ser otro sistema operativo y otra versión de Python. La reproducibilidad del
  build empieza en la base.
- **Multi-stage**: el stage de build instala compiladores y dependencias de
  build; el stage final solo copia artefactos y runtime. La imagen final no
  lleva gcc ni caches de pip.
- **Usuario no root**: `USER` no-root obligatorio. El proceso del contenedor
  no debe correr con permisos del host; si se rompe, el atacante no hereda
  root.
- **`.dockerignore`**: excluir `.git`, `models/`, `data/`, `.venv`,
  `.rag-index/`, notebooks. Si no existe, el contexto del build (y el cache)
  incluye el dataset entero.

## Orden de COPY, cache de capas y apt

El cache de Docker se invalida por capa: si una instrucción cambia, todas las
siguientes se re-ejecutan. Ordenar de lo que cambia menos a lo que cambia más:

```dockerfile
FROM python:3.12-slim

# 1. apt (cambia rara vez)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. dependencias Python (cambia solo cuando pyproject cambia)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# 3. el código (cambia en cada commit)
COPY . .

# 4. runtime
USER 1000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0"]
```

- `COPY` vs `ADD`: usar `COPY` salvo que se necesiten las features de `ADD`
  (URLs, extraer tarballs). `ADD` con URLs mete red y sorpresas en el build.
- `apt`: `--no-install-recommends` y borrar `/var/lib/apt/lists/*` en la misma
  capa (si no, el cache de apt engorda la imagen).
- Un solo `apt-get update && install` en la misma capa: separarlos permite que
  el install use listas viejas y falle en build futuro con paquetes caídos.

**Fallo en producción**: `COPY . .` primero (invalida todo el cache en cada
commit → builds de 10 min en vez de 30 s), o `RUN pip install` después del
código (re-instala deps en cada cambio de código).

## Healthchecks, CMD y ENTRYPOINT

- `HEALTHCHECK` en el Dockerfile: el orquestador solo envía tráfico a
  contenedores sanos. Apuntar a un endpoint de readiness ligero, no a una
  página pesada.
- `ENTRYPOINT` fija el ejecutable (no sobreescribible por `docker run` con
  argumentos); `CMD` da los argumentos por defecto. `ENTRYPOINT ["uvicorn"]`
  + `CMD ["api.main:app"]` permite `docker run ... --port 9000`.
- `exec-form` (`["uvicorn", ...]`) y nunca `shell-form` (`uvicorn ...`): el
  shell-form lanza `sh -c` y los señales (SIGTERM) llegan al shell, no al
  proceso; el graceful shutdown se rompe.

## Tamaño y superficie de ataque

- Preferir `slim`/`distroless`: menos binarios = menos CVEs potenciales y
  imagen más pequeña. `distroless` no trae shell ni package manager: es lo
  correcto para producción y lo incómodo para depurar (no hay `exec` bash).
- Limpiar caches en la misma capa que los crea: `pip install --no-cache-dir`,
  `rm -rf /root/.cache`, `find ... -name '*.pyc' -delete`.
- Instalar solo lo que el runtime necesita. Cada paquete extra es superficie
  de ataque y bytes.

**Fallo en producción**: imagen de 2 GB con gcc y toolchains de build por no
usar multi-stage → más superficie, más lenta de escanear y de desplegar; o
`FROM python:latest` que hoy resuelve y mañana rompe el pipeline de CI.

## docker-compose

{% if use_docker %}
La imagen de este proyecto sirve la interfaz web de chat (`chat/app.py`,
puerto 8080): el contenedor expone el chat, y `docker-compose.yml` arranca el
servicio con volumenes para `models/` y `data/`, healthcheck sobre
`/api/status` y límite de memoria de 4G. `make docker-run` equivale a
`docker compose up -d` + abrir `http://localhost:8080`.
{% endif %}

```yaml
services:
  app:
    build: .
    restart: unless-stopped
    depends_on:
      db:
        condition: service_healthy     # espera el healthcheck, no solo el start
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models:ro        # modelos en solo lectura
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/models/model.joblib
    deploy:
      resources:
        limits: { memory: 4G }
```

- `depends_on` con `condition: service_healthy`: espera a que la dependencia
  esté realmente lista, no a que el contenedor arranque.
- Límites de `memory`/`cpus`: evitan que un runaway mate el host. Con
  `restart: unless-stopped` un crash se recupera solo.
- Volúmenes para modelo y datos: lo que cambia en runtime no vive en la capa
  de la imagen; se monta. El contenedor es efímero, los datos persistentes.

**Fallo en producción**: `depends_on` sin health (arranca el app antes que la
dependencia), o el modelo dentro de la imagen (cada rebuild lo recalcula, y
escalar = duplicar el modelo por réplica).

## Runtime: proceso dentro del contenedor

- **Cómo llega al host**: solo por `ports`/`expose`. Nada de ejecutar el
  servicio contra el network namespace del host.
- **Config por env vars**: config y secretos por `environment`/`.env`, nunca
  en el código ni en la imagen. En producción, secretos vía orquestador
  (Docker secrets, k8s secrets), no en compose.
- **Logs a stdout**: el contenedor escribe logs en stdout/stderr; el
  orquestador los captura y enruta. Escribir a fichero dentro del contenedor
  los pierde en cada recreación.
- **SIGTERM = apagado elegante**: el proceso debe escuchar SIGTERM, parar de
  aceptar tráfico, terminar las peticiones en vuelo y salir. Sin `exec-form`
  o sin manejador de señales, el orquestador acaba con `SIGKILL` a mitad de
  una petición.

**Fallo en producción**: logs a fichero que "desaparecen" al recrear el
contenedor, y apagados que cortan predicciones a mitad porque nadie maneja
SIGTERM.

## Seguridad

- Nunca root: además del `USER` no-root, la imagen no debe instalar `sudo`.
- Nada de secretos en capas: un `ARG`/`ENV` con una clave queda en la capa
  (y en el historial de la imagen). Los build args y las claves van por
  secretos del build (BuildKit `--secret`), que no persisten en capas.
- Fijar checksums donde sea posible: en bases y binarios descargados, pin con
  digest. La cadena de suministro empieza por saber exactamente qué corre.
- Escanear la imagen final (trivy, grype) en CI; bloquear CVEs críticos antes
  del push al registry.

**Fallo en producción**: la clave en una `ENV` del Dockerfile que se commitea
y queda en el historial de la imagen (visible en Docker Hub); o `USER root`
que convierte un RCE en el contenedor en acceso root al host (si el
mount/escape lo permite).

## Fuentes

- Docker docs: Dockerfile best practices, multi-stage builds, .dockerignore.
- Docker compose spec: `depends_on`, `deploy.resources`, healthchecks.
- OWASP Docker Security Cheat Sheet.
- Documentación de BuildKit: `--mount=type=secret` para secretos sin capas.
