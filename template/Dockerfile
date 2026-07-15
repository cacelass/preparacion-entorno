# Dockerfile — {{ project_name }}
# Generado por dskit (https://github.com/cacelass/dskit)
# ─────────────────────────────────────────────────────────────────────────────

# ── Imagen base ───────────────────────────────────────────────────────────────
FROM python:{{ python_version }}-slim

# ── Metadatos OCI ─────────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="{{ project_name }}"
LABEL org.opencontainers.image.version="{{ project_version }}"
LABEL org.opencontainers.image.description="{{ project_description }}"
LABEL dskit.template="https://github.com/cacelass/dskit"
LABEL dskit.ml_type="{{ ml_type }}"

WORKDIR /app

# ── Dependencias del sistema ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    figlet \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── uv — gestor de paquetes rapido ───────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Dependencias Python (capa de cache) ──────────────────────────────────────
# Copiamos solo pyproject.toml primero para aprovechar la cache de Docker.
# Instalamos sin -e: setuptools no necesita el codigo fuente todavia.
COPY pyproject.toml .

# Stub del paquete para que setuptools resuelva los extras sin el codigo real
RUN mkdir -p {{ project_slug }} && touch {{ project_slug }}/__init__.py

RUN uv pip install --system ".[{{ ml_type }}]" 2>/dev/null \
    || { echo "WARN: uv falló, usando pip"; pip install --no-cache-dir ".[{{ ml_type }}]"; }

# ── Interfaz web de chat ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    websockets

# ── Herramientas de documentacion y apuntes ──────────────────────────────────
RUN pip install --no-cache-dir "markitdown[all]" 2>/dev/null \
    || pip install --no-cache-dir markitdown 2>/dev/null \
    || echo "INFO: markitdown no disponible — omitido"

RUN pip install --no-cache-dir graphify 2>/dev/null \
    || pip install --no-cache-dir graphviz 2>/dev/null \
    || echo "INFO: graphify/graphviz no disponible — omitido"

# ── Assets del frontend (sin dependencia de CDN en runtime) ──────────────────
# Si la descarga falla no se crea el fichero:
#   StaticFiles devuelve 404 → onerror → CDN fallback → renderer inline.
RUN curl -fsSL --retry 3 \
      https://cdn.jsdelivr.net/npm/marked@9/marked.min.js \
      -o /tmp/marked.min.js 2>/dev/null || true

# ── Codigo del proyecto ───────────────────────────────────────────────────────
COPY . .

# Copiar marked.min.js solo si se descargo correctamente
RUN if [ -f /tmp/marked.min.js ]; then \
        cp /tmp/marked.min.js /app/chat/static/marked.min.js; \
    fi

# Instalar el paquete en modo editable ahora que el codigo fuente esta presente
RUN uv pip install --system -e . --no-deps 2>/dev/null \
    || { echo "WARN: uv falló, usando pip"; pip install --no-cache-dir -e . --no-deps; }

# ── Permisos del entrypoint ───────────────────────────────────────────────────
RUN chmod +x /app/chat/entrypoint.sh

# ── Usuario no root ──────────────────────────────────────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ── Directorios persistibles (montados como volumenes) ───────────────────────
VOLUME ["/app/models", "/app/data", "/app/apuntes"]

# ── Puerto de la interfaz web ─────────────────────────────────────────────────
EXPOSE 8080

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

# ── Punto de entrada ──────────────────────────────────────────────────────────
ENTRYPOINT ["/app/chat/entrypoint.sh"]
