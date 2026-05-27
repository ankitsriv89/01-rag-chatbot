# ─── Dockerfile ───────────────────────────────────────────────────────────────
# Multi-stage build for the FastAPI backend.
#
# CONCEPT — Multi-stage builds:
#   Stage 1 (builder): installs all deps including build tools (gcc, etc.)
#   Stage 2 (runtime): copies only what's needed — much smaller final image.
#   Result: production image is ~60% smaller than a single-stage build.
#
# CONCEPT — Why non-root user?
#   Running as root inside a container is a security risk.
#   If an attacker escapes the app, they get root on the host.
#   We create a dedicated `appuser` with no special privileges.

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for some Python packages (faiss-cpu, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (Docker layer caching)
# If requirements.txt hasn't changed, this layer is reused — faster builds.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy application source
COPY app/ ./app/

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

USER appuser

# ── Environment ───────────────────────────────────────────────────────────────
# These are defaults — override with real values via docker-compose or .env
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# ── Health check ──────────────────────────────────────────────────────────────
# Docker/Kubernetes checks this to know if the container is healthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

EXPOSE 8000

# ── Start command ─────────────────────────────────────────────────────────────
# --workers 1: single worker (FAISS is in-memory per-process; multi-worker needs Chroma/Pinecone)
# --host 0.0.0.0: listen on all interfaces inside the container
# --timeout-keep-alive 75: keep connections alive longer (good for streaming)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "75"]
