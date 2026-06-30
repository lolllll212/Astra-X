# Multi-stage Docker build for Astra X backend
# Target: production on PostgreSQL. SQLite works in dev mode.

# ============================================================
# Stage 1 — build: install all dependencies into a venv
# ============================================================
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY pyproject.toml README.md ./

RUN python -m venv /venv && \
    /venv/bin/pip install -e ".[redis]"

# ============================================================
# Stage 2 — runtime: minimal image with only the venv + code
# ============================================================
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH"

# Create non-root user
RUN addgroup --system --gid 1001 astra && \
    adduser --system --uid 1001 --gid 1001 astra && \
    mkdir -p /app/data && \
    chown -R astra:astra /app

# Copy venv from builder
COPY --from=builder /venv /venv

# Copy application code
WORKDIR /app
COPY app ./app
COPY alembic.ini .
COPY alembic/ ./alembic/
COPY pyproject.toml README.md ./

USER astra

EXPOSE 8000

# Run migrations on startup, then start the server.
# ASTRA_DATABASE_URL must point to a running PostgreSQL (or SQLite in dev).
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"
