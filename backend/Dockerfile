# Multi-stage Docker build for Astra X backend

# ---- Build stage ----
FROM python:3.13-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir build && \
    pip install --no-cache-dir -e ".[dev]"

# ---- Runtime stage ----
FROM python:3.13-slim AS runtime

# Create non-root user
RUN addgroup --system --gid 1001 astra && \
    adduser --system --uid 1001 --gid 1001 astra

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app ./app
COPY alembic.ini .
COPY alembic/ ./alembic/

RUN mkdir -p /app/data && chown -R astra:astra /app/data

USER astra

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
