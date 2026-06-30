# Astra X Backend

A modular, local-first AI assistant backend with an agent framework, tool-calling,
memory, and pluggable LLM providers.

## Quick start (development)

```bash
cp .env.example .env          # edit as needed
py -3 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload
```

## Environment variables

All config is through `ASTRA_`-prefixed env vars. See `.env.example` for every
option with documentation. See `.env.production.example` for production defaults.

## Run migrations

```bash
alembic upgrade head          # apply all pending migrations
alembic downgrade -1          # roll back one step
alembic revision --autogenerate -m "description"
```

## Run the server

```bash
# development (hot reload)
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# production
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Run tests

```bash
pytest --cov=app --cov-report=term-missing
```

## Lint & type-check

```bash
ruff check app/
mypy app/
```

## Makefile

```bash
make help           # show all targets
make dev-install    # install all deps
make lint           # ruff + mypy
make test           # run tests
make build          # docker build
make up             # docker compose up (dev)
make up-prod        # docker compose up (production)
make migrate        # run alembic migrations
make revision msg="description"  # create migration
make backup         # backup postgres + file store
make deploy         # apply k8s manifests
```

## Docker Compose (development)

```bash
cp .env.example .env
docker compose up -d --build
```

Starts the app, PostgreSQL 16, and Redis 7. The app connects to PostgreSQL
automatically via the compose network.

## Docker Compose (production)

```bash
cp .env.production.example .env.production
# Edit .env.production with real secrets
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Production overrides:
- No host port exposure (reverse-proxy handles ingress)
- Docker secrets for the database password
- Resource limits on every container
- Local log driver (binary, rotation built in)

## Kubernetes

Kustomize manifests are in `k8s/`. They deploy:

| Component   | Kind        | Replicas | Notes                          |
|-------------|-------------|----------|--------------------------------|
| Backend     | Deployment  | 2-10     | HPA at 70% CPU / 80% memory   |
| PostgreSQL  | StatefulSet | 1        | 10 GiB PVC, pg_isready probes  |
| Redis       | Deployment  | 1        |                                |

```bash
kubectl apply -k k8s/
```

Requires:
- An `ingress-nginx` controller in the cluster
- `cert-manager` with a `letsencrypt-prod` ClusterIssuer
- A `postgres-credentials` Secret with keys `password` and `database-url`

## Backup strategy

Two scripts in `scripts/`:

- `scripts/backup-db.sh` — `pg_dump` (custom format, compressed), 30-day retention
- `scripts/backup-files.sh` — tarball of the data directory, 30-day retention
- `scripts/restore.sh` — `pg_restore` + Alembic upgrade

Run via Makefile:

```bash
make backup          # creates files in backups/
make restore file=backups/astra_20260101_120000.sql
```

For production, schedule these via cron or a K8s CronJob:

```yaml
# 2am daily backup
0 2 * * * cd /srv/astra && ./scripts/backup-db.sh && ./scripts/backup-files.sh
```

## CI/CD

| Workflow   | Trigger              | Actions                                                |
|------------|----------------------|--------------------------------------------------------|
| CI         | PR / push to `main`  | Ruff → Mypy → Pytest (with PostgreSQL) → Alembic check |
| Deploy     | Tag `v*.*.*`         | Ruff → Mypy → Pytest → Build & push image → K8s apply  |

## Secrets management

| Method               | Usage                                      |
|----------------------|--------------------------------------------|
| Environment vars     | Development (`ASTRA_SECRET_KEY`, etc.)     |
| Docker secrets       | Production Compose (`secrets/db_password`) |
| Kubernetes Secrets   | K8s (`astra-x-secret`, `postgres-credentials`) |
| SecretStr (Pydantic) | In-app validation prevents exposure in logs |

## Production checklist

- [ ] `ASTRA_SECRET_KEY` set to a 64-char hex string, not the dev default
- [ ] `ASTRA_ALLOWED_HOSTS` locked to the real domain
- [ ] `ASTRA_CORS_ORIGINS` locked to the frontend domain
- [ ] `ASTRA_DATABASE_URL` points to PostgreSQL, not SQLite
- [ ] `ASTRA_LOG_FORMAT=json` for structured logging
- [ ] Database password stored in a secret manager, not in the repo
- [ ] TLS terminates at the ingress/reverse-proxy
- [ ] Backups scheduled and tested

## Architecture

- `app/api/` — FastAPI routes, schemas, middleware
- `app/agents/` — agent framework (planner, executor, reflection, coordinator)
- `app/llm/` — provider adapters (Ollama, LM Studio, OpenAI-compatible) + router
- `app/tools/` — tool registry, executor, built-in tools, filesystem, web, GitHub, Python REPL
- `app/memory/` — memory subsystem (embeddings, chunking, retrieval, consolidation)
- `app/services/` — orchestration layer (chat, conversations, providers, memory)
- `app/domain/` — domain models, enums, streaming events
- `app/database/` — SQLAlchemy models, repositories, Alembic migrations
- `app/config/` — pydantic-settings configuration

## License

MIT
