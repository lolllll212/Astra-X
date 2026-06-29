# Astra X Backend

A modular, local-first AI assistant backend with an agent framework, tool-calling,
memory, and pluggable LLM providers.

## Quick start

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
option with documentation.

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
