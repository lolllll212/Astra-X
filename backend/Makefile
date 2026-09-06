# Astra X — build, test, migrate, and deploy targets
#
# Usage:
#   make help           — show this help
#   make install        — install production deps
#   make dev-install    — install dev + production deps
#   make lint           — ruff check + mypy
#   make test           — run unit tests with coverage
#   make build          — build Docker image
#   make up             — docker compose up (dev)
#   make up-prod        — docker compose up (production)
#   make down           — docker compose down
#   make migrate        — run pending Alembic migrations
#   make revision       — create a new Alembic migration (msg=...)
#   make backup         — backup PostgreSQL + file store
#   make restore        — restore from a backup (file=...)
#   make deploy         — deploy to Kubernetes (kustomize)

.PHONY: help install dev-install lint test build up up-prod down \
        migrate revision backup restore deploy

help:
	@printf "Usage: make <target>\n\n"
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e ".[redis]"

dev-install: ## Install all dependencies (including dev)
	pip install -e ".[redis,dev]"

lint: ## Run Ruff linter + mypy type checker
	ruff check app/ tests/
	mypy app/

test: ## Run unit test suite with coverage
	pytest --cov=app --cov-report=term-missing

build: ## Build the production Docker image
	docker compose build

up: ## Start dev environment with Docker Compose
	docker compose up -d --build

up-prod: ## Start production environment
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

down: ## Tear down Docker Compose (stops all services)
	docker compose down

migrate: ## Run pending Alembic migrations
	alembic upgrade head

revision: ## Create a new Alembic migration (use msg="description")
	alembic revision --autogenerate -m "$(msg)"

backup: ## Backup PostgreSQL database and file store
	@mkdir -p backups
	@./scripts/backup-db.sh
	@./scripts/backup-files.sh
	@echo "Backup complete: backups/"

restore: ## Restore from a backup (use file=backups/astra_YYYYMMDD_HHMMSS.sql)
	@./scripts/restore.sh $(file)

deploy: ## Apply Kubernetes manifests via kustomize
	kubectl apply -k k8s/

down-all: ## Tear down everything, including volumes
	docker compose down -v
