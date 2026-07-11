# Security Architecture

## Secrets Management

### Principles

1. **No hardcoded secrets in source** — all credentials are injected at
   deployment time via environment variables, Docker Secrets, or mounted
   files.
2. **Defense in depth** — secrets are stored encrypted at rest and
   transmitted only over TLS.
3. **Least privilege** — each service gets only the secrets it needs.
4. **Short-lived when possible** — database passwords support rotation
   without downtime.

### Supported injection methods

| Method | Use case | Mechanism |
|--------|----------|-----------|
| **Environment variables** | Local dev, simple deploys | `.env` file loaded by pydantic-settings |
| **`<FIELD>_FILE` env vars** | Docker Secrets, K8s Secrets | Settings reads file at `ASTRA_SECRET_KEY_FILE` |
| **Kubernetes Secret mounts** | K8s production | `env.valueFrom.secretKeyRef` in `deployment.yaml` |
| **Vault agent sidecar** | Enterprise | Vault writes secrets to a shared volume; app reads via `<FIELD>_FILE` |
| **AWS Secrets Manager** | AWS ECS/EKS | Sidecar or init container retrieves and writes to tmpfs |

### Docker Compose secrets

For `docker compose -f docker-compose.yml -f docker-compose.prod.yml`:

```
./secrets/db_password.txt          # Postgres password (referenced by db service)
./secrets/astra_secret_key.txt     # ASTRA_SECRET_KEY (referenced by app service)
./secrets/astra_database_url.txt   # Full database URL (referenced by app service)
```

Create these files **outside version control**:

```bash
python -c "import secrets; pw=secrets.token_hex(16); open('secrets/db_password.txt','w').write(pw); print(f'postgresql+asyncpg://astra:{pw}@db:5432/astra')" > secrets/astra_database_url.txt
python -c "import secrets; open('secrets/astra_secret_key.txt','w').write(secrets.token_hex(32))"
```

### Kubernetes secrets

Edit `k8s/secret.yaml`:

```bash
# Generate values
SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
DB_PASSWORD=$(python -c "import secrets; print(secrets.token_hex(16))")
DATABASE_URL="postgresql+asyncpg://astra:${DB_PASSWORD}@postgres:5432/astra"

# Apply to cluster
kubectl create secret generic postgres-credentials \
  --from-literal=password="$DB_PASSWORD" \
  --from-literal=database-url="$DATABASE_URL" \
  -n astra-x

kubectl create secret generic astra-x-secret \
  --from-literal=ASTRA_SECRET_KEY="$SECRET_KEY" \
  -n astra-x
```

## Secret Rotation

### Key rotation strategy

| Secret | Rotation frequency | Method |
|--------|-------------------|--------|
| `ASTRA_SECRET_KEY` | Every 90 days | Rolling update: deploy new pods, then revoke old key after TTL |
| Database password | Every 180 days | Dual-credential window: update app first, then update database |
| LLM API keys | On compromise or annually | Update provider config, restart pods |

### `ASTRA_SECRET_KEY` rotation procedure

1. Generate a new key:
   ```bash
   NEW_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
   ```
2. Deploy a new Secret with the old key as fallback (if session tokens
   are still signed with the old key):
   ```yaml
   ASTRA_SECRET_KEY: "<new-key>"
   ASTRA_SECRET_KEY_OLD: "<old-key>"   # optional: used to verify existing tokens
   ```
3. After twice the maximum token lifetime (default 24 h), remove
   `ASTRA_SECRET_KEY_OLD` and repeat from step 1.

### Database password rotation (zero-downtime)

1. Set `POSTGRES_PASSWORD` to the new value and reload Postgres:
   ```sql
   ALTER USER astra PASSWORD 'new-password';
   ```
2. Update the app's secret with the new password.
3. Roll out the app deployment (rolling update, no downtime).
4. After all pods are running with the new password, revoke the old one:
   ```sql
   ALTER USER astra PASSWORD 'new-password';  -- already set; just confirms
   ```

## CORS & Host Restrictions

| Setting | Development | Production |
|---------|-------------|------------|
| `ASTRA_ALLOWED_HOSTS` | `localhost,127.0.0.1` | specific domain(s) |
| `ASTRA_CORS_ORIGINS` | `http://localhost:5173` | `https://yourdomain.com` |
| `ASTRA_CORS_ALLOW_CREDENTIALS` | `true` | `true` if using cookies |

The production safety validator (`Settings._enforce_production_safety`)
rejects the application startup if:
- `ASTRA_SECRET_KEY` is not a 64-character hex string
- `ASTRA_ALLOWED_HOSTS` contains `*`
- `ASTRA_CORS_ORIGINS` contains `*`
- `ASTRA_DEBUG` is `true`
- `ASTRA_DATABASE_ECHO` is `true`

## Pre-deployment validation

Run the deployment validation script before any production push:

```bash
python scripts/validate_deployment.py
```

This checks:
- All required env vars are set
- `ASTRA_SECRET_KEY` is a valid 64-char hex string
- `ASTRA_ALLOWED_HOSTS` does not contain `*`
- `ASTRA_CORS_ORIGINS` does not contain `*`
- `ASTRA_DEBUG` is `false`
- `ASTRA_ENVIRONMENT` is `production`
- Database driver is `postgresql+asyncpg` (not `sqlite`)
