#!/usr/bin/env bash
# =============================================================================
# restore.sh — restore PostgreSQL database from a pg_dump custom-format backup
# =============================================================================
# Usage:
#   ./scripts/restore.sh backups/astra_20260101_120000.sql
#   PGHOST=db ./scripts/restore.sh backups/astra_20260101_120000.sql
#
# Environment:
#   PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE
#   DROP_FIRST — set to "yes" to drop the database before restoring (default: no)
#
# ⚠  DROP_FIRST=yes terminates all connections to the target database and
#   drops it. Use with extreme caution in production.
# =============================================================================

set -euo pipefail

: "${PGHOST:=db}"
: "${PGPORT:=5432}"
: "${PGUSER:=astra}"
: "${PGPASSWORD:=astra}"
: "${PGDATABASE:=astra}"
: "${DROP_FIRST:=no}"

BACKUP_FILE="${1:?Usage: $0 <backup-file.sql>}"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: ${BACKUP_FILE}"
    exit 1
fi

export PGPASSWORD

if [ "$DROP_FIRST" = "yes" ]; then
    echo "[$(date +%H:%M:%S)] Dropping database ${PGDATABASE}..."

    # Terminate existing connections and drop
    psql --host="$PGHOST" --port="$PGPORT" --username="$PGUSER" \
        --dbname="postgres" \
        -c "SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '${PGDATABASE}'
              AND pid <> pg_backend_pid();" 2>/dev/null || true

    dropdb --host="$PGHOST" --port="$PGPORT" --username="$PGUSER" "$PGDATABASE"
    createdb --host="$PGHOST" --port="$PGPORT" --username="$PGUSER" "$PGDATABASE"

    echo "[$(date +%H:%M:%S)] Database dropped and recreated."
fi

echo "[$(date +%H:%M:%S)] Restoring ${BACKUP_FILE} to ${PGDATABASE}..."

pg_restore \
    --host="$PGHOST" \
    --port="$PGPORT" \
    --username="$PGUSER" \
    --dbname="$PGDATABASE" \
    --format=custom \
    --verbose \
    --no-owner \
    --no-privileges \
    --exit-on-error \
    "$BACKUP_FILE"

echo "[$(date +%H:%M:%S)] Restore complete."

# Run Alembic migrations to bring the schema to the latest revision
echo "[$(date +%H:%M:%S)] Running Alembic migrations..."
alembic upgrade head
echo "[$(date +%H:%M:%S)] Migrations applied. Restore finished successfully."
