#!/usr/bin/env bash
# =============================================================================
# backup-db.sh — PostgreSQL backup with pg_dump
# =============================================================================
# Runs inside the db container (docker compose) or against a remote host.
#
# Environment:
#   PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE
#   BACKUP_DIR  — local directory to write backups (default: ./backups)
#   RETENTION   — days to keep backups (default: 30)
#
# Usage:
#   ./scripts/backup-db.sh
#   PGHOST=prod-db.example.com ./scripts/backup-db.sh
# =============================================================================

set -euo pipefail

: "${PGHOST:=db}"
: "${PGPORT:=5432}"
: "${PGUSER:=astra}"
: "${PGPASSWORD:=astra}"
: "${PGDATABASE:=astra}"
: "${BACKUP_DIR:=./backups}"
: "${RETENTION:=30}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILENAME="${BACKUP_DIR}/astra_${TIMESTAMP}.sql"
LOGFILE="${BACKUP_DIR}/backup.log"

mkdir -p "$BACKUP_DIR"

export PGPASSWORD

echo "[$(date +%H:%M:%S)] Starting backup of ${PGDATABASE}@${PGHOST}..."
pg_dump \
    --host="$PGHOST" \
    --port="$PGPORT" \
    --username="$PGUSER" \
    --dbname="$PGDATABASE" \
    --format=custom \
    --compress=9 \
    --file="$FILENAME" \
    --verbose \
    --no-owner \
    --no-privileges \
    2>> "$LOGFILE"

echo "[$(date +%H:%M:%S)] Backup written to ${FILENAME}"

# Prune old backups
find "$BACKUP_DIR" -name 'astra_*.sql' -mtime +"$RETENTION" -delete 2>/dev/null || true

echo "[$(date +%H:%M:%S)] Done. Retention: ${RETENTION} days."
echo "---" >> "$LOGFILE"
