#!/usr/bin/env bash
# =============================================================================
# backup-files.sh — archive uploaded file storage (attachments, data dirs)
# =============================================================================
#
# Environment:
#   DATA_DIR   — path to the data directory (default: ./data)
#   BACKUP_DIR — where to store tarballs (default: ./backups)
#   RETENTION  — days to keep file backups (default: 30)
#
# Usage:
#   ./scripts/backup-files.sh
#   DATA_DIR=/srv/astra/data ./scripts/backup-files.sh
# =============================================================================

set -euo pipefail

: "${DATA_DIR:=./data}"
: "${BACKUP_DIR:=./backups}"
: "${RETENTION:=30}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILENAME="${BACKUP_DIR}/astra_files_${TIMESTAMP}.tar.gz"

mkdir -p "$BACKUP_DIR"

if [ ! -d "$DATA_DIR" ]; then
    echo "[$(date +%H:%M:%S)] Data dir ${DATA_DIR} does not exist — nothing to back up."
    exit 0
fi

echo "[$(date +%H:%M:%S)] Archiving ${DATA_DIR} -> ${FILENAME}..."
tar czf "$FILENAME" -C "$(dirname "$DATA_DIR")" "$(basename "$DATA_DIR")"
echo "[$(date +%H:%M:%S)] File backup written to ${FILENAME}"

# Prune old file backups
find "$BACKUP_DIR" -name 'astra_files_*.tar.gz' -mtime +"$RETENTION" -delete 2>/dev/null || true

echo "[$(date +%H:%M:%S)] Done."
