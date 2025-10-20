#!/bin/bash
# WAL compaction script for Fractal LBA backend (CLAUDE_PHASE2 WP6)
#
# Purpose:
#   Remove old WAL entries beyond retention window to prevent disk pressure
#
# Usage:
#   ./scripts/wal-compact.sh /data/wal 14
#
# Arguments:
#   $1: WAL directory path
#   $2: Retention days (default: 14, matches dedup TTL)
#
# Scheduling:
#   Run as Kubernetes CronJob daily:
#     kubectl create cronjob wal-compact \
#       --image=busybox \
#       --schedule="0 2 * * *" \
#       -- /scripts/wal-compact.sh /data/wal 14

set -euo pipefail

# Configuration
WAL_DIR="${1:-/data/wal}"
RETENTION_DAYS="${2:-14}"
DRY_RUN="${DRY_RUN:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $*" >&2
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $*" >&2
    exit 1
}

# Validate inputs
if [[ ! -d "$WAL_DIR" ]]; then
    error "WAL directory not found: $WAL_DIR"
fi

if [[ ! $RETENTION_DAYS =~ ^[0-9]+$ ]] || [[ $RETENTION_DAYS -lt 1 ]]; then
    error "Invalid retention days: $RETENTION_DAYS (must be positive integer)"
fi

log "Starting WAL compaction"
log "WAL directory: $WAL_DIR"
log "Retention: $RETENTION_DAYS days"
log "Dry run: $DRY_RUN"

# Calculate cutoff timestamp
CUTOFF_SECONDS=$(($(date +%s) - (RETENTION_DAYS * 86400)))
CUTOFF_DATE=$(date -d "@$CUTOFF_SECONDS" +'%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r $CUTOFF_SECONDS +'%Y-%m-%d %H:%M:%S')
log "Cutoff date: $CUTOFF_DATE"

# Find WAL files
WAL_FILES=$(find "$WAL_DIR" -type f -name "*.wal" 2>/dev/null | sort)
if [[ -z "$WAL_FILES" ]]; then
    warn "No WAL files found in $WAL_DIR"
    exit 0
fi

TOTAL_FILES=$(echo "$WAL_FILES" | wc -l)
log "Found $TOTAL_FILES WAL files"

# Calculate disk usage before
DISK_USAGE_BEFORE=$(du -sh "$WAL_DIR" | cut -f1)
log "Disk usage before: $DISK_USAGE_BEFORE"

# Compact old files
DELETED_COUNT=0
DELETED_SIZE=0

while IFS= read -r wal_file; do
    # Get file modification time
    FILE_MTIME=$(stat -c %Y "$wal_file" 2>/dev/null || stat -f %m "$wal_file")

    if [[ $FILE_MTIME -lt $CUTOFF_SECONDS ]]; then
        FILE_SIZE=$(du -b "$wal_file" | cut -f1)
        FILE_DATE=$(date -d "@$FILE_MTIME" +'%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r $FILE_MTIME +'%Y-%m-%d %H:%M:%S')

        if [[ "$DRY_RUN" == "true" ]]; then
            log "Would delete: $wal_file (modified: $FILE_DATE, size: $FILE_SIZE bytes)"
        else
            log "Deleting: $wal_file (modified: $FILE_DATE, size: $FILE_SIZE bytes)"
            rm -f "$wal_file"
        fi

        DELETED_COUNT=$((DELETED_COUNT + 1))
        DELETED_SIZE=$((DELETED_SIZE + FILE_SIZE))
    fi
done <<< "$WAL_FILES"

# Calculate disk usage after
DISK_USAGE_AFTER=$(du -sh "$WAL_DIR" | cut -f1)

# Format deleted size
if [[ $DELETED_SIZE -gt 1073741824 ]]; then
    DELETED_SIZE_HUMAN="$(awk "BEGIN {printf \"%.2f\", $DELETED_SIZE/1073741824}")GB"
elif [[ $DELETED_SIZE -gt 1048576 ]]; then
    DELETED_SIZE_HUMAN="$(awk "BEGIN {printf \"%.2f\", $DELETED_SIZE/1048576}")MB"
elif [[ $DELETED_SIZE -gt 1024 ]]; then
    DELETED_SIZE_HUMAN="$(awk "BEGIN {printf \"%.2f\", $DELETED_SIZE/1024}")KB"
else
    DELETED_SIZE_HUMAN="${DELETED_SIZE}B"
fi

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Compaction Summary"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Total WAL files: $TOTAL_FILES"
log "Deleted files: $DELETED_COUNT"
log "Remaining files: $((TOTAL_FILES - DELETED_COUNT))"
log "Space freed: $DELETED_SIZE_HUMAN"
log "Disk usage before: $DISK_USAGE_BEFORE"
log "Disk usage after: $DISK_USAGE_AFTER"

if [[ "$DRY_RUN" == "true" ]]; then
    warn "DRY RUN MODE - No files were actually deleted"
    log "Run without DRY_RUN=true to perform actual compaction"
fi

log "✓ WAL compaction completed successfully"
