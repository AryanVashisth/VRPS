#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FOLDER_ID="10BdxeVioWVp5AVuJl0Pz9RHdyYRRvK02"
REMOTE="${RCLONE_REMOTE:-gdrive}"
rclone copy "$ROOT/meetings" "${REMOTE}:" --drive-root-folder-id "$FOLDER_ID" -P --create-empty-src-dirs
rclone copy "$ROOT/Documentation" "${REMOTE}:" --drive-root-folder-id "$FOLDER_ID" -P
rclone copy "$ROOT/exports" "${REMOTE}:" --drive-root-folder-id "$FOLDER_ID" -P
rclone copy "$ROOT/pilot/INTERN_USER_MANUAL.md" "${REMOTE}:" --drive-root-folder-id "$FOLDER_ID" -P
rclone copy "$ROOT/pilot/VALIDATION_AUDIT.md" "${REMOTE}:" --drive-root-folder-id "$FOLDER_ID" -P
rclone copy "$ROOT/ACCESS.md" "${REMOTE}:" --drive-root-folder-id "$FOLDER_ID" -P
echo "Done. Check https://drive.google.com/drive/folders/$FOLDER_ID"
