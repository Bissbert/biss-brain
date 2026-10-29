#!/usr/bin/env bash
# One-shot v0.6.1 brain remediation runner.
#
# Backs up ~/.claude/brain, shows both migration plans (dry-run), prompts for
# confirmation, applies them, and restarts the dream daemon. Safe to re-run —
# the migrations are idempotent.
#
# Usage:
#   scripts/run_v061_migrations.sh                     # interactive
#   scripts/run_v061_migrations.sh --yes               # skip confirmation
#   TARGET_PROJECT=<hash> scripts/run_v061_migrations.sh
set -euo pipefail

PLUGIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRAIN_ROOT="${HOME}/.claude/brain"
BACKUP_DIR="${HOME}"
TARGET_PROJECT="${TARGET_PROJECT:-fa373a4c0aa1}"   # biss-brain repo hash on this machine
PYTHON="${PLUGIN_ROOT}/.venv/bin/python"
AUTO_YES=0

for arg in "$@"; do
  case "$arg" in
    -y|--yes) AUTO_YES=1 ;;
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: venv missing at $PYTHON — run /biss-brain:install first" >&2
  exit 1
fi
if [ ! -d "$BRAIN_ROOT" ]; then
  echo "ERROR: brain root missing at $BRAIN_ROOT — nothing to migrate" >&2
  exit 1
fi

export PYTHONPATH="${PLUGIN_ROOT}/scripts:${PYTHONPATH:-}"

echo "=== biss-brain v0.6.1 remediation ==="
echo "  plugin:  $PLUGIN_ROOT"
echo "  brain:   $BRAIN_ROOT"
echo "  target:  $TARGET_PROJECT  (override with TARGET_PROJECT=<hash>)"
echo

# --- 1. Backup --------------------------------------------------------------
BACKUP_TARBALL="${BACKUP_DIR}/brain-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
echo "[1/5] Backing up brain to $BACKUP_TARBALL"
tar -czf "$BACKUP_TARBALL" -C "$(dirname "$BRAIN_ROOT")" "$(basename "$BRAIN_ROOT")"
echo "      done ($(du -h "$BACKUP_TARBALL" | cut -f1))"
echo

# --- 2. Dry-run plans -------------------------------------------------------
echo "[2/5] Fold-worktrees plan (dry run):"
"$PYTHON" "${PLUGIN_ROOT}/scripts/migrate_fold_worktrees.py"
echo
echo "[3/5] Recover-orphans plan (dry run):"
"$PYTHON" "${PLUGIN_ROOT}/scripts/migrate_recover_orphans.py" --target-project "$TARGET_PROJECT"
echo

# --- 3. Confirm -------------------------------------------------------------
if [ "$AUTO_YES" -ne 1 ]; then
  printf "Apply both migrations? [y/N] "
  read -r reply
  case "$reply" in
    y|Y|yes|YES) ;;
    *) echo "aborted; backup left at $BACKUP_TARBALL"; exit 0 ;;
  esac
fi

# --- 4. Apply ---------------------------------------------------------------
echo "[4/5] Applying fold-worktrees…"
"$PYTHON" "${PLUGIN_ROOT}/scripts/migrate_fold_worktrees.py" --apply
echo
echo "      Applying recover-orphans…"
"$PYTHON" "${PLUGIN_ROOT}/scripts/migrate_recover_orphans.py" \
  --target-project "$TARGET_PROJECT" --apply
echo

# --- 5. Restart dream daemon -----------------------------------------------
echo "[5/5] Restarting dream daemon (LaunchAgent)…"
DREAM_PLIST="${HOME}/Library/LaunchAgents/ch.bissbert.biss-brain.dream.plist"
if [ -f "$DREAM_PLIST" ]; then
  launchctl unload "$DREAM_PLIST" 2>/dev/null || true
  launchctl load "$DREAM_PLIST"
  echo "      reloaded $DREAM_PLIST"
else
  echo "      WARN: dream plist not found at $DREAM_PLIST — skipping restart"
fi

echo
echo "=== done ==="
echo "Backup:  $BACKUP_TARBALL"
echo "Verify:  open the UI (/biss-brain:ui), watch the status pill — CLI×N should be 0"
echo "         after the next dream pass; consolidation_round should start incrementing."
