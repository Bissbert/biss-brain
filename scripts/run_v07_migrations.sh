#!/usr/bin/env bash
# v0.7 schema migration runner: plan -> confirm -> backup -> apply.
#
# Migrates the existing store in place. The recommended path for v0.7 is a
# fresh start instead (scripts/fresh_start.py); use this only to keep the
# old memories searchable. Honours BISS_BRAIN_ROOT. Never touches launchd:
# the commands to reload the dream job are printed at the end.
#
# Usage:
#   scripts/run_v07_migrations.sh          # interactive
#   scripts/run_v07_migrations.sh --yes    # skip confirmation
set -euo pipefail

PLUGIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRAIN_ROOT="${BISS_BRAIN_ROOT:-$HOME/.claude/brain}"
PYTHON="${PLUGIN_ROOT}/.venv/bin/python"
AUTO_YES=0

for arg in "$@"; do
  case "$arg" in
    -y|--yes) AUTO_YES=1 ;;
    -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: venv missing at $PYTHON; run /biss-brain:install first" >&2
  exit 1
fi
if [ ! -d "$BRAIN_ROOT" ]; then
  echo "ERROR: no brain at $BRAIN_ROOT; nothing to migrate" >&2
  exit 1
fi
export BISS_BRAIN_ROOT="$BRAIN_ROOT"

echo "=== biss-brain v0.7 schema migration ==="
echo "[1/3] Plan (dry run):"
"$PYTHON" "${PLUGIN_ROOT}/scripts/migrate_v07.py"
echo

if [ "$AUTO_YES" -ne 1 ]; then
  printf "Back up and apply? [y/N] "
  read -r reply
  case "$reply" in
    y|Y|yes|YES) ;;
    *) echo "aborted; nothing changed"; exit 0 ;;
  esac
fi

echo "[2/3] Backup and apply:"
"$PYTHON" "${PLUGIN_ROOT}/scripts/migrate_v07.py" --apply
echo

echo "[3/3] Check:"
"$PYTHON" "${PLUGIN_ROOT}/scripts/doctor.py" || true
echo
echo "To reload the nightly dream job with this plugin version, run:"
echo "  launchctl bootout gui/\$(id -u)/ch.bissbert.biss-brain.dream 2>/dev/null"
echo "  launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/ch.bissbert.biss-brain.dream.plist"
