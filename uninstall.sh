#!/usr/bin/env bash
# Uninstall biss-brain. By default removes background services and venv but
# preserves captured memories at ~/.claude/brain/<project-hash>/. Pass
# --purge-data to also delete every project's brain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRAIN_ROOT="$HOME/.claude/brain"
PLIST_DST="$HOME/Library/LaunchAgents/ch.bissbert.biss-brain.dream.plist"
LEGACY_PLIST="$HOME/Library/LaunchAgents/ch.bissbert.biss-brain.decay.plist"

PURGE_DATA=0
KEEP_VENV=0
YES=0

for arg in "$@"; do
  case "$arg" in
    --purge-data) PURGE_DATA=1 ;;
    --keep-venv)  KEEP_VENV=1 ;;
    --yes|-y)     YES=1 ;;
    -h|--help)
      cat <<'EOF'
usage: uninstall.sh [--purge-data] [--keep-venv] [--yes]

Default behaviour:
  - stops the flush daemon
  - stops the UI server
  - unloads and removes the launchd agents (dream, dmn) and their runner
  - removes the venv at ./.venv
  - PRESERVES captured memories at ~/.claude/brain/<project-hash>/

Flags:
  --purge-data   also delete ~/.claude/brain/ (all projects, all memories)
  --keep-venv    keep ./.venv (default removes it)
  --yes, -y      skip the confirmation prompt for --purge-data
EOF
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      echo "run with --help for usage." >&2
      exit 2
      ;;
  esac
done

echo "-> biss-brain uninstall"

# ── Stop UI server ──────────────────────────────────────────────────────
UI_PID_FILE="$BRAIN_ROOT/ui.pid"
if [ -f "$UI_PID_FILE" ]; then
  UI_PID=$(cat "$UI_PID_FILE" 2>/dev/null || true)
  if [ -n "${UI_PID:-}" ] && kill -0 "$UI_PID" 2>/dev/null; then
    echo "-> stopping UI server (pid $UI_PID)"
    kill "$UI_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5 6; do
      kill -0 "$UI_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$UI_PID" 2>/dev/null || true
  fi
  rm -f "$UI_PID_FILE" "$BRAIN_ROOT/ui.port" "$BRAIN_ROOT/ui.log"
fi

# ── Stop flush daemon ───────────────────────────────────────────────────
FLUSH_PID_FILE="$BRAIN_ROOT/flush.pid"
if [ -f "$FLUSH_PID_FILE" ]; then
  FLUSH_PID=$(cat "$FLUSH_PID_FILE" 2>/dev/null || true)
  if [ -n "${FLUSH_PID:-}" ] && kill -0 "$FLUSH_PID" 2>/dev/null; then
    echo "-> stopping flush daemon (pid $FLUSH_PID)"
    kill "$FLUSH_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5 6; do
      kill -0 "$FLUSH_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$FLUSH_PID" 2>/dev/null || true
  fi
  rm -f "$FLUSH_PID_FILE"
fi

# ── Unload launchd dream agent (and any legacy decay plist) ─────────────
if [ -f "$PLIST_DST" ]; then
  echo "-> unloading nightly dream launchd agent"
  launchctl unload "$PLIST_DST" 2>/dev/null || true
  rm -f "$PLIST_DST"
fi
DMN_PLIST="$HOME/Library/LaunchAgents/ch.bissbert.biss-brain.dmn.plist"
if [ -f "$DMN_PLIST" ]; then
  echo "-> unloading DMN launchd agent"
  launchctl unload "$DMN_PLIST" 2>/dev/null || true
  rm -f "$DMN_PLIST"
fi
rm -f "$BRAIN_ROOT/bin/run" "$BRAIN_ROOT/active_root"
rmdir "$BRAIN_ROOT/bin" 2>/dev/null || true
if [ -f "$LEGACY_PLIST" ]; then
  echo "-> unloading legacy decay launchd agent"
  launchctl unload "$LEGACY_PLIST" 2>/dev/null || true
  rm -f "$LEGACY_PLIST"
fi

# ── Remove venv ─────────────────────────────────────────────────────────
if [ "$KEEP_VENV" -eq 0 ] && [ -d "$ROOT/.venv" ]; then
  echo "-> removing $ROOT/.venv"
  rm -rf "$ROOT/.venv" "$ROOT/.venv-build.lock"
fi

# ── Purge data (opt-in) ─────────────────────────────────────────────────
if [ "$PURGE_DATA" -eq 1 ]; then
  if [ -d "$BRAIN_ROOT" ]; then
    PROJECTS=$(find "$BRAIN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
    if [ "$YES" -eq 0 ]; then
      echo ""
      echo "WARNING: this will delete $BRAIN_ROOT and all $PROJECTS project memories."
      printf "type 'yes' to proceed: "
      read -r CONFIRM
      if [ "$CONFIRM" != "yes" ]; then
        echo "-> data preserved (confirmation declined)"
        exit 0
      fi
    fi
    echo "-> deleting $BRAIN_ROOT ($PROJECTS project(s))"
    rm -rf "$BRAIN_ROOT"
  fi
else
  if [ -d "$BRAIN_ROOT" ]; then
    PROJECTS=$(find "$BRAIN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
    echo "-> data preserved at $BRAIN_ROOT ($PROJECTS project(s)). pass --purge-data to remove."
  fi
fi

echo "-> uninstall complete"
echo ""
echo "to also remove the plugin from Claude Code, disable it via your marketplace UI"
echo "or remove the source repo at $ROOT."
