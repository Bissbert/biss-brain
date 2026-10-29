#!/usr/bin/env bash
# Copied to ~/.claude/brain/bin/run by /biss-brain:install; the launchd jobs
# run it instead of a versioned path, so they survive plugin updates.
#
# usage: run <script.py> [args...]
# Runs scripts/<script.py> from the live plugin copy: the one _run.sh last
# recorded in active_root, else the newest installed version.
set -u

BRAIN="${BISS_BRAIN_ROOT:-$HOME/.claude/brain}"
ROOT=""
[ -f "$BRAIN/active_root" ] && read -r ROOT < "$BRAIN/active_root"
if [ -z "$ROOT" ] || [ ! -x "$ROOT/scripts/_run.sh" ]; then
  # Sort on the version folder only; the marketplace folder before it would win otherwise.
  ROOT="$(for d in "$HOME"/.claude/plugins/cache/*/biss-brain/*/; do
            [ -x "${d}scripts/_run.sh" ] || continue
            d="${d%/}"
            printf '%s\t%s\n' "${d##*/}" "$d"
          done | sort -V -k1,1 | tail -1 | cut -f2-)"
fi
if [ -z "$ROOT" ]; then
  echo "biss-brain: no installed plugin copy found" >&2
  exit 0
fi
SCRIPT="$1"
shift
exec "$ROOT/scripts/_run.sh" "$ROOT/scripts/$SCRIPT" "$@"
