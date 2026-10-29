#!/usr/bin/env bash
# Builds the plugin's .venv with `uv sync --frozen`. .venv/.biss-ready marks a
# finished build: uv creates .venv/bin/python before the packages are in, so
# the python alone does not mean the venv is usable.
#
# One build per plugin copy at a time. macOS has no flock(1), so the lock is a
# directory holding the builder's pid; a lock whose pid is gone is stale.
#
# usage: build_venv.sh [--wait]
#   --wait  when another build holds the lock, wait for it instead of returning
# exit: 0 ready, 75 another build is running, 127 uv not found, else uv's code
set -u

ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
READY="$ROOT/.venv/.biss-ready"
LOCK="$ROOT/.venv-build.lock"
UV="${BISS_BRAIN_UV:-uv}"  # tests point this at a stub
WAIT=0
[ "${1:-}" = "--wait" ] && WAIT=1

[ -f "$READY" ] && exit 0
if ! command -v "$UV" >/dev/null 2>&1; then
  echo "biss-brain: uv not found on PATH (brew install uv)" >&2
  exit 127
fi

nopid=0
until mkdir "$LOCK" 2>/dev/null; do
  pid="$(cat "$LOCK/pid" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    [ "$WAIT" = 1 ] || exit 75
    sleep 2
    continue
  fi
  # No pid yet: the owner may be between mkdir and writing it. Give it a moment.
  if [ -z "$pid" ] && [ "$nopid" -lt 3 ]; then
    nopid=$((nopid + 1))
    sleep 1
    continue
  fi
  rm -rf "$LOCK"
  nopid=0
done
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

[ -f "$READY" ] && exit 0  # a build finished while we waited
echo "biss-brain: building $ROOT/.venv ($(date '+%Y-%m-%d %H:%M:%S'))"
(cd "$ROOT" && "$UV" sync --frozen) || exit $?
touch "$READY"
echo "biss-brain: venv ready"
