#!/usr/bin/env bash
# One-time setup: delegates to scripts/install.py so the install command and
# the CLI bootstrap share the same logic.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on PATH" >&2
  exit 1
fi

CLAUDE_PLUGIN_ROOT="$ROOT" exec python3 "$ROOT/scripts/install.py" "$@"
