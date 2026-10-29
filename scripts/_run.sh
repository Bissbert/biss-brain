#!/usr/bin/env bash
# Wrapper that runs the plugin's Python with the bundled venv.
# Self-resolves CLAUDE_PLUGIN_ROOT from its own location so it works even
# when Claude Code does not inject the env var (e.g. in some hook contexts).
# Builds the venv on first use (scripts/build_venv.sh) and never blocks Claude on it.
set -e

# LaunchAgent + hook contexts inherit a sanitised PATH that omits user-shell
# additions. Without this, `shutil.which("claude")` in distill.py / judge.py
# returns None and every chunk falls back to passthrough truncation. Front-load
# the well-known install dirs so subprocess lookups succeed regardless of how
# this wrapper was invoked.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$HOME/.npm-global/bin:/usr/bin:/bin:${PATH:-}"

if [ -z "${CLAUDE_PLUGIN_ROOT:-}" ] || [ ! -d "${CLAUDE_PLUGIN_ROOT:-}" ]; then
  CLAUDE_PLUGIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  export CLAUDE_PLUGIN_ROOT
fi

# Hooks: leave out-of-scope directories before Python starts (scripts/scope.py
# writes the rules; it re-checks when the cache is missing). Never reads stdin.
_brain_out_of_scope() {
  local d="$1" cache="$2" t p inc=0 hit=0
  if [ -f "$cache" ]; then
    while read -r t p; do
      case "$t" in
        e) [ "$d" = "$p" ] && return 0 ;;
        x) case "$d/" in "${p%/}"/*) return 0 ;; esac ;;
        i) inc=1; case "$d/" in "${p%/}"/*) hit=1 ;; esac ;;
      esac
    done < "$cache"
    [ "$inc" = 1 ] && [ "$hit" = 0 ] && return 0
  fi
  p="$d"
  while :; do
    [ -e "$p/.biss-brain-ignore" ] && return 0
    if [ -f "$p/.claude/brain.toml" ] \
      && grep -Eq '^[[:space:]]*enabled[[:space:]]*=[[:space:]]*false' "$p/.claude/brain.toml"; then
      return 0
    fi
    [ -z "$p" ] || [ "$p" = "/" ] && break
    p="${p%/*}"
  done
  return 1
}
SCRIPT="${1##*/}"
MODE="${2:-}"
BRAIN="${BISS_BRAIN_ROOT:-$HOME/.claude/brain}"
case "$SCRIPT" in
  capture.py|inject_context.py)
    if _brain_out_of_scope "$(pwd -P)" "$BRAIN/scope.cache"; then
      exit 0
    fi
    ;;
esac

# Only the SessionStart hook speaks to the user: a hook that exits 0 shows
# nothing from stderr, so the hint is a JSON systemMessage on stdout.
_brain_hint() {
  if [ "$SCRIPT" = inject_context.py ] && [ "$MODE" = session_start ]; then
    printf '{"systemMessage": "biss-brain: %s"}\n' "$1"
  fi
}

# Each plugin version is a fresh copy without a venv. Build it here, so an
# update needs no manual step. .biss-ready marks a finished build.
if [ ! -f "$CLAUDE_PLUGIN_ROOT/.venv/.biss-ready" ]; then
  if ! command -v "${BISS_BRAIN_UV:-uv}" >/dev/null 2>&1; then
    _brain_hint "needs uv to set itself up. Install it (brew install uv), then run /biss-brain:install."
    echo "biss-brain: uv not found on PATH; install it, then run /biss-brain:install" >&2
    exit 0
  fi
  case "$SCRIPT" in
    dream.py|dmn.py|decay.py)
      # launchd jobs can wait for the build
      bash "$CLAUDE_PLUGIN_ROOT/scripts/build_venv.sh" --wait || exit 0
      ;;
    *)
      # hooks and the MCP server have seconds; the first build takes minutes
      LOG="$CLAUDE_PLUGIN_ROOT/venv-build.log"
      nohup bash "$CLAUDE_PLUGIN_ROOT/scripts/build_venv.sh" </dev/null >>"$LOG" 2>&1 &
      _brain_hint "setting up its Python environment in the background (a few minutes after an install or update). Memory capture resumes when it is done."
      echo "biss-brain: building the venv in the background, see $LOG" >&2
      exit 0
      ;;
  esac
fi

if [ ! -f "$BRAIN/config.toml" ] && [ "$SCRIPT" = inject_context.py ] && [ "$MODE" = session_start ]; then
  _brain_hint "is installed but not set up yet. Run /biss-brain:install to check Ollama and the embedding model and to schedule the nightly pass."
  exit 0
fi

# Record which plugin copy is live, for the launchd shim at $BRAIN/bin/run.
case "$CLAUDE_PLUGIN_ROOT" in
  */plugins/cache/*)
    if [ -d "$BRAIN" ]; then
      CUR=""
      [ -f "$BRAIN/active_root" ] && read -r CUR < "$BRAIN/active_root" || true
      [ "$CUR" = "$CLAUDE_PLUGIN_ROOT" ] || printf '%s\n' "$CLAUDE_PLUGIN_ROOT" > "$BRAIN/active_root"
    fi
    ;;
esac

exec "$CLAUDE_PLUGIN_ROOT/.venv/bin/python" "$@"
