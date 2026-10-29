"""The single path to `claude -p`. Public API: `run_claude(prompt, model, timeout) -> str | None`.

Every LLM call (distill, salience, merge, judge, dream arcs, remember classify)
goes through here so they all share:
  - the resolved CLI path (cli_resolver; never a bare "claude" on PATH)
  - cwd=BRAIN_ROOT, so the subprocess never inherits a project directory
  - no setting sources, MCP servers, slash commands or tools: no hooks or
    plugins load, so the child cannot recurse into capture. Not --bare: that
    reads auth only from ANTHROPIC_API_KEY and fails for a keychain login.
  - failure classification into cli_failures.jsonl for the status pill

Failure counter (read by the Wave 3 status pill):
  _FAILURE_COUNTER["claude_cli_missing"]  — find_claude_cli() returned None
  _FAILURE_COUNTER["claude_cli_error"]    — non-zero exit or is_error reply
  _FAILURE_COUNTER["claude_cli_timeout"]  — subprocess.TimeoutExpired
"""
from __future__ import annotations

import json
import logging
import subprocess
import time

import cli_resolver

log = logging.getLogger("biss-brain.cli")

# Pinned model ids. Changing them changes cost and behaviour, so it is a
# deliberate config decision, not a refactor side effect.
MODEL_MAP = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}

_FAILURE_COUNTER: dict[str, int] = {
    "claude_cli_missing": 0,
    "claude_cli_error": 0,
    "claude_cli_timeout": 0,
}


def _record_failure(kind: str) -> None:
    """Increment the in-process counter and persist a JSONL row for the UI.

    The status panel reads BRAIN_ROOT/cli_failures.jsonl and counts entries
    inside the last 24h. Best-effort: never raises to the caller.
    """
    _FAILURE_COUNTER[kind] = _FAILURE_COUNTER.get(kind, 0) + 1
    try:
        from common import BRAIN_ROOT  # noqa: PLC0415 — late: tests reload common
        BRAIN_ROOT.mkdir(parents=True, exist_ok=True)
        with (BRAIN_ROOT / "cli_failures.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": int(time.time()), "kind": kind}) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _workdir():
    """BRAIN_ROOT, created if missing (a missing cwd raises FileNotFoundError,
    which would otherwise be misread as a missing CLI)."""
    from common import BRAIN_ROOT  # noqa: PLC0415
    try:
        BRAIN_ROOT.mkdir(parents=True, exist_ok=True)
        return str(BRAIN_ROOT)
    except OSError:
        return None


def _reason(stdout: str | None) -> str:
    """The error text of a failed JSON reply, else the start of stdout."""
    try:
        data = json.loads(stdout or "")
        if isinstance(data, dict):
            return str(data.get("result") or data.get("subtype") or "")[:300]
    except (json.JSONDecodeError, TypeError):
        pass
    return (stdout or "").strip()[:300]


def run_claude(prompt: str, model: str = "haiku", timeout: int = 120) -> str | None:
    """Run `claude -p` headless and return the reply text, or None on any failure.

    None means "no answer" — callers must not turn it into a verdict or a
    truncated fallback.
    """
    cli = cli_resolver.find_claude_cli()
    if cli is None:
        _record_failure("claude_cli_missing")
        log.error("biss-brain: claude CLI unreachable")
        return None

    resolved = MODEL_MAP.get(model, model)
    # A NUL can come in with transcript text; argv cannot carry one.
    prompt = prompt.replace("\0", "")
    cmd = [
        str(cli), "-p", prompt,
        "--setting-sources", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--output-format", "json",
        "--model", resolved,
        "--tools", "",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
            cwd=_workdir(), stdin=subprocess.DEVNULL,  # -p appends piped stdin to the prompt
        )
    except subprocess.TimeoutExpired:
        _record_failure("claude_cli_timeout")
        log.warning("claude -p timed out (model=%s)", resolved)
        return None
    except FileNotFoundError:
        _record_failure("claude_cli_missing")
        log.error("biss-brain: claude CLI unreachable")
        return None
    except (OSError, ValueError) as e:
        _record_failure("claude_cli_error")
        log.warning("claude -p could not start: %s", e)
        return None

    if proc.returncode != 0:
        _record_failure("claude_cli_error")
        # With --output-format json the reason is in stdout, not stderr.
        log.warning("claude -p exit=%s stderr=%s stdout=%s", proc.returncode,
                    (proc.stderr or "").strip()[:300], _reason(proc.stdout))
        return None

    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        return (proc.stdout or "").strip() or None
    if not isinstance(data, dict):
        return (proc.stdout or "").strip() or None
    if data.get("is_error"):
        # exit 0 but the CLI reports failure (not logged in, rate limited, ...)
        _record_failure("claude_cli_error")
        log.warning("claude -p reported is_error (subtype=%s): %s", data.get("subtype"), _reason(proc.stdout))
        return None
    return (data.get("result") or "").strip() or None
