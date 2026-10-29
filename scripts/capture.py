#!/usr/bin/env python3
"""Hook entry point. Reads the hook JSON from stdin, writes a capture Event
to the project queue, ensures the flush daemon is running, exits in <500 ms.

The event records the transcript's byte offset and the (scrubbed) last
assistant message; it does not parse the transcript. The flush daemon reads
and distills each segment once (see segment.py).

PreCompact also writes a digest of the conversation tail for the compaction
restore (compact.py) and returns at once; the summary is distilled in the
background.

Invoked as: capture.py <kind> [--immediate]
where <kind> in {stop, subagent_stop, pre_compact, session_end}.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from common import (
    BRAIN_ROOT,
    Event,
    daemon_root,
    ensure_project_label,
    load_config,
    plugin_version,
    project_dir,
    queue_dir,
)
import scope
from scrub import scrub
from transcript import strip_private

# Fallback text if the transcript is gone by flush time.
_LAST_MESSAGE_CHARS = 4000


def _read_hook_input() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _replace_older(pid: int, plugin_root: Path) -> bool:
    """Stop daemon *pid* when it runs from an older plugin copy than ours.

    A Claude Code session keeps the plugin copy it started with, so after an
    update its hooks keep starting the old flush.py, old bugs included. The
    newer copy wins; the old session's hooks see a live daemon and leave it.
    """
    theirs = daemon_root(pid)
    if theirs is None or theirs == plugin_root:
        return False
    old, new = plugin_version(theirs), plugin_version(plugin_root)
    if old is None or new is None or old >= new:
        return False
    os.kill(pid, signal.SIGTERM)  # queued events stay on disk; the new daemon picks them up
    return True


def _ensure_daemon(plugin_root: Path) -> None:
    """Start flush.py daemon if not already running. Best-effort, never blocks."""
    pid_file = BRAIN_ROOT / "flush.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # signal 0: probe liveness
            if not _replace_older(pid, plugin_root):
                return  # already running
        except (ValueError, ProcessLookupError, PermissionError):
            pid_file.unlink(missing_ok=True)

    # Spawn detached daemon
    log_path = BRAIN_ROOT / "flush.log"
    BRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    runner = plugin_root / "scripts" / "_run.sh"
    flush_script = plugin_root / "scripts" / "flush.py"
    with open(log_path, "ab") as logf:
        subprocess.Popen(
            [str(runner), str(flush_script)],
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=logf,
            start_new_session=True,
            close_fds=True,
        )


def _write_event(ev: Event) -> Path:
    qdir = queue_dir(ev.cwd)
    fname = f"{ev.enqueued_ts:010d}-{ev.kind}-{uuid.uuid4().hex[:8]}.json"
    path = qdir / fname
    tmp = path.with_suffix(".tmp")
    tmp.write_text(ev.to_json())
    tmp.rename(path)  # atomic
    return path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: capture.py <kind> [--immediate]", file=sys.stderr)
        return 0

    kind = sys.argv[1]
    immediate = "--immediate" in sys.argv[2:]

    cfg = load_config()
    hook = _read_hook_input()

    cwd = hook.get("cwd") or os.getcwd()

    if not scope.check(cwd, cfg):
        return 0  # excluded or opted out ($HOME, Downloads, .biss-brain-ignore, ...)

    ensure_project_label(cwd)
    # A subagent's own transcript, when the harness names it; otherwise the
    # event joins the parent's segment like a Stop.
    own = kind == "subagent_stop" and bool(hook.get("agent_transcript_path"))
    transcript_path = hook["agent_transcript_path"] if own else hook.get("transcript_path", "")
    try:
        offset = os.path.getsize(transcript_path) if transcript_path else 0
    except OSError:
        offset = 0
    last = hook.get("last_assistant_message")
    excerpt = strip_private(last)[-_LAST_MESSAGE_CHARS:] if isinstance(last, str) else ""
    if excerpt and cfg["capture"].get("scrub_secrets", True):
        excerpt = scrub(excerpt)
    if not offset and not excerpt.strip():
        return 0  # nothing to distill

    ev = Event(
        kind=kind,
        session_id=hook.get("session_id", ""),
        agent_id=hook.get("agent_id"),
        cwd=cwd,
        transcript_path=transcript_path,
        transcript_excerpt=excerpt,
        sources=["assistant"] if excerpt else [],
        offset=offset,
        payload={
            k: hook.get(k)
            for k in ("trigger", "exit_reason", "reason", "agent_type", "tool_name")
            if k in hook
        },
    )

    if own:
        ev.payload["own_transcript"] = True
    if immediate:
        ev.payload["__immediate"] = True

    ev_path = _write_event(ev)
    if kind == "pre_compact":
        try:
            import compact

            compact.write_digest(project_dir(cwd), ev.session_id, ev_path.name, transcript_path, offset,
                                 excerpt, ev.enqueued_ts, cfg)
        except Exception:  # noqa: BLE001 — the restore is best effort; the event is queued
            pass

    env_root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    plugin_root = Path(env_root).resolve() if env_root else Path(__file__).resolve().parent.parent
    if (plugin_root / "scripts" / "_run.sh").is_file():
        _ensure_daemon(plugin_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
