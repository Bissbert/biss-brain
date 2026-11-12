"""Session start tells the user when a launchd job is pinned to one plugin copy."""
from __future__ import annotations

import plistlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import inject_context  # noqa: E402


def _plist(agents: Path, label: str, args: list[str]) -> None:
    agents.mkdir(parents=True, exist_ok=True)
    (agents / f"ch.bissbert.biss-brain.{label}.plist").write_bytes(
        plistlib.dumps({"Label": f"ch.bissbert.biss-brain.{label}", "ProgramArguments": args}))


def test_a_pinned_dream_job_is_named(tmp_path: Path) -> None:
    cache = "/x/.claude/plugins/cache/m/biss-brain/0.7.2/scripts"
    _plist(tmp_path, "dream", [f"{cache}/_run.sh", f"{cache}/dream.py"])
    msg = inject_context.pinned_jobs_notice(tmp_path)
    assert "dream job" in msg and "/biss-brain:install" in msg


def test_jobs_on_the_runner_say_nothing(tmp_path: Path) -> None:
    _plist(tmp_path, "dream", ["/x/.claude/brain/bin/run", "dream.py"])
    _plist(tmp_path, "dmn", ["/x/.claude/brain/bin/run", "dmn.py"])
    assert inject_context.pinned_jobs_notice(tmp_path) == ""


def test_no_jobs_or_a_broken_plist_say_nothing(tmp_path: Path) -> None:
    assert inject_context.pinned_jobs_notice(tmp_path) == ""
    (tmp_path / "ch.bissbert.biss-brain.dream.plist").write_text("not a plist")
    assert inject_context.pinned_jobs_notice(tmp_path) == ""


def test_the_notice_rides_in_the_hook_output(capsys) -> None:
    import json

    inject_context._emit("SessionStart", "ctx", "heads up")
    out = json.loads(capsys.readouterr().out)
    assert out["systemMessage"] == "heads up"
    assert out["hookSpecificOutput"]["additionalContext"] == "ctx"
