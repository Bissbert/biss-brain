"""Tests for the $HOME / root capture guard in scripts/capture.py.

The guard must:
- Return 0 immediately (without writing any queue file) when cwd is $HOME.
- Return 0 immediately when cwd is "/".
- Continue normally when cwd is a real project directory that happens to live
  inside $HOME (only the *literal* $HOME root is skipped).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _fresh_capture():
    """Import capture.py with a clean module state."""
    for mod in list(sys.modules.keys()):
        if mod in ("capture", "common", "scrub"):
            del sys.modules[mod]
    import capture  # noqa: PLC0415
    return capture


def _make_hook_payload(cwd: str) -> str:
    """Return a minimal hook JSON payload with the given cwd."""
    return json.dumps({"cwd": cwd, "session_id": "test-session", "transcript_path": "",
                       "last_assistant_message": "done"})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_home_guard_skips_when_cwd_is_home(tmp_path: Path) -> None:
    """main() must return 0 and write NO queue files when cwd == $HOME."""
    home = str(Path.home())

    capture = _fresh_capture()

    written_files: list[Path] = []

    def fake_write_event(ev):
        # Record that a write was attempted
        path = tmp_path / "fake_event.json"
        path.write_text(ev.to_json())
        written_files.append(path)
        return path

    hook_input = _make_hook_payload(home)

    with patch("sys.argv", ["capture.py", "stop"]), \
         patch("sys.stdin") as mock_stdin, \
         patch.object(capture, "_write_event", side_effect=fake_write_event), \
         patch.object(capture, "_ensure_daemon"):
        mock_stdin.read.return_value = hook_input
        result = capture.main()

    assert result == 0, f"Expected return code 0, got {result}"
    assert len(written_files) == 0, (
        f"Expected no queue files written for cwd=$HOME, but got: {written_files}"
    )


def test_home_guard_skips_when_cwd_is_root(tmp_path: Path) -> None:
    """main() must return 0 and write NO queue files when cwd == '/'."""
    capture = _fresh_capture()

    written_files: list[Path] = []

    def fake_write_event(ev):
        path = tmp_path / "fake_event_root.json"
        path.write_text(ev.to_json())
        written_files.append(path)
        return path

    hook_input = _make_hook_payload("/")

    with patch("sys.argv", ["capture.py", "stop"]), \
         patch("sys.stdin") as mock_stdin, \
         patch.object(capture, "_write_event", side_effect=fake_write_event), \
         patch.object(capture, "_ensure_daemon"):
        mock_stdin.read.return_value = hook_input
        result = capture.main()

    assert result == 0
    assert len(written_files) == 0, (
        f"Expected no queue files written for cwd='/', but got: {written_files}"
    )


def test_home_guard_does_not_skip_project_inside_home(tmp_path: Path) -> None:
    """A project directory inside $HOME must NOT be skipped.

    Only the literal $HOME root triggers the guard. Sub-directories are real
    project contexts.
    """
    # Use tmp_path which is typically inside /var/... or /tmp/..., but to be
    # safe we create a sub-directory explicitly to represent a "project inside
    # home" scenario.
    project_dir = tmp_path / "my_project"
    project_dir.mkdir()

    capture = _fresh_capture()

    written_files: list[Path] = []

    def fake_write_event(ev):
        path = tmp_path / "event.json"
        path.write_text(ev.to_json())
        written_files.append(path)
        return path

    hook_input = _make_hook_payload(str(project_dir))

    with patch("sys.argv", ["capture.py", "stop"]), \
         patch("sys.stdin") as mock_stdin, \
         patch.object(capture, "_write_event", side_effect=fake_write_event), \
         patch.object(capture, "_ensure_daemon"), \
         patch("common.ensure_project_label"):  # don't touch real brain root
        mock_stdin.read.return_value = hook_input
        result = capture.main()

    assert result == 0
    assert len(written_files) == 1, (
        f"Expected exactly 1 queue file for a real project directory, "
        f"but got: {written_files}"
    )


def test_home_guard_skips_home_symlink_resolved(tmp_path: Path) -> None:
    """Guard must trigger even when $HOME is passed as a symlink or relative path.

    We point $HOME at a fake home (scope.py expands "~" from it) and pass
    the cwd through a symlink to it.
    """
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    link = tmp_path / "homelink"
    link.symlink_to(fake_home)

    capture = _fresh_capture()

    written_files: list[Path] = []

    def fake_write_event(ev):
        path = tmp_path / "event_sym.json"
        path.write_text(ev.to_json())
        written_files.append(path)
        return path

    hook_input = _make_hook_payload(str(link))

    with patch("sys.argv", ["capture.py", "stop"]), \
         patch("sys.stdin") as mock_stdin, \
         patch.dict("os.environ", {"HOME": str(fake_home)}), \
         patch.object(capture, "_write_event", side_effect=fake_write_event), \
         patch.object(capture, "_ensure_daemon"):
        mock_stdin.read.return_value = hook_input
        result = capture.main()

    assert result == 0
    assert len(written_files) == 0, (
        f"Expected no queue file when cwd resolves to fake $HOME: {written_files}"
    )
