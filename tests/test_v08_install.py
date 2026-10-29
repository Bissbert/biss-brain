"""v0.8: the venv builds itself, the launchd runner follows updates, first-run hints."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import doctor as D  # noqa: E402
import install  # noqa: E402

RUN = REPO / "scripts" / "_run.sh"
BUILD = REPO / "scripts" / "build_venv.sh"
RUNNER = REPO / "scripts" / "launchd_run.sh"


def _root(base: Path) -> Path:
    """A plugin copy with the real launcher scripts and no venv."""
    root = base / "plugin"
    (root / "scripts").mkdir(parents=True)
    for f in (RUN, BUILD):
        (root / "scripts" / f.name).write_text(f.read_text())
        (root / "scripts" / f.name).chmod(0o755)
    return root


def _stub_uv(base: Path, marker: Path, delay: float = 0) -> Path:
    """A uv whose sync makes a python that records its arguments."""
    uv = base / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        f"sleep {delay}\n"
        "mkdir -p .venv/bin\n"
        f"printf '#!/bin/sh\\necho \"$@\" >> {marker}\\n' > .venv/bin/python\n"
        "chmod +x .venv/bin/python\n"
    )
    uv.chmod(0o755)
    return uv


def _env(root: Path, brain: Path, uv: Path | str, home: Path | None = None) -> dict:
    return {"PATH": "/usr/bin:/bin", "HOME": str(home or brain.parent), "CLAUDE_PLUGIN_ROOT": str(root),
            "BISS_BRAIN_ROOT": str(brain), "BISS_BRAIN_UV": str(uv)}


def _hook(root, env, cwd, script="inject_context.py", mode="session_start"):
    return subprocess.run(["bash", str(root / "scripts" / "_run.sh"), str(root / "scripts" / script), mode],
                          cwd=cwd, env=env, input="{}", text=True, capture_output=True, timeout=10)


def _wait_for(path: Path, seconds: float = 10) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def test_hook_builds_the_venv_in_the_background_and_says_so(tmp_path):
    root, brain, marker = _root(tmp_path), tmp_path / "brain", tmp_path / "started"
    brain.mkdir()
    (brain / "config.toml").write_text("")
    env = _env(root, brain, _stub_uv(tmp_path, marker, delay=1))

    t = time.perf_counter()
    r = _hook(root, env, tmp_path)
    assert time.perf_counter() - t < 1.0  # does not wait for the 1 s build
    assert r.returncode == 0
    assert "Python environment" in json.loads(r.stdout)["systemMessage"]
    assert not marker.exists()

    assert _wait_for(root / ".venv" / ".biss-ready")
    assert not (root / ".venv-build.lock").exists()
    r = _hook(root, env, tmp_path)
    assert r.returncode == 0 and r.stdout == ""
    assert marker.read_text().split()[-1] == "session_start"


def test_capture_hook_builds_without_a_message(tmp_path):
    root, brain = _root(tmp_path), tmp_path / "brain"
    env = _env(root, brain, _stub_uv(tmp_path, tmp_path / "started"))
    r = _hook(root, env, tmp_path, "capture.py", "stop")
    assert r.returncode == 0 and r.stdout == ""
    assert _wait_for(root / ".venv" / ".biss-ready")


def test_without_uv_the_session_start_hint_says_how_to_get_it(tmp_path):
    root, brain = _root(tmp_path), tmp_path / "brain"
    r = _hook(root, _env(root, brain, tmp_path / "no-such-uv"), tmp_path)
    assert r.returncode == 0
    assert "brew install uv" in json.loads(r.stdout)["systemMessage"]
    assert not (root / ".venv").exists()


def test_launchd_scripts_wait_for_the_build(tmp_path):
    root, brain, marker = _root(tmp_path), tmp_path / "brain", tmp_path / "started"
    env = _env(root, brain, _stub_uv(tmp_path, marker, delay=0.5))
    r = _hook(root, env, tmp_path, "dream.py", "")
    assert r.returncode == 0 and marker.exists()


def test_missing_config_asks_for_install_once_the_venv_is_ready(tmp_path):
    root, brain, marker = _root(tmp_path), tmp_path / "brain", tmp_path / "started"
    env = _env(root, brain, _stub_uv(tmp_path, marker))
    assert subprocess.run(["bash", str(root / "scripts" / "build_venv.sh")], env=env).returncode == 0
    r = _hook(root, env, tmp_path)
    assert "/biss-brain:install" in json.loads(r.stdout)["systemMessage"]
    assert not marker.exists()
    _hook(root, env, tmp_path, "capture.py", "stop")  # capture still runs on defaults
    assert marker.exists()


def test_a_running_build_is_not_started_twice_and_a_stale_lock_is_cleared(tmp_path):
    root, brain = _root(tmp_path), tmp_path / "brain"
    env = _env(root, brain, _stub_uv(tmp_path, tmp_path / "started"))
    build = ["bash", str(root / "scripts" / "build_venv.sh")]
    lock = root / ".venv-build.lock"
    lock.mkdir()
    holder = subprocess.Popen(["sleep", "30"])
    try:
        (lock / "pid").write_text(f"{holder.pid}\n")
        assert subprocess.run(build, env=env).returncode == 75
    finally:
        holder.kill()
        holder.wait()
    assert subprocess.run(build, env=env, timeout=10).returncode == 0  # the holder is gone
    assert (root / ".venv" / ".biss-ready").is_file() and not lock.exists()


def test_active_root_is_recorded_only_for_an_installed_copy(tmp_path):
    brain, marker = tmp_path / "brain", tmp_path / "started"
    brain.mkdir()
    (brain / "config.toml").write_text("")
    dev = _root(tmp_path / "dev")
    cached = _root(tmp_path / ".claude" / "plugins" / "cache" / "m" / "biss-brain" / "0.8.0")
    for root in (dev, cached):
        env = _env(root, brain, _stub_uv(tmp_path, marker))
        subprocess.run(["bash", str(root / "scripts" / "build_venv.sh")], env=env, check=True)
        _hook(root, env, tmp_path, "capture.py", "stop")
        if root is dev:
            assert not (brain / "active_root").exists()
    assert (brain / "active_root").read_text() == f"{cached}\n"


def _copy(home: Path, market: str, ver: str) -> Path:
    root = home / ".claude" / "plugins" / "cache" / market / "biss-brain" / ver
    (root / "scripts").mkdir(parents=True)
    run = root / "scripts" / "_run.sh"
    run.write_text(f"#!/bin/sh\necho {market}/{ver} \"$@\"\n")
    run.chmod(0o755)
    return root


def _runner(home: Path, brain: Path) -> str:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "BISS_BRAIN_ROOT": str(brain)}
    return subprocess.run(["bash", str(RUNNER), "dream.py", "--stage", "1"], env=env,
                          capture_output=True, text=True, timeout=10, check=True).stdout.strip()


def test_runner_uses_the_recorded_copy_else_the_newest_version(tmp_path):
    home, brain = tmp_path / "home", tmp_path / "brain"
    brain.mkdir()
    _copy(home, "bissbert", "0.9.0")
    newest = _copy(home, "biss-claude-plugins", "0.10.0")  # sorts first by folder, last by version
    out = _runner(home, brain)
    assert out == f"biss-claude-plugins/0.10.0 {newest}/scripts/dream.py --stage 1"

    old = _copy(home, "bissbert", "0.8.0")
    (brain / "active_root").write_text(f"{old}\n")
    assert _runner(home, brain).startswith("bissbert/0.8.0 ")
    (brain / "active_root").write_text(f"{tmp_path / 'gone'}\n")
    assert _runner(home, brain).startswith("biss-claude-plugins/0.10.0 ")


def test_plists_run_the_stable_runner():
    for tpl, script in ((install.DREAM_PLIST_TEMPLATE, "dream.py"), (install.DMN_PLIST_TEMPLATE, "dmn.py")):
        text = tpl.format(runner="/r/bin/run", log_path="/l.log")
        assert "<string>/r/bin/run</string>" in text and f"<string>{script}</string>" in text
        assert "CLAUDE_PLUGIN_ROOT" not in text


def test_doctor_accepts_the_runner_and_flags_a_pinned_plist(tmp_path):
    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    runner = tmp_path / "brain" / "bin" / "run"
    runner.parent.mkdir(parents=True)
    runner.touch()
    pinned = tmp_path / "cache" / "biss-brain" / "0.7.2" / "scripts"
    pinned.mkdir(parents=True)
    (pinned / "_run.sh").touch()
    (pinned / "dmn.py").touch()
    (agents / "ch.bissbert.biss-brain.dream.plist").write_text(json.dumps([str(runner), "dream.py"]))
    (agents / "ch.bissbert.biss-brain.dmn.plist").write_text(
        json.dumps([str(pinned / "_run.sh"), str(pinned / "dmn.py")]))

    def run(cmd):
        if cmd[0] == "launchctl":
            return "-\t0\tch.bissbert.biss-brain.dream\n-\t0\tch.bissbert.biss-brain.dmn\n"
        return Path(cmd[-1]).read_text()

    out = [w for w, _ in D.launchd_checks(pinned.parent, agents=agents, run=run)]
    assert out == ["ch.bissbert.biss-brain.dmn is pinned to one plugin copy and stops after an update"]
    runner.unlink()
    out = [w for w, _ in D.launchd_checks(pinned.parent, agents=agents, run=run)]
    assert any("runner" in w and "missing" in w for w in out)


def test_commands_find_a_copy_from_any_marketplace():
    for cmd in (REPO / "commands").glob("*.md"):
        text = cmd.read_text()
        assert "biss-claude-plugins" not in text, cmd.name
        if "CLAUDE_PLUGIN_ROOT:-" in text:
            assert "cache/*/biss-brain/*/" in text and "sort -V -k1,1" in text, cmd.name
