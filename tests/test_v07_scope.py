"""v0.7 phase 6: launcher and scope (excluded dirs, opt-out files, the shell check)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import common  # noqa: E402
import scope  # noqa: E402

RUN = REPO / "scripts" / "_run.sh"


def _cfg(exclude=(), include_only=()):
    return {"scope": {"exclude": list(exclude), "include_only": list(include_only)}}


def test_defaults_exclude_home_exactly_and_the_listed_trees(tmp_path):
    cfg = common.DEFAULT_CONFIG
    home = Path.home()
    assert not scope.in_scope(str(home), cfg) and not scope.in_scope("/", cfg)
    assert not scope.in_scope(str(home / "Downloads" / "x"), cfg)
    assert not scope.in_scope(str(home / ".claude" / "brain"), cfg)
    assert not scope.in_scope("/tmp/foo", cfg)
    assert scope.in_scope(str(REPO), cfg)  # under ~ but not ~ itself


def test_include_only_and_prefix_is_a_path_not_a_string(tmp_path):
    a, ab = tmp_path / "a", tmp_path / "ab"
    for d in (a / "sub", ab):
        d.mkdir(parents=True)
    cfg = _cfg(exclude=[str(a)])
    assert not scope.in_scope(str(a / "sub"), cfg) and scope.in_scope(str(ab), cfg)
    cfg = _cfg(include_only=[str(ab)])
    assert scope.in_scope(str(ab), cfg) and not scope.in_scope(str(a), cfg)


def test_opt_out_files_apply_to_subdirectories(tmp_path):
    proj = tmp_path / "p"
    (proj / "deep").mkdir(parents=True)
    assert scope.in_scope(str(proj / "deep"), _cfg())
    (proj / ".biss-brain-ignore").touch()
    assert not scope.in_scope(str(proj / "deep"), _cfg())
    (proj / ".biss-brain-ignore").unlink()
    (proj / ".claude").mkdir()
    (proj / ".claude" / "brain.toml").write_text("enabled = true\n")
    assert scope.in_scope(str(proj / "deep"), _cfg())
    (proj / ".claude" / "brain.toml").write_text("enabled = false\n")
    assert not scope.in_scope(str(proj / "deep"), _cfg())


def test_cache_is_written_once_per_rule_change(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "BRAIN_ROOT", tmp_path)
    scope.refresh_cache(_cfg(exclude=["/tmp"]))
    f = tmp_path / scope.CACHE
    first = f.stat().st_mtime_ns
    scope.refresh_cache(_cfg(exclude=["/tmp"]))
    assert f.stat().st_mtime_ns == first
    assert f.read_text().splitlines()[0] == "e /"


def _fake_root(tmp_path):
    """A plugin root whose python only records that it started."""
    root = tmp_path / "plugin"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / "scripts").mkdir()
    marker = tmp_path / "started"
    py = root / ".venv" / "bin" / "python"
    py.write_text(f"#!/bin/sh\necho \"$@\" >> {marker}\n")
    py.chmod(0o755)
    (root / ".venv" / ".biss-ready").touch()
    return root, marker


def _run(root, brain, cwd, script):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"],
           "CLAUDE_PLUGIN_ROOT": str(root), "BISS_BRAIN_ROOT": str(brain)}
    return subprocess.run(["bash", str(RUN), str(root / "scripts" / script), "stop"], cwd=cwd,
                          env=env, input="{}", text=True, capture_output=True, timeout=10)


def test_shell_check_stops_hooks_before_python(tmp_path, monkeypatch):
    root, marker = _fake_root(tmp_path)
    brain = tmp_path / "brain"
    excluded, kept, ignored = tmp_path / "ex", tmp_path / "keep", tmp_path / "ign" / "sub"
    for d in (excluded / "sub", kept, ignored):
        d.mkdir(parents=True)
    (tmp_path / "ign" / ".biss-brain-ignore").touch()
    monkeypatch.setattr(common, "BRAIN_ROOT", brain)
    scope.refresh_cache(_cfg(exclude=[str(excluded)]))

    for cwd in (excluded / "sub", ignored):
        assert _run(root, brain, cwd, "capture.py").returncode == 0
    assert not marker.exists()
    assert _run(root, brain, kept, "capture.py").returncode == 0
    assert _run(root, brain, excluded, "dream.py").returncode == 0  # only hook scripts are checked
    assert len(marker.read_text().splitlines()) == 2


def test_shell_check_without_cache_leaves_it_to_python(tmp_path):
    root, marker = _fake_root(tmp_path)
    cwd = tmp_path / "x"
    cwd.mkdir()
    assert _run(root, tmp_path / "nobrain", cwd, "inject_context.py").returncode == 0
    assert marker.exists()


def test_shell_check_is_fast(tmp_path, monkeypatch):
    root, marker = _fake_root(tmp_path)
    brain = tmp_path / "brain"
    ex = tmp_path / "ex"
    ex.mkdir()
    monkeypatch.setattr(common, "BRAIN_ROOT", brain)
    scope.refresh_cache(_cfg(exclude=[str(ex)]))
    best = min(_timed(root, brain, ex) for _ in range(5))
    assert not marker.exists() and best < 0.2  # ~10 ms on an idle machine; loose for CI load


def _timed(root, brain, cwd):
    t = time.perf_counter()
    _run(root, brain, cwd, "capture.py")
    return time.perf_counter() - t


def test_hook_commands_use_the_plugin_root_and_guard_an_empty_one():
    hooks = json.loads((REPO / "hooks" / "hooks.json").read_text())["hooks"]
    cmds = [h["command"] for groups in hooks.values() for g in groups for h in g["hooks"]]
    assert cmds and all("plugins/cache" not in c for c in cmds)
    for c in cmds:
        assert c.startswith('[ -x "${CLAUDE_PLUGIN_ROOT}/scripts/_run.sh" ] || exit 0; exec ')
    # An unset root makes the guard exit 0 instead of running /scripts/_run.sh.
    r = subprocess.run(["sh", "-c", cmds[0]], env={"PATH": os.environ["PATH"]}, input="{}",
                       text=True, capture_output=True, timeout=10)
    assert r.returncode == 0 and r.stderr == ""
    mcp = json.loads((REPO / ".mcp.json").read_text())["mcpServers"]["biss-brain"]
    assert mcp["command"] == "${CLAUDE_PLUGIN_ROOT}/scripts/_run.sh"


def test_capture_hook_skips_an_excluded_dir_in_python(tmp_path):
    brain, cfg = tmp_path / "brain", tmp_path / "config.toml"
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".biss-brain-ignore").touch()
    cfg.write_text("[distill]\nbackend = \"passthrough\"\n")
    t = tmp_path / "t.jsonl"
    t.write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')
    env = {**os.environ, "BISS_BRAIN_ROOT": str(brain), "BISS_BRAIN_CONFIG": str(cfg),
           "CLAUDE_PLUGIN_ROOT": str(tmp_path / "x")}
    hook = json.dumps({"cwd": str(proj), "session_id": "s", "transcript_path": str(t)})
    subprocess.run([sys.executable, str(REPO / "scripts" / "capture.py"), "stop"], input=hook,
                   text=True, env=env, check=True, timeout=30)
    assert not list(brain.glob("*/queue/*.json"))
    assert (brain / scope.CACHE).is_file()
