"""v0.7 phase 5.7a: strict distill replies, per-window progress, retry with backoff."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

import pending  # noqa: E402
import segment  # noqa: E402
import transcript as T  # noqa: E402
from test_v07_segment import _batch, _cfg, _ev, _json, _line  # noqa: E402


def _chunks(cwd) -> list[str]:
    import common

    meta = common.open_meta(str(cwd))
    try:
        return [r[0] for r in meta.execute("SELECT kind FROM chunks")]
    finally:
        meta.close()


def test_strict_reply_rejects_prose_but_keeps_salvage():
    assert T.split_output("Sure! Here are the facts: ruff.", ["user"], strict=True) is None
    assert T.split_output('{"facts": []}', ["user"], strict=True) is None
    assert T.split_output("[no findings]", ["user"], strict=True) == ({"trusted": [], "low": []}, {})
    groups, _ = T.split_output('{"bullets": [{"source": "user", "text": "Use ruff.", "salience": 7}, {"sou',
                               ["user"], strict=True)
    assert groups["trusted"] == [("user", "- Use ruff.")]
    # passthrough (not strict) still stores its plain text
    assert T.split_output("- [user] Use ruff.", ["user"])[0]["trusted"] == [("user", "- Use ruff.")]


def test_garbage_reply_goes_to_retry_not_no_findings(tmp_path, monkeypatch):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    t = tmp_path / "t.jsonl"
    t.write_text(_line("user", "remember ruff"))
    ev = _ev(offset=t.stat().st_size, path=str(t), cwd=str(cwd))
    seen, qfs = _batch(tmp_path, monkeypatch, [ev], summary="I could not find anything useful.")
    assert len(seen) == 1 and all(f.exists() for f, _ in qfs)
    assert segment.cursor(str(t)) == 0
    st = pending.state(tmp_path, segment.group_key(ev))
    assert st["attempts"] == 1 and st["error"] == "invalid distill reply"
    assert "ruff" not in json.dumps(st)  # no transcript text in the retry state
    assert pending.waiting(tmp_path, segment.group_key(ev))

    # The retry succeeds: events finished, state cleared.
    _, qfs = _batch(tmp_path, monkeypatch, [ev])
    assert not any(f.exists() for f, _ in qfs) and pending.state(tmp_path, segment.group_key(ev)) is None


def test_failed_second_window_keeps_the_first(tmp_path, monkeypatch):
    import common
    import flush

    cwd = tmp_path / "proj"
    cwd.mkdir()
    t = tmp_path / "t.jsonl"
    t.write_text(_line("user", "first decision: ruff " + "x" * 40))
    o1 = t.stat().st_size
    with t.open("a") as f:
        f.write(_line("user", "second decision: uv " + "y" * 40))
    o2 = t.stat().st_size
    cfg = common._deep_merge(_cfg(), {"capture": {"segment_window_chars": 80}})
    replies = iter([_json("The user wants ruff."), None, _json("The user wants uv.")])
    monkeypatch.setenv("BISS_BRAIN_FAKE_EMBED", "1")
    monkeypatch.setattr(flush, "write_backlinks", lambda *a, **k: None)
    monkeypatch.setattr(flush, "distill", lambda kind, text: next(replies))
    qfs = []
    for i, off in enumerate((o1, o2)):
        qf = tmp_path / f"e{i}.json"
        qf.write_text("{}")
        qfs.append((qf, _ev(offset=off, path=str(t), cwd=str(cwd), ts=1000 + i)))

    flush._process_batch(str(tmp_path), qfs, cfg)
    assert segment.cursor(str(t)) == o1 and len(_chunks(cwd)) == 1
    assert not qfs[0][0].exists() and qfs[1][0].exists()  # the first event is finished

    flush._process_batch(str(tmp_path), qfs[1:], cfg)  # the retry reads only window 2
    assert segment.cursor(str(t)) == o2 and len(_chunks(cwd)) == 2 and not qfs[1][0].exists()


def test_backoff_doubles_up_to_the_max_and_never_gives_up(tmp_path):
    pdir = tmp_path / "proj"
    cfg = {"distill": {"retry_base_seconds": 60, "retry_max_seconds": 300}}
    delays = [pending.fail(pdir, "seg:x", "stop", "distill failed", cfg, now=1000)["next_try"] - 1000
              for _ in range(6)]
    assert delays == [60, 120, 240, 300, 300, 300]
    assert pending.waiting(pdir, "seg:x", now=1299) and not pending.waiting(pdir, "seg:x", now=1300)
    f = next((pdir / "pending").glob("*.json"))
    assert oct(f.stat().st_mode & 0o777) == "0o600"
    [b] = pending.backlog(tmp_path)
    assert b["attempts"] == 6 and b["group"] == "seg:x" and b["project_dir"] == str(pdir)
    pending.clear(pdir, "seg:x")
    assert pending.state(pdir, "seg:x") is None and not pending.waiting(pdir, "seg:x")


def test_pass_ends_at_the_first_boundary(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text(_line("user", "before compaction"))
    o1 = t.stat().st_size
    with t.open("a") as f:
        f.write(_line("user", "after compaction"))
    o2 = t.stat().st_size
    evs = [(tmp_path / "a", _ev(offset=o1, path=str(t), ts=1000)),
           (tmp_path / "b", _ev(kind="pre_compact", offset=o1, path=str(t), ts=1001)),
           (tmp_path / "c", _ev(offset=o2, path=str(t), ts=1002))]
    p = segment.plan(evs, _cfg())
    assert p.kind == "pre_compact" and [f.name for f, _ in p.events] == ["a", "b"]
    assert p.end == o1 and "after compaction" not in "".join(w.text for w in p.windows)
    # Same second: the boundary file sorts before the stop, which still belongs to the pass.
    p = segment.plan([evs[1], evs[0], evs[2]], _cfg())
    assert {f.name for f, _ in p.events} == {"a", "b"} and p.kind == "pre_compact"


def test_pass_reads_oldest_first_and_skips_nothing(tmp_path):
    import common

    t = tmp_path / "t.jsonl"
    t.write_text("".join(_line("user", f"turn {i} " + "x" * 200) for i in range(10)))
    ev = (tmp_path / "q", _ev(offset=t.stat().st_size, path=str(t)))
    cfg = common._deep_merge(_cfg(), {"capture": {"raw_tail_bytes": 1000}})
    p = segment.plan([ev], cfg)
    assert "turn 0" in p.windows[0].text and "turn 9" not in "".join(w.text for w in p.windows)
    assert p.end < t.stat().st_size and p.commit(p.end) == []  # event not finished yet
    seen = {i for i in range(10) for w in p.windows if f"turn {i} " in w.text}
    while segment.cursor(str(t)) < t.stat().st_size:
        p = segment.plan([ev], cfg)
        seen |= {i for i in range(10) for w in p.windows if f"turn {i} " in w.text}
        p.commit(p.end)
    assert seen == set(range(10))


def test_daemon_idles_out_past_a_backing_off_segment(tmp_path, monkeypatch):
    import flush

    pdir = tmp_path / "p"
    pending.fail(pdir, "seg:t", "stop", "distill failed", _cfg(), now=None)
    groups = {(str(pdir), "seg:t"): [(Path("x"), _ev(ts=1))]}  # ready, but backing off
    monkeypatch.setattr(flush, "_acquire_pidlock", lambda: object())
    monkeypatch.setattr(flush, "_scan_groups", lambda: groups)
    monkeypatch.setattr(flush, "load_config", lambda: _cfg(idle_exit_seconds=2))
    monkeypatch.setattr(flush.time, "sleep", lambda s: None)
    monkeypatch.setattr(flush, "_process_batch", lambda *a: (_ for _ in ()).throw(AssertionError("waiting")))
    assert flush.main() == 0


# ── no pass can spin: every pass progresses or backs off ────────────────

def test_partial_last_line_and_truncated_transcript_finish_the_events(tmp_path, monkeypatch):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    t = tmp_path / "t.jsonl"
    full = _line("user", "use ruff")
    t.write_text(full + '{"type":"user","mes')  # a line still being written
    _, qfs = _batch(tmp_path, monkeypatch, [_ev(offset=t.stat().st_size, path=str(t), cwd=str(cwd))])
    assert not any(f.exists() for f, _ in qfs) and segment.cursor(str(t)) == len(full)

    s = tmp_path / "s.jsonl"
    s.write_text(full)
    _, qfs = _batch(tmp_path, monkeypatch, [_ev(offset=10_000, path=str(s), cwd=str(cwd))])  # rewritten shorter
    assert not any(f.exists() for f, _ in qfs) and segment.cursor(str(s)) == len(full)


def test_invalid_reply_for_the_same_window_is_skipped_and_recorded(tmp_path, monkeypatch):
    import common

    cwd = tmp_path / "proj"
    cwd.mkdir()
    t = tmp_path / "t.jsonl"
    t.write_text(_line("user", "secret plan: ruff"))
    ev = _ev(offset=t.stat().st_size, path=str(t), cwd=str(cwd))
    cfg = common._deep_merge(_cfg(), {"distill": {"skip_invalid_after": 2}})
    _, qfs = _batch(tmp_path, monkeypatch, [ev], summary="As instructed, prose only.", cfg=cfg)
    assert all(f.exists() for f, _ in qfs) and not list((tmp_path / "pending").glob("skipped.jsonl"))
    _, qfs = _batch(tmp_path, monkeypatch, [ev], summary="As instructed, prose only.", cfg=cfg)
    assert not any(f.exists() for f, _ in qfs) and segment.cursor(str(t)) == t.stat().st_size
    assert pending.state(tmp_path, segment.group_key(ev)) is None
    [rec] = [json.loads(ln) for ln in (tmp_path / "pending" / "skipped.jsonl").read_text().splitlines()]
    assert oct((tmp_path / "pending" / "skipped.jsonl").stat().st_mode & 0o777) == "0o600"
    assert (rec["path"], rec["start"], rec["end"]) == (str(t), 0, t.stat().st_size)
    assert "ruff" not in json.dumps(rec)
    # A transient failure in between resets the count: never skipped.
    assert pending.fail(tmp_path, "g", "stop", "distill failed", cfg, at=0)["stuck"] == 1
    assert pending.fail(tmp_path, "g", "stop", "invalid distill reply", cfg, at=0)["stuck"] == 1
    assert pending.fail(tmp_path, "g", "stop", "invalid distill reply", cfg, at=5)["stuck"] == 1


def test_a_raising_pass_backs_off(tmp_path, monkeypatch):
    import flush

    pdir = tmp_path / "p"
    groups = {(str(pdir), "seg:t"): [(Path("x"), _ev(ts=1))]}
    calls = []
    monkeypatch.setattr(flush, "_acquire_pidlock", lambda: object())
    monkeypatch.setattr(flush, "_scan_groups", lambda: groups)
    monkeypatch.setattr(flush, "load_config", lambda: _cfg(idle_exit_seconds=2))
    monkeypatch.setattr(flush.time, "sleep", lambda s: None)
    monkeypatch.setattr(flush, "_process_batch", lambda *a: calls.append(1) or 1 / 0)
    assert flush.main() == 0 and calls == [1]
    assert pending.state(pdir, "seg:t")["error"] == "error: ZeroDivisionError"


def test_daemon_with_a_failing_backend_keeps_the_segment_and_exits(tmp_path):
    """Real capture.py and flush.py: the pending key matches what the daemon scans."""
    import os
    import subprocess

    brain = tmp_path / "brain"
    brain.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text('[distill]\nbackend = "ollama"\n[embed]\nendpoint = "http://127.0.0.1:9"\n'
                   '[debounce]\nidle_exit_seconds = 1\n[hygiene]\nconflict_judge_enabled = false\n')
    env = {**os.environ, "BISS_BRAIN_ROOT": str(brain), "BISS_BRAIN_CONFIG": str(cfg),
           "BISS_BRAIN_FAKE_EMBED": "1", "CLAUDE_PLUGIN_ROOT": str(tmp_path / "noplugin")}
    cwd = tmp_path / "proj"
    cwd.mkdir()
    t = tmp_path / "t.jsonl"
    t.write_text(_line("user", "we use ruff") + _line("assistant", "noted"))
    hook = json.dumps({"cwd": str(cwd), "session_id": "s1", "transcript_path": str(t)})
    subprocess.run([sys.executable, str(REPO / "scripts" / "capture.py"), "session_end"], input=hook,
                   text=True, env=env, check=True, timeout=30)
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "flush.py")], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert len(list(brain.glob("*/queue/*.json"))) == 1
    [st] = [json.loads(f.read_text()) for f in brain.glob("*/pending/*.json")]
    assert st["attempts"] == 1 and st["error"] == "distill failed"
    assert all(json.loads(f.read_text())["offset"] == 0 for f in (brain / "cursors").glob("*.json"))


# ── 5.7b: distill fallback chain ─────────────────────────────────────────

def test_fallback_chain_order_and_refusals(monkeypatch):
    import common
    import distill

    base = common.load_config()
    cfg = common._deep_merge(base, {"distill": {"fallback": ["passthrough", "ollama", "claude_code", "gpt"]}})
    assert distill.chain(cfg) == ["claude_code", "ollama"]
    assert distill.chain(base) == ["claude_code"]
    assert segment.width(base) == base["capture"]["segment_window_chars"]
    assert segment.width(cfg) == segment.OLLAMA_WINDOW_CHARS  # windows fit every backend in the chain


def test_distill_falls_back_only_when_the_primary_returns_nothing(monkeypatch):
    import common
    import distill

    cfg = common._deep_merge(common.load_config(), {"distill": {"fallback": ["ollama"]}})
    monkeypatch.setattr(distill, "load_config", lambda: cfg)
    tried = []
    replies = {"claude_code": None, "ollama": '{"bullets": []}'}

    def fake(prompt, model="haiku", timeout=120, json_out=False, backend=None):
        tried.append((backend, model, json_out))
        return replies[backend]

    monkeypatch.setattr(distill, "call", fake)
    assert distill.distill("stop", "[user] hi") == '{"bullets": []}'
    assert tried == [("claude_code", "haiku", True), ("ollama", "haiku", True)]
    tried.clear()
    replies["claude_code"] = '{"bullets": [{"source": "user", "text": "x"}]}'
    assert distill.distill("stop", "[user] hi").startswith('{"bullets": [{')
    assert [b for b, _, _ in tried] == ["claude_code"]
    tried.clear()
    replies.update(claude_code=None, ollama=None)
    assert distill.distill("stop", "[user] hi") is None and len(tried) == 2
