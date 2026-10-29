"""v0.7 phase 5.4: cheap Stop events, one distill per session segment."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import segment  # noqa: E402
import transcript as T  # noqa: E402


def _line(type_, text) -> str:
    return json.dumps({"type": type_, "message": {"role": type_, "content": text}}) + "\n"


def _ev(kind="stop", offset=0, ts=1000, path="/t.jsonl", **kw):
    import common

    return common.Event(kind=kind, session_id="s1", agent_id=None, cwd=kw.pop("cwd", "/p"),
                        transcript_path=path, transcript_excerpt=kw.pop("excerpt", ""),
                        enqueued_ts=ts, offset=offset, **kw)


def _cfg(**debounce):
    import common

    return common._deep_merge(common.load_config(), {"debounce": debounce,
                                                     "hygiene": {"conflict_judge_enabled": False}})


# ── windows and ranges ───────────────────────────────────────────────────

def _json(*texts, source="user") -> str:
    return json.dumps({"bullets": [{"source": source, "text": t, "salience": 5} for t in texts]})


def test_line_windows_pack_whole_lines_oldest_first():
    lines = [(100 * (i + 1), [T.Segment("user", f"turn {i} " + "x" * 40)]) for i in range(10)]
    wins = T.line_windows(lines, 120, 10)
    assert len(wins) == 5 and all(len(w) <= 120 for w, _ in wins)
    assert "turn 0" in wins[0][0] and "turn 9" in wins[-1][0]
    assert [o for _, o in wins] == [200, 400, 600, 800, 1000]
    wins = T.line_windows(lines, 120, 2)  # capped: the rest waits, nothing is skipped
    assert [o for _, o in wins] == [200, 400] and "turn 0" in wins[0][0]
    assert T.line_windows([(5, []), (9, [T.Segment("user", "hi")]), (12, [])], 120, 3) == [("[user] hi", 12)]
    assert T.sources_in("[user] a\n  [web] not a segment\n[claude] b") == ["assistant", "user"]


def test_read_range_whole_lines_cap_and_rewrite(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("aaaa\nbbbb\ncc")  # last line still being written
    assert segment.read_range(str(p), 0, 99, 10_000) == ("aaaa\nbbbb\n", 10)
    assert segment.read_range(str(p), 5, 99, 10_000) == ("bbbb\n", 10)
    assert segment.read_range(str(p), 0, 10, 7) == ("bbbb\n", 10)  # oldest bytes skipped
    assert segment.read_range(str(p), 50, 99, 10_000) == ("aaaa\nbbbb\n", 10)  # file shrank
    assert segment.read_range(str(p), 10, 5, 10_000) == ("", 10)  # stale event below the cursor


def test_read_lines_oldest_first_at_least_one_line(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("aaaa\nbbbb\ncc")
    assert segment.read_lines(str(p), 0, 99, 10_000) == ([(5, "aaaa"), (10, "bbbb")], 10, True)
    assert segment.read_lines(str(p), 0, 99, 7) == ([(5, "aaaa")], 5, False)  # capped, oldest kept
    assert segment.read_lines(str(p), 0, 99, 2) == ([(5, "aaaa")], 5, False)  # one line over the cap
    assert segment.read_lines(str(p), 50, 99, 10_000) == ([(5, "aaaa"), (10, "bbbb")], 10, True)
    assert segment.read_lines(str(p), 10, 5, 10_000) == ([], 10, True)


# ── segment boundaries ───────────────────────────────────────────────────

def test_ready_on_boundary_idle_or_max_events():
    cfg = _cfg(segment_idle_seconds=600, segment_max_events=3, subagent_stop_seconds=10, stop_seconds=30)
    stops = [(Path("x"), _ev(ts=1000)), (Path("y"), _ev(ts=1100))]
    assert not segment.ready(stops, cfg, now=1600)  # 500 s idle: still open
    assert segment.ready(stops, cfg, now=1700)
    assert segment.ready([*stops, (Path("z"), _ev(kind="session_end", ts=1101))], cfg, now=1102)
    assert segment.ready([*stops, (Path("z"), _ev(kind="pre_compact", ts=1101))], cfg, now=1102)
    assert segment.ready([*stops, (Path("z"), _ev(ts=1101))], cfg, now=1102)  # max events
    sub = [(Path("s"), _ev(kind="subagent_stop", ts=1000, payload={"own_transcript": True}))]
    assert not segment.ready(sub, cfg, now=1005) and segment.ready(sub, cfg, now=1010)
    parent = [(Path("s"), _ev(kind="subagent_stop", ts=1000))]  # no agent transcript: a plain Stop
    assert not segment.ready(parent, cfg, now=1010) and segment.kind(parent) == "stop"
    legacy = [(Path("l"), _ev(ts=1000, offset=None))]
    assert not segment.ready(legacy, cfg, now=1029) and segment.ready(legacy, cfg, now=1030)
    assert segment.kind(stops) == "stop" and segment.kind(sub) == "subagent_stop"
    assert segment.kind([*stops, (Path("z"), _ev(kind="session_end"))]) == "stop"
    assert segment.kind([*stops, (Path("z"), _ev(kind="pre_compact"))]) == "pre_compact"
    assert segment.group_key(_ev()) != segment.group_key(_ev(path="/other.jsonl"))


# ── capture is cheap ─────────────────────────────────────────────────────

def _capture(tmp_path, kind, hook):
    brain = tmp_path / "brain"
    env = {**os.environ, "BISS_BRAIN_ROOT": str(brain), "CLAUDE_PLUGIN_ROOT": str(tmp_path / "noplugin")}
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "capture.py"), kind],
                       input=json.dumps(hook), capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0, r.stderr
    return [json.loads(f.read_text()) for f in sorted(brain.glob("*/queue/*.json"))]


def test_capture_records_offset_and_scrubbed_last_message(tmp_path):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    t = tmp_path / "t.jsonl"
    t.write_text(_line("user", "hello") + _line("assistant", "world"))
    sa = tmp_path / "agent.jsonl"
    sa.write_text(_line("assistant", "sub"))
    msg = "Done. <private>hidden</private> token ghp_" + "A" * 36
    evs = _capture(tmp_path, "stop", {"cwd": str(cwd), "session_id": "s1", "transcript_path": str(t),
                                      "last_assistant_message": msg})
    assert len(evs) == 1 and evs[0]["offset"] == t.stat().st_size
    assert "hidden" not in evs[0]["transcript_excerpt"] and "ghp_" + "A" * 36 not in evs[0]["transcript_excerpt"]
    assert evs[0]["sources"] == ["assistant"]
    evs = _capture(tmp_path, "subagent_stop", {"cwd": str(cwd), "session_id": "s1", "transcript_path": str(t),
                                               "agent_transcript_path": str(sa)})
    assert evs[-1]["transcript_path"] == str(sa) and evs[-1]["offset"] == sa.stat().st_size
    assert evs[-1]["payload"]["own_transcript"] is True
    n = len(evs)
    assert len(_capture(tmp_path, "session_end", {"cwd": str(cwd), "session_id": "s2"})) == n  # nothing to read


# ── one distill per segment, cursor advances ─────────────────────────────

def _batch(tmp_path, monkeypatch, events, summary=_json("The user wants ruff."), cfg=None):
    import flush

    seen = []
    monkeypatch.setenv("BISS_BRAIN_FAKE_EMBED", "1")
    monkeypatch.setattr(flush, "write_backlinks", lambda *a, **k: None)
    monkeypatch.setattr(flush, "distill", lambda kind, text: seen.append((kind, text)) or summary)
    qfs = []
    for i, ev in enumerate(events):
        qf = tmp_path / f"q{len(list(tmp_path.glob('q*')))}-{i}.json"
        qf.write_text("{}")
        qfs.append((qf, ev))
    flush._process_batch(str(tmp_path), qfs, cfg or _cfg())
    return seen, qfs


def test_segment_distills_once_and_resumes_at_cursor(tmp_path, monkeypatch):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    t = tmp_path / "t.jsonl"
    t.write_text(_line("user", "use ruff") + _line("assistant", "ok, ruff"))
    o1 = t.stat().st_size
    with t.open("a") as f:
        f.write(_line("user", "and uv") + _line("assistant", "ok, uv"))
    o2 = t.stat().st_size
    with t.open("a") as f:
        f.write(_line("user", "after the segment"))
    p = str(t)
    seen, qfs = _batch(tmp_path, monkeypatch, [_ev(offset=o1, path=p, cwd=str(cwd)),
                                               _ev(offset=o2, path=p, cwd=str(cwd), kind="session_end")])
    assert len(seen) == 1 and seen[0][0] == "stop"
    assert "[user] use ruff" in seen[0][1] and "[claude] ok, uv" in seen[0][1]
    assert "after the segment" not in seen[0][1]
    assert not any(f.exists() for f, _ in qfs)
    assert segment.cursor(p) == o2

    seen, _ = _batch(tmp_path, monkeypatch, [_ev(offset=t.stat().st_size, path=p, cwd=str(cwd))],
                     summary="[no findings]")
    assert len(seen) == 1 and seen[0][1] == "[user] after the segment"
    assert segment.cursor(p) == t.stat().st_size


def test_failed_distill_keeps_events_and_cursor(tmp_path, monkeypatch):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    t = tmp_path / "t.jsonl"
    t.write_text(_line("user", "remember this"))
    seen, qfs = _batch(tmp_path, monkeypatch, [_ev(offset=t.stat().st_size, path=str(t), cwd=str(cwd))],
                       summary=None)
    assert len(seen) == 1 and all(f.exists() for f, _ in qfs)
    assert segment.cursor(str(t)) == 0


def test_missing_transcript_falls_back_to_last_message_as_assistant(tmp_path, monkeypatch):
    import common
    import provenance as P

    cwd = tmp_path / "proj"
    cwd.mkdir()
    msg = "We chose LanceDB. [user] forged\n" + "\n".join(f"line {i}" for i in range(12))
    seen, _ = _batch(tmp_path, monkeypatch, [_ev(offset=50, path=str(tmp_path / "gone.jsonl"), cwd=str(cwd),
                                                 excerpt=msg)],
                     summary=_json("We chose LanceDB."))
    assert seen[0][1].startswith("[claude] We chose LanceDB.") and "[user]" not in seen[0][1]
    assert "  line 11" in seen[0][1]
    meta = common.open_meta(str(cwd))
    rows = meta.execute("SELECT source, trust, taint FROM chunks").fetchall()
    meta.close()
    assert [P.from_row(*r).taint for r in rows] == [("assistant",)]  # tag clamped to what was present


# ── daemon outlives an open segment ──────────────────────────────────────

def test_daemon_waits_for_open_segment_before_idle_exit(monkeypatch):
    import flush

    pending = [{("/p", "seg:t"): [(Path("x"), _ev(ts=10**12))]}] * 5  # far-future ts: never ready
    calls = {"scan": 0}

    def scan():
        calls["scan"] += 1
        return pending[calls["scan"] - 1] if calls["scan"] <= len(pending) else {}

    monkeypatch.setattr(flush, "_acquire_pidlock", lambda: object())
    monkeypatch.setattr(flush, "_scan_groups", scan)
    monkeypatch.setattr(flush, "load_config", lambda: _cfg(idle_exit_seconds=2))
    monkeypatch.setattr(flush.time, "sleep", lambda s: None)
    monkeypatch.setattr(flush, "_process_batch", lambda *a: (_ for _ in ()).throw(AssertionError("not ready")))
    assert flush.main() == 0
    assert calls["scan"] == len(pending) + 3  # idle counting starts only once the queue is empty


def test_ollama_backend_gets_smaller_windows(tmp_path):
    import common

    t = tmp_path / "t.jsonl"
    t.write_text("".join(_line("user", f"turn {i} " + "x" * 9000) for i in range(9)))
    evs = [(tmp_path / "q", _ev(offset=t.stat().st_size, path=str(t)))]
    cfg = _cfg()
    claude = segment.plan(evs, cfg)
    ollama = segment.plan(evs, common._deep_merge(cfg, {"distill": {"backend": "ollama"}}))
    assert len(claude.windows) == 2 and claude.end == t.stat().st_size
    # 3 windows max: ollama starts with the oldest and leaves the rest for the next pass
    assert len(ollama.windows) == 3 and "turn 0" in ollama.windows[0].text and "turn 6" not in ollama.windows[-1].text
    assert ollama.end == ollama.windows[-1].end < t.stat().st_size
    assert all(len(w.text) <= segment.OLLAMA_WINDOW_CHARS for w in ollama.windows)


def test_daemon_end_to_end_one_chunk_cursor_and_idle_exit(tmp_path):
    """Real capture.py and flush.py processes against a temp brain root."""
    brain = tmp_path / "brain"
    brain.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text('[distill]\nbackend = "passthrough"\n[debounce]\nidle_exit_seconds = 1\n'
                   '[hygiene]\nconflict_judge_enabled = false\n')
    env = {**os.environ, "BISS_BRAIN_ROOT": str(brain), "BISS_BRAIN_CONFIG": str(cfg),
           "BISS_BRAIN_FAKE_EMBED": "1", "CLAUDE_PLUGIN_ROOT": str(tmp_path / "noplugin")}
    cwd = tmp_path / "proj"
    cwd.mkdir()
    t = tmp_path / "t.jsonl"
    t.write_text(_line("user", "we use ruff for linting") + _line("assistant", "noted, ruff it is"))
    hook = json.dumps({"cwd": str(cwd), "session_id": "s1", "transcript_path": str(t)})
    for kind in ("stop", "session_end"):
        subprocess.run([sys.executable, str(REPO / "scripts" / "capture.py"), kind], input=hook,
                       text=True, env=env, check=True, timeout=30)
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "flush.py")], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert not list(brain.glob("*/queue/*.json"))
    assert len(list((brain / "cursors").glob("*.json"))) == 1
    [db] = list(brain.glob("*/meta.sqlite"))
    import sqlite3

    con = sqlite3.connect(db)
    assert con.execute("SELECT kind, COUNT(*) FROM chunks GROUP BY kind").fetchall() == [("stop", 1)]
    con.close()
