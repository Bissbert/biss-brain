"""v0.7 phase 5.7c: store without a vector when the embedder is down, backfill later."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

import pending  # noqa: E402
import segment  # noqa: E402
from test_v07_segment import _cfg, _ev, _json, _line  # noqa: E402


def _setup(tmp_path, monkeypatch, judge=True):
    import common
    import flush

    root = tmp_path / "brain"
    monkeypatch.setattr(common, "BRAIN_ROOT", root)
    monkeypatch.setattr(flush, "BRAIN_ROOT", root)
    monkeypatch.setenv("BISS_BRAIN_FAKE_EMBED", "1")
    calls = {"links": [], "judge": []}
    monkeypatch.setattr(flush, "write_backlinks", lambda cwd, proj, cid, vec, cfg: calls["links"].append(cid))
    monkeypatch.setattr(flush, "_run_conflict_judges",
                        lambda cwd, proj, kind, items, cfg: calls["judge"].append([i[0] for i in items]))
    cfg = common._deep_merge(_cfg(), {"hygiene": {"conflict_judge_enabled": judge}})
    cwd = tmp_path / "proj"
    cwd.mkdir()
    return common, flush, cfg, cwd, root, calls


def _down(*a, **k):
    raise ConnectionError("ollama down")


def _store_one(tmp_path, monkeypatch, flush, cfg, cwd, text="we lint with ruff"):
    t = tmp_path / f"t{len(list(tmp_path.glob('t*.jsonl')))}.jsonl"  # a new segment per call
    t.write_text(_line("user", text))
    monkeypatch.setattr(flush, "distill", lambda kind, x: _json(f"The user said: {text}."))
    qf = tmp_path / f"{t.stem}.json"
    qf.write_text("{}")
    ev = _ev(offset=t.stat().st_size, path=str(t), cwd=str(cwd))
    flush._process_batch(str(common_pdir(cwd)), [(qf, ev)], cfg)
    return qf, ev


def common_pdir(cwd):
    import common

    return common.project_dir(str(cwd))


def _rows(flush, cfg, cwd):
    tbl = flush._open_table(str(cwd), cfg["embed"]["dim"])
    return tbl.search().where("vector IS NULL").to_list(), tbl.count_rows()


def test_embed_down_stores_without_vector_and_finishes_the_segment(tmp_path, monkeypatch):
    common, flush, cfg, cwd, root, calls = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(flush, "_embed", _down)
    qf, ev = _store_one(tmp_path, monkeypatch, flush, cfg, cwd)
    assert not qf.exists() and pending.state(common_pdir(cwd), segment.group_key(ev)) is None
    nulls, total = _rows(flush, cfg, cwd)
    assert total == 1 and len(nulls) == 1 and "ruff" in nulls[0]["text"]
    meta = common.open_meta(str(cwd))
    assert meta.execute("SELECT salience FROM chunks").fetchall() == [(5,)]
    meta.close()
    assert calls == {"links": [], "judge": []}  # both wait for the vector
    assert pending.embed_marks(root) == [(common_pdir(cwd), str(cwd))]
    # Keyword search still finds it.
    tbl = flush._open_table(str(cwd), cfg["embed"]["dim"])
    assert [r["id"] for r in tbl.search("ruff", query_type="fts").limit(3).to_list()] == [nulls[0]["id"]]


def test_backfill_probes_once_while_down_then_embeds_links_and_judges(tmp_path, monkeypatch):
    import retrieve

    common, flush, cfg, cwd, root, calls = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(flush, "_embed", _down)
    _store_one(tmp_path, monkeypatch, flush, cfg, cwd)
    _store_one(tmp_path, monkeypatch, flush, cfg, cwd, text="and test with pytest")
    probes = []
    monkeypatch.setattr(flush, "_embed", lambda *a: probes.append(1) or _down())
    assert flush._backfill(cfg) == "down" and probes == [1, 1]  # the row, then the probe
    assert pending.embed_marks(root) and pending.embed_skips(common_pdir(cwd)) == []

    monkeypatch.setattr(flush, "_embed", lambda text, cfg: common.embed(text, cfg))
    nulls, _ = _rows(flush, cfg, cwd)
    assert len(nulls) == 2 and flush._backfill(cfg) == "done"
    assert _rows(flush, cfg, cwd)[0] == [] and pending.embed_marks(root) == []
    assert sorted(calls["links"]) == sorted(r["id"] for r in nulls)
    assert [sorted(j) for j in calls["judge"]] == [sorted(r["id"] for r in nulls)]
    assert flush._backfill(cfg) == "done"

    # The updated rows are still found by keyword and by the hybrid search.
    tbl = flush._open_table(str(cwd), cfg["embed"]["dim"])
    ruff = [r["id"] for r in nulls if "ruff" in r["text"]]
    assert [r["id"] for r in tbl.search("ruff", query_type="fts").limit(3).to_list()] == ruff
    qvec = common.embed("ruff", cfg, purpose="query")
    hits = retrieve.search(tbl, qvec, "ruff", "superseded_by = ''", 5, cfg)
    assert ruff[0] in [r["id"] for r in hits]


def test_backfill_skips_a_row_the_embedder_rejects(tmp_path, monkeypatch):
    common, flush, cfg, cwd, root, calls = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(flush, "_embed", _down)
    _store_one(tmp_path, monkeypatch, flush, cfg, cwd)
    _store_one(tmp_path, monkeypatch, flush, cfg, cwd, text="and test with pytest")

    def picky(text, cfg):
        if "ruff" in text:
            raise ValueError("input too long")
        return common.embed(text, cfg)

    monkeypatch.setattr(flush, "_embed", picky)
    assert flush._backfill(cfg) == "done"  # the embedder is up: not a reason to stop
    [bad] = _rows(flush, cfg, cwd)[0]
    assert "ruff" in bad["text"] and pending.embed_skips(common_pdir(cwd)) == [bad["id"]]
    assert pending.embed_marks(root) == [] and len(calls["links"]) == 1


def test_backfill_update_failure_is_retried_not_skipped(tmp_path, monkeypatch):
    import lancedb.table

    common, flush, cfg, cwd, root, calls = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(flush, "_embed", _down)
    _store_one(tmp_path, monkeypatch, flush, cfg, cwd)
    monkeypatch.setattr(flush, "_embed", lambda text, cfg: common.embed(text, cfg))
    real = lancedb.table.LanceTable.update
    fails = []

    def conflict(self, *a, **k):
        if not fails:
            fails.append(1)
            raise RuntimeError("commit conflict")
        return real(self, *a, **k)

    monkeypatch.setattr(lancedb.table.LanceTable, "update", conflict)
    assert flush._backfill(cfg) == "down" and pending.embed_skips(common_pdir(cwd)) == []
    assert flush._backfill(cfg) == "done" and _rows(flush, cfg, cwd)[0] == []


def test_backfill_drains_a_long_backlog_without_waiting(tmp_path, monkeypatch):
    common, flush, cfg, cwd, root, calls = _setup(tmp_path, monkeypatch, judge=False)
    monkeypatch.setattr(flush, "_embed", _down)
    for i in range(5):
        _store_one(tmp_path, monkeypatch, flush, cfg, cwd, text=f"fact number {i}")
    monkeypatch.setattr(flush, "BACKFILL_BATCH", 2)
    monkeypatch.setattr(flush, "BACKFILL_MAX", 3)
    monkeypatch.setattr(flush, "_embed", lambda text, cfg: common.embed(text, cfg))
    assert flush._backfill(cfg) == "more" and len(_rows(flush, cfg, cwd)[0]) == 2
    assert flush._backfill(cfg) == "done" and _rows(flush, cfg, cwd)[0] == []
    assert pending.embed_marks(root) == []


def test_real_embed_against_a_dead_endpoint_stores_null(tmp_path):
    import json
    import os
    import subprocess

    brain = tmp_path / "brain"
    cfg = tmp_path / "config.toml"
    cfg.write_text('[distill]\nbackend = "passthrough"\n[embed]\nendpoint = "http://127.0.0.1:9"\n'
                   '[debounce]\nidle_exit_seconds = 1\n[hygiene]\nconflict_judge_enabled = false\n')
    env = {k: v for k, v in os.environ.items() if k != "BISS_BRAIN_FAKE_EMBED"}
    env.update(BISS_BRAIN_ROOT=str(brain), BISS_BRAIN_CONFIG=str(cfg), CLAUDE_PLUGIN_ROOT=str(tmp_path / "x"))
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
    assert not list(brain.glob("*/queue/*.json")) and pending.backlog(brain) == []
    [(pdir, mark_cwd)] = pending.embed_marks(brain)
    assert mark_cwd == str(cwd)
    import lancedb

    tbl = lancedb.connect(str(pdir / "lancedb")).open_table("chunks")
    assert tbl.count_rows() >= 1 and tbl.count_rows("vector IS NULL") == tbl.count_rows()


def test_store_without_vector_off_retries_the_segment(tmp_path, monkeypatch):
    common, flush, cfg, cwd, root, calls = _setup(tmp_path, monkeypatch)
    cfg = common._deep_merge(cfg, {"embed": {"store_without_vector": False}})
    monkeypatch.setattr(flush, "_embed", _down)
    qf, ev = _store_one(tmp_path, monkeypatch, flush, cfg, cwd)
    assert qf.exists() and _rows(flush, cfg, cwd)[1] == 0
    assert pending.state(common_pdir(cwd), segment.group_key(ev))["error"] == "embed failed"


def test_dream_kicks_the_daemon_only_with_retries_pending(tmp_path, monkeypatch):
    import capture
    import dream

    root = tmp_path / "brain"
    root.mkdir()
    monkeypatch.setattr(dream, "BRAIN_ROOT", root)
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(REPO))
    kicks = []
    monkeypatch.setattr(capture, "_ensure_daemon", lambda r: kicks.append(r))
    dream._kick_flush()
    assert kicks == []
    pending.mark_embed(root / "p1", "/some/cwd")
    dream._kick_flush()
    assert kicks == [REPO]
    pending.clear_embed(root / "p1")
    pending.fail(root / "p2", "seg:x", "stop", "distill failed", _cfg())
    dream._kick_flush()
    assert len(kicks) == 2
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(tmp_path / "noplugin"))
    dream._kick_flush()
    assert len(kicks) == 2
