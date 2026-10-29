"""v0.7 phase 3: temporal validity — valid_from / invalid_at, as_of and history reads."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import temporal as T  # noqa: E402

DAY = 86400
BASE = 1_750_000_000


def _vec(i: int, dim: int = 8) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def _store(tmp_path, rows: list[dict], proj: str = "p"):
    """A Lance table + v0.7 meta db. rows: id, day, text, vec, and optional
    superseded_by / invalid_day / valid_day / kind / ts."""
    import lancedb

    import common

    pdir = tmp_path / "proj"
    pdir.mkdir(exist_ok=True)
    meta = sqlite3.connect(pdir / "meta.sqlite")
    meta.row_factory = sqlite3.Row
    common.ensure_schema(meta)
    lance_rows = []
    for r in rows:
        ts = r.get("ts", BASE + r["day"] * DAY)
        sup = r.get("superseded_by", "")
        lance_rows.append({
            "id": r["id"], "project": proj, "session_id": "", "agent_id": "", "kind": r.get("kind", "stop"),
            "ts": ts, "text": r["text"], "content_hash": r["id"], "superseded_by": sup,
            "vector": r.get("vec", _vec(0)),
        })
        meta.execute(
            "INSERT INTO chunks (id, project, kind, ts, content_hash, superseded_by, valid_from, invalid_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (r["id"], proj, r.get("kind", "stop"), ts, r["id"], sup,
             BASE + r["valid_day"] * DAY if "valid_day" in r else None,
             BASE + r["invalid_day"] * DAY if "invalid_day" in r else None),
        )
    meta.commit()
    tbl = lancedb.connect(str(pdir / "lancedb")).create_table("chunks", data=lance_rows, mode="overwrite")
    return pdir, tbl, meta


# PG15 (day 1) → PG16 (day 40) → PG17 (day 80); a tombstoned note; an unrelated live row.
CHAIN = [
    {"id": "pg15", "day": 1, "text": "database is PostgreSQL 15", "superseded_by": "pg16", "invalid_day": 40},
    {"id": "pg16", "day": 40, "text": "database is PostgreSQL 16", "superseded_by": "pg17", "invalid_day": 80},
    {"id": "pg17", "day": 80, "text": "database is PostgreSQL 17"},
    {"id": "gone", "day": 5, "text": "database password note", "superseded_by": "__forgotten__"},
    {"id": "other", "day": 2, "text": "ci runs on gitea actions", "vec": _vec(3)},
]


# ── migration ────────────────────────────────────────────────────────────

def test_migration_backfills_contradictions_not_tombstones(tmp_path):
    import common

    conn = sqlite3.connect(tmp_path / "old.sqlite")
    conn.execute(
        "CREATE TABLE chunks (id TEXT PRIMARY KEY, project TEXT, session_id TEXT, agent_id TEXT, kind TEXT, "
        "ts INTEGER, content_hash TEXT, superseded_by TEXT DEFAULT '')"
    )
    conn.executemany(
        "INSERT INTO chunks (id, project, kind, ts, content_hash, superseded_by) VALUES (?, ?, 'stop', ?, ?, ?)",
        [("old", "p", 100, "h1", "new"), ("new", "p", 500, "h2", ""),
         ("tomb", "p", 200, "h3", "deleted-by-ui"),
         ("xproj", "p", 300, "h4", "elsewhere"), ("elsewhere", "q", 900, "h5", "")],
    )
    conn.commit()
    common.ensure_schema(conn)
    got = dict(conn.execute("SELECT id, invalid_at FROM chunks"))
    assert got["old"] == 500
    assert got["tomb"] is None and got["new"] is None
    assert got["xproj"] is None  # a successor in another project is not a contradiction
    common.ensure_schema(conn)  # idempotent
    assert dict(conn.execute("SELECT id, invalid_at FROM chunks"))["old"] == 500


# ── temporal.py ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("s,want", [
    (None, None), ("now", None), ("", None), ("1700000000", 1700000000), (1700000000, 1700000000),
    ("30d", 2_000_000_000 - 30 * DAY), ("12h", 2_000_000_000 - 12 * 3600), ("2w", 2_000_000_000 - 14 * DAY),
])
def test_parse_as_of(s, want):
    assert T.parse_as_of(s, now=2_000_000_000) == want


def test_parse_as_of_dates_and_errors():
    from datetime import datetime

    end = T.parse_as_of("2026-03-01")
    assert end == int(datetime(2026, 3, 1).timestamp()) + 86399  # end of that local day
    assert T.parse_as_of("2026-03-01T12:00:00") == int(datetime(2026, 3, 1, 12).timestamp())
    for bad in ("yesterday", "3x", "2026-13-40", "1; DROP TABLE"):
        with pytest.raises(ValueError):
            T.parse_as_of(bad)


def test_validity_head_and_predecessors(tmp_path):
    _, _, meta = _store(tmp_path, CHAIN)
    v = T.validity(meta, "p", ["pg15", "pg17", "gone"])
    assert not v["pg15"].forgotten and v["pg15"].valid_at(BASE + 20 * DAY)
    assert not v["pg15"].valid_at(BASE + 40 * DAY)  # invalid_at is exclusive
    assert v["pg17"].live and v["pg17"].valid_at(BASE + 90 * DAY)
    assert v["gone"].forgotten and not v["gone"].valid_at(BASE + 10 * DAY)

    assert T.head_of(meta, "p", "pg15") == "pg17"
    assert T.head_of(meta, "p", "gone") is None
    assert T.head_of(meta, "p", "pg15") == T.head_of(meta, "p", "pg16")
    assert [p[0] for p in T.predecessors(meta, "p", "pg17")] == ["pg16", "pg15"]
    assert T.predecessors(meta, "p", "other") == []


def test_head_of_survives_a_cycle(tmp_path):
    _, _, meta = _store(tmp_path, [
        {"id": "a", "day": 1, "text": "a", "superseded_by": "b", "invalid_day": 2},
        {"id": "b", "day": 2, "text": "b", "superseded_by": "a", "invalid_day": 3},
    ])
    assert T.head_of(meta, "p", "a") is None


def test_invalidate_uses_new_event_time_never_before_old(tmp_path):
    _, _, meta = _store(tmp_path, [
        {"id": "old", "day": 10, "text": "x"}, {"id": "new", "day": 20, "text": "y"},
        {"id": "late", "day": 30, "text": "z"}, {"id": "backdated", "day": 31, "valid_day": 5, "text": "w"},
    ])
    assert T.invalidate(meta, "p", "old", "new") == BASE + 20 * DAY
    # a successor whose event time predates the old one never yields a negative interval
    assert T.invalidate(meta, "p", "late", "backdated") == BASE + 30 * DAY
    assert T.invalidate(meta, "p", "missing", "new") is None
    assert meta.execute("SELECT superseded_by FROM chunks WHERE id='old'").fetchone()[0] == "new"


# ── retrieve ─────────────────────────────────────────────────────────────

def _retrieve(tbl, meta, **kw):
    import common
    import retrieve

    cfg = common._deep_merge(common.load_config(), {"retrieve": {"hybrid": False, "rerank": False}})
    return retrieve.retrieve(tbl, meta, "p", "postgres", _vec(0), cfg, top_k=5, **kw)


def test_retrieve_now_hides_contradicted_and_forgotten(tmp_path):
    _, tbl, meta = _store(tmp_path, CHAIN)
    ids = [r["id"] for r in _retrieve(tbl, meta)]
    assert "pg17" in ids and not {"pg15", "pg16", "gone"} & set(ids)


@pytest.mark.parametrize("day,want", [(20, "pg15"), (40, "pg16"), (79, "pg16"), (100, "pg17")])
def test_retrieve_as_of_returns_what_was_true(tmp_path, day, want):
    _, tbl, meta = _store(tmp_path, CHAIN)
    rows = _retrieve(tbl, meta, as_of=BASE + day * DAY)
    pg = [r["id"] for r in rows if r["id"].startswith("pg")]
    assert pg == [want]
    assert "gone" not in [r["id"] for r in rows]
    top = next(r for r in rows if r["id"] == want)
    assert top["_valid_from"] == BASE + int(want == "pg15" and 1 or want == "pg16" and 40 or 80) * DAY


def test_retrieve_as_of_before_anything_is_empty(tmp_path):
    _, tbl, meta = _store(tmp_path, CHAIN)
    assert _retrieve(tbl, meta, as_of=BASE) == []


def test_include_history_folds_onto_head(tmp_path):
    _, tbl, meta = _store(tmp_path, CHAIN)
    rows = _retrieve(tbl, meta, include_history=True)
    ids = [r["id"] for r in rows]
    assert ids.count("pg17") == 1 and not {"pg15", "pg16", "gone"} & set(ids)
    head = next(r for r in rows if r["id"] == "pg17")
    assert [h["id"] for h in head["_history"]] == ["pg16", "pg15"]
    assert head["_history"][1] == {"id": "pg15", "valid_from": BASE + DAY, "invalid_at": BASE + 40 * DAY,
                                   "text": "database is PostgreSQL 15"}
    assert next(r for r in rows if r["id"] == "other")["_history"] == []


def test_include_history_fetches_head_outside_pool(tmp_path):
    import retrieve

    _, tbl, meta = _store(tmp_path, CHAIN)
    ranked = [{"id": "pg15", "text": "database is PostgreSQL 15", "_score": 0.9, "_invalid_at": BASE + 40 * DAY}]
    out = retrieve.fold_history(tbl, meta, "p", ranked, None)
    assert [r["id"] for r in out] == ["pg17"] and out[0]["_score"] == 0.9


# ── write paths ──────────────────────────────────────────────────────────

def test_flush_contradicts_sets_invalid_at_and_extends_adds_edge(tmp_path, monkeypatch):
    import common
    import flush

    pdir, tbl, meta = _store(tmp_path, [
        {"id": "old", "day": 1, "text": "PostgreSQL 15"}, {"id": "new", "day": 40, "text": "PostgreSQL 16"},
        {"id": "base", "day": 2, "text": "uses uv", "vec": _vec(5)},
        {"id": "more", "day": 3, "text": "uses uv and ruff", "vec": _vec(5)},
    ])
    meta.close()

    def _open(cwd):
        c = sqlite3.connect(pdir / "meta.sqlite")
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr(flush, "lancedb_dir", lambda cwd: pdir / "lancedb")
    monkeypatch.setattr(flush, "open_meta", _open)
    cfg = common.load_config()

    monkeypatch.setattr(flush, "judge_conflict", lambda new, old, model=None: "CONTRADICTS")
    flush._run_conflict_judge(str(tmp_path), "p", "stop", "new", "PostgreSQL 16", _vec(0), cfg)
    monkeypatch.setattr(flush, "judge_conflict", lambda new, old, model=None: "EXTENDS")
    flush._run_conflict_judge(str(tmp_path), "p", "stop", "more", "uses uv and ruff", _vec(5), cfg)

    m = _open(None)
    assert tuple(m.execute("SELECT superseded_by, invalid_at FROM chunks WHERE id='old'").fetchone()) == (
        "new", BASE + 40 * DAY)
    assert m.execute("SELECT invalid_at FROM chunks WHERE id='base'").fetchone()[0] is None
    related = m.execute("SELECT related_ids FROM chunks WHERE id='more'").fetchone()[0]
    assert "base" in related and "extends" in related
    tbl.checkout_latest()
    lance = {r["id"]: r["superseded_by"] for r in tbl.search().limit(10).to_list()}
    assert lance["old"] == "new" and lance["base"] == ""


def test_flush_valid_from_is_event_time(tmp_path, monkeypatch):
    import common
    import flush

    monkeypatch.setenv("BISS_BRAIN_FAKE_EMBED", "1")
    monkeypatch.setattr(flush, "write_backlinks", lambda *a, **k: None)
    monkeypatch.setattr(flush, "distill", lambda kind, text: '{"bullets": [{"source": "user", "text": "Use uv, not pip.", "salience": 5}]}')
    cfg = common._deep_merge(common.load_config(), {"hygiene": {"conflict_judge_enabled": False}})
    cwd = tmp_path / "proj"
    cwd.mkdir()
    qf = tmp_path / "e.json"
    qf.write_text("{}")
    ev = common.Event(kind="stop", session_id="s1", agent_id=None, cwd=str(cwd), transcript_path="",
                      transcript_excerpt="x", sources=["user"])
    ev.enqueued_ts = 1_700_000_000
    flush._process_batch(str(tmp_path), [(qf, ev)], cfg)
    m = common.open_meta(str(cwd))
    vf, ts = m.execute("SELECT valid_from, ts FROM chunks WHERE kind='stop'").fetchone()
    assert vf == 1_700_000_000 and ts > vf


def test_dream_newer_memory_supersedes_whatever_order(tmp_path, monkeypatch):
    import common
    import dream
    import judge

    now = common.now_unix()
    # "first" is stored first but describes the older state (imported late).
    pdir, tbl, meta = _store(tmp_path, [
        {"id": "first", "day": 0, "ts": now - 100, "valid_day": 0, "text": "MSRV 1.74"},
        {"id": "second", "day": 0, "ts": now - 50, "text": "MSRV raised to 1.80"},
    ])
    meta.close()
    calls = []
    monkeypatch.setattr(judge, "judge_conflict",
                        lambda new, old, model=None: calls.append((new, old)) or "CONTRADICTS")
    cfg = common.load_config()
    out = dream._stage1b_conflicts(pdir, "p", cfg, dry_run=False)
    assert out["contradicts"] == 1
    assert calls == [("MSRV raised to 1.80", "MSRV 1.74")]
    m = sqlite3.connect(pdir / "meta.sqlite")
    assert m.execute("SELECT superseded_by FROM chunks WHERE id='first'").fetchone()[0] == "second"
    assert m.execute("SELECT superseded_by FROM chunks WHERE id='second'").fetchone()[0] == ""


# ── read surfaces ────────────────────────────────────────────────────────

def _server():
    spec = importlib.util.spec_from_file_location("brain_mcp_server_t", REPO / "mcp" / "server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    return server


def test_mcp_rejects_bad_as_of_before_touching_the_store(tmp_path, monkeypatch):
    server = _server()
    monkeypatch.setattr(server, "_table", lambda cwd: pytest.fail("store opened"))
    out = server.search_memory("q", as_of="last tuesday")
    assert len(out) == 1 and "unrecognised as_of" in out[0]["error"]


def test_inject_date_is_event_time():
    import inject_context as IC

    c = {"id": "a", "kind": "stop", "ts": 1_800_000_000, "_valid_from": 1_700_000_000, "text": "t"}
    assert f'date="{T.iso(1_700_000_000)}"' in IC._frame_chunks([c], "H")
    del c["_valid_from"]
    assert f'date="{T.iso(1_800_000_000)}"' in IC._frame_chunks([c], "H")
