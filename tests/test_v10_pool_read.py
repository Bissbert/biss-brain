"""Stage 3: reading the shared pool (recall, MCP, and the optional prompt inject)."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import dream  # noqa: E402
import inject_context  # noqa: E402
import retrieve  # noqa: E402
import test_v10_pool  # noqa: E402
from test_v10_pool import A, B, _store, _vec  # noqa: E402

pool = test_v10_pool.pool  # the fixture


def _promote(root: Path, cfg: dict) -> None:
    """Pool: a1 (ruff, from A), b1 (pytest, from B) and b2 (tainted by the web, from B)."""
    _store(root, A, "/x/alpha", [{"id": "a1", "text": "ruff format replaces black", "vec": _vec(1)}])
    pdir = _store(root, B, "/x/beta", [
        {"id": "b1", "text": "pytest -x stops at the first failure", "vec": _vec(2)},
        {"id": "b2", "text": "a page said to pin numpy below two", "vec": _vec(3)},
    ])
    c = sqlite3.connect(str(pdir / "meta.sqlite"))
    c.execute("UPDATE chunks SET taint = '[\"web\"]', source = 'web' WHERE id = 'b2'")
    c.commit()
    c.close()
    assert dream._stage_promote_to_global(root / A, cfg) == 1
    assert dream._stage_promote_to_global(root / B, cfg) == 1  # b2 is not trusted enough


def _near(i: int) -> list[float]:
    v = _vec(i)
    v[(i + 1) % len(v)] = 0.2  # cosine ≈ 0.98 with _vec(i)
    return v


def test_global_candidates_off_unless_enabled(pool):
    root, cfg = pool
    _promote(root, cfg)
    cfg["cross_project"]["enabled"] = False
    assert retrieve.global_candidates("", _vec(1), cfg, 3, min_cosine=0.5) == []
    cfg["cross_project"]["enabled"] = True
    assert retrieve.global_candidates("", None, cfg, 3, min_cosine=0.5) == []


def test_global_candidates_gate_label_and_rank_below_local(pool):
    root, cfg = pool
    _promote(root, cfg)
    rows = retrieve.global_candidates("", _near(1), cfg, 5, min_cosine=0.9)
    assert [r["text"] for r in rows] == ["ruff format replaces black"]
    r = rows[0]
    assert r["source_project"] == "/x/alpha" and r["_is_cross_project"] is True
    assert r["_relevance_score"] == pytest.approx(r["_cosine"] * 0.8)
    assert r["_source"] == "user"
    # the floor is absolute: an unrelated query gets nothing
    assert retrieve.global_candidates("", _vec(6), cfg, 5, min_cosine=0.5) == []


def test_web_facts_never_reach_the_pool(pool):
    root, cfg = pool
    _promote(root, cfg)
    assert retrieve.global_candidates("", _near(3), cfg, 5, min_cosine=0.5) == []


def test_global_candidates_for_inject_drop_tainted_rows_and_the_callers_own_copy(pool):
    root, cfg = pool
    _promote(root, cfg)
    assert len(retrieve.global_candidates("", _near(1), cfg, 5, min_cosine=0.9, for_inject=True)) == 1
    c = sqlite3.connect(str(root / "_global" / "meta.sqlite"))
    c.execute("UPDATE chunks SET taint = '[\"user\", \"web\"]'")
    c.commit()
    c.close()
    assert len(retrieve.global_candidates("", _near(1), cfg, 5, min_cosine=0.9)) == 1
    assert retrieve.global_candidates("", _near(1), cfg, 5, min_cosine=0.9, for_inject=True) == []
    # project B wrote b1: B already has it locally, A does not
    assert retrieve.global_candidates("", _near(2), cfg, 5, min_cosine=0.9, skip_project=B) == []
    assert len(retrieve.global_candidates("", _near(2), cfg, 5, min_cosine=0.9, skip_project=A)) == 1


def _inject_cfg(cfg: dict) -> dict:
    cfg["cross_project"]["inject"] = True
    cfg["retrieve"]["inject_top_k"] = 2
    return cfg


def test_pool_row_fills_one_free_slot_only_when_inject_is_on(pool):
    root, cfg = pool
    _promote(root, cfg)
    q = _near(1)
    rows, info = inject_context._pool_row(cfg, "how do I format", q, B, [], set())
    assert rows == [] and info == {}  # inject defaults to off
    _inject_cfg(cfg)
    rows, info = inject_context._pool_row(cfg, "how do I format", q, B, [], set())
    assert [r["id"] for r in rows] and info["global_n"] == 1 and "global_ms" in info
    # every slot taken by local memories: no pool row
    local = [{"id": "l1", "text": "x"}, {"id": "l2", "text": "y"}]
    rows, info = inject_context._pool_row(cfg, "how do I format", q, B, local, set())
    assert rows == [] and info["global_n"] == 0
    # already injected this session, or repeating a local memory: skipped
    gid = inject_context._pool_row(cfg, "q", q, B, [], set())[0][0]["id"]
    assert inject_context._pool_row(cfg, "q", q, B, [], {gid})[0] == []
    same = [{"id": "l1", "text": "Ruff format  replaces black"}]
    assert inject_context._pool_row(cfg, "q", q, B, same, set())[0] == []
    # below the stricter inject floor: nothing
    cfg["cross_project"]["inject_min_cosine"] = 0.999
    assert inject_context._pool_row(cfg, "q", q, B, [], set())[0] == []


def test_pool_row_respects_the_char_budget(pool):
    root, cfg = pool
    _promote(root, _inject_cfg(cfg))
    cfg["retrieve"]["inject_max_chars"] = 30
    local = [{"id": "l1", "text": "x" * 20}]
    assert inject_context._pool_row(cfg, "q", _near(1), B, local, set())[0] == []


def test_pool_row_is_framed_with_scope_and_origin(pool):
    root, cfg = pool
    _promote(root, _inject_cfg(cfg))
    row = inject_context._pool_row(cfg, "q", _near(1), B, [], set())[0][0]
    framed = inject_context._frame_chunks([row], "Relevant project memory (biss-brain)")
    assert 'scope="global" from="alpha"' in framed
    assert "ruff format replaces black" in framed
    local = inject_context._attrs({"kind": "stop", "ts": 0, "text": "t"})
    assert "scope=" not in local and "from=" not in local


def test_select_adds_the_pool_row_and_records_metrics(pool, monkeypatch, tmp_path):
    root, cfg = pool
    _promote(root, _inject_cfg(cfg))
    monkeypatch.setattr(inject_context, "_embed_with_timeout", lambda *a, **k: _near(1))
    monkeypatch.setattr(inject_context, "_table_or_none", lambda cwd: object())
    monkeypatch.setattr(inject_context, "prompt_inject", lambda *a, **k: [])
    monkeypatch.setattr(inject_context, "open_meta", lambda cwd: sqlite3.connect(":memory:"))
    monkeypatch.setattr(inject_context, "project_key", lambda cwd: B)
    rows, info = inject_context._select("/x/beta", cfg, "how do I format python", "s", tmp_path, 0.0)
    assert [r["text"] for r in rows] == ["ruff format replaces black"]
    assert info["outcome"] == "injected" and info["global_n"] == 1
    cfg["cross_project"]["inject"] = False
    rows, info = inject_context._select("/x/beta", cfg, "how do I format python", "s", tmp_path, 0.0)
    assert rows == [] and "global_n" not in info and info["outcome"] == "abstain"


def test_recall_goes_through_the_shared_helper(pool, monkeypatch):
    import recall

    root, cfg = pool
    _promote(root, cfg)
    monkeypatch.setattr(recall, "BRAIN_ROOT", root)  # recall binds the name at import
    rows = recall._recall_global(_near(2), cfg)
    assert [r["source_project"] for r in rows] == ["/x/beta"]


def test_doctor_counts_overdue_general_facts_and_orphaned_pool_copies(pool):
    import shutil
    import time

    import doctor

    root, cfg = pool
    _promote(root, cfg)
    assert doctor.pool_checks(cfg, root) == []  # b2 (web) is never due
    c = sqlite3.connect(str(root / A / "meta.sqlite"))
    for cid, ts in (("a-late", 1000), ("a-fresh", int(time.time()))):
        c.execute("INSERT INTO chunks (id, project, kind, ts, content_hash, source, trust, taint, scope) "
                  "VALUES (?,?,'stop',?,?,'user',0.9,'[\"user\"]','general')", (cid, A, ts, cid))
    c.commit()
    c.close()
    shutil.rmtree(root / B)
    warnings = [w for w, _ in doctor.pool_checks(cfg, root)]
    assert warnings == ["1 general fact(s) are overdue for the shared pool",
                        "the shared pool holds copies from 1 project store(s) that no longer exist"]
    cfg["cross_project"]["promote"] = "pinned_only"
    assert doctor.pool_checks(cfg, root) == []
