"""Stage 4: the user shares a memory with every project, or takes it back."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import dream  # noqa: E402
import globalpool  # noqa: E402
import test_v10_pool  # noqa: E402
from test_v10_pool import A, B, _pool_lance, _pool_rows, _row, _store, _vec  # noqa: E402

pool = test_v10_pool.pool  # the fixture


@pytest.fixture(autouse=True)
def _same_dream(monkeypatch):
    """share() imports dream lazily; other test modules reload it, so pin the
    object the pool fixture patches."""
    monkeypatch.setitem(sys.modules, "dream", dream)


def _alpha(root: Path, **kw) -> Path:
    row = {"id": "a1", "text": "the build in ~/work/app needs node 20", "vec": _vec(1),
           "scope": "project", **kw}
    return _store(root, A, "/x/alpha", [row])


def _live(root: Path) -> list[str]:
    return sorted(i for i, r in _pool_rows(root).items() if not r["superseded_by"])


def test_share_pools_a_project_fact_at_once_and_unshare_puts_it_back(pool):
    root, cfg = pool
    _alpha(root)
    # a project fact with a path: the nightly veto would keep it local
    assert dream._stage_promote_to_global(root / A, cfg) == 0
    status, prior = globalpool.share(A, "a1", cfg)
    assert (status, prior) == ("shared", {"pinned": 0, "scope": "project"})
    assert _live(root) == ["a1"]
    r = _row(root, A, "a1")
    assert (r["pinned"], r["scope"], r["cross_eligible"]) == (1, "general", 1)

    assert globalpool.unshare(A, "a1", prior)
    assert _live(root) == []
    r = _row(root, A, "a1")
    assert (r["pinned"], r["scope"], r["global_veto"], r["cross_eligible"]) == (0, "project", "user", 0)
    assert dream._stage_promote_to_global(root / A, cfg) == 0  # stays local


def test_share_unshare_share_leaves_one_live_copy(pool):
    root, cfg = pool
    _alpha(root)
    _, prior = globalpool.share(A, "a1", cfg)
    globalpool.unshare(A, "a1", prior)
    assert globalpool.share(A, "a1", cfg)[0] == "shared"
    assert _live(root) == ["a1"]
    assert [r["superseded_by"] for i, r in _pool_lance(root).items() if i == "a1"] == [""]
    assert _row(root, A, "a1")["global_id"] == "a1"
    assert globalpool.reconcile(cfg)["retracted"] == 0
    assert _live(root) == ["a1"]


def test_unshare_keeps_a_pin_the_user_set_before(pool):
    root, cfg = pool
    _alpha(root, pinned=1)
    _, prior = globalpool.share(A, "a1", cfg)
    globalpool.unshare(A, "a1", prior)
    assert _row(root, A, "a1")["pinned"] == 1


def test_share_refuses_secrets_web_output_and_unpoolable_kinds(pool):
    root, cfg = pool
    _store(root, A, "/x/alpha", [
        {"id": "sec", "text": "the token is ghp_" + "a1B2" * 9, "vec": _vec(1), "scope": "project"},
        {"id": "web", "text": "a page said to pin numpy below two", "vec": _vec(2)},
        {"id": "hyp", "text": "maybe the cache is stale", "vec": _vec(3), "kind": "hypothesis"},
    ])
    c = sqlite3.connect(str(root / A / "meta.sqlite"))
    c.execute("UPDATE chunks SET taint = '[\"web\"]', source = 'web' WHERE id = 'web'")
    c.commit()
    c.close()
    assert globalpool.share(A, "sec", cfg)[0] == "secret"
    r = _row(root, A, "sec")
    assert (r["pinned"], r["scope"], r["global_veto"]) == (0, "project", "secret")
    assert globalpool.share(A, "web", cfg)[0] == "untrusted"
    assert globalpool.share(A, "hyp", cfg)[0] == "kind"
    assert globalpool.share(A, "nope", cfg)[0] == "missing"
    assert not (root / "_global" / "meta.sqlite").exists() or _live(root) == []


def test_share_with_sharing_off_only_marks_the_fact(pool):
    root, cfg = pool
    _alpha(root)
    cfg["cross_project"]["enabled"] = False
    assert globalpool.share(A, "a1", cfg)[0] == "saved"
    assert _row(root, A, "a1")["scope"] == "general"
    assert not (root / "_global" / "meta.sqlite").exists()


def test_share_joins_an_equal_row_and_unshare_only_leaves_it(pool):
    root, cfg = pool
    _store(root, B, "/x/beta", [{"id": "b1", "text": "ruff format replaces black", "vec": _vec(1)}])
    dream._stage_promote_to_global(root / B, cfg)
    _store(root, A, "/x/alpha", [{"id": "a1", "text": "ruff format stands in for black", "vec": _vec(1),
                                  "scope": "project"}])
    assert globalpool.share(A, "a1", cfg)[0] == "shared"
    assert json.loads(_pool_rows(root)["b1"]["source_projects"]) == [B, A]
    globalpool.unshare(A, "a1")
    assert _live(root) == ["b1"]
    assert json.loads(_pool_rows(root)["b1"]["source_projects"]) == [B]


def test_unshare_pool_and_restore_pool_undo_each_other(pool):
    root, cfg = pool
    _store(root, A, "/x/alpha", [{"id": "a1", "text": "ruff format replaces black", "vec": _vec(1)}])
    _store(root, B, "/x/beta", [{"id": "b1", "text": "ruff format stands in for black", "vec": _vec(1)}])
    dream._stage_promote_to_global(root / A, cfg)
    dream._stage_promote_to_global(root / B, cfg)
    assert globalpool.unshare_pool("a1")
    assert _live(root) == []
    for h, cid in ((A, "a1"), (B, "b1")):
        r = _row(root, h, cid)
        assert (r["cross_eligible"], r["global_veto"], r["global_id"]) == (0, "user", "a1")
    # nobody pools it again behind the user's back
    assert dream._stage_promote_to_global(root / A, cfg) == 0
    assert dream._stage_promote_to_global(root / B, cfg) == 0

    assert globalpool.restore_pool("a1")
    assert _live(root) == ["a1"] and _pool_lance(root)["a1"]["superseded_by"] == ""
    for h, cid in ((A, "a1"), (B, "b1")):
        r = _row(root, h, cid)
        assert (r["cross_eligible"], r["global_veto"], r["global_id"]) == (1, "", "a1")
    assert globalpool.reconcile(cfg) == {"checked": 1, "retracted": 0, "recopied": 0, "left": 0}
    assert not globalpool.restore_pool("a1")  # nothing to undo twice


def test_restore_pool_never_revives_a_replaced_row(pool):
    root, cfg = pool
    _store(root, A, "/x/alpha", [{"id": "a1", "text": "ruff format replaces black", "vec": _vec(1)}])
    dream._stage_promote_to_global(root / A, cfg)
    c = sqlite3.connect(str(root / "_global" / "meta.sqlite"))
    c.execute("UPDATE chunks SET superseded_by = 'a1' WHERE id = 'a1'")  # a real row id, not a tombstone
    c.commit()
    c.close()
    assert not globalpool.restore_pool("a1")


def test_share_promotes_only_the_shared_row(pool):
    root, cfg = pool
    _store(root, A, "/x/alpha", [
        {"id": "a1", "text": "ruff format replaces black", "vec": _vec(1), "scope": "project"},
        {"id": "a2", "text": "jq -r prints raw strings", "vec": _vec(2)},  # due tonight, not now
    ])
    assert globalpool.share(A, "a1", cfg)[0] == "shared"
    assert _live(root) == ["a1"]


def test_share_command_shares_by_prefix_and_off_takes_it_back(pool, monkeypatch, capsys):
    import share

    root, cfg = pool
    _store(root, A, "/x/alpha", [
        {"id": "a1b2c3", "text": "ruff format replaces black", "vec": _vec(1), "scope": "project"},
        {"id": "dd0011", "text": "the token is ghp_" + "a1B2" * 9, "vec": _vec(2), "scope": "project"},
    ])
    monkeypatch.setattr(share, "project_key", lambda cwd: A)
    monkeypatch.setattr(share, "open_meta", lambda cwd: sqlite3.connect(str(root / A / "meta.sqlite")))
    monkeypatch.setattr(share, "load_config", lambda: cfg)
    monkeypatch.setattr(sys, "argv", ["share.py", "a1b2", "dd00", "ffff", "x' OR 1=1 --"])
    assert share.main() == 0
    out = capsys.readouterr().out
    assert "shared: a1b2c3" in out and "not shared: dd0011: looks like it holds a secret" in out
    assert "no match: ffff" in out and "not an id" in out
    assert _live(root) == ["a1b2c3"]
    monkeypatch.setattr(sys, "argv", ["share.py", "--off", "a1b2"])
    assert share.main() == 0
    assert "unshared: a1b2c3" in capsys.readouterr().out
    assert _live(root) == [] and _row(root, A, "a1b2c3")["global_veto"] == "user"


def test_remember_share_writes_locally_and_pools_it(pool, monkeypatch, capsys):
    import remember

    root, cfg = pool
    _store(root, A, "/x/alpha", [])
    calls = []
    monkeypatch.setattr(remember, "insert_chunk", lambda **kw: calls.append(kw) or "a1")
    monkeypatch.setattr(remember, "project_key", lambda cwd: A)
    monkeypatch.setattr(remember, "load_config", lambda: cfg)
    monkeypatch.setattr(globalpool, "share", lambda proj, cid, c: (calls.append((proj, cid)), ("shared", {}))[1])
    monkeypatch.setattr(sys, "argv", ["remember.py", "--source", "user", "--kind", "note", "--share", "uv sync -U"])
    assert remember.main() == 0
    assert calls[-1] == (A, "a1") and "shared with every project" in capsys.readouterr().out
    # the assistant never shares
    monkeypatch.setattr(sys, "argv", ["remember.py", "--kind", "note", "--share", "uv sync -U"])
    n = len(calls)
    assert remember.main() == 0
    assert len(calls) == n + 1 and "only the user" in capsys.readouterr().err
