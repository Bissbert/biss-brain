"""Stage 4: the dream suggests pool memories that several projects wrote for global core."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import core  # noqa: E402
import dream  # noqa: E402
import globalpool  # noqa: E402
import test_v10_pool  # noqa: E402
from common import GLOBAL_PROJECT_HASH  # noqa: E402
from test_v10_pool import A, B, _store, _vec  # noqa: E402

pool = test_v10_pool.pool  # the fixture


@pytest.fixture
def brain(pool, monkeypatch):
    """The pool fixture, with core's copy of common pointed at it too."""
    root, cfg = pool
    monkeypatch.setattr(core.common, "BRAIN_ROOT", root)
    return root, cfg


def _two_projects(root: Path, cfg: dict) -> None:
    """Pool: l1 (from A, joined by B) and l2 (from A only)."""
    _store(root, A, "/x/alpha", [
        {"id": "l1", "text": "ruff format replaces black", "vec": _vec(1), "kind": "lesson"},
        {"id": "l2", "text": "jq -r prints raw strings", "vec": _vec(2), "kind": "lesson"},
    ])
    _store(root, B, "/x/beta", [
        {"id": "m1", "text": "ruff format stands in for black", "vec": _vec(1), "kind": "lesson"},
    ])
    dream._stage_promote_to_global(root / A, cfg)
    dream._stage_promote_to_global(root / B, cfg)


def test_widely_written_needs_two_projects(brain):
    root, cfg = brain
    _two_projects(root, cfg)
    rows = globalpool.widely_written()
    assert [(r["id"], r["projects"]) for r in rows] == [("l1", {A, B})]
    assert {r["id"] for r in globalpool.widely_written(1)} == {"l1", "l2"}


def test_dream_proposes_a_lesson_two_projects_wrote_for_global_core(brain):
    root, cfg = brain
    _two_projects(root, cfg)
    assert dream._core_proposals_global(cfg, dry_run=True) == 1
    assert core.proposals([GLOBAL_PROJECT_HASH]) == []
    assert dream._core_proposals_global(cfg, dry_run=False) == 1
    got = [(r["chunk_id"], r["layer"], r["section"], r["origin"], r["text"])
           for r in core.proposals([GLOBAL_PROJECT_HASH])]
    assert got == [("l1", "global", "Lessons", "dream", "ruff format replaces black")]
    assert not core.global_file().exists()  # proposals only, core untouched
    assert dream._core_proposals_global(cfg, dry_run=False) == 0  # once


def test_accepting_it_writes_global_core_and_it_is_not_proposed_again(brain, tmp_path):
    root, cfg = brain
    _two_projects(root, cfg)
    dream._core_proposals_global(cfg, dry_run=False)
    pid = core.proposals([GLOBAL_PROJECT_HASH])[0]["id"]
    cwd = tmp_path / "anywhere"
    cwd.mkdir()
    assert "global" in core.accept(cwd, pid)
    assert [i.text for i in core.load(core.global_file()).items()] == ["ruff format replaces black"]
    c = sqlite3.connect(str(core.index_path()))
    c.execute("DELETE FROM core_proposals")  # a fresh queue: dedup by text against _global.md
    c.commit()
    c.close()
    assert dream._core_proposals_global(cfg, dry_run=False) == 0


def test_no_global_proposals_with_sharing_off_or_proposals_off(brain):
    root, cfg = brain
    _two_projects(root, cfg)
    cfg["core"]["proposals_per_dream"] = 0
    assert dream._core_proposals_global(cfg, dry_run=False) == 0
    cfg["core"]["proposals_per_dream"] = 5
    cfg["cross_project"]["enabled"] = False
    assert dream._core_proposals_global(cfg, dry_run=False) == 0


def test_the_private_pass_leaves_a_widely_written_memory_to_the_global_pass(brain):
    root, cfg = brain
    _two_projects(root, cfg)
    c = sqlite3.connect(str(root / A / "meta.sqlite"))
    c.execute("UPDATE chunks SET retrieval_count = 9")
    c.commit()
    c.close()
    assert dream._stage2c_core_proposals(root / A, A, cfg, dry_run=False) == 1
    assert [r["chunk_id"] for r in core.proposals([A])] == ["l2"]
    # with sharing off the pool is not consulted and both stay private candidates
    cfg["cross_project"]["enabled"] = False
    assert dream._stage2c_core_proposals(root / A, A, cfg, dry_run=False) == 1
    assert sorted(r["chunk_id"] for r in core.proposals([A])) == ["l1", "l2"]
