"""Phase 0 eval harness: metrics, grading, root guards, snapshot, synthetic smoke run."""
from __future__ import annotations

import json
import os
import pwd
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

import brain_snapshot  # noqa: E402
import eval_metrics as m  # noqa: E402

LIVE = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".claude" / "brain"


# ── grading ──────────────────────────────────────────────────────────────

def test_term_groups_and_within_or_across():
    q = {"answers": [["postgres", "16"], ["/pg ?16/"]]}
    assert m.is_relevant("We run Postgres 16 now", "", q)
    assert m.is_relevant("upgraded to PG16", "", q)
    assert not m.is_relevant("postgres 15 for now", "", q)


def test_gold_id_prefix_is_relevant_without_terms():
    q = {"answers": [["nothing matches"]], "gold_ids": ["abc123"]}
    assert m.is_relevant("unrelated", "abc123ffff", q)
    assert not m.is_relevant("unrelated", "abd", q)


def test_no_answer_question():
    assert not m.is_answerable({"answers": []})
    assert m.is_answerable({"answers": [], "gold_ids": ["x"]})
    assert m.relevance_vector([{"text": "a"}, {"text": "b"}], {"answers": []}) == [0, 0]


# ── metrics ──────────────────────────────────────────────────────────────

def test_rank_metrics():
    rel = [0, 1, 0, 1]
    assert m.hit_at_k(rel, 1) == 0.0
    assert m.hit_at_k(rel, 2) == 1.0
    assert m.reciprocal_rank(rel) == 0.5
    assert m.reciprocal_rank([0, 0]) == 0.0
    assert m.ndcg_at_k([1, 1, 0], 3) == pytest.approx(1.0)
    assert 0 < m.ndcg_at_k([0, 1], 2) < 1
    assert m.ndcg_at_k([0, 0], 2) == 0.0
    assert m.precision([1, 0]) == 0.5
    assert m.precision([]) is None


def test_bootstrap_is_deterministic_and_brackets_mean():
    xs = [0.0, 1.0, 0.5, 1.0, 0.0, 1.0]
    lo, hi = m.bootstrap_ci(xs)
    assert lo <= m.mean(xs) <= hi
    assert m.bootstrap_ci(xs) == (lo, hi)


def test_paired_bootstrap_direction():
    a = [0.0] * 20
    b = [1.0] * 20
    better = m.paired_bootstrap(a, b)
    assert better["delta"] == 1.0 and better["p_not_better"] == 0.0
    same = m.paired_bootstrap(a, a)
    assert same["delta"] == 0.0 and same["p_not_better"] == 1.0
    with pytest.raises(AssertionError):
        m.paired_bootstrap([1.0], [1.0, 2.0])


# ── guards ───────────────────────────────────────────────────────────────

def _run_eval_module():
    """Import run_eval without binding a root (guard_root needs no common import)."""
    import importlib

    return importlib.import_module("run_eval")


@pytest.mark.parametrize("p", [LIVE, LIVE / "abc123", LIVE / "x" / ".." / "abc"])
def test_eval_refuses_live_brain(p):
    run_eval = _run_eval_module()
    with pytest.raises(run_eval.RootGuardError):
        run_eval.guard_root(p)


def test_eval_guard_survives_faked_home(tmp_path, monkeypatch):
    run_eval = _run_eval_module()
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(run_eval.RootGuardError):
        run_eval.guard_root(LIVE)
    with pytest.raises(run_eval.RootGuardError):
        run_eval.guard_root(tmp_path / ".claude" / "brain" / "p")


def test_eval_accepts_snapshot_root(tmp_path):
    assert _run_eval_module().guard_root(tmp_path) == tmp_path.resolve()


def test_snapshot_guard_refuses_nesting(tmp_path):
    src = tmp_path / "brain"
    src.mkdir()
    with pytest.raises(brain_snapshot.SnapshotError):
        brain_snapshot.guard_paths(src, src / "snap")
    with pytest.raises(brain_snapshot.SnapshotError):
        brain_snapshot.guard_paths(src / "inner", src.parent)
    brain_snapshot.guard_paths(src, tmp_path / "elsewhere")


def test_snapshot_copies_without_touching_source(tmp_path):
    src = tmp_path / "brain"
    proj = src / "aaaaaaaaaaaa"
    (proj / "lancedb").mkdir(parents=True)
    (proj / "lancedb" / "marker").write_text("x")
    (proj / "project_path.txt").write_text("/work/a\n")
    db = sqlite3.connect(proj / "meta.sqlite")
    db.execute("CREATE TABLE chunks (id TEXT)")
    db.execute("INSERT INTO chunks VALUES ('c1')")
    db.commit()
    db.close()
    other = src / "bbbbbbbbbbbb"
    other.mkdir()
    (other / "meta.sqlite").write_bytes(b"")
    (other / "project_path.txt").write_text("/elsewhere\n")
    before = {p: p.stat().st_mtime_ns for p in src.rglob("*")}

    dest = tmp_path / "snap"
    man = brain_snapshot.snapshot(dest, src=src, prefixes=["/work"])

    assert [p["hash"] for p in man["projects"]] == ["aaaaaaaaaaaa"]
    out = sqlite3.connect(dest / "aaaaaaaaaaaa" / "meta.sqlite")
    assert out.execute("SELECT id FROM chunks").fetchall() == [("c1",)]
    out.close()
    assert (dest / "aaaaaaaaaaaa" / "lancedb" / "marker").read_text() == "x"
    assert not (dest / "bbbbbbbbbbbb").exists()
    assert json.loads((dest / "SNAPSHOT.json").read_text())["src"] == str(src)
    assert {p: p.stat().st_mtime_ns for p in src.rglob("*")} == before
    with pytest.raises(brain_snapshot.SnapshotError):
        brain_snapshot.snapshot(dest, src=src)  # non-empty destination


# ── fallback counter ─────────────────────────────────────────────────────

class _Q:
    def __init__(self, fail):
        self.fail = fail

    def text(self, _):
        if self.fail:
            raise RuntimeError("all columns in a record batch must have the same length")
        return self

    def vector(self, _):
        return self

    def distance_type(self, _):
        return self

    def where(self, _):
        return self

    def limit(self, _):
        return self

    def to_list(self):
        return [{"id": "dense"}]


class _Tbl:
    def search(self, *a, query_type=None):
        return _Q(fail=query_type == "hybrid")


def test_hybrid_failure_falls_back_and_is_counted():
    import common

    before = common.HYBRID_FALLBACKS
    assert [r["id"] for r in common.hybrid_search(_Tbl(), [0.0], "q", "project = 'p'", 3)] == ["dense"]
    assert common.HYBRID_FALLBACKS == before + 1


# ── end to end ───────────────────────────────────────────────────────────

def test_synthetic_suite_runs_with_fake_embeddings(tmp_path):
    out = tmp_path / "res.json"
    env = {k: v for k, v in os.environ.items() if k not in ("BISS_BRAIN_ROOT", "BISS_BRAIN_CONFIG")}
    p = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "eval" / "run_eval.py"), "--suite", "synthetic",
         "--embed", "fake", "--variants", "prod,no_decay", "--out", str(out)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    res = json.loads(out.read_text())
    s = res["summaries"]["prod"]
    assert s["n_answerable"] == 26 and s["n_no_answer"] == 11
    assert "poison_injected" in s and "poison_top1" in s
    assert s["errors"] == 0
    assert set(res["deltas"]) == {"no_decay"}
    assert res["coverage"] == {"answerable": 26, "in_corpus": 26, "missing": []}
    # results carry ids and numbers only, never memory text
    assert all("text" not in pq for r in res["runs"].values() for pq in r["per_question"].values())
