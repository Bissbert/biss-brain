"""v0.7 phase 2.3: taint propagates through merges, arcs, roll-ups and promotion."""
from __future__ import annotations

import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

import provenance as P  # noqa: E402


def _set_prov(conn: sqlite3.Connection, cid: str, prov: P.Provenance) -> None:
    conn.execute("UPDATE chunks SET source = ?, trust = ?, taint = ? WHERE id = ?", (*prov.row(), cid))
    conn.commit()


# ── trust class ──────────────────────────────────────────────────────────

def test_trust_class_and_sql_predicate_agree(tmp_path):
    import common

    conn = sqlite3.connect(tmp_path / "m.sqlite")
    common.ensure_schema(conn)
    provs = {
        "u": P.make("user"),
        "f": P.derive(["assistant", "file"]),
        "i": P.IMPORTED,
        "w": P.make("web"),
        "t": P.derive(["user", "tool"]),
    }
    for cid, prov in provs.items():
        conn.execute("INSERT INTO chunks (id, project, kind, ts, content_hash) VALUES (?, 'p', 'stop', 1, ?)", (cid, cid))
        _set_prov(conn, cid, prov)
    got = {r[0] for r in conn.execute(f"SELECT id FROM chunks WHERE {P.trusted_sql()}")}
    assert got == {cid for cid, p in provs.items() if P.trust_class(p) == "trusted"} == {"u", "f", "i"}
    assert P.trust_class(None) == "trusted"  # unknown ranks as imported; injection still fails closed


# ── reconsolidation ──────────────────────────────────────────────────────

def _labile(tmp_path, prov):
    import test_reconsolidate as R
    from common import open_meta

    cwd = str(tmp_path / "project")
    Path(cwd).mkdir()
    old_text = "The deploy script uses rsync to copy artifacts to /srv/app."
    old_vec = R._normalize(R._fake_embed(old_text))
    cid = R._insert_labile_chunk(cwd, old_text, old_vec)
    meta = open_meta(cwd)
    _set_prov(meta, cid, prov)
    meta.close()
    return R, cwd, cid, old_vec


def test_reconsolidate_skips_other_trust_class(tmp_path):
    import flush
    from common import open_meta

    R, cwd, cid, old_vec = _labile(tmp_path, P.make("web"))
    meta = open_meta(cwd)
    with patch("flush.ask") as ask:
        got = flush._try_reconsolidate(cwd, R._make_lancedb_table(cwd), meta, "deploy via rsync",
                                       R._make_close_vec(old_vec, 0.95), R._make_cfg(cwd), P.make("user"))
    meta.close()
    assert got is None and not ask.called


def test_reconsolidate_merge_combines_provenance(tmp_path):
    import flush
    from common import open_meta

    R, cwd, cid, old_vec = _labile(tmp_path, P.make("user"))
    merged = json.dumps({"merged_text": "The deploy script copies artifacts to /srv/app with rsync."})
    meta = open_meta(cwd)
    with patch("flush.ask", return_value=merged) as ask, \
            patch("flush._embed", return_value=R._make_close_vec(old_vec, 0.95)):
        got = flush._try_reconsolidate(cwd, R._make_lancedb_table(cwd), meta,
                                       "deploy via rsync, then restart nginx with systemctl",
                                       R._make_close_vec(old_vec, 0.95), R._make_cfg(cwd), P.make("file"))
    meta.close()
    assert got == cid and ask.called
    meta = open_meta(cwd)
    row = meta.execute("SELECT source, trust, taint FROM chunks WHERE id = ?", (cid,)).fetchone()
    meta.close()
    assert P.from_row(*row) == P.Provenance("distilled", 0.7, ("file", "user"))


def test_reconsolidate_folds_a_restatement_without_a_merge_call(tmp_path):
    import flush
    from common import open_meta

    R, cwd, cid, old_vec = _labile(tmp_path, P.make("user"))
    meta = open_meta(cwd)
    with patch("flush.ask") as ask:
        got = flush._try_reconsolidate(cwd, R._make_lancedb_table(cwd), meta, "deploy via rsync",
                                       R._make_close_vec(old_vec, 0.95), R._make_cfg(cwd), P.make("file"))
    meta.close()
    assert got == cid and not ask.called
    meta = open_meta(cwd)
    row = meta.execute("SELECT source, trust, taint FROM chunks WHERE id = ?", (cid,)).fetchone()
    meta.close()
    assert P.from_row(*row) == P.Provenance("distilled", 0.7, ("file", "user"))


def test_revises_counts_new_content_terms():
    import flush

    old = "The deploy script uses rsync to copy artifacts to /srv/app."
    assert not flush.revises(old, "Deploy uses rsync to copy the artifacts", 0.2)
    assert flush.revises(old, "Deploy uses rsync, then restarts nginx via systemctl", 0.2)
    assert flush.revises("we use ruff", "we do not use ruff", 0.2)  # a negation is new evidence
    assert not flush.revises(old, "the", 0.2)


# ── dream: arcs, dailies, promotion ──────────────────────────────────────

def _dream_env(tmp_path, monkeypatch, daily: bool = False):
    import test_dream_stage2 as D

    monkeypatch.setenv("BISS_BRAIN_FAKE_EMBED", "1")
    monkeypatch.setenv("BISS_BRAIN_FAKE_DISTILL", "1")
    monkeypatch.setenv("BISS_BRAIN_CONFIG", str(D._write_test_config(tmp_path, daily_enabled=daily)))
    monkeypatch.setenv("HOME", str(tmp_path))
    D._reset_modules()
    import common
    import dream

    cwd = tmp_path / "proj"
    cwd.mkdir()
    return D, common, dream, str(cwd)


def test_arc_excludes_low_trust_members_and_combines_provenance(tmp_path, monkeypatch):
    D, common, dream, cwd = _dream_env(tmp_path, monkeypatch)
    sid = uuid.uuid4().hex[:16]
    base = int(time.time()) - 7200
    provs = [P.make("user"), P.make("assistant"), P.make("web"), P.derive(["assistant", "tool"])]
    ids = [uuid.uuid4().hex for _ in provs]
    for i, cid in enumerate(ids):
        D._insert_chunk_direct(cwd, cid, f"chunk {i}", [0.1 + i * 0.01] * 768, session_id=sid, ts=base + i)
    meta = common.open_meta(cwd)
    for cid, prov in zip(ids, provs):
        _set_prov(meta, cid, prov)
    meta.close()

    cfg = common.load_config()
    assert dream._stage2a_arcs(common.project_dir(cwd), common.project_key(cwd), cfg, dry_run=False) == 1

    meta = common.open_meta(cwd)
    arc = meta.execute("SELECT id, evidence_ids, source, trust, taint FROM chunks WHERE kind = 'arc'").fetchone()
    parents = dict(meta.execute("SELECT id, parent_id FROM chunks WHERE kind != 'arc'").fetchall())
    meta.close()
    assert set(arc[1].split(",")) == set(ids[:2])
    assert P.from_row(*arc[2:]) == P.Provenance("distilled", 0.8, ("assistant", "user"))
    assert parents[ids[2]] == parents[ids[3]] == ""


def test_daily_rolls_up_trusted_arcs_only(tmp_path, monkeypatch):
    D, common, dream, cwd = _dream_env(tmp_path, monkeypatch, daily=True)
    yesterday = int(time.time()) - 86400
    arcs = {"a1": P.make("user"), "a2": P.derive(["file"]), "a3": P.make("web")}
    for i, cid in enumerate(arcs):
        D._insert_chunk_direct(cwd, cid, f"arc {i}", [0.2 + i * 0.01] * 768, kind="arc",
                               ts=yesterday + i, consolidation_round=1)
    meta = common.open_meta(cwd)
    for cid, prov in arcs.items():
        _set_prov(meta, cid, prov)
    meta.close()

    cfg = common.load_config()
    assert dream._stage2b_daily(common.project_dir(cwd), common.project_key(cwd), cfg, dry_run=False) == 1
    meta = common.open_meta(cwd)
    day = meta.execute("SELECT evidence_ids, source, trust, taint FROM chunks WHERE kind = 'daily_summary'").fetchone()
    meta.close()
    assert set(day[0].split(",")) == {"a1", "a2"}
    assert P.from_row(*day[1:]) == P.Provenance("distilled", 0.7, ("file", "user"))


def test_global_promotion_copies_provenance_and_skips_low_trust(tmp_path, monkeypatch):
    D, common, dream, cwd = _dream_env(tmp_path, monkeypatch)
    monkeypatch.setattr(dream, "BRAIN_ROOT", tmp_path / "brain")
    monkeypatch.setattr(dream, "_open_lance", lambda pdir: (None, None))
    cfg = common.load_config()
    kind = cfg["cross_project"]["admission"]["eligible_kinds"][0]
    rnd = cfg["cross_project"]["admission"]["min_consolidation_round"]
    for cid in ("good", "bad"):
        D._insert_chunk_direct(cwd, cid, cid, [0.3] * 768, kind=kind, consolidation_round=rnd)
    meta = common.open_meta(cwd)
    _set_prov(meta, "good", P.derive(["assistant", "file"]))
    _set_prov(meta, "bad", P.derive(["assistant", "web"]))
    meta.close()

    pdir, proj = common.project_dir(cwd), common.project_key(cwd)
    cands = dream._select_global_candidates(pdir, proj, cfg)
    assert [c["id"] for c in cands] == ["good"]
    assert dream._two_phase_promote(cands[0], proj, pdir, cfg, dry_run=False)
    g = sqlite3.connect(tmp_path / "brain" / "_global" / "meta.sqlite")
    row = g.execute("SELECT source, trust, taint FROM chunks WHERE id = 'good'").fetchone()
    g.close()
    assert P.from_row(*row) == P.derive(["assistant", "file"])
