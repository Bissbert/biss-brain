"""Stage 0: fold stores whose project_path now keys to another project."""
from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common  # noqa: E402
import migrate_fold_stores as mfs  # noqa: E402

DIM = 8


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("BISS_BRAIN_CONFIG", str(tmp_path / "absent.toml"))
    root = tmp_path / "brain"
    root.mkdir()
    return home.resolve(), root


def _store(root: Path, path: Path, n: int, h: str | None = None) -> tuple[str, list[str]]:
    """A store for *path* under hash *h* (default: today's key) with n rows."""
    h = h or common.project_key(path)
    d = root / h
    d.mkdir()
    (d / "project_path.txt").write_text(str(path))
    ids = [str(uuid.uuid4()) for _ in range(n)]
    c = sqlite3.connect(str(d / "meta.sqlite"))
    common.ensure_schema(c)
    for i, cid in enumerate(ids):
        c.execute("INSERT INTO chunks (id, project, kind, ts, content_hash, source, taint)"
                  " VALUES (?,?,?,?,?,'user','[\"user\"]')", (cid, h, "stop", 1000 + i, f"h{cid}"))
    c.execute("INSERT INTO edge_bumps (project, chunk_a, chunk_b, bumped_at) VALUES (?,?,?,1)",
              (h, ids[0], ids[-1]))
    c.commit()
    c.close()
    tbl = common.open_or_create_chunks(d / "lancedb", DIM)
    tbl.add([{"id": cid, "project": h, "session_id": "s", "agent_id": "", "kind": "stop", "ts": 1000,
              "text": f"memory {cid}", "content_hash": f"h{cid}", "superseded_by": "",
              "vector": [float(i + 1)] * DIM} for i, cid in enumerate(ids)])
    (d / "queue").mkdir()
    (d / "queue" / f"{h}.json").write_text("{}")
    return h, ids


def _rows(root: Path, h: str) -> tuple[set, set]:
    import lancedb

    c = sqlite3.connect(str(root / h / "meta.sqlite"))
    meta = {tuple(r) for r in c.execute("SELECT id, project FROM chunks")}
    c.close()
    t = lancedb.connect(str(root / h / "lancedb")).open_table("chunks").to_arrow()
    return meta, set(zip(t.column("id").to_pylist(), t.column("project").to_pylist()))


def _layout(home: Path, root: Path):
    proj = home / "modpack"
    (proj / "pack" / "kubejs").mkdir(parents=True)
    (proj / "README.md").write_text("x")
    ws = home / "experiments"  # unmarked folder of projects: must attract nothing
    (ws / "lone").mkdir(parents=True)
    old_child = common.hashlib.sha256(str(proj / "pack" / "kubejs").encode()).hexdigest()[:12]
    parent, pids = _store(root, proj, 2)
    child, cids = _store(root, proj / "pack" / "kubejs", 3, h=old_child)
    wsh, _ = _store(root, ws, 1)
    loneh, _ = _store(root, ws / "lone", 1)
    return parent, pids, child, cids, wsh, loneh


def test_plan_folds_only_stale_stores_and_writes_nothing(world):
    home, root = world
    parent, _, child, _, wsh, loneh = _layout(home, root)
    (root / "abcdefabcdef").mkdir()  # no project_path.txt: left alone
    before = sorted(p.name for p in root.iterdir())
    plan = mfs.build_plan(root)
    assert [(f["hash"], f["target"], f["chunks"]) for f in plan["fold"]] == [(child, parent, 3)]
    assert {"hash": "abcdefabcdef", "reason": "no project_path.txt"} in plan["skipped"]
    assert sorted(p.name for p in root.iterdir()) == before


def test_apply_moves_rows_vectors_files_and_proposals_into_the_target(world, tmp_path):
    home, root = world
    parent, pids, child, cids, wsh, loneh = _layout(home, root)
    core = sqlite3.connect(str(root / "core.sqlite"))
    core.execute("CREATE TABLE core_proposals (id TEXT PRIMARY KEY, project TEXT NOT NULL)")
    core.execute("INSERT INTO core_proposals VALUES ('p1', ?)", (child,))
    core.commit()
    core.close()

    report = mfs.apply_plan(root, mfs.build_plan(root), tmp_path / "backup")
    assert report["errors"] == []
    assert report["folded"][0] | {"moved_to": ""} == {
        "hash": child, "target": parent, "rows": 3, "vectors": 3, "files": 1, "proposals": 1, "pool": 0,
        "moved_to": ""}

    meta, lance = _rows(root, parent)
    want = {(i, parent) for i in pids + cids}
    assert meta == want and lance == want  # project rewritten in both stores
    c = sqlite3.connect(str(root / parent / "meta.sqlite"))
    assert c.execute("SELECT COUNT(*) FROM edge_bumps WHERE project = ?", (parent,)).fetchone()[0] == 2
    c.close()
    assert (root / parent / "queue" / f"{child}.json").exists()
    assert not (root / child).exists()
    assert (Path(report["folded"][0]["moved_to"]) / "meta.sqlite").exists()  # set aside, not deleted
    assert (root / wsh).is_dir() and (root / loneh).is_dir()
    # backup holds both stores and the core index as they were
    snap = json.loads((tmp_path / "backup" / "SNAPSHOT.json").read_text())
    assert {p["hash"] for p in snap["projects"]} == {parent, child}
    c = sqlite3.connect(str(tmp_path / "backup" / "core.sqlite"))
    assert c.execute("SELECT project FROM core_proposals").fetchone()[0] == child
    c.close()
    # rerun: nothing left to fold
    assert mfs.build_plan(root)["fold"] == []


def test_apply_points_pool_copies_at_the_target(world, tmp_path):
    home, root = world
    parent, pids, child, cids, wsh, loneh = _layout(home, root)
    g = root / "_global"
    g.mkdir()
    c = sqlite3.connect(str(g / "meta.sqlite"))
    common.ensure_schema(c)
    rows = [("g1", child, [child]), ("g2", wsh, [wsh, child, parent]), ("g3", wsh, [wsh])]
    for gid, sp, sps in rows:
        c.execute("INSERT INTO chunks (id, project, kind, ts, content_hash, source_project, source_id, "
                  "source_projects) VALUES (?,?,?,?,?,?,?,?)",
                  (gid, common.GLOBAL_PROJECT_HASH, "stop", 1, gid, sp, cids[0], json.dumps(sps)))
    c.commit()
    c.close()

    report = mfs.apply_plan(root, mfs.build_plan(root), tmp_path / "backup")
    assert report["errors"] == [] and report["folded"][0]["pool"] == 2
    c = sqlite3.connect(str(g / "meta.sqlite"))
    got = {r[0]: (r[1], json.loads(r[2])) for r in c.execute(
        "SELECT id, source_project, source_projects FROM chunks")}
    c.close()
    # the reconcile looks the source row up in source_project: it must be the store that now holds it
    assert got == {"g1": (parent, [parent]), "g2": (wsh, [wsh, parent]), "g3": (wsh, [wsh])}
    snap = json.loads((tmp_path / "backup" / "SNAPSHOT.json").read_text())
    assert "_global" in {p["hash"] for p in snap["projects"]}


def test_apply_into_a_target_that_does_not_exist_yet(world, tmp_path):
    home, root = world
    proj = home / "site"
    (proj / "app" / "fonts").mkdir(parents=True)
    (proj / "package.json").write_text("{}")
    old = common.hashlib.sha256(str(proj / "app" / "fonts").encode()).hexdigest()[:12]
    _, ids = _store(root, proj / "app" / "fonts", 2, h=old)
    target = common.project_key(proj)
    mfs.apply_plan(root, mfs.build_plan(root), tmp_path / "b")
    assert (root / target / "project_path.txt").read_text() == str(proj)
    meta, lance = _rows(root, target)
    assert meta == lance == {(i, target) for i in ids}


def test_cli_dry_run_by_default_and_apply_refuses_while_flush_runs(world, capsys, monkeypatch):
    home, root = world
    _layout(home, root)
    assert mfs.main(["--brain-root", str(root)]) == 0
    assert "dry run" in capsys.readouterr().out
    (root / "flush.pid").write_text(str(__import__("os").getpid()))
    assert mfs.main(["--brain-root", str(root), "--apply"]) == 3
    assert len(mfs.build_plan(root)["fold"]) == 1
