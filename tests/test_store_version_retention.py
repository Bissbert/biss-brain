"""Each write's index refresh prunes superseded LanceDB versions, so the store
stays the size of its rows instead of keeping a week of full index copies."""
from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

import lancedb

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
os.environ.setdefault("BISS_BRAIN_FAKE_EMBED", "1")

import common  # noqa: E402


def _row(i: int) -> dict:
    text = f"memory number {i} about the store"
    return {"id": f"c{i}", "project": "p", "session_id": "s", "agent_id": "", "kind": "note",
            "ts": i, "text": text, "vector": common._fake_embed(text, 8)}


def test_refresh_prunes_old_versions_and_keeps_every_row(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(common, "VERSION_RETENTION", timedelta(0))
    db = lancedb.connect(str(tmp_path))
    tbl = db.create_table("chunks", schema=common.chunks_schema(8))
    for i in range(6):
        tbl.add([{**{f.name: None for f in common.chunks_schema(8)}, **_row(i)}])
        assert common.refresh_fts(tbl)
    assert tbl.count_rows() == 6
    assert len(tbl.list_versions()) <= 2
    assert tbl.search("number", query_type="fts").limit(10).to_list()


def test_the_default_retention_is_short() -> None:
    assert common.VERSION_RETENTION <= timedelta(minutes=10)


def test_a_cached_ui_table_survives_another_writer_pruning(tmp_path: Path, monkeypatch) -> None:
    from datetime import timedelta as td

    monkeypatch.setattr(common, "VERSION_RETENTION", td(0))
    writer = lancedb.connect(str(tmp_path)).create_table("chunks", schema=common.chunks_schema(8))
    writer.add([{**{f.name: None for f in common.chunks_schema(8)}, **_row(0)}])
    common.refresh_fts(writer)
    reader = lancedb.connect(str(tmp_path), read_consistency_interval=td(0)).open_table("chunks")
    assert reader.count_rows() == 1
    for i in range(1, 4):
        writer.add([{**{f.name: None for f in common.chunks_schema(8)}, **_row(i)}])
        common.refresh_fts(writer)
    assert reader.count_rows() == 4
    assert len(reader.search("number", query_type="fts").limit(10).to_list()) == 4


def _bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _index_bytes(root: Path) -> int:
    return _bytes(root / "chunks.lance" / "_indices")


def test_old_index_copies_leave_the_disk(tmp_path: Path, monkeypatch) -> None:
    """Pruning keeps one live index, not one copy per write: the bytes grow
    with the partitions of the live index only, and compaction folds those."""
    monkeypatch.setattr(common, "VERSION_RETENTION", timedelta(0))
    tbl = lancedb.connect(str(tmp_path)).create_table("chunks", schema=common.chunks_schema(8))
    empty = {f.name: None for f in common.chunks_schema(8)}
    for i in range(30):
        tbl.add([{**empty, **_row(i)}])
        assert common.refresh_fts(tbl)
    live = [d for d in (tmp_path / "chunks.lance" / "_indices").iterdir() if any(d.iterdir())]
    assert len(live) <= 2
    before = _index_bytes(tmp_path)
    assert common.compact_store(tbl)
    assert _index_bytes(tmp_path) * 4 < before
    assert tbl.count_rows() == 30
    assert len(tbl.search("number", query_type="fts").limit(50).to_list()) == 30


def test_the_dream_compacts_every_store(tmp_path: Path, monkeypatch) -> None:
    import dream

    monkeypatch.setattr(common, "VERSION_RETENTION", timedelta(0))
    # another test may have reloaded common; patch the module dream calls into
    monkeypatch.setitem(dream.compact_store.__globals__, "VERSION_RETENTION", timedelta(0))
    pdir = tmp_path / "abc123abc123"
    tbl = lancedb.connect(str(pdir / "lancedb")).create_table("chunks", schema=common.chunks_schema(8))
    empty = {f.name: None for f in common.chunks_schema(8)}
    for i in range(20):
        tbl.add([{**empty, **_row(i)}])
        common.refresh_fts(tbl)
    before = _index_bytes(pdir / "lancedb")
    assert dream._stage1d_compact(pdir, dry_run=True) is False
    assert dream._stage1d_compact(pdir, dry_run=False) is True
    assert _index_bytes(pdir / "lancedb") * 2 < before
    assert tbl.count_rows() == 20
    assert dream._stage1d_compact(tmp_path / "empty", dry_run=False) is False
