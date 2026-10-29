#!/usr/bin/env python3
"""Validation tests for scripts/remember.py (Feature 4.1).

- Inserts a chunk via remember.py with BISS_BRAIN_FAKE_EMBED=1.
- Asserts kind='decision', pinned=1, superseded_by='' in SQLite.
- Asserts the chunk is queryable via recall.py.
- Mocks the classifier subprocess so no real claude -p calls are made.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import tempfile
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _write_test_config(tmp: Path) -> Path:
    cfg_path = tmp / "config.toml"
    cfg_path.write_text(
        "[distill]\nbackend = \"passthrough\"\n"
        "[embed]\nbackend = \"ollama\"\nmodel = \"nomic-embed-text\"\ndim = 768\n"
        "endpoint = \"http://127.0.0.1:11434\"\n"
        "[retrieve]\nhybrid = false\n"
    )
    return cfg_path


def test_remember_decision_pinned() -> None:
    """remember.py --kind decision should store kind=decision, pinned=1, superseded_by=''."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp)
        fake_cwd = tmp / "project"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)

        # Reload modules so env vars are respected
        import importlib
        for mod in ["common", "remember"]:
            if mod in sys.modules:
                del sys.modules[mod]

        import common
        import remember

        text = "we picked LanceDB over Qdrant"
        chunk_id = remember.insert_chunk(
            text=text,
            kind="decision",
            pinned=False,  # kind=decision should auto-pin
            cwd=str(fake_cwd),
        )
        assert chunk_id, "insert_chunk must return a non-empty id"

        meta = common.open_meta(str(fake_cwd))
        row = meta.execute(
            "SELECT kind, pinned, superseded_by FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        meta.close()

        assert row is not None, f"chunk {chunk_id} not found in SQLite"
        kind_val, pinned_val, superseded_val = row
        assert kind_val == "decision", f"expected kind=decision, got {kind_val!r}"
        assert pinned_val == 1, f"expected pinned=1, got {pinned_val!r}"
        assert superseded_val == "", f"expected superseded_by='', got {superseded_val!r}"
        print(f"  PASS: chunk {chunk_id[:8]} has kind=decision, pinned=1, superseded_by=''")

        # Cleanup
        del os.environ["BISS_BRAIN_CONFIG"]


def test_remember_explicit_pin() -> None:
    """--pin flag should set pinned=1 regardless of kind."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp)
        fake_cwd = tmp / "project2"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)

        for mod in ["common", "remember"]:
            if mod in sys.modules:
                del sys.modules[mod]

        import common
        import remember

        chunk_id = remember.insert_chunk(
            text="some note about architecture",
            kind="note",
            pinned=True,
            cwd=str(fake_cwd),
        )

        meta = common.open_meta(str(fake_cwd))
        row = meta.execute(
            "SELECT kind, pinned FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        meta.close()

        assert row is not None
        assert row[0] == "note"
        assert row[1] == 1, f"expected pinned=1 from --pin flag, got {row[1]}"
        print(f"  PASS: --pin flag sets pinned=1 for kind=note chunk {chunk_id[:8]}")

        del os.environ["BISS_BRAIN_CONFIG"]


def test_remember_queryable_via_recall() -> None:
    """Chunk inserted by remember.py must be retrievable by recall-style search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp)
        fake_cwd = tmp / "project3"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)

        for mod in ["common", "remember"]:
            if mod in sys.modules:
                del sys.modules[mod]

        import common
        import remember

        text = "biss-brain uses LanceDB for vector storage"
        chunk_id = remember.insert_chunk(
            text=text,
            kind="note",
            pinned=False,
            cwd=str(fake_cwd),
        )

        # Directly query LanceDB to confirm the chunk exists
        import lancedb
        from common import lancedb_dir, fetch_by_ids
        db = lancedb.connect(str(lancedb_dir(str(fake_cwd))))
        assert "chunks" in db.list_tables().tables, "LanceDB table missing"
        tbl = db.open_table("chunks")
        rows = fetch_by_ids(tbl, [chunk_id], limit=1)
        assert rows, f"chunk {chunk_id[:8]} not found in LanceDB"
        assert rows[0]["text"] == text, "text mismatch in LanceDB"
        print(f"  PASS: chunk {chunk_id[:8]} queryable from LanceDB")

        del os.environ["BISS_BRAIN_CONFIG"]


def test_classifier_is_mocked() -> None:
    """_classify_kind must not call real subprocess — mock verifies no claude -p call."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp)
        fake_cwd = tmp / "project4"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)

        for mod in ["common", "remember"]:
            if mod in sys.modules:
                del sys.modules[mod]

        import remember

        mock_result = '{"result": "task"}'
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = mock_result
        mock_proc.stderr = ""

        with patch("cli_resolver.find_claude_cli", return_value=Path("/fake/claude")), \
             patch("subprocess.run", return_value=mock_proc) as mock_run:
            kind = remember._classify_kind("implement the new feature by Friday")

        mock_run.assert_called_once()
        assert kind == "task", f"expected 'task', got {kind!r}"
        print(f"  PASS: classifier returned {kind!r} via mocked subprocess")

        del os.environ["BISS_BRAIN_CONFIG"]


def test_dedup_returns_existing_id() -> None:
    """Inserting the same text twice must return the same chunk_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cfg_path = _write_test_config(tmp)
        fake_cwd = tmp / "project5"
        fake_cwd.mkdir()

        os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"
        os.environ["BISS_BRAIN_CONFIG"] = str(cfg_path)

        for mod in ["common", "remember"]:
            if mod in sys.modules:
                del sys.modules[mod]

        import remember

        text = "unique memory for dedup test"
        id1 = remember.insert_chunk(text=text, kind="note", pinned=False, cwd=str(fake_cwd))
        id2 = remember.insert_chunk(text=text, kind="note", pinned=False, cwd=str(fake_cwd))
        assert id1 == id2, f"expected same id on dedup, got {id1!r} vs {id2!r}"
        print(f"  PASS: duplicate insert returned same id {id1[:8]}")

        del os.environ["BISS_BRAIN_CONFIG"]


if __name__ == "__main__":
    print("test_remember.py")
    test_remember_decision_pinned()
    test_remember_explicit_pin()
    test_remember_queryable_via_recall()
    test_classifier_is_mocked()
    test_dedup_returns_existing_id()
    print("\nAll remember tests passed.")
