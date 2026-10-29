#!/usr/bin/env python3
"""Conflict judge: the cosine gate in flush._run_conflict_judge and judge.py itself.

The judge is mocked. Vectors are chosen by hand so the gate outcome is known:
a near-duplicate passes the 0.85 cosine gate, an orthogonal vector does not.
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import lance_filter as F  # noqa: E402
from common import normalize  # noqa: E402

DIM = 8
CFG = {
    "embed": {"dim": DIM},
    "distill": {"models": {"conflict_judge": "haiku"}},
    "hygiene": {"conflict_judge_enabled": True, "conflict_cosine": 0.85},
}


def _flush():
    # Use flush's own bindings: other tests reload `common` with a different root.
    import flush

    return flush


def _insert_chunk(cwd: str, chunk_id: str, text: str, vec: list[float], ts: int, kind: str = "stop") -> None:
    fl = _flush()
    proj = fl.project_key(cwd)
    h = hashlib.sha256(text.encode()).hexdigest()
    tbl = fl._open_table(cwd, DIM)
    tbl.add([{
        "id": chunk_id, "project": proj, "session_id": "", "agent_id": "", "kind": kind,
        "ts": ts, "text": text, "content_hash": h, "superseded_by": "", "vector": vec,
    }])
    meta = fl.open_meta(cwd)
    meta.execute(
        "INSERT INTO chunks (id, project, session_id, agent_id, kind, ts, content_hash, superseded_by, last_accessed) "
        "VALUES (?, ?, '', '', ?, ?, ?, '', ?)",
        (chunk_id, proj, kind, ts, h, ts),
    )
    meta.commit()
    meta.close()


def _run(old_vec: list[float], new_vec: list[float]) -> tuple[str, str, str, int]:
    """Insert old then new, run the judge; return (old_id, new_id, meta superseded_by, judge calls)."""
    fl = _flush()
    with tempfile.TemporaryDirectory() as cwd:
        old_id, new_id = uuid.uuid4().hex, uuid.uuid4().hex
        now = int(time.time())
        _insert_chunk(cwd, old_id, "Authentication uses JWT tokens signed with HS256.", old_vec, now - 60)
        _insert_chunk(cwd, new_id, "Authentication uses server-side session cookies.", new_vec, now)
        with patch.object(fl, "judge_conflict", return_value="CONTRADICTS") as judge:
            fl._run_conflict_judge(cwd=cwd, proj=fl.project_key(cwd), kind="stop", new_id=new_id,
                                new_text="Authentication uses server-side session cookies.",
                                new_vec=new_vec, cfg=CFG)
        meta = fl.open_meta(cwd)
        sup = meta.execute("SELECT superseded_by FROM chunks WHERE id = ?", (old_id,)).fetchone()[0]
        meta.close()
        lance_sup = fl._open_table(cwd, DIM).search().where(F.eq("id", old_id)).limit(1).to_list()
        assert lance_sup[0]["superseded_by"] == sup, "LanceDB and SQLite disagree"
        return old_id, new_id, sup, judge.call_count


def test_near_duplicate_contradiction_supersedes_old_chunk() -> None:
    old = normalize([1, 0, 0, 0, 0, 0, 0, 0])
    new = normalize([1, 0.1, 0, 0, 0, 0, 0, 0])  # cosine ~0.995
    _, new_id, sup, calls = _run(old, new)
    assert calls == 1
    assert sup == new_id


def test_dissimilar_chunk_is_never_judged() -> None:
    old = normalize([1, 0, 0, 0, 0, 0, 0, 0])
    new = normalize([0, 1, 0, 0, 0, 0, 0, 0])  # cosine 0
    _, _, sup, calls = _run(old, new)
    assert calls == 0
    assert sup == ""


def test_judge_conflict_returns_valid_verdict() -> None:
    """Unit test judge_conflict itself with mocked subprocess."""
    import json
    from unittest.mock import patch, MagicMock

    from judge import judge_conflict

    mock_result = json.dumps({"result": "CONTRADICTS"})
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = mock_result
    mock_proc.stderr = ""

    with patch("cli_resolver.find_claude_cli", return_value=Path("/fake/claude")), \
         patch("subprocess.run", return_value=mock_proc):
        verdict = judge_conflict("uses JWT", "uses session cookies")

    assert verdict == "CONTRADICTS", f"expected CONTRADICTS, got {verdict!r}"
    print(f"  PASS: judge_conflict returned {verdict!r}")


def test_judge_conflict_fallback_on_error() -> None:
    """judge_conflict must return None (not a verdict, not raise) when subprocess fails."""
    from unittest.mock import patch, MagicMock

    from judge import judge_conflict

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stdout = ""
    mock_proc.stderr = "some error"

    with patch("cli_resolver.find_claude_cli", return_value=Path("/fake/claude")), \
         patch("subprocess.run", return_value=mock_proc):
        verdict = judge_conflict("A", "B")

    assert verdict is None, f"expected None, got {verdict!r}"
    print(f"  PASS: judge_conflict fallback returned {verdict!r}")


if __name__ == "__main__":
    print("test_judge.py")
    test_judge_conflict_returns_valid_verdict()
    test_judge_conflict_fallback_on_error()
    test_near_duplicate_contradiction_supersedes_old_chunk()
    test_dissimilar_chunk_is_never_judged()
    print("\nAll judge tests passed.")
