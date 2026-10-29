#!/usr/bin/env python3
"""Tests for _try_reconsolidate() in scripts/flush.py (B4 Reconsolidation).

Strategy:
- Use BISS_BRAIN_FAKE_EMBED + deterministic _fake_embed vectors.
- Insert labile chunks directly into SQLite + LanceDB.
- Monkeypatch distill to return deterministic JSON merge output.
- Call _try_reconsolidate directly and assert DB state.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
import math
import random
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import lancedb
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Force fake embeddings throughout
os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"


# ── helpers ──────────────────────────────────────────────────────────────

DIM = 768


def _fake_embed(text: str, dim: int = DIM) -> list[float]:
    """Deterministic fake embedding matching common._fake_embed."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-9:
        return v
    return [x / norm for x in v]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


def _make_close_vec(base: list[float], similarity: float) -> list[float]:
    """Create a vector with approximately `similarity` cosine to `base`.

    Interpolates between base (unit-normalised) and an orthogonal vector.
    Returns a unit-normalised result.
    """
    bn = _normalize(base)
    # Create a perpendicular vector via random projection orthogonalised to bn
    rng = random.Random(42)
    perp = [rng.gauss(0, 1) for _ in range(len(bn))]
    dot = sum(p * b for p, b in zip(perp, bn))
    perp = [p - dot * b for p, b in zip(perp, bn)]
    pn = _normalize(perp)
    # Blend: cos(theta) = similarity means the component along bn = similarity
    sin_t = math.sqrt(max(0.0, 1.0 - similarity * similarity))
    blended = [similarity * b + sin_t * p for b, p in zip(bn, pn)]
    return _normalize(blended)


def _make_lancedb_table(cwd: str, dim: int = DIM):
    """Open or create the LanceDB chunks table for a given cwd."""
    from common import lancedb_dir
    db = lancedb.connect(str(lancedb_dir(cwd)))
    if "chunks" in db.list_tables().tables:
        return db.open_table("chunks")
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("project", pa.string()),
        pa.field("session_id", pa.string()),
        pa.field("agent_id", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("ts", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("content_hash", pa.string()),
        pa.field("superseded_by", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])
    return db.create_table("chunks", schema=schema, mode="create")


def _insert_labile_chunk(
    cwd: str,
    text: str,
    vec: list[float],
    kind: str = "stop",
    reconsolidation_count: int = 0,
    labile_offset: int = 300,
) -> str:
    """Insert a labile chunk into both SQLite and LanceDB.

    labile_offset > 0 means labile_until = now + offset (still labile).
    labile_offset <= 0 is used to create expired labile windows.
    """
    from common import open_meta, project_key, content_hash as _ch

    proj = project_key(cwd)
    chunk_id = uuid.uuid4().hex
    ts = int(time.time())
    labile_until = ts + labile_offset
    h = _ch(text)

    meta = open_meta(cwd)
    meta.execute(
        "INSERT INTO chunks "
        "(id, project, session_id, agent_id, kind, ts, content_hash, "
        "superseded_by, last_accessed, labile_until, reconsolidation_count) "
        "VALUES (?, ?, '', '', ?, ?, ?, '', ?, ?, ?)",
        (chunk_id, proj, kind, ts, h, ts, labile_until, reconsolidation_count),
    )
    meta.commit()
    meta.close()

    tbl = _make_lancedb_table(cwd)
    tbl.add([{
        "id": chunk_id,
        "project": proj,
        "session_id": "",
        "agent_id": "",
        "kind": kind,
        "ts": ts,
        "text": text,
        "content_hash": h,
        "superseded_by": "",
        "vector": [float(x) for x in vec],
    }])
    return chunk_id


def _make_cfg(tmp_dir: str) -> dict:
    """Minimal config dict that enables reconsolidation with fake embed."""
    return {
        "embed": {"backend": "ollama", "model": "nomic-embed-text", "dim": DIM,
                  "endpoint": "http://127.0.0.1:11434"},
        "distill": {
            "backend": "passthrough",
            "models": {"merge": "haiku"},
            "ollama": {"model": "qwen2.5-coder:14b-instruct"},
        },
        "reconsolidation": {
            "enabled": True,
            "labile_window_seconds": 300,
            "merge_cosine_threshold": 0.82,
            "promote_to_lesson_after": 5,
        },
    }


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_cwd(tmp_path: Path) -> str:
    """Return a fresh cwd string backed by a temp dir."""
    cwd = str(tmp_path / "project")
    Path(cwd).mkdir()
    return cwd


# ── tests ─────────────────────────────────────────────────────────────────

def test_reconsolidate_merges_above_threshold(tmp_cwd: str) -> None:
    """Labile chunk with cosine ~0.9 to new text triggers merge; no new row inserted."""
    from common import open_meta, project_key
    import flush as _flush

    cfg = _make_cfg(tmp_cwd)
    proj = project_key(tmp_cwd)

    # Create existing labile chunk
    old_text = "The deploy script uses rsync to copy artifacts to /srv/app."
    old_vec = _normalize(_fake_embed(old_text))

    # New vector close to old (cosine ~0.9)
    new_vec = _make_close_vec(old_vec, 0.90)
    new_text = "The deploy script copies build artifacts to /srv/app via rsync."

    existing_id = _insert_labile_chunk(tmp_cwd, old_text, old_vec)

    tbl = _make_lancedb_table(tmp_cwd)

    merged_json = json.dumps({"merged_text": "The deploy script uses rsync to copy artifacts to /srv/app."})

    with patch("flush.ask", return_value=merged_json), \
         patch("flush._embed", return_value=_make_close_vec(old_vec, 0.95)):
        meta = open_meta(tmp_cwd)
        result = _flush._try_reconsolidate(tmp_cwd, tbl, meta, new_text, new_vec, cfg)
        meta.close()

    assert result == existing_id, f"Expected merge to return {existing_id[:8]}, got {result}"

    # Verify reconsolidation_count bumped to 1 (in SQLite metadata)
    meta = open_meta(tmp_cwd)
    row = meta.execute(
        "SELECT reconsolidation_count FROM chunks WHERE id = ?", (existing_id,)
    ).fetchone()
    meta.close()

    assert row is not None
    assert row[0] == 1, f"Expected reconsolidation_count=1, got {row[0]}"


def test_reconsolidate_skips_below_threshold(tmp_cwd: str) -> None:
    """Cosine below threshold (0.5 < 0.82): no merge, returns None."""
    from common import open_meta, project_key
    import flush as _flush

    cfg = _make_cfg(tmp_cwd)

    old_text = "The database uses PostgreSQL 15 with JSONB columns."
    old_vec = _normalize(_fake_embed(old_text))

    # New vector far from old (~0.5 cosine)
    new_vec = _make_close_vec(old_vec, 0.50)
    new_text = "Frontend builds are done with Vite and React 18."

    existing_id = _insert_labile_chunk(tmp_cwd, old_text, old_vec)
    tbl = _make_lancedb_table(tmp_cwd)

    meta = open_meta(tmp_cwd)
    result = _flush._try_reconsolidate(tmp_cwd, tbl, meta, new_text, new_vec, cfg)
    meta.close()

    assert result is None, f"Expected None (no merge), got {result}"

    # Original chunk must be untouched
    meta = open_meta(tmp_cwd)
    row = meta.execute(
        "SELECT reconsolidation_count FROM chunks WHERE id = ?", (existing_id,)
    ).fetchone()
    meta.close()
    assert row[0] == 0, f"Expected reconsolidation_count=0, got {row[0]}"


def test_reconsolidate_skips_when_no_labile_window(tmp_cwd: str) -> None:
    """Candidate with labile_until <= now is ignored; returns None."""
    from common import open_meta
    import flush as _flush

    cfg = _make_cfg(tmp_cwd)

    old_text = "CI pipeline runs on GitHub Actions with Ubuntu 22.04."
    old_vec = _normalize(_fake_embed(old_text))

    # Insert with expired labile window (labile_offset = -10, i.e. already expired)
    existing_id = _insert_labile_chunk(tmp_cwd, old_text, old_vec, labile_offset=-10)
    tbl = _make_lancedb_table(tmp_cwd)

    new_vec = _make_close_vec(old_vec, 0.95)
    new_text = "CI pipeline runs on GitHub Actions with Ubuntu 22.04 LTS."

    meta = open_meta(tmp_cwd)
    result = _flush._try_reconsolidate(tmp_cwd, tbl, meta, new_text, new_vec, cfg)
    meta.close()

    assert result is None, "Expired labile_until must not trigger merge"


def test_reconsolidate_promotes_to_lesson_after_5(tmp_cwd: str) -> None:
    """After 5 reconsolidations, chunk becomes kind='lesson', consolidation_round=2."""
    from common import open_meta, project_key
    import flush as _flush

    cfg = _make_cfg(tmp_cwd)

    old_text = "Auth tokens expire after 24 hours; refresh via /api/auth/refresh."
    old_vec = _normalize(_fake_embed(old_text))

    # Insert with reconsolidation_count = 4 (one more will hit promote_after=5)
    existing_id = _insert_labile_chunk(
        tmp_cwd, old_text, old_vec, reconsolidation_count=4
    )
    tbl = _make_lancedb_table(tmp_cwd)

    new_vec = _make_close_vec(old_vec, 0.92)
    new_text = "Auth tokens expire in 24h; use /api/auth/refresh to renew."

    merged_json = json.dumps({"merged_text": "Auth tokens expire after 24 hours. Refresh via /api/auth/refresh."})

    with patch("flush.ask", return_value=merged_json), \
         patch("flush._embed", return_value=_make_close_vec(old_vec, 0.97)):
        meta = open_meta(tmp_cwd)
        result = _flush._try_reconsolidate(tmp_cwd, tbl, meta, new_text, new_vec, cfg)
        meta.close()

    assert result == existing_id, "Merge should have occurred"

    meta = open_meta(tmp_cwd)
    row = meta.execute(
        "SELECT reconsolidation_count, kind, consolidation_round FROM chunks WHERE id = ?",
        (existing_id,),
    ).fetchone()
    meta.close()

    assert row is not None
    assert row[0] == 5, f"Expected reconsolidation_count=5, got {row[0]}"
    assert row[1] == "lesson", f"Expected kind='lesson', got {row[1]!r}"
    assert row[2] == 2, f"Expected consolidation_round=2, got {row[2]}"


def test_reconsolidate_preserves_distinct_facts(tmp_cwd: str) -> None:
    """Merged text must contain facts from both chunks (via deterministic stub distill)."""
    from common import open_meta
    import flush as _flush

    cfg = _make_cfg(tmp_cwd)

    old_text = "The logging library is structlog version 23.1."
    new_text = "Error handling uses Sentry with DSN configured in .env."

    old_vec = _normalize(_fake_embed(old_text))
    new_vec = _make_close_vec(old_vec, 0.88)

    # The stub distill returns both facts merged
    combined = "The logging library is structlog version 23.1. Error handling uses Sentry with DSN configured in .env."
    merged_json = json.dumps({"merged_text": combined})

    _insert_labile_chunk(tmp_cwd, old_text, old_vec)
    tbl = _make_lancedb_table(tmp_cwd)

    with patch("flush.ask", return_value=merged_json), \
         patch("flush._embed", return_value=_make_close_vec(old_vec, 0.93)):
        meta = open_meta(tmp_cwd)
        result = _flush._try_reconsolidate(tmp_cwd, tbl, meta, new_text, new_vec, cfg)
        meta.close()

    assert result is not None, "Merge should have occurred"

    # Check merged text contains both facts — text lives in LanceDB
    tbl2 = _make_lancedb_table(tmp_cwd)
    hits = tbl2.search().where(f"id = '{result}'").limit(1).to_list()
    assert hits, f"chunk {result[:8]} not found in LanceDB after merge"
    merged_stored = hits[0].get("text", "")
    assert "structlog" in merged_stored, "Old fact (structlog) missing from merged text"
    assert "Sentry" in merged_stored, "New fact (Sentry) missing from merged text"


def test_drift_gate_rejects_hallucinated_merge(tmp_cwd: str) -> None:
    """Merged embedding with high drift from midpoint rejects the merge; new row must be inserted."""
    from common import open_meta
    import flush as _flush

    cfg = _make_cfg(tmp_cwd)

    old_text = "The cache layer is Redis 7 with 16 GB memory limit."
    old_vec = _normalize(_fake_embed(old_text))
    new_vec = _make_close_vec(old_vec, 0.90)
    new_text = "Redis 7 is used as cache with 16 GB RAM."

    _insert_labile_chunk(tmp_cwd, old_text, old_vec)
    tbl = _make_lancedb_table(tmp_cwd)

    merged_json = json.dumps({"merged_text": "Completely unrelated hallucinated content about machine learning."})

    # Drift vector: orthogonal to midpoint (drift > 0.12 guaranteed)
    midpoint = [o + n for o, n in zip(old_vec, new_vec)]
    mp_norm = math.sqrt(sum(x * x for x in midpoint))
    midpoint_unit = [x / mp_norm for x in midpoint]
    # Create a vector perpendicular to midpoint => drift = 1.0 - 0 = 1.0
    rng = random.Random(99)
    perp = [rng.gauss(0, 1) for _ in range(DIM)]
    dot = sum(p * m for p, m in zip(perp, midpoint_unit))
    perp = [p - dot * m for p, m in zip(perp, midpoint_unit)]
    perp_n = math.sqrt(sum(x * x for x in perp))
    halluc_vec = [x / perp_n for x in perp]

    with patch("flush.ask", return_value=merged_json), \
         patch("flush._embed", return_value=halluc_vec):
        meta = open_meta(tmp_cwd)
        result = _flush._try_reconsolidate(tmp_cwd, tbl, meta, new_text, new_vec, cfg)
        meta.close()

    assert result is None, "Drift gate must reject hallucinated merge (return None)"

    # Original chunk must be untouched
    meta = open_meta(tmp_cwd)
    row = meta.execute(
        "SELECT reconsolidation_count FROM chunks"
    ).fetchone()
    meta.close()
    assert row[0] == 0, "Chunk must remain unchanged after drift-gate rejection"


def test_reconsolidate_atomic(tmp_cwd: str) -> None:
    """Simulate failure mid-merge (_embed raises after distill succeeds).

    When the embedding of merged text fails, no SQLite changes must be committed —
    reconsolidation_count stays at 0 and _try_reconsolidate returns None.
    This validates the atomicity invariant: no partial-merge state is possible
    because the SQLite UPDATE only runs after all preconditions (embed, drift gate)
    succeed.
    """
    from common import open_meta
    import flush as _flush

    cfg = _make_cfg(tmp_cwd)

    old_text = "Migrations run via Alembic; run `alembic upgrade head` to apply."
    old_vec = _normalize(_fake_embed(old_text))
    new_vec = _make_close_vec(old_vec, 0.93)
    new_text = "Apply DB migrations with `alembic upgrade head` (Alembic tool)."

    existing_id = _insert_labile_chunk(tmp_cwd, old_text, old_vec)
    tbl = _make_lancedb_table(tmp_cwd)

    merged_json = json.dumps({"merged_text": "Migrations via Alembic; `alembic upgrade head`."})

    # Simulate: distill succeeds, but embedding the merged text raises.
    # This means the UPDATE is never reached; count must remain 0.
    def embed_fails(text: str, c: dict) -> list[float]:
        raise RuntimeError("simulated embed failure mid-merge")

    with patch("flush.ask", return_value=merged_json), \
         patch("flush._embed", side_effect=embed_fails):
        meta = open_meta(tmp_cwd)
        result = _flush._try_reconsolidate(tmp_cwd, tbl, meta, new_text, new_vec, cfg)
        meta.close()

    # Embed failed => merge aborted before any SQLite write
    assert result is None, "Embed failure must abort merge and return None"

    meta = open_meta(tmp_cwd)
    row = meta.execute(
        "SELECT reconsolidation_count FROM chunks WHERE id = ?", (existing_id,)
    ).fetchone()
    meta.close()

    assert row is not None
    assert row[0] == 0, (
        f"No SQLite writes must happen before embed succeeds; "
        f"expected count=0, got {row[0]}"
    )


def test_reconsolidate_disabled_skips(tmp_cwd: str) -> None:
    """When reconsolidation.enabled=False, _try_reconsolidate always returns None."""
    from common import open_meta
    import flush as _flush

    cfg = _make_cfg(tmp_cwd)
    cfg["reconsolidation"]["enabled"] = False

    old_text = "Feature flags are managed via LaunchDarkly."
    old_vec = _normalize(_fake_embed(old_text))
    _insert_labile_chunk(tmp_cwd, old_text, old_vec)
    tbl = _make_lancedb_table(tmp_cwd)

    new_vec = _make_close_vec(old_vec, 0.95)
    new_text = "LaunchDarkly manages all feature flags."

    meta = open_meta(tmp_cwd)
    result = _flush._try_reconsolidate(tmp_cwd, tbl, meta, new_text, new_vec, cfg)
    meta.close()

    assert result is None, "reconsolidation.enabled=False must always return None"
