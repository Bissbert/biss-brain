"""Deterministic retrieval fixture shared by parity tests and the golden generator.

Builds one project under $BISS_BRAIN_ROOT with a LanceDB table (dim 8, fake
embeddings, FTS index) and a migrated meta.sqlite with varied decay scores,
kinds and superseded rows. Callers run retrieval with cwd = the returned dir.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import lancedb
import pyarrow as pa

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

DIM = 8
TOPICS = [
    "lancedb hybrid search uses bm25 and dense vectors",
    "ollama serves nomic embed text on localhost",
    "the dream pass consolidates sessions into arcs nightly",
    "salience scoring weights novelty against recent embeddings",
    "reconsolidation merges a labile chunk with new evidence",
    "rif lowers decay for siblings of retrieved chunks",
    "stdp bumps edges between co retrieved chunks",
    "scrub removes api keys and tokens before storage",
    "flush daemon drains the queue and distills events",
    "the ui shows a scatter projection of embeddings",
]
KINDS = ["stop", "decision", "arc", "daily_summary", "note", "subagent_stop", "hypothesis"]
QUERIES = [
    "how does hybrid search work",
    "nightly consolidation of sessions",
    "secrets removal before storing",
    "edges between chunks retrieved together",
    "embedding server",
    "decay of sibling memories",
]


def fake_vec(text: str) -> list[float]:
    from common import _fake_embed

    return _fake_embed(text, DIM)


def write_config(root: Path, hybrid: bool, summary_first: bool) -> Path:
    p = root / "config.toml"
    p.write_text(
        "[distill]\nbackend = \"passthrough\"\n"
        f"[embed]\nmodel = \"fake\"\ndim = {DIM}\nendpoint = \"http://127.0.0.1:9\"\n"
        f"[retrieve]\nhybrid = {'true' if hybrid else 'false'}\nrerank = false\ndecay_weight = 0.1\n"  # goldens were generated at 0.1
        "user_prompt_inject = true\ninject_min_cosine = -1.0\ninject_min_bm25 = 0\ninject_min_words = 1\n"
        "inject_top_k = 3\n"
        f"[dream]\nprefer_summaries_in_recall = {'true' if summary_first else 'false'}\n"
        "[cross_project]\nenabled = false\n"
    )
    return p


def build(cwd: Path) -> list[str]:
    """Populate the project for *cwd*. Returns chunk ids in insertion order."""
    from common import lancedb_dir, open_meta, project_key

    proj = project_key(cwd)
    rows, ids = [], []
    for i in range(30):
        topic = TOPICS[i % len(TOPICS)]
        text = f"{topic}; variant {i} note {'alpha' if i % 2 else 'beta'}"
        cid = hashlib.sha256(f"chunk-{i}".encode()).hexdigest()[:32]
        ids.append(cid)
        rows.append({
            "id": cid,
            "project": proj,
            "session_id": f"sess{i % 5:02d}",
            "agent_id": "",
            "kind": KINDS[i % len(KINDS)],
            "ts": 1_700_000_000 + i * 3600,
            "text": text,
            "content_hash": hashlib.sha256(text.encode()).hexdigest(),
            "superseded_by": ids[0] if i % 11 == 10 else "",
            "vector": fake_vec(text),
        })
    schema = pa.schema([
        pa.field("id", pa.string()), pa.field("project", pa.string()),
        pa.field("session_id", pa.string()), pa.field("agent_id", pa.string()),
        pa.field("kind", pa.string()), pa.field("ts", pa.int64()),
        pa.field("text", pa.string()), pa.field("content_hash", pa.string()),
        pa.field("superseded_by", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), DIM)),
    ])
    db = lancedb.connect(str(lancedb_dir(cwd)))
    tbl = db.create_table("chunks", schema=schema, mode="overwrite")
    tbl.add(rows)
    tbl.create_fts_index("text", replace=True)
    meta = open_meta(cwd)
    for i, r in enumerate(rows):
        meta.execute(
            "INSERT INTO chunks (id, project, session_id, agent_id, kind, ts, content_hash, "
            "superseded_by, last_accessed, decay_score, pinned) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (r["id"], proj, r["session_id"], "", r["kind"], r["ts"], r["content_hash"],
             r["superseded_by"], r["ts"], round(0.2 + (i * 37 % 80) / 100, 2), 1 if i % 9 == 0 else 0),
        )
    meta.commit()
    meta.close()
    return ids


def scenarios() -> list[tuple[str, bool, bool]]:
    return [("hybrid", True, False), ("dense", False, False), ("hybrid_summary", True, True)]


def env_for(root: Path, hybrid: bool, summary_first: bool) -> dict:
    cfg = write_config(root, hybrid, summary_first)
    env = dict(os.environ)
    env["BISS_BRAIN_ROOT"] = str(root)
    env["BISS_BRAIN_CONFIG"] = str(cfg)
    env["BISS_BRAIN_FAKE_EMBED"] = "1"
    return env
