#!/usr/bin/env python3
"""Probe the installed LanceDB API to find the working hybrid search form.

Creates a temp DB with 10 rows, builds FTS, tries all plausible hybrid search
API forms and reports which ones work.
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

import lancedb
import pyarrow as pa

SEED = 42
DIM = 16
random.seed(SEED)


def rand_vec() -> list[float]:
    return [random.random() for _ in range(DIM)]


TEXTS = [
    "The database schema uses primary keys for deduplication",
    "Python async tasks run concurrently via TaskGroup",
    "Vector embeddings capture semantic similarity",
    "The SQLite WAL mode improves concurrent write performance",
    "LanceDB stores vectors and text in a single file-backed store",
    "Hybrid search combines BM25 full-text and dense vector retrieval",
    "Session transcripts are chunked and embedded for retrieval",
    "The debounce window collapses rapid events into one batch",
    "Config overrides allow per-project customisation of model choice",
    "Metadata indexes speed up filtering by kind, ts, and project",
]


def make_table(db):
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("project", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("ts", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), DIM)),
    ])
    rows = [
        {
            "id": f"id_{i:03d}",
            "project": "testproj",
            "kind": "stop",
            "ts": 1700000000 + i * 100,
            "text": TEXTS[i],
            "vector": rand_vec(),
        }
        for i in range(len(TEXTS))
    ]
    tbl = db.create_table("chunks", data=rows, schema=schema, mode="create")
    return tbl


def try_fts(tbl):
    print("\n--- FTS index creation ---")
    try:
        tbl.create_fts_index("text", replace=True)
        print("  FTS index: OK")
        return True
    except Exception as e:
        print(f"  FTS index: FAILED — {e}")
        return False


def try_forms(tbl, fts_ok: bool) -> str | None:
    """Try all plausible search API forms. Return name of first working hybrid."""
    qvec = rand_vec()
    qtext = "vector database hybrid search"

    working_hybrid = None

    # Form 1: tuple form
    print("\n--- Form 1: tbl.search((qvec, qtext), query_type='hybrid') ---")
    try:
        rows = (
            tbl.search((qvec, qtext), query_type="hybrid")
            .limit(3)
            .to_list()
        )
        print(f"  Result count: {len(rows)}")
        if rows:
            print(f"  First: {rows[0]['text'][:60]}")
            working_hybrid = working_hybrid or "tuple"
            print("  STATUS: WORKS")
    except Exception as e:
        print(f"  STATUS: FAILED — {e}")

    # Form 2: .search(qvec).vector(qvec)  — fluent builder with both
    print("\n--- Form 2: tbl.search(qtext).vector(qvec) ---")
    try:
        rows = (
            tbl.search(qtext)
            .vector(qvec)
            .limit(3)
            .to_list()
        )
        print(f"  Result count: {len(rows)}")
        if rows:
            print(f"  First: {rows[0]['text'][:60]}")
            working_hybrid = working_hybrid or "text_then_vector"
            print("  STATUS: WORKS")
    except Exception as e:
        print(f"  STATUS: FAILED — {e}")

    # Form 3: pure dense vector
    print("\n--- Form 3: tbl.search(qvec) pure dense ---")
    try:
        rows = (
            tbl.search(qvec)
            .limit(3)
            .to_list()
        )
        print(f"  Result count: {len(rows)}")
        if rows:
            print(f"  First: {rows[0]['text'][:60]}")
            print("  STATUS: WORKS")
    except Exception as e:
        print(f"  STATUS: FAILED — {e}")

    # Form 4: .where() style listing (no query, just filter)
    print("\n--- Form 4: tbl.search().where('project = ...') listing ---")
    try:
        rows = (
            tbl.search()
            .where("project = 'testproj'")
            .limit(5)
            .to_list()
        )
        print(f"  Result count: {len(rows)}")
        print("  STATUS: WORKS")
    except Exception as e:
        print(f"  STATUS: FAILED — {e}")

    # Form 5: to_lance().to_table() for id IN listing
    print("\n--- Form 5: tbl.to_lance().to_table(filter='id IN (...)') ---")
    try:
        ids = ["id_000", "id_001", "id_002"]
        id_list = ", ".join(f"'{i}'" for i in ids)
        arrow_table = tbl.to_lance().to_table(filter=f"id IN ({id_list})")
        rows = arrow_table.to_pylist()
        print(f"  Result count: {len(rows)}")
        if rows:
            print(f"  First: {rows[0]['text'][:60]}")
            print("  STATUS: WORKS")
    except Exception as e:
        print(f"  STATUS: FAILED — {e}")

    # Form 6: filter via search().where() with id IN
    print("\n--- Form 6: tbl.search(qvec).where('id IN (...)') ---")
    try:
        ids = ["id_000", "id_001", "id_002"]
        id_list = ", ".join(f"'{i}'" for i in ids)
        rows = (
            tbl.search(qvec)
            .where(f"id IN ({id_list})")
            .limit(3)
            .to_list()
        )
        print(f"  Result count: {len(rows)}")
        print("  STATUS: WORKS")
    except Exception as e:
        print(f"  STATUS: FAILED — {e}")

    # Form 7: check if query_type="fts" works standalone
    print("\n--- Form 7: tbl.search(qtext, query_type='fts') ---")
    try:
        rows = (
            tbl.search(qtext, query_type="fts")
            .limit(3)
            .to_list()
        )
        print(f"  Result count: {len(rows)}")
        if rows:
            print(f"  First: {rows[0]['text'][:60]}")
            print("  STATUS: WORKS")
    except Exception as e:
        print(f"  STATUS: FAILED — {e}")

    return working_hybrid


def main():
    print(f"LanceDB version: {lancedb.__version__}")
    with tempfile.TemporaryDirectory() as tmpdir:
        db = lancedb.connect(tmpdir)
        tbl = make_table(db)
        print(f"Table created with {len(TEXTS)} rows, dim={DIM}")

        fts_ok = try_fts(tbl)
        best = try_forms(tbl, fts_ok)

        print("\n=== SUMMARY ===")
        print(f"FTS index: {'OK' if fts_ok else 'FAILED'}")
        print(f"Best hybrid form: {best or 'none — fall back to pure dense'}")


if __name__ == "__main__":
    main()
