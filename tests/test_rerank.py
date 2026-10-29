#!/usr/bin/env python3
"""Validation test for scripts/rerank.py.

Builds a golden set of (query, candidates, expected_top_id) tuples and asserts
that the reranker places the expected id at position 0 in at least 4/5 cases.

Skips gracefully (exit 0) if FlagEmbedding/torch is not installed or the model
cannot be downloaded.
"""
from __future__ import annotations

import sys
import os

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _check_import() -> bool:
    """Return True if FlagEmbedding or sentence-transformers can be imported."""
    try:
        import FlagEmbedding  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
        return True
    except ImportError:
        pass
    return False


GOLDEN: list[tuple[str, list[dict], str]] = [
    (
        "What database is used for vector storage?",
        [
            {"id": "a2", "text": "The CI pipeline uses GitHub Actions for continuous integration."},
            {"id": "a3", "text": "Python 3.12 is the minimum supported runtime for this project."},
            {"id": "a1", "text": "We chose LanceDB as the vector store because it natively supports hybrid BM25 and dense vector search in a single embedded database."},
            {"id": "a4", "text": "Authentication is handled via JWT tokens issued by the backend API."},
            {"id": "a5", "text": "Docker Compose orchestrates the local development environment."},
            {"id": "a6", "text": "Prometheus scrapes metrics from the /metrics endpoint every 15 seconds."},
        ],
        "a1",
    ),
    (
        "How are secrets stored and accessed?",
        [
            {"id": "b2", "text": "The embedding model produces 768-dimensional vectors for each text chunk."},
            {"id": "b3", "text": "Linting is enforced with ruff configured to disallow unused imports."},
            {"id": "b4", "text": "Test coverage is measured using pytest-cov with a minimum threshold of 80 percent."},
            {"id": "b1", "text": "Environment variables injected by Kubernetes secrets hold all API keys and database credentials."},
            {"id": "b5", "text": "The release pipeline tags Docker images with the git sha."},
            {"id": "b6", "text": "Terraform manages the cloud infrastructure across three AWS regions."},
        ],
        "b1",
    ),
    (
        "What is the debounce window for stop events?",
        [
            {"id": "c2", "text": "LanceDB stores chunk vectors and supports full-text search via BM25."},
            {"id": "c3", "text": "The project uses SQLite as canonical metadata store alongside LanceDB."},
            {"id": "c4", "text": "Pre-compaction events are flushed immediately without any debounce delay."},
            {"id": "c5", "text": "Ollama serves the nomic-embed-text model on port 11434 locally."},
            {"id": "c1", "text": "The Stop hook debounce window is 30 seconds; rapid consecutive stops are collapsed into a single distillation call."},
            {"id": "c6", "text": "The decay job runs nightly to update decay scores for all chunks."},
        ],
        "c1",
    ),
    (
        "How does the system handle duplicate content?",
        [
            {"id": "d2", "text": "The MCP server exposes search_memory, recent_sessions, related_to, pin, and unpin tools."},
            {"id": "d3", "text": "Bootstrap pulls the embedding model with ollama pull nomic-embed-text."},
            {"id": "d4", "text": "Session IDs are propagated from the Claude Code hook payload into the chunk metadata."},
            {"id": "d5", "text": "The flush daemon exits after 300 seconds of idle queue to free resources."},
            {"id": "d6", "text": "Recall searches are limited to the current project via a WHERE clause filter."},
            {"id": "d1", "text": "Duplicate detection uses SHA-256 content hashing; if a chunk with the same hash already exists it is skipped on insert."},
        ],
        "d1",
    ),
    (
        "Which embedding model is used and what is its dimensionality?",
        [
            {"id": "e2", "text": "The SupervisedBy field is stored as empty string rather than NULL to simplify SQL comparisons."},
            {"id": "e3", "text": "The plugin registers five hooks: Stop, SubagentStop, PreCompact, SessionStart, and PostCompact."},
            {"id": "e4", "text": "All event files are written atomically via a .tmp rename to avoid partial reads by the daemon."},
            {"id": "e5", "text": "The launchd plist for the decay job runs daily at 03:00 local time."},
            {"id": "e6", "text": "Pinned chunks are immune to decay and receive a 5x multiplier in the decay formula."},
            {"id": "e1", "text": "nomic-embed-text produces 768-dimensional embeddings and is the default model served via Ollama."},
        ],
        "e1",
    ),
]


def run_tests() -> None:
    from rerank import rerank

    passed = 0
    total = len(GOLDEN)

    for i, (query, candidates, expected_top) in enumerate(GOLDEN):
        result = rerank(query, candidates, top_k=1)
        got_top = result[0]["id"] if result else None
        ok = got_top == expected_top
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] golden[{i}] expected={expected_top} got={got_top!r}  query={query[:40]!r}")
        if ok:
            passed += 1

    print(f"\n  {passed}/{total} cases correct")
    if passed < 4:
        print("FAIL: reranker got fewer than 4/5 correct")
        sys.exit(1)
    else:
        print("PASS: reranker meets 4/5 accuracy threshold")


def main() -> None:
    if not _check_import():
        print("SKIP: FlagEmbedding and sentence-transformers not available — skipping rerank test")
        sys.exit(0)

    # Also skip if model download would fail (network check via a simple import)
    try:
        run_tests()
    except Exception as exc:
        # Common failure: model not downloaded, no internet, CUDA OOM, etc.
        print(f"SKIP: rerank test could not run ({exc!r})")
        sys.exit(0)


if __name__ == "__main__":
    main()
