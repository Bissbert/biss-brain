#!/usr/bin/env python3
"""Tests for compute_salience() in scripts/flush.py (B1 Capture stream).

Tests:
  - test_salience_heuristic_high_for_novel_long_text
  - test_salience_heuristic_low_for_short_repetitive
  - test_salience_llm_falls_back_on_parse_error
  - test_migration_idempotent
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Use fake embeddings throughout
os.environ["BISS_BRAIN_FAKE_EMBED"] = "1"


def _make_cfg(use_llm: bool = False, length_weight: float = 0.4, novelty_weight: float = 0.6) -> dict:
    return {
        "capture": {
            "salience": {
                "use_llm": use_llm,
                "heuristic": {
                    "length_weight": length_weight,
                    "novelty_weight": novelty_weight,
                },
            }
        }
    }


def _unit_vec(values: list[float]) -> list[float]:
    """Normalise a vector to unit length."""
    import math
    norm = math.sqrt(sum(v * v for v in values))
    if norm < 1e-9:
        return values
    return [v / norm for v in values]


def test_salience_heuristic_high_for_novel_long_text() -> None:
    """Long text + low max cosine with recent embeddings -> score > 6."""
    # Reload fresh to pick up env overrides
    for mod in list(sys.modules.keys()):
        if mod in ("flush", "common", "distill"):
            del sys.modules[mod]

    import flush

    # Long text saturates length_norm at 1.0 (500+ chars)
    long_text = "A" * 600
    # Target embedding: [1, 0, 0, ...]
    dim = 10
    target_vec = _unit_vec([1.0] + [0.0] * (dim - 1))
    # Recent embeddings orthogonal to target -> cosine ~ 0
    recent = [_unit_vec([0.0, 1.0] + [0.0] * (dim - 2))]

    cfg = _make_cfg(use_llm=False)
    score = flush.compute_salience(long_text, target_vec, recent, cfg)

    assert isinstance(score, int), f"score must be int, got {type(score)}"
    assert 1 <= score <= 10, f"score out of range: {score}"
    assert score > 6, (
        f"Expected score > 6 for long novel text, got {score}. "
        f"(length_norm=1.0, cosine~0 => novelty~1.0 => raw~1.0 => score~10)"
    )
    print(f"  PASS: novel long text => score={score}")


def test_salience_heuristic_low_for_short_repetitive() -> None:
    """Short text + high max cosine with recent embeddings -> score < 4."""
    for mod in list(sys.modules.keys()):
        if mod in ("flush", "common", "distill"):
            del sys.modules[mod]

    import flush

    # Short text => length_norm close to 0 (50 chars / 500 = 0.1)
    short_text = "A" * 50
    # Target embedding and recent embeddings are nearly identical -> cosine ~ 1.0
    dim = 10
    same_vec = _unit_vec([1.0] * dim)
    # Slightly perturb to avoid exact 1.0 but keep cosine very high
    near_same = _unit_vec([1.0] * (dim - 1) + [0.01])

    cfg = _make_cfg(use_llm=False)
    score = flush.compute_salience(short_text, same_vec, [near_same], cfg)

    assert isinstance(score, int), f"score must be int, got {type(score)}"
    assert 1 <= score <= 10, f"score out of range: {score}"
    assert score < 4, (
        f"Expected score < 4 for short repetitive text, got {score}. "
        f"(length_norm~0.1, cosine~1.0 => novelty~0 => raw~0.04 => score~1)"
    )
    print(f"  PASS: short repetitive text => score={score}")


def test_salience_llm_falls_back_on_parse_error() -> None:
    """When distill returns invalid JSON, compute_salience falls back to heuristic."""
    for mod in list(sys.modules.keys()):
        if mod in ("flush", "common", "distill"):
            del sys.modules[mod]

    import flush

    # Force LLM mode
    cfg = _make_cfg(use_llm=True)
    text = "some chunk text of moderate length to test fallback"
    dim = 10
    vec = _unit_vec([1.0] + [0.0] * (dim - 1))
    recent = [_unit_vec([0.5] * dim)]

    # Monkeypatch distill to return garbage JSON
    with patch("flush.ask", return_value="not valid json at all"):
        score = flush.compute_salience(text, vec, recent, cfg)

    assert isinstance(score, int), f"Fallback must return int, got {type(score)}"
    assert 1 <= score <= 10, f"Fallback score out of range: {score}"
    print(f"  PASS: LLM parse error => heuristic fallback score={score}")


def test_migration_idempotent() -> None:
    """open_meta on a fresh dir called twice must produce salience column exactly once."""
    for mod in list(sys.modules.keys()):
        if mod in ("flush", "common", "distill"):
            del sys.modules[mod]

    import common

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_cwd = Path(tmpdir) / "project"
        fake_cwd.mkdir()

        # First open — creates the DB with all v05 columns
        conn1 = common.open_meta(str(fake_cwd))
        conn1.close()

        # Second open — must be idempotent (no duplicate-column error)
        conn2 = common.open_meta(str(fake_cwd))
        cols = {row[1] for row in conn2.execute("PRAGMA table_info(chunks)").fetchall()}
        conn2.close()

        assert "salience" in cols, "salience column must exist after open_meta"

        # Confirm it appears exactly once by checking the column list length
        import sqlite3
        raw = sqlite3.connect(str(common.meta_db_path(str(fake_cwd))))
        all_cols = [row[1] for row in raw.execute("PRAGMA table_info(chunks)").fetchall()]
        raw.close()

        count = all_cols.count("salience")
        assert count == 1, f"Expected salience column exactly once, found {count} times"
        print(f"  PASS: salience column present exactly once after two open_meta calls")


if __name__ == "__main__":
    print("test_salience.py")
    test_salience_heuristic_high_for_novel_long_text()
    test_salience_heuristic_low_for_short_repetitive()
    test_salience_llm_falls_back_on_parse_error()
    test_migration_idempotent()
    print("\nAll salience tests passed.")
