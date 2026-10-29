#!/usr/bin/env python3
"""Decay job (legacy entry point). Now part of the nightly dream pass — Stage 1c.

decay_score = exp(-Δdays / half_life) * (1 + usage_boost * retrieval_count) * (5 if pinned else 1)

This file is preserved as a shim so installations whose launchd plist still
points at `decay.py` keep working. New installs schedule `dream.py` directly.

Idempotent. Does NOT delete anything.
"""
from __future__ import annotations

import logging
import math
import sqlite3
import sys
import time
from pathlib import Path

# Make scripts/ importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BRAIN_ROOT, load_config

log = logging.getLogger("biss-brain.decay")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def _compute_decay(ts: int, retrieval_count: int, pinned: int, half_life: float,
                   usage_boost: float = 0.0) -> float:
    """Compute decay score for a single chunk.

    decay_score = exp(-Δdays / half_life) * (1 + usage_boost * retrieval_count) * (5 if pinned else 1)
    """
    now = time.time()
    delta_days = (now - ts) / 86400.0
    base = math.exp(-delta_days / half_life)
    retrieval_boost = 1.0 + usage_boost * retrieval_count
    pin_boost = 5.0 if pinned else 1.0
    return base * retrieval_boost * pin_boost


def run_decay(cfg: dict | None = None) -> int:
    """Recompute decay scores across all project databases. Returns number of chunks updated."""
    if cfg is None:
        cfg = load_config()

    half_life = float(cfg.get("hygiene", {}).get("decay_half_life_days", 30))
    usage_boost = float(cfg.get("hygiene", {}).get("decay_usage_boost", 0.0))
    total_updated = 0

    if not BRAIN_ROOT.exists():
        log.info("brain root does not exist; nothing to decay")
        return 0

    for proj_dir in BRAIN_ROOT.iterdir():
        if not proj_dir.is_dir():
            continue
        meta_path = proj_dir / "meta.sqlite"
        if not meta_path.exists():
            continue

        try:
            # Open SQLite directly — do not use open_meta() which re-hashes the path
            conn = sqlite3.connect(str(meta_path), timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")

            # Migration: add decay_score column if missing
            existing_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
            }
            if "decay_score" not in existing_cols:
                conn.execute(
                    "ALTER TABLE chunks ADD COLUMN decay_score REAL NOT NULL DEFAULT 1.0"
                )
                conn.commit()

            rows = conn.execute(
                "SELECT id, ts, retrieval_count, pinned FROM chunks"
            ).fetchall()

            if not rows:
                conn.close()
                continue

            updates = []
            for row_id, ts, retrieval_count, pinned in rows:
                score = _compute_decay(ts, retrieval_count or 0, pinned or 0, half_life, usage_boost)
                updates.append((score, row_id))

            conn.executemany(
                "UPDATE chunks SET decay_score = ? WHERE id = ?", updates
            )
            conn.commit()
            conn.close()

            log.info(
                "decay: updated %d chunks in %s (half_life=%.1f days)",
                len(updates),
                proj_dir.name,
                half_life,
            )
            total_updated += len(updates)

        except Exception as e:  # noqa: BLE001
            log.error("decay failed for %s: %s", proj_dir, e)

    log.info("decay run complete: %d total chunks updated", total_updated)
    return total_updated


def main() -> int:
    cfg = load_config()
    run_decay(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
