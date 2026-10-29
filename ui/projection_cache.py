"""UMAP projection cache for the scatter panel.

Owns UMAP fit/transform + on-disk caching.  Kept separate from data.py to
avoid importing umap-learn (a heavy dependency) whenever other panels start.

Cache layout under ui/projections/:
  <project_hash>_proj.parquet   – projected points (no vectors)
  <project_hash>_umap.joblib    – serialised umap.UMAP model
  <project_hash>_proj.json      – fallback if pyarrow write fails
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup – allow `from ui.data import ...` from any working directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent      # ui/
_ROOT = _HERE.parent                          # plugin root
_SCRIPTS = _ROOT / "scripts"
for _p in (_ROOT, _SCRIPTS):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from ui.data import fetch_chunks_meta, open_lance, open_meta_ro, validate_project  # noqa: E402

log = logging.getLogger("biss-brain.projection_cache")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_TTL_SECONDS = 3600
MIN_POINTS = 5  # UMAP needs at least n_neighbors+1 points; guard at 5
PROJECTIONS_DIR = Path(__file__).parent / "projections"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_compute(project_hash: str, force: bool = False) -> dict:
    """Return projection data for a project, using the on-disk cache when fresh.

    Return shape:
      {
        "points": [{id, x, y, kind, ts, pinned, decay_score, superseded}, ...],
        "computed_ts": int,
        "model_n": int,   # number of points the UMAP model was fit on
        "stale": bool,
      }

    Logic:
      1. Cache exists and age < TTL and not force  -> load parquet and return.
      2. Cache exists but new chunks since computed_ts              -> incremental.
      3. No cache or force=True                                     -> full fit.
    """
    PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = PROJECTIONS_DIR / f"{project_hash}_proj.parquet"
    json_path = PROJECTIONS_DIR / f"{project_hash}_proj.json"
    model_path = PROJECTIONS_DIR / f"{project_hash}_umap.joblib"

    # Validate project exists (raises HTTPException if not)
    validate_project(project_hash)

    # ── 1. Load vectors + metadata ─────────────────────────────────────────
    tbl = open_lance(project_hash)
    conn = open_meta_ro(project_hash)
    try:
        meta_rows = fetch_chunks_meta(conn, "1=1", (), 100000)
    finally:
        conn.close()

    if not meta_rows:
        return {
            "points": [],
            "computed_ts": int(time.time()),
            "model_n": 0,
            "stale": False,
        }

    # Build meta lookup by id
    meta_by_id: dict[str, dict] = {r["id"]: r for r in meta_rows}

    # Pull vectors from LanceDB
    if tbl is None:
        return {
            "points": [],
            "computed_ts": int(time.time()),
            "model_n": 0,
            "stale": False,
        }

    try:
        arrow_tbl = tbl.search().select(["id", "vector"]).limit(100000).to_arrow()
        ids_from_lance: list[str] = arrow_tbl["id"].to_pylist()
        vectors_raw = arrow_tbl["vector"].to_pylist()
    except Exception as e:
        log.warning("LanceDB vector fetch failed: %s", e)
        return {
            "points": [],
            "computed_ts": int(time.time()),
            "model_n": 0,
            "stale": False,
        }

    if not ids_from_lance:
        return {
            "points": [],
            "computed_ts": int(time.time()),
            "model_n": 0,
            "stale": False,
        }

    # Apply superseded filter: keep chunk unless superseded AND not pinned
    kept_ids: list[str] = []
    kept_vectors: list[list[float]] = []
    for chunk_id, vec in zip(ids_from_lance, vectors_raw):
        meta = meta_by_id.get(chunk_id)
        if meta is None:
            continue  # not in SQLite; skip
        is_superseded = bool(meta.get("superseded_by", ""))
        is_pinned = bool(meta.get("pinned", 0))
        if is_superseded and not is_pinned:
            continue  # filtered out
        kept_ids.append(chunk_id)
        kept_vectors.append(vec)

    if len(kept_ids) < MIN_POINTS:
        return {
            "points": [],
            "computed_ts": int(time.time()),
            "model_n": 0,
            "stale": False,
        }

    vectors_np = np.array(kept_vectors, dtype=np.float32)
    now_ts = int(time.time())

    # ── 2. Determine cache state ────────────────────────────────────────────
    cached_ts: int | None = None
    cached_points: list[dict] | None = None

    if not force and (parquet_path.exists() or json_path.exists()):
        cached_ts, cached_points = _load_cache(parquet_path, json_path)

    if cached_ts is not None and cached_points is not None:
        age = now_ts - cached_ts
        if age < CACHE_TTL_SECONDS and not force:
            # Cache is fresh — return as-is
            return {
                "points": cached_points,
                "computed_ts": cached_ts,
                "model_n": len(cached_points),
                "stale": False,
            }

        # Cache exists but stale: try incremental update
        if model_path.exists() and not force:
            new_cached_ids = {p["id"] for p in cached_points}
            new_chunk_ids = [cid for cid in kept_ids if cid not in new_cached_ids]
            if new_chunk_ids:
                log.info(
                    "Incremental UMAP for %s: %d new chunks",
                    project_hash,
                    len(new_chunk_ids),
                )
                try:
                    model = _load_model(model_path)
                    idx_map = {cid: i for i, cid in enumerate(kept_ids)}
                    new_indices = [idx_map[cid] for cid in new_chunk_ids if cid in idx_map]
                    if new_indices:
                        new_vecs = vectors_np[new_indices]
                        new_coords = _transform_incremental(model, new_vecs)
                        new_points = []
                        for cid, (x, y) in zip(new_chunk_ids, new_coords):
                            meta = meta_by_id.get(cid, {})
                            new_points.append(_make_point(cid, float(x), float(y), meta))
                        all_points = cached_points + new_points
                        computed_ts = now_ts
                        _save_cache(all_points, parquet_path, json_path, computed_ts=now_ts)
                        return {
                            "points": all_points,
                            "computed_ts": computed_ts,
                            "model_n": len(cached_points),
                            "stale": False,
                        }
                except Exception as e:
                    log.warning("Incremental UMAP failed, falling back to full fit: %s", e)
            else:
                # No new chunks but cache is stale; return with stale flag
                return {
                    "points": cached_points,
                    "computed_ts": cached_ts,
                    "model_n": len(cached_points),
                    "stale": True,
                }

    # ── 3. Full fit ─────────────────────────────────────────────────────────
    log.info("Full UMAP fit for %s with %d vectors", project_hash, len(kept_ids))
    coords, model = _fit_full(vectors_np)

    points: list[dict] = []
    for cid, (x, y) in zip(kept_ids, coords):
        meta = meta_by_id.get(cid, {})
        points.append(_make_point(cid, float(x), float(y), meta))

    _save_cache(points, parquet_path, json_path, computed_ts=now_ts)
    _save_model(model, model_path)

    return {
        "points": points,
        "computed_ts": now_ts,
        "model_n": len(points),
        "stale": False,
    }


# ---------------------------------------------------------------------------
# UMAP helpers
# ---------------------------------------------------------------------------

def _fit_full(vectors: np.ndarray) -> tuple[np.ndarray, Any]:
    """Fit a UMAP model from scratch and return (coords, model)."""
    import umap  # lazy import — keeps server cold-start fast

    n = len(vectors)
    n_neighbors = min(15, n - 1)
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        random_state=42,
    )
    coords = reducer.fit_transform(vectors)
    return coords, reducer


def _transform_incremental(model: Any, new_vectors: np.ndarray) -> np.ndarray:
    """Transform new vectors using an already-fitted UMAP model."""
    return model.transform(new_vectors)


# ---------------------------------------------------------------------------
# Cache I/O helpers
# ---------------------------------------------------------------------------

def _make_point(chunk_id: str, x: float, y: float, meta: dict) -> dict:
    """Build the point dict stored in cache and returned to callers."""
    return {
        "id": chunk_id,
        "x": x,
        "y": y,
        "kind": meta.get("kind", ""),
        "ts": int(meta.get("ts", 0)),
        "pinned": int(meta.get("pinned", 0)),
        "decay_score": float(meta.get("decay_score") or 1.0),
        "superseded": bool(meta.get("superseded_by", "")),
    }


def _load_cache(
    parquet_path: Path, json_path: Path
) -> tuple[int | None, list[dict] | None]:
    """Load cached points from parquet or JSON fallback.

    Returns (computed_ts, points) or (None, None) if neither file exists
    or parsing fails.
    """
    if parquet_path.exists():
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(str(parquet_path))
            df = tbl.to_pydict()
            n = len(df.get("id", []))
            points = [
                {
                    "id": df["id"][i],
                    "x": float(df["x"][i]),
                    "y": float(df["y"][i]),
                    "kind": df["kind"][i],
                    "ts": int(df["ts"][i]),
                    "pinned": int(df["pinned"][i]),
                    "decay_score": float(df["decay_score"][i]),
                    "superseded": bool(df["superseded"][i]),
                }
                for i in range(n)
            ]
            computed_ts = int(parquet_path.stat().st_mtime)
            # Check for a sidecar timestamp file
            ts_path = parquet_path.with_suffix(".ts")
            if ts_path.exists():
                try:
                    computed_ts = int(ts_path.read_text().strip())
                except Exception:
                    pass
            return computed_ts, points
        except Exception as e:
            log.warning("Parquet load failed: %s", e)

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            computed_ts = int(json_path.stat().st_mtime)
            ts_path = json_path.with_suffix(".ts")
            if ts_path.exists():
                try:
                    computed_ts = int(ts_path.read_text().strip())
                except Exception:
                    pass
            return computed_ts, data.get("points", [])
        except Exception as e:
            log.warning("JSON cache load failed: %s", e)

    return None, None


def _save_cache(
    points: list[dict],
    parquet_path: Path,
    json_path: Path,
    computed_ts: int | None = None,
) -> None:
    """Persist points to parquet, falling back to JSON on failure."""
    now_ts = computed_ts if computed_ts is not None else int(time.time())

    # Write timestamp sidecar (shared by both formats)
    ts_path = parquet_path.with_suffix(".ts")
    ts_path.write_text(str(now_ts))

    # Attempt parquet write
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("x", pa.float64()),
            pa.field("y", pa.float64()),
            pa.field("kind", pa.string()),
            pa.field("ts", pa.int64()),
            pa.field("pinned", pa.int64()),
            pa.field("decay_score", pa.float64()),
            pa.field("superseded", pa.bool_()),
        ])
        tbl = pa.table(
            {
                "id": [p["id"] for p in points],
                "x": [p["x"] for p in points],
                "y": [p["y"] for p in points],
                "kind": [p["kind"] for p in points],
                "ts": [p["ts"] for p in points],
                "pinned": [p["pinned"] for p in points],
                "decay_score": [p["decay_score"] for p in points],
                "superseded": [p["superseded"] for p in points],
            },
            schema=schema,
        )
        pq.write_table(tbl, str(parquet_path))
        log.info("Saved %d points to %s", len(points), parquet_path)
        return
    except Exception as e:
        log.warning("Parquet write failed, falling back to JSON: %s", e)

    # JSON fallback
    try:
        json_path.write_text(json.dumps({"points": points}, ensure_ascii=False))
        # Also update ts_path to point to json
        ts_path2 = json_path.with_suffix(".ts")
        ts_path2.write_text(str(now_ts))
        log.info("Saved %d points to JSON fallback %s", len(points), json_path)
    except Exception as e2:
        log.error("JSON fallback write also failed: %s", e2)


def _save_model(model: Any, model_path: Path) -> None:
    """Serialise the fitted UMAP model with joblib."""
    try:
        import joblib
        joblib.dump(model, str(model_path))
        log.info("Saved UMAP model to %s", model_path)
    except Exception as e:
        log.warning("Model save failed (incremental updates disabled): %s", e)


def _load_model(model_path: Path) -> Any:
    """Load a previously serialised UMAP model."""
    import joblib
    return joblib.load(str(model_path))
