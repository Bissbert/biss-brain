#!/usr/bin/env python3
"""Background daemon. Polls every project's queue/, groups events into
session segments, distills each segment once, embeds, and upserts into
LanceDB + SQLite metadata.

Lifecycle:
  - capture.py spawns this on first event if not already running
  - PID file at ~/.claude/brain/flush.pid (with fcntl lock)
  - Chunks stored without a vector (embedder down) are embedded later;
    while it stays down, one probe per `backfill_interval_seconds` (_backfill)
  - Loops at 1 Hz; never exits while events wait for their segment to close
    (segment.ready); exits after `idle_exit_seconds` with nothing queued but
    segments backing off after a failure (pending.py)
"""
from __future__ import annotations

import fcntl
import json
import logging
import logging.handlers
import os
import re
import sys
import time
import uuid
from pathlib import Path

import numpy as np

import lance_filter as F
import common
from common import (
    BRAIN_ROOT,
    Event,
    content_hash,
    embed,
    fetch_by_ids,
    lancedb_dir,
    load_config,
    now_unix,
    open_meta,
    open_or_create_chunks,
    project_dir,
    project_key,
    refresh_fts,
    vector_search,
)
import pending
import provenance
import segment
import temporal
from backlinks import _add_typed_edge, write_backlinks
from distill import ask, distill
from judge import judge_conflict, judge_conflicts
from transcript import split_output

log = logging.getLogger("biss-brain.flush")

_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def _setup_logging() -> None:
    """Configure rotating file handler + stderr WARNING handler."""
    BRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = BRAIN_ROOT / "flush.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Rotating file handler — 2 MB, keep 3 backups
    fh = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(fh)

    # Console handler at WARNING so foreground triage is easy
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(ch)


_setup_logging()


# ── salience ────────────────────────────────────────────────────────────
def compute_salience(
    text: str,
    embedding: list[float],
    recent_embeddings: list[list[float]],
    cfg: dict,
) -> int:
    """Compute salience score 1-10 for a chunk.

    Heuristic (default):
        score = alpha * length_norm + beta * (1 - max_cosine_recent)
        scaled to integer 1-10.
        alpha = cfg.capture.salience.heuristic.length_weight (default 0.4)
        beta  = cfg.capture.salience.heuristic.novelty_weight (default 0.6)

        length_norm = min(len(text) / 500, 1.0)
        novelty = 1.0 - max(cosine(embedding, e) for e in recent_embeddings)
                  if recent_embeddings else 1.0

    LLM path (opt-in via cfg.capture.salience.use_llm):
        ask('salience', ...) renders prompts/score_salience.md and calls the
        LLM; parse integer 1-10 from JSON.
        On parse error falls back to heuristic.
    """
    sal_cfg = cfg.get("capture", {}).get("salience", {})
    heuristic_cfg = sal_cfg.get("heuristic", {})
    alpha = float(heuristic_cfg.get("length_weight", 0.4))
    beta = float(heuristic_cfg.get("novelty_weight", 0.6))

    def _heuristic() -> int:
        length_norm = min(len(text) / 500.0, 1.0)
        if recent_embeddings:
            vec = np.array(embedding, dtype=float)
            sims = [
                float(np.dot(vec, np.array(e, dtype=float)) /
                      max(np.linalg.norm(vec) * np.linalg.norm(np.array(e, dtype=float)), 1e-9))
                for e in recent_embeddings
            ]
            novelty = 1.0 - max(sims)
        else:
            novelty = 1.0
        raw = alpha * length_norm + beta * novelty
        return max(1, min(10, int(round(raw * 10))))

    if not sal_cfg.get("use_llm", False):
        return _heuristic()

    # LLM path — opt-in only
    try:
        # Build recent context: last 3 items truncated to 200 chars each
        recent_snippets = []
        for e_text in _recent_texts_cache.get("texts", [])[-3:]:
            recent_snippets.append(str(e_text)[:200])
        recent_context = "\n---\n".join(recent_snippets) if recent_snippets else "(none)"

        raw_response = ask("salience", chunk_text=text, recent_context=recent_context)
        if raw_response is None:
            raise ValueError("no LLM reply")
        # Parse JSON: {"score": <int>, "reason": "..."}
        parsed = json.loads(raw_response)
        score = int(parsed["score"])
        if 1 <= score <= 10:
            return score
        log.warning("LLM salience score %d out of range; falling back to heuristic", score)
    except Exception as e:  # noqa: BLE001
        log.warning("LLM salience failed (%s); falling back to heuristic", e)

    return _heuristic()


# Module-level cache for passing recent chunk texts to the LLM path
# without changing compute_salience's public signature.
_recent_texts_cache: dict[str, list[str]] = {"texts": []}


def _fetch_recent_embeddings(tbl: object, meta, proj: str, limit: int = 20) -> list[list[float]]:
    """Return the embeddings of the project's `limit` newest live chunks."""
    try:
        ids = [
            r[0]
            for r in meta.execute(
                "SELECT id FROM chunks WHERE project = ? AND superseded_by = '' "
                "ORDER BY ts DESC LIMIT ?",
                (proj, limit),
            ).fetchall()
        ]
        if not ids:
            return []
        return [list(r["vector"]) for r in fetch_by_ids(tbl, ids, limit=len(ids)) if r.get("vector") is not None]
    except Exception as e:  # noqa: BLE001
        log.debug("_fetch_recent_embeddings failed (non-fatal): %s", e)
        return []


# ── locking ─────────────────────────────────────────────────────────────
def _acquire_pidlock(wait: float = 5.0) -> Path | None:
    """Take flush.pid. Waits *wait* seconds, so a daemon that capture.py just
    stopped (an older plugin copy) can let go."""
    BRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    pid_file = BRAIN_ROOT / "flush.pid"
    fd = os.open(str(pid_file), os.O_RDWR | os.O_CREAT, 0o644)
    end = time.monotonic() + wait
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= end:
                os.close(fd)
                return None
            time.sleep(0.2)
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}".encode())
    return pid_file


# ── embedding and table ─────────────────────────────────────────────────
def _embed(text: str, cfg: dict) -> list[float]:
    """Embed a document text as a unit vector (patched in tests)."""
    return embed(text, cfg, purpose="document", timeout=60)


def _open_table(cwd: str, dim: int):
    return open_or_create_chunks(lancedb_dir(cwd), dim)


# ── reconsolidation ─────────────────────────────────────────────────────
def _cosine(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1]."""
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / denom)


_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to via was were with"
    " we our you your i".split()
)


def _terms(text: str) -> set[str]:
    """Content terms, lowercased, with a crude plural/tense fold (not and no stay terms)."""
    out = set()
    for w in re.findall(r"[a-z0-9_./-]+", text.lower()):
        w = w.strip("./-")
        if len(w) < 2 or w in _STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


def revises(old_text: str, new_text: str, min_novelty: float) -> bool:
    """True when *new_text* adds enough terms the old memory lacks to be worth a merge."""
    new = _terms(new_text)
    return bool(new) and len(new - _terms(old_text)) / len(new) >= min_novelty


def _try_reconsolidate(
    cwd: str,
    tbl: object,
    meta,
    new_text: str,
    new_embedding: list[float],
    cfg: dict,
    prov: provenance.Provenance | None = None,
) -> str | None:
    """Pre-insert merge check.

    Only candidates of the same trust class as *prov* (provenance.trust_class)
    are considered; the merged row takes the combined provenance.

    Returns the existing chunk id if a merge occurred (caller must skip insert),
    or None if no merge happened (caller proceeds with normal insert).

    Steps:
    1. Query labile chunks for this project from SQLite using idx_chunks_labile.
    2. Compute cosine between new_embedding and each candidate's vector in LanceDB.
    3. If best cosine < merge_cosine_threshold  -> return None.
    3b. HiMem-style gate: the candidate was *used* (retrieved or injected within
        labile_window_seconds, step 1) and the new text *revises* it (at least
        revise_min_novelty of its content terms are new). A restatement that
        adds nothing is folded into the candidate without an LLM call.
    4. ask('merge', ...) renders prompts/reconsolidate_merge.md and calls the LLM.
    5. Drift gate: cosine(merged_vec, midpoint(old, new)) must be within 0.12.
    6. Atomic SQLite UPDATE: text, reconsolidation_count, labile_until.
    7. If reconsolidation_count >= promote_to_lesson_after -> promote kind/round.
    8. Update LanceDB text and vector fields.
    9. Return existing chunk id.
    """
    recon_cfg = cfg.get("reconsolidation", {})
    if not recon_cfg.get("enabled", True):
        return None

    proj = project_key(cwd)
    now = now_unix()
    merge_threshold = float(recon_cfg.get("merge_cosine_threshold", 0.82))
    labile_window = int(recon_cfg.get("labile_window_seconds", 300))
    promote_after = int(recon_cfg.get("promote_to_lesson_after", 5))
    drift_max = 0.12  # hard-coded safety gate per spec (doc 04 §C.2)

    # Step 1: find labile candidates via idx_chunks_labile
    # Note: SQLite chunks table has no 'text' column — text lives in LanceDB.
    try:
        rows = meta.execute(
            "SELECT id, reconsolidation_count FROM chunks "
            "WHERE project = ? AND labile_until > ? AND superseded_by = ''",
            (proj, now),
        ).fetchall()
    except Exception as e:  # noqa: BLE001
        log.debug("reconsolidate: labile query failed (non-fatal): %s", e)
        return None

    provs = provenance.lookup(meta, proj, [r[0] for r in rows])
    cls = provenance.trust_class(prov)
    rows = [r for r in rows if provenance.trust_class(provs.get(r[0])) == cls]
    if not rows:
        return None

    # Step 2: compute cosine against each candidate's LanceDB vector + fetch text
    new_vec = np.asarray(new_embedding, dtype=float)
    best_id: str | None = None
    best_cos = 0.0
    best_text = ""
    best_count = 0
    best_vec: np.ndarray | None = None

    for (cid, ccount) in rows:
        try:
            hits = (
                tbl.search()  # type: ignore[union-attr]
                .where(F.eq("id", cid))
                .limit(1)
                .to_list()
            )
        except Exception as e:  # noqa: BLE001
            log.debug("reconsolidate: vector fetch failed for %s: %s", cid[:8], e)
            continue
        if not hits:
            continue
        raw_vec = hits[0].get("vector")
        if raw_vec is None:
            continue
        c_vec = np.asarray(raw_vec, dtype=float)
        cos = _cosine(new_vec, c_vec)
        if cos > best_cos:
            best_cos = cos
            best_id = cid
            best_text = hits[0].get("text", "")
            best_count = ccount or 0
            best_vec = c_vec

    # Step 3: threshold gate
    if best_cos < merge_threshold or best_id is None or best_vec is None:
        log.debug(
            "reconsolidate: best cosine %.3f below threshold %.3f; skipping",
            best_cos, merge_threshold,
        )
        return None

    if not revises(best_text, new_text, float(recon_cfg.get("revise_min_novelty", 0.2))):
        log.info("reconsolidate: %s already holds the new text (cosine=%.3f); folded, no merge call",
                 best_id[:8], best_cos)
        try:
            # the fold still counts as evidence: provenance combines as in a merge
            meta.execute(
                "UPDATE chunks SET last_accessed = ?, source = ?, trust = ?, taint = ? WHERE id = ? AND project = ?",
                (now, *provenance.combine([provs.get(best_id) or provenance.IMPORTED, prov or provenance.IMPORTED],
                                          cfg).row(), best_id, proj),
            )
            meta.commit()
        except Exception as e:  # noqa: BLE001
            log.warning("reconsolidate: fold update failed (%s); inserting instead", e)
            return None
        return best_id

    log.info(
        "reconsolidate: candidate %s cosine=%.3f >= %.3f; attempting merge",
        best_id[:8], best_cos, merge_threshold,
    )

    # Step 4: LLM merge via prompts/reconsolidate_merge.md
    try:
        raw_merged = ask("merge", old_text=best_text, new_text=new_text)
        if not raw_merged:
            log.warning("reconsolidate: distill returned empty; skipping merge")
            return None
        parsed = json.loads(raw_merged)
        merged_text: str = parsed["merged_text"]
    except Exception as e:  # noqa: BLE001
        log.warning("reconsolidate: merge LLM failed (%s); skipping", e)
        return None

    if not merged_text or not merged_text.strip():
        log.warning("reconsolidate: merged_text is empty; skipping")
        return None

    # Embed the merged text
    try:
        merged_vec_list = _embed(merged_text, cfg)
        merged_vec = np.asarray(merged_vec_list, dtype=float)
    except Exception as e:  # noqa: BLE001
        log.warning("reconsolidate: embed of merged text failed (%s); skipping", e)
        return None

    # Step 5: Drift gate — cosine(merged, midpoint(old, new)) must be close enough
    midpoint = (best_vec + new_vec) / 2.0
    midpoint_norm = np.linalg.norm(midpoint)
    if midpoint_norm > 1e-9:
        midpoint = midpoint / midpoint_norm
    merged_norm = np.linalg.norm(merged_vec)
    if merged_norm > 1e-9:
        merged_unit = merged_vec / merged_norm
    else:
        merged_unit = merged_vec
    drift = 1.0 - float(merged_unit @ midpoint)
    if drift > drift_max:
        log.warning(
            "reconsolidate: drift %.4f > %.4f for candidate %s; "
            "rejecting hallucinated merge",
            drift, drift_max, best_id[:8],
        )
        return None

    log.info(
        "reconsolidate: drift=%.4f OK; committing merge for chunk %s",
        drift, best_id[:8],
    )

    # Step 6 + 7: Atomic SQLite UPDATE — single transaction
    # Note: SQLite has no 'text' column; text update goes to LanceDB in Step 8.
    new_count = best_count + 1
    new_labile_until = now + labile_window
    try:
        meta.execute(
            "UPDATE chunks SET "
            "reconsolidation_count = ?, "
            "labile_until = ?, "
            "last_accessed = ?, "
            "source = ?, trust = ?, taint = ? "
            "WHERE id = ? AND project = ?",
            (new_count, new_labile_until, now,
             *provenance.combine([provs.get(best_id) or provenance.IMPORTED, prov or provenance.IMPORTED], cfg).row(),
             best_id, proj),
        )

        # Step 7: Promote to lesson after N reconsolidations (within same tx)
        if new_count >= promote_after:
            meta.execute(
                "UPDATE chunks SET kind = 'lesson', consolidation_round = 2 "
                "WHERE id = ? AND project = ?",
                (best_id, proj),
            )
            log.info(
                "reconsolidate: chunk %s promoted to kind=lesson "
                "after %d reconsolidations",
                best_id[:8], new_count,
            )

        meta.commit()
    except Exception as e:  # noqa: BLE001
        log.error("reconsolidate: SQLite update failed (%s); rolling back", e)
        try:
            meta.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None

    # Step 8: Update LanceDB text and vector fields (best-effort; non-fatal)
    try:
        tbl.update(  # type: ignore[union-attr]
            where=F.eq("id", best_id),
            values={"text": merged_text, "vector": merged_vec_list},
        )
        refresh_fts(tbl)
    except Exception as e:  # noqa: BLE001
        log.warning(
            "reconsolidate: LanceDB update for %s failed (non-fatal): %s",
            best_id[:8], e,
        )

    # Step 9: return the existing chunk id so caller skips insert
    return best_id


# ── conflict judge helper ────────────────────────────────────────────────
def _run_conflict_judge(
    cwd: str,
    proj: str,
    kind: str,
    new_id: str,
    new_text: str,
    new_vec: list[float],
    cfg: dict,
    prov: provenance.Provenance | None = None,
) -> None:
    """Judge one new chunk against its most similar live chunk (_run_conflict_judges)."""
    _run_conflict_judges(cwd, proj, kind, [(new_id, new_text, new_vec, prov)], cfg)


def _run_conflict_judges(
    cwd: str,
    proj: str,
    kind: str,
    items: list[tuple[str, str, list[float], provenance.Provenance | None]],
    cfg: dict,
) -> None:
    """Judge each new chunk of a segment against its most similar live chunk.

    *items* are (new_id, text, vector, provenance). A chunk is judged when its
    top live chunk of the same kind (not from this segment) has cosine >= 0.85
    and may be superseded by it: web or tool text never supersedes a user
    memory (DEFER without a call). Pairs whose new chunk is all trusted
    sources and pairs with web/tool text are judged in separate calls, so
    low-trust text never shares a prompt with a pair it could sway. In each
    call, one pair is judged with judge_conflict; more go to one
    judge_conflicts call, where an unparseable verdict is DEFER.

    On CONTRADICTS: the OLD chunk becomes invalid from the new chunk's event
    time (temporal.invalidate; superseded in SQLite and LanceDB, never
    deleted). On EXTENDS: an `extends` edge from new to old.
    """
    try:
        import lancedb as _ldb

        db = _ldb.connect(str(lancedb_dir(cwd)))
        if "chunks" not in db.list_tables().tables:
            return
        tbl = db.open_table("chunks")
        new_ids = [i[0] for i in items]
        gate = float(cfg.get("hygiene", {}).get("conflict_cosine", 0.85))

        # (new_id, new_text, old_id, old_text), by whether the new chunk is all trusted sources
        batches: dict[bool, list[tuple[str, str, str, str]]] = {True: [], False: []}
        meta = open_meta(cwd)
        try:
            for new_id, new_text, new_vec, prov in items:
                candidates = vector_search(
                    tbl, new_vec, F.live_in(proj, F.eq("kind", kind), F.notin("id", new_ids)), 1
                )
                if not candidates or candidates[0]["_cosine"] < gate:
                    continue
                old_id = candidates[0]["id"]
                old_prov = provenance.lookup(meta, proj, [old_id]).get(old_id)
                if not provenance.may_supersede(prov, old_prov):
                    log.info("conflict judge: %s is less trusted than %s — DEFER", new_id[:8], old_id[:8])
                    continue
                trusted = set((prov or provenance.IMPORTED).taint) <= provenance.TRUSTED_SOURCES
                batches[trusted].append((new_id, new_text, old_id, candidates[0]["text"]))
        finally:
            meta.close()

        judge_model = cfg.get("distill", {}).get("models", {}).get("conflict_judge", "haiku")
        judged: list[tuple[tuple[str, str, str, str], str | None]] = []
        for pairs in batches.values():
            if len(pairs) == 1:
                verdicts = [judge_conflict(pairs[0][1], pairs[0][3], model=judge_model)]
            elif pairs:
                verdicts = judge_conflicts([(n, o) for _, n, _, o in pairs], model=judge_model)
            else:
                continue
            judged.extend(zip(pairs, verdicts))

        superseded: set[str] = set()
        for (new_id, _, old_id, _), verdict in judged:
            if verdict == "CONTRADICTS" and old_id not in superseded:
                superseded.add(old_id)
                log.info(
                    "conflict judge: chunk %s CONTRADICTS %s — old is invalid from now on",
                    new_id[:8],
                    old_id[:8],
                )
                meta = open_meta(cwd)
                try:
                    temporal.invalidate(meta, proj, old_id, new_id)
                    meta.commit()
                finally:
                    meta.close()
                # Also update LanceDB superseded_by for consistent WHERE filtering
                try:
                    tbl.update(
                        where=F.eq("id", old_id),
                        values={"superseded_by": new_id},
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning("lancedb update superseded_by failed: %s", e)
            elif verdict == "EXTENDS":
                meta = open_meta(cwd)
                try:
                    _add_typed_edge(meta, new_id, old_id, "extends", 1.0, now_unix(), project=proj)
                    meta.commit()
                finally:
                    meta.close()
            else:
                log.debug("conflict judge: %s vs %s → %s", new_id[:8], old_id[:8], verdict)

    except Exception as e:  # noqa: BLE001
        log.warning("conflict judge step failed (non-fatal): %s", e)


# ── batch processing ────────────────────────────────────────────────────
def _scan_groups() -> dict[tuple[str, str], list[tuple[Path, Event]]]:
    """Scan all project queue dirs, group events by (project_dir, segment.group_key)."""
    groups: dict[tuple[str, str], list[tuple[Path, Event]]] = {}
    if not BRAIN_ROOT.exists():
        return groups
    for proj_dir in BRAIN_ROOT.iterdir():
        qdir = proj_dir / "queue"
        if not qdir.is_dir():
            continue
        for f in sorted(qdir.glob("*.json")):
            try:
                ev = Event.from_json(f.read_text())
            except Exception:  # noqa: BLE001
                f.unlink(missing_ok=True)
                continue
            groups.setdefault((str(proj_dir), segment.group_key(ev)), []).append((f, ev))
    return groups


def _process_batch(proj_dir: str, events: list[tuple[Path, Event]], cfg: dict) -> None:
    """One distill pass over a ready segment (segment.plan).

    Windows are distilled and stored in order; after each one the cursor
    moves past it and the events it finished are deleted. The first failure
    ends the pass and schedules a retry (pending.py), from that window on.
    A window whose reply stays invalid `skip_invalid_after` times is skipped
    and recorded (pending.skip): it would otherwise block the session for good.
    A pass that finishes nothing backs off too, so no bug can pin the daemon.
    """
    if not events:
        return
    cwd = events[0][1].cwd
    proj = project_key(cwd)
    gk = segment.group_key(events[0][1])
    pas = segment.plan(events, cfg)
    kind = pas.kind
    log.info("flush %s/%s n=%d windows=%d", proj, kind, len(pas.events), len(pas.windows))

    pdir = Path(proj_dir)
    finished = 0

    def finish(offset: int | None, final: bool = False) -> None:
        nonlocal finished
        for f, _ in pas.commit(offset, final):
            if f.exists():
                f.unlink(missing_ok=True)
                finished += 1

    stored: list[tuple[str, str, list[float], str]] = []
    error, at = None, pas.start
    for w in pas.windows:
        got, error = _store_window(cwd, proj, kind, pas.events[-1][1], w, cfg)
        if error:
            break
        stored += got
        if w.end is not None:
            finish(w.end)
            at = w.end
    if error:
        st = pending.fail(pdir, gk, kind, error, cfg, at=at if pas.path else None)
        if error == "invalid distill reply" and st["stuck"] >= int(cfg["distill"].get("skip_invalid_after", 5)):
            pending.skip(pdir, gk, pas.path, at if pas.path else None, w.end, error)
            log.warning("skipping window %s:%s-%s of %s/%s after %d invalid replies", pas.path, at, w.end,
                        proj, kind, st["stuck"])
            finish(w.end, final=w is pas.windows[-1])
            pending.clear(pdir, gk)
        else:
            log.warning("%s for %s/%s (attempt %d); retry after %ds", error, proj, kind,
                        st["attempts"], st["next_try"] - now_unix())
    else:
        finish(pas.end, final=True)
        if finished or (pas.path and pas.end != pas.start):
            pending.clear(pdir, gk)
        else:
            pending.fail(pdir, gk, kind, "no progress", cfg)
            log.warning("no progress for %s/%s; backing off", proj, kind)

    # ── conflict judge (post-insert, one call per pass) ──────────────────
    # A quarantined chunk never supersedes anything.
    if stored and cfg.get("hygiene", {}).get("conflict_judge_enabled", True):
        meta = open_meta(cwd)
        try:
            provs = provenance.lookup(meta, proj, [r[0] for r in stored])
        finally:
            meta.close()
        judged = [(cid, t, v, provs.get(cid)) for cid, t, v, k in stored if k != "quarantine"]
        if judged:
            _run_conflict_judges(cwd, proj, kind, judged, cfg)


def _store_window(
    cwd: str, proj: str, kind: str, last: Event, w: segment.Window, cfg: dict
) -> tuple[list[tuple[str, str, list[float], str]], str | None]:
    """Distill, embed and store one window: (stored chunks, None) or ([], error).

    Trusted (user/assistant/file) and low-trust (tool/web) facts become
    separate chunks, so web content never rides on user trust. One distill
    call returns every bullet with its source and salience. Outside
    passthrough, a reply that is not the JSON the prompt asked for is a
    failure, never "no findings".
    """
    summary = distill(kind, w.text)
    if not summary:
        return [], "distill failed"
    out = split_output(summary, w.present, strict=cfg["distill"]["backend"] != "passthrough")
    if out is None:
        return [], "invalid distill reply"
    groups, salience = out
    if not any(groups.values()):
        log.info("no findings for %s/%s", proj, kind)
        return [], None
    parts = [("\n".join(t for _, t in lines), provenance.derive((s for s, _ in lines), cfg), salience.get(g))
             for g, lines in groups.items() if lines]

    meta = open_meta(cwd)
    try:
        prepared = []
        for text, prov, sal in parts:
            h = content_hash(text)
            cur = meta.execute("SELECT id FROM chunks WHERE content_hash = ?", (h,)).fetchone()
            if cur:
                log.info("dedup hit %s/%s id=%s", proj, kind, cur[0])
                continue
            try:
                vec = _embed(text, cfg)
            except Exception as e:  # noqa: BLE001
                if not cfg["embed"].get("store_without_vector", True):
                    log.error("embed failed: %s", e)
                    return [], "embed failed"  # nothing of this window stored yet
                # Keep the paid distill: store now, embed later (_backfill).
                log.warning("embed failed (%s); storing without a vector", e)
                vec = None
            prepared.append((text, h, prov, vec, sal))
        stored = [
            r for r in (_store_chunk(cwd, proj, kind, last, meta, t, h, pv, v, cfg, salience=sal)
                        for t, h, pv, v, sal in prepared) if r
        ]
    finally:
        meta.close()
    if any(v is None for _, _, v, _ in stored):
        pending.mark_embed(project_dir(cwd), cwd)
    stored = [r for r in stored if r[2] is not None]  # backlinks and judge wait for the vector
    for chunk_id, _, vec, _ in stored:
        write_backlinks(cwd, proj, chunk_id, vec, cfg)
    return stored, None


BACKFILL_BATCH = 20
BACKFILL_MAX = 200  # rows per call; the loop calls again on the next tick


def _backfill(cfg: dict) -> str:
    """Embed chunks stored without a vector, then link and judge them.

    Returns "down" when the embedder failed (wait an interval), "more" when
    rows remain (call again at once), or "done". A row that fails while a
    probe text embeds fine is bad, not the embedder: it is listed in
    pending/embed_skip.json and keeps its NULL vector (keyword search only).
    Reconsolidation is skipped for these chunks, so a near-duplicate may stay
    until dream merges it.
    """
    budget = BACKFILL_MAX
    for pdir, cwd in pending.embed_marks(BRAIN_ROOT):
        if not cwd:
            pending.clear_embed(pdir)
            continue
        tbl = open_or_create_chunks(pdir / "lancedb", cfg["embed"]["dim"])
        while True:
            bad = pending.embed_skips(pdir)
            where = "vector IS NULL" + (f" AND NOT ({F.isin('id', bad)})" if bad else "")
            lim = min(BACKFILL_BATCH, budget)
            rows = tbl.search().where(where).limit(lim).to_list()
            done: list[tuple[dict, list[float]]] = []
            down = False
            for r in rows:
                try:
                    vec = _embed(r["text"], cfg)
                except Exception as e:  # noqa: BLE001
                    try:
                        _embed("probe", cfg)
                    except Exception:  # noqa: BLE001
                        log.info("backfill: embedder still down (%s)", e)
                        down = True
                        break
                    log.warning("backfill: chunk %s cannot be embedded (%s); left without a vector",
                                r["id"][:8], type(e).__name__)
                    pending.skip_embed(pdir, r["id"])
                    continue
                try:
                    tbl.update(where=F.eq("id", r["id"]), values={"vector": vec})
                except Exception as e:  # noqa: BLE001
                    # A write conflict (dream) is not the row's fault: retry next interval.
                    log.warning("backfill: update failed (%s); retrying later", type(e).__name__)
                    down = True
                    break
                done.append((r, vec))
            if done:
                refresh_fts(tbl)  # an update moves the row out of the FTS index
                _link_and_judge(cwd, done, cfg)
                log.info("backfill: embedded %d chunk(s) in %s", len(done), pdir.name)
            budget -= len(rows)
            if down:
                return "down"
            if len(rows) < lim:
                pending.clear_embed(pdir)
                break
            if budget <= 0:
                return "more"
    return "done"


def _link_and_judge(cwd: str, done: list[tuple[dict, list[float]]], cfg: dict) -> None:
    proj = done[0][0]["project"]
    live = [(r, v) for r, v in done if r["kind"] != "quarantine" and not r.get("superseded_by")]
    for r, vec in live:
        write_backlinks(cwd, proj, r["id"], vec, cfg)
    if not live or not cfg.get("hygiene", {}).get("conflict_judge_enabled", True):
        return
    meta = open_meta(cwd)
    try:
        provs = provenance.lookup(meta, proj, [r["id"] for r, _ in live])
    finally:
        meta.close()
    by_kind: dict[str, list] = {}
    for r, vec in live:
        by_kind.setdefault(r["kind"], []).append((r["id"], r["text"], vec, provs.get(r["id"])))
    for kind, items in by_kind.items():
        _run_conflict_judges(cwd, proj, kind, items, cfg)


def _store_chunk(
    cwd: str, proj: str, kind: str, last: Event, meta, text: str, h: str,
    prov: provenance.Provenance, vec: list[float] | None, cfg: dict, salience: int | None = None,
) -> tuple[str, str, list[float] | None, str] | None:
    """Salience, reconsolidation and insert for one provenance group.

    *salience* is the distiller's score for the group; without one the
    length/novelty heuristic (compute_salience) is used.

    Low-trust text that reads like an instruction is stored as
    kind=quarantine and never merged. Returns (chunk_id, text, vec, kind) for
    a new chunk, None when merged into an existing one.
    """
    kind = provenance.chunk_kind(kind, text, prov, cfg)
    chunk_id = uuid.uuid4().hex
    ts = now_unix()
    # Event time: when the session produced it, not when the queue drained.
    valid_from = min(int(last.enqueued_ts or ts), ts)
    tbl = _open_table(cwd, cfg["embed"]["dim"])

    # ── salience (pre-insert) ────────────────────────────────────────────
    if vec is None:
        salience = 5 if salience is None else salience  # no vector, no novelty score
    elif salience is None:
        recent_embeddings = _fetch_recent_embeddings(tbl, meta, proj, limit=20)
        salience = compute_salience(text, vec, recent_embeddings, cfg)
    log.debug("salience=%d for chunk %s", salience, chunk_id[:8])

    # ── reconsolidation (pre-insert) ─────────────────────────────────────
    # If a labile candidate of the same trust class is close enough, merge
    # in-place and skip insert.
    merged_id = None if kind == "quarantine" or vec is None else _try_reconsolidate(cwd, tbl, meta, text, vec, cfg, prov)
    if merged_id is not None:
        log.info("reconsolidate: merged into existing chunk %s; skipping insert for %s/%s",
                 merged_id[:8], proj, kind)
        return None

    tbl.add(
        [
            {
                "id": chunk_id,
                "project": proj,
                "session_id": last.session_id or "",
                "agent_id": last.agent_id or "",
                "kind": kind,
                "ts": ts,
                "text": text,
                "content_hash": h,
                "superseded_by": "",
                "vector": vec,
            }
        ]
    )
    refresh_fts(tbl)
    meta.execute(
        "INSERT INTO chunks (id, project, session_id, agent_id, kind, ts, content_hash, superseded_by, "
        "last_accessed, salience, source, trust, taint, valid_from) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?)",
        (chunk_id, proj, last.session_id, last.agent_id, kind, ts, h, ts, salience, *prov.row(), valid_from),
    )
    meta.commit()
    log.info("indexed chunk %s (%s/%s, %s)", chunk_id[:8], proj, kind, provenance.label(prov))
    return chunk_id, text, vec, kind


# ── main loop ───────────────────────────────────────────────────────────
def main() -> int:
    if _acquire_pidlock() is None:
        log.info("another flush daemon is running; exiting")
        return 0

    cfg = load_config()
    common.claim_store(cfg)
    bad = common.store_mismatch(cfg)
    if bad:
        # Queued events stay put until the config matches the store again.
        log.error("embed settings differ from the store (%s); not indexing. Fix: %s", bad, common.MISMATCH_HINT)
        return 0
    idle_seconds = 0
    last_backfill = 0.0

    while True:
        if time.monotonic() - last_backfill >= float(cfg["embed"].get("backfill_interval_seconds", 60)):
            # Throttle only while the embedder is down; drain a backlog at once.
            try:
                status = _backfill(cfg)
            except Exception:  # noqa: BLE001
                log.exception("backfill failed")
                status = "down"
            last_backfill = 0.0 if status == "more" else time.monotonic()
        groups = _scan_groups()
        alive = False
        for (proj_dir, gk), events in groups.items():
            if pending.waiting(Path(proj_dir), gk):
                continue  # backing off after a failure
            alive = True
            if segment.ready(events, cfg):
                try:
                    _process_batch(proj_dir, events, cfg)
                except Exception as e:  # noqa: BLE001
                    log.exception("batch failed")
                    pending.fail(Path(proj_dir), gk, segment.kind(events), f"error: {type(e).__name__}", cfg)

        # Waiting events keep the daemon alive until their segment closes. A
        # segment that is only backing off does not: the next capture respawns
        # the daemon, and the nightly dream kicks it.
        if alive:
            idle_seconds = 0
        else:
            idle_seconds += 1
            if idle_seconds > cfg["debounce"]["idle_exit_seconds"]:
                log.info("idle %ds; exiting", idle_seconds)
                return 0
        time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
