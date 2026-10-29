"""Pure retrieval: candidate search + ranking, no side effects.

Every read path (recall CLI, MCP search_memory, UserPromptSubmit inject) and
the eval harness call these functions, so what is measured is what ships.
Side effects (counts, STDP, RIF, labile stamping) live in effects.py and run
on the rows returned here.

Ranking: a candidate pool (top_k × pool_factor, at least pool_min) comes from
hybrid BM25 + dense search (RRF) or dense search; an optional cross-encoder
reranks it; the final score is

    relevance / max(relevance in pool)  ×  decay_score ** decay_weight
                                        ×  provenance.rank_multiplier

so decay can only break near-ties and never reorders by recency alone. (v0.6
sorted by decay_score alone, which cost 0.2–0.3 MRR in the Phase 0 eval.)
Ranked rows carry `_prov` and `_source` (display label). Quarantined rows are
excluded unless asked for by kind; prompt_inject drops rows whose taint is in
`provenance.no_inject_sources` and rows with no meta row at all.

Time (temporal.py): by default only live rows are searched. With `as_of` the
pool includes contradicted rows and keeps those valid at that time; with
`include_history` each result is folded onto the live memory that replaced it
and carries `_history` (the replaced memories, newest first). Ranked rows
carry `_valid_from` / `_invalid_at`.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Any

import lance_filter as F
import provenance
import temporal
from common import fetch_by_ids, hybrid_search, vector_search


def base_where(proj: str, kind: str | None = None, since_ts: int | None = None, live: bool = True) -> str:
    """Lance filter for a query; live=False also admits superseded rows
    (temporal queries filter them on the meta side)."""
    return (F.live_in if live else F.in_project)(
        proj,
        F.NOT_HYPOTHESIS,
        F.NOT_QUARANTINE if kind != "quarantine" else None,
        F.eq("kind", kind) if kind else None,
        F.ge("ts", int(since_ts)) if since_ts is not None else None,
    )


def search(tbl: Any, qvec: list[float], qtext: str, where_sql: str, limit: int, cfg: dict) -> list[dict]:
    if cfg["retrieve"].get("hybrid", True):
        return hybrid_search(tbl, qvec, qtext, where_sql, limit)
    return vector_search(tbl, qvec, where_sql, limit)


def pool_size(top_k: int, cfg: dict) -> int:
    r = cfg["retrieve"]
    return max(top_k * int(r.get("pool_factor", 5)), int(r.get("pool_min", 25)))


def candidates(
    tbl: Any,
    qvec: list[float],
    qtext: str,
    where_sql: str,
    top_k: int,
    cfg: dict,
    summary_first: bool = False,
) -> list[dict]:
    """Fetch the candidate pool; with summary_first, arcs/daily summaries come first."""
    fetch_k = pool_size(top_k, cfg)
    results: list[dict] = []
    if summary_first:
        try:
            results = search(
                tbl, qvec, qtext, F.and_(where_sql, F.isin("kind", ["arc", "daily_summary"])), fetch_k, cfg
            )
        except Exception:  # noqa: BLE001
            results = []
        for r in results:
            r["_summary"] = True
    if len(results) < top_k:
        seen = {r["id"] for r in results}
        for r in search(tbl, qvec, qtext, where_sql, fetch_k, cfg):
            if r["id"] not in seen:
                results.append(r)
                seen.add(r["id"])
    return results[:fetch_k]


def decay_scores(meta: sqlite3.Connection | None, proj: str, ids: list[str]) -> dict[str, float]:
    if meta is None or not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = meta.execute(
        f"SELECT id, decay_score FROM chunks WHERE project = ? AND id IN ({placeholders})",
        (proj, *ids),
    ).fetchall()
    return {cid: (d if d is not None else 1.0) for cid, d in rows}


def relevance(row: dict) -> float:
    """The row's query relevance: rerank score, else RRF score, else cosine."""
    for key in ("_rerank_score", "_relevance_score"):
        if row.get(key) is not None:
            return float(row[key])
    return float(row.get("_cosine", 0.0))


def rank(
    qtext: str,
    rows: list[dict],
    meta: sqlite3.Connection | None,
    proj: str,
    top_k: int,
    cfg: dict,
) -> list[dict]:
    """Optional rerank, then relevance × decay^decay_weight × trust multiplier; stable on ties."""
    if not rows:
        return []
    if cfg["retrieve"].get("rerank"):
        from rerank import rerank  # noqa: PLC0415

        rows = rerank(qtext, rows, top_k=len(rows))
    w = float(cfg["retrieve"].get("decay_weight", 0.0))
    rels = [relevance(r) for r in rows]
    top = max(rels) or 1.0
    decay = decay_scores(meta, proj, [r["id"] for r in rows]) if w else {}
    provs = provenance.lookup(meta, proj, [r["id"] for r in rows])
    valid = temporal.validity(meta, proj, [r["id"] for r in rows])
    for r, rel in zip(rows, rels):
        p = r["_prov"] = provs.get(r["id"])
        r["_source"] = provenance.label(p)
        v = valid.get(r["id"]) or temporal.from_lance(r)
        r["_valid_from"], r["_invalid_at"] = v.valid_from, v.invalid_at
        score = (rel / top if top > 0 else 0.0) * (max(decay.get(r["id"], 1.0), 1e-6) ** w)
        score *= provenance.rank_multiplier(p, cfg)
        # summary_first: summaries stay ahead of raw chunks
        r["_score"] = score + (1.0 if r.get("_summary") else 0.0)
    return sorted(rows, key=lambda r: r["_score"], reverse=True)[:top_k]


def retrieve(
    tbl: Any,
    meta: sqlite3.Connection | None,
    proj: str,
    qtext: str,
    qvec: list[float],
    cfg: dict,
    top_k: int = 5,
    kind: str | None = None,
    since_ts: int | None = None,
    summary_first: bool = False,
    as_of: int | None = None,
    include_history: bool = False,
) -> list[dict]:
    timed = as_of is not None or include_history
    where_sql = base_where(proj, kind, since_ts, live=not timed)
    rows = candidates(tbl, qvec, qtext, where_sql, top_k, cfg, summary_first=summary_first)
    if not timed:
        return rank(qtext, rows, meta, proj, top_k, cfg)
    rows = select_valid(meta, proj, rows, as_of)
    ranked = rank(qtext, rows, meta, proj, len(rows) if include_history else top_k, cfg)
    if include_history:
        ranked = fold_history(tbl, meta, proj, ranked, as_of, kind)[:top_k]
    return ranked


def select_valid(meta: sqlite3.Connection | None, proj: str, rows: list[dict], as_of: int | None) -> list[dict]:
    """Drop forgotten rows; with as_of, keep only rows valid at that time."""
    valid = temporal.validity(meta, proj, [r["id"] for r in rows])
    out = []
    for r in rows:
        v = valid.get(r["id"]) or temporal.from_lance(r)
        if v.forgotten or (as_of is not None and not v.valid_at(as_of)):
            continue
        out.append(r)
    return out


def fold_history(
    tbl: Any,
    meta: sqlite3.Connection | None,
    proj: str,
    ranked: list[dict],
    as_of: int | None,
    kind: str | None = None,
) -> list[dict]:
    """Fold each ranked row onto its head and attach the chain it replaced.

    The head is the live memory at the end of the row's contradiction chain
    (with as_of, every row is already the version valid then, so it is its own
    head). A head keeps the best score of its chain; heads not in the pool are
    fetched by id.
    """
    if meta is None:
        return ranked
    by_id = {r["id"]: r for r in ranked}
    order: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for r in ranked:
        head = r["id"] if as_of is not None or not r.get("_invalid_at") else temporal.head_of(meta, proj, r["id"])
        if head is None or head in seen:
            continue
        seen.add(head)
        order.append((head, r))
    missing = [h for h, _ in order if h not in by_id]
    by_id.update({r["id"]: r for r in fetch_by_ids(tbl, missing)})
    valid = temporal.validity(meta, proj, missing)
    blocked = {"hypothesis", "quarantine"} - {kind}
    chains = {h: temporal.predecessors(meta, proj, h) for h, _ in order}
    texts = {r["id"]: r for r in fetch_by_ids(tbl, [p[0] for c in chains.values() for p in c])}
    out = []
    for head, best in order:
        row = by_id.get(head)
        if row is None or row.get("kind") in blocked:
            continue
        if head in valid:
            row["_valid_from"], row["_invalid_at"] = valid[head].valid_from, valid[head].invalid_at
        row.setdefault("_score", best["_score"])
        row.setdefault("_prov", best.get("_prov"))
        row.setdefault("_source", best.get("_source", "imported"))
        row["_history"] = [
            {"id": cid, "valid_from": vf, "invalid_at": inv, "text": texts.get(cid, {}).get("text", "")}
            for cid, vf, inv in chains[head]
        ]
        out.append(row)
    return out


# ── proactive injection (UserPromptSubmit) ───────────────────────────────

# A prompt made only of these words carries no topic ("thanks, looks good",
# "yes do it", "continue please"): retrieval would only find noise.
_ACK_WORDS = frozenset(
    "yes y yep yeah sure ok okay k no n nope thanks thank you thx ty please pls lgtm continue go "
    "on ahead do it proceed next done great nice cool perfect good fine looks that works again "
    "keep going sounds right correct exactly agreed carry".split()
)
_WORD = re.compile(r"[\w'-]+")


def trivial_prompt(text: str, cfg: dict) -> bool:
    """True for prompts not worth a retrieval: slash commands, bash mode, acks, very short text."""
    t = text.strip()
    if not t or t[0] in "/!":
        return True
    words = [w.lower() for w in _WORD.findall(t)]
    if len(words) < int(cfg["retrieve"].get("inject_min_words", 3)):
        return True
    return all(w in _ACK_WORDS for w in words)


def fts_search(tbl: Any, qtext: str, where_sql: str, limit: int) -> list[dict]:
    """BM25-only search, for when the query could not be embedded in time.

    Rows carry `_bm25` and `_relevance_score` (= BM25) and no `_cosine`.
    """
    q = tbl.search(qtext, query_type="fts")
    if where_sql:
        q = q.where(where_sql)
    rows = q.limit(limit).to_list()
    for r in rows:
        r["_bm25"] = float(r.pop("_score", 0.0) or 0.0)
        r["_relevance_score"] = r["_bm25"]
    return rows


def inject_candidates(
    tbl: Any,
    meta: sqlite3.Connection | None,
    proj: str,
    qtext: str,
    qvec: list[float] | None,
    cfg: dict,
    exclude: frozenset[str] | set[str] = frozenset(),
) -> list[dict]:
    """Ranked, injectable rows not already in context, before the gate.

    `exclude` holds ids already injected this session (the ledger). With
    qvec=None the pool is BM25-only. At most inject_max_k rows are returned,
    each with `_bm25` (0 when the text has no query term) for the gate.
    """
    r = cfg["retrieve"]
    max_k = int(r.get("inject_max_k", 3))
    want = pool_size(max_k, cfg)
    try:
        lexical = fts_search(tbl, qtext, base_where(proj), want)
    except Exception:  # noqa: BLE001 — no FTS index yet: the cosine gate still works
        lexical = []
    if qvec is None:
        rows = rank(qtext, select_valid(meta, proj, lexical, None), meta, proj, want, cfg)
    else:
        rows = retrieve(tbl, meta, proj, qtext, qvec, cfg, top_k=want)
        bm25 = {row["id"]: row["_bm25"] for row in lexical}
        for row in rows:
            row["_bm25"] = bm25.get(row["id"], 0.0)
    out = [row for row in rows if row["id"] not in exclude and provenance.injectable(row.get("_prov"), cfg)]
    return out[:max_k]


def inject_gate(rows: list[dict], cfg: dict, vector: bool = True) -> list[dict]:
    """Keep the candidates confident enough to inject unasked.

    A row passes on either signal: cosine >= inject_min_cosine (only with a
    query vector) or BM25 >= inject_min_bm25 (0 disables the lexical side).
    On the v0.7 eval BM25 separates relevant from unrelated memories far
    better than cosine; the cosine side catches paraphrases with no shared
    term. Then top inject_top_k in rank order, then the char budget.
    """
    r = cfg["retrieve"]
    min_cos = float(r.get("inject_min_cosine", 0.5))
    min_bm25 = float(r.get("inject_min_bm25", 0.0))

    def passes(row: dict) -> bool:
        if vector and row.get("_cosine", 0.0) >= min_cos:
            return True
        return min_bm25 > 0 and row.get("_bm25", 0.0) >= min_bm25

    rows = [row for row in rows if passes(row)][: int(r.get("inject_top_k", 2))]
    budget = int(r.get("inject_max_chars", 6000))
    out, used = [], 0
    for row in rows:
        n = len(row.get("text", ""))
        if out and used + n > budget:
            break
        if not out and n > budget:
            row = {**row, "text": row["text"][:budget].rstrip() + " …"}
            n = budget
        out.append(row)
        used += n
    return out


def prompt_inject(
    tbl: Any,
    meta: sqlite3.Connection | None,
    proj: str,
    qtext: str,
    qvec: list[float] | None,
    cfg: dict,
    exclude: frozenset[str] | set[str] = frozenset(),
) -> list[dict]:
    """UserPromptSubmit selection: trivial-prompt skip, candidates, then the gate.

    Every caller (hook and eval) goes through here, so the eval measures the
    shipped behaviour. The gate uses absolute cosine, which hybrid and dense
    rows both carry (v0.6 read `_distance`, absent on hybrid rows, and never
    injected). Non-injectable rows are dropped before the cut, so they can't
    take a slot.
    """
    if trivial_prompt(qtext, cfg):
        return []
    rows = inject_candidates(tbl, meta, proj, qtext, qvec, cfg, exclude)
    return inject_gate(rows, cfg, vector=qvec is not None)
