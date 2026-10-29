# Implementation Research: Orchestration, DMN, Observability

> Synthesis layer over `03-impl-research-retrieval.md` (sparse signatures,
> Hopfield blend, STDP queue, reconsolidation, RIF) and
> `04-impl-research-llm-stages.md` (salience, schema extraction,
> reconsolidation merge prompt, REM, cross-project conflict judge).
> This document covers what those two leave open: the **glue** — the
> dream-cycle state machine, the DMN daemon's idle detection on macOS,
> the cross-project promotion ledger, observability hooks, performance
> budget, and the migration sequencing inside v0.5. All recommendations
> reference real symbols audited from the existing biss-brain codebase
> (see audit summary in §1).

---

## 1. What's Already in the Codebase (Audit Summary)

The Explore-agent audit of `scripts/common.py`, `scripts/flush.py`,
`scripts/dream.py`, `scripts/recall.py`, `scripts/inject_context.py`,
`scripts/backlinks.py`, `scripts/decay.py`, `mcp/server.py`, and
`prompts/*.md` produced these load-bearing facts the v2 build must
respect:

**Helpers to reuse (don't reinvent):**

- `load_config()` (`common.py:92`) — TOML deep-merge over
  `DEFAULT_CONFIG`. New v2 keys land here.
- `project_key(cwd)` / `project_dir(cwd)` (`common.py:101`, `:106`) —
  12-char SHA-256 hash; the cross-project layer's `__global__`
  identifier follows the same primitive.
- `open_meta(cwd)` (`common.py:126`) — WAL-mode SQLite with
  idempotent migrations. v2 schema additions slot in here.
- `hybrid_search(tbl, qvec, qtext, where_sql, top_k)` (`common.py:307`)
  — BM25+dense fallback. v2 retrieval pipeline calls this as step 1
  before sparse-Jaccard, graph expand, Hopfield.
- `_compute_decay(ts, retrieval_count, pinned, half_life)` (`decay.py:32`,
  also at `dream.py:426`) — the existing decay formula. v0.8 modifies
  this, v0.5 leaves it alone.
- `judge_conflict(new_text, old_text, model)` (`judge.py:28`) — already
  returns three verdicts (`CONTRADICTS | EXTENDS | UNRELATED`). The
  cross-project update only adds the fourth verdict (`DIVERGES`) and
  two prompt placeholders.
- `write_backlinks(...)` / `fetch_related(...)` / `_parse_ids` /
  `_serialize_ids` / `_cap` (`backlinks.py:42, :111, :28-39`) — the
  existing related_ids codec. The typed-edges adapter wraps this.
- `distill(kind, text, model)` (`distill.py:111`) — backend dispatch to
  `claude_code | ollama | passthrough`. Every new prompt
  (reconsolidate-merge, schema-extract, REM-recombine, salience-score)
  must go through this dispatcher, not call subprocess directly.
- `_load_prompt(kind)` / `_render(prompt, **vars)` (`distill.py:31-43`)
  — the `template.replace("{key}", value)` idiom. v2 prompts are
  loaded the same way.

**Schema columns already present** (relevant to v2):

`decay_score`, `retrieval_count`, `last_accessed`, `consolidation_round`
(0/1/2), `parent_id`, `evidence_ids`, `last_dream_ts`. The v0.5
migration is purely additive on top of this — `salience`, `labile_until`,
`reconsolidation_count`, `co_retrieval_count`, `cross_links`,
`cross_eligible`, `retrieval_count_local`.

**Bug-class issue surfaced:** `_fake_embed` is duplicated across
`flush.py:92`, `recall.py:23`, `dream.py:173`, `inject_context.py:29`.
v0.5 is the right time to centralise it in `common.py` — the migration
risk is low and every team will need it for tests.

---

## 2. Dream Cycle as a State Machine

### 2.1 State diagram

```
                                ┌──────────┐
                                │   IDLE   │
                                └─────┬────┘
                                      │ trigger:
                                      │   - Stop event (debounced)
                                      │   - DMN tick
                                      │   - manual /dream
                                      ▼
                                ┌──────────┐
                                │  NREM-1  │ — backlink refresh,
                                │ (existing│   dedup, same-project
                                │ stage1a/b)│   conflict sweep
                                └─────┬────┘
                                      │ exit if no new chunks
                                      ▼
                                ┌──────────┐
                                │  NREM-2  │ — sparse-sig compute
                                │ (v0.5/0.6)│   (v0.6); STDP edge
                                │           │   drain (v0.5);
                                │           │   RIF apply (v0.5)
                                └─────┬────┘
                                      │ NO-LLM stage; cheap
                                      ▼
                                ┌──────────┐
                                │  NREM-3  │ — arc distillation
                                │ (existing │   (existing); schema
                                │  stage2a) │   extract sub-stage
                                │  + v0.7   │   (v0.7); SHY decay
                                │  + v0.8   │   sweep (v0.8)
                                └─────┬────┘
                                      │ HEAVY-LLM stage
                                      ▼
                                ┌──────────┐
                                │   REM    │ — generative
                                │  (v0.7,   │   recombination,
                                │   default-│   quarantined output
                                │   off)    │   (kind=hypothesis)
                                └─────┬────┘
                                      │ HEAVY-LLM stage; flag-gated
                                      ▼
                          ┌────────────────────┐
                          │ POST: cross-project │ — promotion to
                          │  promotion (v0.5,   │   _global if
                          │  flag-off default)  │   eligibility met
                          └────────┬───────────┘
                                   │
                                   ▼
                                ┌──────────┐
                                │   IDLE   │
                                └──────────┘
```

### 2.2 Why a state machine, not a procedural pipeline

The current `dream.py` runs stages 1a → 1b → 2a → 2b in a single linear
function call per project. Two reasons to convert that to an explicit
state machine in v2:

1. **Resumability.** NREM-3 is the only stage that calls the LLM. If
   the LLM call fails halfway through arc distillation for project P,
   the next dream-pass invocation on P should resume at NREM-3, not
   re-run NREM-1/2 (which already mutated the SQLite). Today's code
   re-runs everything; that's safe but wasteful. Persisting the
   per-project state in a small SQLite table
   (`dream_state(project, stage, last_ts, last_error)`) makes this
   resumable.
2. **Observability.** Each stage has different cost characteristics.
   The DMN daemon should be able to run **only NREM-2** during waking
   rest (cheap, no LLM); the nightly dream should run all stages.
   Tagging stages explicitly is the precondition for that scheduling.

### 2.3 Concrete state record

```python
# new in scripts/common.py schema (v0.5)
CREATE TABLE IF NOT EXISTS dream_state (
    project        TEXT PRIMARY KEY,
    last_stage     TEXT NOT NULL DEFAULT 'IDLE',  -- IDLE|NREM1|NREM2|NREM3|REM|POST
    last_complete_ts INTEGER NOT NULL DEFAULT 0,
    last_error     TEXT NOT NULL DEFAULT '',
    last_error_ts  INTEGER NOT NULL DEFAULT 0
);
```

Idempotency rule: each stage takes `last_complete_ts` as its lower
bound. NREM-2 only processes chunks with `ts > last_complete_ts` AND
`sparse_sig IS NULL`. NREM-3 only processes sessions whose newest chunk
has `ts > last_complete_ts`. Re-runs are no-ops once the stage has
caught up.

---

## 3. The DMN Daemon (Waking Offline)

### 3.1 Why DMN is not the dream pass

Per `docs/research/01-neuroscience-of-memory.md` §9 (Buckner 2008,
Staresina 2023): hippocampal ripple-coupled reactivation occurs during
**waking rest**, not just sleep. In software terms this means a cheap,
no-LLM idle daemon that runs micro-replay between user activity. It is
**not** a substitute for the heavier nightly dream pass — it just keeps
the index warm.

### 3.2 Cost model

DMN's job is everything in the cycle that is no-LLM and cheap:

- STDP edge_bumps drain (cost: SQL aggregate on a ≤ 600k row table).
- Sparse-signature backfill for any chunk with `sparse_sig IS NULL`
  (cost: 6 µs per chunk per the retrieval research).
- Backlink refresh for the most recent session (cost: a handful of
  ANN calls).
- Optionally: pre-warm the LanceDB FTS index after writes.

Non-goals: no LLM calls, no schema extraction, no REM, no global
promotion. Those are dream-pass only.

### 3.3 Idle detection on macOS

The user's environment is `darwin` (macOS). The natural idle signals:

- **No `Stop` event in the queue for ≥ 5 min** (config:
  `dmn.idle_seconds_before_run`). Implementation: read the most recent
  row in `~/.claude/brain/<hash>/queue/` directories or a global "last
  Stop" timestamp file.
- **Optional: macOS `ioreg` user-idle time** for true keyboard idle
  (`ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print $NF/1e9; exit}'`).
  Skip if cross-platform compatibility is desired in the future.

### 3.4 Process model

```python
# scripts/dmn.py
def main():
    cfg = load_config()
    if not cfg["dmn"]["enabled"]:
        return
    while True:
        time.sleep(cfg["dmn"]["idle_seconds_before_run"])
        if _has_recent_stop_within(cfg["dmn"]["idle_seconds_before_run"]):
            continue
        deadline = time.monotonic() + cfg["dmn"]["max_runtime_seconds"]
        for pdir in _list_active_projects():
            if time.monotonic() > deadline:
                break
            try:
                _drain_edge_bumps(pdir, cfg)              # v0.5
                _backfill_sparse_signatures(pdir, cfg)    # v0.6 onward
                _refresh_recent_backlinks(pdir, cfg)
            except Exception as exc:
                _log_dmn_error(pdir, exc)
```

**Process supervision:** v0.5 ships as a manually started script. The
user runs `biss-brain-dmn &` once. v0.6 may add a `launchd` plist
template under `scripts/dmn.launchd.plist` for automatic startup, but
that is opt-in (no daemon-installation surprise).

**Signal handling:** SIGTERM → finish current project, exit cleanly.
SIGINT → exit immediately (drop in-flight work — it's all idempotent).

### 3.5 Lock contention with concurrent flush / dream / recall

The four producers writing to SQLite simultaneously:

| Writer | Concurrent? | Conflict point |
|---|---|---|
| `flush.py` (`Stop` event) | yes | inserts new chunks |
| `dream.py` (nightly) | rare | mostly mutates `related_ids`, `superseded_by`, writes arcs |
| `recall.py` / `mcp/server.py` | very frequent | enqueues `edge_bumps`, bumps `retrieval_count`, sets `labile_until` |
| `dmn.py` (idle) | frequent | drains `edge_bumps`, computes `sparse_sig` |

WAL mode (already enabled at `common.py:126`) handles read-write
concurrency well, but the DMN drainer and dream drainer can both target
`edge_bumps`. Per the retrieval research §C, the drainer must use
`BEGIN IMMEDIATE` to avoid SQLite's deadlock-on-upgrade. The pattern:

```python
def _drain_edge_bumps(pdir, cfg):
    with open_meta_direct(pdir) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute("""
                SELECT id_a, id_b, COUNT(*) as cnt FROM edge_bumps
                WHERE drained = 0 AND ts < ?
                GROUP BY id_a, id_b
                HAVING cnt >= ?
            """, (now_unix() - cfg["stdp"]["edge_drain_window_days"]*86400,
                  cfg["stdp"]["co_retrieval_min_count"]))
            for id_a, id_b, cnt in cur.fetchall():
                _add_typed_edge(conn, id_a, id_b,
                                type="co_occurs",
                                weight=min(1.0, cnt / 10.0))
            conn.execute("UPDATE edge_bumps SET drained=1 "
                         "WHERE drained=0 AND ts < ?",
                         (now_unix() - cfg["stdp"]["edge_drain_window_days"]*86400,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
```

Only one drainer (DMN or dream) holds `BEGIN IMMEDIATE` at a time;
the other waits with `busy_timeout` set to 5 s.

---

## 4. Cross-Project Promotion Flow

### 4.1 What 02-v0.5-spec leaves open

The v0.5 spec defines the gate (kind ∈ {schema, lesson, hypothesis};
`consolidation_round ≥ 1`; pinned OR frequency-gate) but is fuzzy on
**how** the frequency gate counts "distinct projects" since each project
is its own SQLite. Two implementation paths:

**Path A (v0.5, conservative):** Treat the frequency gate as
`retrieval_count_local ≥ 5` AND `pinned = 1`. This is what the spec
documents and is correct-by-default — it never over-admits. It does
under-admit (a chunk retrieved 5× in project A but never pinned won't
graduate just because some other project also accesses similar
material).

**Path B (v0.7, principled):** Maintain a global retrieval ledger in
`_global/retrievals.sqlite`:

```sql
CREATE TABLE chunk_retrievals (
    chunk_id TEXT NOT NULL,
    source_project TEXT NOT NULL,
    accessing_project TEXT NOT NULL,
    ts INTEGER NOT NULL
);
CREATE INDEX idx_chunk_retrievals_chunk ON chunk_retrievals(chunk_id);
```

Each cross-project recall hit appends a row. The frequency gate
becomes:

```sql
SELECT COUNT(DISTINCT accessing_project) FROM chunk_retrievals
WHERE chunk_id = ? AND ts > ?
```

Only enable Path B once Path A has shipped and the cross-project layer
is proven not to leak (R1–R4 audit clean).

### 4.2 Atomic promotion writer

Promotion to `_global` writes to a different project dir. The
operation must be transactional in the source project (mark
`cross_eligible=1`) AND in the global project (insert chunk row +
LanceDB row). Two-phase pattern:

```python
def _promote_to_global(src_pdir, chunk_id, cfg):
    with open_meta_direct(src_pdir) as src_conn:
        row = src_conn.execute(
            "SELECT * FROM chunks WHERE id=? AND cross_eligible=0", (chunk_id,)
        ).fetchone()
        if row is None:
            return  # already promoted or not eligible
        # Phase 1: write to global (idempotent — chunk_id is unique)
        global_pdir = BRAIN_ROOT / "_global"
        with open_meta_direct(global_pdir) as g_conn:
            g_conn.execute(
                "INSERT OR IGNORE INTO chunks(id, project, source_project, ...) "
                "VALUES(?, '__global__', ?, ...)",
                (row["id"], _proj_hash_from_dir(src_pdir), ...))
            g_conn.commit()
        _write_global_lance_row(row, global_pdir)
        # Phase 2: mark source eligible (only after global write confirmed)
        src_conn.execute(
            "UPDATE chunks SET cross_eligible=1 WHERE id=?", (chunk_id,))
        src_conn.commit()
```

Failure mode: phase 1 succeeds, phase 2 fails. Idempotency on phase 1
(`INSERT OR IGNORE`) means the next promotion attempt is safe.

### 4.3 The R2 invariant in code (verifiable by grep)

The cross-project conflict-judge research (`04-impl-research-llm-stages.md`
§E) specifies a structural caller-side guard. The code shape:

```python
def judge_with_project_context(new_chunk, old_chunk, cfg):
    same_project = (new_chunk["project"] == old_chunk["project"])
    verdict = judge_conflict(
        new_chunk["text"], old_chunk["text"],
        new_label=_project_label(new_chunk["project"]),
        old_label=_project_label(old_chunk["project"]),
        cfg=cfg
    )
    # STRUCTURAL GUARD — even if the LLM ignores the prompt instruction,
    # cross-project supersession is impossible at the caller level.
    if not same_project and verdict == "CONTRADICTS":
        verdict = "DIVERGES"
    return verdict
```

The reviewer-grep contract from `99-cross-cutting.md` §6.1 becomes:
every code path that writes `superseded_by` must be reachable only via
`judge_with_project_context`, never directly via `judge_conflict`. A
`grep -rn "superseded_by" scripts/ mcp/` produces a finite call-site
list; each must be inspected.

---

## 5. Performance Budget

Budgets per stage, M-series CPU, on a project with 100k chunks
(reasonable upper bound for an active workspace).

### 5.1 Hot path: `recall` / `inject_context` / MCP `search_memory`

Target: < 200 ms median end-to-end.

| Step | Budget | Source |
|---|---|---|
| Embed query (Ollama `/api/embeddings`) | ~80 ms | network/process |
| Dense top-k (LanceDB) | ~10 ms | existing |
| Sparse-Jaccard filter (v0.6+) | 41 µs | research §A |
| Graph 1-hop expand (v0.5) | ~5 ms | SQLite IN query |
| Hopfield blend (v0.8) | 490 µs | research §B |
| Final rerank | ~30 ms | bge-reranker |
| Sum | ~125 ms | |

Headroom: ~75 ms for the per-result side-effects (RIF apply, labile
window stamp, STDP enqueue). Per the retrieval research §E, RIF must
**not** run inline — it goes async to NREM-2. The labile stamp and
STDP enqueue are sub-millisecond SQL writes; they run inline.

### 5.2 NREM-2 (DMN tick)

Target: ≤ 60 s total runtime (config: `dmn.max_runtime_seconds`).
Per project budget: 60 s ÷ N active projects.

| Step | Budget per project | At 100k chunks |
|---|---|---|
| Edge-bump drain | < 1 s | — |
| Sparse-sig backfill (lazy) | 6 µs × n_missing | resumable; usually small |
| RIF apply (queued from prev hot-path) | 20 ms × n_retrieved | research §E |
| Backlink refresh (last session only) | < 5 s | n_chunks_in_session × 10 ms |

### 5.3 NREM-3 (nightly dream)

Heavy LLM stage; budget is best-effort, not bounded.

- Arc distillation: ~5 s/session × LLM call (Haiku-equivalent)
- Schema extraction (v0.7): ~3 s/arc × LLM call
- SHY decay sweep: ~1 s for the SQL update, regardless of corpus size

### 5.4 REM (v0.7, opt-in)

- Sample 3–5 arcs at low pairwise cosine: ~10 ms
- LLM recombination: ~10 s (Haiku-equivalent), one-shot
- Cosine grounding gate (0.30): ~50 ms (LanceDB lookup of evidence
  arcs)

REM is bounded to one run per dream cycle; it is not a pipeline.

---

## 6. Observability

The audit confirmed there is no metrics surface today. v2 needs three
things to be operationally legible:

### 6.1 Stage timings

Every dream stage emits a JSON line to
`~/.claude/brain/<hash>/dream.log`:

```json
{
  "ts": 1714492800,
  "project": "5da0e6b1ad8a",
  "stage": "NREM-2",
  "duration_ms": 412,
  "chunks_processed": 47,
  "edges_drained": 12,
  "errors": []
}
```

The `/biss-brain:doctor` command tails this file and reports the last
24 h of stage outcomes per project.

### 6.2 Counters in `/biss-brain:stats`

Add to the existing stats output:

- `salience` distribution (mean, p50, p90)
- `reconsolidation_count` distribution
- `co_retrieval_count` non-zero count
- `cross_eligible` count vs total chunks
- `kind = hypothesis` count (REM output, should be small)
- per-stage last-runtime

### 6.3 Reviewer-grep invariants as runtime checks

The `99-cross-cutting.md` §6 invariants ("hypotheses are quarantined,"
"cross-project supersession is forbidden," etc.) are compile-time grep
checks, but two of them benefit from runtime assertions:

- After every write to `superseded_by`, assert that
  `chunks.project[new] == chunks.project[old]`. If not, raise — the
  R2 invariant is violated and the violating row should be visible
  before any further damage.
- Before every retrieval response is returned, assert that no row
  with `kind = 'hypothesis'` is in the result list, unless the caller
  passed `include_hypotheses=True`. Default is False everywhere.

These run cheaply (a single tuple comparison) and are the kind of
defence that makes a regression surface in seconds, not weeks.

---

## 7. Migration Sequencing Inside v0.5

The v0.5 multi-agent plan (`01-multi-agent-teams.md` §8) requires the
schema migration to land before any builder branches off. The exact
landing order:

```
Commit M0 (coordinator):  scripts/common.py — schema migration v05
                           (the seven new chunks columns + edge_bumps
                            table + dream_state table)
                          + centralised _fake_embed in common.py
                          + parse_related_ids tolerant adapter

Commit M1 (Edges):        typed-edge writers; consume parse_related_ids

Commit M2 (Capture):      salience compute (heuristic); flush insert
                          path

Commit M3 (Recall):       STDP enqueue; labile stamp; RIF queue
                          (RIF apply moved to NREM-2)

Commit M4 (Reconsolidation): pre-insert merge with cosine drift gate;
                             prompt added to prompts/reconsolidate_merge.md

Commit M5 (DMN):          scripts/dmn.py; STDP drainer; backlink refresh

Commit M6 (Global):       _global project init; promotion writer;
                          judge_with_project_context wrapper;
                          updated prompts/judge_conflict.md
```

Commits M1–M5 are independent and can land in any order after M0.
M6 depends on M1 (typed edges for `cross_links`) and on the prompt
changes in `04-impl-research-llm-stages.md` §E.

The reviewer audits run after M6:

1. Quality reviewer: regression suite + cosine-drift gate test +
   merge-faithfulness golden set.
2. Security reviewer: R1/R2/R3/R4 grep audit + structural guard
   inspection.
3. Documentation reviewer: update 99-cross-cutting and architecture.

Flag-flip happens only after all three sign off.

---

## 8. Failure Modes and Recovery

| Failure | Detection | Recovery |
|---|---|---|
| LLM merge invents facts | cosine drift > 0.12 (research §C in 04) | reject merge, plain insert; close labile window |
| Schema extraction returns malformed JSON | `json.loads` raises | retry once with T=0; if still bad, log + skip |
| REM hypothesis fails grounding gate | cosine < 0.30 to evidence centroid | drop, do not write |
| `_global` promotion phase 2 fails | next dream pass sees row in global but `cross_eligible=0` in source | idempotent retry |
| DMN hangs on a slow ANN call | `dmn.max_runtime_seconds` deadline | kill stage cleanly, log, continue |
| SQLite WAL grows unboundedly | disk-watcher (`du`) | `PRAGMA wal_checkpoint(TRUNCATE)` in DMN tick |
| LanceDB index corruption (rare) | `tbl.search()` raises | rebuild via `tbl.create_fts_index(replace=True)`; re-embed missing rows |
| Cross-project supersession event (R2 violation) | runtime assertion fires (§6.3) | abort write, alert, dump violating row to `dream.log` for forensic review |

Two of these (R2 violation, cosine drift) are not just operational —
they are correctness gates. A single occurrence is investigation-worthy,
not noise.

---

## 9. What Is Still Open

### 9.1 Embedding model versioning

The audit flagged: no `embed_model` or `embed_version` column on
chunks. If the user upgrades from `nomic-embed-text` to a newer
embedding model, today's code silently mixes 768-dim vectors from two
distinct semantic spaces. This is a v0.6 priority — a one-line column
add and a `_compatible_embed?` check on retrieval.

### 9.2 Prompt versioning

Same shape of risk, different surface. If the merge prompt in
`prompts/reconsolidate_merge.md` is edited mid-flight, chunks merged
under prompt vN are not distinguishable from chunks merged under
prompt vN+1. v0.7 should add a `prompt_hash` column to chunks
that record their distillation prompt's SHA-256 at write time. A
prompt rewrite then triggers an opt-in re-distill in the next dream
pass.

### 9.3 Multi-host federation

Beyond v1.0. If a user has `~/.claude/brain` on two machines, they
are independent corpora today. The `_global` design is opt-in
cross-project on **one host**; a multi-host story requires a sync
protocol (rsync? CRDT? cloud-backed LanceDB?) and is well outside the
v0.5–v1.0 plan.

### 9.4 Privacy gates for `_global`

Pinning a chunk in project A and admitting it to `_global` means it
becomes visible from project B's recall. There is no per-chunk privacy
flag today. Two natural additions: a `private` column that excludes
chunks from `_global` even when other gates pass, and a redaction pass
in the promotion writer that strips paths/secrets/identifiers before
the global write. Both are v0.7+ enhancements; v0.5 ships with admission
selectivity (kind-gate + frequency) as the only privacy mechanism.

---

## 10. Open Empirical Questions

The two synthesist agents highlighted three numbers that are
literature-grounded but should be calibrated empirically on real
biss-brain data before flag-flip:

1. **Salience weight α = 0.4 (length), β = 0.6 (surprise)**.
   SynapticRAG's empirical finding favours surprise-dominant. The right
   way to confirm: pin 50 chunks the user judges "high-salience" and
   measure the heuristic's correlation with the manual rating. If
   correlation < 0.5, retune.
2. **Hopfield β = 10–25 for normalised cosines**. Lower β is
   smoother; higher is more discrete. The empirical fit: at β = 20
   approximately 3 patterns dominate. Measure: log the entropy of the
   softmax distribution per recall. If median entropy is < 0.5 nats,
   raise β; if > 2 nats, lower.
3. **Cosine drift gate = 0.12** (reconsolidation merge). The novel
   contribution from the LLM-stages research — calibrated against
   "paraphrase shift produces drift 0.04–0.08; topic-introducing
   hallucination produces drift > 0.15". Build a 30-merge golden set
   from the user's actual reconsolidation events and confirm the
   threshold.

These three knobs become the first config tuning targets after v0.5
ships.

---

## 11. Source Index

This document synthesises:

- `docs/research/01-neuroscience-of-memory.md` — invariants
- `docs/research/05-dream-pass-hook-mapping.md` — code touch points
- `docs/architecture/00-overview.md` — sleep cycle state machine
- `docs/implementation/02-v0.5-spec.md` — schema and flag scope
- `docs/implementation/03-impl-research-retrieval.md` — sparse,
  Hopfield, STDP, RIF, reconsolidation gate
- `docs/implementation/04-impl-research-llm-stages.md` — salience,
  schema, merge, REM, conflict-judge prompts
- `docs/implementation/99-cross-cutting.md` — config, flags,
  invariants
- Audit transcript of the existing biss-brain codebase (in-session,
  not separately committed)

No new external citations — the verified bibliography in 03/04 is
the substantive evidence base.
