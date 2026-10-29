# Dream Pass: Internal Architecture Mapping

> Internal map of the existing biss-brain dream pass + recall pipeline. Identifies
> where cross-project linking and brain-inspired layers (sparse signatures, schema
> graph, REM, RIF, reconsolidation) plug in cleanly.

---

## Current Isolation Boundaries

**Physical boundaries (separate DB files)** are the primary enforcement mechanism.
Each project lives in `~/.claude/brain/<12-hex-hash>/meta.sqlite` and
`~/.claude/brain/<12-hex-hash>/lancedb/`. Every read or write in `dream.py`,
`recall.py`, `inject_context.py`, and `mcp/server.py` opens **one specific project's**
LanceDB connection via `_open_lance(pdir)` (`dream.py:157`) or `lancedb_dir(cwd)`
(`common.py:118`) and **one specific SQLite file** via `_open_meta_direct(pdir)` or
`open_meta(cwd)`.

**Logical boundary** also exists: every row in every `chunks` table carries a
`project TEXT NOT NULL` column (`common.py:132`), and every LanceDB row also stores
`project` as a string field (`dream.py:539`). All WHERE clauses filter on it — e.g.,
`project = '{proj}'` in backlink search (`dream.py:205`), conflict sweep
(`dream.py:358`), arc distillation (`dream.py:567`), and all three recall paths.

**Project identity is doubly encoded**: in directory location AND in a column on
every row.

**Cross-DB UNION cost.** LanceDB has no native cross-instance federation. A
cross-project vector search would require opening N separate LanceDB connections,
issuing N parallel `.search()` calls, merging result lists in Python, and re-ranking
the merged set. For 5–10 projects this is low-overhead; for 50+ it becomes a fan-out
problem. The `project` column on every row means a single merged LanceDB table would
eliminate that fan-out entirely — rows are already self-describing.

---

## Natural Cross-Project Hooks in `dream.py`

### Hook A — Stage 1a Backlink Refresh

**Location.** `_stage1a_backlinks`, `dream.py:248-292`

Currently the candidate search is scoped hard to one project:
`where = f"project = '{proj}' AND superseded_by = '' AND id != '{chunk_id}'"`
(`dream.py:205`). The vector search runs against a single `pdir`'s LanceDB table.

**What would change.** If a global / shared LanceDB table existed (or if the search
opened all project tables and merged), removing the `project = '{proj}'` constraint
from the WHERE clause is the only change to the query logic. The backlink write-back
(`UPDATE chunks SET related_ids`) goes to SQLite, so a new column
`cross_project_links TEXT` alongside `related_ids` could store cross-project IDs
without touching the existing schema.

**Invasiveness.** Low — one WHERE clause change + new column.

### Hook B — Stage 1b Conflict Sweep

**Location.** `_stage1b_conflicts`, `dream.py:298-420`

The top-similar-neighbour search at `dream.py:354-363` is also scoped to
`project = '{proj}'`. A cross-project contradiction detector ("project A decided X,
project B decided not-X") is the natural output. The `superseded_by` write already
works per-row in individual SQLites; **cross-project contradictions would need a new
link type** (a `cross_superseded_by TEXT` column, or a global conflict ledger).

**Invasiveness.** Medium — need a place to store cross-project supersession without
corrupting single-project indices. Per the risk audit (`04-cross-project-risk-audit.md`,
R2): **cross-project supersession must never write `superseded_by`**; it can only
record a typed `contradicts` link.

### Hook C — Stage 2a Arc Distillation

**Location.** `_stage2a_arcs`, `dream.py:552-691`

Each arc today packages one session from one project. **Arc distillation could
instead be run as a "cross-project theme" pass**: group arcs from multiple projects
by semantic similarity (topic cluster), then distill a higher-level
`kind='theme_arc'` at `consolidation_round=3`. The session query
(`dream.py:563-573`) is entirely per-project SQLite; a cross-project variant would
query all projects' SQLites for `kind='arc' AND consolidation_round=1`, gather them,
and group by semantic distance.

The `evidence_ids` column already exists to record provenance.

**Invasiveness.** Medium — no schema changes needed; new stage function; `project`
field on the new chunk would be set to a sentinel (`"__global__"`) or stored in a
new global project dir.

### Hook D — Stage 2b Daily Summary

**Location.** `_stage2b_daily`, `dream.py:694-800`

Daily summaries currently collapse arcs for a single project's day. **Cross-project
daily summaries** ("what happened across all projects on 2026-04-30") are a trivial
extension: iterate all project dirs, collect their `kind='arc'` rows for the same
date string, concatenate, and call `_distill_arc`. The LanceDB text-fetch loop
(`dream.py:734-737`) would need to open each project's table separately and join
texts before passing to the LLM. The insert of the resulting `daily_summary` chunk
needs a home — the cleanest option is a dedicated global project dir
(`BRAIN_ROOT / "_global" / ...`).

**Invasiveness.** Low for the distillation logic; medium for deciding where to
store the result.

---

## Recall Integration Cost

### Where the project filter is applied

In all three recall paths the filter is built as a Python string before any DB call:

- `mcp/server.py:84` — `where_parts = [f"project = '{proj}'", "superseded_by = ''"]`
- `scripts/recall.py:76` — same pattern
- `scripts/inject_context.py:226` — same pattern
- `inject_context.py`'s `_recent_decisions` (lines 100-109) — SQLite query
  `WHERE project = ? AND superseded_by = ''`

The filter is applied **before** the vector search
(`tbl.search(qvec).where(where_sql)`), meaning it is a pre-filter on the LanceDB ANN
index. To surface cross-project results the filter just needs a `project IN (...)`
list or the clause dropped entirely if a global table is used.

### Schema support

The existing `project` column on every chunk (`common.py:132`, `dream.py:539`) means
cross-project results are already self-labelling — a merged result set carries its
provenance. No schema migration is required for basic cross-project recall; results
can be annotated with their source project hash / path from `project_path.txt`.

### Retrieval-count and decay bump

`mcp/server.py:103-115`, `recall.py:119-132` call `open_meta(cwd)` which resolves to
the **current** project's SQLite. For cross-project hits these bumps would need to
open the correct remote project's SQLite by hash. That is 1–2 lines of change per
call site: replace `open_meta(cwd)` with
`_open_meta_direct(BRAIN_ROOT / r['project'])` for each result row.

---

## Hooks for v2 Brain-Inspired Layers

Beyond cross-project, the v2 architecture (`docs/architecture/00-overview.md`) adds:

### Sparse signature column (DG analog)

**Insertion point.** `scripts/flush.py` distill pipeline. After embedding, compute
sparse signature (top-k binarisation of dense embedding to a 4096-bit bitmap),
store as new column `sparse_sig BLOB` on chunks.

**Read paths.** `recall.py` and `mcp/server.py` need to apply sparse-Jaccard filter
between dense top-k and final return. New helper function in `common.py`.

### Co-retrieval STDP edge bumps

**Insertion point.** Recall paths at `recall.py:119-132` and `mcp/server.py:103-115`.
After collecting result IDs in a session, queue pairwise edge bumps to a new SQLite
table `edge_bumps(session_id, id_a, id_b, ts)`. Dream pass NREM-2 stage drains
these into `related_ids` typed edges.

### Reconsolidation labile window

**Insertion point.** Recall returns mark each chunk with
`labile_until = now + 5min`. Capture path in `flush.py` checks if any new chunk's
content has cosine ≥ 0.82 against any labile chunk; if yes, **merge** instead of
insert.

### Retrieval-induced suppression (RIF)

**Insertion point.** Same recall paths. After retrieval of chunk A, identify chunks
sharing ≥ 3 of A's top-10 dense neighbours and apply `decay_score *= 0.98` via
SQLite update.

### Schema graph (mPFC scaffold)

**Insertion point.** New table `schema_edges` defined in `common.py` schema
migration. New module `scripts/schema_extract.py` called from dream NREM-3 with the
arc text → LLM extraction → temporal-versioned writes to `schema_edges`.

### REM generative recombination

**Insertion point.** New stage `_stage3_rem` in `dream.py` after `_stage2b_daily`.
Samples 3–5 high-salience but unrelated arcs across projects (or within), prompts
LLM for hypothesis chunks, writes with `kind='hypothesis'` and a quarantine flag.

### DMN daemon (waking offline)

**Insertion point.** New file `scripts/dmn.py`. Watches for idle (no Stop event >
5 min). Runs micro-replay over the last session — refresh backlinks, recompute
sparse signatures, drain co-retrieval edge bump queue. No LLM cost.

---

## Cross-Cutting File Touch List for v2

| File | What changes |
|---|---|
| `scripts/common.py` | Schema migrations: `salience`, `sparse_sig`, `labile_until`, `reconsolidation_count`, `cross_links`, `cross_superseded_by`, `co_retrieval_count`, `schema_edges` table, sparse-signature helper, sparse-Jaccard helper |
| `scripts/flush.py` | Salience scoring; sparse signature compute; reconsolidation-merge check before insert; cross-project hash-key dedup |
| `scripts/recall.py` | Sparse-signature filter step; co-retrieval edge bump enqueue; RIF apply; labile window mark; cross-project fan-out |
| `scripts/inject_context.py` | Same as recall.py for the inject path |
| `scripts/dream.py` | New NREM-2 stage (sparse signature backlinks, edge bump drain, RIF); new NREM-3 schema integration sub-stage; new REM stage (`_stage3_rem`); SHY global decay; cross-project arc + daily summary writers to `_global` project dir |
| `scripts/decay.py` | Per-project retrieval count; SHY-style multiplicative decay |
| `scripts/dmn.py` | NEW — idle-detection daemon |
| `scripts/schema_extract.py` | NEW — LLM-based entity / relation extraction |
| `mcp/server.py` | Same as recall.py + cross-project label annotation |
| `prompts/judge_conflict.md` | Add project labels to template |
| `prompts/distill_arc.md` | (Possibly) add salience hint |
| `prompts/extract_schema.md` | NEW — entity / relation extraction prompt |
| `prompts/rem_recombine.md` | NEW — generative recombination prompt |
| `ui/` templates and routes | New global dashboard route; hypothesis chunk UI; schema graph viewer |
