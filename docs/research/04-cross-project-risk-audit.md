# Cross-Project Risk Audit

> Audit of contamination, correctness, and privacy risks if biss-brain's existing
> dedup / conflict-judge / decay / backlink logic is exposed to a cross-project
> embedding space. Frame: **what would silently break** under cross-project data?

---

## R1 — Hard Contamination: Hash Dedup Collapses Distinct Chunks

**Location.** `scripts/flush.py:277-285`, `scripts/remember.py:181-188`

The dedup gate is a pure SHA-256 equality check on the distilled summary string —
**not cosine distance**. If two projects produce a distillation that comes out
byte-for-byte identical (`"- Added config.toml at repo root"`) the second write is
silently dropped and the project-scoped `project` column is never stored for it. In a
per-project DB this is harmless. In a shared DB, one project's chunk would simply
never be written for the other.

The `dedup_cosine = 0.94` config value (`scripts/common.py:53`) is defined in
`DEFAULT_CONFIG["hygiene"]` but **is never read anywhere in the codebase** — no file
loads it and applies a cosine gate. There is therefore no vector-based dedup at all
today; the only dedup is hash equality. In a cross-project corpus, a shared DB would
collapse one of two identical-text chunks regardless of project, suppressing the
second project's memory entirely.

**Mitigation requirement.** Cross-project mode must scope dedup hash matching to
within-project, OR the project column must be part of the dedup key, OR a cosine gate
must finally be implemented and applied per-project.

---

## R2 — Hard Contamination: Conflict Judge Has No Project Context

**Location.** `scripts/flush.py:162-233`, `scripts/dream.py:298-420`,
`prompts/judge_conflict.md:1-24`

Both the flush-time `_run_conflict_judge` and the dream Stage 1b `_stage1b_conflicts`
scope their candidate search with `project = '{proj}'` and `kind = '{kind}'`. In
today's architecture this is safe.

If a cross-project pass relaxed the `project =` filter, **the LLM judge receives two
chunks with no project context** — the prompt (`prompts/judge_conflict.md`) contains
only `{new_text}` and `{old_text}`, with zero metadata about project of origin. The
judge is instructed: *"Direct factual reversal → CONTRADICTS"*. Two chunks that say
"uses session cookies" vs "uses JWT" from completely different stacks would produce
CONTRADICTS, causing the older chunk's `superseded_by` to be written
(`flush.py:214-227`). The superseded chunk is then **excluded from all future
retrieval** via the universal `superseded_by = ''` filter.

**This is a silent, destructive, irreversible write with no undo path.**

**Mitigation requirement.** Either (a) the conflict judge prompt must include
project labels for both chunks, or (b) cross-project supersession must be **disabled
entirely** — only the link is recorded, never the kill.

---

## R3 — Hard Contamination: Global Retrieval Count Biases Decay

**Location.** `scripts/decay.py:32-42`, `scripts/dream.py:426-429`,
`scripts/recall.py:128-131`

`retrieval_count` is incremented unconditionally on every retrieval event
(`recall.py:128-131`, `mcp/server.py:112-113`). The decay formula is:

```
exp(-Δdays / half_life) * (1 + 0.2 * retrieval_count) * pin_boost
```

If a global chunk is retrieved 20 times from project A it accumulates
`retrieval_boost = 5.0`, which more than doubles its `decay_score` vs a never-retrieved
chunk of the same age. If this chunk also appears relevant to project B, project B
retrieves it too, pushing the score even higher. **Conversely**, a high-quality
project-B-only chunk that was never retrieved from project A starts at
`retrieval_count = 0`, decay-ranks lower, and is buried behind project-A-dominant
hits.

There is no per-project retrieval counter, no normalisation, and the decay run
(`decay.py:45-111`) is a single global sweep that mixes all projects' `retrieval_count`
values under one formula.

**Mitigation requirement.** Per-project retrieval counters (e.g.,
`retrieval_count_by_project TEXT JSON`) OR exclude cross-project hits from
`retrieval_count` increments OR normalise by exposure (count / projects_exposed_to).

---

## R4 — Hard Contamination: Backlink Cross-Vocabulary Collision

**Location.** `scripts/backlinks.py:20-24`, `scripts/dream.py:197-245`

Both `write_backlinks` and `_write_backlinks_direct` include a `project = '{proj}'`
guard in their WHERE clause, so today backlinks are project-scoped. However the
`_stage1a_backlinks` dream pass calls `_write_backlinks_direct(pdir, tbl, proj, cid,
vec)` with the project-specific `tbl`.

If a cross-project pool were introduced, the only safeguard is that `project` filter.
At the **0.70 cosine threshold** (`_L2_GATE = 0.78` corresponds to L2≈0.78 → cosine
≈ 0.70), generic vocabulary ("config file", "auth flow", "migration", "database
connection") between unrelated projects will routinely produce false positive
backlinks, **hard-linking chunk IDs across project boundaries** in `related_ids`. The
`related_to` MCP tool (`mcp/server.py:156-193`) then surfaces these to the model
with no cross-project label.

**Mitigation requirement.** Cross-project backlink threshold must be ≥ 0.85 cosine,
not 0.70; cross-project links must be stored in a separate column
(`cross_links` not `related_ids`); MCP tool output must annotate cross-project results
with their source project name.

---

## S1 — Soft Degradation: BM25 IDF Skew Across Vocabularies

**Location.** `scripts/common.py:307-337`, `scripts/flush.py:137-139`

The FTS index is created with `tbl.create_fts_index("text", replace=True)` on the
full LanceDB table. If multiple projects share one table, BM25 IDF weights will be
computed **globally**. Terms that are rare within a single project ("Axum", "Prisma",
"SQLx") but common across two Rust projects would have artificially deflated IDF,
making them poor discriminators. Conversely, terms common to all projects ("auth",
"config", "error", "migration") that are normally filtered out by BM25's IDF
saturation would resurface as if rare if the cross-project corpus is small.

The `hybrid_search` function (`common.py:327-337`) has no project filter injected at
the BM25 level; the `where_sql` filter (`project = '{proj}'` today) is applied
**after** the hybrid fusion, so BM25 scores from a global index bleed into the RRF /
linear-combination ranking even when only one project's results survive the
post-filter.

**Mitigation.** Either keep per-project FTS indexes (one per LanceDB table, no
global table), or apply the project filter as a pre-filter on the FTS path.

---

## S2 — Soft Degradation: `prefer_summaries_in_recall` Amplifies Pollution

**Location.** `scripts/recall.py:85-109`, `scripts/inject_context.py:89-138`

When `prefer_summaries_in_recall` is true, `recall.py` issues a first pass restricted
to `kind IN ('arc', 'daily_summary')`. Arc chunks (`dream.py:552-691`) are synthesised
narrative summaries that intentionally strip session-level noise and preserve
high-level decisions. **They are the most portable (project-agnostic in language)
chunks in the corpus.**

In a cross-project pool, arc chunks from project A would score very highly against
project B queries because their language is clean, dense, and topically coherent.
They would be promoted *above* project B's own raw chunks by the kind-priority sort.

**Mitigation.** Cross-project recall must label arcs with their source project AND
require a stricter cosine gate than within-project arcs.

---

## S3 — Soft Degradation: Auto-Inject Threshold Too Permissive

**Location.** `scripts/inject_context.py:226-238`, `scripts/common.py:48-49`

The threshold is applied as `_distance < 0.8` where `_distance` is L2. For 768-dim
unit-norm vectors, L2 = 0.8 corresponds to cosine ≈ 0.68 — a very low bar. Within a
single project most false positives are low-signal-but-harmless. Across projects,
hits at cosine 0.68–0.75 in domain-adjacent corpora (two web APIs, two CLI tools)
would inject plausible-sounding but wrong project context **silently into every
user prompt** via the UserPromptSubmit hook.

**Mitigation.** Cross-project auto-inject must require cosine ≥ 0.85 AND
`kind ∈ {schema, lesson}` AND user opt-in.

---

## Mitigations Already Implicit in the Schema

### M1 — `project` column is first-class

Location: `scripts/common.py:131-148`, every `WHERE` in `flush.py`, `recall.py`,
`inject_context.py`, `mcp/server.py`

The `project` column (12-hex SHA-256 of `realpath(cwd)`) is stored in both LanceDB
and SQLite for every chunk. All current retrieval and maintenance queries filter on
it. A cross-project mode could implement a two-tier query: a global pass that records
the source `project` on every returned row, then a caller-side filter or ranking
bonus for same-project hits.

### M2 — `superseded_by` and `kind` filters universally applied

Both columns would remain valid cross-project discriminators. A global conflict
sweep could safely gate on `kind = '{kind}'` AND `project != '{proj}'` to explicitly
restrict judge calls to cross-project pairs only when that is the desired semantic.

### M3 — `pinned` and `consolidation_round` survive decay

Pinned chunks receive a 5× decay multiplier regardless of `retrieval_count`. Arc /
daily_summary chunks carry `consolidation_round = 1/2`, which could be used as a
cross-project "safe-to-link" tier since they have already been through LLM synthesis
and have project-agnostic language. **A cross-project dream pass could restrict
linking to `consolidation_round >= 1` chunks only**, reducing the surface area for
false conflict verdicts and noisy backlinks.

### M4 — `project_path.txt` provides a human-readable namespace label

Location: `scripts/common.py:215-223`, `scripts/dream.py:110-127`

The file written by `ensure_project_label` stores the real filesystem path. Any
cross-project UI or prompt injection could prefix retrieved chunks with this label,
giving the model enough context to self-correct if a cross-project chunk is
irrelevant.

---

## Risk Register Summary

| ID | Type | Mechanism | Severity | Mitigation cost |
|---|---|---|---|---|
| R1 | Hard | Hash dedup collapses distinct chunks | High | Low (per-project hash key) |
| R2 | Hard | Conflict judge silently kills cross-project chunks | **Critical** | Medium (prompt change + new column) |
| R3 | Hard | Decay biased by global retrieval_count | Medium | Medium (per-project counter) |
| R4 | Hard | Backlink false positives across vocabularies | Medium | Low (raise threshold + new column) |
| S1 | Soft | BM25 IDF skew | Low | Low (per-project FTS) |
| S2 | Soft | Arc dominance in cross-project recall | Medium | Low (label + stricter gate) |
| S3 | Soft | Auto-inject too permissive | High | Low (raise threshold + opt-in) |

**Conclusion.** All hard risks have known mitigations. The combination of
(a) **opt-in** cross-project promotion, (b) **kind-gating** to
`{schema, lesson, hypothesis}` with `consolidation_round ≥ 1`, (c) **stricter cosine
gate** (≥ 0.85), (d) **conflict-judge prompt enrichment with project labels**, and
(e) **separate `cross_links` column** distinct from `related_ids` is sufficient to
ship safely.

The single most important rule: **cross-project supersession is permanently
forbidden**. Cross-project conflicts may be flagged, surfaced in the UI, presented
to the user — but never used to write `superseded_by`.
