# biss-brain v2 Implementation Roadmap

> Phased rollout plan for the brain-inspired architecture. Each phase ships
> independently behind a feature flag, with explicit exit criteria. The order
> minimises risk: cheap, additive changes first; expensive / generative changes
> last; opt-in cross-project after the within-project plumbing is proven.

---

## Phasing Principles

1. **Additive over invasive.** Each phase adds columns, tables, and stages
   without rewriting existing code paths. Old behaviour stays the default until
   the new code has soaked.
2. **Feature-flag everything.** Every new stage, retrieval step, or write rule
   gates on a config key. `git revert` is always one commit away.
3. **Schema migrations are forward-only.** No phase deletes columns from a
   previous phase; they may go unused.
4. **Reviewer audit before flag-on.** A phase is "shipped" when the
   feature-flag default is flipped, not when the code lands. Builder teams write
   the code; reviewer teams set the flag.
5. **Each phase has a measurable exit criterion** so we can tell whether to
   continue or roll back.

---

## v0.5 — Hebbian Foundation

**Headline.** Capture salience, type the edges, learn from co-retrieval, accept
that recall mutates memory, prune by retrieval-induced suppression, run a waking
DMN, optionally promote to a global semantic store.

**Scope.**

- Salience scoring at flush (`scripts/flush.py`): heuristic v1 (length × cosine
  novelty vs recent), LLM v2 behind `capture.salience.use_llm`.
- Schema migration: `chunks.salience INTEGER`, `chunks.labile_until INTEGER`,
  `chunks.reconsolidation_count INTEGER`, `chunks.co_retrieval_count INTEGER`,
  `chunks.cross_links TEXT`, `chunks.cross_eligible INTEGER`.
- Typed `related_ids`: migrate from `["id1", "id2"]` to
  `[{"id": ..., "type": ..., "weight": ...}]`. Read-side parser tolerates both
  shapes during migration.
- STDP edge bumps: recall paths queue pairwise (id_a, id_b, ts) into new
  `edge_bumps` SQLite table. NREM-2 stage drains the queue into typed
  `co_occurs` edges (count ≥ 3).
- RIF on retrieval: after retrieval of A, sibling chunks (≥ 3 of A's top-10
  dense neighbours) get `decay_score *= 0.98` via SQLite update.
- Reconsolidation labile window: recall stamps `labile_until = now + 5min`. New
  capture with cosine ≥ 0.82 against any labile chunk merges instead of
  inserting (LLM call: tighten + preserve facts). Counter increments. After 5
  reconsolidations the chunk is promoted `kind=lesson, consolidation_round=2`.
- DMN daemon (`scripts/dmn.py`): idle-detection (no Stop event > 5 min), runs
  micro-replay (backlink refresh, sparse-sig recompute when v0.6 lands, STDP
  drain). No LLM cost.
- Cross-project layer (opt-in): `~/.claude/brain/_global/`. Admission:
  `kind ∈ {schema, lesson, hypothesis}` with `consolidation_round ≥ 1` AND
  (`pinned=1` OR retrieval_count ≥ 5 from ≥ 3 distinct projects in 30d).
  Cosine gate ≥ 0.85. Conflict judge prompt **must** carry project labels.
  `cross_links` column never aliases `related_ids`.

**Out of scope.** Sparse signatures (v0.6). Hopfield blend (v0.8). REM
(v0.7). Schema graph (v0.7). Trained SAE (v1.0).

**Exit criteria.**

- Tests pass for all new stages and migrations.
- Reconsolidation merge produces fewer total chunks than baseline on a 1-week
  replay corpus, with no decision-chunk loss measured by manual spot-check on
  10 sampled sessions.
- Cross-project promotion is verifiable: pin a chunk in project A, search from
  project B with the global flag on, the chunk is returned with the correct
  source-project label.
- Reviewer team signs off. Default-on for typed edges, STDP, RIF,
  reconsolidation, DMN. Cross-project remains opt-in (default-off) regardless
  of soak — that is a permanent product decision, not a soak gate.

**Estimated effort.** ~1 week with parallel teams.

---

## v0.6 — Pattern Separation

**Headline.** Sparse signatures distinguish surface-similar chunks; dual-index
retrieval kills cross-domain false positives.

**Scope.**

- Schema migration: `chunks.sparse_sig BLOB` (4096 bits packed, 512 bytes).
- Sparse signature compute: top-k binarisation of dense embedding (keep
  top-200 magnitudes, hash to 4096-bit bitmap). Helper in `scripts/common.py`.
- Flush integration: every new chunk stores its signature.
- DMN backfill: a one-shot pass computes signatures for all existing chunks
  (resumable, batched).
- Retrieval pipeline: insert sparse-Jaccard filter step between dense top-k
  and final rerank. Drop candidates with Jaccard < 0.10 against query
  signature.
- Reconsolidation merge gate refinement: require sparse-Jaccard ≥ 0.20 in
  addition to cosine ≥ 0.82. Prevents merging across surface-similar but
  domain-different chunks.

**Out of scope.** Trained SAE (v1.0).

**Exit criteria.**

- Backfill completes for projects with ≤ 100k chunks in < 5 min on M-series
  CPU.
- Recall@10 on a held-out query set does not degrade vs v0.5 baseline.
- Cross-domain false-positive rate (manual: hand-curated test queries) drops
  measurably vs v0.5.

**Estimated effort.** ~1 week.

---

## v0.7 — Semantic Scaffold

**Headline.** Schema graph captures generalisable structure; new arcs that fit
existing schema fast-path to L2; REM generates hypotheses, quarantined.

**Scope.**

- Schema migration: `schema_edges` table (subject, relation, object,
  evidence_ids, confidence, created_ts, invalidated_ts, superseded_by).
- New `scripts/schema_extract.py`: LLM call over arc text → typed entity /
  relation tuples. Conflicting tuples invalidate older edges (Graphiti-style
  bi-temporal versioning).
- NREM-3 sub-stage calls schema_extract on each new arc.
- Schema-consistent fast path: arcs whose extracted relations all match
  existing edges write `consolidation_round=2` directly, skipping the slow
  distillation queue.
- New `_stage3_rem` in `dream.py`: samples 3–5 high-salience but
  low-similarity arcs, prompts LLM for hypothesis chunks (`prompts/rem_recombine.md`).
  Output is `kind=hypothesis`, `consolidation_round=2`,
  `cross_eligible=0` (must be promoted by user).
- Quarantine semantics: hypotheses are NEVER returned by auto-recall or
  inject_context. Only surfaced via explicit `/recall hypothesis` or in a
  dedicated UI panel.
- UI: schema graph viewer route; hypothesis panel with a "promote" /
  "discard" affordance.

**Out of scope.** PPR over the schema graph (v1.0). Trained SAE (v1.0).
Hopfield blend (v0.8).

**Exit criteria.**

- Schema extraction produces ≥ 80% precision on a 50-arc hand-labelled set.
- REM hypotheses do not surface in default recall paths under any code path
  (verified by reviewer: search the codebase for `kind = 'hypothesis'`
  filters; every retrieval path must exclude it).
- UI presents hypotheses with a distinct visual treatment.

**Estimated effort.** ~2 weeks. REM is the highest-risk stage — see
`01-multi-agent-teams.md` for dedicated security review.

---

## v0.8 — Forgetting & Soft Recall

**Headline.** Active forgetting prunes the corpus to keep generalisation
working; Hopfield blend converts brittle top-1 retrieval into attractor-style
soft recall; iterative graph expansion converges on relevant clusters.

**Scope.**

- TTL pruning: NREM-3 deletes chunks with `decay_score < 0.05` AND
  `pinned = 0` AND `kind NOT IN ('lesson', 'schema')`. Soft delete via
  `superseded_by = '__decayed__'` for 30 days, then hard delete.
- Cold archive: pruned chunks move to `~/.claude/brain/<hash>/archive/` as
  newline-delimited JSON, queryable but not indexed.
- Hopfield soft blend: replace top-1 / top-k return with softmax-weighted
  blend over the candidate set (Ramsauer 2020). Temperature β config-tuned
  so 2–4 patterns dominate.
- Iterative retrieval: take the Hopfield-blended state, re-query the index
  for its top-k neighbours, re-blend. Up to 3 iterations or until convergence
  (state delta < ε). This is the explicit CA3 attractor settle.

**Out of scope.** Trained SAE (v1.0).

**Exit criteria.**

- Pruning measurably reduces total chunk count without measurable
  recall@10 regression on a held-out query set.
- Iterative retrieval improves coverage of long-tail queries (manual:
  hand-curated set of underspecified queries).

**Estimated effort.** ~1 week.

---

## v1.0 — Trained Sparse Features (Research)

**Headline.** Replace top-k binarisation with SAE-learned sparse features;
PPR over schema graph for principled multi-hop retrieval.

**Scope.**

- Train a small sparse autoencoder per project (Bricken 2023, Cunningham
  2023) on the project's chunk embeddings. Hidden dim = 4× dense (≈ 3072),
  L1 penalty for sparsity.
- Replace `sparse_sig` compute with SAE feature activations (still packed as
  bitmap above an activation threshold).
- Personalised PageRank over `schema_edges` for graph expansion (HippoRAG-style),
  replacing the v0.5 1-hop walk.

**Out of scope.** This phase is exploratory. Acceptance is contingent on
empirical lift over v0.8.

**Exit criteria.**

- Recall@10 on held-out queries improves measurably over v0.8 baseline.
- Train cost amortises (one-shot per project, < 30 min on M-series CPU).
- If neither holds, ship as opt-in research mode and revert default to
  v0.8 top-k binarisation.

**Estimated effort.** months. Research timeline, not engineering.

---

## Phase Dependencies

```
v0.5 ──► v0.6 ──► v0.7 ──► v0.8 ──► v1.0
  │        │        │        │
  │        │        │        └─► requires Hopfield (v0.8) is correct before
  │        │        │            iterative retrieval can land
  │        │        └─► REM benefits from sparse signatures (v0.6) for diversity
  │        │            sampling but does not strictly require them
  │        └─► sparse signatures benefit from typed edges (v0.5) for the
  │            DMN backfill scheduler but do not strictly require them
  └─► no dependencies; v0.5 is the foundation
```

Within a phase, the team breakdown in `01-multi-agent-teams.md` describes
which work streams are parallel and which serialise.

---

## Rollback Strategy

Every phase ships behind feature flags. Rollback is `set <flag> = false` in
config and re-run flush / dream — the new columns and tables remain (they're
inert when the flag is off) and the code paths revert to the previous
behaviour.

The one exception is **typed `related_ids`** in v0.5: the column shape changes
from `["id"]` to `[{id, type, weight}]`. Read paths must tolerate both
forever (the migration is one-way, and rollback means tolerating the new
shape under the old behaviour). The cost is a 6-line shape adapter in
`scripts/common.py`.

---

## Out-of-Scope (Permanent)

These were considered and explicitly rejected in
`docs/architecture/00-overview.md` §7:

- Full HTM stack (Spatial Pooler / Temporal Memory).
- NTM / DNC — superseded by tool-mediated retrieval.
- Predictive coding networks — research-stage, no demonstrated lift on
  coding tasks.
- Spiking neural networks — hardware story.
- VSA over dense embeddings — impedance mismatch with `nomic-embed-text`.
- Full GraphRAG community summaries — ingestion cost too high for a
  live coding workspace.

---

## Success Metrics (End-State, Post-v0.8)

- Median recall latency stays under 200 ms on M-series.
- Reconsolidation merges absorb ≥ 30% of would-be-duplicate captures.
- Cross-project promotion graduates ≤ 5% of all chunks (high admission
  selectivity).
- Schema graph contains ≥ 50 edges per active project after 30 days of use.
- REM hypothesis chunks are surfaced to the user; promotion-to-lesson rate
  is non-zero (the user finds at least some hypotheses useful).
- Zero observed cross-project supersession events (this is the R2 invariant
  and must be verifiable by code audit, not just by absence of incidents).
