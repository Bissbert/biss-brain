# Cross-Cutting Reference

> One-stop reference for state that spans phases: schema migration history,
> config-key index, feature-flag list, glossary. Updated by the documentation
> reviewer after each phase ships.

---

## 1. Schema Migration Register

| Version | Phase | Tables / Columns | One-way? | Notes |
|---|---|---|---|---|
| `v05` | v0.5 | + `chunks.salience`, `chunks.labile_until`, `chunks.reconsolidation_count`, `chunks.co_retrieval_count`, `chunks.cross_links`, `chunks.cross_eligible`, `chunks.retrieval_count_local`; + table `edge_bumps` | yes (related_ids shape) | Read adapter tolerates legacy `related_ids` shape forever. **Shipped in v0.5.** |
| `v06` | v0.6 | + `chunks.sparse_sig BLOB` | no | DMN backfill computes for legacy chunks |
| `v07` | v0.7 | + table `schema_edges` | no | Bi-temporal versioning fields built in from the start |
| `v08` | v0.8 | (no new schema; uses existing `decay_score`, `superseded_by`) | n/a | Soft-delete uses sentinel `superseded_by='__decayed__'` |
| `v10` | v1.0 | (TBD — SAE feature persistence; PPR cache) | TBD | Research phase |

Rule: every migration is additive and idempotent. Re-running after
upgrade is a no-op. Downgrade leaves the column unused.

---

## 2. Config-Key Index

All keys live under `~/.claude/brain/config.toml` with `DEFAULT_CONFIG`
baked into `scripts/common.py`. Phase that introduced the key is in
parens.

```toml
[capture]
salience.use_llm = false                      # (v0.5)
salience.heuristic.length_weight = 0.4        # (v0.5)
salience.heuristic.novelty_weight = 0.6       # (v0.5)

[reconsolidation]
enabled = true                                # (v0.5)
labile_window_seconds = 300                   # (v0.5)
merge_cosine_threshold = 0.82                 # (v0.5)
merge_sparse_jaccard_threshold = 0.20         # (v0.6)
promote_to_lesson_after = 5                   # (v0.5)

[stdp]
enabled = true                                # (v0.5)
co_retrieval_min_count = 3                    # (v0.5)
edge_drain_window_days = 7                    # (v0.5)

[rif]
enabled = true                                # (v0.5)
shared_neighbour_threshold = 3                # (v0.5)
decay_multiplier = 0.98                       # (v0.5)

[dmn]
enabled = true                                # (v0.5)
idle_seconds_before_run = 300                 # (v0.5)
max_runtime_seconds = 60                      # (v0.5)

[sparse_signature]
enabled = false                               # (v0.6) — flipped on after backfill
dim = 4096                                    # (v0.6)
top_k = 200                                   # (v0.6)
jaccard_filter_threshold = 0.10               # (v0.6)

[schema]
enabled = false                               # (v0.7)
extraction.confidence_threshold = 0.6         # (v0.7)
fast_path.enabled = true                      # (v0.7)

[rem]
enabled = false                               # (v0.7) — DEFAULT-OFF for one release minimum
sample_size = 4                               # (v0.7)
min_salience = 7                              # (v0.7)
max_pairwise_cosine = 0.6                     # (v0.7)

[forget]
enabled = false                               # (v0.8)
decay_floor = 0.05                            # (v0.8)
soft_delete_window_days = 30                  # (v0.8)

[hopfield]
enabled = false                               # (v0.8)
beta = 8.0                                    # (v0.8)

[retrieve]
iterative.enabled = false                     # (v0.8)
iterative.max_iters = 3                       # (v0.8)
iterative.convergence_eps = 0.01              # (v0.8)
use_sae = false                               # (v1.0) — research opt-in

[cross_project]
enabled = false                               # (v0.5) — PERMANENT default-off
admission.min_consolidation_round = 1         # (v0.5)
admission.eligible_kinds = ["schema", "lesson", "hypothesis"]  # (v0.5)
admission.frequency.min_retrievals = 5        # (v0.5)
admission.frequency.distinct_projects = 3     # (v0.5)
admission.frequency.window_days = 30          # (v0.5)
recall.cosine_gate = 0.85                     # (v0.5)
recall.label_with_source_project = true       # (v0.5)
backlinks.cosine_gate = 0.85                  # (v0.5)
```

---

## 3. Feature Flag List

| Flag | Default | Phase shipped | Status | Promotes-to-default-on after |
|---|---|---|---|---|
| `reconsolidation.enabled` | true | v0.5 | **shipped** | shipped on |
| `stdp.enabled` | true | v0.5 | **shipped** | shipped on |
| `rif.enabled` | true | v0.5 | **shipped** | shipped on |
| `dmn.enabled` | true | v0.5 | **shipped** | shipped on |
| `cross_project.enabled` | false | v0.5 | **shipped** | **never** (product decision) |
| `capture.salience.use_llm` | false | v0.5 | **shipped** | user-only opt-in |
| `sparse_signature.enabled` | false | v0.6 | pending | after backfill complete + Recall@10 non-regression |
| `schema.enabled` | false | v0.7 | pending | after extraction precision ≥ 0.8 on labelled set |
| `schema.fast_path.enabled` | true (when schema on) | v0.7 | pending | on with parent flag |
| `rem.enabled` | false | v0.7 | pending | one full release minimum after security review sign-off |
| `forget.enabled` | false | v0.8 | pending | after restore tooling proven |
| `hopfield.enabled` | false | v0.8 | pending | after Recall@10 non-regression |
| `retrieve.iterative.enabled` | false | v0.8 | pending | after latency budget test |
| `retrieve.use_sae` | false | v1.0 | pending | only if measured lift; else permanent opt-in |

The two flags that **never** flip on by user policy:

- `cross_project.enabled` — product decision; sharing across projects is
  always opt-in per `docs/research/04-cross-project-risk-audit.md`.
- `capture.salience.use_llm` — cost decision; the LLM call is opt-in for
  users who want richer salience at higher token spend.

---

## 4. Reserved Sentinels

| Sentinel | Meaning | Phase |
|---|---|---|
| `project = '__global__'` | Cross-project store identifier | v0.5 |
| `superseded_by = '__decayed__'` | Soft-deleted by TTL prune; restorable for 30 days | v0.8 |
| `kind = 'arc'`, `'daily_summary'` | Existing summary kinds | pre-v0.5 |
| `kind = 'lesson'` | Promoted via reconsolidation N=5 | v0.5 |
| `kind = 'schema'` | Promoted from schema graph extraction | v0.7 |
| `kind = 'hypothesis'` | REM output; quarantined from auto-recall | v0.7 |

---

## 5. Glossary

**Active forgetting.** Energy-consuming memory removal (vs passive
decay). Implemented as TTL prune in v0.8. Bio source: AMPAR endocytosis
(Hardt 2013).

**CA3.** Hippocampal subfield with dense recurrent connections; pattern
completion. Architectural analog: the iterative graph-expand + Hopfield
blend retrieval pipeline.

**CLS (Complementary Learning Systems).** McClelland 1995. The thesis
that a fast/episodic and slow/semantic store coexist because neither can
do both jobs well. Architectural analog: L1 hippocampal store +
L2 neocortical store.

**Consolidation.** Offline transformation of episodic traces into
distributed, generalised semantic knowledge. Architectural analog: dream
pass → arc → daily summary → schema.

**DG (dentate gyrus).** Hippocampal subfield producing sparse,
near-orthogonal codes. Architectural analog: sparse signature column.

**DMN (Default Mode Network).** Brain network active during waking rest.
Architectural analog: idle daemon `scripts/dmn.py`.

**Engram.** Physical memory trace. Architectural analog: a chunk row.

**Labile window.** ~5-min post-retrieval period during which a memory
is destabilised and rewrite-eligible. Architectural analog:
`labile_until` column.

**NREM / REM.** Sleep stages. Architectural analog: dream pass stages
NREM-1 / NREM-2 / NREM-3 / REM.

**Pattern completion.** Reconstructing a full memory from a partial cue.
Architectural analog: iterative retrieval (v0.8).

**Pattern separation.** Mapping similar inputs to distinct codes.
Architectural analog: sparse-signature filter (v0.6).

**Reconsolidation.** Restabilisation of a destabilised trace; allows
update. Architectural analog: pre-insert merge (v0.5).

**Replay.** Sequenced reactivation of waking patterns during sleep.
Architectural analog: NREM-3 arc distillation.

**RIF (Retrieval-Induced Forgetting).** Retrieval of A suppresses
similar non-retrieved siblings. Architectural analog: `decay_score *=
0.98` on shared-neighbour chunks (v0.5).

**Salience.** Attention/importance weight that biases consolidation
priority. Architectural analog: `chunks.salience INTEGER` column (v0.5).

**Schema.** Relational scaffold of generalised facts. Architectural
analog: `schema_edges` table (v0.7).

**SHY (Synaptic Homeostasis Hypothesis).** Tononi & Cirelli — sleep
globally down-scales synaptic weights to combat saturation.
Architectural analog: NREM-3 multiplicative decay sweep (v0.5/v0.8).

**Sparse signature.** ~5%-active high-dim binary vector. v1: top-k
binarisation; v2: SAE features.

**SDR (Sparse Distributed Representation).** Numenta's term for the
same. Architectural analog: same.

**STDP (Spike-Timing-Dependent Plasticity).** Hebbian rule:
co-activation strengthens, mistimed activation weakens. Architectural
analog: co-retrieval `edge_bumps` queue (v0.5).

**Soft delete.** Mark for deletion but retain restorability. v0.8 uses
`superseded_by='__decayed__'`.

---

## 6. Code-Path Invariants (Verified by Reviewer Grep)

These invariants must be checkable by codebase grep at any phase.
Reviewers run these checks per their charter.

1. **Cross-project supersession is forbidden.** Every
   `UPDATE chunks SET superseded_by` query has a same-project guard in
   the WHERE.

2. **Hypotheses are quarantined.** Every retrieval path
   (`recall.py`, `inject_context.py`, `mcp/server.py`) has a
   `kind != 'hypothesis'` filter.

3. **Cross-project hits do not bump local retrieval_count.** Local
   retrieval increments only when the row's project matches the caller's
   `cwd`-derived project hash.

4. **`cross_links` is never written to `related_ids`.** Search for
   writes to `related_ids` — none should touch cross-project IDs.

5. **`_global` admission is gated.** The function that writes to
   `_global` checks `kind ∈ admission.eligible_kinds`,
   `consolidation_round >= 1`, AND (pinned OR frequency-gate).

6. **Default-off flags are default-off.** `cross_project.enabled = false`
   and `rem.enabled = false` (after v0.7) in `DEFAULT_CONFIG`.

---

## 7. Where to Find Things

| Topic | File |
|---|---|
| Why this architecture | `docs/research/01-neuroscience-of-memory.md`, `docs/architecture/00-overview.md` |
| Why we picked these primitives | `docs/research/02-bio-inspired-ai-architectures.md` |
| Why cross-project is opt-in | `docs/research/03-cross-project-memory-prior-art.md`, `docs/research/04-cross-project-risk-audit.md` |
| Where in `dream.py` things plug in | `docs/research/05-dream-pass-hook-mapping.md` |
| When to ship what | `docs/implementation/00-roadmap.md` |
| Who builds it | `docs/implementation/01-multi-agent-teams.md` |
| What to write for v0.5 | `docs/implementation/02-v0.5-spec.md` |
| How to code retrieval primitives | `docs/implementation/03-impl-research-retrieval.md` |
| How to code LLM-driven stages | `docs/implementation/04-impl-research-llm-stages.md` |
| Dream-cycle orchestration, perf budget, migration sequencing | `docs/implementation/05-impl-research-orchestration.md` |
| Extended literature (REM safety, sleep-time compute, SAEs, salience) | `docs/research/06-extended-literature.md` |
| Citation audit trail | `docs/research/verification-report.md` |
| Schema, flags, glossary | this file |
