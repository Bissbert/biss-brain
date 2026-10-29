# biss-brain v2 Architecture: Brain-Inspired Memory

> Synthesis document. Grounded in `docs/research/01-neuroscience-of-memory.md` and
> `docs/research/02-bio-inspired-ai-architectures.md`. The design is bio-inspired,
> not bio-mimetic — biological mechanisms are the source of computational
> *invariants*, but implementations use the most pragmatic available primitive
> (LanceDB, SQLite, Ollama) rather than spiking neurons or differential equations.

---

## 1. Design Invariants (Non-Negotiable)

These seven properties are derived from the neuroscience research and any v2
implementation must respect them:

1. **Two-store write architecture (CLS).** Fast / sparse / episodic and slow /
   distributed / semantic stores. One store cannot do both jobs.

2. **Sparse pattern separation at write (DG).** Similar inputs must map to
   near-orthogonal codes, otherwise interference dominates as the corpus grows.

3. **Pattern completion at read (CA3).** Partial cues should converge to stored
   attractors via iterative retrieval, not single-shot top-k.

4. **Offline consolidation is mandatory (NREM / replay / SHY).** No optional dream
   pass — the episodic store saturates without it.

5. **Every retrieval is a write (reconsolidation).** Recall is destabilising; the
   system must accept that and let traces update.

6. **Active forgetting is first-class (AMPAR turnover, RIF).** TTL with deletion,
   retrieval-induced suppression, global down-scaling. Forgetting nothing prevents
   generalisation.

7. **Salience and prediction error gate write strength (STDP).** Surprise + reward
   + co-activation should weigh more than recency alone.

---

## 2. Architecture Overview

```
                 ┌─────────────────────────────────────────┐
                 │  Capture (Stop / SubagentStop / hooks)  │  ← perforant path
                 │  + salience score (LLM 1–10 at ingest)  │
                 └────────────────┬────────────────────────┘
                                  ▼
   ┌──────────────────────── HIPPOCAMPUS ────────────────────────┐
   │ L1a: episodic LanceDB (per project)                         │
   │      dense embedding (768d, nomic) +                        │
   │      sparse signature (4096b, ~5% active) ← DG analog       │
   │ L1b: SQLite chunks: salience, decay, related_ids (typed),   │
   │      co_retrieval_count, labile_until, last_access          │
   └─────────────┬───────────────────────────────────┬──────────┘
                 │ (sleep cycle: NREM 1/2/3 + REM)   │ (recall)
                 ▼                                   ▼
   ┌──── NEOCORTEX ────┐         ┌──── CA3 retrieval loop ────┐
   │ L2: semantic       │         │ 1. dense top-k             │
   │     LanceDB        │         │ 2. sparse-sig pattern sep  │
   │     kind=arc,      │         │ 3. graph expand (1-hop)    │
   │     daily, schema, │         │ 4. Hopfield soft blend     │
   │     hypothesis,    │         │ 5. recency × salience      │
   │     lesson         │         │    × decay rerank          │
   ├────────────────────┤         └────────────────────────────┘
   │ L3: schema graph   │
   │     SQLite typed   │
   │     edges, temp-   │
   │     versioned      │ ← mPFC analog (Tse 2007)
   └────────────────────┘
                 │
                 ▼
   ┌──── GLOBAL NEOCORTEX (~/.claude/brain/_global/) ────┐
   │ Cross-project store, opt-in promotion                │
   │ Accepts only kind ∈ {schema, lesson, hypothesis}     │
   │ Cosine gate ≥ 0.85, project labels in conflict prompt│
   └─────────────────────────────────────────────────────┘
```

---

## 3. Layer Definitions

### L0 — Sensory buffer (perforant path)

Existing `queue/` directory in each project. **Add salience scoring at flush time**:
a one-shot LLM call rates each chunk 1–10 on (novelty × decision-weight × surprise
vs prior chunks). Stored in `chunks.salience INTEGER`. This becomes the gate signal
for replay weighting and consolidation priority — the analog of waking
ripple-tagging that biases overnight replay (Singer & Frank 2009 / Kumaran 2016).

### L1 — Hippocampal store (episodic, sparse, high-fidelity)

Today's per-project LanceDB + SQLite, with two additions:

**Sparse signature column on every chunk.**

- v1 (cheap): top-k binarisation of the dense embedding — keep the 200 highest-magnitude
  dims out of 768, zero the rest, hash to a 4096-bit bitmap.
- v2 (research): train a small SAE on the project's embeddings (Bricken 2023), use
  its sparse feature activations.

The signature is a **pattern-separation index**: two surface-similar chunks from
different domains (the cross-project R1/R4 risk) will share dense-cosine ≈ 0.85 but
sparse-Jaccard ≈ 0.15, exposing them as distinct.

**Typed `related_ids`.** Replace the current opaque list with
`[{id, type, weight}]` where `type ∈ {refines, contradicts, generalises, instance_of,
caused, co_occurs}`. Type is set by the dream pass's conflict judge or by structural
rules (e.g., same session = `co_occurs`).

### L2 — Neocortical store (semantic, distributed, lower-fidelity)

A **second LanceDB table per project**, taking only LLM-distilled chunks:
`kind ∈ {arc, daily_summary, schema, hypothesis, lesson}` with
`consolidation_round ≥ 1`. Has its own retrieval budget at recall time (config:
`retrieve.semantic_top_k`).

The hippocampal store surfaces "what specifically happened in session X." The
neocortical store surfaces "what did I learn / what patterns recur."

### L3 — Schema graph (mPFC analog)

SQLite table of typed temporal edges:

```sql
CREATE TABLE schema_edges (
    id INTEGER PRIMARY KEY,
    project TEXT NOT NULL,
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    object TEXT NOT NULL,
    evidence_ids TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_ts INTEGER NOT NULL,
    invalidated_ts INTEGER NOT NULL DEFAULT 0,  -- Graphiti-style
    superseded_by INTEGER NOT NULL DEFAULT 0
);
```

Built by an LLM extractor over arcs during NREM-3 consolidation. Conflicting facts
get the older edge `invalidated_ts`-stamped, never deleted.

**Schema-consistent fast path (Tse 2007).** New arcs whose extracted relations all
match existing edges bypass the slow distillation queue and go straight to L2 with
`consolidation_round=2`.

### L4 — Global semantic store (opt-in cross-project)

At `~/.claude/brain/_global/`. Same shape as L2 but cross-project. Strict admission
control:

- only chunks with `cross_project_eligible=1`, set by either user pin OR
- a frequency gate: `retrieval_count ≥ 5` from `≥ 3` distinct projects within
  30 days.

Recall fans out at cosine ≥ 0.85 only, results labelled with source project from
`project_path.txt`. **Conflict judge prompts MUST include both projects' labels**
(fixes the R2 silent-supersession risk).

---

## 4. The Sleep Cycle as a State Machine

Replace the current monolithic dream pass with four ordered stages mapped to
NREM-1/2/3 + REM, gated on session-debounce and cooldown.

| Stage | Biological analog | What it does | Cost |
|---|---|---|---|
| **NREM-1** | Light sleep, K-complexes | Backlink refresh + dedup + same-project conflict sweep | Existing |
| **NREM-2** | Spindles | Compute sparse signatures for new chunks; build sparse-signature edges; STDP edge bumps from co-retrieval logs | Cheap (no LLM) |
| **NREM-3** | Slow-wave + ripples | **Replay**: sample top-salience sessions, distill arcs. **Schema integration**: extract typed edges; fast-path schema-consistent arcs to L2. **SHY**: global multiplicative `decay_score *= 0.97`, prune chunks < 0.05 (active forgetting via AMPAR-removal analog) unless pinned | Heavy (LLM) |
| **REM** | Paradoxical sleep | **Generative recombination**: sample 3–5 high-salience but unrelated arcs, prompt LLM "what hypotheses connect these?" Output is `kind=hypothesis`. **Quarantined**: never injected into auto-recall. Only surfaced via explicit `/recall hypothesis` or in the UI. | Heavy (LLM) |

### STDP — co-retrieval edge bumps

When two chunks are co-retrieved in the same recall response within a session,
their edge `co_retrieval_count` increments. The dream pass converts this into a
`co_occurs` typed edge if count ≥ 3. **This is the timing-dependent write rule
operationalised** — without it, the graph only learns from explicit LLM judgment.

### RIF — retrieval-induced suppression

When chunk A is retrieved, chunks sharing ≥ 3 of A's top-10 dense neighbours get a
small `decay_score *= 0.98`. Prevents the "everything is similar to everything"
interference collapse that the cross-project audit flagged as risk S2.

### Reconsolidation

Every retrieved chunk gets `labile_until = now + 5min`. If a subsequent `Stop` event
within that window contains content with cosine ≥ 0.82 to the labile chunk, the new
content is **merged** into the existing chunk (LLM call: "tighten this, preserve
facts") rather than written as a new chunk. Counter `reconsolidation_count`
increments. After N=5 reconsolidations the chunk is marked `kind=lesson` and
promoted to L2 — repeated rewrite is the strongest signal of generalisable
knowledge.

### DMN — waking offline

New daemon `scripts/dmn.py` watches for idle (no Stop event > 5 min). Runs
micro-replay: refresh backlinks for the last session, recompute sparse signatures,
run STDP edge bumps. **No LLM cost** — purely indexing. Mirrors Buckner 2008 /
Staresina 2023 — the hippocampus replays during waking rest, not just sleep.

---

## 5. Retrieval Pipeline (CA3 Attractor Dynamics)

Replace today's single-shot hybrid search with a five-step settle:

1. **Dense top-k** over hippocampal LanceDB (current behaviour, k=20).
2. **Sparse-signature filter**: drop candidates whose sparse-Jaccard with the query
   signature is < 0.10. This is pattern separation — kills the surface-similar-but-
   domain-mismatched hits.
3. **Graph expansion**: 1-hop walk over typed `related_ids`
   (only `refines | generalises | caused`, not `contradicts`); add expanded set to
   candidate pool.
4. **Hopfield soft blend** (Ramsauer 2020): instead of returning the top-1, return
   the softmax-weighted blend over the candidate set with temperature β chosen so
   2–4 patterns dominate. This is mathematically attention; one matrix-multiply over
   the candidate matrix.
5. **Final rerank**: `score = α·cosine + β·salience + γ·decay + δ·recency`.
   Coefficients in config.

Steps 2–4 are all O(top_k · d) in the candidate set, so the whole pipeline stays
under 50ms for k=20 even on CPU.

---

## 6. Cross-Project Layer

See `docs/research/03-cross-project-memory-prior-art.md` and
`docs/research/04-cross-project-risk-audit.md` for evidence base. Architectural
rules:

- **Opt-in only.** `dream.cross_project.enabled` defaults to `false`.
- **Kind-gated.** Only `kind ∈ {schema, lesson, hypothesis}` with
  `consolidation_round ≥ 1` are eligible for cross-project promotion.
- **Cosine ≥ 0.85** for cross-project edges (vs the within-project 0.70 at
  `_L2_GATE = 0.78`).
- **Conflict judge gets project labels** injected into the prompt.
- **Cross-project supersession permanently forbidden.** Only a `contradicts` typed
  link is recorded; `superseded_by` is never written cross-project.
- **Recall labels cross-project hits** with the source project name (from
  `project_path.txt`) so the model can self-correct.
- **Separate `cross_links` column** distinct from `related_ids` — never overload
  existing per-project structures.

---

## 7. What We Deliberately Do NOT Build

The AI-primitives research surfaced several technologies that look biological but
don't deliver in practice:

- **Full HTM stack** (Numenta) — SDRs are useful (we're stealing them as the sparse
  signature), but the Spatial Pooler / Temporal Memory / Thousand Brains framework
  has no production track record.
- **NTM / DNC** — superseded conceptually by tool use over external stores, which
  is what biss-brain already has.
- **Predictive coding networks** — research-stage; would require training
  infrastructure with no demonstrated lift on coding tasks.
- **Spiking neural networks** — a hardware story (Loihi 2), not a software-layer
  primitive.
- **VSA over dense embeddings** — impedance mismatch with `nomic-embed-text`; the
  binding/bundling operations don't compose with non-VSA embeddings.
- **Full GraphRAG community summaries** — ingestion cost too high for a
  continuously-updating coding workspace.

**Steal:** SDRs (as sparse signatures), modern Hopfield (as blend), HippoRAG-style
PPR (as graph expand), Park et al. reflection (as arc distill — already there),
DER++ (as the labile/reconsolidation logits memory).

---

## 8. Open Design Questions

Three things to settle before / during implementation:

1. **Sparse signature dimensionality.** 4096-bit at 5% active is a guess; needs
   empirical capacity testing on real chunk corpora. Numenta argues 2% is the
   sweet spot; SAE work suggests overcompleteness > 4× the dense dim.

2. **REM safety.** Generative recombination is the highest-risk new component
   (self-reinforcing false beliefs is the #1 failure mode in the literature).
   Quarantine + provenance is the safety net, but we should also surface
   `hypothesis` chunks in the UI with a distinct visual treatment so the user can
   prune them aggressively.

3. **Salience model cost.** Scoring every chunk via a Haiku call adds latency and
   tokens. Alternative: heuristic salience (chunk length × novelty-vs-recent-chunks
   via cosine) as v1, LLM scoring opt-in.

---

## 9. Phased Rollout

See `docs/implementation/00-roadmap.md` for the full plan.

| Phase | Headline | Effort | Status |
|---|---|---|---|
| **v0.5** | Salience + typed edges + STDP + reconsolidation + RIF + DMN + cross-project (opt-in, kind-gated) | ~1 week | **Shipped** |
| **v0.6** | Sparse signature column + dual-index retrieval + reconsolidation merge | ~1 week | Planned |
| **v0.7** | Hardening instead of the schema graph: eval harness, provenance, temporal validity, core memory, cheap capture, scope, ablation, store fingerprint (see §10) | ~2 weeks | **Shipped** |
| **later** | Schema graph + schema fast-path + REM (quarantined), the former v0.7 plan | ~2 weeks | Planned |
| **v0.8** | Active TTL prune + cold archive + Hopfield soft blend + iterative graph-expand retrieval | ~1 week | Planned |
| **v1.0** (research) | SAE-trained sparse features replacing top-k binarisation; PPR over schema graph | months | Planned |

The most impactful single change for least effort is **typed `related_ids` +
co-retrieval STDP + reconsolidation merge** — those three together convert
backlinks from a static index into an actively-learning Hebbian graph and would
produce visible recall-quality lift within a week.

---

## 10. v0.7: correctness, safety and evidence

v0.7 spent its budget on making the existing design trustworthy, not on new layers. The plan is
`docs/plans/v0.7-upgrade.md`, the per-phase record `docs/plans/v0.7-progress.md`, and the
user-facing list of behaviour changes `CHANGELOG.md`.

- **Evidence first.** A frozen eval harness (`scripts/eval/`) with a real and a synthetic
  suite scores every ranking change. Phase 7 ablated each mechanism of §4 against it
  (`docs/eval/v0.7-ablation.md`). RIF and the usage boost in nightly decay did not earn their
  place and are off by default. STDP, backlinks and DMN stay, but they feed only the graph
  tools, not ranking.
- **Provenance (L0/L1).** Every chunk carries `source`, `trust` and `taint`. Tool and web
  content is never auto-injected, ranks lower, cannot supersede first-party memory, and cannot
  launder into a trusted merge, arc or global row. Injected memories are escaped data.
- **Temporal validity.** `valid_from` (event time) and `invalid_at` sit beside `ts` (record
  time). A contradiction closes the older fact instead of deleting it, and every read path takes
  `as_of` and `include_history`.
- **Core memory**, a curated tier outside the vector store: Markdown files in their own git repo
  (global, per repo, and an opt-in shared file in the repo), injected at SessionStart and after
  compaction. Writes from the model are proposals; only the user promotes them.
- **Capture by segment.** Hooks only queue a byte offset. The daemon distills each session
  segment once, with one JSON call per window, retries failures with backoff, and stores without
  a vector when the embedder is down.
- **Reconsolidation** merges only memories that were used and that new evidence revises; a
  restatement is folded in without an LLM call.
- **Scope and identity.** One launcher, excluded directories with a shell-level check, and one
  project key per repository (worktrees fold into the main checkout).
- **Store fingerprint.** `store.json` records the schema version and the embed model, dim and
  prefixes. Every reader and writer refuses a store whose fingerprint differs from the config.
  `fresh_start.py` archives the old brain and starts an empty store (decision D8);
  `migrate_v07.py` is the in-place alternative.
